
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import json
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
import warnings
warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from common import *


@dataclass
class WeakTrainConfig:
    # =========================
    # 只改这里：数据、模型、训练和 UniMatch V2 参数都集中在一个配置中
    # =========================
    root_dir: str = "./data/train"
    hr_dir: str = "hr"
    mask_dir: str = "mask"
    hr_suffix: str = "hr"
    mask_suffix: str = "mask"

    save_dir: str = "./experiment_worldview/unimatch_v2"
    method: str = "unimatch_v2"
    backbone: str = "resnet101"  # resnet50 / resnet101

    # common.py 中 build_model / make_optimizer / ramp_weight 需要的通用字段
    # 这些字段统一放在这里，后面不再二次覆盖，避免配置分散。
    output_stride: int = 16
    pretrained_backbone: bool = False
    num_classes: int = 1
    momentum: float = 0.9
    nesterov: bool = True
    lr_power: float = 0.9
    min_lr: float = 0.0
    rampup_epochs: int = 0
    num_epochs: int = 60
    batch_size: int = 8
    num_workers: int = 0
    val_ratio: float = 0.1
    seed: int = 42
    split_json: str = ""
    patience: int = 5

    lr: float = 1e-4
    weight_decay: float = 1e-4

    # optimizer settings required by common.py::make_optimizer
    backbone_lr_mult: float = 1.0

    # compatibility fields kept consistent with common.py::TrainConfig
    ema_momentum: float = 0.99
    temperature: float = 0.5
    rank_conf_thresh: float = 0.7
    u2pl_conf_thresh: float = 0.95
    rank_weight: float = 0.5
    edge_weight: float = 0.15
    proto_weight: float = 0.2
    neg_weight: float = 0.2
    rank_pairs: int = 2048
    rank_margin: float = 0.0
    u2pl_queue_size: int = 512
    u2pl_alpha0: float = 0.20
    u2pl_delta_p: float = 0.30
    u2pl_contrast_weight: float = 0.5
    u2pl_low_rank: int = 1
    u2pl_high_rank: int = 2
    u2pl_num_anchors: int = 256
    u2pl_num_negatives: int = 512
    contrast_temperature: float = 0.1
    grad_clip: float = 1.0

    hr_divisor: float = 2048.0
    mask_threshold: float = 50.0
    mask_binarize: bool = True

    conf_thresh: float = 0.0
    ema_momentum: float = 0.996
    cutmix_ratio: float = 0.5

    log_interval: int = 50
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def strip_suffix(stem: str, suffix: str) -> str:
    token = "_" + suffix
    if stem.endswith(token):
        return stem[:-len(token)]
    if stem.endswith(suffix):
        return stem[:-len(suffix)]
    return stem


def find_samples(cfg: WeakTrainConfig) -> List[Dict[str, str]]:
    hr_root = Path(cfg.root_dir) / cfg.hr_dir
    mask_root = Path(cfg.root_dir) / cfg.mask_dir
    if not hr_root.exists():
        raise FileNotFoundError(f"hr folder not found: {hr_root}")
    if not mask_root.exists():
        raise FileNotFoundError(f"mask folder not found: {mask_root}")

    hr_map = {strip_suffix(p.stem, cfg.hr_suffix): str(p) for p in hr_root.iterdir() if p.is_file()}
    mask_map = {strip_suffix(p.stem, cfg.mask_suffix): str(p) for p in mask_root.iterdir() if p.is_file()}
    ids = sorted(set(hr_map) & set(mask_map))
    if len(ids) == 0:
        raise RuntimeError(f"No matched weak samples found in {hr_root} and {mask_root}. Expected xxx_{cfg.hr_suffix}.tif / xxx_{cfg.mask_suffix}.tif")
    return [{"id": sid, "hr_path": hr_map[sid], "mask_path": mask_map[sid]} for sid in ids]


def split_samples(cfg: WeakTrainConfig):
    if cfg.split_json:
        split = json.loads(Path(cfg.split_json).read_text(encoding="utf-8"))
        by_id = {sample["id"]: sample for sample in find_samples(cfg)}
        train_ids = list(split.get("train", []))
        val_ids = list(split.get("val", split.get("validation", [])))
        missing = [sid for sid in train_ids + val_ids if sid not in by_id]
        if missing:
            raise RuntimeError(f"split_json contains ids missing from dataset: {missing[:5]}")
        return [by_id[sid] for sid in train_ids], [by_id[sid] for sid in val_ids]
    samples = find_samples(cfg)
    rng = random.Random(cfg.seed)
    rng.shuffle(samples)
    n_val = max(1, int(round(len(samples) * cfg.val_ratio))) if len(samples) > 1 else 0
    return samples[n_val:], samples[:n_val]


def read_tif(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
    return arr


def process_img(arr: np.ndarray, divisor: float) -> torch.Tensor:
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32) / float(divisor)
    arr = np.clip(arr, 0.0, 1.0)
    return torch.from_numpy(arr)


def process_mask(arr: np.ndarray, threshold: float, binarize: bool) -> torch.Tensor:
    if arr.ndim == 2:
        arr = arr[None]
    if arr.shape[0] > 1:
        arr = arr[:1]
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if binarize:
        arr = (arr > float(threshold)).astype(np.float32)
    else:
        arr = np.clip(arr / 100.0, 0.0, 1.0).astype(np.float32)
    return torch.from_numpy(arr)


def photometric_aug(x: torch.Tensor, strength: float = 0.12) -> torch.Tensor:
    y = x.clone()
    scale = 1.0 + (torch.rand(1).item() * 2.0 - 1.0) * strength
    bias = (torch.rand(1).item() * 2.0 - 1.0) * strength * 0.25
    y = y * scale + bias
    if strength > 0:
        y = y + torch.randn_like(y) * (strength * 0.08)
    return y.clamp(0.0, 1.0)


class WeakWaterDataset(Dataset):
    def __init__(self, samples: List[Dict[str, str]], cfg: WeakTrainConfig, train: bool = True):
        self.samples = samples
        self.cfg = cfg
        self.train = train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = process_img(read_tif(s["hr_path"]), self.cfg.hr_divisor)
        mask = process_mask(read_tif(s["mask_path"]), self.cfg.mask_threshold, self.cfg.mask_binarize)
        if mask.shape[-2:] != img.shape[-2:]:
            mask = F.interpolate(mask.unsqueeze(0), size=img.shape[-2:], mode="nearest").squeeze(0)

        if self.train:
            return {
                "id": s["id"],
                "img": img,
                "img_weak": photometric_aug(img, 0.04),
                "img_strong": photometric_aug(img, 0.16),
                "img_strong2": photometric_aug(img, 0.20),
                "mask_hr": mask.float(),
            }
        return {"id": s["id"], "img": img, "mask_hr": mask.float()}


def build_weak_loaders(cfg: WeakTrainConfig):
    train_samples, val_samples = split_samples(cfg)
    train_loader = DataLoader(WeakWaterDataset(train_samples, cfg, True), batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(WeakWaterDataset(val_samples, cfg, False), batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, val_loader


def dice_loss_with_logits(logits, target, eps=1e-6):
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1.0 - ((2.0 * inter + eps) / (den + eps)).mean()


def supervised_loss_hr(logits, target):
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    if target.ndim == 3:
        target = target.unsqueeze(1)
    if target.shape[-2:] != logits.shape[-2:]:
        target = F.interpolate(target.float(), size=logits.shape[-2:], mode="nearest")
    bce = F.binary_cross_entropy_with_logits(logits, target.float())
    dice = dice_loss_with_logits(logits, target.float())
    return bce + dice, logits


def binary_iou_hr(logits, target, thresh=0.5, eps=1e-6):
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    if target.ndim == 3:
        target = target.unsqueeze(1)
    if target.shape[-2:] != logits.shape[-2:]:
        target = F.interpolate(target.float(), size=logits.shape[-2:], mode="nearest")
    pred = (torch.sigmoid(logits) >= thresh)
    target = (target > 0.5)
    tp = (pred & target).sum(dim=(1, 2, 3)).float()
    tn = ((~pred) & (~target)).sum(dim=(1, 2, 3)).float()
    fp = (pred & (~target)).sum(dim=(1, 2, 3)).float()
    fn = ((~pred) & target).sum(dim=(1, 2, 3)).float()
    return (0.5 * (tp / (tp + fp + fn + eps) + tn / (tn + fp + fn + eps))).mean().item()


def masked_mean(loss_map, valid, eps=1e-6):
    while valid.ndim < loss_map.ndim:
        valid = valid.unsqueeze(1)
    valid = valid.float().expand_as(loss_map)
    return (loss_map * valid).sum() / valid.sum().clamp_min(eps)


def save_simple_ckpt(path, models, optimizers, cfg, epoch, best_iou):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "best_iou": best_iou,
        "config": asdict(cfg),
        "models": {k: v.state_dict() for k, v in models.items()},
        "optimizers": {k: v.state_dict() for k, v in optimizers.items()},
    }, path)


def validate_single(models, loader, cfg, model_key="student"):
    model = models[model_key]
    model.eval()
    loss_meter = AverageMeter(); iou_meter = AverageMeter()
    with torch.no_grad():
        for batch in loader:
            img = batch["img"].to(cfg.device)
            target = batch["mask_hr"].to(cfg.device)
            logits = model(img)
            loss, logits_hr = supervised_loss_hr(logits, target)
            iou = binary_iou_hr(logits_hr, target)
            bs = img.size(0)
            loss_meter.update(loss.item(), bs); iou_meter.update(iou, bs)
    return {"loss": loss_meter.avg, "iou_hr": iou_meter.avg}


def validate_dual_avg(models, loader, cfg):
    models["student1"].eval(); models["student2"].eval()
    loss_meter = AverageMeter(); iou_meter = AverageMeter()
    with torch.no_grad():
        for batch in loader:
            img = batch["img"].to(cfg.device)
            target = batch["mask_hr"].to(cfg.device)
            logits1 = models["student1"](img)
            logits2 = models["student2"](img)
            loss1, logits1 = supervised_loss_hr(logits1, target)
            loss2, logits2 = supervised_loss_hr(logits2, target)
            avg_logits = 0.5 * (logits1 + logits2)
            iou = binary_iou_hr(avg_logits, target)
            bs = img.size(0)
            loss_meter.update(0.5 * (loss1.item() + loss2.item()), bs); iou_meter.update(iou, bs)
    return {"loss": loss_meter.avg, "iou_hr": iou_meter.avg}


def run_loop(cfg, build_models_fn, train_one_epoch_fn, validate_fn):
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)
    Path(cfg.save_dir, "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    train_loader, val_loader = build_weak_loaders(cfg)
    sample = next(iter(train_loader))
    in_channels = sample["img"].shape[1]
    models, aux = build_models_fn(in_channels, cfg)
    total_iters = max(1, cfg.num_epochs * len(train_loader))
    optimizers = {k: make_optimizer(v, cfg, total_iters) for k, v in models.items() if any(p.requires_grad for p in v.parameters())}

    print(f"method={cfg.method} backbone={cfg.backbone} input_channels={in_channels} device={cfg.device}")
    print(f"root={cfg.root_dir} train={len(train_loader.dataset)} val={len(val_loader.dataset)} save_dir={cfg.save_dir}")

    best_iou = -1.0
    stale_epochs = 0
    timer = AverageMeter()
    for epoch in range(1, cfg.num_epochs + 1):
        t0 = time.time()
        train_stats = train_one_epoch_fn(models, optimizers, train_loader, epoch, cfg, aux)
        val_stats = validate_fn(models, val_loader, cfg) if len(val_loader.dataset) > 0 else {"loss": 0.0, "iou_hr": 0.0}
        epoch_sec = time.time() - t0
        timer.update(epoch_sec)
        eta = timer.avg * (cfg.num_epochs - epoch)
        print(
            f"[{cfg.method}] {epoch:03d}/{cfg.num_epochs:03d} | train={train_stats['loss']:.4f} "
            f"(sup={train_stats.get('loss_sup', train_stats.get('loss_lr', 0)):.4f}, unsup={train_stats.get('loss_unsup', 0):.4f}, "
            f"rank={train_stats.get('loss_rank', 0):.4f}, iou={train_stats.get('iou_hr', 0):.4f}, valid={train_stats.get('valid_ratio', 1):.3f}) | "
            f"val={val_stats['loss']:.4f} (iou={val_stats['iou_hr']:.4f}) | time={epoch_sec:.1f}s avg={timer.avg:.1f}s eta={format_sec(eta)}"
        )
        save_simple_ckpt(os.path.join(cfg.save_dir, "latest.pth"), models, optimizers, cfg, epoch, best_iou)
        if val_stats["iou_hr"] > best_iou:
            best_iou = val_stats["iou_hr"]
            save_simple_ckpt(os.path.join(cfg.save_dir, "best.pth"), models, optimizers, cfg, epoch, best_iou)
            print(f"  -> save best, best_iou={best_iou:.4f}")
            stale_epochs = 0
        else:
            stale_epochs += 1
        if cfg.patience > 0 and stale_epochs >= cfg.patience:
            print(f"Early stopping at epoch {epoch}: no weak-validation IoU improvement for {cfg.patience} epochs.")
            break
    print(f"Finished. Best val IoU(HR weak mask) = {best_iou:.4f}")


# 全局配置实例：后续代码只使用这一个 cfg，避免前后两处参数不一致
cfg = WeakTrainConfig()


def build_models(in_channels, cfg):
    student = build_model(in_channels, cfg).to(cfg.device)
    teacher = build_model(in_channels, cfg).to(cfg.device)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False
    return {"student": student, "teacher": teacher}, {"global_step": 0}


def sample_cutmix_box(h, w, ratio=0.5, device="cpu"):
    mask = torch.zeros((1, 1, h, w), device=device)
    cut_h = max(1, int(h * ratio * torch.empty(1, device=device).uniform_(0.5, 1.0).item()))
    cut_w = max(1, int(w * ratio * torch.empty(1, device=device).uniform_(0.5, 1.0).item()))
    y1 = int(torch.randint(0, max(1, h - cut_h + 1), (1,), device=device).item())
    x1 = int(torch.randint(0, max(1, w - cut_w + 1), (1,), device=device).item())
    mask[:, :, y1:y1 + cut_h, x1:x1 + cut_w] = 1.0
    return mask


@torch.no_grad()
def update_ema(student, teacher, base_momentum, global_step):
    momentum = min(1.0 - 1.0 / float(global_step + 1), base_momentum)
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(momentum).add_(ps.data, alpha=1.0 - momentum)
    for bs, bt in zip(student.buffers(), teacher.buffers()):
        bt.data.copy_(bs.data)
    return momentum


def train_one_epoch(models, optimizer_map, loader, epoch, cfg, aux):
    meters = {k: AverageMeter() for k in ["loss", "loss_sup", "loss_unsup", "iou_hr", "valid_ratio"]}
    student, teacher = models["student"], models["teacher"]
    student.train(); teacher.eval()
    ema_used = cfg.ema_momentum

    for batch in loader:
        img = batch["img"].to(cfg.device)
        weak = batch["img_weak"].to(cfg.device)
        strong1 = batch["img_strong"].to(cfg.device)
        strong2 = batch["img_strong2"].to(cfg.device)
        target = batch["mask_hr"].to(cfg.device)
        b, _, h, w = strong1.shape

        box1 = sample_cutmix_box(h, w, cfg.cutmix_ratio, strong1.device).expand(b, 1, h, w)
        box2 = sample_cutmix_box(h, w, cfg.cutmix_ratio, strong1.device).expand(b, 1, h, w)
        strong1_mix = strong1.clone(); strong2_mix = strong2.clone()
        strong1_mix[box1.expand_as(strong1_mix).bool()] = strong1.flip(0)[box1.expand_as(strong1_mix).bool()]
        strong2_mix[box2.expand_as(strong2_mix).bool()] = strong2.flip(0)[box2.expand_as(strong2_mix).bool()]

        optimizer_map["student"].zero_grad()
        with torch.no_grad():
            logits_w = teacher(weak)
            prob_w = torch.sigmoid(logits_w)
            conf_w = torch.maximum(prob_w, 1.0 - prob_w)
            pseudo_w = (prob_w > 0.5).float()
            pseudo1 = pseudo_w.clone(); conf1 = conf_w.clone()
            pseudo2 = pseudo_w.clone(); conf2 = conf_w.clone()
            pseudo1[box1.bool()] = pseudo_w.flip(0)[box1.bool()]
            conf1[box1.bool()] = conf_w.flip(0)[box1.bool()]
            pseudo2[box2.bool()] = pseudo_w.flip(0)[box2.bool()]
            conf2[box2.bool()] = conf_w.flip(0)[box2.bool()]

        logits_x = student(img)
        loss_sup, logits_hr = supervised_loss_hr(logits_x, target)
        logits_s1 = student(strong1_mix)
        logits_s2 = student(strong2_mix)
        valid1 = (conf1 >= cfg.conf_thresh).float()
        valid2 = (conf2 >= cfg.conf_thresh).float()
        loss_u1 = masked_mean(F.binary_cross_entropy_with_logits(logits_s1, pseudo1, reduction="none"), valid1)
        loss_u2 = masked_mean(F.binary_cross_entropy_with_logits(logits_s2, pseudo2, reduction="none"), valid2)
        loss_unsup = 0.5 * (loss_u1 + loss_u2)
        loss = 0.5 * (loss_sup + loss_unsup)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
        optimizer_map["student"].step()

        aux["global_step"] += 1
        ema_used = update_ema(student, teacher, cfg.ema_momentum, aux["global_step"])
        bs = img.size(0)
        vals = {"loss": loss.item(), "loss_sup": loss_sup.item(), "loss_unsup": loss_unsup.item(),
                "iou_hr": binary_iou_hr(logits_hr, target), "valid_ratio": 0.5 * (valid1.mean().item() + valid2.mean().item())}
        for k, v in vals.items(): meters[k].update(v, bs)
    out = {k: v.avg for k, v in meters.items()}
    out["ema_used"] = ema_used
    return out


def parse_args():
    parser = argparse.ArgumentParser(description="Auditable UniMatch-style Sentinel weak-supervision training")
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--split-json", required=True)
    parser.add_argument("--backbone", choices=["resnet50", "resnet101"], default="resnet101")
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hr-divisor", type=float, default=4096.0)
    parser.add_argument("--mask-threshold", type=float, default=50.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--conf-thresh", type=float, default=0.95)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = WeakTrainConfig(root_dir=args.root_dir, save_dir=args.save_dir, split_json=args.split_json,
                          num_epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
                          seed=args.seed, hr_divisor=args.hr_divisor, mask_threshold=args.mask_threshold,
                          lr=args.lr, weight_decay=args.weight_decay, conf_thresh=args.conf_thresh, num_workers=args.num_workers,
                          backbone=args.backbone, pretrained_backbone=args.pretrained_backbone)
    run_loop(cfg, build_models, train_one_epoch, lambda models, loader, cfg: validate_single(models, loader, cfg, "student"))
