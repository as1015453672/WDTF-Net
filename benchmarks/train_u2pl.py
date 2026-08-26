
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import json
import random
import time
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from common import *


@dataclass
class WeakTrainConfig:
    # =========================
    # 只改这里：数据、模型、训练和 RankMatch 参数都集中在一个配置中
    # =========================
    root_dir: str = "./data/train"
    hr_dir: str = "hr"
    mask_dir: str = "mask"
    hr_suffix: str = "hr"
    mask_suffix: str = "mask"

    save_dir: str = "./experiment_worldview/u2pl"
    method: str = "u2pl"
    backbone: str = "resnet101"  # resnet50 / resnet101

    # common.py 中 build_model / make_optimizer / ramp_weight 需要的通用字段
    # 这些字段统一放在这里，后面不再二次覆盖，避免配置分散。
    output_stride: int = 16
    pretrained_backbone: bool = False
    sup_only_epochs: int = 0
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
    pos_weight: float = 1.0
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
    unsup_weight: float = 1.0
    cutmix_ratio: float = 0.5
    fp_dropout: float = 0.5
    fp_noise_std: float = 0.1
    rank_corr_weight: float = 0.1
    rank_num_landmarks: int = 64
    rank_topk_permutation: int = 4

    # TorchSemiSeg / CPS-style parameters
    cps_weight: float = 1.0
    cutmix_box_ratio: float = 0.5

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
    samples = find_samples(cfg)
    if cfg.split_json:
        split = json.loads(Path(cfg.split_json).read_text(encoding="utf-8"))
        by_id = {sample["id"]: sample for sample in samples}
        train_ids = list(split.get("train", [])); val_ids = list(split.get("val", split.get("validation", [])))
        missing = [sid for sid in train_ids + val_ids if sid not in by_id]
        if missing:
            raise RuntimeError(f"split_json contains ids missing from dataset: {missing[:5]}")
        return [by_id[sid] for sid in train_ids], [by_id[sid] for sid in val_ids]
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
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=len(train_samples) > 1)
    val_loader = DataLoader(WeakWaterDataset(val_samples, cfg, False), batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, val_loader


def dice_loss_with_logits(logits, target, eps=1e-6):
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(1, 2, 3))
    den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1.0 - ((2.0 * inter + eps) / (den + eps)).mean()


def supervised_loss_hr(logits, target, pos_weight: float = 1.0):
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    if target.ndim == 3:
        target = target.unsqueeze(1)
    if target.shape[-2:] != logits.shape[-2:]:
        target = F.interpolate(target.float(), size=logits.shape[-2:], mode="nearest")
    if pos_weight and pos_weight != 1.0:
        weight = logits.new_tensor([float(pos_weight)])
        bce = F.binary_cross_entropy_with_logits(logits, target.float(), pos_weight=weight)
    else:
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
    water_iou = tp / (tp + fp + fn + eps)
    background_iou = tn / (tn + fp + fn + eps)
    return (0.5 * (water_iou + background_iou)).mean().item()


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
            loss, logits_hr = supervised_loss_hr(logits, target, cfg.pos_weight)
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
            loss1, logits1 = supervised_loss_hr(logits1, target, cfg.pos_weight)
            loss2, logits2 = supervised_loss_hr(logits2, target, cfg.pos_weight)
            avg_logits = 0.5 * (logits1 + logits2)
            iou = binary_iou_hr(avg_logits, target)
            bs = img.size(0)
            loss_meter.update(0.5 * (loss1.item() + loss2.item()), bs); iou_meter.update(iou, bs)
    return {"loss": loss_meter.avg, "iou_hr": iou_meter.avg}


def run_loop(cfg, build_models_fn, train_one_epoch_fn, validate_fn):
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)
    train_loader, val_loader = build_weak_loaders(cfg)
    sample = next(iter(train_loader))
    in_channels = sample["img"].shape[1]
    models, aux = build_models_fn(in_channels, cfg)
    total_iters = max(1, cfg.num_epochs * len(train_loader))
    optimizers = {k: make_optimizer(v, cfg, total_iters) for k, v in models.items() if any(p.requires_grad for p in v.parameters())}

    print(f"method={cfg.method} backbone={cfg.backbone} input_channels={in_channels} device={cfg.device}")
    print(f"root={cfg.root_dir} train={len(train_loader.dataset)} val={len(val_loader.dataset)} save_dir={cfg.save_dir}")
    Path(cfg.save_dir, "config.json").write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")

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



class CategoryMemoryBank:
    def __init__(self, num_classes=2, feat_dim=256, queue_size=512):
        from collections import deque
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.queue_size = queue_size
        self.queues = {c: deque(maxlen=queue_size) for c in range(num_classes)}

    @torch.no_grad()
    def enqueue(self, class_id, feats):
        if feats is None or feats.numel() == 0:
            return
        feats = F.normalize(feats.detach(), dim=1).cpu()
        for v in feats:
            self.queues[class_id].append(v)

    def sample(self, class_id, num_samples, device):
        q = self.queues[class_id]
        if len(q) == 0:
            return None
        num = min(num_samples, len(q))
        idx = torch.randperm(len(q))[:num].tolist()
        out = torch.stack([q[i] for i in idx], dim=0).to(device)
        return F.normalize(out, dim=1)


def build_models(in_channels, cfg):
    student = build_model(in_channels, cfg).to(cfg.device)
    teacher = build_model(in_channels, cfg).to(cfg.device)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False
    aux = {"bank": CategoryMemoryBank(num_classes=2, feat_dim=256, queue_size=cfg.u2pl_queue_size)}
    return {"student": student, "teacher": teacher}, aux


def compute_entropy(prob):
    prob = prob.clamp(1e-6, 1 - 1e-6)
    return -(prob * torch.log(prob) + (1 - prob) * torch.log(1 - prob))


@torch.no_grad()
def get_reliable_masks(prob, epoch, cfg):
    entropy = compute_entropy(prob)
    alpha_t = cfg.u2pl_alpha0 * (1.0 - float(epoch - 1) / max(1, cfg.num_epochs))
    gamma = torch.quantile(entropy.reshape(-1), 1.0 - alpha_t).item()
    reliable = (entropy < gamma).float()
    pseudo = (prob > 0.5).float()
    return pseudo, entropy, reliable, gamma, alpha_t


@torch.no_grad()
def compute_binary_rank(prob):
    rank1 = (prob <= 0.5).long()
    rank0 = (prob > 0.5).long()
    return rank0, rank1


@torch.no_grad()
def gather_anchor_and_negative_sets(teacher_feat, teacher_prob, pseudo, reliable_mask, target_hr, cfg):
    feat_hw = teacher_feat.shape[-2:]
    label_feat = F.interpolate(target_hr.float(), size=feat_hw, mode="nearest")
    prob_feat = F.interpolate(teacher_prob, size=feat_hw, mode="bilinear", align_corners=False)
    pseudo_feat = F.interpolate(pseudo.float(), size=feat_hw, mode="nearest")
    reliable_feat = F.interpolate(reliable_mask.float(), size=feat_hw, mode="nearest") > 0.5
    entropy_feat = compute_entropy(prob_feat)
    gamma = torch.quantile(entropy_feat.reshape(-1), 1.0 - cfg.u2pl_alpha0 * 0.5).item()
    unreliable_feat = entropy_feat > gamma

    feat_flat = teacher_feat.permute(0, 2, 3, 1).reshape(-1, teacher_feat.shape[1])
    feat_flat = F.normalize(feat_flat, dim=1)
    label_flat = label_feat.reshape(-1)
    pseudo_flat = pseudo_feat.reshape(-1)
    reliable_flat = reliable_feat.reshape(-1)
    prob_flat = prob_feat.reshape(-1)
    rank0, rank1 = compute_binary_rank(prob_feat)
    rank0 = rank0.reshape(-1)
    rank1 = rank1.reshape(-1)
    unreliable_flat = unreliable_feat.reshape(-1)

    out = {}
    for c in [0, 1]:
        prob_c = prob_flat if c == 1 else (1.0 - prob_flat)
        rank_c = rank1 if c == 1 else rank0
        labeled_anchors = (label_flat == float(c)) & (prob_c > cfg.u2pl_delta_p)
        unlabeled_anchors = (pseudo_flat == float(c)) & reliable_flat & (prob_c > cfg.u2pl_delta_p)
        anchor_mask = labeled_anchors | unlabeled_anchors
        pos_center = feat_flat[anchor_mask]
        labeled_neg = (label_flat != float(c)) & (rank_c < cfg.u2pl_low_rank)
        unlabeled_neg = unreliable_flat & (rank_c >= cfg.u2pl_low_rank) & (rank_c < cfg.u2pl_high_rank)
        neg_mask = labeled_neg | unlabeled_neg
        neg_feats = feat_flat[neg_mask]
        out[c] = {"anchor_mask": anchor_mask, "positive_center_feats": pos_center, "negative_feats": neg_feats}
    return out


def contrastive_u2pl_loss(student_feat, teacher_sets, bank, cfg):
    feat_flat = student_feat.permute(0, 2, 3, 1).reshape(-1, student_feat.shape[1])
    feat_flat = F.normalize(feat_flat, dim=1)
    total_loss = student_feat.new_tensor(0.0)
    used_classes = 0
    for c in [0, 1]:
        cur = teacher_sets[c]
        anchor_idx = torch.nonzero(cur["anchor_mask"], as_tuple=False).flatten()
        if anchor_idx.numel() == 0 or cur["positive_center_feats"].numel() == 0:
            continue
        bank.enqueue(c, cur["negative_feats"])
        negatives = bank.sample(c, cfg.u2pl_num_negatives, student_feat.device)
        if negatives is None or negatives.numel() == 0:
            continue
        num_anchors = min(cfg.u2pl_num_anchors, anchor_idx.numel())
        sel = anchor_idx[torch.randperm(anchor_idx.numel(), device=anchor_idx.device)[:num_anchors]]
        anchors = feat_flat[sel]
        pos_center = F.normalize(cur["positive_center_feats"].mean(dim=0, keepdim=True).to(student_feat.device), dim=1)
        pos_logit = torch.sum(anchors * pos_center, dim=1, keepdim=True) / cfg.contrast_temperature
        neg_logits = anchors @ negatives.t() / cfg.contrast_temperature
        logits = torch.cat([pos_logit, neg_logits], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        total_loss = total_loss + F.cross_entropy(logits, labels)
        used_classes += 1
    if used_classes == 0:
        return student_feat.new_tensor(0.0)
    return total_loss / used_classes


def train_one_epoch(models, optimizer_map, loader, epoch, cfg, aux):
    meters = {k: AverageMeter() for k in ["loss", "loss_sup", "loss_unsup", "loss_rank", "loss_edge", "loss_proto", "loss_neg", "iou_hr", "valid_ratio"]}
    models["student"].train()
    models["teacher"].eval()
    bank = aux["bank"]

    for batch in loader:
        img = batch["img"].to(cfg.device)
        weak = batch["img_weak"].to(cfg.device)
        strong = batch["img_strong"].to(cfg.device)
        target = batch["mask_hr"].to(cfg.device)

        optimizer_map["student"].zero_grad()
        with torch.no_grad():
            t_logits, t_feat = models["teacher"](weak, return_feat=True)
            t_prob = torch.sigmoid(t_logits)
            pseudo, entropy, reliable, gamma, alpha_t = get_reliable_masks(t_prob, epoch, cfg)
            teacher_sets = gather_anchor_and_negative_sets(t_feat, t_prob, pseudo, reliable, target, cfg)

        s_logits, s_feat = models["student"](strong, return_feat=True)
        loss_sup, logits_hr = supervised_loss_hr(s_logits, target, cfg.pos_weight)
        unsup_map = F.binary_cross_entropy_with_logits(s_logits, pseudo, reduction="none")
        reliable_count = max(float(reliable.sum().item()), 1.0)
        total_count = float(reliable.numel())
        lambda_u = cfg.unsup_weight * (total_count / reliable_count)
        loss_unsup = masked_mean(unsup_map, reliable)
        loss_contrast = contrastive_u2pl_loss(s_feat, teacher_sets, bank, cfg)
        if epoch <= cfg.sup_only_epochs:
            aux_unsup_weight = 0.0
            aux_contrast_weight = 0.0
        else:
            ramp_epoch = epoch - cfg.sup_only_epochs
            aux_unsup_weight = ramp_weight(ramp_epoch, cfg, lambda_u)
            aux_contrast_weight = ramp_weight(ramp_epoch, cfg, cfg.u2pl_contrast_weight)
        loss = loss_sup + aux_unsup_weight * loss_unsup + aux_contrast_weight * loss_contrast
        loss.backward()
        torch.nn.utils.clip_grad_norm_(models["student"].parameters(), max_norm=cfg.grad_clip)
        optimizer_map["student"].step()
        update_ema(models["student"], models["teacher"], cfg.ema_momentum)

        bs = img.size(0)
        vals = {
            "loss": loss.item(), "loss_sup": loss_sup.item(), "loss_unsup": loss_unsup.item(),
            "loss_rank": 0.0, "loss_edge": 0.0, "loss_proto": loss_contrast.item(), "loss_neg": 0.0,
            "iou_hr": binary_iou_hr(logits_hr, target), "valid_ratio": reliable.mean().item(),
        }
        for k, v in vals.items():
            meters[k].update(v, bs)
    return {k: v.avg for k, v in meters.items()}


def parse_args():
    parser = argparse.ArgumentParser(description="Auditable U2PL training for Sentinel weak supervision")
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
    parser.add_argument("--pos-weight", type=float, default=1.0,
                        help="Positive-class BCE weight; 1 disables balancing.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--rampup-epochs", type=int, default=5,
                        help="Warm up pseudo-label and contrastive terms; 0 reproduces the legacy immediate start.")
    parser.add_argument("--unsup-weight", type=float, default=0.5)
    parser.add_argument("--contrast-weight", type=float, default=0.05)
    parser.add_argument("--ema-momentum", type=float, default=0.99)
    parser.add_argument("--sup-only-epochs", type=int, default=0,
                        help="Run supervised-only epochs before pseudo-label and contrastive losses.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = replace(WeakTrainConfig(), root_dir=args.root_dir, save_dir=args.save_dir,
                  split_json=args.split_json, num_epochs=args.epochs, patience=args.patience,
                  backbone=args.backbone, pretrained_backbone=args.pretrained_backbone,
                  batch_size=args.batch_size, seed=args.seed, hr_divisor=args.hr_divisor,
                  mask_threshold=args.mask_threshold, lr=args.lr, weight_decay=args.weight_decay, pos_weight=args.pos_weight, num_workers=args.num_workers,
                  rampup_epochs=args.rampup_epochs, unsup_weight=args.unsup_weight,
                  u2pl_contrast_weight=args.contrast_weight, ema_momentum=args.ema_momentum,
                  sup_only_epochs=args.sup_only_epochs)
    run_loop(cfg, build_models, train_one_epoch, lambda models, loader, cfg: validate_single(models, loader, cfg, "student"))
