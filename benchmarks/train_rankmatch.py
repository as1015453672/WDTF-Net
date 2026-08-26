
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
    # 只改这里：数据、模型、训练和 RankMatch 参数都集中在一个配置中
    # =========================
    root_dir: str = "./data/train"
    hr_dir: str = "hr"
    mask_dir: str = "mask"
    hr_suffix: str = "hr"
    mask_suffix: str = "mask"

    save_dir: str = "./experiment_worldview/rankmatch"
    method: str = "rankmatch"
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
    unsup_weight: float = 1.0
    cutmix_ratio: float = 0.5
    fp_dropout: float = 0.5
    fp_noise_std: float = 0.1
    rank_corr_weight: float = 0.1
    rank_num_landmarks: int = 64
    rank_topk_permutation: int = 4

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


from itertools import permutations

# 全局配置实例：后续代码只使用这一个 cfg，避免前后两处参数不一致
cfg = WeakTrainConfig()


def build_models(in_channels, cfg):
    student = build_model(in_channels, cfg).to(cfg.device)
    return {"student": student}, {}


def _rand_bbox(h, w, ratio):
    cut_h = max(1, int(h * ratio * np.random.uniform(0.5, 1.0)))
    cut_w = max(1, int(w * ratio * np.random.uniform(0.5, 1.0)))
    y1 = np.random.randint(0, max(1, h - cut_h + 1))
    x1 = np.random.randint(0, max(1, w - cut_w + 1))
    return y1, y1 + cut_h, x1, x1 + cut_w


def cutmix_batch(img, ratio=0.5):
    b, c, h, w = img.shape
    perm = torch.randperm(b, device=img.device)
    mixed = img.clone()
    mask = torch.zeros((b, 1, h, w), device=img.device, dtype=img.dtype)
    for i in range(b):
        y1, y2, x1, x2 = _rand_bbox(h, w, ratio)
        mixed[i, :, y1:y2, x1:x2] = img[perm[i], :, y1:y2, x1:x2]
        mask[i, :, y1:y2, x1:x2] = 1.0
    return mixed, perm, mask


def prob2rank(prob_w, prob_s, k=4):
    n = prob_w.shape[-1]
    k = max(2, min(int(k), int(n)))
    perms = torch.tensor(list(permutations(range(k))), device=prob_w.device, dtype=torch.long)
    _, topk_idx = prob_w.topk(k, dim=-1)
    gather_idx = topk_idx[:, :, :, perms]
    bw = prob_w.unsqueeze(3).expand(-1, -1, -1, perms.shape[0], -1)
    bs = prob_s.unsqueeze(3).expand(-1, -1, -1, perms.shape[0], -1)
    cw = torch.gather(bw, -1, gather_idx).clamp_min(1e-10)
    cs = torch.gather(bs, -1, gather_idx).clamp_min(1e-10)
    rank_w = cw[..., 0] / cw[..., 0:].sum(dim=-1).clamp_min(1e-10)
    rank_s = cs[..., 0] / cs[..., 0:].sum(dim=-1).clamp_min(1e-10)
    for i in range(1, k):
        rank_w = rank_w * (cw[..., i] / cw[..., i:].sum(dim=-1).clamp_min(1e-10))
        rank_s = rank_s * (cs[..., i] / cs[..., i:].sum(dim=-1).clamp_min(1e-10))
    rank_w = rank_w.clamp_min(1e-10); rank_s = rank_s.clamp_min(1e-10)
    rank_w = rank_w / rank_w.sum(dim=-1, keepdim=True).clamp_min(1e-10)
    rank_s = rank_s / rank_s.sum(dim=-1, keepdim=True).clamp_min(1e-10)
    return rank_w, rank_s


def orthogonal_landmarks(q, q_s, num_landmarks=64):
    B, D, H, W = q.shape
    N = H * W
    qf = q.permute(0, 2, 3, 1).reshape(B, N, D)
    qsf = q_s.permute(0, 2, 3, 1).reshape(B, N, D)
    if N > 4096:
        sample_idx = torch.randint(N, (4096,), device=q.device)
        qk_norm = F.normalize(qf[:, sample_idx, :], p=2, dim=-1)
        source_q = qf[:, sample_idx, :]; source_qs = qsf[:, sample_idx, :]
    else:
        qk_norm = F.normalize(qf, p=2, dim=-1)
        source_q = qf; source_qs = qsf
    K = qk_norm.size(1)
    selected_mask = torch.zeros((B, K, 1), device=q.device)
    random_idx = torch.randint(K, (B, 1, 1), device=q.device)
    selected = qk_norm[torch.arange(B, device=q.device), random_idx.view(-1), :].view(B, D)
    selected_mask.scatter_(-2, random_idx, 1.0)
    cos_sims = torch.empty((B, K, num_landmarks), device=q.device, dtype=q.dtype)
    for m in range(1, num_landmarks):
        cos = torch.einsum("bnd,bd->bn", qk_norm, selected).abs()
        cos_sims[:, :, m - 1] = cos
        cos_set = cos_sims[:, :, :m]
        cos_set.view(-1, m)[selected_mask.flatten().bool(), :] = 10
        idx = cos_set.amax(-1).argmin(-1)
        selected = qk_norm[torch.arange(B, device=q.device), idx, :].view(B, D)
        selected_mask.scatter_(-2, idx.view(B, 1, 1), 1.0)
    landmarks = torch.masked_select(source_q, selected_mask.bool()).reshape(B, -1, D)
    landmarks_s = torch.masked_select(source_qs, selected_mask.bool()).reshape(B, -1, D)
    return landmarks, landmarks_s


def corr_rank_loss(feat_w, feat_s, num_landmarks=64, topk=4):
    if not torch.isfinite(feat_w).all() or not torch.isfinite(feat_s).all():
        return feat_w.new_tensor(0.0)
    refers_w, refers_s = orthogonal_landmarks(feat_w, feat_s, num_landmarks=num_landmarks)
    p2r_w = torch.einsum("bchw,bnc->bhwn", feat_w, refers_w).softmax(dim=-1)
    p2r_s = torch.einsum("bchw,bnc->bhwn", feat_s, refers_s).softmax(dim=-1)
    rank_w, rank_s = prob2rank(p2r_w, p2r_s, k=topk)
    # Average over spatial locations/permutations.  ``batchmean`` summed all
    # full-resolution pixels and produced million-scale losses.
    loss = F.kl_div(rank_s.clamp_min(1e-10).log(), rank_w.clamp_min(1e-10), reduction="none").mean()
    return loss if torch.isfinite(loss) else feat_w.new_tensor(0.0)


def perturb_feature(feat, dropout_p=0.5, noise_std=0.1):
    if dropout_p > 0:
        mask = (torch.rand_like(feat) > dropout_p).float() / max(1e-6, 1.0 - dropout_p)
        feat = feat * mask
    if noise_std > 0:
        feat = feat + torch.randn_like(feat) * noise_std
    return feat


def train_one_epoch(models, optimizer_map, loader, epoch, cfg, aux):
    meters = {k: AverageMeter() for k in ["loss", "loss_sup", "loss_unsup", "loss_rank", "iou_hr", "valid_ratio"]}
    model = models["student"]
    model.train()

    for batch in loader:
        img = batch["img"].to(cfg.device)
        weak = batch["img_weak"].to(cfg.device)
        strong1 = batch["img_strong"].to(cfg.device)
        strong2 = batch["img_strong2"].to(cfg.device)
        target = batch["mask_hr"].to(cfg.device)

        weak_mix, perm_w, mixmask_w = cutmix_batch(weak, ratio=cfg.cutmix_ratio)
        strong1_mix, perm_s1, mixmask_s1 = cutmix_batch(strong1, ratio=cfg.cutmix_ratio)
        strong2_mix, perm_s2, mixmask_s2 = cutmix_batch(strong2, ratio=cfg.cutmix_ratio)

        optimizer_map["student"].zero_grad()
        with torch.no_grad():
            model.eval()
            logits_w_mix, feat_w_mix = model(weak_mix, return_feat=True, normalize_feat=False)
            prob_w_mix = torch.softmax(torch.cat([torch.zeros_like(logits_w_mix), logits_w_mix], dim=1), dim=1)
            conf_w_mix = prob_w_mix.max(dim=1)[0]
            pseudo_w_mix = prob_w_mix.argmax(dim=1)
        model.train()

        sup_logits = model(img)
        loss_sup, logits_hr = supervised_loss_hr(sup_logits, target)
        logits_w, feat_w = model(weak, return_feat=True, normalize_feat=False)
        logits_s1, feat_s1 = model(strong1_mix, return_feat=True, normalize_feat=False)
        logits_s2, feat_s2 = model(strong2_mix, return_feat=True, normalize_feat=False)

        feat_w_fp = perturb_feature(feat_w, cfg.fp_dropout, cfg.fp_noise_std)
        logits_w_fp = model.logits_from_feat(feat_w_fp)
        prob_w = torch.softmax(torch.cat([torch.zeros_like(logits_w), logits_w], dim=1), dim=1).detach()
        conf_w = prob_w.max(dim=1)[0]
        pseudo_w = prob_w.argmax(dim=1)

        pseudo_s1 = pseudo_w.clone(); conf_s1 = conf_w.clone()
        pseudo_s2 = pseudo_w.clone(); conf_s2 = conf_w.clone()
        for i in range(pseudo_s1.shape[0]):
            m1 = mixmask_s1[i, 0].bool(); m2 = mixmask_s2[i, 0].bool()
            pseudo_s1[i][m1] = pseudo_w_mix[perm_s1[i]][m1]
            conf_s1[i][m1] = conf_w_mix[perm_s1[i]][m1]
            pseudo_s2[i][m2] = pseudo_w_mix[perm_s2[i]][m2]
            conf_s2[i][m2] = conf_w_mix[perm_s2[i]][m2]

        valid_s1 = (conf_s1 >= cfg.conf_thresh).float()
        valid_s2 = (conf_s2 >= cfg.conf_thresh).float()
        valid_w = (conf_w >= cfg.conf_thresh).float()
        loss_u_s1 = masked_mean(F.binary_cross_entropy_with_logits(logits_s1, pseudo_s1.unsqueeze(1).float(), reduction="none"), valid_s1.unsqueeze(1))
        loss_u_s2 = masked_mean(F.binary_cross_entropy_with_logits(logits_s2, pseudo_s2.unsqueeze(1).float(), reduction="none"), valid_s2.unsqueeze(1))
        loss_u_fp = masked_mean(F.binary_cross_entropy_with_logits(logits_w_fp, pseudo_w.unsqueeze(1).float(), reduction="none"), valid_w.unsqueeze(1))

        feat_w_s1 = feat_w.detach(); feat_w_s2 = feat_w.detach()
        feat_mask1 = F.interpolate(mixmask_s1, size=feat_w.shape[-2:], mode="nearest").expand_as(feat_w_s1)
        feat_mask2 = F.interpolate(mixmask_s2, size=feat_w.shape[-2:], mode="nearest").expand_as(feat_w_s2)
        feat_w_mix_up = F.interpolate(feat_w_mix, size=feat_w.shape[-2:], mode="bilinear", align_corners=False)
        feat_w_s1 = torch.where(feat_mask1.bool(), feat_w_mix_up[perm_s1], feat_w_s1)
        feat_w_s2 = torch.where(feat_mask2.bool(), feat_w_mix_up[perm_s2], feat_w_s2)
        # Rank correlation is defined on a compact feature grid.  The local
        # DeepLab wrapper upsamples decoder features to 512x512; running
        # landmark ranking at that resolution is prohibitively slow and
        # inconsistent with the official low-resolution feature operation.
        rank_size = min(32, feat_w_s1.shape[-1], feat_w_s1.shape[-2])
        if feat_w_s1.shape[-2:] != (rank_size, rank_size):
            feat_w_s1 = F.adaptive_avg_pool2d(feat_w_s1, (rank_size, rank_size))
            feat_s1 = F.adaptive_avg_pool2d(feat_s1, (rank_size, rank_size))
            feat_w_s2 = F.adaptive_avg_pool2d(feat_w_s2, (rank_size, rank_size))
            feat_s2 = F.adaptive_avg_pool2d(feat_s2, (rank_size, rank_size))
        loss_rank = 0.5 * (
            corr_rank_loss(feat_w_s1, feat_s1, cfg.rank_num_landmarks, cfg.rank_topk_permutation) +
            corr_rank_loss(feat_w_s2, feat_s2, cfg.rank_num_landmarks, cfg.rank_topk_permutation)
        )
        if not torch.isfinite(loss_rank):
            loss_rank = img.new_tensor(0.0)

        loss_unsup = (0.25 * loss_u_s1 + 0.25 * loss_u_s2 + 0.5 * loss_u_fp) / 2.0
        loss = 0.5 * loss_sup + cfg.unsup_weight * loss_unsup + cfg.rank_corr_weight * loss_rank
        if not torch.isfinite(loss):
            optimizer_map["student"].zero_grad(set_to_none=True)
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer_map["student"].step()

        bs = img.size(0)
        vals = {"loss": loss.item(), "loss_sup": loss_sup.item(), "loss_unsup": loss_unsup.item(),
                "loss_rank": loss_rank.item(), "iou_hr": binary_iou_hr(logits_hr, target),
                "valid_ratio": (valid_w.mean().item() + valid_s1.mean().item() + valid_s2.mean().item()) / 3.0}
        for k, v in vals.items(): meters[k].update(v, bs)
    return {k: v.avg for k, v in meters.items()}


def parse_args():
    parser = argparse.ArgumentParser(description="Auditable RankMatch-style Sentinel weak-supervision training")
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
