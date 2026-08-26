from __future__ import annotations

import argparse
import math
import os
import random
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
import warnings
import json
warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Dataset
# -----------------------------
@dataclass
class WeakDataConfig:
    root_dir: str
    hr_dir: str = 'hr'
    mask_dir: str = 'mask'
    hr_suffix: str = 'hr'
    mask_suffix: str = 'mask'
    val_ratio: float = 0.1
    seed: int = 42
    batch_size: int = 8
    num_workers: int = 0
    pin_memory: bool = True
    hr_divisor: float = 1024.0
    mask_threshold: float = 20.0
    mask_binarize: bool = True
    boundary_kernel: int = 9
    boundary_weight: float = 0.35
    interior_weight: float = 1.0
    min_confidence: float = 0.15
    split_json: str = ''
    ignore_value: float | None = None


def _strip_known_suffix(name_no_ext: str, suffix: str) -> str:
    token = '_' + suffix
    if name_no_ext.endswith(token):
        return name_no_ext[:-len(token)]
    if name_no_ext.endswith(suffix):
        return name_no_ext[:-len(suffix)]
    return name_no_ext


def _find_samples(cfg: WeakDataConfig) -> List[Dict[str, str]]:
    hr_root = Path(cfg.root_dir) / cfg.hr_dir
    mask_root = Path(cfg.root_dir) / cfg.mask_dir
    if not hr_root.exists():
        raise FileNotFoundError(f'hr folder not found: {hr_root}')
    if not mask_root.exists():
        raise FileNotFoundError(f'mask folder not found: {mask_root}')

    hr_map = {_strip_known_suffix(p.stem, cfg.hr_suffix): str(p) for p in hr_root.iterdir() if p.is_file()}
    mask_map = {_strip_known_suffix(p.stem, cfg.mask_suffix): str(p) for p in mask_root.iterdir() if p.is_file()}
    common_ids = sorted(set(hr_map) & set(mask_map))
    if not common_ids:
        raise RuntimeError(
            f'No matched samples found. Expected names like xxx_{cfg.hr_suffix}.tif / xxx_{cfg.mask_suffix}.tif'
        )
    return [{'id': sid, 'hr_path': hr_map[sid], 'mask_path': mask_map[sid]} for sid in common_ids]


def split_samples(cfg: WeakDataConfig) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    samples = _find_samples(cfg)
    if cfg.split_json:
        split = json.loads(Path(cfg.split_json).read_text(encoding='utf-8'))
        by_id = {sample['id']: sample for sample in samples}
        train_ids = split.get('train', [])
        val_ids = split.get('val', [])
        missing = [sid for sid in train_ids + val_ids if sid not in by_id]
        if missing:
            raise RuntimeError(f'split_json contains ids missing from dataset: {missing[:5]}')
        return [by_id[sid] for sid in train_ids], [by_id[sid] for sid in val_ids]
    rng = random.Random(cfg.seed)
    rng.shuffle(samples)
    n_total = len(samples)
    n_val = max(1, int(round(n_total * cfg.val_ratio))) if n_total > 1 else 0
    val_samples = samples[:n_val]
    train_samples = samples[n_val:] if n_total > 1 else samples
    return train_samples, val_samples


def read_raster(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read()
    return arr.astype(np.float32)


def normalize_image(arr: np.ndarray, divisor: float) -> np.ndarray:
    return arr.astype(np.float32) / float(divisor)


def process_mask(mask: np.ndarray, threshold: float, binarize: bool = True) -> np.ndarray:
    if binarize:
        mask = (mask > threshold).astype(np.float32)
    else:
        mask = mask.astype(np.float32) / 100.0
    return mask


def _make_boundary_band(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size % 2 == 0:
        kernel_size += 1
    if mask.ndim == 3:
        mask = mask.unsqueeze(0)
    pad = kernel_size // 2
    dil = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad)
    ero = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=kernel_size, stride=1, padding=pad)
    band = (dil - ero).clamp(0.0, 1.0)
    return band.squeeze(0)


class WeakHRDataset(Dataset):
    def __init__(self, samples: List[Dict[str, str]], cfg: WeakDataConfig):
        self.samples = samples
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        hr = read_raster(s['hr_path'])
        raw_mask = read_raster(s['mask_path'])
        hr = normalize_image(hr, self.cfg.hr_divisor)
        valid_mask = np.ones_like(raw_mask, dtype=np.float32) if self.cfg.ignore_value is None else \
            (~np.isclose(raw_mask, self.cfg.ignore_value)).astype(np.float32)
        weak_mask = process_mask(raw_mask, self.cfg.mask_threshold, self.cfg.mask_binarize) * valid_mask

        img = torch.from_numpy(hr)
        mask = torch.from_numpy(weak_mask)
        valid = torch.from_numpy(valid_mask)
        hard_mask = (mask > 0.5).float()

        boundary = _make_boundary_band(hard_mask, self.cfg.boundary_kernel) * valid
        conf = torch.full_like(mask, float(self.cfg.interior_weight))
        conf = (conf * (1.0 - boundary) + float(self.cfg.boundary_weight) * boundary) * valid
        conf = torch.where(valid > 0, conf.clamp(min=float(self.cfg.min_confidence), max=1.0), torch.zeros_like(conf))

        return {
            'id': s['id'],
            'img': img,
            'mask': mask,
            'hard_mask': hard_mask,
            'conf': conf,
            'boundary': boundary,
            'valid': valid,
        }


def build_loaders(cfg: WeakDataConfig) -> Tuple[DataLoader, DataLoader]:
    train_samples, val_samples = split_samples(cfg)
    train_ds = WeakHRDataset(train_samples, cfg)
    val_ds = WeakHRDataset(val_samples, cfg)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    return train_loader, val_loader


# -----------------------------
# Model
# -----------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBNReLU(c_in, c_out)

    def forward(self, x):
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, c_in: int, c_skip: int, c_out: int):
        super().__init__()
        self.conv = ConvBNReLU(c_in + c_skip, c_out)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class PCREWeakNet(nn.Module):
    """
    Adaptation of WSSS-PCRE for noisy weak masks:
    - progressive confidence region expansion (PCR expansion)
    - class-prototype enhancement (CPE)
    - binary water segmentation
    """
    def __init__(self, in_channels: int = 3, feat_dim: int = 128):
        super().__init__()
        self.stem = ConvBNReLU(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.up2 = Up(512, 256, 256)
        self.up1 = Up(256, 128, 128)
        self.up0 = Up(128, 64, 64)
        self.embed = nn.Sequential(
            nn.Conv2d(64, feat_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )
        self.mask_head = nn.Conv2d(feat_dim, 1, 1)
        self.region_head = nn.Conv2d(feat_dim, 1, 1)

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        y2 = self.up2(x3, x2)
        y1 = self.up1(y2, x1)
        y0 = self.up0(y1, x0)
        feat = self.embed(y0)
        logits = self.mask_head(feat)
        region_logits = self.region_head(feat)
        return {
            'logits': logits,
            'region_logits': region_logits,
            'feat': feat,
        }


# -----------------------------
# Losses and utilities
# -----------------------------
def sigmoid_dice_loss(logits: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    target = target.float()
    if weight is None:
        weight = torch.ones_like(target)
    prob = prob * weight
    target = target * weight
    inter = (prob * target).sum(dim=(1, 2, 3))
    den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1e-6
    return (1.0 - (2.0 * inter + 1e-6) / den).mean()


def weighted_bce_loss(logits: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target.float(), reduction='none')
    if weight is not None:
        loss = loss * weight
        return loss.sum() / (weight.sum() + 1e-6)
    return loss.mean()


def compute_metrics(prob: torch.Tensor, target: torch.Tensor, valid: Optional[torch.Tensor] = None, thr: float = 0.5) -> Dict[str, float]:
    pred = (prob >= thr).float()
    target = (target > 0.5).float()
    keep = torch.ones_like(target) if valid is None else valid.float()
    tp = (pred * target * keep).sum().item()
    fp = (pred * (1 - target) * keep).sum().item()
    fn = ((1 - pred) * target * keep).sum().item()
    tn = ((1 - pred) * (1 - target) * keep).sum().item()
    iou = tp / (tp + fp + fn + 1e-6)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    return {'iou': iou, 'dice': dice, 'acc': acc, 'recall': recall, 'precision': precision}


def update_ema(student: nn.Module, teacher: nn.Module, momentum: float = 0.99):
    with torch.no_grad():
        for ps, pt in zip(student.parameters(), teacher.parameters()):
            pt.data.mul_(momentum).add_(ps.data, alpha=1.0 - momentum)


def confidence_schedule(epoch: int, total_epochs: int, start: float, end: float) -> float:
    ratio = min(max(epoch / max(total_epochs - 1, 1), 0.0), 1.0)
    return start + (end - start) * ratio


def build_progressive_targets(
    teacher_prob: torch.Tensor,
    weak_mask: torch.Tensor,
    conf_map: torch.Tensor,
    proto_sim: torch.Tensor,
    epoch: int,
    total_epochs: int,
    start_conf: float,
    end_conf: float,
    proto_mix: float,
    available: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Progressive Confidence Region Expansion (adapted):
    - keep original weak labels as anchors
    - progressively add confident teacher regions
    - use prototype similarity to suppress false expansion
    """
    conf_thr = confidence_schedule(epoch, total_epochs, start_conf, end_conf)
    teacher_fg = teacher_prob >= conf_thr
    teacher_bg = teacher_prob <= (1.0 - conf_thr)

    proto_prob = torch.sigmoid(proto_sim)
    fused_conf = (1.0 - proto_mix) * teacher_prob + proto_mix * proto_prob
    fused_fg = fused_conf >= conf_thr
    fused_bg = fused_conf <= (1.0 - conf_thr)

    known = torch.ones_like(weak_mask, dtype=torch.bool) if available is None else available > 0
    known_fg = (weak_mask > 0.5) & known
    known_bg = (weak_mask <= 0.5) & known

    # Expand from high-confidence regions, but avoid low-confidence weak boundaries.
    expand_fg = fused_fg & (conf_map >= conf_map.mean()) & known
    expand_bg = fused_bg & (conf_map >= conf_map.mean()) & known

    pseudo = weak_mask.clone()
    valid = torch.zeros_like(weak_mask)

    pseudo[known_fg] = 1.0
    pseudo[known_bg] = 0.0
    valid[known_fg | known_bg] = 1.0

    add_fg = expand_fg & (~known_fg)
    add_bg = expand_bg & (~known_fg)
    pseudo[add_fg] = 1.0
    pseudo[add_bg] = 0.0
    valid[add_fg | add_bg] = 1.0
    return pseudo, valid


def extract_batch_prototypes(feat: torch.Tensor, prob: torch.Tensor, valid_weight: torch.Tensor, eps: float = 1e-6):
    """Class-Prototype Enhancement (adapted to binary case)."""
    fg_w = (prob * valid_weight).detach()
    bg_w = ((1.0 - prob) * valid_weight).detach()
    fg_proto = (feat * fg_w).sum(dim=(0, 2, 3)) / (fg_w.sum(dim=(0, 2, 3)) + eps)
    bg_proto = (feat * bg_w).sum(dim=(0, 2, 3)) / (bg_w.sum(dim=(0, 2, 3)) + eps)
    fg_proto = F.normalize(fg_proto, dim=0)
    bg_proto = F.normalize(bg_proto, dim=0)
    return fg_proto, bg_proto


def prototype_similarity_map(feat: torch.Tensor, fg_proto: torch.Tensor, bg_proto: torch.Tensor) -> torch.Tensor:
    feat_n = F.normalize(feat, dim=1)
    fg_sim = (feat_n * fg_proto.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
    bg_sim = (feat_n * bg_proto.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
    return fg_sim - bg_sim


def maybe_fix_omp_env():
    omp = os.environ.get('OMP_NUM_THREADS', '').strip()
    if omp:
        try:
            if int(omp) <= 0:
                raise ValueError
        except Exception:
            os.environ['OMP_NUM_THREADS'] = '1'
    else:
        os.environ['OMP_NUM_THREADS'] = '1'


def format_eta(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f'{h:02d}:{m:02d}:{s:02d}'


# -----------------------------
# Train / validate
# -----------------------------
@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    total_loss = 0.0
    n = 0
    meter = {'iou': 0.0, 'dice': 0.0, 'acc': 0.0, 'recall': 0.0, 'precision': 0.0}
    for batch in loader:
        img = batch['img'].to(device, non_blocking=True)
        mask = batch['hard_mask'].to(device, non_blocking=True)
        conf = batch['conf'].to(device, non_blocking=True)
        valid = batch['valid'].to(device, non_blocking=True)
        out = model(img)
        logits = out['logits']
        loss = weighted_bce_loss(logits, mask, conf) + sigmoid_dice_loss(logits, mask, conf)
        prob = torch.sigmoid(logits)
        metrics = compute_metrics(prob, mask, valid)
        bs = img.size(0)
        total_loss += loss.item() * bs
        n += bs
        for k in meter:
            meter[k] += metrics[k] * bs
    if n == 0:
        return {'loss': 0.0, **meter}
    return {'loss': total_loss / n, **{k: v / n for k, v in meter.items()}}


def train_one_epoch(
    epoch: int,
    args,
    model: nn.Module,
    teacher: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    global_proto_fg: torch.Tensor,
    global_proto_bg: torch.Tensor,
):
    model.train()
    total = 0.0
    stat = {'sup': 0.0, 'dice': 0.0, 'region': 0.0, 'proto': 0.0, 'cons': 0.0, 'ent': 0.0, 'iou': 0.0}
    n = 0

    for batch in loader:
        img = batch['img'].to(device, non_blocking=True)
        weak_mask = batch['mask'].to(device, non_blocking=True)
        hard_mask = batch['hard_mask'].to(device, non_blocking=True)
        conf = batch['conf'].to(device, non_blocking=True)
        boundary = batch['boundary'].to(device, non_blocking=True)
        valid = batch['valid'].to(device, non_blocking=True)

        out = model(img)
        logits = out['logits']
        region_logits = out['region_logits']
        feat = out['feat']
        prob = torch.sigmoid(logits)
        region_prob = torch.sigmoid(region_logits)

        with torch.no_grad():
            teacher_prob = torch.sigmoid(teacher(img)['logits'])

        # Class-Prototype Enhancement: local prototypes + EMA global prototypes
        batch_fg_proto, batch_bg_proto = extract_batch_prototypes(feat, weak_mask, conf)
        mix_m = args.proto_momentum
        global_proto_fg = F.normalize(global_proto_fg * mix_m + batch_fg_proto.detach() * (1.0 - mix_m), dim=0)
        global_proto_bg = F.normalize(global_proto_bg * mix_m + batch_bg_proto.detach() * (1.0 - mix_m), dim=0)

        proto_sim_local = prototype_similarity_map(feat, batch_fg_proto, batch_bg_proto)
        proto_sim_global = prototype_similarity_map(feat, global_proto_fg, global_proto_bg)
        proto_sim = 0.5 * proto_sim_local + 0.5 * proto_sim_global

        # Progressive Confidence Region Expansion
        pseudo_mask, pseudo_valid = build_progressive_targets(
            teacher_prob=teacher_prob,
            weak_mask=weak_mask,
            conf_map=conf,
            proto_sim=proto_sim.detach(),
            epoch=epoch,
            total_epochs=args.epochs,
            start_conf=args.start_conf,
            end_conf=args.end_conf,
            proto_mix=args.proto_mix,
            available=valid,
        )

        weak_sup = weighted_bce_loss(logits, weak_mask, conf)
        weak_dice = sigmoid_dice_loss(logits, weak_mask, conf)

        # CRME: region head learns the progressively expanded confidence region.
        region_weight = pseudo_valid * (1.0 - boundary) + boundary * args.boundary_weight
        region_loss = weighted_bce_loss(region_logits, pseudo_mask, region_weight)

        # CPE: segmentation logits should agree with prototype similarity induced labels.
        proto_target = torch.sigmoid(proto_sim.detach())
        proto_weight = (pseudo_valid + conf).clamp(0.0, 1.0)
        proto_loss = F.binary_cross_entropy_with_logits(logits, proto_target, reduction='none')
        proto_loss = (proto_loss * proto_weight).sum() / (proto_weight.sum() + 1e-6)

        # Teacher-student consistency.
        cons_weight = ((teacher_prob > args.cons_threshold).float() + (teacher_prob < (1.0 - args.cons_threshold)).float()) * valid
        cons_loss = F.mse_loss(prob * cons_weight, teacher_prob * cons_weight)

        # entropy minimization on uncertain regions
        ent = -(prob.clamp(1e-6, 1 - 1e-6) * torch.log(prob.clamp(1e-6, 1 - 1e-6)) +
                (1.0 - prob).clamp(1e-6, 1 - 1e-6) * torch.log((1.0 - prob).clamp(1e-6, 1 - 1e-6)))
        ent_weight = valid * (1.0 - conf)
        ent_loss = (ent * ent_weight).mean()

        loss = (
            args.lambda_sup * weak_sup +
            args.lambda_dice * weak_dice +
            args.lambda_region * region_loss +
            args.lambda_proto * proto_loss +
            args.lambda_cons * cons_loss +
            args.lambda_ent * ent_loss
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        update_ema(model, teacher, args.ema_momentum)

        bs = img.size(0)
        metrics = compute_metrics(prob.detach(), hard_mask, valid)
        total += loss.item() * bs
        stat['sup'] += weak_sup.item() * bs
        stat['dice'] += weak_dice.item() * bs
        stat['region'] += region_loss.item() * bs
        stat['proto'] += proto_loss.item() * bs
        stat['cons'] += cons_loss.item() * bs
        stat['ent'] += ent_loss.item() * bs
        stat['iou'] += metrics['iou'] * bs
        n += bs

    if n == 0:
        return {'loss': 0.0, **stat}, global_proto_fg, global_proto_bg
    out = {'loss': total / n}
    out.update({k: v / n for k, v in stat.items()})
    return out, global_proto_fg, global_proto_bg


# -----------------------------
# Main
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description='Adapted WSSS-PCRE training for HR + noisy mask weak supervision')
    parser.add_argument('--root_dir', type=str, default="./data/train")
    parser.add_argument('--save_dir', type=str, default='./experiment/wsss_pcre_weak')
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--feat_dim', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val_ratio', type=float, default=0.1)
    parser.add_argument('--hr_divisor', type=float, default=1024.0)
    parser.add_argument('--mask_threshold', type=float, default=20.0)
    parser.add_argument('--ignore_value', type=float, default=None,
                        help='Weak-mask no-data code excluded from all supervised losses and validation metrics.')
    parser.add_argument('--boundary_kernel', type=int, default=9)
    parser.add_argument('--boundary_weight', type=float, default=0.35)
    parser.add_argument('--min_confidence', type=float, default=0.15)

    parser.add_argument('--lambda_sup', type=float, default=1.0)
    parser.add_argument('--lambda_dice', type=float, default=0.7)
    parser.add_argument('--lambda_region', type=float, default=0.8)
    parser.add_argument('--lambda_proto', type=float, default=0.3)
    parser.add_argument('--lambda_cons', type=float, default=0.2)
    parser.add_argument('--lambda_ent', type=float, default=0.05)

    parser.add_argument('--ema_momentum', type=float, default=0.99)
    parser.add_argument('--proto_momentum', type=float, default=0.9)
    parser.add_argument('--start_conf', type=float, default=0.92)
    parser.add_argument('--end_conf', type=float, default=0.70)
    parser.add_argument('--cons_threshold', type=float, default=0.7)
    parser.add_argument('--proto_mix', type=float, default=0.35)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--split_json', type=str, default='')
    parser.add_argument('--patience', type=int, default=5)
    return parser.parse_args()


def main():
    maybe_fix_omp_env()
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cfg = WeakDataConfig(
        root_dir=args.root_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        hr_divisor=args.hr_divisor,
        mask_threshold=args.mask_threshold,
        boundary_kernel=args.boundary_kernel,
        boundary_weight=args.boundary_weight,
        min_confidence=args.min_confidence,
        split_json=args.split_json,
        ignore_value=args.ignore_value,
    )
    train_loader, val_loader = build_loaders(cfg)

    model = PCREWeakNet(in_channels=args.in_channels, feat_dim=args.feat_dim).to(device)
    teacher = deepcopy(model).to(device)
    for p in teacher.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    global_proto_fg = F.normalize(torch.randn(args.feat_dim, device=device), dim=0)
    global_proto_bg = F.normalize(torch.randn(args.feat_dim, device=device), dim=0)

    print(f'method=WSSS-PCRE-weak input_channels={args.in_channels} device={device}')
    print(f'train={len(train_loader.dataset)} val={len(val_loader.dataset)} save_dir={args.save_dir}')

    best_iou = -1.0
    epoch_times: List[float] = []

    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_stats, global_proto_fg, global_proto_bg = train_one_epoch(
            epoch=epoch - 1,
            args=args,
            model=model,
            teacher=teacher,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            global_proto_fg=global_proto_fg,
            global_proto_bg=global_proto_bg,
        )
        val_stats = validate(model, val_loader, device)

        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)
        avg_time = sum(epoch_times) / len(epoch_times)
        eta = avg_time * (args.epochs - epoch)

        ckpt = {
            'epoch': epoch,
            'model': model.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'args': vars(args),
            'best_iou': best_iou,
            'global_proto_fg': global_proto_fg.detach().cpu(),
            'global_proto_bg': global_proto_bg.detach().cpu(),
        }
        torch.save(ckpt, os.path.join(args.save_dir, 'last.pth'))

        if val_stats['iou'] > best_iou:
            best_iou = val_stats['iou']
            ckpt['best_iou'] = best_iou
            torch.save(ckpt, os.path.join(args.save_dir, 'best.pth'))
            stale_epochs = 0
        else:
            stale_epochs += 1

        print(
            f"[WSSS-PCRE] {epoch:03d}/{args.epochs:03d} | "
            f"train={train_stats['loss']:.6f} "
            f"(sup={train_stats['sup']:.4f}, dice={train_stats['dice']:.4f}, region={train_stats['region']:.4f}, "
            f"proto={train_stats['proto']:.4f}, cons={train_stats['cons']:.4f}, ent={train_stats['ent']:.4f}, iou={train_stats['iou']:.4f}) | "
            f"val={val_stats['loss']:.6f} "
            f"(iou={val_stats['iou']:.4f}, dice={val_stats['dice']:.4f}, acc={val_stats['acc']:.4f}, recall={val_stats['recall']:.4f}, precision={val_stats['precision']:.4f}) | "
            f"time={epoch_time:.1f}s avg={avg_time:.1f}s eta={format_eta(eta)}"
        )
        if args.patience > 0 and stale_epochs >= args.patience:
            print(f'Early stopping at epoch {epoch}: no validation IoU improvement for {args.patience} epochs.')
            break

    print(f'Best val IoU: {best_iou:.4f}')


if __name__ == '__main__':
    main()
