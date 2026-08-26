import argparse
import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning
import warnings

warnings.filterwarnings('ignore', category=NotGeoreferencedWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, resnet101, ResNet50_Weights, ResNet101_Weights


# -----------------------------
# Utilities
# -----------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_valid_omp() -> None:
    val = os.environ.get('OMP_NUM_THREADS', '').strip()
    if val:
        try:
            if int(val) <= 0:
                raise ValueError
        except Exception:
            os.environ['OMP_NUM_THREADS'] = '1'
    else:
        os.environ['OMP_NUM_THREADS'] = '1'


def format_seconds(sec: float) -> str:
    sec = max(0, int(sec))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def sigmoid_iou_metrics(logits: torch.Tensor, target: torch.Tensor, thr: float = 0.5) -> Dict[str, float]:
    prob = torch.sigmoid(logits)
    pred = (prob > thr).float()
    tgt = (target > 0.5).float()

    tp = (pred * tgt).sum().item()
    tn = ((1 - pred) * (1 - tgt)).sum().item()
    fp = (pred * (1 - tgt)).sum().item()
    fn = ((1 - pred) * tgt).sum().item()

    eps = 1e-7
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    recall = tp / (tp + fn + eps)
    precision = tp / (tp + fp + eps)
    return {
        'iou': iou,
        'dice': dice,
        'acc': acc,
        'recall': recall,
        'precision': precision,
    }


# -----------------------------
# Dataset (adapted from user's mydataset_weak.py)
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
        raise RuntimeError('No matched samples found. Expected xxx_hr.tif and xxx_mask.tif')
    return [{'id': sid, 'hr_path': hr_map[sid], 'mask_path': mask_map[sid]} for sid in common_ids]


def split_samples(cfg: WeakDataConfig) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    samples = _find_samples(cfg)
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


def mask_to_edge(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 3:
        mask = mask.unsqueeze(0)
    max_pool = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    min_pool = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
    edge = (max_pool - min_pool).clamp(0.0, 1.0)
    return edge.squeeze(0)


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


def image_gradient_weight(img: torch.Tensor) -> torch.Tensor:
    gray = img.mean(dim=0, keepdim=True)
    gx = gray[:, :, 1:] - gray[:, :, :-1]
    gy = gray[:, 1:, :] - gray[:, :-1, :]
    gx = F.pad(gx.abs(), (0, 1, 0, 0))
    gy = F.pad(gy.abs(), (0, 0, 0, 1))
    grad = (gx + gy)
    grad = grad / (grad.amax(dim=(-2, -1), keepdim=True) + 1e-6)
    return grad.clamp(0.0, 1.0)


class MPFWeakDataset(Dataset):
    def __init__(self, samples: List[Dict[str, str]], cfg: WeakDataConfig):
        self.samples = samples
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        hr = normalize_image(read_raster(s['hr_path']), self.cfg.hr_divisor)
        weak_mask = process_mask(read_raster(s['mask_path']), self.cfg.mask_threshold, self.cfg.mask_binarize)

        hr_t = torch.from_numpy(hr)
        mask_t = torch.from_numpy(weak_mask)
        boundary = _make_boundary_band((mask_t > 0.5).float(), self.cfg.boundary_kernel)
        conf = torch.full_like(mask_t, float(self.cfg.interior_weight))
        conf = conf * (1.0 - boundary) + float(self.cfg.boundary_weight) * boundary
        conf = conf.clamp(min=float(self.cfg.min_confidence), max=1.0)
        edge = mask_to_edge((mask_t > 0.5).float())
        grad = image_gradient_weight(hr_t)
        return {
            'id': s['id'],
            'img': hr_t,
            'mask': mask_t,
            'conf': conf,
            'boundary': boundary,
            'edge': edge,
            'grad': grad,
        }


def build_loaders(cfg: WeakDataConfig) -> Tuple[DataLoader, DataLoader]:
    train_samples, val_samples = split_samples(cfg)
    train_ds = MPFWeakDataset(train_samples, cfg)
    val_ds = MPFWeakDataset(val_samples, cfg)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    return train_loader, val_loader


# -----------------------------
# MPF-inspired network
# -----------------------------

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FPNDecoder(nn.Module):
    def __init__(self, chs=(256, 512, 1024, 2048), fpn_ch=256):
        super().__init__()
        self.lat2 = nn.Conv2d(chs[0], fpn_ch, 1)
        self.lat3 = nn.Conv2d(chs[1], fpn_ch, 1)
        self.lat4 = nn.Conv2d(chs[2], fpn_ch, 1)
        self.lat5 = nn.Conv2d(chs[3], fpn_ch, 1)
        self.smooth2 = ConvBNReLU(fpn_ch, fpn_ch)
        self.smooth3 = ConvBNReLU(fpn_ch, fpn_ch)
        self.smooth4 = ConvBNReLU(fpn_ch, fpn_ch)
        self.smooth5 = ConvBNReLU(fpn_ch, fpn_ch)

    def forward(self, c2, c3, c4, c5):
        p5 = self.smooth5(self.lat5(c5))
        p4 = self.smooth4(self.lat4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode='bilinear', align_corners=False))
        p3 = self.smooth3(self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode='bilinear', align_corners=False))
        p2 = self.smooth2(self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode='bilinear', align_corners=False))
        return p2, p3, p4, p5


class MPFWeakNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, use_imagenet: bool = True,
                 backbone_name: str = "resnet101"):
        super().__init__()
        if backbone_name == "resnet101":
            weights = ResNet101_Weights.DEFAULT if use_imagenet else None
            backbone = resnet101(weights=weights)
        elif backbone_name == "resnet50":
            weights = ResNet50_Weights.DEFAULT if use_imagenet else None
            backbone = resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        if in_channels != 3:
            old = backbone.conv1
            backbone.conv1 = nn.Conv2d(in_channels, old.out_channels, kernel_size=old.kernel_size,
                                       stride=old.stride, padding=old.padding, bias=False)
            with torch.no_grad():
                if in_channels > 3:
                    backbone.conv1.weight[:, :3] = old.weight
                    mean_w = old.weight.mean(dim=1, keepdim=True)
                    for i in range(3, in_channels):
                        backbone.conv1.weight[:, i:i+1] = mean_w
                else:
                    backbone.conv1.weight[:] = old.weight[:, :in_channels]
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.decoder = FPNDecoder()
        self.head2 = nn.Conv2d(256, out_channels, 1)
        self.head3 = nn.Conv2d(256, out_channels, 1)
        self.head4 = nn.Conv2d(256, out_channels, 1)
        self.head5 = nn.Conv2d(256, out_channels, 1)
        self.fuse_head = nn.Sequential(
            ConvBNReLU(256 * 4, 256),
            nn.Conv2d(256, out_channels, 1)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h, w = x.shape[-2:]
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p2, p3, p4, p5 = self.decoder(c2, c3, c4, c5)
        l2 = F.interpolate(self.head2(p2), size=(h, w), mode='bilinear', align_corners=False)
        l3 = F.interpolate(self.head3(p3), size=(h, w), mode='bilinear', align_corners=False)
        l4 = F.interpolate(self.head4(p4), size=(h, w), mode='bilinear', align_corners=False)
        l5 = F.interpolate(self.head5(p5), size=(h, w), mode='bilinear', align_corners=False)
        fused_feat = torch.cat([
            F.interpolate(p2, size=(h, w), mode='bilinear', align_corners=False),
            F.interpolate(p3, size=(h, w), mode='bilinear', align_corners=False),
            F.interpolate(p4, size=(h, w), mode='bilinear', align_corners=False),
            F.interpolate(p5, size=(h, w), mode='bilinear', align_corners=False),
        ], dim=1)
        fused = self.fuse_head(fused_feat)
        return {'l2': l2, 'l3': l3, 'l4': l4, 'l5': l5, 'fused': fused}


# -----------------------------
# MPF-inspired loss
# -----------------------------


def weighted_bce_with_logits(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    return (loss * weight).sum() / (weight.sum() + 1e-6)


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    prob = prob * weight
    target = target * weight
    inter = (prob * target).sum(dim=(-2, -1))
    den = prob.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    dice = (2 * inter + 1e-6) / (den + 1e-6)
    return 1 - dice.mean()


def confidence_from_logits(logits: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    return (prob - 0.5).abs() * 2.0


def fuse_multilevel_pseudo(outputs: Dict[str, torch.Tensor], grad: torch.Tensor, boundary: torch.Tensor) -> torch.Tensor:
    levels = [outputs['l2'], outputs['l3'], outputs['l4'], outputs['l5'], outputs['fused']]
    confs = [confidence_from_logits(x) for x in levels]
    conf_stack = torch.stack(confs, dim=0)
    prob_stack = torch.stack([torch.sigmoid(x) for x in levels], dim=0)

    grad_boost = 1.0 + 0.35 * grad.unsqueeze(0)
    boundary_penalty = 1.0 - 0.25 * boundary.unsqueeze(0)
    weights = (conf_stack * grad_boost * boundary_penalty).clamp(min=1e-4)
    pseudo = (prob_stack * weights).sum(dim=0) / weights.sum(dim=0)
    return pseudo.detach()


def consistency_loss(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    ref = torch.sigmoid(outputs['fused']).detach()
    losses = []
    for key in ('l2', 'l3', 'l4', 'l5'):
        losses.append(F.l1_loss(torch.sigmoid(outputs[key]), ref))
    return torch.stack(losses).mean()


def boundary_entropy_loss(logits: torch.Tensor, boundary: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    ent = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
    return (ent * boundary).sum() / (boundary.sum() + 1e-6)


# -----------------------------
# Train / Validate
# -----------------------------

@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    metrics_sum = {'iou': 0.0, 'dice': 0.0, 'acc': 0.0, 'recall': 0.0, 'precision': 0.0}
    for batch in loader:
        img = batch['img'].to(device)
        mask = batch['mask'].to(device)
        conf = batch['conf'].to(device)
        outputs = model(img)
        logits = outputs['fused']
        loss = weighted_bce_with_logits(logits, mask, conf) + soft_dice_loss(logits, mask, conf)
        bs = img.size(0)
        total_loss += loss.item() * bs
        total += bs
        mb = sigmoid_iou_metrics(logits, mask)
        for k in metrics_sum:
            metrics_sum[k] += mb[k] * bs
    out = {'loss': total_loss / max(total, 1)}
    for k, v in metrics_sum.items():
        out[k] = v / max(total, 1)
    return out


def train_one_epoch(model: nn.Module, ema_model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, args) -> Dict[str, float]:
    model.train()
    meters = {
        'loss': 0.0,
        'sup': 0.0,
        'dice': 0.0,
        'mpf': 0.0,
        'cons': 0.0,
        'ent': 0.0,
        'pseudo': 0.0,
        'iou': 0.0,
    }
    total = 0
    for batch in loader:
        img = batch['img'].to(device)
        mask = batch['mask'].to(device)
        conf = batch['conf'].to(device)
        boundary = batch['boundary'].to(device)
        grad = batch['grad'].to(device)

        outputs = model(img)

        sup_losses = []
        dice_losses = []
        for key in ('l2', 'l3', 'l4', 'l5', 'fused'):
            sup_losses.append(weighted_bce_with_logits(outputs[key], mask, conf))
            dice_losses.append(soft_dice_loss(outputs[key], mask, conf))
        sup_loss = torch.stack(sup_losses).mean()
        dice_loss = torch.stack(dice_losses).mean()

        pseudo = fuse_multilevel_pseudo(outputs, grad, boundary)
        mpf_loss = F.l1_loss(torch.sigmoid(outputs['fused']), pseudo)
        cons_loss = consistency_loss(outputs)
        ent_loss = boundary_entropy_loss(outputs['fused'], boundary)

        pseudo_loss = torch.tensor(0.0, device=device)
        if epoch >= args.warmup_epochs:
            with torch.no_grad():
                ema_out = ema_model(img)
                teacher_pseudo = fuse_multilevel_pseudo(ema_out, grad, boundary)
                teacher_conf = ((teacher_pseudo - 0.5).abs() * 2.0).clamp(min=args.min_pseudo_conf)
            pseudo_loss = weighted_bce_with_logits(outputs['fused'], teacher_pseudo, teacher_conf)

        loss = (
            args.lambda_sup * sup_loss +
            args.lambda_dice * dice_loss +
            args.lambda_mpf * mpf_loss +
            args.lambda_cons * cons_loss +
            args.lambda_ent * ent_loss +
            args.lambda_pseudo * pseudo_loss
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        with torch.no_grad():
            for p_ema, p in zip(ema_model.parameters(), model.parameters()):
                p_ema.data.mul_(args.ema_decay).add_(p.data, alpha=1.0 - args.ema_decay)

        bs = img.size(0)
        total += bs
        meters['loss'] += loss.item() * bs
        meters['sup'] += sup_loss.item() * bs
        meters['dice'] += dice_loss.item() * bs
        meters['mpf'] += mpf_loss.item() * bs
        meters['cons'] += cons_loss.item() * bs
        meters['ent'] += ent_loss.item() * bs
        meters['pseudo'] += pseudo_loss.item() * bs
        meters['iou'] += sigmoid_iou_metrics(outputs['fused'], mask)['iou'] * bs

    for k in meters:
        meters[k] /= max(total, 1)
    return meters


# -----------------------------
# Main
# -----------------------------


def parse_args():
    p = argparse.ArgumentParser(description='MPF-style weak segmentation for HR imagery + noisy mask')
    p.add_argument('--root_dir', type=str, default="./data/train")
    p.add_argument('--save_dir', type=str, default='./experiment/mpf_weak')
    p.add_argument('--in_channels', type=int, default=3)
    p.add_argument('--backbone', choices=['resnet50', 'resnet101'], default='resnet101')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--warmup_epochs', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--val_ratio', type=float, default=0.1)
    p.add_argument('--hr_divisor', type=float, default=1024.0)
    p.add_argument('--mask_threshold', type=float, default=20.0)
    p.add_argument('--boundary_kernel', type=int, default=9)
    p.add_argument('--boundary_weight', type=float, default=0.35)
    p.add_argument('--lambda_sup', type=float, default=1.0)
    p.add_argument('--lambda_dice', type=float, default=0.5)
    p.add_argument('--lambda_mpf', type=float, default=0.4)
    p.add_argument('--lambda_cons', type=float, default=0.15)
    p.add_argument('--lambda_ent', type=float, default=0.05)
    p.add_argument('--lambda_pseudo', type=float, default=0.35)
    p.add_argument('--min_pseudo_conf', type=float, default=0.2)
    p.add_argument('--ema_decay', type=float, default=0.99)
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--no_imagenet_pretrain', action='store_true')
    return p.parse_args()


def main():
    ensure_valid_omp()
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

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
    )

    train_loader, val_loader = build_loaders(cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MPFWeakNet(in_channels=args.in_channels, use_imagenet=not args.no_imagenet_pretrain,
                       backbone_name=args.backbone).to(device)
    ema_model = MPFWeakNet(in_channels=args.in_channels, use_imagenet=not args.no_imagenet_pretrain,
                           backbone_name=args.backbone).to(device)
    ema_model.load_state_dict(model.state_dict())
    for p in ema_model.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_iou = -1.0
    stale_epochs = 0
    epoch_times = []

    print(f'method=MPF-weak-hr input_channels={args.in_channels} device={device}')
    print(f'train={len(train_loader.dataset)} val={len(val_loader.dataset)} save_dir={args.save_dir}')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_stats = train_one_epoch(model, ema_model, train_loader, optimizer, device, epoch - 1, args)
        val_stats = validate(model, val_loader, device)
        scheduler.step()

        if val_stats['iou'] > best_iou:
            best_iou = val_stats['iou']
            stale_epochs = 0
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'ema_model': ema_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': vars(args),
                'best_iou': best_iou,
            }, os.path.join(args.save_dir, 'best.pth'))
        else:
            stale_epochs += 1

        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'ema_model': ema_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'args': vars(args),
            'best_iou': best_iou,
        }, os.path.join(args.save_dir, 'last.pth'))

        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)
        avg_epoch_time = sum(epoch_times) / len(epoch_times)
        eta = avg_epoch_time * (args.epochs - epoch)

        print(
            f"[MPF-weak] {epoch:03d}/{args.epochs:03d} | "
            f"train={train_stats['loss']:.4f} "
            f"(sup={train_stats['sup']:.4f}, dice={train_stats['dice']:.4f}, mpf={train_stats['mpf']:.4f}, "
            f"cons={train_stats['cons']:.4f}, ent={train_stats['ent']:.4f}, pseudo={train_stats['pseudo']:.4f}, iou={train_stats['iou']:.4f}) | "
            f"val={val_stats['loss']:.4f} "
            f"(iou={val_stats['iou']:.4f}, dice={val_stats['dice']:.4f}, acc={val_stats['acc']:.4f}, "
            f"recall={val_stats['recall']:.4f}, precision={val_stats['precision']:.4f}) | "
            f"time={epoch_time:.1f}s avg={avg_epoch_time:.1f}s eta={format_seconds(eta)}"
        )
        if args.patience > 0 and stale_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}: no validation IoU improvement for {args.patience} epochs.")
            break

    print(f'Best val IoU: {best_iou:.4f}')
    print(f'Best checkpoint: {os.path.join(args.save_dir, "best.pth")}')


if __name__ == '__main__':
    main()
