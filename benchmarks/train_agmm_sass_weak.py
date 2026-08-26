import argparse
import math
import os
import random
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.semseg.deeplabv3plus import DeepLabV3Plus
import models.backbone.resnet as resnet
from models.tools import build_cur_cls_label, clean_mask, get_cls_loss, cal_protypes, GMM, cal_gmm_loss


# -----------------------------
# basic utils
# -----------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


# -----------------------------
# dataset
# -----------------------------
@dataclass
class WeakDataConfig:
    root_dir: str
    hr_dir: str = "hr"
    mask_dir: str = "mask"
    hr_suffix: str = "hr"
    mask_suffix: str = "mask"
    val_ratio: float = 0.1
    seed: int = 42
    hr_divisor: float = 1024.0
    mask_threshold: float = 20.0
    batch_size: int = 8
    num_workers: int = 0
    pin_memory: bool = True
    crop_size: int = 512
    min_scale: float = 0.75
    max_scale: float = 1.25
    hflip_p: float = 0.5
    boundary_kernel: int = 9
    boundary_weight: float = 0.35
    interior_weight: float = 1.0
    min_confidence: float = 0.15


def _strip_known_suffix(name_no_ext: str, suffix: str) -> str:
    token = "_" + suffix
    if name_no_ext.endswith(token):
        return name_no_ext[: -len(token)]
    if name_no_ext.endswith(suffix):
        return name_no_ext[: -len(suffix)]
    return name_no_ext


def _find_samples(cfg: WeakDataConfig) -> List[Dict[str, str]]:
    hr_root = Path(cfg.root_dir) / cfg.hr_dir
    mask_root = Path(cfg.root_dir) / cfg.mask_dir

    if not hr_root.exists():
        raise FileNotFoundError(f"hr folder not found: {hr_root}")
    if not mask_root.exists():
        raise FileNotFoundError(f"mask folder not found: {mask_root}")

    hr_map = {
        _strip_known_suffix(p.stem, cfg.hr_suffix): str(p)
        for p in hr_root.iterdir() if p.is_file()
    }
    mask_map = {
        _strip_known_suffix(p.stem, cfg.mask_suffix): str(p)
        for p in mask_root.iterdir() if p.is_file()
    }

    common_ids = sorted(set(hr_map) & set(mask_map))
    if not common_ids:
        raise RuntimeError(
            f"No matched samples found. Expected xxx_{cfg.hr_suffix}.tif and xxx_{cfg.mask_suffix}.tif"
        )

    return [{"id": sid, "hr_path": hr_map[sid], "mask_path": mask_map[sid]} for sid in common_ids]


def split_samples(cfg: WeakDataConfig) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    samples = _find_samples(cfg)
    rng = random.Random(cfg.seed)
    rng.shuffle(samples)

    n_total = len(samples)
    if n_total <= 1:
        return samples, []
    n_val = max(1, int(round(n_total * cfg.val_ratio)))
    return samples[n_val:], samples[:n_val]


def read_raster(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read()
    return arr.astype(np.float32)


def normalize_image(arr: np.ndarray, divisor: float) -> np.ndarray:
    arr = arr.astype(np.float32) / float(divisor)
    return arr


def process_mask(mask: np.ndarray, threshold: float) -> np.ndarray:
    return (mask > threshold).astype(np.float32)


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


def resize_pair(img: torch.Tensor, mask: torch.Tensor, scale: float) -> Tuple[torch.Tensor, torch.Tensor]:
    _, h, w = img.shape
    new_h = max(32, int(round(h * scale)))
    new_w = max(32, int(round(w * scale)))
    img = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False).squeeze(0)
    mask = F.interpolate(mask.unsqueeze(0), size=(new_h, new_w), mode="nearest").squeeze(0)
    return img, mask


def crop_pair(img: torch.Tensor, mask: torch.Tensor, size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    _, h, w = img.shape
    pad_h = max(0, size - h)
    pad_w = max(0, size - w)
    if pad_h > 0 or pad_w > 0:
        img = F.pad(img, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
        mask = F.pad(mask, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
        _, h, w = img.shape
    top = 0 if h == size else random.randint(0, h - size)
    left = 0 if w == size else random.randint(0, w - size)
    return img[:, top:top + size, left:left + size], mask[:, top:top + size, left:left + size]


class WeakHRAGMMDataset(Dataset):
    def __init__(self, samples: List[Dict[str, str]], cfg: WeakDataConfig, train: bool = True):
        self.samples = samples
        self.cfg = cfg
        self.train = train

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        hr = read_raster(s["hr_path"])
        weak_mask = read_raster(s["mask_path"])

        hr = normalize_image(hr, self.cfg.hr_divisor)
        weak_mask = process_mask(weak_mask, self.cfg.mask_threshold)

        hr_t = torch.from_numpy(hr)
        mask_t = torch.from_numpy(weak_mask)

        if self.train:
            scale = random.uniform(self.cfg.min_scale, self.cfg.max_scale)
            hr_t, mask_t = resize_pair(hr_t, mask_t, scale)
            hr_t, mask_t = crop_pair(hr_t, mask_t, self.cfg.crop_size)
            if random.random() < self.cfg.hflip_p:
                hr_t = torch.flip(hr_t, dims=[2])
                mask_t = torch.flip(mask_t, dims=[2])

        mask_hard = (mask_t > 0.5).long().squeeze(0)
        cls_label = torch.zeros(2, dtype=torch.float32)
        uniq = torch.unique(mask_hard)
        for v in uniq.tolist():
            if v in [0, 1]:
                cls_label[v] = 1.0

        boundary = _make_boundary_band(mask_t.float(), self.cfg.boundary_kernel)
        conf = torch.full_like(mask_t, float(self.cfg.interior_weight))
        conf = conf * (1.0 - boundary) + float(self.cfg.boundary_weight) * boundary
        conf = conf.clamp(min=float(self.cfg.min_confidence), max=1.0)

        return {
            "id": s["id"],
            "img": hr_t.float(),
            "mask": mask_hard,
            "cls_label": cls_label,
            "conf": conf.float(),
            "boundary": boundary.float(),
        }


def build_loaders(cfg: WeakDataConfig):
    train_samples, val_samples = split_samples(cfg)
    train_ds = WeakHRAGMMDataset(train_samples, cfg, train=True)
    val_ds = WeakHRAGMMDataset(val_samples, cfg, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=len(train_ds) >= cfg.batch_size,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    return train_loader, val_loader


# -----------------------------
# model + loss
# -----------------------------


def patch_backbone_pretrain_fallback() -> None:
    """
    AGMM-SASS 原仓库里 DeepLabV3Plus 会强制调用 resnet50(True, ...)，
    即默认要求 pretrained/resnet50.pth 存在。这里改成：
    - 若本地存在预训练权重，则正常加载
    - 若不存在，则自动退化为随机初始化，不报错
    """
    original__resnet = resnet._resnet

    def _safe_resnet(arch, block, layers, pretrained, **kwargs):
        from pathlib import Path
        pretrained_path = Path('pretrained') / f'{arch}.pth'
        if pretrained and not pretrained_path.exists():
            print(f'[Warn] pretrained weight not found: {pretrained_path}. Use random initialization instead.')
            pretrained = False
        return original__resnet(arch, block, layers, pretrained, **kwargs)

    resnet._resnet = _safe_resnet


def patch_backbone_input_channels(model: nn.Module, in_channels: int) -> None:
    if in_channels == 3:
        return
    old_conv = model.backbone.conv1[0]
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )
    with torch.no_grad():
        if in_channels > 3:
            new_conv.weight[:, :3] = old_conv.weight
            mean_w = old_conv.weight.mean(dim=1, keepdim=True)
            for c in range(3, in_channels):
                new_conv.weight[:, c:c+1] = mean_w
        else:
            new_conv.weight[:] = old_conv.weight[:, :in_channels]
    model.backbone.conv1[0] = new_conv


def weighted_ce_loss(logits: torch.Tensor, target: torch.Tensor, conf: torch.Tensor) -> torch.Tensor:
    per_pixel = F.cross_entropy(logits, target.long(), reduction="none")
    weight = conf.squeeze(1)
    return (per_pixel * weight).sum() / weight.sum().clamp_min(1.0)


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.softmax(logits, dim=1)[:, 1]
    target = target.float()
    inter = (prob * target).sum(dim=(1, 2))
    denom = prob.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def compute_metrics(logits: torch.Tensor, target: torch.Tensor):
    pred = torch.argmax(logits, dim=1)
    target = target.long()

    tp = ((pred == 1) & (target == 1)).sum().item()
    fp = ((pred == 1) & (target == 0)).sum().item()
    fn = ((pred == 0) & (target == 1)).sum().item()
    tn = ((pred == 0) & (target == 0)).sum().item()

    iou = tp / max(tp + fp + fn, 1)
    dice = 2 * tp / max(2 * tp + fp + fn, 1)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "iou": iou,
        "dice": dice,
        "acc": acc,
        "precision": precision,
        "recall": recall,
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    losses = []
    stats = {"iou": 0.0, "dice": 0.0, "acc": 0.0, "precision": 0.0, "recall": 0.0}
    n = 0

    for batch in loader:
        img = batch["img"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        conf = batch["conf"].to(device, non_blocking=True)

        out = model(img)
        logits = out if isinstance(out, torch.Tensor) else out[-1]
        loss = weighted_ce_loss(logits, mask, conf) + dice_loss_from_logits(logits, mask)
        losses.append(loss.item())

        cur = compute_metrics(logits, mask)
        for k in stats:
            stats[k] += cur[k]
        n += 1

    if n == 0:
        return {"loss": 0.0, **stats}
    return {"loss": float(np.mean(losses)), **{k: v / n for k, v in stats.items()}}


# -----------------------------
# train
# -----------------------------
def main():
    parser = argparse.ArgumentParser("AGMM-SASS weak HR training")
    parser.add_argument("--root_dir", type=str, default="./data/train")
    parser.add_argument("--save_dir", type=str, default="./experiment/agmm_sass_weak")
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--backbone", type=str, default="resnet101", choices=["resnet50", "resnet101"])
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_multi", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--hr_divisor", type=float, default=1024.0)
    parser.add_argument("--mask_threshold", type=float, default=20.0)
    parser.add_argument("--boundary_kernel", type=int, default=9)
    parser.add_argument("--boundary_weight", type=float, default=0.35)
    parser.add_argument("--interior_weight", type=float, default=1.0)
    parser.add_argument("--min_confidence", type=float, default=0.15)
    parser.add_argument("--min_scale", type=float, default=0.75)
    parser.add_argument("--max_scale", type=float, default=1.25)
    parser.add_argument("--lambda_gmm", type=float, default=1.0)
    parser.add_argument("--lambda_proto", type=float, default=1.0)
    parser.add_argument("--lambda_cls", type=float, default=1.0)
    parser.add_argument("--lambda_dice", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_cfg = WeakDataConfig(
        root_dir=args.root_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        hr_divisor=args.hr_divisor,
        mask_threshold=args.mask_threshold,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        crop_size=args.crop_size,
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        boundary_kernel=args.boundary_kernel,
        boundary_weight=args.boundary_weight,
        interior_weight=args.interior_weight,
        min_confidence=args.min_confidence,
    )
    train_loader, val_loader = build_loaders(data_cfg)

    cfg = {
        "nclass": 2,
        "aux": False,
        "backbone": args.backbone,
        "multi_grid": False,
        "replace_stride_with_dilation": [False, True, True],
        "dilations": [6, 12, 18],
    }
    patch_backbone_pretrain_fallback()
    model = DeepLabV3Plus(cfg, aux=False)
    patch_backbone_input_channels(model, args.in_channels)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": args.lr},
            {
                "params": [p for n, p in model.named_parameters() if "backbone" not in n],
                "lr": args.lr * args.lr_multi,
            },
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_iters = max(1, len(train_loader) * args.epochs)
    iters = 0
    best_iou = -1.0
    stale_epochs = 0
    epoch_times: List[float] = []

    print("=" * 72)
    print(f"method         = AGMM-SASS-weak-HR")
    print(f"root_dir       = {args.root_dir}")
    print(f"save_dir       = {args.save_dir}")
    print(f"device         = {device}")
    print(f"in_channels    = {args.in_channels}")
    print(f"train_samples  = {len(train_loader.dataset)}")
    print(f"val_samples    = {len(val_loader.dataset)}")
    print(f"backbone       = {args.backbone}")
    print("=" * 72)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()

        run = {
            "loss": 0.0,
            "seg": 0.0,
            "gmm": 0.0,
            "proto": 0.0,
            "cls": 0.0,
            "dice": 0.0,
            "iou": 0.0,
        }
        steps = 0

        for batch in train_loader:
            img = batch["img"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            cls_label = batch["cls_label"].to(device, non_blocking=True)
            conf = batch["conf"].to(device, non_blocking=True)

            feat, logits = model(img)

            seg_loss = weighted_ce_loss(logits, mask, conf)
            dice = dice_loss_from_logits(logits, mask)

            cur_cls_label = build_cur_cls_label(mask, 2)
            pred_cl = clean_mask(logits, cls_label, True)
            vecs, proto_loss = cal_protypes(feat, mask, 2)
            res = GMM(feat, vecs, pred_cl, mask, cur_cls_label)
            cls_loss = get_cls_loss(logits, cls_label, mask)
            gmm_align = cal_gmm_loss(torch.softmax(logits, 1), res, cur_cls_label, mask)
            gmm_loss = gmm_align + args.lambda_proto * proto_loss + args.lambda_cls * cls_loss

            loss = seg_loss + args.lambda_dice * dice + args.lambda_gmm * gmm_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            iters += 1
            lr = args.lr * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * args.lr_multi

            cur_metrics = compute_metrics(logits.detach(), mask)
            run["loss"] += loss.item()
            run["seg"] += seg_loss.item()
            run["gmm"] += gmm_align.item()
            run["proto"] += proto_loss.item()
            run["cls"] += cls_loss.item()
            run["dice"] += dice.item()
            run["iou"] += cur_metrics["iou"]
            steps += 1

        train_stats = {k: v / max(steps, 1) for k, v in run.items()}
        val_stats = evaluate(model, val_loader, device) if len(val_loader.dataset) > 0 else {
            "loss": 0.0, "iou": 0.0, "dice": 0.0, "acc": 0.0, "precision": 0.0, "recall": 0.0
        }

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_epoch = sum(epoch_times) / len(epoch_times)
        eta = avg_epoch * (args.epochs - epoch)

        print(
            f"[AGMM-SASS] {epoch:03d}/{args.epochs:03d} | "
            f"train={train_stats['loss']:.4f} (seg={train_stats['seg']:.4f}, gmm={train_stats['gmm']:.4f}, "
            f"proto={train_stats['proto']:.4f}, cls={train_stats['cls']:.4f}, dice={train_stats['dice']:.4f}, "
            f"iou={train_stats['iou']:.4f}) | "
            f"val={val_stats['loss']:.4f} (iou={val_stats['iou']:.4f}, dice={val_stats['dice']:.4f}, "
            f"acc={val_stats['acc']:.4f}, rec={val_stats['recall']:.4f}, prec={val_stats['precision']:.4f}) | "
            f"time={epoch_time:.1f}s avg={avg_epoch:.1f}s eta={format_seconds(eta)}"
        )

        latest_path = os.path.join(args.save_dir, "agmm_sass_weak_latest.pth")
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "val_stats": val_stats,
        }, latest_path)

        if val_stats["iou"] > best_iou:
            best_iou = val_stats["iou"]
            stale_epochs = 0
            best_path = os.path.join(args.save_dir, "agmm_sass_weak_best.pth")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
                "val_stats": val_stats,
            }, best_path)
        else:
            stale_epochs += 1
        if args.patience > 0 and stale_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}: no validation IoU improvement for {args.patience} epochs.")
            break

    print(f"Best val IoU = {best_iou:.4f}")


if __name__ == "__main__":
    main()
