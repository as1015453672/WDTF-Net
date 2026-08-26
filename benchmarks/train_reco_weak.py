from __future__ import annotations

import os
import math
import time
import random
import warnings
import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler
import torchvision.models as models

from models.deeplabv2 import DeepLabv2

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


# -----------------------------------------------------------------------------
# Data
# -----------------------------------------------------------------------------
@dataclass
class ReCoWeakConfig:
    root_dir: str
    hr_dir: str = "hr"
    mask_dir: str = "mask"
    hr_suffix: str = "hr"
    mask_suffix: str = "mask"
    val_ratio: float = 0.1
    seed: int = 42
    batch_size: int = 8
    num_workers: int = 0
    pin_memory: bool = True
    hr_divisor: float = 1024.0
    mask_threshold: float = 20.0
    boundary_kernel: int = 9
    boundary_weight: float = 0.35
    min_confidence: float = 0.15
    crop_size: int = 0
    hflip_prob: float = 0.5
    split_json: str = ""
    ignore_value: float | None = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def _strip_known_suffix(name_no_ext: str, suffix: str) -> str:
    token = "_" + suffix
    if name_no_ext.endswith(token):
        return name_no_ext[: -len(token)]
    if name_no_ext.endswith(suffix):
        return name_no_ext[: -len(suffix)]
    return name_no_ext



def _find_samples(cfg: ReCoWeakConfig) -> List[Dict[str, str]]:
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
            "No matched samples found. Expected names like xxx_hr.tif / xxx_mask.tif"
        )
    return [{"id": sid, "hr_path": hr_map[sid], "mask_path": mask_map[sid]} for sid in common_ids]



def split_samples(cfg: ReCoWeakConfig) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    samples = _find_samples(cfg)
    if cfg.split_json:
        split = json.loads(Path(cfg.split_json).read_text(encoding="utf-8"))
        by_id = {sample["id"]: sample for sample in samples}
        train_ids = split.get("train", [])
        val_ids = split.get("val", [])
        missing = [sid for sid in train_ids + val_ids if sid not in by_id]
        if missing:
            raise RuntimeError(f"split_json contains ids missing from dataset: {missing[:5]}")
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


class ReCoWeakHRDataset(Dataset):
    def __init__(self, samples: List[Dict[str, str]], cfg: ReCoWeakConfig, is_train: bool = True):
        self.samples = samples
        self.cfg = cfg
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.samples)

    def _random_crop(self, hr_t: torch.Tensor, mask_t: torch.Tensor, valid_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        crop = int(self.cfg.crop_size)
        if crop <= 0:
            return hr_t, mask_t, valid_t
        _, h, w = hr_t.shape
        if h < crop or w < crop:
            return hr_t, mask_t, valid_t
        top = random.randint(0, h - crop)
        left = random.randint(0, w - crop)
        return (hr_t[:, top:top+crop, left:left+crop], mask_t[:, top:top+crop, left:left+crop],
                valid_t[:, top:top+crop, left:left+crop])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        hr = normalize_image(read_raster(s["hr_path"]), self.cfg.hr_divisor)
        raw_mask = read_raster(s["mask_path"])
        valid_mask = np.ones_like(raw_mask, dtype=np.float32) if self.cfg.ignore_value is None else \
            (~np.isclose(raw_mask, self.cfg.ignore_value)).astype(np.float32)
        weak_mask = process_mask(raw_mask, self.cfg.mask_threshold) * valid_mask

        hr_t = torch.from_numpy(hr)
        mask_t = torch.from_numpy(weak_mask)
        valid_t = torch.from_numpy(valid_mask)

        if self.is_train:
            hr_t, mask_t, valid_t = self._random_crop(hr_t, mask_t, valid_t)
            if random.random() < self.cfg.hflip_prob:
                hr_t = torch.flip(hr_t, dims=[2])
                mask_t = torch.flip(mask_t, dims=[2])
                valid_t = torch.flip(valid_t, dims=[2])

        boundary = _make_boundary_band((mask_t > 0.5).float(), self.cfg.boundary_kernel) * valid_t
        conf = torch.full_like(mask_t, 1.0)
        conf = (conf * (1.0 - boundary) + float(self.cfg.boundary_weight) * boundary) * valid_t
        conf = torch.where(valid_t > 0, conf.clamp(min=float(self.cfg.min_confidence), max=1.0), torch.zeros_like(conf))
        target = (mask_t > 0.5).long().squeeze(0)

        return {
            "id": s["id"],
            "img": hr_t,
            "target": target,
            "conf": conf.squeeze(0),
            "boundary": boundary.squeeze(0),
            "valid": valid_t.squeeze(0),
        }


# -----------------------------------------------------------------------------
# Model utils
# -----------------------------------------------------------------------------
def build_model(in_channels: int = 3, output_dim: int = 128, pretrained_backbone: bool = False,
                backbone_name: str = "resnet101") -> nn.Module:
    weights = None
    if pretrained_backbone:
        try:
            weights = (models.ResNet101_Weights.IMAGENET1K_V1 if backbone_name == "resnet101"
                       else models.ResNet50_Weights.IMAGENET1K_V1)
        except Exception:
            weights = None
    backbone = models.resnet101(weights=weights) if backbone_name == "resnet101" else models.resnet50(weights=weights)
    if in_channels != 3:
        old_conv = backbone.conv1
        new_conv = nn.Conv2d(
            in_channels, old_conv.out_channels,
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
                new_conv.weight.copy_(old_conv.weight[:, :in_channels])
        backbone.conv1 = new_conv
    model = DeepLabv2(backbone, num_classes=2, output_dim=output_dim)
    return model


class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
        self.power = power
        self.max_iters = max_iters
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [max(base_lr * (1 - self.last_epoch / self.max_iters) ** self.power, self.min_lr)
                for base_lr in self.base_lrs]


# -----------------------------------------------------------------------------
# ReCo loss (adapted from official repo)
# -----------------------------------------------------------------------------
def label_onehot(inputs: torch.Tensor, num_segments: int) -> torch.Tensor:
    batch_size, im_h, im_w = inputs.shape
    inputs = torch.relu(inputs)
    outputs = torch.zeros([batch_size, num_segments, im_h, im_w], device=inputs.device)
    return outputs.scatter_(1, inputs.unsqueeze(1), 1.0)



def negative_index_sampler(samp_num: torch.Tensor, seg_num_list: List[int]) -> List[int]:
    negative_index = []
    for i in range(samp_num.shape[0]):
        for j in range(samp_num.shape[1]):
            n = int(samp_num[i, j])
            if n <= 0:
                continue
            low = sum(seg_num_list[:j])
            high = sum(seg_num_list[:j + 1])
            if high <= low:
                continue
            negative_index += np.random.randint(low=low, high=high, size=n).tolist()
    return negative_index



def compute_reco_loss(
    rep: torch.Tensor,
    label: torch.Tensor,
    mask: torch.Tensor,
    prob: torch.Tensor,
    strong_threshold: float = 0.97,
    temp: float = 0.5,
    num_queries: int = 128,
    num_negatives: int = 128,
) -> torch.Tensor:
    batch_size, num_feat, im_h, im_w = rep.shape
    num_segments = label.shape[1]
    device = rep.device

    valid_pixel = label * mask
    rep = rep.permute(0, 2, 3, 1)

    seg_feat_all_list = []
    seg_feat_hard_list = []
    seg_num_list = []
    seg_proto_list = []

    for i in range(num_segments):
        valid_pixel_seg = valid_pixel[:, i]
        if valid_pixel_seg.sum() == 0:
            continue
        prob_seg = prob[:, i, :, :]
        rep_mask_hard = (prob_seg < strong_threshold) * valid_pixel_seg.bool()
        all_feat = rep[valid_pixel_seg.bool()]
        if all_feat.numel() == 0:
            continue
        seg_proto_list.append(torch.mean(all_feat, dim=0, keepdim=True))
        seg_feat_all_list.append(all_feat)
        seg_feat_hard_list.append(rep[rep_mask_hard])
        seg_num_list.append(int(valid_pixel_seg.sum().item()))

    if len(seg_num_list) <= 1:
        return torch.tensor(0.0, device=device)

    reco_loss = torch.tensor(0.0, device=device)
    seg_proto = torch.cat(seg_proto_list, dim=0)
    valid_seg = len(seg_num_list)
    seg_len = torch.arange(valid_seg, device=device)

    for i in range(valid_seg):
        if len(seg_feat_hard_list[i]) == 0:
            continue
        seg_hard_idx = torch.randint(len(seg_feat_hard_list[i]), size=(num_queries,), device=device)
        anchor_feat = seg_feat_hard_list[i][seg_hard_idx]

        with torch.no_grad():
            seg_mask = torch.cat((seg_len[i:], seg_len[:i]))
            proto_sim = torch.cosine_similarity(seg_proto[seg_mask[0]].unsqueeze(0), seg_proto[seg_mask[1:]], dim=1)
            proto_prob = torch.softmax(proto_sim / temp, dim=0)
            negative_dist = torch.distributions.categorical.Categorical(probs=proto_prob)
            samp_class = negative_dist.sample(sample_shape=[num_queries, num_negatives])
            samp_num = torch.stack([(samp_class == c).sum(1) for c in range(len(proto_prob))], dim=1)

            negative_num_list = seg_num_list[i + 1:] + seg_num_list[:i]
            negative_index = negative_index_sampler(samp_num.cpu(), negative_num_list)
            negative_feat_all = torch.cat(seg_feat_all_list[i + 1:] + seg_feat_all_list[:i], dim=0)
            if len(negative_index) == 0 or negative_feat_all.numel() == 0:
                continue
            negative_feat = negative_feat_all[negative_index].reshape(num_queries, num_negatives, num_feat)

            positive_feat = seg_proto[i].unsqueeze(0).unsqueeze(0).repeat(num_queries, 1, 1)
            all_feat = torch.cat((positive_feat, negative_feat), dim=1)

        seg_logits = torch.cosine_similarity(anchor_feat.unsqueeze(1), all_feat, dim=2)
        targets = torch.zeros(num_queries, dtype=torch.long, device=device)
        reco_loss = reco_loss + F.cross_entropy(seg_logits / temp, targets)

    return reco_loss / valid_seg


# -----------------------------------------------------------------------------
# Metrics and helpers
# -----------------------------------------------------------------------------
def weighted_ce_loss(logits: torch.Tensor, target: torch.Tensor, conf: torch.Tensor) -> torch.Tensor:
    loss = F.cross_entropy(logits, target, reduction='none')
    return (loss * conf).sum() / conf.sum().clamp_min(1.0)



def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, conf: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.softmax(logits, dim=1)[:, 1]
    target_f = target.float()
    conf = conf.float()
    intersection = (prob * target_f * conf).sum(dim=(1, 2))
    denom = (prob * conf).sum(dim=(1, 2)) + (target_f * conf).sum(dim=(1, 2))
    dice = (2 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()



def compute_binary_metrics(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor | None = None) -> Dict[str, float]:
    pred = torch.argmax(logits, dim=1)
    pred_b = pred.bool()
    tar_b = target.bool()
    keep = torch.ones_like(pred_b, dtype=torch.bool) if valid is None else valid.bool()
    tp = (pred_b & tar_b & keep).sum().item()
    fp = (pred_b & ~tar_b & keep).sum().item()
    fn = (~pred_b & tar_b & keep).sum().item()
    tn = (~pred_b & ~tar_b & keep).sum().item()
    iou = tp / max(tp + fp + fn, 1)
    dice = 2 * tp / max(2 * tp + fp + fn, 1)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    return {
        "iou": iou,
        "dice": dice,
        "acc": acc,
        "recall": recall,
        "precision": precision,
    }



def format_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


# -----------------------------------------------------------------------------
# Train / eval
# -----------------------------------------------------------------------------
def run_epoch(model, loader, optimizer, scheduler, device, args, train=True):
    model.train(train)
    loss_meter = {"total": 0.0, "ce": 0.0, "dice": 0.0, "reco": 0.0}
    metric_meter = {"iou": 0.0, "dice": 0.0, "acc": 0.0, "recall": 0.0, "precision": 0.0}
    count = 0

    for batch in loader:
        img = batch["img"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        conf = batch["conf"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            logits_low, rep_low = model(img)
            logits = F.interpolate(logits_low, size=target.shape[-2:], mode="bilinear", align_corners=False)
            rep = F.interpolate(rep_low, size=target.shape[-2:], mode="bilinear", align_corners=False)
            prob = torch.softmax(logits, dim=1)

            loss_ce = weighted_ce_loss(logits, target, conf)
            loss_dice = dice_loss_from_logits(logits, target, conf)

            label_oh = label_onehot(target, 2)
            mask_oh = conf.unsqueeze(1).repeat(1, 2, 1, 1)
            loss_reco = compute_reco_loss(
                rep=rep,
                label=label_oh,
                mask=mask_oh,
                prob=prob.detach(),
                strong_threshold=args.strong_threshold,
                temp=args.temp,
                num_queries=args.num_queries,
                num_negatives=args.num_negatives,
            ) if args.apply_reco else torch.tensor(0.0, device=device)

            loss = args.lambda_ce * loss_ce + args.lambda_dice * loss_dice + args.lambda_reco * loss_reco

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

        bs = img.shape[0]
        count += bs
        loss_meter["total"] += loss.item() * bs
        loss_meter["ce"] += loss_ce.item() * bs
        loss_meter["dice"] += loss_dice.item() * bs
        loss_meter["reco"] += float(loss_reco.item()) * bs

        metrics = compute_binary_metrics(logits.detach(), target, valid)
        for k, v in metrics.items():
            metric_meter[k] += v * bs

    for k in loss_meter:
        loss_meter[k] /= max(count, 1)
    for k in metric_meter:
        metric_meter[k] /= max(count, 1)
    return loss_meter, metric_meter



def save_checkpoint(path: str, model: nn.Module, optimizer, epoch: int, best_iou: float, args) -> None:
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_iou": best_iou,
        "args": vars(args),
    }, path)



def build_loaders(cfg: ReCoWeakConfig):
    train_samples, val_samples = split_samples(cfg)
    train_ds = ReCoWeakHRDataset(train_samples, cfg, is_train=True)
    val_ds = ReCoWeakHRDataset(val_samples, cfg, is_train=False)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    return train_loader, val_loader



def parse_args():
    parser = argparse.ArgumentParser(description="ReCo adapted training for HR image + noisy weak mask")
    parser.add_argument("--root_dir", type=str, default="./data/train")
    parser.add_argument("--save_dir", type=str, default="./experiment/reco_weak")
    parser.add_argument("--in_channels", type=int, default=3)
    parser.add_argument("--output_dim", type=int, default=128)
    parser.add_argument("--backbone", choices=["resnet50", "resnet101"], default="resnet101")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--no_pin_memory", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--hr_divisor", type=float, default=1024.0)
    parser.add_argument("--mask_threshold", type=float, default=20.0)
    parser.add_argument("--ignore_value", type=float, default=None,
                        help="Weak-mask no-data code excluded from all supervised losses and validation metrics.")
    parser.add_argument("--boundary_kernel", type=int, default=9)
    parser.add_argument("--boundary_weight", type=float, default=0.35)
    parser.add_argument("--min_confidence", type=float, default=0.15)
    parser.add_argument("--crop_size", type=int, default=0)
    parser.add_argument("--hflip_prob", type=float, default=0.5)
    parser.add_argument("--split_json", type=str, default="")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--pretrained_backbone", action="store_true")
    parser.add_argument("--apply_reco", action="store_true")
    parser.add_argument("--lambda_ce", type=float, default=1.0)
    parser.add_argument("--lambda_dice", type=float, default=1.0)
    parser.add_argument("--lambda_reco", type=float, default=0.05)
    parser.add_argument("--strong_threshold", type=float, default=0.97)
    parser.add_argument("--temp", type=float, default=0.5)
    parser.add_argument("--num_queries", type=int, default=128)
    parser.add_argument("--num_negatives", type=int, default=128)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    args = parser.parse_args()
    if args.no_pin_memory:
        args.pin_memory = False
    return args



def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ReCoWeakConfig(
        root_dir=args.root_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        hr_divisor=args.hr_divisor,
        mask_threshold=args.mask_threshold,
        boundary_kernel=args.boundary_kernel,
        boundary_weight=args.boundary_weight,
        min_confidence=args.min_confidence,
        crop_size=args.crop_size,
        hflip_prob=args.hflip_prob,
        split_json=args.split_json,
        ignore_value=args.ignore_value,
    )

    train_loader, val_loader = build_loaders(cfg)
    print(f"method=reco input_channels={args.in_channels} device={device}")
    print(f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} save_dir={args.save_dir}")

    model = build_model(args.in_channels, args.output_dim, args.pretrained_backbone, args.backbone).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = PolyLR(optimizer, max_iters=max(1, args.epochs * len(train_loader)), power=0.9, min_lr=1e-6)

    best_iou = -1.0
    epoch_times = []

    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_metrics = run_epoch(model, train_loader, optimizer, scheduler, device, args, train=True)
        val_loss, val_metrics = run_epoch(model, val_loader, None, None, device, args, train=False)
        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        avg_time = sum(epoch_times) / len(epoch_times)
        eta = avg_time * (args.epochs - epoch)

        if val_metrics["iou"] > best_iou:
            best_iou = val_metrics["iou"]
            save_checkpoint(os.path.join(args.save_dir, "best.pth"), model, optimizer, epoch, best_iou, args)
            stale_epochs = 0
        else:
            stale_epochs += 1
        save_checkpoint(os.path.join(args.save_dir, "last.pth"), model, optimizer, epoch, best_iou, args)

        print(
            f"[ReCo] {epoch:03d}/{args.epochs:03d} | "
            f"train={train_loss['total']:.4f} (ce={train_loss['ce']:.4f}, dice={train_loss['dice']:.4f}, reco={train_loss['reco']:.4f}, iou={train_metrics['iou']:.4f}) | "
            f"val={val_loss['total']:.4f} (iou={val_metrics['iou']:.4f}, dice={val_metrics['dice']:.4f}, acc={val_metrics['acc']:.4f}, recall={val_metrics['recall']:.4f}, precision={val_metrics['precision']:.4f}) | "
            f"time={elapsed:.1f}s avg={avg_time:.1f}s eta={format_eta(eta)}"
        )
        if args.patience > 0 and stale_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}: no validation IoU improvement for {args.patience} epochs.")
            break

    print(f"Best val IoU = {best_iou:.4f}")


if __name__ == "__main__":
    main()
