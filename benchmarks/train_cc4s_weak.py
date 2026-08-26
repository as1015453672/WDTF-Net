import argparse
import copy
import math
import os
import random
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

import model_RW

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class WeakHRConfig:
    root_dir: str
    hr_dir: str = "hr"
    mask_dir: str = "mask"
    hr_suffix: str = "hr"
    mask_suffix: str = "mask"
    val_ratio: float = 0.1
    seed: int = 42
    hr_divisor: float = 1024.0
    mask_threshold: float = 20.0
    mask_binarize: bool = True
    boundary_kernel: int = 9
    boundary_weight: float = 0.35
    interior_weight: float = 1.0
    min_confidence: float = 0.15
    num_workers: int = 0
    pin_memory: bool = True


def _strip_known_suffix(name_no_ext: str, suffix: str) -> str:
    token = "_" + suffix
    if name_no_ext.endswith(token):
        return name_no_ext[: -len(token)]
    if name_no_ext.endswith(suffix):
        return name_no_ext[: -len(suffix)]
    return name_no_ext


def _find_samples(cfg: WeakHRConfig) -> List[Dict[str, str]]:
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


def split_samples(cfg: WeakHRConfig) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
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


def make_boundary_band(mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size % 2 == 0:
        kernel_size += 1
    if mask.ndim == 3:
        mask = mask.unsqueeze(0)
    pad = kernel_size // 2
    dil = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad)
    ero = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=kernel_size, stride=1, padding=pad)
    band = (dil - ero).clamp(0.0, 1.0)
    return band.squeeze(0)


class CC4SWeakHRDataset(Dataset):
    def __init__(self, samples: List[Dict[str, str]], cfg: WeakHRConfig, train: bool = True):
        self.samples = samples
        self.cfg = cfg
        self.train = train

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.train:
            return img, mask

        if random.random() < 0.5:
            img = torch.flip(img, dims=[2])
            mask = torch.flip(mask, dims=[2])

        if random.random() < 0.5:
            img = torch.flip(img, dims=[1])
            mask = torch.flip(mask, dims=[1])

        k = random.randint(0, 3)
        if k > 0:
            img = torch.rot90(img, k=k, dims=[1, 2])
            mask = torch.rot90(mask, k=k, dims=[1, 2])

        return img.contiguous(), mask.contiguous()

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        hr = read_raster(s["hr_path"])
        weak_mask = read_raster(s["mask_path"])

        hr = normalize_image(hr, self.cfg.hr_divisor)
        weak_mask = process_mask(weak_mask, self.cfg.mask_threshold, self.cfg.mask_binarize)

        hr_t = torch.from_numpy(hr)
        mask_t = torch.from_numpy(weak_mask)
        hr_t, mask_t = self._augment(hr_t, mask_t)

        hard = (mask_t > 0.5).float()
        boundary = make_boundary_band(hard, self.cfg.boundary_kernel)
        conf = torch.full_like(mask_t, float(self.cfg.interior_weight))
        conf = conf * (1.0 - boundary) + float(self.cfg.boundary_weight) * boundary
        conf = conf.clamp(min=float(self.cfg.min_confidence), max=1.0)
        edge = mask_to_edge(hard)

        return {
            "id": s["id"],
            "img": hr_t,
            "mask": mask_t,
            "hard_mask": hard,
            "conf": conf,
            "boundary": boundary,
            "edge": edge,
        }


class SoftPseudoDataset(CC4SWeakHRDataset):
    def __init__(self, samples: List[Dict[str, str]], cfg: WeakHRConfig, pseudo_root: str, train: bool = True):
        super().__init__(samples, cfg, train=train)
        self.pseudo_root = Path(pseudo_root)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        out = super().__getitem__(idx)
        pseudo_path = self.pseudo_root / f"{out['id']}.npy"
        if not pseudo_path.exists():
            raise FileNotFoundError(f"pseudo label not found: {pseudo_path}")
        pseudo = np.load(pseudo_path).astype(np.float32)
        if pseudo.ndim == 2:
            pseudo = pseudo[None, ...]
        pseudo_t = torch.from_numpy(pseudo)
        if self.train:
            # reuse same flips/rotations is hard without storing ops; safest is no heavy aug in stage2
            pass
        out["pseudo"] = pseudo_t
        return out


class ConfidenceWeightedBCEDice(nn.Module):
    def __init__(self, dice_weight: float = 1.0, bce_weight: float = 1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor, conf: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        prob = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        bce = (bce * conf).sum() / (conf.sum() + 1e-6)

        inter = (prob * target * conf).sum(dim=(1, 2, 3))
        denom = ((prob + target) * conf).sum(dim=(1, 2, 3))
        dice = 1.0 - ((2.0 * inter + 1e-6) / (denom + 1e-6)).mean()

        loss = self.bce_weight * bce + self.dice_weight * dice
        return loss, {"bce": float(bce.detach()), "dice": float(dice.detach())}


class BoundaryEntropyLoss(nn.Module):
    def forward(self, logits: torch.Tensor, boundary: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
        entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
        return (entropy * boundary).sum() / (boundary.sum() + 1e-6)


class SoftConsistencyLoss(nn.Module):
    def forward(self, logits: torch.Tensor, pseudo: torch.Tensor, conf: Optional[torch.Tensor] = None) -> torch.Tensor:
        prob = torch.sigmoid(logits)
        loss = F.binary_cross_entropy(prob, pseudo, reduction="none")
        if conf is not None:
            loss = loss * conf
            return loss.sum() / (conf.sum() + 1e-6)
        return loss.mean()


class BinaryCC4SModel(nn.Module):
    def __init__(self, layers: int = 18, shrink_factor: int = 2, in_channels: int = 3):
        super().__init__()
        self.net = model_RW.Res_Deeplab(num_classes=2, layers=layers, shrink_factor=shrink_factor)
        if in_channels != 3:
            old = self.net.model_sed.conv1
            new = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                if in_channels == 1:
                    new.weight.copy_(old.weight.mean(dim=1, keepdim=True))
                elif in_channels > 3:
                    new.weight[:, :3].copy_(old.weight)
                    extra = new.weight[:, 3:]
                    extra.copy_(old.weight.mean(dim=1, keepdim=True).repeat(1, in_channels - 3, 1, 1))
                else:
                    new.weight[:, :in_channels].copy_(old.weight[:, :in_channels])
            self.net.model_sed.conv1 = new
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits2, P, sed, pred4 = self.net(x)
        logits = logits2[:, 1:2] - logits2[:, 0:1]
        return logits, P, sed, pred4


@torch.no_grad()
def update_ema(student: nn.Module, teacher: nn.Module, decay: float) -> None:
    student_state = student.state_dict()
    for k, v in teacher.state_dict().items():
        v.copy_(v * decay + student_state[k] * (1.0 - decay))


def compute_binary_metrics(prob: torch.Tensor, target: torch.Tensor, thr: float = 0.5) -> Dict[str, float]:
    pred = (prob >= thr).astype(np.uint8).ravel()
    gt = (target >= 0.5).astype(np.uint8).ravel()
    if gt.sum() == 0 and pred.sum() == 0:
        iou = dice = precision = recall = f1 = 1.0
    else:
        iou = jaccard_score(gt, pred, zero_division=0)
        precision = precision_score(gt, pred, zero_division=0)
        recall = recall_score(gt, pred, zero_division=0)
        f1 = f1_score(gt, pred, zero_division=0)
        dice = f1
    acc = float((pred == gt).mean())
    return {"iou": iou, "dice": dice, "precision": precision, "recall": recall, "f1": f1, "acc": acc}


@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float = 0.5) -> Dict[str, float]:
    model.eval()
    probs_all, gts_all = [], []
    for batch in loader:
        img = batch["img"].to(device, non_blocking=True)
        gt_t = batch["hard_mask"]
        logits, _, _, _ = model(img)
        logits = F.interpolate(logits, size=gt_t.shape[-2:], mode="bilinear", align_corners=True)
        prob = torch.sigmoid(logits).cpu().numpy()
        gt = gt_t.cpu().numpy()
        probs_all.append(prob)
        gts_all.append(gt)
    probs = np.concatenate(probs_all, axis=0)
    gts = np.concatenate(gts_all, axis=0)
    return compute_binary_metrics(probs, gts, thr=threshold)

@torch.no_grad()
def generate_pseudo_labels(model: nn.Module, loader: DataLoader, device: torch.device, out_dir: str) -> None:
    model.eval()
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    for batch in loader:
        img = batch["img"].to(device, non_blocking=True)
        ids = batch["id"]
        target_size = batch["hard_mask"].shape[-2:]
        logits, _, _, _ = model(img)
        logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=True)
        prob = torch.sigmoid(logits).cpu().numpy()
        for sid, p in zip(ids, prob):
            np.save(out_root / f"{sid}.npy", p.astype(np.float32))


def build_loaders(cfg: WeakHRConfig, batch_size: int, train_mode: str, pseudo_root: Optional[str] = None):
    train_samples, val_samples = split_samples(cfg)
    if train_mode == "stage2":
        train_ds = SoftPseudoDataset(train_samples, cfg, pseudo_root=pseudo_root, train=False)
        val_ds = SoftPseudoDataset(val_samples, cfg, pseudo_root=pseudo_root, train=False) if val_samples else None
    else:
        train_ds = CC4SWeakHRDataset(train_samples, cfg, train=True)
        val_ds = CC4SWeakHRDataset(val_samples, cfg, train=False) if val_samples else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
    )
    val_loader = None
    if val_ds is not None and len(val_ds) > 0:
        val_loader = DataLoader(
            val_ds,
            batch_size=max(1, batch_size // 2),
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=False,
        )
    return train_loader, val_loader


def train_stage1(args) -> str:
    cfg = WeakHRConfig(
        root_dir=args.root_dir,
        hr_dir=args.hr_dir,
        mask_dir=args.mask_dir,
        hr_suffix=args.hr_suffix,
        mask_suffix=args.mask_suffix,
        val_ratio=args.val_ratio,
        seed=args.seed,
        hr_divisor=args.hr_divisor,
        mask_threshold=args.mask_threshold,
        mask_binarize=not args.mask_soft,
        boundary_kernel=args.boundary_kernel,
        boundary_weight=args.boundary_weight,
        interior_weight=args.interior_weight,
        min_confidence=args.min_confidence,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    train_loader, val_loader = build_loaders(cfg, args.batch_size, train_mode="stage1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BinaryCC4SModel(layers=args.layers, shrink_factor=args.shrink_factor, in_channels=args.in_channels).to(device)
    teacher = copy.deepcopy(model).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    sup_criterion = ConfidenceWeightedBCEDice(args.dice_weight, args.bce_weight)
    boundary_entropy = BoundaryEntropyLoss()
    consistency = nn.MSELoss()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "stage1_best.pth"
    last_path = save_dir / "stage1_last.pth"

    best_iou = -1.0
    stale_epochs = 0
    epoch_times: List[float] = []

    for epoch in range(1, args.epochs + 1):
        start_t = time.time()
        model.train()
        meter = {"loss": 0.0, "sup": 0.0, "entropy": 0.0, "cons": 0.0, "n": 0}

        for batch in train_loader:
            img = batch["img"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            conf = batch["conf"].to(device, non_blocking=True)
            boundary = batch["boundary"].to(device, non_blocking=True)

            logits, _, feat, _ = model(img)
            logits = F.interpolate(logits, size=mask.shape[-2:], mode="bilinear", align_corners=True)
            sup_loss, loss_dict = sup_criterion(logits, mask, conf)

            entropy_loss = boundary_entropy(logits, boundary)

            img_flip = torch.flip(img, dims=[3])
            _, _, feat_flip, _ = model(img_flip)
            cons_loss = consistency(feat, torch.flip(feat_flip, dims=[3]))

            with torch.no_grad():
                teacher_logits, _, _, _ = teacher(img)
                teacher_logits = F.interpolate(teacher_logits, size=mask.shape[-2:], mode="bilinear", align_corners=True)
                teacher_prob = torch.sigmoid(teacher_logits)
                teacher_conf = (teacher_prob - 0.5).abs() * 2.0

            student_prob = torch.sigmoid(logits)
            teacher_align = F.binary_cross_entropy(student_prob, teacher_prob, reduction="none")
            teacher_align = (teacher_align * teacher_conf).sum() / (teacher_conf.sum() + 1e-6)

            loss = sup_loss + args.lambda_entropy * entropy_loss + args.lambda_cons * cons_loss + args.lambda_teacher * teacher_align

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            update_ema(model, teacher, args.ema_decay)

            bsz = img.size(0)
            meter["loss"] += float(loss.detach()) * bsz
            meter["sup"] += float(sup_loss.detach()) * bsz
            meter["entropy"] += float(entropy_loss.detach()) * bsz
            meter["cons"] += float((cons_loss + teacher_align).detach()) * bsz
            meter["n"] += bsz

        scheduler.step()
        epoch_time = time.time() - start_t
        epoch_times.append(epoch_time)
        avg_epoch = sum(epoch_times) / len(epoch_times)
        eta = avg_epoch * (args.epochs - epoch)

        train_log = {k: (v / max(meter["n"], 1)) for k, v in meter.items() if k != "n"}
        msg = (
            f"[Stage1] {epoch:03d}/{args.epochs:03d} | "
            f"train={train_log['loss']:.4f} "
            f"(sup={train_log['sup']:.4f}, ent={train_log['entropy']:.4f}, cons={train_log['cons']:.4f}) | "
            f"time={epoch_time:.1f}s avg={avg_epoch:.1f}s eta={eta/3600:.2f}h"
        )

        if val_loader is not None:
            metrics = validate(model, val_loader, device, threshold=args.threshold)
            msg += f" | val(iou={metrics['iou']:.4f}, dice={metrics['dice']:.4f}, acc={metrics['acc']:.4f})"
            is_best = metrics["iou"] > best_iou
            if is_best:
                best_iou = metrics["iou"]
                stale_epochs = 0
                torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, best_path)
            else:
                stale_epochs += 1
        else:
            metrics = None
            is_best = False

        print(msg)
        torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, last_path)
        if args.patience > 0 and stale_epochs >= args.patience:
            print(f"Early stopping stage 1 at epoch {epoch}: no validation IoU improvement for {args.patience} epochs.")
            break

    pseudo_dir = save_dir / "pseudo_stage1"
    full_ds = CC4SWeakHRDataset(_find_samples(cfg), cfg, train=False)
    full_loader = DataLoader(full_ds, batch_size=max(1, args.batch_size // 2), shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    best_model = BinaryCC4SModel(layers=args.layers, shrink_factor=args.shrink_factor, in_channels=args.in_channels).to(device)
    ckpt = torch.load(best_path if best_path.exists() else last_path, map_location=device)
    best_model.load_state_dict(ckpt["model"])
    generate_pseudo_labels(best_model, full_loader, device, str(pseudo_dir))
    print(f"Stage1 pseudo labels saved to: {pseudo_dir}")
    return str(best_path if best_path.exists() else last_path)


def train_stage2(args) -> str:
    cfg = WeakHRConfig(
        root_dir=args.root_dir,
        hr_dir=args.hr_dir,
        mask_dir=args.mask_dir,
        hr_suffix=args.hr_suffix,
        mask_suffix=args.mask_suffix,
        val_ratio=args.val_ratio,
        seed=args.seed,
        hr_divisor=args.hr_divisor,
        mask_threshold=args.mask_threshold,
        mask_binarize=not args.mask_soft,
        boundary_kernel=args.boundary_kernel,
        boundary_weight=args.boundary_weight,
        interior_weight=args.interior_weight,
        min_confidence=args.min_confidence,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    train_loader, val_loader = build_loaders(cfg, args.batch_size_stage2, train_mode="stage2", pseudo_root=args.pseudo_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BinaryCC4SModel(layers=args.layers, shrink_factor=args.shrink_factor, in_channels=args.in_channels).to(device)
    ckpt = torch.load(args.stage1_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr_stage2, weight_decay=args.weight_decay_stage2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_stage2)

    soft_consistency = SoftConsistencyLoss()
    boundary_entropy = BoundaryEntropyLoss()
    sup_criterion = ConfidenceWeightedBCEDice(args.dice_weight, args.bce_weight)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "stage2_best.pth"
    last_path = save_dir / "stage2_last.pth"

    best_iou = -1.0
    stale_epochs = 0
    epoch_times: List[float] = []

    for epoch in range(1, args.epochs_stage2 + 1):
        start_t = time.time()
        model.train()
        meter = {"loss": 0.0, "soft": 0.0, "weak": 0.0, "ent": 0.0, "n": 0}

        for batch in train_loader:
            img = batch["img"].to(device, non_blocking=True)
            weak = batch["mask"].to(device, non_blocking=True)
            hard = batch["hard_mask"].to(device, non_blocking=True)
            conf = batch["conf"].to(device, non_blocking=True)
            boundary = batch["boundary"].to(device, non_blocking=True)
            pseudo = batch["pseudo"].to(device, non_blocking=True)

            logits, _, _, _ = model(img)
            logits = F.interpolate(logits, size=weak.shape[-2:], mode="bilinear", align_corners=True)

            soft_loss = soft_consistency(logits, pseudo, conf=None)
            weak_loss, _ = sup_criterion(logits, hard, conf)
            ent_loss = boundary_entropy(logits, boundary)
            loss = args.lambda_soft * soft_loss + args.lambda_weak_stage2 * weak_loss + args.lambda_entropy_stage2 * ent_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bsz = img.size(0)
            meter["loss"] += float(loss.detach()) * bsz
            meter["soft"] += float(soft_loss.detach()) * bsz
            meter["weak"] += float(weak_loss.detach()) * bsz
            meter["ent"] += float(ent_loss.detach()) * bsz
            meter["n"] += bsz

        scheduler.step()
        epoch_time = time.time() - start_t
        epoch_times.append(epoch_time)
        avg_epoch = sum(epoch_times) / len(epoch_times)
        eta = avg_epoch * (args.epochs_stage2 - epoch)
        train_log = {k: (v / max(meter["n"], 1)) for k, v in meter.items() if k != "n"}

        msg = (
            f"[Stage2] {epoch:03d}/{args.epochs_stage2:03d} | "
            f"train={train_log['loss']:.4f} "
            f"(soft={train_log['soft']:.4f}, weak={train_log['weak']:.4f}, ent={train_log['ent']:.4f}) | "
            f"time={epoch_time:.1f}s avg={avg_epoch:.1f}s eta={eta/3600:.2f}h"
        )

        if val_loader is not None:
            metrics = validate(model, val_loader, device, threshold=args.threshold)
            msg += f" | val(iou={metrics['iou']:.4f}, dice={metrics['dice']:.4f}, acc={metrics['acc']:.4f})"
            is_best = metrics["iou"] > best_iou
            if is_best:
                best_iou = metrics["iou"]
                stale_epochs = 0
                torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, best_path)
            else:
                stale_epochs += 1
        else:
            metrics = None
        print(msg)
        torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics}, last_path)
        if args.patience > 0 and stale_epochs >= args.patience:
            print(f"Early stopping stage 2 at epoch {epoch}: no validation IoU improvement for {args.patience} epochs.")
            break

    return str(best_path if best_path.exists() else last_path)


def parse_args():
    p = argparse.ArgumentParser("CC4S weak HR training for noisy water masks")
    p.add_argument("--root_dir", type=str, default="./data/train")
    p.add_argument("--hr_dir", type=str, default="hr")
    p.add_argument("--mask_dir", type=str, default="mask")
    p.add_argument("--hr_suffix", type=str, default="hr")
    p.add_argument("--mask_suffix", type=str, default="mask")
    p.add_argument("--save_dir", type=str, default="./experiment/cc4s_weak")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--hr_divisor", type=float, default=1024.0)
    p.add_argument("--mask_threshold", type=float, default=20.0)
    p.add_argument("--mask_soft", action="store_true")
    p.add_argument("--boundary_kernel", type=int, default=9)
    p.add_argument("--boundary_weight", type=float, default=0.35)
    p.add_argument("--interior_weight", type=float, default=1.0)
    p.add_argument("--min_confidence", type=float, default=0.15)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--in_channels", type=int, default=3)

    p.add_argument("--layers", type=int, default=18)
    p.add_argument("--shrink_factor", type=int, default=2)

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--bce_weight", type=float, default=1.0)
    p.add_argument("--dice_weight", type=float, default=1.0)
    p.add_argument("--lambda_entropy", type=float, default=0.10)
    p.add_argument("--lambda_cons", type=float, default=0.05)
    p.add_argument("--lambda_teacher", type=float, default=0.10)
    p.add_argument("--ema_decay", type=float, default=0.99)
    p.add_argument("--patience", type=int, default=5)

    p.add_argument("--run_stage2", action="store_true", default=True)
    p.add_argument("--pseudo_dir", type=str, default="")
    p.add_argument("--stage1_ckpt", type=str, default="")
    p.add_argument("--batch_size_stage2", type=int, default=8)
    p.add_argument("--epochs_stage2", type=int, default=20)
    p.add_argument("--lr_stage2", type=float, default=1e-4)
    p.add_argument("--weight_decay_stage2", type=float, default=1e-4)
    p.add_argument("--lambda_soft", type=float, default=1.0)
    p.add_argument("--lambda_weak_stage2", type=float, default=0.30)
    p.add_argument("--lambda_entropy_stage2", type=float, default=0.05)

    p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    stage1_ckpt = args.stage1_ckpt
    if not stage1_ckpt:
        stage1_ckpt = train_stage1(args)

    if args.run_stage2:
        if not args.pseudo_dir:
            args.pseudo_dir = str(Path(args.save_dir) / "pseudo_stage1")
        args.stage1_ckpt = stage1_ckpt
        final_ckpt = train_stage2(args)
        print(f"Final stage2 checkpoint: {final_ckpt}")
    else:
        print(f"Final stage1 checkpoint: {stage1_ckpt}")
