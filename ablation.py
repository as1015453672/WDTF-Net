"""Controlled Sentinel ablation trainer using the revised training protocol.

Supported variants remove one module family while retaining the same group-wise
validation split, weak labels, early stopping, Stage-1 teacher, and Stage-2
BatchNorm handling as the full revised WDTF-Net run.
"""
from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from wdtf.protocol import (RunConfig, TestDataset, WeakTrainDataset, mean_loss,
                           pairs, set_seed, test)
from wdtf.losses import Stage2AdaptiveGraphPrototypeLoss
from wdtf.stage1_loss import Stage1WeakPrototypeLoss
from wdtf.model import DoubleConv, WDTFNetConfig, WDTFNetOptimized, WaveletPriorAnalyzer


class IdentityDecoderGate(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x


class IdentityAdapter(nn.Module):
    """Parameter-free replacement used to isolate the Stage-2 WA contribution."""
    def forward(self, x):
        return x


class PlainPriorBlock(nn.Module):
    """Convolutional replacement for WGDC that retains frequency cues for FG."""
    def __init__(self, channels: int, num_templates: int = 6, wavelet_basis: str = "haar"):
        super().__init__()
        self.prior = WaveletPriorAnalyzer(channels, num_templates=num_templates, basis=wavelet_basis)
        self.refine = DoubleConv(channels, channels)
        self.res_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        prior = self.prior(x)
        y = x + self.res_scale * self.refine(x)
        return y, {key: prior[key] for key in ("alpha", "theta", "rho", "scale", "bfreq")}


def make_variant(cfg: RunConfig, variant: str, device: torch.device, wavelet_basis: str = "haar"):
    model = WDTFNetOptimized(WDTFNetConfig(in_channels=cfg.in_channels, base_channels=cfg.base_channels, num_templates=6, wavelet_basis=wavelet_basis))
    c = cfg.base_channels
    if variant == "no_wgdc":
        model.wdtf2 = PlainPriorBlock(c * 2, wavelet_basis=wavelet_basis)
        model.wdtf3 = PlainPriorBlock(c * 4, wavelet_basis=wavelet_basis)
        model.wdtf4 = PlainPriorBlock(c * 8, wavelet_basis=wavelet_basis)
    elif variant == "no_fg":
        model.dec_gate3 = IdentityDecoderGate(); model.dec_gate2 = IdentityDecoderGate(); model.dec_gate1 = IdentityDecoderGate()
    elif variant == "no_wa":
        model.adapter_wdtf2 = IdentityAdapter(); model.adapter_wdtf3 = IdentityAdapter(); model.adapter_wdtf4 = IdentityAdapter()
        model.adapter_up3 = IdentityAdapter(); model.adapter_up2 = IdentityAdapter()
    elif variant != "full":
        raise ValueError(f"Unknown variant: {variant}")
    return model.to(device)


def save(path, model, cfg, variant, stage, val_loss, wavelet_basis):
    torch.save({"model": model.state_dict(), "config": asdict(cfg), "variant": variant,
                "stage": stage, "val_loss": val_loss, "wavelet_basis": wavelet_basis}, path)


def train_stage1(model, train_loader, val_loader, cfg, device, out, variant, wavelet_basis):
    criterion = Stage1WeakPrototypeLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr1, weight_decay=cfg.weight_decay)
    best, bad, path = float("inf"), 0, out / "stage1_best.pt"
    last_path = out / "stage1_last.pt"
    for epoch in range(cfg.stage1_epochs):
        model.train(); train_loss = 0.0
        for batch in train_loader:
            image = batch["img"].to(device)
            result = criterion(model(image, False), image, batch["mask"].to(device), batch["conf"].to(device), batch["boundary"].to(device), epoch)
            optimizer.zero_grad(); result["loss"].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            train_loss += float(result["loss"])
        val = mean_loss(model, val_loader, criterion, device)
        print(f"{variant}/stage1 {epoch+1:03d}: train={train_loss/len(train_loader):.5f}; val_weak_loss={val:.5f}", flush=True)
        save(last_path, model, cfg, variant, 1, val, wavelet_basis)
        if val < best - cfg.min_delta:
            best, bad = val, 0; save(path, model, cfg, variant, 1, val, wavelet_basis)
        else:
            bad += 1
            if bad >= cfg.patience: print(f"{variant}/stage1 early stop", flush=True); break
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"], strict=True)
    return path


def train_stage2(model, train_loader, val_loader, cfg, device, out, variant, wavelet_basis):
    teacher = copy.deepcopy(model).eval()
    for parameter in teacher.parameters(): parameter.requires_grad = False
    model.freeze_for_stage2(train_seg_head=True, tune_geometry=False)
    criterion = Stage2AdaptiveGraphPrototypeLoss(lambda_proto=1.0, lambda_cons=0.22, lambda_smooth=0.05,
        lambda_prior=0.15, lambda_absent=0.32, proto_uncertain_weight=0.35,
        gc_stage1_weight=0.85, gc_prior_weight=0.15, gc_strong_weight=1.0, gc_mid_weight=0.30).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr2, weight_decay=cfg.weight_decay)
    best, bad, path = float("inf"), 0, out / "stage2_best.pt"
    last_path = out / "stage2_last.pt"
    for epoch in range(cfg.stage2_epochs):
        model.set_stage2_train_mode(); train_loss = 0.0
        for batch in train_loader:
            image = batch["img"].to(device); weak = batch["mask"].to(device)
            with torch.no_grad(): anchor = teacher(image, False)
            result = criterion(image, anchor, model(image, True), weak, compute_spec=False)
            optimizer.zero_grad(); result["loss"].backward(); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0); optimizer.step()
            train_loss += float(result["loss"])
        val = mean_loss(model, val_loader, criterion, device, stage2=True, teacher=teacher)
        print(f"{variant}/stage2 {epoch+1:03d}: train={train_loss/len(train_loader):.5f}; val_weak_loss={val:.5f}", flush=True)
        save(last_path, model, cfg, variant, 2, val, wavelet_basis)
        if val < best - cfg.min_delta:
            best, bad = val, 0; save(path, model, cfg, variant, 2, val, wavelet_basis)
        else:
            bad += 1
            if bad >= cfg.patience: print(f"{variant}/stage2 early stop", flush=True); break
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", required=True); parser.add_argument("--test-root", help="Optional held-out hr/ and label/ test root."); parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variant", choices=("no_wgdc", "no_fg", "no_wa", "full"), required=True)
    parser.add_argument("--wavelet-basis", choices=("haar", "db2", "sobel", "lowpass"), default="haar")
    parser.add_argument("--val-group", default="05"); parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=5); parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--in-channels", type=int, default=4); parser.add_argument("--base-channels", type=int, default=64); parser.add_argument("--divisor", type=float, default=4096.0)
    args = parser.parse_args()
    cfg = RunConfig(args.train_root, args.test_root or "", args.output_dir, args.val_group, seed=args.seed, in_channels=args.in_channels, base_channels=args.base_channels, batch_size=args.batch_size,
                    stage1_epochs=args.epochs, stage2_epochs=args.epochs, patience=args.patience, divisor=args.divisor)
    set_seed(cfg.seed); out = Path(cfg.output_dir); out.mkdir(parents=True, exist_ok=True)
    items = pairs(cfg.train_root); train = [x for x in items if x["id"].split("_")[0] != cfg.val_group]; val = [x for x in items if x["id"].split("_")[0] == cfg.val_group]
    (out / "split.json").write_text(json.dumps({"variant": args.variant, "wavelet_basis": args.wavelet_basis, "seed": args.seed, "train": [x["id"] for x in train], "val": [x["id"] for x in val]}, indent=2), encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(WeakTrainDataset(train, cfg, True), batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers, pin_memory=True)
    val_loader = DataLoader(WeakTrainDataset(val, cfg, False), batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.workers, pin_memory=True)
    model = make_variant(cfg, args.variant, device, args.wavelet_basis)
    print(f"variant={args.variant}; basis={args.wavelet_basis}; seed={args.seed}; device={device}; train={len(train)}; val={len(val)}", flush=True)
    stage1 = train_stage1(model, train_loader, val_loader, cfg, device, out, args.variant, args.wavelet_basis)
    stage2 = train_stage2(model, train_loader, val_loader, cfg, device, out, args.variant, args.wavelet_basis)
    if args.test_root:
        model.load_state_dict(torch.load(stage2, map_location=device, weights_only=False)["model"], strict=True)
        test(model, DataLoader(TestDataset(args.test_root, cfg), batch_size=1, shuffle=False), device, out)
    print(json.dumps({"stage1": str(stage1), "stage2": str(stage2), "variant": args.variant, "wavelet_basis": args.wavelet_basis, "seed": args.seed}), flush=True)


if __name__ == "__main__":
    main()
