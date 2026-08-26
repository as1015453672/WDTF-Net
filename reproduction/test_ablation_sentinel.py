# -*- coding: utf-8 -*-
"""
Ablation test script for WDTF-Net variants 0-7.

Output for each sample:
    HR | Label | 0 Baseline | 1 +WGDC | 2 +FG | 3 +WA | 4 w/o WGDC | 5 w/o FG | 6 w/o WA | 7 Full

Dataset structure:
    TEST_ROOT/
        hr/     xxx_hr.tif
        label/  xxx_label.tif

Label rule:
    label > 0 => water

Usage:
    Put this file in the same folder as wdtf_net.py and your checkpoints.
    Edit the CONFIG section below, then run:
        python test_wdtfnet_ablation.py
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import csv
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torch.nn.functional as F

import rasterio
from rasterio.errors import NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

from wdtf_net import (
    WDTFNetConfig,
    WDTFNetOptimized,
    WaveletPriorAnalyzer,
    DoubleConv,
)

# ============================================================
# 1. CONFIG: directly edit here
# ============================================================

TEST_ROOT = "./data/sentinel/test"       # must contain hr/ and label/
OUTPUT_DIR = r"./experiment_sentinel/test_ablation_wdtfnet"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IN_CHANNELS = 4
HR_DIVISOR = 4096.0
THRESHOLD = 0.5
SAVE_PROB = False
SAVE_BIN = False

# Visualization settings. Metrics use original size; only panels are resized for display.
PANEL_MAX_SIDE = 512
TITLE_HEIGHT = 34
GAP = 6
RGB_BANDS = (0, 1, 2)      # If WorldView order is B,G,R,NIR, set (2,1,0)

# Model hyperparameters. Must match training.
BASE_CHANNELS = 64
ADAPTER_REDUCTION = 4
MAX_RESIDUAL_OFFSET = 0.35
NUM_TEMPLATES = 6

# ------------------------------------------------------------
# Ablation variants
# ------------------------------------------------------------
# ID definition:
#   0 Baseline       : WGDC x, FG x, WA x, Stage-1 checkpoint
#   1 Baseline+WGDC  : WGDC √, FG x, WA x, Stage-1 checkpoint
#   2 Baseline+FG    : WGDC x, FG √, WA x, Stage-1 checkpoint
#   3 Baseline+WA    : WGDC x, FG x, WA √, Stage-2 checkpoint
#   4 w/o WGDC       : WGDC x, FG √, WA √, Stage-2 checkpoint
#   5 w/o FG         : WGDC √, FG x, WA √, Stage-2 checkpoint
#   6 w/o WA         : WGDC √, FG √, WA x, your existing Stage-1 checkpoint
#   7 Full WDTF-Net  : WGDC √, FG √, WA √, your existing Stage-2 checkpoint
#
# Edit ckpt paths according to your training outputs. The first existing path in
# ckpt_candidates is used.

ABLATION_CONFIGS: List[Dict[str, Any]] = [
    dict(
        id=0, name="ablation_0_baseline", display="0 Baseline", enabled=True,
        use_wgdc=False, use_fg=False, use_wa=False,
        ckpt=r"./experiment_sentinel/ablation_sentinel/00_baseline/stage1/stage1_last.pth",
    ),
    dict(
        id=1, name="ablation_1_wgdc", display="1 +WGDC", enabled=True,
        use_wgdc=True, use_fg=False, use_wa=False,
        ckpt=r"./experiment_sentinel/ablation_wdtfnet/01_baseline_wgdc/stage1/stage1_best.pth",
    ),
    dict(
        id=2, name="ablation_2_fg", display="2 +FG", enabled=True,
        use_wgdc=False, use_fg=True, use_wa=False,
        ckpt=r"./experiment_sentinel/ablation_wdtfnet/04_wo_wgdc/stage1/stage1_best.pth",
    ),
    dict(
        id=3, name="ablation_3_wa", display="3 +WA", enabled=True,
        use_wgdc=False, use_fg=False, use_wa=True,
        ckpt=r"./experiment_sentinel/ablation_sentinel/03_baseline_wa/stage2/stage2_best.pth",
    ),
    dict(
        id=4, name="ablation_4_wo_wgdc", display="4 w/o WGDC", enabled=True,
        use_wgdc=False, use_fg=True, use_wa=True,
        ckpt=r"./experiment_sentinel/ablation_wdtfnet/04_wo_wgdc/stage2/stage2_best.pth",
    ),
    dict(
        id=5, name="ablation_5_wo_fg", display="5 w/o FG", enabled=True,
        use_wgdc=True, use_fg=False, use_wa=True,
        ckpt=r"./experiment_sentinel/ablation_wdtfnet/05_wo_fg/stage2/stage2_best.pth",
    ),
    dict(
        id=6, name="ablation_6_wo_wa", display="6 w/o WA", enabled=True,
        use_wgdc=True, use_fg=True, use_wa=False,
        ckpt=r"./experiment_sentinel/ablation_wdtfnet/07_full_wdtfnet/stage1_best.pth",
    ),
    dict(
        id=7, name="ablation_7_full", display="7 Full", enabled=True,
        use_wgdc=True, use_fg=True, use_wa=True,
        ckpt=r"./experiment_sentinel/ablation_wdtfnet/07_full_wdtfnet/stage2_best.pth",
    ),
]

# ============================================================
# 2. Basic utilities
# ============================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def strip_suffix(name_no_ext: str, suffix: str) -> str:
    token = "_" + suffix
    if name_no_ext.endswith(token):
        return name_no_ext[:-len(token)]
    if name_no_ext.endswith(suffix):
        return name_no_ext[:-len(suffix)]
    return name_no_ext


def read_raster(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        arr = src.read()
    return arr.astype(np.float32)


def normalize_image(arr: np.ndarray, divisor: float) -> np.ndarray:
    return arr.astype(np.float32) / float(divisor)


def binarize_label(arr: np.ndarray) -> np.ndarray:
    return (arr > 0).astype(np.float32)


def find_test_samples(test_root: str) -> List[Dict[str, str]]:
    hr_root = Path(test_root) / "hr"
    label_root = Path(test_root) / "label"
    if not hr_root.exists():
        raise FileNotFoundError(f"hr folder not found: {hr_root}")
    if not label_root.exists():
        raise FileNotFoundError(f"label folder not found: {label_root}")

    hr_map = {strip_suffix(p.stem, "hr"): str(p) for p in hr_root.iterdir() if p.is_file()}
    label_map = {strip_suffix(p.stem, "label"): str(p) for p in label_root.iterdir() if p.is_file()}
    common = sorted(set(hr_map) & set(label_map))
    if not common:
        raise RuntimeError("No matched samples found. Expected xxx_hr.tif and xxx_label.tif")
    return [{"id": sid, "hr_path": hr_map[sid], "label_path": label_map[sid]} for sid in common]


def resize_logits_to(logits: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    if logits.shape[-2:] != size_hw:
        logits = F.interpolate(logits, size=size_hw, mode="bilinear", align_corners=False)
    return logits


def compute_binary_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    pred = pred.astype(np.uint8)
    target = target.astype(np.uint8)
    tp = np.logical_and(pred == 1, target == 1).sum()
    tn = np.logical_and(pred == 0, target == 0).sum()
    fp = np.logical_and(pred == 1, target == 0).sum()
    fn = np.logical_and(pred == 0, target == 1).sum()
    eps = 1e-6
    return {
        "IoU": float(tp / (tp + fp + fn + eps)),
        "Dice": float((2 * tp) / (2 * tp + fp + fn + eps)),
        "F1": float((2 * tp) / (2 * tp + fp + fn + eps)),
        "Acc": float((tp + tn) / (tp + tn + fp + fn + eps)),
        "Recall": float(tp / (tp + fn + eps)),
        "Precision": float(tp / (tp + fp + eps)),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


def metrics_from_counts(tp: int, tn: int, fp: int, fn: int, prefix: str = "") -> Dict[str, float]:
    eps = 1e-6
    return {
        prefix + "IoU": float(tp / (tp + fp + fn + eps)),
        prefix + "Dice": float((2 * tp) / (2 * tp + fp + fn + eps)),
        prefix + "F1": float((2 * tp) / (2 * tp + fp + fn + eps)),
        prefix + "Acc": float((tp + tn) / (tp + tn + fp + fn + eps)),
        prefix + "Recall": float(tp / (tp + fn + eps)),
        prefix + "Precision": float(tp / (tp + fp + eps)),
    }


def format_metrics(m: Dict[str, float]) -> str:
    return (f"IoU={m['IoU']:.4f} Dice={m['Dice']:.4f} F1={m['F1']:.4f} "
            f"Acc={m['Acc']:.4f} Recall={m['Recall']:.4f} Precision={m['Precision']:.4f}")

# ============================================================
# 3. Visualization utilities
# ============================================================

def chw_to_rgb_uint8(img_chw: np.ndarray) -> np.ndarray:
    c, h, w = img_chw.shape
    if c == 1:
        arr = np.repeat(img_chw, 3, axis=0)
    elif c >= 3:
        idx = [b for b in RGB_BANDS if b < c]
        if len(idx) < 3:
            idx = list(range(min(c, 3)))
        arr = img_chw[idx[:3]]
    else:
        arr = np.concatenate([img_chw, img_chw[:1]], axis=0)[:3]
    arr = np.transpose(arr, (1, 2, 0))
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)
    return (arr * 255.0).round().astype(np.uint8)


def gray_to_rgb_uint8(arr_hw: np.ndarray) -> np.ndarray:
    arr = np.clip(arr_hw, 0.0, 1.0)
    g = (arr * 255.0).round().astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


def fit_panel(arr_rgb: np.ndarray, max_side: int) -> np.ndarray:
    h, w = arr_rgb.shape[:2]
    scale = min(max_side / max(h, w), 1.0)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    if (new_h, new_w) == (h, w):
        return arr_rgb
    return np.asarray(Image.fromarray(arr_rgb).resize((new_w, new_h), Image.BILINEAR))


def add_title(panel: np.ndarray, title: str, title_h: int = TITLE_HEIGHT) -> np.ndarray:
    h, w = panel.shape[:2]
    canvas = Image.new("RGB", (w, h + title_h), (255, 255, 255))
    canvas.paste(Image.fromarray(panel), (0, title_h))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    draw.text((6, 8), title, fill=(0, 0, 0), font=font)
    return np.asarray(canvas)


def pad_to_height(panel: np.ndarray, height: int) -> np.ndarray:
    h, w = panel.shape[:2]
    if h == height:
        return panel
    out = np.full((height, w, 3), 255, dtype=np.uint8)
    out[:h, :, :] = panel
    return out


def save_compare_row(panels: List[Tuple[str, np.ndarray]], out_path: str) -> None:
    titled = []
    for title, rgb in panels:
        p = fit_panel(rgb, PANEL_MAX_SIDE)
        titled.append(add_title(p, title))
    max_h = max(x.shape[0] for x in titled)
    titled = [pad_to_height(x, max_h) for x in titled]
    gaps = [np.full((max_h, GAP, 3), 255, dtype=np.uint8) for _ in range(max(0, len(titled) - 1))]
    row = []
    for i, p in enumerate(titled):
        row.append(p)
        if i < len(gaps):
            row.append(gaps[i])
    canvas = np.concatenate(row, axis=1)
    Image.fromarray(canvas).save(out_path)


def save_gray(arr_hw: np.ndarray, path: str) -> None:
    g = (np.clip(arr_hw, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    Image.fromarray(g).save(path)

# ============================================================
# 4. Checkpoint helpers
# ============================================================

def resolve_checkpoint_path(cfg: Dict[str, Any]) -> str:
    candidates = []
    if "ckpt" in cfg:
        candidates.append(cfg["ckpt"])
    candidates.extend(cfg.get("ckpt_candidates", []))
    seen = set()
    candidates = [x for x in candidates if not (x in seen or seen.add(x))]
    for x in candidates:
        if x and os.path.exists(x):
            return x
    return cfg.get("ckpt", candidates[0] if candidates else "")


def load_torch(path: str, device: torch.device) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return torch.load(path, map_location=device, weights_only=False)


def clean_state_dict(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        nk = k
        for prefix in ["module.", "model.", "student."]:
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        out[nk] = v
    return out


def extract_state_dict(ckpt: Any) -> Dict[str, torch.Tensor]:
    if not isinstance(ckpt, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")
    for key in [
        "model_state_dict", "state_dict", "model", "net", "ema_model",
        "student", "teacher", "model_student", "model_teacher",
    ]:
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    if len(ckpt) > 0 and all(torch.is_tensor(v) for v in ckpt.values()):
        return ckpt
    raise KeyError(f"Could not find state_dict in checkpoint. keys={list(ckpt.keys())}")


def safe_load_model(model: nn.Module, ckpt: Any, strict: bool = False) -> None:
    sd = clean_state_dict(extract_state_dict(ckpt))
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    if missing or unexpected:
        print(f"[Load] {model.__class__.__name__}: missing={len(missing)} unexpected={len(unexpected)} strict={strict}")

# ============================================================
# 5. Ablation model wrappers
# ============================================================

class IdentityGate(nn.Module):
    def forward(self, x: torch.Tensor, rho: Optional[torch.Tensor] = None, bfreq: Optional[torch.Tensor] = None) -> torch.Tensor:
        return x


class NoWGDCBlock(nn.Module):
    """Replace WDTFBlock by wavelet-prior analysis + ordinary residual DoubleConv.

    It still returns aux rho/bfreq so that FG can be tested independently.
    """
    def __init__(self, channels: int, num_templates: int = 6):
        super().__init__()
        self.prior = WaveletPriorAnalyzer(channels, num_templates=num_templates)
        self.refine = DoubleConv(channels, channels)
        self.res_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor):
        prior = self.prior(x)
        y = self.refine(x)
        y = x + self.res_scale * y
        aux = {
            "alpha": prior["alpha"],
            "theta": prior["theta"],
            "rho": prior["rho"],
            "scale": prior["scale"],
            "bfreq": prior["bfreq"],
        }
        return y, aux


class AblationWDTFWrapper(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        net_cfg = WDTFNetConfig(
            in_channels=IN_CHANNELS,
            num_classes=1,
            base_channels=BASE_CHANNELS,
            use_skeleton_head=False,
            max_residual_offset=MAX_RESIDUAL_OFFSET,
            adapter_reduction=ADAPTER_REDUCTION,
            num_templates=NUM_TEMPLATES,
        )
        self.model = WDTFNetOptimized(net_cfg)
        self.use_wgdc = bool(cfg.get("use_wgdc", True))
        self.use_fg = bool(cfg.get("use_fg", True))
        self.use_wa = bool(cfg.get("use_wa", False))

        c = BASE_CHANNELS
        if not self.use_wgdc:
            self.model.wdtf2 = NoWGDCBlock(c * 2, num_templates=NUM_TEMPLATES)
            self.model.wdtf3 = NoWGDCBlock(c * 4, num_templates=NUM_TEMPLATES)
            self.model.wdtf4 = NoWGDCBlock(c * 8, num_templates=NUM_TEMPLATES)
        if not self.use_fg:
            self.model.dec_gate3 = IdentityGate()
            self.model.dec_gate2 = IdentityGate()
            self.model.dec_gate1 = IdentityGate()

    def forward(self, x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        out = self.model(x, use_adapters=self.use_wa)
        logits = out["logits"] if isinstance(out, dict) and "logits" in out else out
        logits = resize_logits_to(logits, target_hw)
        return torch.sigmoid(logits)


def build_ablation_model(cfg: Dict[str, Any], device: torch.device) -> nn.Module:
    ckpt_path = resolve_checkpoint_path(cfg)
    ckpt = load_torch(ckpt_path, device)
    wrapper = AblationWDTFWrapper(cfg)
    safe_load_model(wrapper.model, ckpt, strict=False)
    wrapper.to(device)
    wrapper.eval()
    return wrapper

# ============================================================
# 6. Main test process
# ============================================================

def main() -> None:
    device = torch.device(DEVICE)
    ensure_dir(OUTPUT_DIR)
    vis_dir = os.path.join(OUTPUT_DIR, "compare_png")
    prob_dir = os.path.join(OUTPUT_DIR, "prob")
    bin_dir = os.path.join(OUTPUT_DIR, "bin")
    ensure_dir(vis_dir)
    if SAVE_PROB:
        ensure_dir(prob_dir)
    if SAVE_BIN:
        ensure_dir(bin_dir)

    samples = find_test_samples(TEST_ROOT)
    variant_cfgs = [cfg for cfg in ABLATION_CONFIGS if cfg.get("enabled", False)]

    print("=" * 80)
    print(f"test_root   = {TEST_ROOT}")
    print(f"output_dir  = {OUTPUT_DIR}")
    print(f"device      = {device}")
    print(f"in_channels = {IN_CHANNELS}")
    print(f"hr_divisor  = {HR_DIVISOR}")
    print(f"samples     = {len(samples)}")
    print("variants    = " + ", ".join([v.get("display", v["name"]) for v in variant_cfgs]))
    print("=" * 80)

    models: Dict[str, nn.Module] = {}
    active_cfgs: List[Dict[str, Any]] = []
    for cfg in variant_cfgs:
        name = cfg["name"]
        display = cfg.get("display", name)
        try:
            ckpt_path = resolve_checkpoint_path(cfg)
            print(f"[Build] {display}: {ckpt_path}")
            models[name] = build_ablation_model(cfg, device)
            active_cfgs.append(cfg)
        except Exception as e:
            print(f"[Skip] {display}: {repr(e)}")

    if not active_cfgs:
        raise RuntimeError("No ablation variant was successfully loaded. Check checkpoint paths and wdtf_net.py.")

    rows: List[Dict[str, Any]] = []
    global_counts = {cfg["name"]: {"TP": 0, "TN": 0, "FP": 0, "FN": 0} for cfg in active_cfgs}

    with torch.no_grad():
        for idx, sample in enumerate(samples, 1):
            sid = sample["id"]
            img_np_raw = read_raster(sample["hr_path"])
            label_np_raw = read_raster(sample["label_path"])

            img_np = normalize_image(img_np_raw, HR_DIVISOR)
            label_np = binarize_label(label_np_raw)
            gt = label_np[0] if label_np.ndim == 3 else label_np
            h, w = gt.shape[-2:]

            if img_np.shape[0] != IN_CHANNELS:
                print(f"[Warn] {sid}: image channels={img_np.shape[0]}, expected {IN_CHANNELS}")

            img_t = torch.from_numpy(img_np).unsqueeze(0).to(device)
            panels: List[Tuple[str, np.ndarray]] = [
                ("HR", chw_to_rgb_uint8(img_np)),
                ("Label", gray_to_rgb_uint8(gt)),
            ]

            print(f"[{idx:03d}/{len(samples):03d}] {sid}")
            for cfg in active_cfgs:
                name = cfg["name"]
                display = cfg.get("display", name)
                model = models[name]
                prob_t = model(img_t, (h, w))
                prob = prob_t[0, 0].detach().cpu().numpy()
                pred = (prob >= THRESHOLD).astype(np.float32)

                m = compute_binary_metrics(pred, gt)
                rows.append({"id": sid, "method": name, "display": display, "variant_id": cfg.get("id", -1), **m})
                for k in ["TP", "TN", "FP", "FN"]:
                    global_counts[name][k] += int(m[k])

                panels.append((display, gray_to_rgb_uint8(prob)))

                if SAVE_PROB:
                    out_d = os.path.join(prob_dir, name)
                    ensure_dir(out_d)
                    save_gray(prob, os.path.join(out_d, f"{sid}_prob.png"))
                if SAVE_BIN:
                    out_d = os.path.join(bin_dir, name)
                    ensure_dir(out_d)
                    save_gray(pred, os.path.join(out_d, f"{sid}_pred.png"))

                print(f"    {display:<14s} {format_metrics(m)}")

            save_compare_row(panels, os.path.join(vis_dir, f"{sid}_ablation.png"))

    # Save per-sample metrics.
    per_sample_csv = os.path.join(OUTPUT_DIR, "metrics_per_sample.csv")
    with open(per_sample_csv, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["id", "variant_id", "method", "display", "IoU", "Dice", "F1", "Acc", "Recall", "Precision", "TP", "TN", "FP", "FN"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Macro metrics and global pixel-level metrics.
    summary_rows: List[Dict[str, Any]] = []
    for cfg in active_cfgs:
        name = cfg["name"]
        display = cfg.get("display", name)
        ms = [r for r in rows if r["method"] == name]
        macro = {"m" + k: float(np.mean([r[k] for r in ms])) if ms else 0.0
                 for k in ["IoU", "Dice", "F1", "Acc", "Recall", "Precision"]}
        c = global_counts[name]
        global_m = metrics_from_counts(c["TP"], c["TN"], c["FP"], c["FN"], prefix="g")
        summary_rows.append({
            "variant_id": cfg.get("id", -1),
            "method": name,
            "display": display,
            "WGDC": int(bool(cfg.get("use_wgdc", False))),
            "FG": int(bool(cfg.get("use_fg", False))),
            "WA": int(bool(cfg.get("use_wa", False))),
            **macro,
            **global_m,
            "TP": c["TP"], "TN": c["TN"], "FP": c["FP"], "FN": c["FN"],
        })
    summary_rows = sorted(summary_rows, key=lambda x: x["variant_id"])

    summary_csv = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "variant_id", "method", "display", "WGDC", "FG", "WA",
            "mIoU", "mDice", "mF1", "mAcc", "mRecall", "mPrecision",
            "gIoU", "gDice", "gF1", "gAcc", "gRecall", "gPrecision",
            "TP", "TN", "FP", "FN",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow(r)

    print("\n" + "=" * 80)
    print("Summary: macro metrics over samples")
    for r in summary_rows:
        print(f"{r['display']:<14s} | IoU={r['mIoU']:.4f} Dice={r['mDice']:.4f} "
              f"F1={r['mF1']:.4f} Acc={r['mAcc']:.4f} Recall={r['mRecall']:.4f} Precision={r['mPrecision']:.4f}")
    print("=" * 80)
    print(f"Saved per-sample metrics: {per_sample_csv}")
    print(f"Saved summary metrics   : {summary_csv}")
    print(f"Saved visualizations    : {vis_dir}")


if __name__ == "__main__":
    main()
