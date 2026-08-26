# -*- coding: utf-8 -*-
"""
WorldView 4-channel comparison test script for 11 HR water extraction methods.

Output for each sample:
    HR | Label | Mask | WDTFNet | U2PL | UniMatch V2 | RankMatch | CPS | TorchSemiSeg | MPF | ReCo | AGMM-SASS | CC4S | WSSS-PCRE

Dataset structure:
    ./data/worldview/test/
        hr/     xxx_hr.tif
        label/  xxx_label.tif
        mask/   xxx_mask.tif    # optional weak/coarse mask for comparison

Label rule:
    label > 0 => water

"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import csv
import json
import sys
import warnings
import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = REPO_ROOT / "benchmarks"
for _path in (REPO_ROOT, BASELINE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torch.nn.functional as F

import rasterio
from rasterio.errors import NotGeoreferencedWarning
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

# ============================================================
# 1. CONFIG: directly edit here
# ============================================================

TEST_ROOT = "./data/worldview/test"     # must contain hr/ and label/
OUTPUT_DIR = r"./experiment_worldview/test_compare"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IN_CHANNELS = 4
HR_DIVISOR = 2048.0
THRESHOLD = 0.5
SAVE_PROB = False
SAVE_BIN = True

# Visualization settings. Metrics use original size; only panels are resized for display.
PANEL_MAX_SIDE = 512
TITLE_HEIGHT = 34
GAP = 6

# Optional RGB band order for display only.
# If your 4-band WorldView order is B,G,R,NIR, set RGB_BANDS = (2,1,0).
# If your order is R,G,B,NIR, keep (0,1,2).
RGB_BANDS = (2, 1, 0)

# ------------------------------------------------------------
# Method selection
# ------------------------------------------------------------
# enabled=True/False controls whether the method participates in comparison.
# ckpt paths below follow the save_dir defaults in the uploaded training scripts.
# Modify paths if your checkpoints are in other folders.

WDTF_METHOD = dict(
    name="wdtf",
    display="WDTFNet",
    arch="wdtf",
    enabled=True,
    # The first existing path in ckpt_candidates will be used.
    # Keep ckpt as the main path; edit it if your checkpoint is elsewhere.
    ckpt=r"./experiment_worldview/ablation_wdtfnet/07_full_wdtfnet/stage2_best.pth",
    ckpt_candidates=[
    ],
    use_adapters=True,
    # These must match WDTFNetConfig used in training. Your train_wdtfnet.py uses base_channels=64 by default.
    base_channels=64,
    adapter_reduction=4,
    max_residual_offset=0.35,
    num_templates=6,
)

METHOD_CONFIGS = [
    dict(name="u2pl", display="U2PL", arch="common_single", enabled=True,
         module="train_u2pl", ckpt=r"./experiment_worldview/u2pl/best.pth", model_key="student"),
    dict(name="unimatch_v2", display="UniMatch V2", arch="common_single", enabled=True,
         module="train_unimatch_v2", ckpt=r"./experiment_worldview/unimatch_v2/best.pth", model_key="student"),
    dict(name="rankmatch", display="RankMatch", arch="common_single", enabled=True,
         module="train_rankmatch", ckpt=r"./experiment_worldview/rankmatch/best.pth", model_key="student"),

    dict(name="cps", display="CPS", arch="common_dual_avg", enabled=False,
         module="train_cps", ckpt=r"./experiment_worldview/cps/best.pth", model_keys=("student1", "student2")),
    dict(name="torchsemiseg", display="TorchSemiSeg", arch="common_dual_avg", enabled=True,
         module="train_torchsemiseg", ckpt=r"./experiment_worldview/torchsemiseg/best.pth", model_keys=("student1", "student2")),

    dict(name="mpf", display="MPF", arch="mpf", enabled=True,
         ckpt=r"./experiment_worldview/mpf_weak/best.pth", in_channels=IN_CHANNELS),
    dict(name="reco", display="ReCo", arch="reco", enabled=True,
         ckpt=r"./experiment_worldview/reco_weak/best.pth", in_channels=IN_CHANNELS,
         output_dim=128, pretrained_backbone=False),
    dict(name="agmm_sass", display="AGMM-SASS", arch="agmm", enabled=True,
         ckpt=r"./experiment_worldview/agmm_sass_weak/agmm_sass_weak_best.pth", in_channels=IN_CHANNELS, backbone="resnet50"),
    dict(name="cc4s", display="CC4S", arch="cc4s", enabled=True,
         ckpt=r"./experiment_worldview/cc4s_weak/stage2_best.pth", in_channels=IN_CHANNELS, layers=18, shrink_factor=2),
    dict(name="wsss_pcre", display="WSSS-PCRE", arch="wsss_pcre", enabled=True,
         ckpt=r"./experiment_worldview/wsss_pcre_weak/best.pth", in_channels=IN_CHANNELS, feat_dim=128),
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


def binarize_prob_like(arr: np.ndarray, threshold: float = THRESHOLD) -> np.ndarray:
    """Binarize prediction/weak-mask arrays with a threshold.

    If the input looks like an integer mask with values larger than 1, it is first
    normalized by its maximum value, so 0/255, 0/100 and 0/1 probability-style
    masks can all use the same THRESHOLD.
    """
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    max_v = float(arr.max()) if arr.size else 0.0
    if max_v > 1.0:
        arr = arr / max_v
    return (arr >= threshold).astype(np.float32)


def find_test_samples(test_root: str) -> List[Dict[str, str]]:
    hr_root = Path(test_root) / "hr"
    label_root = Path(test_root) / "label"
    mask_root = Path(test_root) / "mask"
    if not hr_root.exists():
        raise FileNotFoundError(f"hr folder not found: {hr_root}")
    if not label_root.exists():
        raise FileNotFoundError(f"label folder not found: {label_root}")
    if not mask_root.exists():
        raise FileNotFoundError(f"mask folder not found: {mask_root}")

    hr_map = {strip_suffix(p.stem, "hr"): str(p) for p in hr_root.iterdir() if p.is_file()}
    label_map = {strip_suffix(p.stem, "label"): str(p) for p in label_root.iterdir() if p.is_file()}
    mask_map = {strip_suffix(p.stem, "mask"): str(p) for p in mask_root.iterdir() if p.is_file()}

    common = sorted(set(hr_map) & set(label_map) & set(mask_map))
    if not common:
        raise RuntimeError("No matched samples found. Expected xxx_hr.tif, xxx_label.tif and xxx_mask.tif")
    return [
        {"id": sid, "hr_path": hr_map[sid], "label_path": label_map[sid], "mask_path": mask_map[sid]}
        for sid in common
    ]


def resize_logits_to(logits: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
    if logits.shape[-2:] != size_hw:
        logits = F.interpolate(logits, size=size_hw, mode="bilinear", align_corners=False)
    return logits


def as_prob_from_output(out: Any, target_hw: Tuple[int, int]) -> torch.Tensor:
    """Convert common binary model outputs to [B,1,H,W] probability."""
    if isinstance(out, dict):
        if "logits" in out:
            logits = out["logits"]
        elif "fused" in out:
            logits = out["fused"]
        elif "out" in out:
            logits = out["out"]
        else:
            first_key = list(out.keys())[0]
            logits = out[first_key]
    elif isinstance(out, (tuple, list)):
        candidates = [x for x in out if torch.is_tensor(x) and x.ndim == 4]
        if not candidates:
            raise RuntimeError("No 4D tensor found in model output tuple/list.")
        logits = candidates[-1]
        for x in candidates:
            if x.shape[1] in (1, 2):
                logits = x
                break
    else:
        logits = out

    if logits.ndim != 4:
        raise RuntimeError(f"Expected logits [B,C,H,W], got {tuple(logits.shape)}")
    logits = resize_logits_to(logits, target_hw)
    if logits.shape[1] == 1:
        return torch.sigmoid(logits)
    if logits.shape[1] == 2:
        return torch.softmax(logits, dim=1)[:, 1:2]
    raise RuntimeError(f"Only binary logits with C=1 or C=2 are supported, got C={logits.shape[1]}")


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
    """Use cfg['ckpt'] or the first existing path in cfg['ckpt_candidates']."""
    candidates = []
    if "ckpt" in cfg:
        candidates.append(cfg["ckpt"])
    candidates.extend(cfg.get("ckpt_candidates", []))
    seen = set()
    candidates = [x for x in candidates if not (x in seen or seen.add(x))]
    for x in candidates:
        if x and os.path.exists(x):
            return x
    # return the main path so load_torch gives a clear FileNotFoundError
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


def extract_state_dict(ckpt: Any, model_key: Optional[str] = None) -> Dict[str, torch.Tensor]:
    if not isinstance(ckpt, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")

    if model_key is not None:
        candidates = [model_key]
        alias_map = {
            "student": ["student", "model_student", "model"],
            "teacher": ["teacher", "model_teacher"],
            "student1": ["student1", "model_student1"],
            "student2": ["student2", "model_student2"],
        }
        candidates += alias_map.get(model_key, [])
        if not str(model_key).startswith("model_"):
            candidates.append("model_" + str(model_key))
        candidates = list(dict.fromkeys(candidates))

        if "models" in ckpt and isinstance(ckpt["models"], dict):
            for k in candidates:
                if k in ckpt["models"] and isinstance(ckpt["models"][k], dict):
                    return ckpt["models"][k]
        for k in candidates:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]

    for key in [
        "model_state_dict", "state_dict", "model", "net", "ema_model",
        "student", "teacher", "model_student", "model_teacher",
        "student1", "student2", "model_student1", "model_student2",
    ]:
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]

    if len(ckpt) > 0 and all(torch.is_tensor(v) for v in ckpt.values()):
        return ckpt

    raise KeyError(f"Could not find state_dict in checkpoint. keys={list(ckpt.keys())}")


def safe_load_model(model: nn.Module, ckpt: Any, model_key: Optional[str] = None, strict: bool = False) -> None:
    sd = clean_state_dict(extract_state_dict(ckpt, model_key=model_key))
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    if missing or unexpected:
        print(f"[Load] {model.__class__.__name__}: missing={len(missing)} unexpected={len(unexpected)} strict={strict}")

# ============================================================
# 5. Model wrappers
# ============================================================

class GenericWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        return as_prob_from_output(self.model(x), target_hw)


class DualAverageWrapper(nn.Module):
    def __init__(self, model1: nn.Module, model2: nn.Module):
        super().__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        p1 = as_prob_from_output(self.model1(x), target_hw)
        p2 = as_prob_from_output(self.model2(x), target_hw)
        return 0.5 * (p1 + p2)


class WDTFWrapper(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        # Build directly from your uploaded wdtf_net.py. This avoids importing training-only
        # files such as stage2_loss_simplified.py during testing.
        from wdtf_net import WDTFNetConfig, WDTFNetOptimized
        net_cfg = WDTFNetConfig(
            in_channels=IN_CHANNELS,
            num_classes=1,
            base_channels=int(cfg.get("base_channels", 64)),
            use_skeleton_head=False,
            max_residual_offset=float(cfg.get("max_residual_offset", 0.35)),
            adapter_reduction=int(cfg.get("adapter_reduction", 4)),
            num_templates=int(cfg.get("num_templates", 6)),
        )
        self.model = WDTFNetOptimized(net_cfg)
        self.use_adapters = bool(cfg.get("use_adapters", True))

    def forward(self, x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        out = self.model(x, use_adapters=self.use_adapters)
        logits = out["logits"] if isinstance(out, dict) and "logits" in out else out
        logits = resize_logits_to(logits, target_hw)
        return torch.sigmoid(logits)


def build_common_models_from_train_module(module_name: str, device: torch.device, checkpoint: Optional[str] = None):
    mod = importlib.import_module(module_name)
    cfg = mod.WeakTrainConfig() if hasattr(mod, "WeakTrainConfig") else mod.cfg
    if checkpoint:
        cfg_path = Path(checkpoint).parent / "config.json"
        if cfg_path.exists():
            try:
                saved = json.loads(cfg_path.read_text(encoding="utf-8"))
                for key, value in saved.items():
                    if hasattr(cfg, key):
                        setattr(cfg, key, value)
            except Exception as exc:
                print(f"[Warn] could not apply saved config {cfg_path}: {exc}")
    cfg.device = str(device)
    cfg.backbone = getattr(cfg, "backbone", "resnet50")
    cfg.pretrained_backbone = False
    cfg.num_classes = 1
    cfg.hr_divisor = HR_DIVISOR
    cfg.root_dir = TEST_ROOT
    models, _ = mod.build_models(IN_CHANNELS, cfg)
    return models



class CommonSingleWrapper(nn.Module):
    def __init__(self, cfg: Dict[str, Any], ckpt: Any, device: torch.device):
        super().__init__()
        # ``ckpt`` is the already-loaded checkpoint dictionary.  Pass the
        # configured checkpoint *path* to recover its adjacent config.json
        # (notably the Sentinel U2PL ResNet-101 setting) before construction.
        models = build_common_models_from_train_module(cfg["module"], device, str(cfg.get("ckpt", "")))
        key = cfg.get("model_key", "student")
        if key not in models:
            key = "student" if "student" in models else list(models.keys())[0]
        self.model = models[key]
        safe_load_model(self.model, ckpt, model_key=key, strict=False)

    def forward(self, x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        return as_prob_from_output(self.model(x), target_hw)


class CommonDualAvgWrapper(nn.Module):
    def __init__(self, cfg: Dict[str, Any], ckpt: Any, device: torch.device):
        super().__init__()
        # Keep the two deployed branches consistent with the checkpoint's
        # saved architecture/configuration, just as for the single-model path.
        models = build_common_models_from_train_module(cfg["module"], device, str(cfg.get("ckpt", "")))
        k1, k2 = cfg.get("model_keys", ("student1", "student2"))
        self.model1 = models[k1]
        self.model2 = models[k2]
        safe_load_model(self.model1, ckpt, model_key=k1, strict=False)
        safe_load_model(self.model2, ckpt, model_key=k2, strict=False)

    def forward(self, x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        p1 = as_prob_from_output(self.model1(x), target_hw)
        p2 = as_prob_from_output(self.model2(x), target_hw)
        return 0.5 * (p1 + p2)


class MPFWrapper(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        mod = importlib.import_module("train_mpf_weak")
        self.model = mod.MPFWeakNet(in_channels=IN_CHANNELS, use_imagenet=False)

    def forward(self, x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        out = self.model(x)
        return as_prob_from_output(out, target_hw)


class ReCoWrapper(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        mod = importlib.import_module("train_reco_weak")
        self.model = mod.build_model(IN_CHANNELS, int(cfg.get("output_dim", 128)), bool(cfg.get("pretrained_backbone", False)))

    def forward(self, x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, (tuple, list)):
            logits = out[0]
        else:
            logits = out
        logits = resize_logits_to(logits, target_hw)
        if logits.shape[1] == 2:
            return torch.softmax(logits, dim=1)[:, 1:2]
        return torch.sigmoid(logits)


class AGMMWrapper(nn.Module):
    def __init__(self, cfg_user: Dict[str, Any]):
        super().__init__()
        mod = importlib.import_module("train_agmm_sass_weak")
        if hasattr(mod, "patch_backbone_pretrain_fallback"):
            mod.patch_backbone_pretrain_fallback()
        cfg = {
            "nclass": 2,
            "aux": False,
            "backbone": cfg_user.get("backbone", "resnet50"),
            "multi_grid": False,
            "replace_stride_with_dilation": [False, True, True],
            "dilations": [6, 12, 18],
        }
        self.model = mod.DeepLabV3Plus(cfg, aux=False)
        if hasattr(mod, "patch_backbone_input_channels"):
            mod.patch_backbone_input_channels(self.model, IN_CHANNELS)

    def forward(self, x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        return as_prob_from_output(self.model(x), target_hw)


class CC4SWrapper(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        mod = importlib.import_module("train_cc4s_weak")
        self.model = mod.BinaryCC4SModel(
            layers=int(cfg.get("layers", 18)),
            shrink_factor=int(cfg.get("shrink_factor", 2)),
            in_channels=IN_CHANNELS,
        )

    def forward(self, x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        out = self.model(x)
        return as_prob_from_output(out, target_hw)


class PCREWrapper(nn.Module):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        mod = importlib.import_module("train_wsss_pcre_weak")
        self.model = mod.PCREWeakNet(in_channels=IN_CHANNELS, feat_dim=int(cfg.get("feat_dim", 128)))

    def forward(self, x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        return as_prob_from_output(self.model(x), target_hw)


def build_compare_model(cfg: Dict[str, Any], device: torch.device) -> nn.Module:
    arch = cfg["arch"]
    ckpt_path = resolve_checkpoint_path(cfg)
    ckpt = load_torch(ckpt_path, device)

    if arch == "wdtf":
        wrapper = WDTFWrapper(cfg)
        safe_load_model(wrapper.model, ckpt, strict=False)
    elif arch == "common_single":
        wrapper = CommonSingleWrapper(cfg, ckpt, device)
    elif arch == "common_dual_avg":
        wrapper = CommonDualAvgWrapper(cfg, ckpt, device)
    elif arch == "mpf":
        wrapper = MPFWrapper(cfg)
        safe_load_model(wrapper.model, ckpt, model_key="model", strict=False)
    elif arch == "reco":
        wrapper = ReCoWrapper(cfg)
        safe_load_model(wrapper.model, ckpt, model_key="model", strict=False)
    elif arch == "agmm":
        wrapper = AGMMWrapper(cfg)
        safe_load_model(wrapper.model, ckpt, model_key="model", strict=False)
    elif arch == "cc4s":
        wrapper = CC4SWrapper(cfg)
        safe_load_model(wrapper.model, ckpt, model_key="model", strict=False)
    elif arch == "wsss_pcre":
        wrapper = PCREWrapper(cfg)
        safe_load_model(wrapper.model, ckpt, model_key="model", strict=False)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    wrapper.to(device)
    wrapper.eval()
    return wrapper

# ============================================================
# 6. Main test process
# ============================================================

def main() -> None:
    here = Path(__file__).resolve().parent
    cwd = Path.cwd()
    for p in [str(here), str(cwd)]:
        if p not in sys.path:
            sys.path.insert(0, p)

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
    method_cfgs = [dict(name="mask", display="Mask", arch="mask", enabled=True), dict(WDTF_METHOD, enabled=True)] + [m for m in METHOD_CONFIGS if m.get("enabled", False)]

    print("=" * 80)
    print(f"test_root   = {TEST_ROOT}")
    print(f"output_dir  = {OUTPUT_DIR}")
    print(f"device      = {device}")
    print(f"in_channels = {IN_CHANNELS}")
    print(f"hr_divisor  = {HR_DIVISOR}")
    print(f"samples     = {len(samples)}")
    print("methods     = " + ", ".join([m.get("display", m["name"]) for m in method_cfgs]))
    print("=" * 80)

    models: Dict[str, nn.Module] = {}
    active_cfgs: List[Dict[str, Any]] = []
    for cfg in method_cfgs:
        name = cfg["name"]
        display = cfg.get("display", name)
        if name == "mask":
            print(f"[Build] {display}: use test_root/mask/xxx_mask.tif directly")
            active_cfgs.append(cfg)
            continue
        try:
            ckpt_path = resolve_checkpoint_path(cfg)
            print(f"[Build] {display}: {ckpt_path}")
            models[name] = build_compare_model(cfg, device)
            active_cfgs.append(cfg)
        except Exception as e:
            print(f"[Skip] {display}: {repr(e)}")

    if not active_cfgs:
        raise RuntimeError("No method was successfully loaded. Check checkpoint paths and model files.")

    rows = []
    global_counts = {cfg["name"]: {"TP": 0, "TN": 0, "FP": 0, "FN": 0} for cfg in active_cfgs}

    with torch.no_grad():
        for idx, sample in enumerate(samples, 1):
            sid = sample["id"]
            img_np_raw = read_raster(sample["hr_path"])
            label_np_raw = read_raster(sample["label_path"])
            mask_np_raw = read_raster(sample["mask_path"])

            img_np = normalize_image(img_np_raw, HR_DIVISOR)
            label_np = binarize_label(label_np_raw)
            gt = label_np[0] if label_np.ndim == 3 else label_np
            h, w = gt.shape[-2:]

            mask_np = mask_np_raw[0] if mask_np_raw.ndim == 3 else mask_np_raw
            if mask_np.shape[-2:] != (h, w):
                mask_t = torch.from_numpy(mask_np[None, None].astype(np.float32))
                mask_np = F.interpolate(mask_t, size=(h, w), mode="nearest")[0, 0].numpy()
            mask_bin = binarize_prob_like(mask_np, THRESHOLD)

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

                if name == "mask":
                    prob = mask_bin.copy()
                    pred = mask_bin.copy()
                else:
                    model = models[name]
                    prob_t = model(img_t, (h, w))
                    prob = prob_t[0, 0].detach().cpu().numpy()
                    pred = (prob >= THRESHOLD).astype(np.float32)

                m = compute_binary_metrics(pred, gt)
                rows.append({"id": sid, "method": name, "display": display, **m})
                for k in ["TP", "TN", "FP", "FN"]:
                    global_counts[name][k] += int(m[k])

                # Visualization uses binary maps after thresholding, not probability maps.
                panels.append((display, gray_to_rgb_uint8(pred)))

                if SAVE_PROB and name != "mask":
                    out_d = os.path.join(prob_dir, name)
                    ensure_dir(out_d)
                    save_gray(prob, os.path.join(out_d, f"{sid}_prob.png"))
                if SAVE_BIN:
                    out_d = os.path.join(bin_dir, name)
                    ensure_dir(out_d)
                    save_gray(pred, os.path.join(out_d, f"{sid}_pred.png"))

                print(f"    {display:<16s} {format_metrics(m)}")

            save_compare_row(panels, os.path.join(vis_dir, f"{sid}_compare.png"))

    # Save per-sample metrics.
    per_sample_csv = os.path.join(OUTPUT_DIR, "metrics_per_sample.csv")
    with open(per_sample_csv, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = ["id", "method", "display", "IoU", "Dice", "F1", "Acc", "Recall", "Precision", "TP", "TN", "FP", "FN"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    # Macro metrics and global pixel-level metrics.
    summary_rows = []
    for cfg in active_cfgs:
        name = cfg["name"]
        display = cfg.get("display", name)
        ms = [r for r in rows if r["method"] == name]
        macro = {"m" + k: float(np.mean([r[k] for r in ms])) if ms else 0.0
                 for k in ["IoU", "Dice", "F1", "Acc", "Recall", "Precision"]}
        c = global_counts[name]
        global_m = metrics_from_counts(c["TP"], c["TN"], c["FP"], c["FN"], prefix="g")
        summary_rows.append({
            "method": name,
            "display": display,
            **macro,
            **global_m,
            "TP": c["TP"], "TN": c["TN"], "FP": c["FP"], "FN": c["FN"],
        })

    summary_csv = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        fieldnames = [
            "method", "display",
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
        print(f"{r['display']:<16s} | IoU={r['mIoU']:.4f} Dice={r['mDice']:.4f} "
              f"F1={r['mF1']:.4f} Acc={r['mAcc']:.4f} Recall={r['mRecall']:.4f} Precision={r['mPrecision']:.4f}")
    print("=" * 80)
    print(f"Saved per-sample metrics: {per_sample_csv}")
    print(f"Saved summary metrics   : {summary_csv}")
    print(f"Saved visualizations    : {vis_dir}")


if __name__ == "__main__":
    main()
