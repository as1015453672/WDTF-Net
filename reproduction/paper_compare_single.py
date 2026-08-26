"""Run one checkpoint through the common full-resolution comparison evaluator.

The batch runner invokes this script one method at a time so that a 12-GB GPU
never has all paper-table models resident simultaneously.  The only writes are
under ``experiment_paper_style_comparison_20260821``.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
BASELINE_ROOT = REPO_ROOT / "benchmarks"
for _path in (REPO_ROOT, BASELINE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import test_compare_worldview as cmp
from paper_comparison_inventory import METHODS, OUT, ROOT


DATA_ROOT = Path(os.environ.get("WDTF_DATA_ROOT", REPO_ROOT / "data"))
DATA = {
    "GF": {"test_root": str(DATA_ROOT / "gf" / "test"), "channels": 3, "divisor": 1024.0, "rgb": (0, 1, 2)},
    "Sentinel": {"test_root": str(DATA_ROOT / "sentinel" / "test"), "channels": 4, "divisor": 4096.0, "rgb": (0, 1, 2)},
    "WorldView": {"test_root": str(DATA_ROOT / "worldview" / "test"), "channels": 4, "divisor": 2048.0, "rgb": (2, 1, 0)},
}


class ParaformerWrapper(nn.Module):
    def __init__(self, checkpoint: dict[str, Any], channels: int):
        super().__init__()
        from models.paraformer import Paraformer
        args = checkpoint.get("args", {})
        self.model = Paraformer(in_channels=channels, num_classes=1, token_pool=int(args.get("token_pool", 16)))
        self.model.load_state_dict(checkpoint["model"], strict=True)

    def forward(self, image: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        _, logits = self.model(image)
        logits = F.interpolate(logits, size=target_hw, mode="bilinear", align_corners=False)
        return torch.sigmoid(logits)


class OfficialUniMatchWrapper(nn.Module):
    def __init__(self, checkpoint_path: str, device: torch.device):
        super().__init__()
        evaluator = importlib.import_module("evaluate_sentinel_phase3_baselines")
        self.model, _, _ = evaluator.load_model("unimatch_v2_official", Path(checkpoint_path), device)

    def forward(self, image: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        source_hw = image.shape[-2:]
        x = image
        if source_hw[0] % 14 or source_hw[1] % 14:
            x = F.interpolate(image, size=(504, 504), mode="bilinear", align_corners=False)
        logits = self.model(x)
        logits = F.interpolate(logits, size=target_hw, mode="bilinear", align_corners=False)
        return torch.softmax(logits, dim=1)[:, 1:2]


class OfficialRankMatchWrapper(nn.Module):
    def __init__(self, checkpoint_path: str, device: torch.device):
        super().__init__()
        evaluator = importlib.import_module("evaluate_sentinel_phase3_baselines")
        self.model, _, _ = evaluator.load_model("rankmatch_official", Path(checkpoint_path), device)

    def forward(self, image: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        logits = self.model(image)[0]
        logits = F.interpolate(logits, size=target_hw, mode="bilinear", align_corners=False)
        return torch.softmax(logits, dim=1)[:, 1:2]


def find_item(dataset: str, method: str) -> dict[str, Any]:
    for item in METHODS:
        if item["dataset"] == dataset and item["method"] == method:
            return dict(item)
    raise KeyError(f"No inventory entry for {dataset}/{method}")


def config_for(item: dict[str, Any]) -> dict[str, Any]:
    method = item["method"]
    checkpoint = str(ROOT / item["checkpoint"]) if item["checkpoint"] else ""
    name = method.lower().replace("-", "_").replace(" ", "_")
    if method == "WDTF-Net":
        return {"name": "wdtfnet", "display": method, "arch": "wdtf", "enabled": True, "ckpt": checkpoint,
                "use_adapters": True, "base_channels": 64, "adapter_reduction": 4,
                "max_residual_offset": 0.35, "num_templates": 6}
    if method == "ParaFormer":
        return {"name": "paraformer", "display": method, "arch": "paraformer", "enabled": True, "ckpt": checkpoint}
    if method in {"U2PL", "UniMatch V2", "RankMatch"}:
        if item["dataset"] == "Sentinel" and method == "UniMatch V2":
            return {"name": "unimatch_v2", "display": method, "arch": "official_unimatch", "enabled": True, "ckpt": checkpoint}
        if item["dataset"] == "Sentinel" and method == "RankMatch":
            return {"name": "rankmatch", "display": method, "arch": "official_rankmatch", "enabled": True, "ckpt": checkpoint}
        module = {"U2PL": "train_u2pl", "UniMatch V2": "train_unimatch_v2", "RankMatch": "train_rankmatch"}[method]
        return {"name": name, "display": method, "arch": "common_single", "enabled": True,
                "module": module, "ckpt": checkpoint, "model_key": "student"}
    if method == "CPS":
        return {"name": "cps", "display": method, "arch": "common_dual_avg", "enabled": True,
                "module": "train_cps", "ckpt": checkpoint, "model_keys": ("student1", "student2")}
    arch = {"MPF": "mpf", "ReCo": "reco", "AGMM-SASS": "agmm", "CC4S": "cc4s", "WSSS-PCRE": "wsss_pcre"}[method]
    cfg: dict[str, Any] = {"name": name, "display": method, "arch": arch, "enabled": True, "ckpt": checkpoint}
    if arch == "reco": cfg.update({"output_dim": 128, "pretrained_backbone": False})
    if arch == "agmm": cfg.update({"backbone": "resnet50"})
    if arch == "cc4s": cfg.update({"layers": 18, "shrink_factor": 2})
    if arch == "wsss_pcre": cfg.update({"feat_dim": 128})
    return cfg


def install_extra_builders() -> None:
    original = cmp.build_compare_model

    def build(cfg: dict[str, Any], device: torch.device) -> nn.Module:
        arch = cfg["arch"]
        path = cmp.resolve_checkpoint_path(cfg)
        if arch == "paraformer":
            checkpoint = cmp.load_torch(path, device)
            wrapper: nn.Module = ParaformerWrapper(checkpoint, cmp.IN_CHANNELS)
        elif arch == "official_unimatch":
            wrapper = OfficialUniMatchWrapper(path, device)
        elif arch == "official_rankmatch":
            wrapper = OfficialRankMatchWrapper(path, device)
        else:
            return original(cfg, device)
        wrapper.to(device).eval()
        return wrapper

    cmp.build_compare_model = build


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATA), required=True)
    parser.add_argument("--method", required=True)
    args = parser.parse_args()
    if args.method == "Mask":
        raise SystemExit("Mask is evaluated implicitly with every model; it has no checkpoint run.")
    item = find_item(args.dataset, args.method)
    spec = DATA[args.dataset]
    cfg = config_for(item)
    out = OUT / "raw" / args.dataset.lower() / cfg["name"]
    status = out / "metrics_summary.csv"
    if status.exists() and f"\n{cfg['name']}," in status.read_text(encoding="utf-8-sig"):
        print(f"already complete: {status}", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_spec.json").write_text(json.dumps({"inventory": item, "config": cfg, "dataset": spec}, ensure_ascii=False, indent=2), encoding="utf-8")
    cmp.TEST_ROOT = spec["test_root"]
    cmp.OUTPUT_DIR = str(out)
    cmp.IN_CHANNELS = spec["channels"]
    cmp.HR_DIVISOR = spec["divisor"]
    cmp.RGB_BANDS = spec["rgb"]
    cmp.SAVE_PROB = False
    cmp.SAVE_BIN = True
    # cmp.main always includes WDTF_METHOD, hence make it a deliberate missing
    # path unless this is the locked original WDTF checkpoint itself.
    cmp.WDTF_METHOD = cfg if cfg["arch"] == "wdtf" else {"name": "wdtf_skipped", "display": "WDTF-Net", "arch": "wdtf", "ckpt": str(out / "not_used.pth")}
    cmp.METHOD_CONFIGS = [] if cfg["arch"] == "wdtf" else [cfg]
    install_extra_builders()
    cmp.main()


if __name__ == "__main__":
    main()
