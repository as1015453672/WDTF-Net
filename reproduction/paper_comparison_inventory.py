"""Create a non-mutating checkpoint/protocol inventory for the paper comparison.

This deliberately does not train, modify checkpoints, or select checkpoints from
test data.  Loading each file on CPU verifies that it is readable and records
only structural metadata.  Runtime model compatibility is verified later by the
same inference wrappers used for the final-label evaluation.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
# Place the released checkpoint tree here (or set WDTF_ARTIFACT_ROOT). The
# repository intentionally does not include large model files.
ROOT = Path(os.environ.get("WDTF_ARTIFACT_ROOT", REPO_ROOT / "artifacts"))
OUT = Path(os.environ.get("WDTF_REPRO_OUTPUT", REPO_ROOT / "runs" / "paper_comparison"))


def entry(dataset: str, method: str, ckpt: str | None, arch: str, *, source: str,
          divisor: float, bands: int, threshold: float = 0.5) -> dict[str, Any]:
    return {
        "dataset": dataset, "method": method, "checkpoint": ckpt,
        "expected_architecture": arch, "source": source,
        "in_channels": bands, "inference_divisor": divisor,
        "threshold": threshold,
    }


METHODS = [
    # GF: the three SSL entries are the post-audit runs, as required.
    entry("GF", "Mask", None, "test weak mask", source="data/gf/test/mask", divisor=1024, bands=3),
    entry("GF", "U2PL", "experiment_cross_dataset_ssl_20260820/gf_reaudit_retrain/u2pl/best.pth", "train_u2pl ResNet101", source="post-audit formal evaluation", divisor=1024, bands=3),
    entry("GF", "UniMatch V2", "experiment_cross_dataset_ssl_20260820/gf_reaudit_retrain/unimatch_v2/best.pth", "train_unimatch_v2 ResNet101", source="post-audit formal evaluation", divisor=1024, bands=3),
    entry("GF", "RankMatch", "experiment_cross_dataset_ssl_20260820/gf_reaudit_retrain/rankmatch/best.pth", "train_rankmatch ResNet101", source="post-audit formal evaluation", divisor=1024, bands=3),
    entry("GF", "CPS", "experiment_gf/cps/best.pth", "train_cps dual ResNet", source="existing checkpoint", divisor=1024, bands=3),
    entry("GF", "MPF", "experiment_gf/mpf_weak/best.pth", "MPFWeakNet", source="existing checkpoint", divisor=1024, bands=3),
    entry("GF", "ReCo", "experiment_gf/reco_weak/best.pth", "DeepLabv2 ReCo", source="existing checkpoint", divisor=1024, bands=3),
    entry("GF", "AGMM-SASS", "experiment_gf/agmm_sass_weak/best.pth", "DeepLabV3Plus ResNet50", source="existing checkpoint", divisor=1024, bands=3),
    entry("GF", "CC4S", "experiment_gf/cc4s_weak/stage1_best.pth", "BinaryCC4SModel", source="existing checkpoint", divisor=1024, bands=3),
    entry("GF", "WSSS-PCRE", "experiment_gf/wsss_pcre_weak/best.pth", "PCREWeakNet", source="existing checkpoint", divisor=1024, bands=3),
    entry("GF", "ParaFormer", "experiment_gf/paraformer/best.pth", "Paraformer", source="user-locked checkpoint", divisor=1024, bands=3),
    entry("GF", "WDTF-Net", "experiment_gf/wdtf/stage2_weak_best.pth", "WDTFNetOptimized", source="user-locked original checkpoint", divisor=1024, bands=3),
    # Sentinel official SSL adaptations are retained as independent checkpoint sources.
    entry("Sentinel", "Mask", None, "test weak mask", source="data/sentinel/test/mask", divisor=4096, bands=4),
    entry("Sentinel", "U2PL", "experiment_sentinel/ssl_baseline_audit_20260819/u2pl_corrected_full_resnet101/best.pth", "official U2PL adaptation", source="official adaptation audit", divisor=4096, bands=4),
    entry("Sentinel", "UniMatch V2", "experiment_sentinel/ssl_baseline_audit_20260819/unimatch_v2_official_dpt_sentinel/best.pth", "official UniMatch V2 adaptation", source="official adaptation audit", divisor=4096, bands=4),
    entry("Sentinel", "RankMatch", "experiment_sentinel/ssl_baseline_audit_20260819/rankmatch_official_sentinel_retry/best.pth", "official RankMatch adaptation", source="official adaptation audit", divisor=4096, bands=4),
    entry("Sentinel", "CPS", "experiment_sentinel/ssl_baseline_audit_20260819/cps_repro/best.pth", "train_cps dual ResNet", source="reproduction audit", divisor=4096, bands=4),
    entry("Sentinel", "MPF", "experiment_sentinel/mpf_weak/best.pth", "MPFWeakNet", source="existing checkpoint", divisor=4096, bands=4),
    entry("Sentinel", "ReCo", "experiment_sentinel/reco_weak/best.pth", "DeepLabv2 ReCo", source="existing checkpoint", divisor=4096, bands=4),
    entry("Sentinel", "AGMM-SASS", "experiment_sentinel/agmm_sass_weak/agmm_sass_weak_best.pth", "DeepLabV3Plus ResNet50", source="existing checkpoint", divisor=4096, bands=4),
    entry("Sentinel", "CC4S", "experiment_sentinel/cc4s_weak/stage2_best.pth", "BinaryCC4SModel", source="existing checkpoint", divisor=4096, bands=4),
    entry("Sentinel", "WSSS-PCRE", "experiment_sentinel/wsss_pcre_weak/best.pth", "PCREWeakNet", source="existing checkpoint", divisor=4096, bands=4),
    entry("Sentinel", "ParaFormer", "experiment_sentinel/paraformer/best.pth", "Paraformer", source="user-locked checkpoint", divisor=4096, bands=4),
    entry("Sentinel", "WDTF-Net", "experiment_sentinel/wdtf/stage2_weak_best.pth", "WDTFNetOptimized", source="user-locked original checkpoint", divisor=4096, bands=4),
    # WorldView uses the completed cross-dataset SSL outputs, not older local copies.
    entry("WorldView", "Mask", None, "test weak mask", source="data/worldview/test/mask", divisor=2048, bands=4),
    entry("WorldView", "U2PL", "experiment_cross_dataset_ssl_20260819/worldview/u2pl/best.pth", "train_u2pl ResNet101", source="cross-dataset formal evaluation", divisor=2048, bands=4),
    entry("WorldView", "UniMatch V2", "experiment_cross_dataset_ssl_20260819/worldview/unimatch_v2/best.pth", "train_unimatch_v2 ResNet101", source="cross-dataset formal evaluation", divisor=2048, bands=4),
    entry("WorldView", "RankMatch", "experiment_cross_dataset_ssl_20260819/worldview/rankmatch/best.pth", "train_rankmatch ResNet101", source="cross-dataset formal evaluation", divisor=2048, bands=4),
    entry("WorldView", "CPS", "experiment_worldview/cps/best.pth", "train_cps dual ResNet", source="existing checkpoint", divisor=2048, bands=4),
    entry("WorldView", "MPF", "experiment_worldview/mpf_weak/best.pth", "MPFWeakNet", source="existing checkpoint", divisor=2048, bands=4),
    entry("WorldView", "ReCo", "experiment_worldview/reco_weak/best.pth", "DeepLabv2 ReCo", source="existing checkpoint", divisor=2048, bands=4),
    entry("WorldView", "AGMM-SASS", "experiment_worldview/agmm_sass_weak/agmm_sass_weak_best.pth", "DeepLabV3Plus ResNet50", source="existing checkpoint", divisor=2048, bands=4),
    entry("WorldView", "CC4S", "experiment_worldview/cc4s_weak/stage2_best.pth", "BinaryCC4SModel", source="existing checkpoint", divisor=2048, bands=4),
    entry("WorldView", "WSSS-PCRE", "experiment_worldview/wsss_pcre_weak/best.pth", "PCREWeakNet", source="existing checkpoint", divisor=2048, bands=4),
    entry("WorldView", "ParaFormer", "experiment_worldview/paraformer/best.pth", "Paraformer", source="user-locked checkpoint", divisor=2048, bands=4),
    entry("WorldView", "WDTF-Net", "experiment_worldview/wdtf/stage2_weak_best.pth", "WDTFNetOptimized", source="user-locked original checkpoint", divisor=2048, bands=4),
]


def state_dict_from_checkpoint(ckpt: Any) -> dict[str, Any] | None:
    if not isinstance(ckpt, dict):
        return None
    for key in ("model", "state_dict", "model_state_dict", "student", "model_student", "net"):
        value = ckpt.get(key)
        if isinstance(value, dict) and value:
            return value
    if ckpt and all(torch.is_tensor(value) for value in ckpt.values()):
        return ckpt
    return None


def first_conv_shape(state: dict[str, Any] | None) -> str:
    if not state:
        return ""
    for key, value in state.items():
        if torch.is_tensor(value) and value.ndim == 4:
            return f"{key}:{list(value.shape)}"
    return ""


def inspect(item: dict[str, Any]) -> dict[str, Any]:
    row = dict(item)
    ckpt_rel = item["checkpoint"]
    if ckpt_rel is None:
        row.update({"exists": True, "readable": True, "bytes": 0, "checkpoint_keys": "N/A (mask)",
                    "first_conv": "N/A (mask)", "compatibility": "not-applicable"})
        return row
    path = ROOT / ckpt_rel
    row["checkpoint"] = str(path)
    row["exists"] = path.is_file()
    row["bytes"] = path.stat().st_size if path.is_file() else 0
    row["readable"] = False
    row["checkpoint_keys"] = ""
    row["first_conv"] = ""
    row["compatibility"] = "pending-runtime-wrapper-check"
    if not path.is_file():
        row["compatibility"] = "N/A: missing checkpoint"
        return row
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        row["readable"] = True
        row["checkpoint_keys"] = ";".join(map(str, ckpt.keys()))[:800] if isinstance(ckpt, dict) else type(ckpt).__name__
        state = state_dict_from_checkpoint(ckpt)
        row["first_conv"] = first_conv_shape(state)
    except Exception as exc:  # recorded rather than hidden
        row["compatibility"] = f"N/A: unreadable ({type(exc).__name__}: {exc})"
    return row


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [inspect(item) for item in METHODS]
    (OUT / "checkpoint_inventory.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (OUT / "checkpoint_inventory.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"rows": len(rows), "readable": sum(bool(row["readable"]) for row in rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
