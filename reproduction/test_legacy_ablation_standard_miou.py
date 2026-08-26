"""Fresh paper-metric evaluation for the original 0--7 WDTF-Net ablations.

This runner preserves every legacy experiment's model switches, input divisor,
and 0.5 decision threshold.  Its primary results exactly follow the paper
tables: calculate water-foreground IoU/F1/Acc/Recall/Precision for every test
image with eps=1e-6, then average images.  It never trains or changes a
checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("WDTF_ARTIFACT_ROOT", REPO_ROOT / "artifacts"))
OUT_ROOT = Path(os.environ.get("WDTF_REPRO_OUTPUT", REPO_ROOT / "runs")) / "legacy_ablation"
CURRENT_SENTINEL_FULL = ROOT / "experiment_revised_all_method_retest_20260821" / "sentinel" / "predictions" / "wdtfnet" / "summary.json"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    test_root: Path
    ablation_root: Path
    channels: int
    divisor: float
    source_module: str


DATASETS = (
    DatasetSpec("GF", Path(os.environ.get("WDTF_GF_TEST_ROOT", "data/gf/test")), ROOT / "experiment_gf" / "ablation_wdtfnet", 3, 1024.0, "test_ablation_sentinel"),
    DatasetSpec("Sentinel", Path(os.environ.get("WDTF_SENTINEL_TEST_ROOT", "data/sentinel/test")), ROOT / "experiment_sentinel" / "ablation_wdtfnet", 4, 4096.0, "test_ablation_sentinel"),
    DatasetSpec("WorldView", Path(os.environ.get("WDTF_WORLDVIEW_TEST_ROOT", "data/worldview/test")), ROOT / "experiment_worldview" / "ablation_wdtfnet", 4, 2048.0, "test_ablation_worldview"),
)

VARIANTS = (
    (0, "Baseline", "00_baseline", "stage1/stage1_best.pth", False, False, False),
    (1, "+WGDC", "01_baseline_wgdc", "stage1/stage1_best.pth", True, False, False),
    (2, "+FG", "02_baseline_fg", "stage1/stage1_best.pth", False, True, False),
    (3, "+WA", "03_baseline_wa", "stage2/stage2_best.pth", False, False, True),
    (4, "w/o WGDC", "04_wo_wgdc", "stage2/stage2_best.pth", False, True, True),
    (5, "w/o FG", "05_wo_fg", "stage2/stage2_best.pth", True, False, True),
    (6, "w/o WA", "07_full_wdtfnet", "stage1_best.pth", True, True, False),
    (7, "Full WDTF-Net", "07_full_wdtfnet", "stage2_best.pth", True, True, True),
)


def counts_for(prediction: np.ndarray, target: np.ndarray) -> dict[str, int]:
    return {
        "TP": int(np.logical_and(prediction, target).sum()),
        "TN": int(np.logical_and(~prediction, ~target).sum()),
        "FP": int(np.logical_and(prediction, ~target).sum()),
        "FN": int(np.logical_and(~prediction, target).sum()),
    }


def global_metrics(counts: dict[str, int]) -> dict[str, float]:
    tp, tn, fp, fn = (counts[key] for key in ("TP", "TN", "FP", "FN"))
    eps = 1e-12
    water_iou = tp / (tp + fp + fn + eps)
    background_iou = tn / (tn + fp + fn + eps)
    water_f1 = 2 * tp / (2 * tp + fp + fn + eps)
    background_f1 = 2 * tn / (2 * tn + fp + fn + eps)
    return {
        "miou": (water_iou + background_iou) / 2,
        "background_iou": background_iou,
        "water_iou": water_iou,
        "mf1": (water_f1 + background_f1) / 2,
        "water_f1": water_f1,
        "pixel_acc": (tp + tn) / (tp + tn + fp + fn + eps),
        "water_recall": tp / (tp + fn + eps),
        "water_precision": tp / (tp + fp + eps),
    }


def paper_metrics(counts: dict[str, int]) -> dict[str, float]:
    """Original manuscript metrics for one image: water is the foreground."""
    tp, tn, fp, fn = (counts[key] for key in ("TP", "TN", "FP", "FN"))
    eps = 1e-6
    return {
        "IoU": tp / (tp + fp + fn + eps),
        "F1": 2 * tp / (2 * tp + fp + fn + eps),
        "Acc": (tp + tn) / (tp + tn + fp + fn + eps),
        "Recall": tp / (tp + fn + eps),
        "Precision": tp / (tp + fp + eps),
    }


def variant_config(root: Path, item: tuple[Any, ...]) -> tuple[dict[str, Any], Path]:
    index, display, directory, relative_checkpoint, use_wgdc, use_fg, use_wa = item
    checkpoint = root / directory / relative_checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return {
        "id": index, "display": display, "use_wgdc": use_wgdc,
        "use_fg": use_fg, "use_wa": use_wa, "ckpt": str(checkpoint),
    }, checkpoint


def build_model(source: Any, config: dict[str, Any], device: torch.device) -> tuple[torch.nn.Module, str]:
    """Load an exact segmentation state, allowing only the unused legacy edge head."""
    checkpoint = source.load_torch(config["ckpt"], device)
    wrapper = source.AblationWDTFWrapper(config)
    state = source.clean_state_dict(source.extract_state_dict(checkpoint))
    missing, unexpected = wrapper.model.load_state_dict(state, strict=False)
    allowed_auxiliary = {key for key in unexpected if key.startswith("edge_head.")}
    disallowed_unexpected = set(unexpected) - allowed_auxiliary
    if missing or disallowed_unexpected:
        raise RuntimeError(
            f"Incompatible state dict for {config['display']}: missing={len(missing)}, "
            f"disallowed unexpected={len(disallowed_unexpected)}"
        )
    compatibility = "exact segmentation state; ignored unused edge_head auxiliary weights" if allowed_auxiliary else "exact state dict"
    return wrapper.to(device).eval(), compatibility


def evaluate_dataset(spec: DatasetSpec) -> list[dict[str, Any]]:
    source = importlib.import_module(spec.source_module)
    source.TEST_ROOT = str(spec.test_root)
    source.IN_CHANNELS = spec.channels
    source.HR_DIVISOR = spec.divisor
    source.THRESHOLD = 0.5
    items = source.find_test_samples(str(spec.test_root))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = OUT_ROOT / spec.name.lower()
    output.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict[str, Any]] = []

    for item in VARIANTS:
        config, checkpoint_path = variant_config(spec.ablation_root, item)
        print(f"[{spec.name}] {config['display']}: {checkpoint_path}", flush=True)
        model, compatibility = build_model(source, config, device)
        total = {key: 0 for key in ("TP", "TN", "FP", "FN")}
        per_sample: list[dict[str, Any]] = []
        with torch.no_grad():
            for position, sample in enumerate(items, 1):
                image = source.normalize_image(source.read_raster(sample["hr_path"]), spec.divisor)
                target = source.binarize_label(source.read_raster(sample["label_path"])[0]).astype(bool)
                tensor = torch.from_numpy(image).unsqueeze(0).to(device)
                prediction = (model(tensor, target.shape).squeeze().cpu().numpy() >= 0.5)
                sample_counts = counts_for(prediction, target)
                for key, value in sample_counts.items():
                    total[key] += value
                per_sample.append({"id": sample["id"], **sample_counts, **paper_metrics(sample_counts)})
                if position == 1 or position % 20 == 0 or position == len(items):
                    print(f"  {position}/{len(items)}", flush=True)
        method_dir = output / f"{int(config['id']):02d}_{config['display'].replace('/', '_').replace(' ', '_')}"
        method_dir.mkdir(parents=True, exist_ok=True)
        with (method_dir / "per_sample_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(per_sample[0]))
            writer.writeheader(); writer.writerows(per_sample)
        paper_means = {key: float(np.mean([row[key] for row in per_sample]))
                       for key in ("IoU", "F1", "Acc", "Recall", "Precision")}
        diagnostics = global_metrics(total)
        summary = {
            "dataset": spec.name, "variant_id": config["id"], "variant": config["display"],
            "checkpoint": str(checkpoint_path), "use_wgdc": config["use_wgdc"],
            "use_fg": config["use_fg"], "use_wa": config["use_wa"], "samples": len(items),
            "input_normalization": f"float32 / {spec.divisor} (legacy protocol; no new clipping)",
            "threshold": 0.5,
            "primary_metric": "paper per-image water-foreground metrics (mean over test images)",
            "metric_protocol": "paper_per_image_water_foreground",
            "metric_definition": "Per image: IoU=TP/(TP+FP+FN), F1=2TP/(2TP+FP+FN), Acc=(TP+TN)/N, Recall=TP/(TP+FN), eps=1e-6; then mean across images.",
            "state_dict_compatibility": compatibility,
            **total, **paper_means,
            # Retain the previous global aggregate only as an explicit diagnostic.
            "global_class_miou": diagnostics["miou"], "global_background_iou": diagnostics["background_iou"],
            "global_water_iou": diagnostics["water_iou"], "global_class_mf1": diagnostics["mf1"],
            "global_pixel_acc": diagnostics["pixel_acc"], "global_water_f1": diagnostics["water_f1"],
            "global_water_recall": diagnostics["water_recall"], "global_water_precision": diagnostics["water_precision"],
        }
        (method_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        result_rows.append(summary)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fields = list(result_rows[0])
    with (output / "legacy_ablation_metrics_paper_foreground.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(result_rows)
    return result_rows


def replace_with_current_sentinel_full(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create a paper-presentation view while preserving the raw legacy retest."""
    current = json.loads(CURRENT_SENTINEL_FULL.read_text(encoding="utf-8"))
    replaced: list[dict[str, Any]] = []
    for row in rows:
        if row["dataset"] == "Sentinel" and row["variant_id"] == 7:
            updated = dict(row)
            updated.update({
                "variant": "Full WDTF-Net (current paper source; non-paired)",
                "checkpoint": current["checkpoint"],
                "samples": current["samples"],
                "TP": current["TP"], "TN": current["TN"], "FP": current["FP"], "FN": current["FN"],
                "IoU": current["IoU"], "F1": current["F1"], "Acc": current["Acc"],
                "Recall": current["Recall"], "Precision": current["Precision"],
                "state_dict_compatibility": "current paper-source checkpoint; inference provenance in experiment_revised_all_method_retest_20260821",
            })
            replaced.append(updated)
        else:
            replaced.append(row)
    return replaced


def write_master(rows: list[dict[str, Any]], paper_sentinel_full: bool = False) -> None:
    if paper_sentinel_full:
        rows = replace_with_current_sentinel_full(rows)
    fields = list(rows[0])
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    suffix = "_paper_sentinel_full" if paper_sentinel_full else ""
    with (OUT_ROOT / f"legacy_ablation_metrics_paper_foreground{suffix}.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    lines = [
        "# Original 0--7 ablation checkpoints: paper-metric retest", "",
        "All rows use the original checkpoint, model switches, input divisor, and 0.5 threshold. The primary metrics exactly match the manuscript protocol: per-image water-foreground IoU/F1/Acc/Recall/Precision, then mean across test images.", "",
        "| Dataset | Variant | IoU | F1 | Acc | Recall | Precision | Global water IoU (diagnostic) |", "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append("| {dataset} | {variant} | {IoU:.4f} | {F1:.4f} | {Acc:.4f} | {Recall:.4f} | {Precision:.4f} | {global_water_iou:.4f} |".format(**row))
    lines.append("")
    if paper_sentinel_full:
        lines.append("The Sentinel Full row is intentionally replaced with the current paper-source checkpoint. It is not the same training batch as the other legacy Sentinel variants and must not be used for strict paired causal differences.")
    else:
        lines.append("These are legacy experiments and are reported separately from the later controlled Sentinel Haar ablation.")
    (OUT_ROOT / f"legacy_ablation_metrics_paper_foreground{suffix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_existing(specs: list[DatasetSpec]) -> list[dict[str, Any]]:
    """Rebuild the master table from completed per-dataset inference results."""
    rows: list[dict[str, Any]] = []
    for spec in specs:
        path = OUT_ROOT / spec.name.lower() / "legacy_ablation_metrics_paper_foreground.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                for key in ("variant_id", "samples", "TP", "TN", "FP", "FN"):
                    row[key] = int(row[key])
                for key in ("IoU", "F1", "Acc", "Recall", "Precision", "global_class_miou", "global_background_iou", "global_water_iou", "global_class_mf1", "global_pixel_acc", "global_water_f1", "global_water_recall", "global_water_precision", "threshold"):
                    row[key] = float(row[key])
                for key in ("use_wgdc", "use_fg", "use_wa"):
                    row[key] = row[key].strip().lower() == "true"
                rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=[item.name for item in DATASETS], default=[item.name for item in DATASETS])
    parser.add_argument("--summarize-existing", action="store_true", help="rebuild the master table without inference")
    parser.add_argument("--paper-sentinel-full", action="store_true", help="use the current 0.9582 Sentinel paper-source Full row in a separate presentation table")
    args = parser.parse_args()
    selected = [item for item in DATASETS if item.name in args.datasets]
    if args.summarize_existing:
        write_master(load_existing(selected), paper_sentinel_full=args.paper_sentinel_full)
        return
    all_rows: list[dict[str, Any]] = []
    for spec in selected:
        all_rows.extend(evaluate_dataset(spec))
    write_master(all_rows, paper_sentinel_full=args.paper_sentinel_full)


if __name__ == "__main__":
    main()
