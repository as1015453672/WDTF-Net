"""Run WDTF-Net on GeoTIFF images and optionally calculate binary water metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import torch

from wdtf.model import WDTFNetConfig, WDTFNetOptimized


def stem_without_suffix(path: Path, suffix: str) -> str:
    token = "_" + suffix
    return path.stem[:-len(token)] if path.stem.endswith(token) else path.stem


def load_model(path: str, stage: str, device: torch.device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("format") == "wdtf_dual_stage_inference_v1":
        checkpoint = checkpoint[stage]
    state = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
    config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    if any(key.startswith("edge_head.") for key in state) and not any(key.startswith("skel_head.") for key in state):
        state = dict(state)
        state["skel_head.weight"] = state.pop("edge_head.weight")
        state["skel_head.bias"] = state.pop("edge_head.bias")
    stem_weight = state["stem.block.0.block.0.weight"]
    model = WDTFNetOptimized(WDTFNetConfig(
        in_channels=int(config.get("in_channels", stem_weight.shape[1])),
        base_channels=int(config.get("base_channels", stem_weight.shape[0])),
        num_templates=int(config.get("num_templates", 6)),
        use_skeleton_head=any(key.startswith("skel_head.") for key in state),
    )).to(device)
    model.load_state_dict(state, strict=True)
    return model.eval(), checkpoint.get("stage", 2) if isinstance(checkpoint, dict) else 2


def metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred, target = pred.astype(bool), target.astype(bool)
    tp, tn = np.logical_and(pred, target).sum(), np.logical_and(~pred, ~target).sum()
    fp, fn = np.logical_and(pred, ~target).sum(), np.logical_and(~pred, target).sum()
    return metrics_from_counts(int(tp), int(tn), int(fp), int(fn))


def metrics_from_counts(tp: int, tn: int, fp: int, fn: int) -> dict[str, float]:
    eps = 1e-8
    return {"water_iou": float(tp / (tp + fp + fn + eps)),
            "water_f1": float(2 * tp / (2 * tp + fp + fn + eps)),
            "accuracy": float((tp + tn) / (tp + tn + fp + fn + eps)),
            "water_recall": float(tp / (tp + fn + eps)),
            "water_precision": float(tp / (tp + fp + eps)),
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input-dir", required=True, help="Folder of input GeoTIFFs.")
    parser.add_argument("--output-dir", default="predictions")
    parser.add_argument("--label-dir", help="Optional label GeoTIFF folder for evaluation.")
    parser.add_argument("--image-suffix", default="hr", help="Suffix removed to match labels, e.g. sample_hr.tif.")
    parser.add_argument("--label-suffix", default="label")
    parser.add_argument("--stage", choices=("stage1", "stage2"), default="stage2")
    parser.add_argument("--divisor", type=float, default=1024.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model, checkpoint_stage = load_model(args.checkpoint, args.stage, device)
    use_adapters = args.stage == "stage2" or checkpoint_stage == 2
    inputs = sorted([p for p in Path(args.input_dir).iterdir() if p.suffix.lower() in {".tif", ".tiff"}])
    if not inputs:
        raise FileNotFoundError(f"No GeoTIFFs found in {args.input_dir}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label_dir = Path(args.label_dir) if args.label_dir else None
    totals = {key: 0 for key in ("tp", "tn", "fp", "fn")}

    with torch.inference_mode():
        for image_path in inputs:
            with rasterio.open(image_path) as source:
                image, profile = source.read().astype(np.float32), source.profile.copy()
            x = torch.from_numpy(np.clip(image / args.divisor, 0, 1)).unsqueeze(0).to(device)
            probability = torch.sigmoid(model(x, use_adapters=use_adapters)["logits"])[0, 0].cpu().numpy()
            binary = (probability >= args.threshold).astype(np.uint8)
            profile.update(count=1, dtype="uint8", nodata=0)
            with rasterio.open(out_dir / f"{image_path.stem}_water.tif", "w", **profile) as target:
                target.write(binary, 1)
            if label_dir:
                sample_id = stem_without_suffix(image_path, args.image_suffix)
                label_path = label_dir / f"{sample_id}_{args.label_suffix}{image_path.suffix}"
                if label_path.is_file():
                    with rasterio.open(label_path) as source:
                        label = source.read(1) > 0
                    item = metrics(binary, label)
                    for key in totals:
                        totals[key] += item[key]

    if label_dir:
        summary = metrics_from_counts(**totals)
        (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
    print(f"Saved {len(inputs)} water masks to {out_dir}")


if __name__ == "__main__":
    main()
