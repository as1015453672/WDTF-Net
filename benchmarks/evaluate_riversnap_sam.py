"""Evaluate a downloaded RiverSnap-SAM fine-tuned checkpoint locally.

The checkpoint saved by ``train_riversnap_sam_weak.py`` contains the full SAM
state dict under ``model``.  This runner reloads the official ViT-B base model,
restores that state dict, and evaluates only against the manual test labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from segment_anything import sam_model_registry

from train_riversnap_sam_weak import evaluate_test, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-root", required=True, type=Path)
    parser.add_argument("--fine-tuned-checkpoint", required=True, type=Path)
    parser.add_argument("--base-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--divisor", required=True, type=float)
    parser.add_argument("--rgb-bands", default="0,1,2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rgb_bands = tuple(int(token) for token in args.rgb_bands.split(","))
    if len(rgb_bands) != 3 or min(rgb_bands) < 0:
        raise ValueError("--rgb-bands must contain exactly three non-negative indices")
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry["vit_b"](checkpoint=str(args.base_checkpoint)).to(device)
    saved = torch.load(args.fine_tuned_checkpoint, map_location=device, weights_only=False)
    if not isinstance(saved, dict) or "model" not in saved:
        raise RuntimeError(f"Expected a training checkpoint with a 'model' state dict: {args.fine_tuned_checkpoint}")
    sam.load_state_dict(saved["model"], strict=True)
    result = evaluate_test(sam, args.test_root, args.divisor, device, args.output_dir, rgb_bands)
    result.update({
        "evaluation_source": "local re-evaluation of downloaded cloud checkpoint",
        "fine_tuned_checkpoint": str(args.fine_tuned_checkpoint),
        "base_checkpoint": str(args.base_checkpoint),
        "device": str(device),
        "rgb_bands": rgb_bands,
        "checkpoint_epoch": saved.get("epoch"),
    })
    (args.output_dir / "local_test_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()


