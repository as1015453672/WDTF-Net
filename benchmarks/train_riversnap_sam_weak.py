"""Weak-label fine-tuning and test evaluation of the public RiverSnap SAM protocol.

The public RiverSnap notebook fine-tunes SAM's mask decoder for river-water
segmentation.  This runner adapts that *training protocol* to the WDTF data:
the resampled GSW mask is the only training/validation target and manual HR
labels are read exclusively in ``evaluate_test`` after the selected checkpoint
has been fixed.  It intentionally uses no prompts, matching the RiverSnap
fine-tuning setup rather than the project's earlier frozen-SAM point baseline.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read(path: Path) -> np.ndarray:
    with rasterio.open(path) as source:
        return source.read().astype(np.float32)


def sample_id(path: Path, suffix: str) -> str:
    token = "_" + suffix
    return path.stem[:-len(token)] if path.stem.endswith(token) else path.stem


def train_samples(root: Path, split_path: Path) -> tuple[list[dict], list[dict], dict]:
    images = {sample_id(p, "hr"): p for p in (root / "hr").glob("*") if p.is_file()}
    masks = {sample_id(p, "mask"): p for p in (root / "mask").glob("*") if p.is_file()}
    split = json.loads(split_path.read_text(encoding="utf-8"))
    available = set(images) & set(masks)

    def select(ids: list[str]) -> list[dict]:
        missing = [sid for sid in ids if sid not in available]
        if missing:
            raise RuntimeError(f"Split references missing paired files, e.g. {missing[:5]}")
        return [{"id": sid, "image": images[sid], "mask": masks[sid]} for sid in ids]

    validation_ids = split.get("val", split.get("validation"))
    if validation_ids is None:
        raise KeyError("Split needs either a 'val' or 'validation' ID list.")
    return select(split["train"]), select(validation_ids), split


def make_input(image: np.ndarray, sam, device: torch.device,
               rgb_bands: tuple[int, int, int]) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
    # SAM expects an RGB image in [0, 255].  The band order is declared from
    # the dataset's documented visualization convention, never inferred from
    # labels or predictions.
    rgb = np.moveaxis(np.clip(image[list(rgb_bands)], 0.0, 1.0) * 255.0, 0, -1).round().astype(np.uint8)
    original_size = rgb.shape[:2]
    resized = ResizeLongestSide(sam.image_encoder.img_size).apply_image(rgb)
    tensor = torch.as_tensor(resized, device=device).permute(2, 0, 1).contiguous()[None]
    return sam.preprocess(tensor), tuple(tensor.shape[-2:]), original_size


def logits_for_batch(sam, images: list[np.ndarray], device: torch.device,
                     rgb_bands: tuple[int, int, int]) -> torch.Tensor:
    """Return decoder logits for equally sized images in one GPU batch.

    The RiverSnap notebook processes one patch at a time.  All WDTF patches
    are 512x512, so batching preserves its computation while avoiding needless
    image-encoder launch overhead.
    """
    prepared = [make_input(image, sam, device, rgb_bands) for image in images]
    model_inputs = torch.cat([entry[0] for entry in prepared], dim=0)
    input_sizes = {entry[1] for entry in prepared}
    original_sizes = {entry[2] for entry in prepared}
    if len(input_sizes) != 1 or len(original_sizes) != 1:
        raise RuntimeError("RiverSnap SAM batches require a common input and original image size.")
    input_size, original_size = next(iter(input_sizes)), next(iter(original_sizes))
    with torch.no_grad():
        embeddings = sam.image_encoder(model_inputs)
        sparse, dense = sam.prompt_encoder(points=None, boxes=None, masks=None)
    # SAM's released MaskDecoder batches prompts for *one* image, rather than
    # independent images.  Encode the images together, then decode each
    # embedding separately so no image/prompt pairs are crossed.
    outputs = []
    for embedding in embeddings.split(1, dim=0):
        low_res, _ = sam.mask_decoder(
            image_embeddings=embedding,
            image_pe=sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        outputs.append(sam.postprocess_masks(low_res, input_size, original_size))
    return torch.cat(outputs, dim=0)


def weak_target(path: Path, shape: tuple[int, int], threshold: float) -> torch.Tensor:
    raw = read(path)[0] > threshold
    target = torch.from_numpy(raw.astype(np.float32))[None, None]
    if tuple(raw.shape) != shape:
        target = F.interpolate(target, size=shape, mode="nearest")
    return target


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, float]:
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    intersection = (prob * target).sum()
    dice = 1.0 - (2.0 * intersection + 1.0) / (prob.sum() + target.sum() + 1.0)
    loss = bce + dice
    pred = prob >= 0.5
    truth = target.bool()
    iou = float((pred & truth).sum().item() / max((pred | truth).sum().item(), 1))
    return loss, iou


def weak_epoch(sam, samples: list[dict], divisor: float, threshold: float, device: torch.device,
               optimizer: torch.optim.Optimizer | None, batch_size: int,
               rgb_bands: tuple[int, int, int]) -> tuple[float, float]:
    train = optimizer is not None
    sam.mask_decoder.train(train)
    total_loss = total_iou = 0.0
    for start in range(0, len(samples), batch_size):
        batch = samples[start:start + batch_size]
        images = [read(item["image"]) / divisor for item in batch]
        target = torch.cat([weak_target(item["mask"], tuple(image.shape[-2:]), threshold) for item, image in zip(batch, images)], dim=0).to(device)
        with torch.set_grad_enabled(train):
            logits = logits_for_batch(sam, images, device, rgb_bands)
            loss, iou = segmentation_loss(logits, target)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(sam.mask_decoder.parameters(), 1.0)
                optimizer.step()
        total_loss += float(loss.detach().item()) * len(batch)
        total_iou += iou * len(batch)
    return total_loss / len(samples), total_iou / len(samples)


def test_items(root: Path) -> list[dict]:
    groups = {name: {sample_id(p, name): p for p in (root / name).glob("*") if p.is_file()}
              for name in ("hr", "label", "mask")}
    ids = sorted(set(groups["hr"]) & set(groups["label"]) & set(groups["mask"]))
    if not ids:
        raise RuntimeError("No matched hr/label/mask test triples found.")
    return [{"id": sid, "image": groups["hr"][sid], "label": groups["label"][sid]} for sid in ids]


def evaluate_test(sam, root: Path, divisor: float, device: torch.device, output: Path,
                  rgb_bands: tuple[int, int, int]) -> dict:
    """Read manual test labels only after model selection has concluded."""
    sam.eval()
    totals = dict(tp=0, tn=0, fp=0, fn=0)
    rows: list[dict] = []
    with torch.no_grad():
        for index, item in enumerate(test_items(root), 1):
            image = read(item["image"]) / divisor
            pred = torch.sigmoid(logits_for_batch(sam, [image], device, rgb_bands))[0, 0].cpu().numpy() >= 0.5
            label = read(item["label"])[0] > 0
            if pred.shape != label.shape:
                raise RuntimeError(f"Prediction/label size mismatch for {item['id']}: {pred.shape} vs {label.shape}")
            tp = int((pred & label).sum()); tn = int((~pred & ~label).sum())
            fp = int((pred & ~label).sum()); fn = int((~pred & label).sum())
            for key, value in (("tp", tp), ("tn", tn), ("fp", fp), ("fn", fn)):
                totals[key] += value
            iou = tp / max(tp + fp + fn, 1)
            rows.append({"id": item["id"], "IoU": iou, "F1": 2 * tp / max(2 * tp + fp + fn, 1),
                         "TP": tp, "TN": tn, "FP": fp, "FN": fn})
            if index == 1 or index % 20 == 0:
                print(f"evaluated {index}/{len(test_items(root))} test patches", flush=True)
    tp, tn, fp, fn = (totals[k] for k in ("tp", "tn", "fp", "fn"))
    water_iou = tp / max(tp + fp + fn, 1)
    bg_iou = tn / max(tn + fp + fn, 1)
    result = {
        "method": "RiverSnap SAM ViT-B weak-label fine-tuned",
        "architecture": "public RiverSnap mask-decoder fine-tuning protocol; official SAM ViT-B base checkpoint",
        "selection": "fixed weak-mask validation split only; manual HR labels excluded from training and checkpoint selection",
        "test_label_use": "final evaluation only",
        "samples": len(rows),
        "mIoU": (water_iou + bg_iou) / 2.0,
        "water_iou": water_iou,
        "mF1": ((2 * tp / max(2 * tp + fp + fn, 1)) + (2 * tn / max(2 * tn + fp + fn, 1))) / 2.0,
        "pixel_acc": (tp + tn) / max(tp + tn + fp + fn, 1),
        "mrecall": ((tp / max(tp + fn, 1)) + (tn / max(tn + fp, 1))) / 2.0,
        "counts": totals,
        "mean_patch_iou": float(np.mean([row["IoU"] for row in rows])),
    }
    with (output / "test_per_patch.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    (output / "test_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", required=True, type=Path)
    parser.add_argument("--test-root", required=True, type=Path)
    parser.add_argument("--split-json", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--divisor", type=float, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--weak-threshold", type=float, default=50.0)
    parser.add_argument("--rgb-bands", default="0,1,2",
                        help="Zero-based native-band indices to use as SAM RGB input, e.g. 2,1,0.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rgb_bands = tuple(int(token) for token in args.rgb_bands.split(","))
    if len(rgb_bands) != 3 or min(rgb_bands) < 0:
        raise ValueError("--rgb-bands must contain exactly three non-negative comma-separated indices.")

    set_seed(args.seed)
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, valid, split = train_samples(args.train_root, args.split_json)
    sam = sam_model_registry["vit_b"](checkpoint=str(args.checkpoint)).to(device)
    for parameter in sam.image_encoder.parameters(): parameter.requires_grad_(False)
    for parameter in sam.prompt_encoder.parameters(): parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(sam.mask_decoder.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    protocol = {
        "source": "https://github.com/ArminMoghimi/Fine-tune-the-Segment-Anything-Model-SAM-",
        "base_checkpoint": str(args.checkpoint), "model": "SAM ViT-B", "device": str(device),
        "train_root": str(args.train_root), "test_root": str(args.test_root), "split": split,
        "supervision": "GSW weak masks only; no prompts; no manual labels during training/selection",
        "optimization": "mask decoder only, BCEWithLogits + soft Dice", "seed": args.seed,
        "epochs_cap": args.epochs, "patience": args.patience, "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size, "rgb_bands": rgb_bands,
    }
    (out / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8")
    history: list[dict] = []; best_loss = float("inf"); stale = 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_iou = weak_epoch(sam, train, args.divisor, args.weak_threshold, device, optimizer, args.batch_size, rgb_bands)
        with torch.no_grad():
            valid_loss, valid_iou = weak_epoch(sam, valid, args.divisor, args.weak_threshold, device, None, args.batch_size, rgb_bands)
        record = {"epoch": epoch, "train_loss": train_loss, "train_weak_iou": train_iou,
                  "val_loss": valid_loss, "val_weak_iou": valid_iou, "elapsed_s": time.time() - started}
        history.append(record); print(json.dumps(record), flush=True)
        if valid_loss < best_loss:
            best_loss, stale = valid_loss, 0
            torch.save({"model": sam.state_dict(), "epoch": epoch, "val_weak_loss": valid_loss,
                        "protocol": protocol}, out / "best_mask_decoder_sam_vit_b.pth")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stopping at epoch {epoch}", flush=True); break
    (out / "training_history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    saved = torch.load(out / "best_mask_decoder_sam_vit_b.pth", map_location=device, weights_only=False)
    sam.load_state_dict(saved["model"], strict=True)
    result = evaluate_test(sam, args.test_root, args.divisor, device, out, rgb_bands)
    result["best_epoch"] = saved["epoch"]; result["best_val_weak_loss"] = saved["val_weak_loss"]
    (out / "test_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

