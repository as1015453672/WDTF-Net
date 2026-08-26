"""Clean Sentinel-2 training/testing entry point for the revised WDTF-Net.

This script is intentionally self-contained at the workflow level.  It uses a
fixed Stage-1 teacher in Stage 2, freezes BatchNorm buffers in the student
backbone, performs group-wise validation, applies early stopping, and stores
the full configuration with each checkpoint.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import rasterio
import torch
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset

from .losses import Stage2AdaptiveGraphPrototypeLoss
from .stage1_loss import Stage1WeakPrototypeLoss
from .model import WDTFNetConfig, WDTFNetOptimized


@dataclass
class RunConfig:
    train_root: str
    test_root: str
    output_dir: str
    val_group: str = "05"
    seed: int = 42
    in_channels: int = 4
    base_channels: int = 64
    batch_size: int = 8
    workers: int = 0
    stage1_epochs: int = 60
    stage2_epochs: int = 60
    patience: int = 5
    min_delta: float = 1e-4
    lr1: float = 1e-4
    lr2: float = 1e-5
    weight_decay: float = 1e-4
    divisor: float = 4096.0
    mask_threshold: float = 50.0
    # Fixed synthetic weak-label perturbation used only for the robustness study.
    # It is stored in every checkpoint configuration.
    weak_perturb: str = "none"  # none | shift_2px | drop_20pct_components


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def key(path: Path, suffix: str) -> str:
    stem = path.stem
    token = "_" + suffix
    return stem[:-len(token)] if stem.endswith(token) else stem


def read(path: str) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)


def boundary_band(mask: torch.Tensor, kernel: int = 9) -> torch.Tensor:
    import torch.nn.functional as F
    pad = kernel // 2
    x = mask.unsqueeze(0)
    dil = F.max_pool2d(x, kernel, 1, pad)
    ero = 1.0 - F.max_pool2d(1.0 - x, kernel, 1, pad)
    return (dil - ero).clamp(0, 1).squeeze(0)


class WeakTrainDataset(Dataset):
    def __init__(self, items: List[Dict[str, str]], cfg: RunConfig, augment: bool):
        self.items, self.cfg, self.augment = items, cfg, augment

    def __len__(self): return len(self.items)

    @staticmethod
    def _zero_padded_shift(mask: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
        """Translate a CHW mask without wrap-around artefacts."""
        out = torch.zeros_like(mask)
        h, w = mask.shape[-2:]
        src_y0, src_y1 = max(0, -dy), min(h, h - dy)
        src_x0, src_x1 = max(0, -dx), min(w, w - dx)
        dst_y0, dst_y1 = max(0, dy), min(h, h + dy)
        dst_x0, dst_x1 = max(0, dx), min(w, w + dx)
        out[..., dst_y0:dst_y1, dst_x0:dst_x1] = mask[..., src_y0:src_y1, src_x0:src_x1]
        return out

    def _perturb_weak_mask(self, mask: torch.Tensor, sample_id: str) -> torch.Tensor:
        """Apply a deterministic corruption so train and validation conditions match."""
        mode = self.cfg.weak_perturb
        if mode == "none":
            return mask
        key = zlib.crc32(sample_id.encode("utf-8")) + int(self.cfg.seed)
        if mode == "shift_2px":
            shifts = [(-2, -2), (-2, 0), (-2, 2), (0, -2), (0, 2), (2, -2), (2, 0), (2, 2)]
            dy, dx = shifts[key % len(shifts)]
            return self._zero_padded_shift(mask, dy, dx)
        if mode == "drop_20pct_components":
            binary = mask[0].numpy() > 0.5
            components, number = ndimage.label(binary, structure=np.ones((3, 3), dtype=np.uint8))
            if number == 0:
                return mask
            rng = np.random.default_rng(key)
            to_drop = rng.choice(np.arange(1, number + 1), size=max(1, int(round(0.20 * number))), replace=False)
            corrupted = mask.clone()
            corrupted[0][torch.from_numpy(np.isin(components, to_drop))] = 0.0
            return corrupted
        raise ValueError(f"Unsupported weak_perturb={mode!r}")

    def __getitem__(self, index):
        item = self.items[index]
        image = torch.from_numpy(read(item["image"]) / self.cfg.divisor)
        mask = torch.from_numpy((read(item["mask"]) > self.cfg.mask_threshold).astype(np.float32))
        if self.augment:
            if random.random() < 0.5: image, mask = image.flip(-1), mask.flip(-1)
            if random.random() < 0.5: image, mask = image.flip(-2), mask.flip(-2)
        mask = self._perturb_weak_mask(mask, item["id"])
        boundary = boundary_band(mask)
        conf = (1.0 - boundary) + 0.35 * boundary
        return {"id": item["id"], "img": image, "mask": mask, "conf": conf.clamp(0.15, 1), "boundary": boundary}


class TestDataset(Dataset):
    def __init__(self, root: str, cfg: RunConfig):
        r = Path(root); hrs = {key(p, "hr"): p for p in (r / "hr").glob("*")}; labels = {key(p, "label"): p for p in (r / "label").glob("*")}
        self.items = [{"id": i, "image": str(hrs[i]), "label": str(labels[i])} for i in sorted(hrs.keys() & labels.keys())]
        self.cfg = cfg
        if not self.items: raise RuntimeError("No matching hr/label test pairs found.")

    def __len__(self): return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        return {"id": item["id"], "img": torch.from_numpy(read(item["image"]) / self.cfg.divisor), "label": torch.from_numpy((read(item["label"]) > 0).astype(np.float32))}


def pairs(root: str) -> List[Dict[str, str]]:
    r = Path(root); hrs = {key(p, "hr"): p for p in (r / "hr").glob("*")}; masks = {key(p, "mask"): p for p in (r / "mask").glob("*")}
    ans = [{"id": i, "image": str(hrs[i]), "mask": str(masks[i])} for i in sorted(hrs.keys() & masks.keys())]
    if not ans: raise RuntimeError("No matching hr/mask training pairs found.")
    return ans


def make_model(cfg: RunConfig, device: torch.device) -> WDTFNetOptimized:
    return WDTFNetOptimized(WDTFNetConfig(in_channels=cfg.in_channels, base_channels=cfg.base_channels, num_templates=6)).to(device)


def mean_loss(model, loader, criterion, device, stage2=False, teacher=None):
    model.eval(); total = 0.0
    with torch.no_grad():
        for batch in loader:
            img = batch["img"].to(device); out = model(img, use_adapters=stage2)
            if stage2:
                loss = criterion(img, teacher(img, use_adapters=False), out, batch["mask"].to(device), compute_spec=False)["loss"]
            else:
                loss = criterion(out, img, batch["mask"].to(device), batch["conf"].to(device), batch["boundary"].to(device), 0)["loss"]
            total += float(loss)
    return total / max(1, len(loader))


def fit_stage1(model, train_loader, val_loader, cfg, device, out_dir):
    criterion = Stage1WeakPrototypeLoss().to(device); optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr1, weight_decay=cfg.weight_decay)
    best, bad, path = float("inf"), 0, out_dir / "stage1_best.pt"
    for epoch in range(cfg.stage1_epochs):
        model.train()
        for b in train_loader:
            img=b["img"].to(device); loss=criterion(model(img, False), img, b["mask"].to(device), b["conf"].to(device), b["boundary"].to(device), epoch)["loss"]
            optim.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optim.step()
        val = mean_loss(model, val_loader, criterion, device)
        print(f"stage1 {epoch+1:03d}: val_weak_loss={val:.5f}", flush=True)
        if val < best-cfg.min_delta:
            best, bad = val, 0; torch.save({"model": model.state_dict(), "config": asdict(cfg), "val_loss": val}, path)
        else:
            bad += 1
            if bad >= cfg.patience: print("stage1 early stop", flush=True); break
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"], strict=True)
    return path


def fit_stage2(model, train_loader, val_loader, cfg, device, out_dir):
    teacher = copy.deepcopy(model).to(device).eval()
    for p in teacher.parameters(): p.requires_grad = False
    model.freeze_for_stage2(train_seg_head=True, tune_geometry=False)
    criterion = Stage2AdaptiveGraphPrototypeLoss(lambda_proto=1.0, lambda_cons=0.22, lambda_smooth=0.05, lambda_prior=0.15, lambda_absent=0.32, proto_uncertain_weight=0.35, gc_stage1_weight=0.85, gc_prior_weight=0.15, gc_strong_weight=1.0, gc_mid_weight=0.30).to(device)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr2, weight_decay=cfg.weight_decay)
    best, bad, path = float("inf"), 0, out_dir / "stage2_best.pt"
    for epoch in range(cfg.stage2_epochs):
        model.set_stage2_train_mode()
        for b in train_loader:
            img=b["img"].to(device)
            with torch.no_grad(): anchor=teacher(img, use_adapters=False)
            loss=criterion(img, anchor, model(img, use_adapters=True), b["mask"].to(device), compute_spec=False)["loss"]
            optim.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0); optim.step()
        val = mean_loss(model, val_loader, criterion, device, True, teacher)
        print(f"stage2 {epoch+1:03d}: val_weak_loss={val:.5f}", flush=True)
        if val < best-cfg.min_delta:
            best, bad = val, 0; torch.save({"model": model.state_dict(), "config": asdict(cfg), "val_loss": val}, path)
        else:
            bad += 1
            if bad >= cfg.patience: print("stage2 early stop", flush=True); break
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model"], strict=True)
    return path


def test(model, loader, device, out_dir):
    model.eval(); sums = dict(tp=0, tn=0, fp=0, fn=0); rows=[]
    with torch.no_grad():
        for b in loader:
            p=(torch.sigmoid(model(b["img"].to(device), True)["logits"])>=0.5).cpu(); y=b["label"]>0.5
            tp=int((p & y).sum()); tn=int((~p & ~y).sum()); fp=int((p & ~y).sum()); fn=int((~p & y).sum())
            for k,v in zip(sums,(tp,tn,fp,fn)): sums[k]+=v
            rows.append({"id":b["id"][0],"TP":tp,"TN":tn,"FP":fp,"FN":fn})
    e=1e-6; tp,tn,fp,fn=(sums[k] for k in ("tp","tn","fp","fn")); metrics={"IoU":tp/(tp+fp+fn+e),"F1":2*tp/(2*tp+fp+fn+e),"Acc":(tp+tn)/(tp+tn+fp+fn+e),"Recall":tp/(tp+fn+e),"Precision":tp/(tp+fp+e)}
    with open(out_dir/"test_metrics.json","w",encoding="utf-8") as f: json.dump({"metrics":metrics,"counts":sums,"warning":"Test scenes overlap training scene prefixes; use only as a local verification run."},f,indent=2)
    with open(out_dir/"test_per_patch.csv","w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    print("test", metrics, flush=True)


def main():
    p=argparse.ArgumentParser(description="Paper-aligned two-stage WDTF-Net training and final test.")
    p.add_argument("--train-root",required=True); p.add_argument("--test-root",required=True); p.add_argument("--output-dir",required=True)
    p.add_argument("--val-group",default="05", help="Prefix of the held-out validation scene group.")
    p.add_argument("--seed",type=int,default=42); p.add_argument("--epochs",type=int,default=60); p.add_argument("--patience",type=int,default=5)
    p.add_argument("--batch-size",type=int,default=8); p.add_argument("--in-channels",type=int,default=4); p.add_argument("--base-channels",type=int,default=64)
    p.add_argument("--divisor",type=float,default=4096.0); p.add_argument("--weak-perturb",choices=["none", "shift_2px", "drop_20pct_components"],default="none")
    args=p.parse_args()
    cfg=RunConfig(args.train_root,args.test_root,args.output_dir,args.val_group,seed=args.seed,in_channels=args.in_channels,base_channels=args.base_channels,batch_size=args.batch_size,stage1_epochs=args.epochs,stage2_epochs=args.epochs,patience=args.patience,divisor=args.divisor,weak_perturb=args.weak_perturb); set_seed(cfg.seed); out=Path(cfg.output_dir); out.mkdir(parents=True,exist_ok=True)
    all_pairs=pairs(cfg.train_root); train=[x for x in all_pairs if x["id"].split("_")[0] != cfg.val_group]; val=[x for x in all_pairs if x["id"].split("_")[0] == cfg.val_group]
    if not train or not val: raise RuntimeError("Group split is empty; choose a valid --val-group.")
    (out/"split.json").write_text(json.dumps({"train": [x["id"] for x in train], "val": [x["id"] for x in val]},indent=2),encoding="utf-8")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"device={device}; train={len(train)}; val={len(val)}; test={len(TestDataset(cfg.test_root,cfg))}")
    tr=DataLoader(WeakTrainDataset(train,cfg,True),batch_size=cfg.batch_size,shuffle=True,num_workers=cfg.workers,pin_memory=True); va=DataLoader(WeakTrainDataset(val,cfg,False),batch_size=cfg.batch_size,shuffle=False,num_workers=cfg.workers,pin_memory=True); te=DataLoader(TestDataset(cfg.test_root,cfg),batch_size=1,shuffle=False)
    model=make_model(cfg,device); fit_stage1(model,tr,va,cfg,device,out); fit_stage2(model,tr,va,cfg,device,out); test(model,te,device,out)

if __name__ == "__main__": main()
