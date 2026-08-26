# WDTF-Net

Standalone implementation of WDTF-Net for weakly supervised water extraction.
It contains the paper-aligned two-stage WDTF training/test protocol, controlled
WGDC/FG/WA ablations, and generic GeoTIFF inference. It deliberately excludes
third-party comparison-method implementations, paper figures, profiling code,
datasets, checkpoints, and experiment logs.

## Install

```bash
conda create -n wdtf python=3.10 -y
conda activate wdtf
pip install -r requirements.txt
```

Install the PyTorch build matching your CUDA environment from
https://pytorch.org when GPU acceleration is required.

## Dataset layout

Training uses weak masks on the same grid as the input image. Filename prefixes
must begin with a scene-group ID (for example `05_...`) so the fixed validation
scene can be selected with `--val-group`:

```text
data_root/
├── hr/        sample_001_hr.tif
└── mask/      sample_001_mask.tif
```

Each image/mask pair must share the filename prefix. Test labels are optional:

```text
test_root/
├── hr/        sample_001_hr.tif
└── label/     sample_001_label.tif
```

Images are normalized by `--divisor`; use the value appropriate for the source
data. The revised-paper experiments use seed 42, batch size 8, weight decay
`1e-4`, and 60 epochs per stage.

## Train

```bash
python train.py --train-root /path/to/train_root --test-root /path/to/test_root \
  --output-dir runs/sentinel --val-group 05 --epochs 60 --patience 5 \
  --batch-size 8 --seed 42 --divisor 4096 --in-channels 4
```

The best checkpoints are saved as `runs/sentinel/stage1_best.pt` and
`runs/sentinel/stage2_best.pt`; the final test metrics are stored in
`test_metrics.json`. The training defaults reproduce the article's reported
optimization schedule: AdamW, Stage-1/Stage-2 learning rates `1e-4`/`1e-5`,
weight decay `1e-4`, 60-epoch budget per stage, patience 5, batch size 8, and
seed 42. Use the dataset-specific divisor and channel count (Sentinel/WV: 4;
GF: 3).

## Controlled ablation

```bash
python ablation.py --train-root /path/to/train_root --test-root /path/to/test_root \
  --output-dir runs/no_wgdc --variant no_wgdc --val-group 05 \
  --epochs 60 --patience 5 --batch-size 8 --in-channels 4 --divisor 4096
```

Supported variants are `full`, `no_wgdc`, `no_fg`, and `no_wa`. Each uses the
same split, weak supervision, optimizer schedule, validation selection, and
test protocol as the full WDTF run.

## Inference and evaluation

```bash
python infer.py --checkpoint /path/to/stage2_best.pth \
  --input-dir /path/to/test_root/hr --label-dir /path/to/test_root/label \
  --output-dir predictions --stage stage2 --divisor 1024
```

Predicted binary water masks are saved as GeoTIFFs with the source image
profile. When `--label-dir` is supplied, `metrics.json` reports Water IoU,
Water F1, accuracy, recall, and precision.

## Checkpoints and data

Weights and data are intentionally ignored by Git. Download/release them
separately and place them anywhere; pass their paths to the commands above.

## Comparison and paper-table reproduction

The adapted comparison-method code is in [benchmarks/](benchmarks/README.md),
while the unified paper-table and legacy-ablation evaluators are in
[reproduction/](reproduction/README.md). They are kept separate from the
proposed-method implementation and use released artifacts through an explicit
path, rather than hard-coded local experiment directories.

## License

Add your preferred license before publishing. No license is included here so
the repository does not accidentally grant rights you did not intend to grant.
