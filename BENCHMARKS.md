# Comparative-experiment release guide

Keep comparison methods in a separate `benchmarks/` release or Git submodules,
not inside the minimal WDTF-Net package. This keeps the proposed-method code
small and preserves the licensing/attribution requirements of third-party
implementations.

For each reported method, release or cite the exact implementation used, its
license, the training configuration, the selected checkpoint, and a test entry
that emits one common JSON/CSV metric schema:

```text
mIoU, water_iou, water_f1, accuracy, water_recall, water_precision
```

The paper comparison includes U2PL, UniMatch V2, RankMatch, CPS, MPF, ReCo,
AGMM-SASS, CC4S, WSSS-PCRE, ParaFormer, fine-tuned SAM ViT-B, and WDTF-Net.
For every method, use the same released train/validation/test partition, weak
masks, input-band order, normalization divisor, threshold (0.5), and final
test-only HR labels. Do not redistribute an official repository's files unless
its license permits this; pin its commit/version and link to the upstream
source instead.

The proposed-method controlled ablations are included directly in `ablation.py`
because they only modify WDTF-Net modules. They support `full`, `no_wgdc`,
`no_fg`, and `no_wa` and use the same paper-aligned protocol as `train.py`.
