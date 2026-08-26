# Comparison-method training code

This directory contains the adapted training/inference code used for the
learning-based comparison methods: U2PL, UniMatch V2, RankMatch, CPS, MPF,
ReCo, AGMM-SASS, CC4S, WSSS-PCRE, and fine-tuned RiverSnap SAM. The local
`models/` directory contains only the shared modules needed by these scripts.

Run a trainer from this directory so its local imports resolve:

```bash
cd benchmarks
python train_u2pl.py --help
python train_wsss_pcre_weak.py --help
python train_riversnap_sam_weak.py --help
```

`train_riversnap_sam_weak.py` and `evaluate_riversnap_sam.py` additionally
require Meta's Segment Anything package and a ViT-B base checkpoint. The
official UniMatch-V2 and RankMatch variants used in the Sentinel audit remain
external upstream dependencies; retain their upstream repositories and licenses
as Git submodules under `third_party/official_sources/` before reproducing those
two specific rows.

All baseline scripts must use the released fixed split, weak masks, channel
order, normalization divisor, 0.5 decision threshold, and final test labels.

## Default paper protocol

The public training entry points default to the protocol used for the revised
comparison study: seed 42, AdamW, learning rate `1e-4`, weight decay `1e-4`,
batch size 8, an epoch budget of 60, and patience 5. U2PL, UniMatch V2,
RankMatch, CPS, MPF, ReCo, and AGMM-SASS default to ResNet-101 where their
implementation exposes a ResNet backbone selector. WSSS-PCRE and CC4S retain
their published method-specific encoder structures. CC4S retains its two-stage
schedule of 40 + 20 epochs (60 total); both stages use batch size 8 and AdamW.
Fine-tuned RiverSnap SAM trains only the ViT-B mask decoder with the same common
optimizer settings. WDTF-Net is documented separately in the repository root:
it retains the paper's two-stage `1e-4` / `1e-5` learning-rate schedule.

Every setting remains a command-line option. Existing released inference
artifacts are not overwritten or changed by these defaults.
