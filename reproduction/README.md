# Paper-result reproduction

This directory contains the migrated unified comparison evaluator and the
legacy three-dataset ablation evaluator. It works with the corresponding
training modules in `../benchmarks/` and does not train or alter checkpoints.

Large weights are intentionally excluded from Git. Download the released
inference-only package and construct the expected artifact tree, or set:

```bash
export WDTF_ARTIFACT_ROOT=/path/to/checkpoint_tree
export WDTF_REPRO_OUTPUT=/path/to/new_outputs
```

Then run one comparison item at a time:

```bash
cd reproduction
python paper_compare_single.py --dataset Sentinel --method "WDTF-Net"
```

The unified metric schema is `mIoU`, Water IoU, Water F1, Accuracy, Water
Recall, and Water Precision. The current-paper Sentinel Full result must use
the `sentinel_main` dual-stage bundle from the released manifest, never the
deprecated legacy `sentinel_ablation_full` artifact.
