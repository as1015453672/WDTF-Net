# Released artifacts (not tracked by Git)

Put downloaded model artifacts below this directory, preserving the checkpoint
paths selected by the reproduction configuration, or set `WDTF_ARTIFACT_ROOT`
to another location. Do not commit checkpoints to the source repository.

The authoritative mapping between paper rows and inference-only weights is the
`manifest.json` produced in the private experiment workspace. Publish that JSON
with the model release (Zenodo, Hugging Face, or GitHub Release) after replacing
local absolute source paths with release-relative paths.

For the revised paper, use `wdtf_dual_stage/sentinel_main.pth` for the Sentinel
Full WDTF result. The legacy `sentinel_ablation_full.pth` artifact is retained
only for audit and must not be referenced by current-paper reproduction scripts.
