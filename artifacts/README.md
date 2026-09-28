# Portable experiment artifacts

The complete `qwen38_27b_harness_v4` artifact tree is stored as eight ordinary
Git files named `qwen38_27b_harness_v4.tar.zst.part-00` through `part-07`.
Each part is at most 90,000,000 bytes, below GitHub's 100 MB regular-file limit;
Git LFS is not used.

Restore the tree from the repository root with:

```bash
bash scripts/restore_qwen38_27b_harness_v4.sh
```

The script verifies the SHA-256 of the concatenated archive before extracting
it to `artifacts/qwen38_27b_harness_v4`. It refuses to overwrite an existing
path. An alternate destination may be passed as its first argument.

The archive is a dereferenced copy of the full v4 checkpoint artifact: 44,941
files and 10,078,848,252 uncompressed payload bytes. It includes the complete
5,535-run solve output, checkpoints, logs, prepared manifests, Slurm records,
and source snapshot. Model weights are not part of this artifact tree.

Prepared model manifests contain machine-local absolute paths. Before judging
from another machine, regenerate those paths and the required tokenizers with
the exact code and config from this commit:

```bash
uv run value-as-tool --config experiment_qwen38_27b_harness.yaml prepare --tokenizers-only
uv run value-as-tool --config experiment_qwen38_27b_harness.yaml status
```
