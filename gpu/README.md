# HAMMER / CRAF GPU code package

Start with [GPU_KULLANIM.md](../docs/GPU_KULLANIM.md). All commands there run
from the package root (the directory containing `gpu`, `scripts`, `docs`).

Production Python files, portable tests, both original launchers and original
requirements are copied byte for byte from the existing outputs. See
`SOURCE_SHA256.json` for relative source names and hashes. Shared training/import
helpers are contained in `train_hammer_4gpu.py`; no separate `trainhelper` source
file was present. The wrapper dispatches these scripts without a new model.

Use `bash gpu/run_project.sh --help`. No-argument invocation only prints help.
Existing feature caches should be reused by `v1-train`, `v2-variants`,
`v1-evaluate`, `v2-evaluate`. Evaluation requires an existing test manifest and
shards and never calls extraction. Original `run_craf_server.sh train` invokes
train/val extraction and `test` invokes test extraction; these are retained for
provenance, not recommended as the default cache-reuse workflow.

`craf_residual.py` and `craf_residual_v2.py` must retain their original bytes:
old checkpoints check source hashes. Preserve the original matching head,
trainer, extractor, manifests and shards; renaming/moving cache content can
also invalidate extraction identities. Do not edit code to bypass checks.

Official HAMMER source is governed by **S-Lab License 1.0, noncommercial**.
This package does not declare that code MIT. Consult `../NOTICE.md`, `../docs/THIRD_PARTY_SLAB_LICENSE.txt` and the official repository.

Local verification is CPU-only using synthetic fixtures, not a 4-GPU execution
or benchmark validation. Optional setup creates a new environment only when
explicitly requested; no local CUDA installation is implied.
