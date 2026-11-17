# GPU usage guide

Run commands from the package root (the directory containing `gpu/`, `scripts/`,
and `docs/`). Official HAMMER sources are distributed under S-Lab License 1.0
for noncommercial use; read `NOTICE.md` and `docs/THIRD_PARTY_SLAB_LICENSE.txt`.
This package does not define a new backbone model. Production Python source
bytes are unchanged. Local verification uses synthetic CPU tests; CUDA and
four-GPU execution have not been verified locally.

## First choice: reuse your existing environment and cache

Do not reinstall a working environment. The following check installs or downloads nothing:

```bash
export PYTHON=python
bash scripts/setup_gpu.sh --check
bash gpu/run_project.sh --help
```

Common server defaults and configurable paths:

```bash
export REPO_DIR=/mnt/sdb/home/mekocak/projects/MultiModal-DeepFake
export DATA_ROOT=/mnt/sdb/home/mekocak/projects/datasets
export CONFIG="$REPO_DIR/configs/train.yaml"
export GPU_IDS=0,1,2,3
export HEAD_GPU=0
export TEXT_ENCODER=bert-base-uncased
# Replace these values with your ACTUAL existing cache and result paths.
export CACHE_DIR="$PWD/craf_features"
export RESULTS_ROOT="$PWD/gpu_runs"
```

`CACHE_DIR` contains the existing `train.manifest.json`, `val.manifest.json`,
`test.manifest.json`, and the `.pt` shards they reference. Use the example path
only if the actual cache is there; there is no need to copy/move a large existing
cache or restart extraction. V1/v2 training reads only train/val caches; the same
HAMMER cache can be used for v2. Cache integrity, backbone identity, and disjoint
news IDs are checked. Keep train/val manifests in the same directory for test
evaluation as well. Missing or mismatched caches cause an error; extraction is
not triggered automatically.

## V1: a low learning rate with an existing cache

Choose a new `OUTPUT_DIR` for a new experiment. This command does not run the backbone:

```bash
export OUTPUT_DIR="$RESULTS_ROOT/craf_v1_lr1e-5_seed42"
CRAF_LR=1e-5 SEED=42 EPOCHS=30 PATIENCE=8 bash gpu/run_project.sh v1-train
```

The original `run_craf_server.sh train` invokes train/val extraction, and `test`
invokes test extraction. For users with an existing cache, the recommended
commands are `v1-train` and `v1-evaluate` here. The original launcher is preserved;
`head` does not extract, but its default LR is `1e-4`, so explicitly use this
guide's `1e-5` setting through the wrapper.

Resume the same interrupted experiment from `last_craf.pth`:

```bash
CRAF_LR=1e-5 SEED=42 EPOCHS=40 PATIENCE=8 bash gpu/run_project.sh v1-resume
```

Keep the same `OUTPUT_DIR`, cache, LR, seed, batch, head, ablation, and selection
settings. Check `OUTPUT_DIR/run.json` and the checkpoint config first. The epoch
target may be increased; these example settings apply only to the new run above.
Do not resume an old `1e-4` run with `1e-5`; choose a new output directory.

## V2: four variants using the same cache

```bash
CRAF_LR=1e-5 CORRECTION_LAMBDA=0.01 SEED=42 EPOCHS=30 \
  bash gpu/run_project.sh v2-variants all
```

Four runs execute sequentially on one GPU, without extraction or automatic testing:

| Variant | Decision conditioning | Correction penalty | Output directory |
| --- | --- | --- | --- |
| control | no | 0 | `$RESULTS_ROOT/craf_v2_control_seed42` |
| penalty | no | 0.01 | `$RESULTS_ROOT/craf_v2_penalty_seed42` |
| decision | yes | 0 | `$RESULTS_ROOT/craf_v2_decision_seed42` |
| combined | yes | 0.01 | `$RESULTS_ROOT/craf_v2_combined_seed42` |

Replace `all` with a variant name to launch one run. A new seed produces a
different output name. Do not start a new run in an existing output directory.
Example combined resume:

```bash
export OUTPUT_DIR="$RESULTS_ROOT/craf_v2_combined_seed42"
CRAF_LR=1e-5 SEED=42 EPOCHS=40 PATIENCE=8 \
  bash gpu/run_project.sh v2-resume \
  --condition-on-decision --correction-lambda 0.01
```

For control resume, omit both extra arguments; for penalty, use only
`--correction-lambda 0.01`; for decision, use only `--condition-on-decision`.
The actual existing run's `run.json` settings always take precedence over examples.

## V1 / V2: explicit evaluation using an existing test cache

Run the test after completing model/variant selection on validation data. These
commands explicitly pass `--split test` and do not extract features. Choose a new
output filename for each evaluation. Create its parent directory first:

```bash
mkdir -p "$RESULTS_ROOT/evaluation"
CRAF_CHECKPOINT="$RESULTS_ROOT/craf_v1_lr1e-5_seed42/best_craf.pth" \
TEST_OUTPUT="$RESULTS_ROOT/evaluation/v1_test.json" \
  bash gpu/run_project.sh v1-evaluate

# The exact combined directory name produced by the v2 launcher above:
CRAF_CHECKPOINT="$RESULTS_ROOT/craf_v2_combined_seed42/best_craf.pth" \
TEST_OUTPUT="$RESULTS_ROOT/evaluation/v2_combined_test.json" \
  bash gpu/run_project.sh v2-evaluate
```

V2 testing does not require variant flags; head settings are loaded from the
checkpoint. Evaluation creates a JSON report and a matching `_predictions.jsonl`
file. If the cache lacks a test split, first locate your actual existing test
cache; do not repeat existing extraction merely to run training or evaluation.

## Full HAMMER training from ALBEF initialization (only if a new baseline is needed)

The official source repository is required: `train.py`, `models/HAMMER.py`,
`configs/train.yaml`, the BERT JSON referenced by the config, and other official
modules. Expected commit: `33fc04f3a1f550f923848737c864eea32556f699`.
Do not modify these sources. Supply ALBEF weights beforehand; ALBEF is not a
trained HAMMER checkpoint.

```bash
export ALBEF_CHECKPOINT="$REPO_DIR/ALBEF_4M.pth"
export OUTPUT_DIR="$RESULTS_ROOT/hammer_4gpu"
# Preflight: metadata/path checks only; does not run a GPU.
"$PYTHON" gpu/train_hammer_4gpu.py --repo-dir "$REPO_DIR" \
  --config "$CONFIG" --data-root "$DATA_ROOT" --checkpoint "$ALBEF_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" --batch-size 32 --epochs 50 --check-only
# Explicit training action; requires four CUDA GPUs and sufficient memory.
bash gpu/run_project.sh hammer-train --batch-size 32 --val-batch-size 32 --epochs 50
```

The trainer uses the six official losses and training flow. Batch size is per
GPU; the queue size must be divisible by the global batch. Preflight does not
verify images, weight compatibility, or CUDA. Training writes `config.yaml`,
`checkpoint_last.pth`, `checkpoint_best.pth`, and logs to the output directory.
The initial `CONFIG` path may be relative to the repository root or absolute;
the wrapper default is absolute. Keep the recorded config and checkpoint
together. HAMMER resume:

```bash
export HAMMER_RESUME="$OUTPUT_DIR/checkpoint_last.pth"
bash gpu/run_project.sh hammer-resume --batch-size 32 --val-batch-size 32 --epochs 50
```

**HAMMER resume limitation:** This existing production code loads the
model/optimizer/scheduler/epoch; it does not automatically compare new settings
against the checkpoint config or preserve per-rank RNG state. Exact stochastic
continuation is not guaranteed. It also rewrites `config.yaml` and `run_args.json`
in the output directory. Before resuming, copy these two files and manually
compare the command with the previous run's settings. Do not present HAMMER
resume as offering the same guarantees as CRAF's stricter hash/RNG checks.
This package preserves production Python source bytes for compatibility.

Keep the previous model/config/batch settings. `GPU_IDS` must expose four
devices; the wrapper starts four processes. This package has not been tested
locally on four GPUs.

## Standalone full HAMMER test (not feature extraction)

This path runs the complete official DGM4 test split on one GPU using a trained
HAMMER checkpoint. It requires neither CRAF nor extraction; the result file
must be new:

```bash
export HAMMER_CHECKPOINT="$RESULTS_ROOT/hammer_4gpu/checkpoint_best.pth"
mkdir -p "$RESULTS_ROOT/evaluation"
TEST_OUTPUT="$RESULTS_ROOT/evaluation/hammer_fulltest.json" \
  bash gpu/run_project.sh hammer-test --check-only
TEST_OUTPUT="$RESULTS_ROOT/evaluation/hammer_fulltest.json" \
  bash gpu/run_project.sh hammer-test
```

Preflight writes no output. The config must match the checkpoint's training settings.

## Frozen HAMMER extraction: only when the cache is genuinely absent

Preserve the existing cache. Choose a separate cache directory for a new HAMMER
checkpoint/config/preprocessing setup. Trained HAMMER is required; ALBEF is not
an extraction input.

```bash
export HAMMER_CHECKPOINT="$RESULTS_ROOT/hammer_4gpu/checkpoint_best.pth"
export CACHE_DIR="$PWD/craf_features_new_backbone"
EXTRACT_BATCH=32 WORKERS=4 bash gpu/run_project.sh extract --splits train val
# Separate explicit action only if the test cache is absent and model selection is complete:
EXTRACT_BATCH=32 WORKERS=4 bash gpu/run_project.sh extract --splits test
```

Extraction uses a frozen/eval backbone in four processes and records cache
identities and hashes. Subsequent head training uses one GPU. The original
extractor's shard checks are preserved; avoid unnecessary repeat extraction.

## Optional clean CUDA environment / data preparation

For an existing working environment, use only `--check`. If a new installation is needed:

```bash
VENV_DIR="$PWD/.venv-gpu" bash scripts/setup_gpu.sh --new-venv
export PYTHON="$PWD/.venv-gpu/bin/python"
bash scripts/setup_gpu.sh --check
```

This uses Python 3.10, torch 2.5.1 / torchvision 0.20.1 CUDA 12.1 wheels, and
the direct dependency pins in `gpu/requirements_gpu_pinned.txt` (NumPy 1.26.4).
It is not a complete transitive lock. Creating a new environment requires
network access and will not overwrite an existing venv. It does not overwrite
or modify an existing official repository. Only if the official repository is
also absent, explicitly add `--clone-official` to the new-venv command:

```bash
VENV_DIR="$PWD/.venv-gpu-new" bash scripts/setup_gpu.sh --new-venv --clone-official
```

The clone checks out the specified official commit. It does not download the
dataset or ALBEF. If full DGM4 already exists, verify it; downloading occurs
only through the explicit `download-data` action:

```bash
bash gpu/run_project.sh download-data --verify-only
bash gpu/run_project.sh download-data --plan-only
# Only if you want to download the missing full dataset:
bash gpu/run_project.sh download-data
```

`TEXT_ENCODER` may point to a local tokenizer/encoder snapshot directory. For
offline use, supply required Hugging Face files in advance and, if needed, set
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`. The package does not secretly download
weights; official `from_pretrained` loading may request network access when its
cache is absent.

## Source integrity and CPU checks

`gpu/SOURCE_SHA256.json` lists files copied exactly from original outputs.
In particular, do not reformat the bytes of `craf_residual.py` or
`craf_residual_v2.py`: old checkpoints require the original head hashes.
Trainer/extractor sources are also unchanged for provenance; pass paths and
hyperparameters through the wrapper/CLI. No separate `trainhelper` source
file exists; shared training helpers are in `train_hammer_4gpu.py`.
Use the correct original head and the cache/config matching the checkpoint
instead of bypassing hash checks.

Portable synthetic CPU tests (not benchmark or CUDA tests):

```bash
(cd gpu && "$PYTHON" -m unittest -v test_hammer_craf_server \
  test_craf_residual_v2 test_hammer_craf_v2_server)
```

Tests create small synthetic caches in temporary directories and do not touch
existing real caches. Actual four-GPU training, extraction throughput, and DGM4
performance must be verified separately from these local packaging checks.

## November 13 server-record note

Actual CRAF run settings and histories are under `results/server_records/`;
code comparisons are in `docs/SERVER_CODE_AUDIT.md`. The supplied
directories contain cache manifests but no `.pt` shards or `.pth` weights.
Those server files must be supplied separately for reruns. Run records do not
replace a complete shell history or environment inventory. The separately
supplied HAMMER config/run_args, 11-epoch training log, and test report are under
`results/server_records/hammer_training/`.
