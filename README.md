# HAMMER + CRAF: Decision-Level Multimodal Correction on a Frozen HAMMER

YZM5110 Multimodal Learning research project by Muhammed Emir Koçak.

The project has two parts:

1. **Reproduction.** HAMMER, a detector of manipulated news image–text pairs, is
   re-trained on DGM4 from ALBEF initialisation and evaluated on all four tasks.
2. **Improvement.** CRAF is a small residual head (0.92 M parameters) trained on the
   frozen HAMMER features. It adds a zero-initialised correction to the binary logits.
   Version 2 adds HAMMER's own confidence as an input and a margin penalty, evaluated
   in a 2×2 ablation.

- **Report:** [`reports/Kocak_YZM5110_Research.pdf`](reports/Kocak_YZM5110_Research.pdf)
  (English, submitted version). A Turkish version and editable Word files are in
  [`reports/`](reports/).
- **Reference paper:** [Detecting and Grounding Multi-Modal Media Manipulation (CVPR 2023)](https://arxiv.org/abs/2304.02556)
- **HAMMER source:** [rshaojimmy/MultiModal-DeepFake](https://github.com/rshaojimmy/MultiModal-DeepFake),
  target commit `33fc04f3a1f550f923848737c864eea32556f699`
- **Dataset:** [DGM4](https://huggingface.co/datasets/rshaojimmy/DGM4)

## Main results

Test split, 50,705 image–text pairs. The values below are copied from the original run
files in `results/server_records/` and checked by `scripts/check_report_numbers.py`.

| Model | Accuracy (%) | AUC (%) | Cross-entropy |
|---|---:|---:|---:|
| HAMMER, our re-training (frozen baseline) | 85.343 | 92.666 | 0.5261 |
| HAMMER + CRAF v1, lr 1e-4 | 85.514 | 92.861 | 0.5631 |
| HAMMER + CRAF v1, lr 1e-5 | 85.534 | 92.859 | 0.5647 |
| HAMMER + CRAF v2, combined | 85.416 | 92.747 | 0.5304 |

For reference, the original paper reports 93.19% AUC and 86.39% accuracy for HAMMER.

- **Ranking gain:** CRAF v1 raises AUC by 0.19–0.20 points (paired-bootstrap 95%
  intervals 0.15–0.25 and 0.14–0.24 for the two v1 runs).
- **Probability cost:** every CRAF variant has a higher cross-entropy than HAMMER,
  because its remaining errors become more confident.
- **Limits:** all runs use a single seed, and the v1 test results influenced the design
  of v2, so the comparisons are exploratory.

See Sections 7–8 of the report for the full analysis.

## Repository structure

```text
.
├── gpu/                    # HAMMER/CRAF code, launchers and CPU tests (single canonical copy)
├── scripts/                # Setup, analysis, verification and number-check scripts
├── docs/                   # GPU guide, literature matrix, reading guide, source verification, licence
├── results/
│   ├── server_records/     # Original configs, histories, metrics, manifests and test predictions
│   ├── user_reported/      # Historical transcriptions and their fingerprints
│   └── *.json, *.md        # Analysis outputs (summaries, bootstrap, calibration, number check)
├── reports/
│   ├── Kocak_YZM5110_Research{,_TR}.pdf    # Report (English / Turkish)
│   ├── Kocak_YZM5110_Research{,_TR}.docx   # Editable Word versions
│   └── build/              # HTML source, figures and build scripts
├── requirements-cpu.txt    # Dependencies for CPU tests and analyses
├── MANIFEST.sha256         # Integrity manifest for the distributed files
├── NOTICE.md               # Attribution and third-party terms
└── AI_USE.md               # AI use disclosure
```

The [evidence index](results/SERVER_EVIDENCE_INDEX.json) maps the original server file
names to the files kept in `results/server_records/`. Paths starting with `hammer-craf/`
or `hammer-craf-v2/` in older audit documents refer to those original server folders.

Model and training source files keep their original bytes, so the checkpoint and
cache source-hash checks remain valid (`gpu/SOURCE_SHA256.json`). V1 and V2 use separate
checkpoint formats and settings and share the cache-loading and metric helpers.

**The dataset, model weights and feature `.pt` shards are not included.** Their SHA-256
identities are recorded in the run reports and manifests.

## Quick start: CPU tests and record analysis

Run commands from the repository root. Python 3.10 and a virtual environment are
recommended. This section needs neither the dataset, the official HAMMER source nor CUDA.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt

# Model, cache, training/resume and evaluation tests on synthetic features
OMP_NUM_THREADS=2 python -m unittest discover -s gpu -p 'test_*craf*.py' -v
python -m unittest discover -s scripts -p 'test_*.py' -v

# File, hash, syntax and archived-result integrity checks
python scripts/verify_package.py
```

## Reproduce the report numbers (CPU only, no model is run)

```bash
python scripts/analyze_hammer_training.py      # HAMMER re-training record
python scripts/analyze_server_results.py       # CRAF runs, paired errors, McNemar tests
python scripts/summarize_results.py --check-only
python scripts/bootstrap_uncertainty.py --reps 2000 --seed 0   # about 2 minutes
python scripts/calibration_analysis.py
python scripts/check_report_numbers.py         # expected: 116/116 checks passed
```

The first two scripts record environment versions in their JSON output, so re-running
them can change file hashes without changing any result. Numbers taken from other papers
are documented with their source tables in
[`docs/EXTERNAL_SOURCE_VERIFICATION.md`](docs/EXTERNAL_SOURCE_VERIFICATION.md).
How to rebuild the report PDF and Word files is described in
[`docs/REPORT_GENERATION.md`](docs/REPORT_GENERATION.md).

## GPU setup

Full HAMMER training and feature extraction with these launchers require Linux, CUDA
and four GPUs; the reported runs used 4 × RTX A6000. CRAF training can use an existing
cache on a single GPU. Provide the official source, DGM4 and the ALBEF or trained HAMMER
weights separately.

With an existing working environment:

```bash
export PYTHON=python
export REPO_DIR=/absolute/path/to/MultiModal-DeepFake
export DATA_ROOT=/absolute/path/to/datasets  # parent directory containing DGM4/
bash scripts/setup_gpu.sh --check
bash gpu/run_project.sh --help
```

To create a new Linux/CUDA environment with Python 3.10:

```bash
export REPO_DIR="$PWD/official-hammer"
VENV_DIR="$PWD/.venv-gpu" bash scripts/setup_gpu.sh --new-venv --clone-official
export PYTHON="$PWD/.venv-gpu/bin/python"
bash scripts/setup_gpu.sh --check
```

Setup installs PyTorch 2.5.1 / torchvision 0.20.1 for CUDA 12.1 and the pinned direct
dependencies in `gpu/requirements_gpu_pinned.txt`. It never overwrites an existing
environment or source repository, and it does not download the dataset or weights or
start training. Keep the official source repository outside Git.

## Run CRAF with an existing cache

`CACHE_DIR` must contain `train.manifest.json`, `val.manifest.json`,
`test.manifest.json` and the `.pt` shards they reference. Use a new output directory
for each experiment.

```bash
export PYTHON="$PWD/.venv-gpu/bin/python"  # or Python from your working environment
export CACHE_DIR=/absolute/path/to/craf_features
export RESULTS_ROOT="$PWD/gpu_runs"
export HEAD_GPU=0
export OUTPUT_DIR="$RESULTS_ROOT/craf_v1_lr1e-5_seed42"

CRAF_LR=1e-5 SEED=42 EPOCHS=30 PATIENCE=8 bash gpu/run_project.sh v1-train

# V2 control, penalty, decision and combined, one after another
CRAF_LR=1e-5 CORRECTION_LAMBDA=0.01 SEED=42 EPOCHS=30 \
  bash gpu/run_project.sh v2-variants all

# Evaluate V1 on the test split
mkdir -p "$RESULTS_ROOT/evaluation"
CRAF_CHECKPOINT="$OUTPUT_DIR/best_craf.pth" \
TEST_OUTPUT="$RESULTS_ROOT/evaluation/v1_test.json" \
  bash gpu/run_project.sh v1-evaluate

# Evaluate V2 combined on the test split
CRAF_CHECKPOINT="$RESULTS_ROOT/craf_v2_combined_seed42/best_craf.pth" \
TEST_OUTPUT="$RESULTS_ROOT/evaluation/v2_combined_test.json" \
  bash gpu/run_project.sh v2-evaluate
```

Evaluation writes JSON metrics and a `_predictions.jsonl` file. To build a feature
cache, and for full HAMMER training, testing and resume commands, see the
[GPU guide](docs/GPU_USAGE.md). Use each checkpoint with its matching V1/V2 evaluator
and do not bypass the cache or source-hash checks.

## Before pushing to GitHub

`.gitignore` excludes environments, secret configuration files, weights, datasets,
feature caches and new experiment outputs.

```bash
python scripts/verify_package.py
# After intentional changes to sources or documents, refresh the manifest:
python scripts/verify_package.py --write-manifest
python scripts/verify_package.py
git status --short
```

Refreshing the manifest only updates the distribution manifest; it does not disable the
hash checks for preserved model sources or server evidence.

## Attribution, licensing and AI use

HAMMER is subject to the S-Lab License 1.0 and its non-commercial-use terms. See the
[third-party licence](docs/THIRD_PARTY_SLAB_LICENSE.txt), [NOTICE](NOTICE.md) and the
[bibliography](docs/references.bib). No MIT or Apache licence is claimed for the project
as a whole. DGM4 and the model weights are subject to their own access and usage terms.
AI assistance is disclosed in [AI_USE.md](AI_USE.md).
