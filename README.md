# HAMMER + CRAF

A YZM5110 research project on manipulation detection in news containing images
and text. CRAF uses frozen HAMMER features to learn a residual correction to
binary classification logits. V2 provides four experimental variants for
decision conditioning and a correction penalty.

Reference paper: [Detecting and Grounding Multi-Modal Media Manipulation (CVPR 2023)](https://arxiv.org/abs/2304.02556).
HAMMER source: [MultiModal-DeepFake](https://github.com/rshaojimmy/MultiModal-DeepFake).
Dataset: [DGM4](https://huggingface.co/datasets/rshaojimmy/DGM4).

## Repository structure

```text
.
├── gpu/                    # HAMMER/CRAF code, launchers, and CPU tests
├── scripts/                # Setup, analysis, report, and integrity tools
├── docs/                   # Usage, methodology, literature, and license documents
├── results/
│   ├── server_records/     # Original experiment configs, histories, metrics, manifests, predictions
│   └── user_reported/      # Historical transcription hashes and a unique epoch excerpt
├── reports/                # Research reports in Markdown and PDF
├── qa/                     # Previous audits and repository cleanup records
├── requirements-cpu.txt    # Dependencies for CPU tests and numerical analysis
├── MANIFEST.sha256         # Integrity manifest for distributed files
├── NOTICE.md               # Attribution and third-party terms
└── AI_USE.md               # AI usage disclosure
```

The single canonical copy of the model and training code lives in `gpu/`.
The [evidence index](results/SERVER_EVIDENCE_INDEX.json) maps original server
filenames to their retained files. Paths beginning with `hammer-craf/` or
`hammer-craf-v2/` in older audits refer to the original imported directories;
these directories are no longer needed at the repository root.

Model and training source files retain their original bytes so checkpoint and
cache source-hash checks remain valid. V1 and V2 use separate checkpoint formats
and experimental settings, while sharing cache-loading and metric helpers.
Similar layers in the two heads are preserved for compatibility with existing
source hashes.

**The dataset, model weights, and feature `.pt` shards are not included.**
Archived manifests do not replace these files. Archived server documents contain
historical commands; use the commands below and the GPU guide for current usage.

## Quick start: CPU tests and record analysis

Run commands from the repository root. Python 3.10 and a virtual environment
are recommended. This section does not require the dataset, the official HAMMER
source repository, or CUDA.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt

# Model, cache, training/resume, and evaluation tests using synthetic features
OMP_NUM_THREADS=2 python -m unittest discover -s gpu -p 'test_*craf*.py' -v
python -m unittest discover -s scripts -p 'test_*.py' -v

# File, hash, syntax, and archived-result integrity checks
python scripts/verify_package.py

# Reanalyze archived results without running a model
python scripts/analyze_hammer_training.py
python scripts/analyze_server_results.py
python scripts/summarize_results.py
python scripts/summarize_results.py --check-only
```

Outputs are written to `results/HAMMER_TRAINING_SUMMARY.md`,
`results/SERVER_SUMMARY.md`, `results/SUMMARY.md`, and the corresponding JSON
analysis files. Analysis JSON also records environment versions, so rerunning
an analysis can change file hashes. `verify_package.py` uses only the Python
standard library and can run without NumPy or PyTorch.
CPU tests do not validate actual DGM4 performance or GPU training.

## GPU setup

Full HAMMER training and feature extraction with these launchers require Linux,
CUDA, and four GPUs. CRAF training can use an existing cache on a single GPU.
The target commit of the official HAMMER repository is
`33fc04f3a1f550f923848737c864eea32556f699`.
Provide the official source, DGM4, and the required ALBEF or trained HAMMER
weights separately.

If you already have a working environment, use it:

```bash
export PYTHON=python
export REPO_DIR=/absolute/path/to/MultiModal-DeepFake
export DATA_ROOT=/absolute/path/to/datasets  # Parent directory containing DGM4/
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

Setup installs PyTorch 2.5.1 / torchvision 0.20.1 for CUDA 12.1 and pinned direct
dependencies. It refuses to overwrite an existing environment or source
repository. It does not download the dataset or weights, or start training.
You can also keep the official source repository outside this project;
do not add a second copy of it to Git.

## Run CRAF with an existing cache

`CACHE_DIR` must contain `train.manifest.json`, `val.manifest.json`,
`test.manifest.json`, and the actual `.pt` shards referenced by those manifests.
Choose a new output directory for each new experiment.

```bash
export PYTHON="$PWD/.venv-gpu/bin/python"  # Or Python from your working environment
export CACHE_DIR=/absolute/path/to/craf_features
export RESULTS_ROOT="$PWD/gpu_runs"
export HEAD_GPU=0
export OUTPUT_DIR="$RESULTS_ROOT/craf_v1_lr1e-5_seed42"

CRAF_LR=1e-5 SEED=42 EPOCHS=30 PATIENCE=8 bash gpu/run_project.sh v1-train

# Run V2 control, penalty, decision, and combined sequentially
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

Evaluation produces JSON metrics and a `_predictions.jsonl` file. If you do not
have a test cache, first follow the `extract` step in the
[GPU guide](docs/GPU_USAGE.md). That guide also covers full HAMMER training,
testing, and resume commands. Use each existing checkpoint with its matching
V1/V2 evaluator; do not bypass cache or source-hash checks.


## Checks before pushing to GitHub

`.gitignore` excludes environments, secret configuration files, weights, large
datasets, new experiment outputs, and generated page previews. Small archived
experiment files and report PDFs remain in the repository.

```bash
python scripts/verify_package.py
# Refresh the manifest after intentional source or documentation changes:
python scripts/verify_package.py --write-manifest
python scripts/verify_package.py

git status --short
```

Refreshing the manifest only updates the distribution manifest; it does not
disable the hash checks for preserved model sources or server evidence.
See [qa/REPOSITORY_CLEANUP.md](qa/REPOSITORY_CLEANUP.md) for cleanup details.

## Attribution and licensing

HAMMER is subject to S-Lab License 1.0 and its noncommercial-use terms.
Read the [third-party license](docs/THIRD_PARTY_SLAB_LICENSE.txt),
[NOTICE](NOTICE.md), and [bibliography](docs/references.bib). This repository
does not claim an MIT or Apache license for the entire project. DGM4 and model
weights are subject to their own access and usage terms.
AI usage is disclosed in [AI_USE.md](AI_USE.md). Outstanding course-submission
fields are retained in [TESLIM_KONTROLU.md](TESLIM_KONTROLU.md).
