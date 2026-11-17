#!/usr/bin/env bash
set -euo pipefail
PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${1:---help}"
if [[ $# -gt 0 ]]; then shift; fi
CLONE=0
for arg in "$@"; do
  case "$arg" in --clone-official) CLONE=1 ;; *) echo "Unknown flag: $arg" >&2; exit 2 ;; esac
done
PYTHON="${PYTHON:-python}"
REPO_DIR="${REPO_DIR:-/mnt/sdb/home/mekocak/projects/MultiModal-DeepFake}"
case "$MODE" in
  --check)
    [[ "$CLONE" == 0 ]] || { echo '--check never clones; use --new-venv --clone-official.' >&2; exit 2; }
    "$PYTHON" - <<'PY'
import importlib
import sys
print('Python:', sys.executable, sys.version)
for name in ('torch', 'torchvision', 'numpy', 'transformers', 'timm', 'PIL',
             'scipy', 'sklearn', 'yaml', 'cv2', 'tensorboard', 'tqdm'):
    module = importlib.import_module(name)
    print(name, getattr(module, '__version__', 'import OK'))
import torch
print('CUDA runtime:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available(), 'visible GPUs:', torch.cuda.device_count())
PY
    ;;
  --new-venv)
    : "${VENV_DIR:?Set VENV_DIR to a NEW environment directory}"
    [[ ! -e "$VENV_DIR" ]] || { echo "Refusing to overwrite: $VENV_DIR" >&2; exit 2; }
    if [[ "$CLONE" == 1 && -e "$REPO_DIR" ]]; then
      echo "Refusing to overwrite existing repository: $REPO_DIR" >&2; exit 2
    fi
    "${PYTHON_BOOTSTRAP:-python3.10}" -m venv "$VENV_DIR"
    NEW_PYTHON="$VENV_DIR/bin/python"
    "$NEW_PYTHON" -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
    "$NEW_PYTHON" -m pip install -r "$PACKAGE_DIR/gpu/requirements_gpu_pinned.txt"
    "$NEW_PYTHON" -m pip check
    if [[ "$CLONE" == 1 ]]; then
      git clone https://github.com/rshaojimmy/MultiModal-DeepFake.git "$REPO_DIR"
      git -C "$REPO_DIR" checkout --detach 33fc04f3a1f550f923848737c864eea32556f699
      [[ "$(git -C "$REPO_DIR" rev-parse HEAD)" == 33fc04f3a1f550f923848737c864eea32556f699 ]]
    fi
    echo "Created environment. Use PYTHON=$VENV_DIR/bin/python; then run setup_gpu.sh --check."
    ;;
  --help|-h)
    cat <<'HELP'
Optional setup; never required for an existing working environment.
  PYTHON=python bash scripts/setup_gpu.sh --check
    Read-only import/CUDA inventory; installs and clones nothing.
  VENV_DIR="$PWD/.venv-gpu" bash scripts/setup_gpu.sh --new-venv
    Creates NEW Python 3.10 venv, CUDA 12.1 torch 2.5.1/torchvision 0.20.1,
    NumPy 1.26.4 and pinned direct dependencies; network required.
  Add --clone-official to --new-venv ONLY to request official source clone.
  REPO_DIR controls the destination; existing repo/venv is never overwritten.
No dataset, ALBEF, tokenizer weights, extraction or training is run by setup.
HELP
    ;;
  *) echo "Unknown mode: $MODE" >&2; exit 2 ;;
esac
