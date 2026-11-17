#!/usr/bin/env bash
set -euo pipefail

CRAF_V2_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CACHE_DIR="${CACHE_DIR:-$CRAF_V2_DIR/craf_features}"
RESULTS_ROOT="${RESULTS_ROOT:-$CRAF_V2_DIR}"
PYTHON="${PYTHON:-python}"
HEAD_GPU="${HEAD_GPU:-0}"
CRAF_LR="${CRAF_LR:-1e-5}"
CORRECTION_LAMBDA="${CORRECTION_LAMBDA:-0.01}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-30}"
PATIENCE="${PATIENCE:-8}"
HEAD_BATCH="${HEAD_BATCH:-512}"

run_variant() {
  local variant="$1"
  local penalty="0"
  local -a extra=(--device cuda)
  case "$variant" in
    control) ;;
    penalty) penalty="$CORRECTION_LAMBDA" ;;
    decision) extra+=(--condition-on-decision) ;;
    combined) penalty="$CORRECTION_LAMBDA"; extra+=(--condition-on-decision) ;;
    *) echo "Unknown variant: $variant" >&2; exit 1 ;;
  esac
  local result="$RESULTS_ROOT/craf_v2_${variant}_seed${SEED}"
  echo "Starting $variant -> $result (lr=$CRAF_LR, lambda=$penalty, seed=$SEED)"
  CUDA_VISIBLE_DEVICES="$HEAD_GPU" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
    "$PYTHON" "$CRAF_V2_DIR/hammer_craf_v2_server.py" train \
    --cache-dir "$CACHE_DIR" --output-dir "$result" \
    --lr "$CRAF_LR" --correction-lambda "$penalty" \
    --epochs "$EPOCHS" --patience "$PATIENCE" --batch-size "$HEAD_BATCH" \
    --selection-metric auc --seed "$SEED" "${extra[@]}"
}

case "${1:-help}" in
  all)
    for variant in control penalty decision combined; do run_variant "$variant"; done
    ;;
  control|penalty|decision|combined) run_variant "$1" ;;
  *)
    echo "Usage: bash run_craf_v2_experiments.sh {all|control|penalty|decision|combined}"
    echo "Reuses existing cache. No HAMMER extraction and no automatic test evaluation."
    ;;
esac
