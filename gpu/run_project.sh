#!/usr/bin/env bash
set -euo pipefail
GPU_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
REPO_DIR="${REPO_DIR:-/mnt/sdb/home/mekocak/projects/MultiModal-DeepFake}"
DATA_ROOT="${DATA_ROOT:-/mnt/sdb/home/mekocak/projects/datasets}"
CONFIG="${CONFIG:-$REPO_DIR/configs/train.yaml}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
HEAD_GPU="${HEAD_GPU:-${GPU_IDS%%,*}}"
TEXT_ENCODER="${TEXT_ENCODER:-bert-base-uncased}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
require() { if [[ -z "${!1:-}" ]]; then echo "Set $1 explicitly." >&2; exit 2; fi; }
head_train() {
  require CACHE_DIR; require OUTPUT_DIR
  CUDA_VISIBLE_DEVICES="$HEAD_GPU" "$PYTHON" "$GPU_DIR/hammer_craf_server.py" train \
    --cache-dir "$CACHE_DIR" --output-dir "$OUTPUT_DIR" --lr "${CRAF_LR:-1e-5}" \
    --epochs "${EPOCHS:-30}" --batch-size "${HEAD_BATCH:-512}" --seed "${SEED:-42}" \
    --selection-metric auc --patience "${PATIENCE:-5}" --ablation "${ABLATION:-full}" "$@"
}
command="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$command" in
  hammer-train|hammer-resume)
    require OUTPUT_DIR
    if [[ "$command" == hammer-resume ]]; then
      require HAMMER_RESUME; init=(--resume "$HAMMER_RESUME")
    else
      require ALBEF_CHECKPOINT; init=(--checkpoint "$ALBEF_CHECKPOINT")
    fi
    CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON" -m torch.distributed.run \
      --standalone --nnodes=1 --nproc_per_node=4 "$GPU_DIR/train_hammer_4gpu.py" \
      --repo-dir "$REPO_DIR" --config "$CONFIG" --data-root "$DATA_ROOT" \
      --output-dir "$OUTPUT_DIR" --text-encoder "$TEXT_ENCODER" "${init[@]}" "$@"
    ;;
  hammer-test)
    require HAMMER_CHECKPOINT; require TEST_OUTPUT
    CUDA_VISIBLE_DEVICES="$HEAD_GPU" "$PYTHON" "$GPU_DIR/test_hammer_server.py" \
      --repo-dir "$REPO_DIR" --config "$CONFIG" --data-root "$DATA_ROOT" \
      --checkpoint "$HAMMER_CHECKPOINT" --output "$TEST_OUTPUT" --text-encoder "$TEXT_ENCODER" "$@"
    ;;
  extract)
    require HAMMER_CHECKPOINT; require CACHE_DIR
    CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON" -m torch.distributed.run \
      --standalone --nnodes=1 --nproc_per_node=4 "$GPU_DIR/hammer_craf_server.py" extract \
      --repo-dir "$REPO_DIR" --config "$CONFIG" --data-root "$DATA_ROOT" \
      --hammer-checkpoint "$HAMMER_CHECKPOINT" --cache-dir "$CACHE_DIR" \
      --text-encoder "$TEXT_ENCODER" --batch-size "${EXTRACT_BATCH:-32}" --workers "${WORKERS:-4}" "$@"
    ;;
  v1-train) head_train "$@" ;;
  v1-resume) head_train --resume "$@" ;;
  v2-variants)
    require CACHE_DIR; require RESULTS_ROOT
    export CACHE_DIR RESULTS_ROOT PYTHON HEAD_GPU
    variant="${1:-all}"
    [[ $# -le 1 ]] || { echo 'Pass one v2 variant or all.' >&2; exit 2; }
    case "$variant" in all|control|penalty|decision|combined) ;; *) echo "Unknown variant: $variant" >&2; exit 2 ;; esac
    exec bash "$GPU_DIR/run_craf_v2_experiments.sh" "$variant"
    ;;
  v2-train|v2-resume)
    require CACHE_DIR; require OUTPUT_DIR
    if [[ "$command" == v2-resume ]]; then set -- --resume "$@"; fi
    CUDA_VISIBLE_DEVICES="$HEAD_GPU" "$PYTHON" "$GPU_DIR/hammer_craf_v2_server.py" train \
      --cache-dir "$CACHE_DIR" --output-dir "$OUTPUT_DIR" --lr "${CRAF_LR:-1e-5}" \
      --epochs "${EPOCHS:-30}" --batch-size "${HEAD_BATCH:-512}" --seed "${SEED:-42}" \
      --patience "${PATIENCE:-8}" --selection-metric auc "$@"
    ;;
  v1-evaluate|v2-evaluate)
    require CACHE_DIR; require CRAF_CHECKPOINT; require TEST_OUTPUT
    script=hammer_craf_server.py
    if [[ "$command" == v2-evaluate ]]; then script=hammer_craf_v2_server.py; fi
    CUDA_VISIBLE_DEVICES="$HEAD_GPU" "$PYTHON" "$GPU_DIR/$script" evaluate \
      --cache-dir "$CACHE_DIR" --craf-checkpoint "$CRAF_CHECKPOINT" \
      --split test --output "$TEST_OUTPUT" "$@"
    ;;
  download-data)
    exec "$PYTHON" "$GPU_DIR/download_dgm4.py" --data-root "$DATA_ROOT" "$@"
    ;;
  help|-h|--help)
    cat <<'HELP'
Usage: bash gpu/run_project.sh COMMAND [original script arguments]
Explicit actions only; no default download, extraction, training or test.
  hammer-train / hammer-resume  Four-process HAMMER training
  hammer-test                  Standalone full official test, one GPU
  extract --splits train val    Frozen four-process extraction (only when needed)
  v1-train / v1-resume          Existing cache, lr=1e-5 default, one GPU
  v2-variants [all|control|penalty|decision|combined]  Existing cache
  v2-train / v2-resume          Existing cache; pass original variant flags
  v1-evaluate / v2-evaluate     Existing test cache, no extraction
  download-data                Explicit dataset download/validation
Read docs/GPU_USAGE.md for required environment variables and resume settings.
HELP
    ;;
  *) echo "Unknown command: $command" >&2; exit 2 ;;
esac
