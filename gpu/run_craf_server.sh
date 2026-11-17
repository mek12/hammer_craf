#!/usr/bin/env bash
set -euo pipefail

PACKAGE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-/mnt/sdb/home/mekocak/projects/MultiModal-DeepFake}"
DATA_ROOT="${DATA_ROOT:-/mnt/sdb/home/mekocak/projects/datasets}"
HAMMER_CHECKPOINT="${HAMMER_CHECKPOINT:-$REPO_DIR/results/hammer_4gpu/checkpoint_best.pth}"
CACHE_DIR="${CACHE_DIR:-$PACKAGE_DIR/craf_features}"
OUTPUT_DIR="${OUTPUT_DIR:-$PACKAGE_DIR/craf_results}"
PYTHON="${PYTHON:-python}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
HEAD_GPU="${HEAD_GPU:-${GPU_IDS%%,*}}"
TEXT_ENCODER="${TEXT_ENCODER:-bert-base-uncased}"
EXTRACT_BATCH="${EXTRACT_BATCH:-32}"
WORKERS="${WORKERS:-4}"
EPOCHS="${EPOCHS:-30}"
HEAD_BATCH="${HEAD_BATCH:-512}"
SEED="${SEED:-42}"
ABLATION="${ABLATION:-full}"

extract_splits() {
  if [[ ! -f "$HAMMER_CHECKPOINT" ]]; then
    echo "Missing trained HAMMER checkpoint: $HAMMER_CHECKPOINT" >&2
    echo "Set HAMMER_CHECKPOINT to your completed training's checkpoint_best.pth." >&2
    exit 1
  fi
  CUDA_VISIBLE_DEVICES="$GPU_IDS" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
    "$PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \
    "$PACKAGE_DIR/hammer_craf_server.py" extract \
    --repo-dir "$REPO_DIR" --config configs/train.yaml \
    --hammer-checkpoint "$HAMMER_CHECKPOINT" --data-root "$DATA_ROOT" \
    --cache-dir "$CACHE_DIR" --text-encoder "$TEXT_ENCODER" \
    --batch-size "$EXTRACT_BATCH" --workers "$WORKERS" --splits "$@"
}

train_head() {
  CUDA_VISIBLE_DEVICES="$HEAD_GPU" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
    "$PYTHON" "$PACKAGE_DIR/hammer_craf_server.py" train \
    --cache-dir "$CACHE_DIR" --output-dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" --batch-size "$HEAD_BATCH" --seed "$SEED" \
    --ablation "$ABLATION" --selection-metric auc --patience 5 "$@"
}

case "${1:-help}" in
  train)
    extract_splits train val
    train_head
    ;;
  resume)
    train_head --resume
    ;;
  test)
    if [[ ! -f "$OUTPUT_DIR/best_craf.pth" ]]; then
      echo "Train CRAF first: missing $OUTPUT_DIR/best_craf.pth" >&2
      exit 1
    fi
    extract_splits test
    CUDA_VISIBLE_DEVICES="$HEAD_GPU" OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" \
      "$PYTHON" "$PACKAGE_DIR/hammer_craf_server.py" evaluate \
      --cache-dir "$CACHE_DIR" --craf-checkpoint "$OUTPUT_DIR/best_craf.pth" \
      --split test --output "$OUTPUT_DIR/test_comparison.json"
    ;;
  head)
    train_head
    ;;
  *)
    echo "Usage: bash run_craf_server.sh {train|resume|test|head}"
    echo "train: extract full train/val on 4 GPUs, then train the head on 1 GPU"
    echo "test: explicitly extract/evaluate test AFTER model selection"
    echo "head: reuse caches for a new seed/ablation with a new OUTPUT_DIR"
    ;;
esac
