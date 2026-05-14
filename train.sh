#!/usr/bin/env bash
set -euo pipefail

GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
if [[ "$GPU_COUNT" -eq 0 ]]; then
    GPU_COUNT=1
fi

export MASTER_ADDR=${MASTER_ADDR:-localhost}
export MASTER_PORT=${MASTER_PORT:-32501}
export WORLD_SIZE=${WORLD_SIZE:-$GPU_COUNT}
export NODE_RANK=${NODE_RANK:-0}
export GLOBAL_RANK_OFFSET=${GLOBAL_RANK_OFFSET:-0}

echo "[INFO] DDP environment:"
echo "[INFO] MASTER_ADDR=${MASTER_ADDR}"
echo "[INFO] MASTER_PORT=${MASTER_PORT}"
echo "[INFO] WORLD_SIZE=${WORLD_SIZE}"
echo "[INFO] NODE_RANK=${NODE_RANK}"
echo "[INFO] GLOBAL_RANK_OFFSET=${GLOBAL_RANK_OFFSET}"

echo "[INFO] Running: python ./train.py $*"
python ./train.py "$@"
