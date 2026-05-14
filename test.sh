#!/usr/bin/env bash
set -euo pipefail

PORT=${PORT:-32500}
PROCS_PER_GPU=${TEST_PROCS_PER_GPU:-2}

EXTRA_ARGS=()
if [[ "${1:-}" == "-h" && -n "${2:-}" ]]; then
  EXTRA_ARGS=("-h" "$2")
fi

PORT_IN_USE=""
if command -v lsof >/dev/null 2>&1; then
  PORT_IN_USE=$(lsof -i ":${PORT}" || true)
elif command -v ss >/dev/null 2>&1; then
  PORT_IN_USE=$(ss -ltn "sport = :${PORT}" || true)
fi

if [[ ${#EXTRA_ARGS[@]} -eq 0 && -z "$PORT_IN_USE" ]]; then
  echo "[INFO] Port ${PORT} is free. Starting test server..."
  echo "[INFO] Test server is waiting for workers."
  echo "[INFO] Open another terminal and run test workers to start evaluation (e.g. 'bash ./test.sh -h <server_host>')."
  python test_server.py
  exit 0
fi

echo "[INFO] Starting test workers on GPUs..."
NUM_GPU=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
if [[ "$NUM_GPU" -eq 0 ]]; then
  echo "[ERROR] No GPU detected by nvidia-smi."
  exit 1
fi

for ((i=0; i<NUM_GPU; i++)); do
  for ((j=0; j<PROCS_PER_GPU; j++)); do
    echo "[INFO] Launching worker $j on GPU $i"
    CUDA_VISIBLE_DEVICES=$i python test.py "${EXTRA_ARGS[@]}" &
  done
done

wait
echo "[INFO] All test workers completed."
