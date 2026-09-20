#!/usr/bin/env bash
# Starts the NVIDIA MPS (Multi-Process Service) daemon on the HOST — not
# inside any application image. Run this once per GPU-having host before
# bringing up compose services that reference CUDA_MPS_PIPE_DIRECTORY
# (see deploy/docker-compose.ner_translate_server.yml, and
# docs/CONCURRENCY_AND_UDF_ENHANCEMENTS.md §4 for why this matters: without
# MPS, multiple processes sharing one physical GPU get crudely time-sliced
# by the driver instead of genuinely co-scheduled).
#
# Safe to run repeatedly — checks whether the daemon is already up first.
# Matches the manual per-node step docs/internet_to_airgapped_transfer.md
# §3.5 already documents for the multi-GPU-node cluster case; this script
# is the same thing, just runnable as one command instead of copy-pasted
# by hand, and also usable for the single-GPU waiter/kitchen case.
set -euo pipefail

PIPE_DIR="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}"
LOG_DIR="${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-mps-log}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found — this host has no NVIDIA driver installed, MPS cannot run here." >&2
    exit 1
fi

if ! nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi found but failed to run — no GPU visible to this host/user." >&2
    exit 1
fi

mkdir -p "$PIPE_DIR" "$LOG_DIR"
export CUDA_MPS_PIPE_DIRECTORY="$PIPE_DIR"
export CUDA_MPS_LOG_DIRECTORY="$LOG_DIR"

if echo "get_server_list" | nvidia-cuda-mps-control 2>/dev/null; then
    echo "MPS daemon already running (pipe: $PIPE_DIR)."
    exit 0
fi

echo "Starting MPS daemon (pipe: $PIPE_DIR, log: $LOG_DIR)..."
nvidia-cuda-mps-control -d
echo "MPS daemon started. Point containers at CUDA_MPS_PIPE_DIRECTORY=$PIPE_DIR"
echo "(docker-compose.ner_translate_server.yml already does, via the same /tmp/nvidia-mps volume mount)."
