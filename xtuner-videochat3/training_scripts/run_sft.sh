#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash training_scripts/run_sft.sh <config-path>" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="$1"

if [[ "${CONFIG_PATH}" != /* ]]; then
  CONFIG_PATH="${PROJECT_ROOT}/${CONFIG_PATH}"
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 2
fi

NNODES="${NNODES:-8}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-40000}"
TASK_NAME="$(basename "${CONFIG_PATH}" .py)"
RDZV_ID="${RDZV_ID:-${SLURM_JOB_ID:-${TASK_NAME}}}"

if ! [[ "${NNODES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NNODES must be a positive integer, got: ${NNODES}" >&2
  exit 2
fi
if ! [[ "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NPROC_PER_NODE must be a positive integer, got: ${NPROC_PER_NODE}" >&2
  exit 2
fi

if [[ -n "${SLURM_JOB_NODELIST:-}" ]]; then
  if ! command -v scontrol >/dev/null 2>&1; then
    echo "SLURM_JOB_NODELIST is set, but scontrol is unavailable." >&2
    exit 2
  fi
  mapfile -t allocated_hosts < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
  if (( ${#allocated_hosts[@]} < NNODES )); then
    echo "NNODES=${NNODES}, but the Slurm allocation has only ${#allocated_hosts[@]} nodes." >&2
    exit 2
  fi
  MASTER_ADDR="${MASTER_ADDR:-${allocated_hosts[0]}}"
else
  if (( NNODES > 1 )) && [[ -z "${MASTER_ADDR:-}" ]]; then
    echo "MASTER_ADDR must be set when running ${NNODES} nodes without Slurm." >&2
    exit 2
  fi
  MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
fi

export XTUNER_USE_FA3="${XTUNER_USE_FA3:-0}"
export XTUNER_GC_ENABLE="${XTUNER_GC_ENABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/work_dir/logs/${TASK_NAME}}"
mkdir -p "${LOG_DIR}"
timestamp="$(date '+%Y%m%d_%H%M%S')"
node_name="$(hostname -s 2>/dev/null || hostname)"
log_file="${LOG_DIR}/training_${timestamp}_${node_name}.log"

cd "${PROJECT_ROOT}"

echo "Config: ${CONFIG_PATH}"
echo "Nodes: ${NNODES}; processes per node: ${NPROC_PER_NODE}"
echo "Rendezvous: ${MASTER_ADDR}:${MASTER_PORT}; id: ${RDZV_ID}"
echo "Log: ${log_file}"

torchrun \
  --nnodes="${NNODES}" \
  --nproc-per-node="${NPROC_PER_NODE}" \
  --rdzv-backend=c10d \
  --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  --rdzv-id="${RDZV_ID}" \
  xtuner/v1/train/cli/sft.py \
  --config "${CONFIG_PATH}" 2>&1 | tee -a "${log_file}"
