#!/usr/bin/env bash
set -euo pipefail

if (( $# == 0 )); then
  echo "Usage: $0 <dataset> [<dataset> ...]" >&2
  echo "Example: LMUData=/path/to/LMUData $0 Video-MME_2fps_limit_1024_448px_80kctx" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
MODEL_NAME="${MODEL_NAME:-VideoChat3-4B}"
MODE="${MODE:-all}"

: "${LMUData:?Set LMUData to the VLMEvalKit data directory.}"

cd "${REPO_ROOT}"
python run.py \
  --data "$@" \
  --model "${MODEL_NAME}" \
  --mode "${MODE}"
