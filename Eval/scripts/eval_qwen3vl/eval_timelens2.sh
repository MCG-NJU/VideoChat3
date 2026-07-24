#!/usr/bin/env bash
set -euo pipefail

if (( $# == 0 )); then
  echo "Usage: $0 <dataset> [<dataset> ...]" >&2
  echo "Example: LMUData=/path/to/LMUData $0 TimeLens_Charades_4fps" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
MODEL_NAME="${MODEL_NAME:-TimeLens2-4B}"
MODE="${MODE:-all}"

: "${LMUData:?Set LMUData to the VLMEvalKit data directory.}"

cd "${REPO_ROOT}"
python run.py \
  --data "$@" \
  --model "${MODEL_NAME}" \
  --mode "${MODE}"
