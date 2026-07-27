#!/usr/bin/env bash
set -euo pipefail

if (( $# == 0 )); then
  echo "Usage: $0 --dataset <dataset>" >&2
  echo "Example: LMUData=/path/to/LMUData $0 --dataset Video-MME_2fps_limit_1024_448px_80kctx" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

: "${LMUData:?Set LMUData to the VLMEvalKit data directory.}"

exec python -u "${SCRIPT_DIR}/extract_video_frames.py" "$@"
