#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/../run_sft.sh" \
  "training_configs/stage1/VideoChat3_4B_train_stage1-1.py"
