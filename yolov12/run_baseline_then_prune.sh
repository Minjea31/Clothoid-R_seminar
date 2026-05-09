#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/yolov12"
BMODEL_PATH="${PROJECT_DIR}/checkpoints/yolov12_0510/weights/best.pt"

cd "${PROJECT_DIR}"

python train_baseline_model.py --epoch 600

if [[ ! -f "${BMODEL_PATH}" ]]; then
  echo "Expected checkpoint not found: ${BMODEL_PATH}" >&2
  exit 1
fi

python pruning_finetuning.py --bmodel "${BMODEL_PATH}" --epoch 600
