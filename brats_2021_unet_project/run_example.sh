#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${1:-/path/to/BraTS2021}
OUTPUT_DIR=${2:-./outputs/brats_2021_unet}

python brats_2021_unet_project/train.py \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --epochs 1 \
  --batch-size 8 \
  --num-workers 8 \
  --device cuda \
  --skip-empty-slices
