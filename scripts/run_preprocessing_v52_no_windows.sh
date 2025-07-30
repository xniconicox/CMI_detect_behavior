#!/bin/bash

# v52 preprocessing without window segmentation
set -e

EXPERIMENT_NAME="preprocess_v52"

cd "$(dirname "$0")/.."

python scripts/run_preprocessing_v52_no_windows.py \
    --experiment-name "$EXPERIMENT_NAME" \
    --config config/config_v52.yaml \
    --mode train
