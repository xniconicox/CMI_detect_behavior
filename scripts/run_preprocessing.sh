#!/usr/bin/env bash
# 前処理パイプライン実行用スクリプト
# 使用例:
#   bash scripts/run_preprocessing.sh exp1 train
#   bash scripts/run_preprocessing.sh exp1 predict
set -e

EXPERIMENT_NAME=${1:?実験名を指定してください}
MODE=${2:-train}
CONFIG_PATH="config/config_v2.yaml"

python scripts/run_preprocessing.py \
    --experiment-name "$EXPERIMENT_NAME" \
    --config "$CONFIG_PATH" \
    --use-cache \
    --mode "$MODE"

