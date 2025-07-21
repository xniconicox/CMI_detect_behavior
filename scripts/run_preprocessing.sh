#!/usr/bin/env bash
# 前処理パイプライン実行用スクリプト
# 使用例:
#   bash scripts/run_preprocessing.sh exp1 train
#   bash scripts/run_preprocessing.sh exp1 predict
set -e

EXPERIMENT_NAME=${1:?実験名を指定してください}
MODE=${2:-train}

# ウィンドウサイズごとに実行
for WS in 64 128; do
    if [ "$WS" = "64" ]; then
        CONFIG="config/config_v2_ws64.yaml"
    else
        CONFIG="config/config_v2.yaml"
    fi

    python scripts/run_preprocessing.py \
        --experiment-name "${EXPERIMENT_NAME}_ws${WS}" \
        --config "$CONFIG" \
        --use-cache \
        --mode "$MODE"
done

