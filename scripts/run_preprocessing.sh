
#!/bin/bash
# 前処理一括実行スクリプト
# 使用例:
#   ./scripts/run_preprocessing.sh exp1
#   ./scripts/run_preprocessing.sh exp1 --mode predict --no-cache

EXPERIMENT="preprocess_v2"
MODE="train"
CONFIG_PATH="config/config_v2.yaml"
USE_CACHE=true
LOG_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2 ;;
    --config) CONFIG_PATH="$2"; shift 2 ;;
    --no-cache) USE_CACHE=false; shift ;;
    --log) LOG_FILE="$2"; shift 2 ;;
    *) EXPERIMENT="$1"; shift ;;
  esac
done

if [[ -z "$EXPERIMENT" ]]; then
  echo "使用法: $0 <experiment_name> [--mode train|predict] [--config PATH] [--no-cache] [--log LOG]"
  exit 1
fi

CACHE_OPT=""
if $USE_CACHE; then
  CACHE_OPT="--use-cache"
fi
LOG_OPT=""
if [[ -n "$LOG_FILE" ]]; then
  LOG_OPT="--log-file $LOG_FILE"
fi

python scripts/run_preprocessing.py \
    --experiment-name "$EXPERIMENT" \
    --config "$CONFIG_PATH" \
    $CACHE_OPT \
    --mode "$MODE" \
    $LOG_OPT
