#!/bin/bash

# 前処理実行スクリプト
# 使用方法: ./scripts/run_preprocessing.sh

set -e  # エラー時に停止

# ============================================================
# 設定変数（必要に応じて変更してください）
# ============================================================

EXPERIMENT_NAME="preproc-v2"                 # 実験名
CONFIG_PATH="config/config_v2.yaml"    # 設定ファイルパス
USE_CACHE=false                         # キャッシュ利用
MODE="train"                           # train または predict
AUGMENT_HANDEDNESS=false                # 利き手反転によるデータ拡張

# ============================================================
# 実行前準備
# ============================================================

# プロジェクトルートへ移動
cd "$(dirname "$0")/.."

# 仮想環境チェック
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  仮想環境が有効化されていません"
    echo "source .venv/bin/activate を実行してください"
    exit 1
fi

# ============================================================
# コマンド生成
# ============================================================

CMD=(python -m scripts.run_preprocessing)
CMD+=(--experiment-name "$EXPERIMENT_NAME")
CMD+=(--config "$CONFIG_PATH")
CMD+=(--mode "$MODE")

if [[ "$USE_CACHE" == "true" ]]; then
    CMD+=(--use-cache)
fi

if [[ "$AUGMENT_HANDEDNESS" == "true" ]]; then
    CMD+=(--augment-handedness)
fi

# ============================================================
# 実行
# ============================================================

echo "🚀 前処理を開始します"
# コマンド配列を実行して前処理スクリプトを起動
exec "${CMD[@]}"
