#!/bin/bash

# 最適化された前処理実行スクリプト
# 使用方法: ./scripts/run_preprocessing_optimized.sh [mode]

set -e  # エラー時に停止

# ============================================================
# 設定変数（必要に応じて変更してください）
# ============================================================

EXPERIMENT_NAME="preproc-v2-optimized"        # 実験名
CONFIG_PATH="config/config_v2.yaml"           # 設定ファイルパス
USE_CACHE=true                               # キャッシュ利用（デフォルト有効）
MODE="${1:-train}"                           # train または predict（引数で指定可能）
AUGMENT_HANDEDNESS=true                      # 利き手反転によるデータ拡張（デフォルト有効）

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

# 設定ファイルの存在確認
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "❌ 設定ファイルが見つかりません: $CONFIG_PATH"
    exit 1
fi

# ============================================================
# 設定の表示
# ============================================================

echo "🔧 前処理設定"
echo "  実験名: $EXPERIMENT_NAME"
echo "  設定ファイル: $CONFIG_PATH"
echo "  モード: $MODE"
echo "  キャッシュ利用: $USE_CACHE"
echo "  利き手拡張: $AUGMENT_HANDEDNESS"

# ============================================================
# コマンド生成
# ============================================================

CMD=(python -m scripts.run_preprocessing)
CMD+=(--experiment-name "$EXPERIMENT_NAME")
CMD+=(--config "$CONFIG_PATH")
CMD+=(--mode "$MODE")

if [[ "$USE_CACHE" == "true" ]]; then
    CMD+=(--use-cache)
    echo "  キャッシュを利用します"
else
    echo "  キャッシュを無効にします"
fi

if [[ "$AUGMENT_HANDEDNESS" == "true" ]]; then
    CMD+=(--augment-handedness)
    echo "  利き手反転によるデータ拡張を有効にします"
else
    echo "  利き手反転によるデータ拡張を無効にします"
fi

# ============================================================
# 実行
# ============================================================

echo ""
echo "🚀 最適化された前処理を開始します"
echo "   コマンド: ${CMD[*]}"
echo ""

# 実行時間の計測開始
START_TIME=$(date +%s)

"${CMD[@]}"

# 実行時間の計測終了
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "✅ 前処理が完了しました"
echo "   実行時間: ${DURATION}秒"
echo "   出力先: output/experiments/$EXPERIMENT_NAME/" 