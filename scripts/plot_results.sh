#!/bin/bash

# 学習結果の可視化スクリプト
# 使用方法: ./scripts/plot_results.sh [experiment_name] [trainer_name]

set -e

# ============================================================
# 設定変数
# ============================================================

# デフォルト値
DEFAULT_EXPERIMENT_NAME="20250717_preproc_train_v30"
DEFAULT_TRAINER_NAME="20250717_preproc_train_v30"

# 引数から取得（指定がない場合はデフォルト値を使用）
EXPERIMENT_NAME="${1:-$DEFAULT_EXPERIMENT_NAME}"
TRAINER_NAME="${2:-$DEFAULT_TRAINER_NAME}"

# ============================================================
# 環境設定
# ============================================================

echo "📊 学習結果の可視化開始"
echo "=================================================="
echo "実験名: $EXPERIMENT_NAME"
echo "トレーナー: $TRAINER_NAME"
echo "=================================================="

# プロジェクトルートディレクトリに移動
cd "$(dirname "$0")/.."

# 仮想環境の確認
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  仮想環境が有効化されていません"
    echo "source .venv/bin/activate を実行してください"
    exit 1
fi

# ============================================================
# 結果ディレクトリ確認
# ============================================================

RESULT_DIR="output/experiments/$TRAINER_NAME/results"

if [[ ! -d "$RESULT_DIR" ]]; then
    echo "❌ 結果ディレクトリが見つかりません: $RESULT_DIR"
    echo "学習を先に実行してください:"
    echo "./scripts/run_multimodal_training_v30.sh"
    exit 1
fi

echo "✅ 結果ディレクトリ確認完了: $RESULT_DIR"

# ============================================================
# 可視化実行
# ============================================================

echo "🎯 可視化開始..."

python scripts/plot_training_results.py \
    --experiment-name "$EXPERIMENT_NAME" \
    --trainer-name "$TRAINER_NAME"

# ============================================================
# 結果確認
# ============================================================

echo "📈 可視化完了"
echo "=================================================="

# 生成されたファイルの確認
if [[ -d "$RESULT_DIR" ]]; then
    echo "📊 生成されたファイル:"
    ls -la "$RESULT_DIR"/*.png 2>/dev/null || echo "PNGファイルが見つかりません"
fi

echo "✅ 学習結果の可視化完了"
echo "=================================================="

# ============================================================
# 使用法表示
# ============================================================

echo ""
echo "📖 使用法:"
echo "  デフォルト実行:"
echo "    ./scripts/plot_results.sh"
echo ""
echo "  実験名を指定:"
echo "    ./scripts/plot_results.sh 20250717_preproc_train_v30"
echo ""
echo "  実験名とトレーナー名を指定:"
echo "    ./scripts/plot_results.sh 20250717_preproc_train_v30 multimodal_v30"
echo ""
echo "  別の出力ディレクトリに保存:"
echo "    python scripts/plot_training_results.py --experiment-name $EXPERIMENT_NAME --output-dir ./my_plots"
echo "" 