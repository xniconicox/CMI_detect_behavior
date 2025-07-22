#!/bin/bash

# マルチモーダル学習実行スクリプト（v40モデル用）
# 使用方法: ./scripts/run_multimodal_training_v40.sh
# 64と128の窓サイズで連続実行します

set -e # エラー時に停止

# ============================================================
# 固定設定
# ============================================================
TRAINER_NAME="multimodal_v40"
MODEL_NAME="multimodal_model_v40"
EPOCHS=50
BATCH_SIZE=32
GPU_MEMORY_GROWTH=true
SAVE_LOGS=true

# ============================================================
# メインループ
# ============================================================
for WS in 64 128; do

    # ============================================================
    # 動的設定
    # ============================================================
    EXPERIMENT_NAME="preprocess_v40_ws${WS}"
    CONFIG_PATH="./config/config_v40_ws${WS}.yaml"

    # ============================================================
    # 環境設定 & データ確認
    # ============================================================
    echo "=================================================="
    echo "🚀 マルチモーダル学習開始 (v40モデル - ws${WS})"
    echo "=================================================="
    echo "実験名: $EXPERIMENT_NAME"
    echo "設定ファイル: $CONFIG_PATH"
    echo "=================================================="

    cd "$(dirname "$0")/.."

    if [[ -z "$VIRTUAL_ENV" ]]; then
        echo "⚠️  仮想環境が有効化されていません"
        exit 1
    fi

    echo "📊 データ確認中 (ws${WS})..."
    DATA_DIR="output/experiments/$EXPERIMENT_NAME/preprocessed"

    if [[ ! -d "$DATA_DIR" ]]; then
        echo "❌ データディレクトリが見つかりません: $DATA_DIR"
        echo "次のウィンドウサイズに進みます..."
        continue
    fi

    echo "✅ データ確認完了 (ws${WS})"

    # ============================================================
    # 環境変数設定
    # ============================================================
    echo "🔧 環境変数設定中..."
    export TF_FORCE_GPU_ALLOW_GROWTH=$GPU_MEMORY_GROWTH
    export CUDA_VISIBLE_DEVICES=0
    export TF_CPP_MIN_LOG_LEVEL=2
    export PYTHONPATH="${PYTHONPATH}:$(pwd)"

    # ============================================================
    # 学習実行
    # ============================================================
    echo "🎯 学習開始 (ws${WS})..."
    LOG_DIR="output/experiments/$EXPERIMENT_NAME/logs"
    mkdir -p "$LOG_DIR"
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_FILE="$LOG_DIR/training_${TIMESTAMP}_ws${WS}.log"

    PYTHON_SCRIPT="
import yaml
from src.trainers.multimodal_trainer_v40 import MultimodalTrainerV40
with open('$CONFIG_PATH', 'r') as f:
    config = yaml.safe_load(f)
train_params = config.get('training_params', {})
trainer = MultimodalTrainerV40('$EXPERIMENT_NAME')
data = trainer.load_all_data()
history = trainer.train_cross_validation(
    data,
    epochs=train_params.get('epochs', $EPOCHS),
    batch_size=train_params.get('batch_size', $BATCH_SIZE)
)
trainer.save_model()
results = trainer.evaluate(data)
trainer.save_training_history()
trainer.save_evaluation_results(results)
print('Macro F1 Score:', results.get('macro_f1', None))
print('CMI Score:', results.get('cmi_score', None))
print('学習完了！ (ws${WS})')
"
    if [[ "$SAVE_LOGS" == "true" ]]; then
        echo "📝 ログファイル: $LOG_FILE"
        python -c "$PYTHON_SCRIPT" 2>&1 | tee "$LOG_FILE"
    else
        python -c "$PYTHON_SCRIPT"
    fi

    echo "✅ マルチモーダル学習完了 (v40モデル - ws${WS})"

done

echo "=================================================="
echo "✅ 全ての学習プロセスが完了しました"
echo "=================================================="