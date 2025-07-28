#!/bin/bash

# マルチモーダル学習実行スクリプト（v50モデル用）
# 使用方法: ./scripts/run_multimodal_training_v50.sh

set -e

TRAINER_NAME="multimodal_v50"
MODEL_NAME="multimodal_model_v50"
EXPERIMENT_NAME="preprocess_v50"
CONFIG_PATH="./config/config_v50.yaml"
EPOCHS=50
BATCH_SIZE=32
GPU_MEMORY_GROWTH=true
SAVE_LOGS=true

cd "$(dirname "$0")/.."

if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  仮想環境が有効化されていません"
    exit 1
fi

DATA_DIR="output/experiments/$EXPERIMENT_NAME/preprocessed"
if [[ ! -d "$DATA_DIR" ]]; then
    echo "❌ データディレクトリが見つかりません: $DATA_DIR"
    exit 1
fi

export TF_FORCE_GPU_ALLOW_GROWTH=$GPU_MEMORY_GROWTH
export CUDA_VISIBLE_DEVICES=0
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

LOG_DIR="output/experiments/$EXPERIMENT_NAME/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/training_${TIMESTAMP}.log"

PYTHON_SCRIPT="
import yaml
from src.trainers.multimodal_trainer_v50 import MultimodalTrainerV50
with open('$CONFIG_PATH', 'r') as f:
    config = yaml.safe_load(f)
train_params = config.get('training_params', {})
trainer = MultimodalTrainerV50('$EXPERIMENT_NAME')
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
print('学習完了！')
"

if [[ "$SAVE_LOGS" == "true" ]]; then
    echo "📝 ログファイル: $LOG_FILE"
    python -c "$PYTHON_SCRIPT" 2>&1 | tee "$LOG_FILE"
else
    python -c "$PYTHON_SCRIPT"
fi

echo "✅ マルチモーダル学習完了 (v50モデル)"
