#!/bin/bash

# マルチモーダル学習実行スクリプト（v40モデル用）
# 使用方法: ./scripts/run_multimodal_training_v40.sh

set -e  # エラー時に停止

# ============================================================
# 設定変数（ここを変更して実行条件を調整）
# ============================================================

# 実験設定
EXPERIMENT_NAME="20250717_preproc_train_v40"  # 前処理済みデータの実験名
TRAINER_NAME="multimodal_v40"                              # トレーナー名
MODEL_NAME="multimodal_model_v40"                          # 保存するモデル名
CONFIG_PATH="./config/config_v40.yaml"                    # 設定ファイルパス

# 学習設定（configファイルから読み込むためここはデフォルト値）
EPOCHS=50
BATCH_SIZE=32
VALIDATION_SPLIT=0.2
RANDOM_SEED=42

# ハードウェア設定
GPU_MEMORY_GROWTH=true
NUM_GPUS=1

# ログ設定
LOG_LEVEL="INFO"
SAVE_LOGS=true

# ============================================================
# 環境設定
# ============================================================

echo "🚀 マルチモーダル学習開始 (v40モデル)"
echo "=================================================="
echo "実験名: $EXPERIMENT_NAME"
echo "トレーナー: $TRAINER_NAME"
echo "エポック数: $EPOCHS (※configファイルで上書きされます)"
echo "バッチサイズ: $BATCH_SIZE (※configファイルで上書きされます)"
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
# データ確認
# ============================================================

echo "📊 データ確認中..."

DATA_DIR="output/experiments/$EXPERIMENT_NAME/preprocessed"

if [[ ! -d "$DATA_DIR" ]]; then
    echo "❌ データディレクトリが見つかりません: $DATA_DIR"
    echo "前処理を先に実行してください:"
    echo "python -m scripts.run_preprocessing --experiment-name $EXPERIMENT_NAME"
    exit 1
fi

# 必要なファイルの存在確認
REQUIRED_FILES=(
    "train_windows.pkl"
    "train_demographics.pkl"
    "train_tabular.pkl"
    "train_tof_windows.pkl"
    "train_labels.pkl"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$DATA_DIR/$file" ]]; then
        echo "❌ 必要なファイルが見つかりません: $file"
        exit 1
    fi
done

echo "✅ データ確認完了"

# ============================================================
# 環境変数設定
# ============================================================

echo "🔧 環境変数設定中..."

# GPU設定
export TF_FORCE_GPU_ALLOW_GROWTH=$GPU_MEMORY_GROWTH
export CUDA_VISIBLE_DEVICES=0

# TensorFlow設定
export TF_CPP_MIN_LOG_LEVEL=2  # 警告を抑制

# Python設定
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "✅ 環境変数設定完了"

# ============================================================
# 学習実行
# ============================================================

echo "🎯 学習開始..."

# ログディレクトリ作成
LOG_DIR="output/experiments/$TRAINER_NAME/logs"
mkdir -p "$LOG_DIR"

# タイムスタンプ
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/training_${TIMESTAMP}.log"

# 学習実行
if [[ "$SAVE_LOGS" == "true" ]]; then
    echo "📝 ログファイル: $LOG_FILE"
    python -c "
import yaml
from src.trainers.multimodal_trainer_v40 import MultimodalTrainerV40
with open('$CONFIG_PATH', 'r') as f:
    config = yaml.safe_load(f)
model_params = config.get('model_params', {})
train_params = config.get('training_params', {})
trainer = MultimodalTrainerV40('$EXPERIMENT_NAME')
data = trainer.load_all_data()
history = trainer.train_cross_validation(
    data,
    epochs=train_params.get('epochs', 50),
    batch_size=train_params.get('batch_size', 32)
)
trainer.save_model()
results = trainer.evaluate(data)
trainer.save_training_history()
trainer.save_evaluation_results(results)
print('Macro F1 Score:', results.get('macro_f1', None))
print('CMI Score:', results.get('cmi_score', None))
print('学習完了！')
" 2>&1 | tee "$LOG_FILE"
else
    python -c "
import yaml
from src.trainers.multimodal_trainer_v40 import MultimodalTrainerV40
with open('$CONFIG_PATH', 'r') as f:
    config = yaml.safe_load(f)
model_params = config.get('model_params', {})
train_params = config.get('training_params', {})
trainer = MultimodalTrainerV40('$EXPERIMENT_NAME')
data = trainer.load_all_data()
history = trainer.train_cross_validation(
    data,
    epochs=train_params.get('epochs', 50),
    batch_size=train_params.get('batch_size', 32)
)
trainer.save_model()
results = trainer.evaluate(data)
trainer.save_training_history()
trainer.save_evaluation_results(results)
print('Macro F1 Score:', results.get('macro_f1', None))
print('CMI Score:', results.get('cmi_score', None))
print('学習完了！')
"
fi

# ============================================================
# 結果確認
# ============================================================

echo "📈 学習完了"
echo "=================================================="

# モデルファイルの確認
MODEL_DIR="output/experiments/$TRAINER_NAME/models"
if [[ -d "$MODEL_DIR" ]]; then
    echo "📁 保存されたモデル:"
    ls -la "$MODEL_DIR"
fi

# 結果ファイルの確認
RESULT_DIR="output/experiments/$TRAINER_NAME/results"
if [[ -d "$RESULT_DIR" ]]; then
    echo "📊 学習結果:"
    ls -la "$RESULT_DIR"
fi

# ログファイルの確認
if [[ "$SAVE_LOGS" == "true" && -f "$LOG_FILE" ]]; then
    echo "📝 ログファイル: $LOG_FILE"
    echo "最終行（エラー確認）:"
    tail -5 "$LOG_FILE"
fi

echo "✅ マルチモーダル学習完了 (v40モデル)"
echo "=================================================="

# ============================================================
# 使用法表示
# ============================================================

echo ""
echo "📖 使用法:"
echo "  変数を変更する場合:"
echo "    EXPERIMENT_NAME=\"your_experiment\" ./scripts/run_multimodal_training_v40.sh"
echo ""
echo "  別の実験を実行する場合:"
echo "    EXPERIMENT_NAME=\"20250713_preproc_pipeline_first_try\" EPOCHS=100 ./scripts/run_multimodal_training_v40.sh"
echo ""
echo "  デバッグ実行する場合:"
echo "    EPOCHS=5 BATCH_SIZE=16 ./scripts/run_multimodal_training_v40.sh"
echo ""
echo "  64と128でアンサンブルする例:"
echo "    EXPERIMENT_NAME=\"exp_ws64\" ./scripts/run_multimodal_training_v40.sh"
echo "    EXPERIMENT_NAME=\"exp_ws128\" ./scripts/run_multimodal_training_v40.sh"
echo ""