#!/bin/bash

# マルチモーダル学習実行スクリプト
# 使用方法: ./scripts/run_multimodal_training.sh

set -e  # エラー時に停止

# ============================================================
# 設定変数（ここを変更して実行条件を調整）
# ============================================================

# 実験設定
EXPERIMENT_NAME="20250713_preproc_pipeline_first_try"  # 前処理済みデータの実験名
TRAINER_NAME="multimodal"                              # トレーナー名
MODEL_NAME="multimodal_model"                          # 保存するモデル名

# 学習設定
EPOCHS=50                                              # 学習エポック数
BATCH_SIZE=32                                          # バッチサイズ
VALIDATION_SPLIT=0.2                                   # 検証データ分割率
RANDOM_SEED=42                                         # 乱数シード

# ハードウェア設定
GPU_MEMORY_GROWTH=true                                 # GPUメモリ動的割り当て
NUM_GPUS=1                                             # 使用GPU数

# ログ設定
LOG_LEVEL="INFO"                                        # ログレベル
SAVE_LOGS=true                                         # ログ保存フラグ

# ============================================================
# 環境設定
# ============================================================

echo "🚀 マルチモーダル学習開始"
echo "=================================================="
echo "実験名: $EXPERIMENT_NAME"
echo "トレーナー: $TRAINER_NAME"
echo "エポック数: $EPOCHS"
echo "バッチサイズ: $BATCH_SIZE"
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

echo "�� 環境変数設定中..."

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
from src.trainers.multimodal_trainer import MultimodalTrainer
trainer = MultimodalTrainer('$EXPERIMENT_NAME')
data = trainer.load_all_data()
trainer.train(data, epochs=$EPOCHS)
trainer.save_model()
results = trainer.evaluate(data)
print('F1 Score:', results['f1_macro'])
print('学習完了！')
" 2>&1 | tee "$LOG_FILE"
else
    python -c "
from src.trainers.multimodal_trainer import MultimodalTrainer
trainer = MultimodalTrainer('$EXPERIMENT_NAME')
data = trainer.load_all_data()
trainer.train(data, epochs=$EPOCHS)
trainer.save_model()
results = trainer.evaluate(data)
print('F1 Score:', results['f1_macro'])
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

echo "✅ マルチモーダル学習完了"
echo "=================================================="

# ============================================================
# 使用法表示
# ============================================================

echo ""
echo "📖 使用法:"
echo "  変数を変更する場合:"
echo "    EXPERIMENT_NAME=\"your_experiment\" ./scripts/run_multimodal_training.sh"
echo ""
echo "  別の実験を実行する場合:"
echo "    EXPERIMENT_NAME=\"20250713_preproc_pipeline_first_try\" EPOCHS=100 ./scripts/run_multimodal_training.sh"
echo ""
echo "  デバッグ実行する場合:"
echo "    EPOCHS=5 BATCH_SIZE=16 ./scripts/run_multimodal_training.sh"
echo ""