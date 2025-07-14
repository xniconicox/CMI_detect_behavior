#!/bin/bash

# 修正版マルチモーダル学習スクリプト
# NaN問題と学習率問題を修正

set -e

# デフォルト設定
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"20250713_preproc_pipeline_first_try"}
TRAINER_TYPE=${TRAINER_TYPE:-"multimodal_fixed"}
EPOCHS=${EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-32}

echo "🚀 修正版マルチモーダル学習開始"
echo "=================================================="
echo "実験名: $EXPERIMENT_NAME"
echo "トレーナー: $TRAINER_TYPE"
echo "エポック数: $EPOCHS"
echo "バッチサイズ: $BATCH_SIZE"
echo "=================================================="

# データ確認
echo "📊 データ確認中..."
if [ ! -d "output/experiments/$EXPERIMENT_NAME/preprocessed" ]; then
    echo "❌ 前処理済みデータが見つかりません: output/experiments/$EXPERIMENT_NAME/preprocessed"
    echo "前処理を先に実行してください"
    exit 1
fi
echo "✅ データ確認完了"

# 環境変数設定
echo " 環境変数設定中..."
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0
echo "✅ 環境変数設定完了"

# 学習開始
echo "🎯 学習開始..."
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="output/experiments/multimodal/logs/training_${timestamp}.log"

# ログディレクトリ作成
mkdir -p "output/experiments/multimodal/logs"

# 修正版トレーナーを実行
python3 -c "
import sys
sys.path.append('src')

from trainers.multimodal_trainer_fixed import MultimodalTrainerFixed

# トレーナー初期化
trainer = MultimodalTrainerFixed('$EXPERIMENT_NAME')

try:
    # データ読み込み
    data = trainer.load_all_data()
    
    # 学習実行
    history = trainer.train(data, epochs=$EPOCHS, batch_size=$BATCH_SIZE)
    
    # モデル保存
    trainer.save_model()
    
    # 評価実行
    results = trainer.evaluate(data)
    print(f'F1 Score: {results[\"f1_macro\"]}')
    print('学習完了！')
    
except Exception as e:
    print(f'❌ エラー: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
" 2>&1 | tee "$log_file"

# 結果確認
echo "📈 学習完了"
echo "=================================================="

# 保存されたモデル確認
echo "📁 保存されたモデル:"
ls -la "output/experiments/$EXPERIMENT_NAME/models/" 2>/dev/null || echo "モデルディレクトリが見つかりません"

# 学習結果確認
echo "📊 学習結果:"
ls -la "output/experiments/$EXPERIMENT_NAME/results/" 2>/dev/null || echo "結果ディレクトリが見つかりません"

# ログファイル確認
echo "📝 ログファイル: $log_file"
echo "最終行（エラー確認）:"
tail -5 "$log_file" 2>/dev/null || echo "ログファイルが見つかりません"

echo "✅ 修正版マルチモーダル学習完了"
echo "=================================================="

echo ""
echo "📖 使用法:"
echo "  変数を変更する場合:"
echo "    EXPERIMENT_NAME=\"your_experiment\" ./scripts/run_multimodal_training_fixed.sh"
echo ""
echo "  別の実験を実行する場合:"
echo "    EXPERIMENT_NAME=\"20250713_preproc_pipeline_first_try\" EPOCHS=100 ./scripts/run_multimodal_training_fixed.sh"
echo ""
echo "  デバッグ実行する場合:"
echo "    EPOCHS=5 BATCH_SIZE=16 ./scripts/run_multimodal_training_fixed.sh" 