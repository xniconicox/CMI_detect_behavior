#!/bin/bash

# 窓分割なし前処理スクリプト
# シーケンス単位の特徴量抽出を行い、異常検知用のデータを生成

set -e

# 実験名を設定
EXPERIMENT_NAME="preprocess_v51_anomaly"

echo "=== 窓分割なし前処理開始 ==="
echo "実験名: $EXPERIMENT_NAME"

# 前処理実行
python scripts/run_preprocessing_no_windows.py \
    --experiment-name $EXPERIMENT_NAME \
    --config config/config_v51.yaml \
    --mode train

echo "=== 前処理完了 ==="
echo "出力ディレクトリ: output/experiments/$EXPERIMENT_NAME/preprocessed/"

# 生成されたファイルの確認
echo "=== 生成されたファイル ==="
ls -la output/experiments/$EXPERIMENT_NAME/preprocessed/

# メタデータの確認
echo "=== メタデータ情報 ==="
if [ -f "output/experiments/$EXPERIMENT_NAME/preprocessed/train_metadata.json" ]; then
    cat output/experiments/$EXPERIMENT_NAME/preprocessed/train_metadata.json | jq '.shapes'
    echo ""
    echo "特徴量数:"
    cat output/experiments/$EXPERIMENT_NAME/preprocessed/train_metadata.json | jq '.feature_names.tabular | length'
else
    echo "メタデータファイルが見つかりません"
fi

echo "=== 前処理完了 ===" 