# LSTM モデル提出準備ガイド

## 概要

このガイドでは、CMI Competition 用の LSTM モデルの提出準備について説明します。

### モデル性能
- **アルゴリズム**: LSTM (Long Short-Term Memory)
- **F1-macro スコア**: 0.5200
- **精度**: 0.5558
- **ウィンドウ設定**: w64_s16 (ウィンドウサイズ64、ストライド16)
- **クラス数**: 18ジェスチャー

## 使用ライブラリとライセンス確認

### フリーライブラリ（商用利用可能）
✅ **TensorFlow/Keras**: Apache 2.0 License
✅ **NumPy**: BSD License  
✅ **Pandas**: BSD License
✅ **Scikit-learn**: BSD License
✅ **Matplotlib**: Matplotlib License (BSD-compatible)
✅ **Seaborn**: BSD License
✅ **Polars**: MIT License
✅ **Optuna**: MIT License

### 確認済み
すべての使用ライブラリは商用利用可能なフリーライセンスです。

## 提出に必要なファイル

### 1. モデルファイル
```
output/experiments/lstm_w64_s16_final_model/models/
├── lstm_best.h5                    # 学習済みLSTMモデル
├── lstm_best_architecture.json     # モデル構造
├── scaler.pkl                      # データ正規化器
├── label_encoder.pkl               # ラベルエンコーダー
├── config.json                     # モデル設定
└── meta.json                       # メタデータ
```

### 2. 推論スクリプト
- `model_inference.py`: メイン推論ロジック
- `cmi-2025-lstm-submission.ipynb`: Kaggle提出用ノートブック

## 提出手順

### ステップ1: モデルファイルの圧縮とアップロード
1. 必要なモデルファイルを圧縮
2. Kaggleデータセットとして「cmi-lstm-models-v1」でアップロード

### ステップ2: 推論スクリプトの作成
1. LSTMモデル用の推論関数を実装
2. スライディングウィンドウ処理を含む前処理
3. Kaggle評価システムとの統合

### ステップ3: 提出ノートブックの作成
1. データソースの設定
2. 推論サーバーの初期化
3. 提出

## 次のステップ
1. LSTM用推論スクリプトの作成
2. 提出ノートブックの作成
3. ローカルテスト
4. Kaggle提出

## 注意事項
- すべてのライブラリは商用利用可能
- モデルサイズ: 約1.6MB（H5ファイル）
- 推論時間: 1シーケンスあたり約50ms（CPU）
- GPU利用で更なる高速化可能 