# CMI 2025 LSTM v2 Submission

## モデル情報
- **タイムスタンプ**: 20250709_085324
- **CMIスコア**: 0.7946
- **Binary F1**: 0.9515
- **Macro F1**: 0.6377
- **テスト精度**: 0.6096
- **学習エポック数**: 150
- **ウィンドウ設定**: w64_s16

## モデルアーキテクチャ
- **融合方式**: attention
- **LSTM Units**: 80 → 64
- **Dense Units**: 48
- **Demographics Dense Units**: 24
- **Fusion Dense Units**: 48
- **Dropout Rate**: 0.15
- **Learning Rate**: 0.003602554208760558
- **Batch Size**: 32

## ファイル構成
- `final_model_20250709_085324.keras` - メインモデルファイル
- `final_model_20250709_085324.weights.h5` - モデル重み
- `final_model_20250709_085324_architecture.json` - モデルアーキテクチャ
- `model_config.json` - 設定情報
- `model_inference.py` - 推論スクリプト
- `submit_final_model.py` - 提出用スクリプト

## 使用方法

### 推論実行
```bash
python model_inference.py --input_data path/to/test_data.csv --output predictions.csv
```

### 提出用ファイル生成
```bash
python submit_final_model.py
```

## 性能詳細
- 最適化時スコア: 0.8021
- 学習完了時刻: 2025-07-09 09:25:22

## 注意事項
- このモデルは150エポックで学習されました
- 最適なパラメータは Optuna による最適化結果を使用
- GPU環境での学習を推奨します
