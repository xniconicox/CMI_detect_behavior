# 統合モデル学習フロー

このドキュメントでは、`MultimodalTrainer` を用いた統合モデルの学習手順をまとめます。
IMU ウィンドウ、Demographics、Tabular、ToF ボクセルの四種類をまとめて学習する構成です。

## 1. 前提条件
- `scripts/run_preprocessing.py` を実行し、`train_windows.pkl` や `train_tof_windows.pkl` など
  必要な前処理ファイルを生成しておきます。
- 実験用ディレクトリは `output/experiments/<experiment_name>/preprocessed/` とします。

## 2. ディレクトリ構成
```
output/
└─ experiments/
   └─ <experiment_name>/
       ├─ preprocessed/
       │   ├─ train_windows.pkl
       │   ├─ train_demographics.pkl
       │   ├─ train_tabular.pkl
       │   ├─ train_tof_windows.pkl
       │   └─ train_labels.pkl
       └─ models/
```
各ファイルの内容は [preprocessing_outputs.md](preprocessing_outputs.md) を参照してください。

## 3. 学習の実行
1. Python スクリプトを実行します。
   ```bash
   python src/trainers/multimodal_trainer.py
   ```
2. `MultimodalTrainer` は下記ファイル名でデータを読み込みます。
   - `train_windows.pkl`
   - `train_demographics.pkl`
   - `train_tabular.pkl`
   - `train_tof_windows.pkl`
   - `train_labels.pkl`
   これらは `load_all_data()` 内で読み込まれます。

## 4. モデル概要
`build_multimodal_model()` では時系列用の LSTM、人口統計と表形式入力の全結合層、
ToF 用の 3D-CNN をそれぞれタワーとして構築し、最後に結合します。
必要に応じてエポック数やバッチサイズを `train()` の引数で変更してください。

## 5. 出力
学習後、モデルは `output/experiments/<experiment_name>/models/multimodal_model.keras`
として保存されます。学習履歴は `MultimodalTrainer.history` プロパティから取得できます。

また、実験ディレクトリには次のファイルが生成されます。
- `models/` ディレクトリ: 学習済みモデル
- `results/` ディレクトリ: `training_history.json` と `evaluation_results.json`
これらの履歴ファイルは `src/scripts/visualize_training_history.py` を使って可視化できます。

## 6. 推論・評価
`evaluate()` を呼び出すとテストデータに対する Macro F1 が計算されます。
予測値を保存して Kaggle 形式に変換することで提出ファイルを作成できます。




## usage

bash scripts/run_multimodal_training.sh