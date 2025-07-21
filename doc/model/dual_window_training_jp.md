# 64と128ウィンドウ併用学習ガイド

このドキュメントでは、ウィンドウサイズ64と128で前処理したデータを両方用いて学習する方法の例を示します。以下の2通りのアプローチを紹介します。

- **モデルアンサンブル**: それぞれのウィンドウ長で独立にモデルを学習し、予測確率を平均して最終予測を求める方法。
- **特徴量統合**: 両ウィンドウのテンソルを特徴次元で連結し、1つのモデルに入力する方法。

## 1. 前処理の準備

まず `scripts/run_preprocessing.sh` をウィンドウサイズ64と128で実行し、
`output/experiments/<exp_ws64>/preprocessed/` と
`output/experiments/<exp_ws128>/preprocessed/` を用意します。

```bash
# 例: ウィンドウ64と128の前処理
WINDOW_SIZE=64 bash scripts/run_preprocessing.sh
WINDOW_SIZE=128 bash scripts/run_preprocessing.sh
```

## 2. モデルアンサンブル例

それぞれの前処理結果を用いて `run_multimodal_training_v40.sh` を実行し、個別にモデルを学習します。

```bash
# ウィンドウ64で学習
EXPERIMENT_NAME="exp_ws64" bash scripts/run_multimodal_training_v40.sh

# ウィンドウ128で学習
EXPERIMENT_NAME="exp_ws128" bash scripts/run_multimodal_training_v40.sh
```

学習後、各実験ディレクトリの `results/prediction_probs.npy` を読み込み、平均を取ってアンサンブル予測を作成します。以下は簡易的なPythonコード例です。

```python
import numpy as np

probs64 = np.load('output/experiments/exp_ws64/results/prediction_probs.npy')
probs128 = np.load('output/experiments/exp_ws128/results/prediction_probs.npy')
ensemble_probs = (probs64 + probs128) / 2
pred_labels = ensemble_probs.argmax(axis=1)
```

## 3. 特徴量統合例

両方の前処理データを読み込んで特徴次元で連結し、ひとつの `MultimodalTrainerV31` に渡します。ウィンドウ長が異なるため、短い方をゼロパディングして長さを揃える方法が簡単です。

```python
from pathlib import Path
import numpy as np
import pickle
from src.trainers.multimodal_trainer_v31 import MultimodalTrainerV31

base = Path('output/experiments')
ws64 = base / 'exp_ws64' / 'preprocessed'
ws128 = base / 'exp_ws128' / 'preprocessed'

with open(ws64 / 'train_windows.pkl', 'rb') as f:
    x64 = pickle.load(f)
with open(ws128 / 'train_windows.pkl', 'rb') as f:
    x128 = pickle.load(f)

pad_len = x128.shape[1] - x64.shape[1]
if pad_len > 0:
    x64 = np.pad(x64, [(0,0), (0,pad_len), (0,0)])

x_concat = np.concatenate([x64, x128], axis=2)

with open(ws64 / 'train_demographics.pkl', 'rb') as f:
    demo = pickle.load(f)
with open(ws64 / 'train_labels.pkl', 'rb') as f:
    labels = pickle.load(f)

trainer = MultimodalTrainerV31('dual_window_example')
data = {
    'sensor': x_concat,
    'demographics': demo,
    'tabular': np.load(ws64 / 'train_tabular.npy'),
    'tof': np.load(ws64 / 'train_tof_windows.npy'),
    'labels': labels,
}
trainer.train_cross_validation(data)
```

これにより、2種類のウィンドウ情報を組み合わせたモデルを学習できます。
