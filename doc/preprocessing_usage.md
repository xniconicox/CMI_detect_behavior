# 前処理パイプラインの使い方

このリポジトリでは `src/utils/pipeline.py` に定義された `Preprocessor` クラスを用いて、学習時と推論時で同じ前処理を適用できます。以下ではリポジトリ更新方法から前処理実行までをまとめます。

## 1. リポジトリを最新状態に更新

リモートリポジトリが設定されている場合は次のコマンドで最新の変更を取得できます。

```bash
git pull
```

依存パッケージをインストールしていない場合は、以下を実行してください。

```bash
pip install -r requirements.txt
```

## 2. データの配置

`data/` ディレクトリに `train.csv`、`train_demographics.csv`、`test.csv`、`test_demographics.csv` を配置します。

## 3. 前処理の実行

基本的な実行方法は `scripts/run_preprocessing.py` を使用します。実験名を指定すると、出力は `output/experiments/<experiment_name>/preprocessed/` に保存されます。

```bash
# 学習用データで前処理を学習しテストも変換
python scripts/run_preprocessing.py \
    --experiment-name exp1 \
    --config config/config_v2.yaml \
    --use-cache \
    --mode train
```

推論専用で実行する場合は保存済みの `preprocessor.pkl` を読み込み、次のように実行します。

```bash
python scripts/run_preprocessing.py \
    --experiment-name exp1 \
    --mode predict \
    --use-cache
```

各オプションの意味は以下の通りです。

- `--config` : YAML 形式の設定ファイル。デフォルトは `config/config_v2.yaml`。
- `--use-cache` : 中間結果のキャッシュを利用して計算を省略します。
- `--mode train` : `fit` と `transform` を実行して学習済み `Preprocessor` を保存します。
- `--mode predict` : `preprocessor.pkl` を読み込み `transform` のみ実行します。
- `--log-file` : ログを保存するファイルパス。省略時は
  `output_dir/<experiment_name>/preprocessed/preprocess.log` に出力されます。

## 4. 出力ファイル

生成されるファイルの詳細は [preprocessing_outputs.md](preprocessing_outputs.md) を参照してください。主に以下が保存されます。

- `train_windows.pkl` / `test_windows.pkl`
- `train_tabular.pkl` / `test_tabular.pkl`
- `train_tof_windows.pkl` / `test_tof_windows.pkl` (旧 `train_tof_voxel.pkl` / `test_tof_voxel.pkl`)
- `preprocessor.pkl` : 学習済み前処理器

## 5. コードからの利用例

前処理済みファイルを使わず直接クラスを利用する場合は次のようにします。

```python
from pathlib import Path
import pandas as pd
from src.utils.pipeline import Preprocessor

# 設定ファイル読み込み
config = yaml.safe_load(Path("config/config_v2.yaml").read_text())

# 学習時
df_train = pd.read_csv("data/train.csv")
pp = Preprocessor(config)
pp.fit(df_train)
preprocessed = pp.transform(df_train)
pp.save(Path("output/experiments/exp1/preprocessed/preprocessor.pkl"))

# 推論時
df_test = pd.read_csv("data/test.csv")
pp = Preprocessor.load(Path("output/experiments/exp1/preprocessed/preprocessor.pkl"))
preprocessed_test = pp.transform(df_test)
```

このように、`Preprocessor.save` と `Preprocessor.load` を使うことで学習時と同じ正規化パラメータを再利用できます。
`pp.save` で生成される `preprocessor.pkl` には `sensor_scaler`、`demo_scaler`、`tab_scaler` の
各 `StandardScaler` が保存され、推論時は `Preprocessor.load` により自動的に読み込まれます。
