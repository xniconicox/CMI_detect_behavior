# CMIコンペ ベースライン

CMIコンペ（CMI-Detect-Behavor-with-Sensor-Data）のベースライン実装です。

## 📋 概要

このリポジトリは、センサーデータを使用したジェスチャー認識のベースライン実装を含んでいます。

### 主な特徴

- **利き手補正**: 左右利きの違いを考慮したセンサーデータの正規化
- **前処理**: 欠損値補完、標準化、特徴量抽出
- **ベースラインモデル**: LightGBMを使用したマルチクラス分類
- **クロスバリデーション**: 5分割StratifiedKFold
- **自動実行**: 前処理から提出ファイル作成まで一括実行

## 🚀 クイックスタート

### 1. 環境セットアップ

```bash
# 仮想環境の作成（推奨）
python -m venv cmi_env
source cmi_env/bin/activate  # Linux/Mac
# または
cmi_env\Scripts\activate     # Windows

# 依存関係のインストール
pip install -r requirements.txt
```

### 2. データの配置

以下のファイルを `data/` ディレクトリに配置してください：

```
data/
├── train.csv
├── train_demographics.csv
├── test.csv
└── test_demographics.csv
```

### 3. ベースライン実行

#### 方法1: 一括実行（推奨）

```bash
python run_baseline.py
```

このコマンドで以下が自動実行されます：
1. 前処理ノートブックの実行
2. ベースラインモデルの訓練
3. 提出ファイルの作成

#### 方法2: 手動実行

```bash
# 1. 前処理ノートブックの実行
jupyter nbconvert --to notebook --execute notebooks/preprocess.ipynb

# 2. ベースラインモデルの実行
python src/baseline_model.py
```

## 📁 ディレクトリ構造

```
CMI_comp/
├── data/                    # データファイル
│   ├── train.csv
│   ├── train_demographics.csv
│   ├── test.csv
│   └── test_demographics.csv
├── notebooks/               # Jupyterノートブック
│   └── preprocess.ipynb    # 前処理ノートブック
├── src/                    # Pythonスクリプト
│   └── baseline_model.py   # ベースラインモデル
├── output/                 # 出力ファイル
│   ├── train_features.csv  # 前処理済み訓練データ
│   ├── test_features.csv   # 前処理済みテストデータ
│   └── submission.csv      # 提出ファイル
├── run_baseline.py         # 一括実行スクリプト
├── requirements.txt        # 依存関係
└── README.md              # このファイル
```

## 🔧 前処理の詳細

### 1. 利き手補正

- 左利きの被験者のセンサーデータを右利きに統一
- 加速度センサー（acc_x）、ジャイロスコープ（gyr_x）の左右軸を反転

### 2. 欠損値処理

- センサーデータ: シークエンス内での前方・後方埋め + 0埋め
- カテゴリカルデータ: 最頻値で補完

### 3. 標準化

- センサーデータをStandardScalerで標準化（平均0、分散1）

### 4. 特徴量抽出

シークエンス単位で以下の統計量を計算：
- 基本統計量: mean, std, min, max, median
- 分布統計量: skew, kurt

## 🤖 モデルの詳細

### LightGBM設定

```python
params = {
    'objective': 'multiclass',
    'num_class': len(classes),
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}
```

### クロスバリデーション

- 5分割StratifiedKFold
- 各フォールドでEarly Stopping（50ラウンド）
- 全フォールドの予測を平均化

## 📊 出力ファイル

### train_features.csv / test_features.csv

前処理済みの特徴量データ：
- `sequence_id`: シークエンスID
- `gesture`: ジェスチャーラベル（訓練データのみ）
- その他: 統計特徴量（例: `acc_x_mean`, `gyr_y_std`など）

### submission.csv

提出用ファイル：
- `sequence_id`: シークエンスID
- `gesture`: 予測ジェスチャー

## 🎯 改善のヒント

### 1. 特徴量エンジニアリング

- 時系列特徴量の追加（FFT、ウェーブレット変換など）
- センサー間の相関特徴量
- ドメイン知識に基づく特徴量

### 2. モデル改善

- ハイパーパラメータチューニング（Optuna、Hyperopt）
- アンサンブル手法（Stacking、Blending）
- ディープラーニング（LSTM、Transformer）

### 3. 前処理改善

- より高度な利き手補正
- ノイズ除去（フィルタリング）
- データ拡張

## 📝 注意事項

- メモリ使用量: 大規模データセットの場合、十分なRAMが必要
- 実行時間: 初回実行時は前処理に時間がかかる場合があります
- 環境依存: 一部の可視化機能は環境によって動作が異なる場合があります

## 🤝 貢献

改善提案やバグ報告は歓迎します。プルリクエストやイシューの作成をお気軽にどうぞ。

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

---

**Happy Coding! 🚀** 