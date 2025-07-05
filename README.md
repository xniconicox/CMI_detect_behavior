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

### 1. データクリーニング

- 欠損値の処理（前方・後方埋め、0埋め）
- 異常値の検出と処理
- データ型の統一

### 2. 特徴量抽出

シークエンス単位で以下の統計量を計算：
- **基本統計量**: mean, std, min, max, median
- **分布統計量**: skew, kurtosis
- **センサー別特徴量**: 加速度、回転、熱、ToFセンサー

### 3. データ正規化

- StandardScalerによる標準化
- センサー間のスケール統一

## 🤖 モデルの詳細

### LightGBM設定

```python
params = {
    'objective': 'multiclass',
    'num_class': 18,
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

- **5分割StratifiedKFold**
- **Early Stopping**: 50ラウンド
- **評価指標**: F1-Score (macro average)

## 📊 結果と分析

### ベースライン性能

- **CV F1-Score**: 0.5890 (±0.0148)
- **特徴量数**: 2,324
- **クラス数**: 18

### 特徴量重要度（Top 5）

1. `acc_y_max` (12,681.93) - 加速度センサーY軸最大値
2. `thm_2_median` (9,472.27) - 熱センサー2中央値
3. `thm_3_median` (6,255.71) - 熱センサー3中央値
4. `thm_2_mean` (6,182.40) - 熱センサー2平均値
5. `rot_y_std` (4,876.89) - 回転センサーY軸標準偏差

### センサー別重要度

- **Accelerometer**: 2,008.63 (最も重要)
- **Thermal**: 1,211.76
- **Rotation**: 1,008.46
- **ToF**: 98.55 (最も低い)

## 📈 可視化機能

### 1. データ分布

- ラベル分布（棒グラフ・円グラフ）
- 特徴量分布（ヒストグラム）
- センサー別データ分布

### 2. 学習結果

- クロスバリデーションスコア
- 特徴量重要度
- 混同行列
- クラス別性能

### 3. 予測分析

- 予測分布
- 確率分布
- 誤分類分析

## 🎯 改善のヒント

### 1. 特徴量エンジニアリング

- **時系列特徴量**: FFT、ウェーブレット変換
- **センサー融合**: マルチモーダル特徴量
- **ドメイン知識**: ジェスチャー固有の特徴量

### 2. モデル改善

- **ハイパーパラメータチューニング**: Optuna、Hyperopt
- **アンサンブル**: Stacking、Blending
- **ディープラーニング**: LSTM、Transformer、CNN

### 3. 前処理改善

- **ノイズ除去**: フィルタリング、スムージング
- **データ拡張**: 回転、ノイズ追加
- **クラス重み**: 不均衡データの処理

## 📝 使用方法

### 1. 前処理の実行

```python
# ノートブック内で実行
# notebooks/preprocess.ipynb を参照
```

### 2. モデル学習

```python
# ノートブック内で実行
# notebooks/train_and_visualize.ipynb を参照
```

### 3. 結果の確認

```bash
# 提出ファイルの確認
head output/submission.csv

# 実験結果の確認
ls output/experiments/
```

## 📝 注意事項

- **メモリ要件**: 大規模データセットの場合、8GB以上のRAMを推奨
- **実行時間**: 初回実行時は前処理に時間がかかる場合があります
- **環境依存**: 一部の可視化機能は環境によって動作が異なる場合があります

## 🔧 トラブルシューティング

### よくある問題

1. **メモリ不足**
   - バッチサイズの削減
   - 特徴量数の削減

2. **パスエラー**
   - ノートブックが正しいディレクトリから実行されているか確認
   - 相対パスの確認

3. **ライブラリエラー**
   - `pip install -r requirements.txt` の再実行
   - 仮想環境の確認

## 🤝 貢献

改善提案やバグ報告は歓迎します。プルリクエストやイシューの作成をお気軽にどうぞ。

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。

---

**Happy Coding! 🚀**

*CMIコンペで良い結果を目指しましょう！* 