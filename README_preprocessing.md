# CMI競技前処理ガイド

## 📋 概要

CMI競技プロジェクトの前処理パイプラインについて説明します。センサーデータとデモグラフィクス情報を統合し、機械学習モデル用の特徴量を生成します。

## 🚀 クイックスタート

### 基本的な前処理実行

```bash
# 環境準備
source .venv/bin/activate

# 標準的な前処理実行
./scripts/run_preprocessing.sh

# 最適化された前処理実行
./scripts/run_preprocessing_optimized.sh

# 予測モードでの実行
./scripts/run_preprocessing_optimized.sh predict
```

## 📁 ファイル構成

```
scripts/
├── run_preprocessing.sh              # 標準前処理スクリプト
└── run_preprocessing_optimized.sh    # 最適化前処理スクリプト

config/
└── config_v2.yaml                    # 前処理設定ファイル

src/utils/
├── preprocessing.py                  # メイン前処理モジュール
├── pipeline.py                       # パイプラインコンポーネント
├── feature_engineering.py            # 特徴量エンジニアリング
└── io_utils.py                       # I/Oユーティリティ
```

## 🔧 設定オプション

### 1. 基本設定

```yaml
# config/config_v2.yaml
preprocessing:
  window_size: 128                    # ウィンドウサイズ
  stride: 64                          # ストライド
  sampling_rate: 50.0                 # サンプリングレート
  use_world_acc: true                 # ワールド座標変換
```

### 2. 特徴量設定（デフォルトで有効）

```yaml
preprocessing:
  # 各特徴量計算の有無を制御（デフォルトで有効）
  use_wavelet_features: true          # ウェーブレット特徴量
  use_tda_features: true              # TDA特徴量
  use_tof_rate_features: true         # ToF変化量特徴量
  use_temperature_change_features: true # 温度変化量特徴量
```

### 3. スクリプト設定

```bash
# scripts/run_preprocessing.sh
EXPERIMENT_NAME="preproc-v2"          # 実験名
USE_CACHE=false                       # キャッシュ利用
MODE="train"                          # train または predict
AUGMENT_HANDEDNESS=true               # 利き手反転によるデータ拡張
```

## 🧪 特徴量の詳細

### 1. 基本統計量特徴量（常に有効）
- **機能**: 平均、標準偏差、RMS、エネルギー、範囲
- **特徴量数**: 約60次元
- **計算関数**: `compute_basic_statistics`

### 2. ピーク特徴量（常に有効）
- **機能**: ピーク数、周期情報
- **特徴量数**: 約15次元
- **計算関数**: `compute_peak_features`

### 3. FFTバンドエネルギー（常に有効）
- **機能**: 0.5-20Hzの周波数帯エネルギー
- **特徴量数**: 約16次元
- **計算関数**: `compute_fft_band_energy`

### 4. オプション特徴量

| 特徴量タイプ | デフォルト | 特徴量数 | 計算時間 |
|-------------|-----------|----------|----------|
| ウェーブレット | 有効 | ~15次元 | +20-30% |
| TDA | 有効 | ~8次元 | +50-100% |
| ToF変化量 | 有効 | 5次元 | +10-15% |
| 温度変化量 | 有効 | 5次元 | +5-10% |

## ⚙️ 実行方法

### 1. 標準実行
```bash
./scripts/run_preprocessing.sh
```

### 2. 最適化実行
```bash
./scripts/run_preprocessing_optimized.sh
```

### 3. 予測モード
```bash
./scripts/run_preprocessing_optimized.sh predict
```

### 4. カスタム設定
```bash
# 設定を変更して実行
vim config/config_v2.yaml
./scripts/run_preprocessing_optimized.sh
```

## 📊 出力結果

### 1. ファイル構造
```
output/experiments/preproc-v2-optimized/
├── preprocessed/
│   ├── train_features.npy           # 学習用特徴量
│   ├── train_labels.npy             # 学習用ラベル
│   ├── test_features.npy            # テスト用特徴量
│   ├── test_labels.npy              # テスト用ラベル
│   └── preprocess.log               # 処理ログ
└── config/
    └── config_v2.yaml               # 使用した設定ファイル
```

### 2. 特徴量の形状
- **Tabular特徴量**: (N, ~131)
- **IMUウィンドウ**: (N, 128, 15)
- **ToFボクセル**: (N, 128, 5, 8, 8)
- **デモグラフィクス**: (N, 7)

## 🔄 最適化オプション

### 1. メモリ不足の場合
```yaml
# config/config_v2.yaml
preprocessing:
  use_tda_features: false             # TDA特徴量を無効化
  use_wavelet_features: false         # ウェーブレット特徴量を無効化
```

### 2. 計算時間を短縮したい場合
```yaml
# config/config_v2.yaml
preprocessing:
  use_tda_features: false             # TDA特徴量を無効化
  use_tof_rate_features: false        # ToF変化量特徴量を無効化
```

### 3. 精度を重視する場合
```yaml
# config/config_v2.yaml
preprocessing:
  use_wavelet_features: true          # 全特徴量を有効化
  use_tda_features: true
  use_tof_rate_features: true
  use_temperature_change_features: true
```

## ⚠️ トラブルシューティング

### 1. メモリ不足エラー
```bash
# 軽量設定で実行
# config/config_v2.yamlで特徴量を無効化
./scripts/run_preprocessing_optimized.sh
```

### 2. キャッシュの問題
```bash
# キャッシュをクリア
rm -rf cache/
./scripts/run_preprocessing_optimized.sh
```

### 3. 設定ファイルエラー
```bash
# 設定ファイルの構文チェック
python -c "import yaml; yaml.safe_load(open('config/config_v2.yaml'))"
```

## 📝 ログとデバッグ

### 1. ログの確認
```bash
# リアルタイムログ確認
tail -f output/experiments/*/preprocessed/preprocess.log

# エラーログの確認
grep -i error output/experiments/*/preprocessed/preprocess.log
```

### 2. 進捗の確認
```bash
# 処理済みファイルの確認
ls -la output/experiments/*/preprocessed/

# 特徴量の形状確認
python -c "
import numpy as np
features = np.load('output/experiments/preproc-v2-optimized/preprocessed/train_features.npy')
print(f'特徴量形状: {features.shape}')
print(f'特徴量数: {features.shape[1]}')
"
```

## 🎯 ベストプラクティス

### 1. 開発時
- 軽量設定を使用
- キャッシュを有効化
- 小規模データでテスト

### 2. 本番実行時
- 全特徴量を有効化
- キャッシュを有効化
- 十分なメモリを確保

### 3. デバッグ時
- ログを詳細に確認
- 段階的に特徴量を有効化
- メモリ使用量を監視

## 📚 詳細ドキュメント

- **特徴量設定ガイド**: `docs/feature_configuration.md`
- **前処理パイプライン**: `doc/preprocess/preprocessing_pipeline.md`
- **テストガイド**: `docs/testing_guide.md`

## 🔗 関連ファイル

- **設定ファイル**: `config/config_v2.yaml`
- **前処理スクリプト**: `scripts/run_preprocessing*.sh`
- **メイン処理**: `src/utils/preprocessing.py`
- **特徴量計算**: `src/utils/feature_engineering.py`

---

このガイドに従って前処理を実行することで、高品質な特徴量を効率的に生成できます。 