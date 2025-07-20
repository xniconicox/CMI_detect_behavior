# 特徴量設定ガイド

## 📋 概要

CMI競技プロジェクトでは、複数の特徴量計算オプションが設定可能です。このドキュメントでは、各特徴量の詳細と設定方法について説明します。

## 🔧 設定可能な特徴量

### 1. 基本統計量特徴量（常に有効）
- **機能**: 平均、標準偏差、RMS、エネルギー、範囲など
- **計算関数**: `compute_basic_statistics`
- **特徴量数**: 約60次元
- **設定**: 常に計算される（無効化不可）

### 2. ピーク特徴量（常に有効）
- **機能**: ピーク数、周期情報
- **計算関数**: `compute_peak_features`
- **特徴量数**: 約15次元
- **設定**: 常に計算される（無効化不可）

### 3. FFTバンドエネルギー（常に有効）
- **機能**: 0.5-20Hzの周波数帯エネルギー
- **計算関数**: `compute_fft_band_energy`
- **特徴量数**: 約16次元
- **設定**: 常に計算される（無効化不可）

### 4. ウェーブレット特徴量（オプション）
- **機能**: DWTバンドエネルギー
- **計算関数**: `compute_wavelet_features`
- **特徴量数**: 約15次元
- **設定**: `use_wavelet_features: true/false`
- **デフォルト**: `true`

### 5. TDA特徴量（オプション）
- **機能**: 位相的特徴量（Persistent Homology）
- **計算関数**: `compute_persistence_image_features_batch`
- **特徴量数**: 約8次元
- **設定**: `use_tda_features: true/false`
- **デフォルト**: `true`

### 6. ToF変化量特徴量（オプション）
- **機能**: ToFセンサのフレーム間変化量
- **計算関数**: `compute_tof_rate_of_change`
- **特徴量数**: 5次元（深度ごと）
- **設定**: `use_tof_rate_features: true/false`
- **デフォルト**: `true`（新規設定）

### 7. 温度変化量特徴量（オプション）
- **機能**: 温度センサーの変化量
- **計算関数**: `compute_temperature_change_features`
- **特徴量数**: 5次元（センサーごと）
- **設定**: `use_temperature_change_features: true/false`
- **デフォルト**: `true`（新規設定）

### 8. オートエンコーダー再構成誤差（オプション）
- **機能**: AE再構成誤差
- **計算関数**: `compute_autoencoder_reconstruction_error`
- **特徴量数**: 約15次元
- **設定**: `ae_model_path`が設定されている場合のみ

## 📊 特徴量次元数の内訳

| 特徴量タイプ | 次元数 | デフォルト |
|-------------|--------|-----------|
| 基本統計量 | ~60 | 有効 |
| ピーク特徴量 | ~15 | 有効 |
| FFTバンドエネルギー | ~16 | 有効 |
| ウェーブレット | ~15 | 有効 |
| TDA | ~8 | 有効 |
| ToF変化量 | 5 | 有効 |
| 温度変化量 | 5 | 有効 |
| デモグラフィクス | 7 | 有効 |
| **合計** | **~131** | - |

## ⚙️ 設定方法

### 1. 設定ファイルでの変更

```yaml
# config/config_v2.yaml
preprocessing:
  # 各特徴量計算の有無を制御（デフォルトで有効）
  use_wavelet_features: true
  use_tda_features: true
  # ToFセンサのフレーム間変化量特徴を計算するか
  use_tof_rate_features: true
  # 温度センサーの変化量特徴を計算するか
  use_temperature_change_features: true
```

### 2. スクリプトでの変更

```bash
# scripts/run_preprocessing.sh
AUGMENT_HANDEDNESS=true                 # 利き手反転によるデータ拡張（デフォルト有効）
```

## 🎯 推奨設定

### 1. 高精度設定（デフォルト）
```yaml
use_wavelet_features: true
use_tda_features: true
use_tof_rate_features: true
use_temperature_change_features: true
```
- **特徴量数**: ~131次元
- **精度**: 最高
- **計算時間**: 長い

### 2. バランス設定
```yaml
use_wavelet_features: true
use_tda_features: false
use_tof_rate_features: true
use_temperature_change_features: true
```
- **特徴量数**: ~123次元
- **精度**: 高
- **計算時間**: 中

### 3. 軽量設定
```yaml
use_wavelet_features: false
use_tda_features: false
use_tof_rate_features: false
use_temperature_change_features: false
```
- **特徴量数**: ~108次元
- **精度**: 中
- **計算時間**: 短

## 🔄 特徴量の有効/無効化の影響

### 1. 計算時間への影響
- **ウェーブレット**: +20-30%
- **TDA**: +50-100%
- **ToF変化量**: +10-15%
- **温度変化量**: +5-10%

### 2. 精度への影響
- **ウェーブレット**: 時系列パターンの詳細な分析
- **TDA**: 位相的な構造の検出
- **ToF変化量**: 動的な距離変化の検出
- **温度変化量**: 温度変化パターンの検出

### 3. メモリ使用量への影響
- **ウェーブレット**: +10-15%
- **TDA**: +20-30%
- **ToF変化量**: +5-10%
- **温度変化量**: +5%

## 🚀 最適化のヒント

### 1. メモリ不足の場合
```yaml
# メモリ使用量を削減
use_tda_features: false
use_wavelet_features: false
```

### 2. 計算時間を短縮したい場合
```yaml
# 計算時間を短縮
use_tda_features: false
use_tof_rate_features: false
```

### 3. 精度を重視する場合
```yaml
# 全特徴量を有効化
use_wavelet_features: true
use_tda_features: true
use_tof_rate_features: true
use_temperature_change_features: true
```

## 📝 設定変更の手順

### 1. 設定ファイルの編集
```bash
# config/config_v2.yamlを編集
vim config/config_v2.yaml
```

### 2. キャッシュのクリア
```bash
# 古いキャッシュを削除
rm -rf cache/
```

### 3. 前処理の再実行
```bash
# 新しい設定で前処理実行
./scripts/run_preprocessing.sh
```

## 🔍 特徴量の確認方法

### 1. ログでの確認
```bash
# 前処理ログで特徴量数を確認
tail -f output/experiments/*/preprocessed/preprocess.log
```

### 2. コードでの確認
```python
# 特徴量の形状を確認
print(f"特徴量形状: {features.shape}")
print(f"特徴量数: {features.shape[1]}")
```

## ⚠️ 注意事項

### 1. キャッシュの影響
- 設定変更後はキャッシュをクリアする必要があります
- キャッシュが残っていると古い設定が使用されます

### 2. メモリ使用量
- 全特徴量を有効にするとメモリ使用量が増加します
- システムリソースに応じて設定を調整してください

### 3. 計算時間
- TDA特徴量は特に計算時間が長いです
- 開発時は軽量設定を使用することを推奨します

## 📚 参考資料

- [前処理パイプライン概要](doc/preprocess/preprocessing_pipeline.md)
- [特徴量エンジニアリング](src/utils/feature_engineering.py)
- [パイプライン実装](src/utils/pipeline.py)

---

このガイドに従って特徴量設定を調整することで、プロジェクトの要件に最適化された前処理を実行できます。 