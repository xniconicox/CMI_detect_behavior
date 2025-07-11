# 入力データ概要

このプロジェクトではセンサーデータとDemographicsデータを組み合わせてジェスチャー認識を行います。基本的なデータ構成は次の通りです。

## データセット構成

```
data/
├── train.csv
├── train_demographics.csv
├── test.csv
└── test_demographics.csv
```

- `train.csv` / `test.csv` : センサ時系列データ
- `train_demographics.csv` / `test_demographics.csv` : 被験者属性情報

## センサ特徴量

`train.csv` の列名から `acc_`, `rot_`, `tof_`, `thm_` で始まるものをセンサ列として扱います【F:notebooks/lstm_v2_preprocessing_fixed.py†L253-L259】。
主なセンサは以下の4種類です。

| センサ種別 | 例 | 説明 |
|------------|----|------|
| Accelerometer | `acc_x`, `acc_y`, `acc_z` | 加速度 (3軸) |
| Rotation/Quaternion | `rot_w`, `rot_x`, `rot_y`, `rot_z` | 回転 (姿勢) |
| Thermal | `thm_1`, `thm_2`, `thm_3` | 温度センサ |
| ToF (Time-of-Flight) | `tof_1_v0` ～ | 距離画像 (複数ピクセル) |

## Demographics特徴量

`create_demographics_features` 関数では年齢グループや身長カテゴリなどの派生特徴量を生成し、最終的には One-Hot を含む複数の属性列を作成します【F:notebooks/lstm_v2_preprocessing_fixed.py†L96-L131】。

## 前処理パイプライン

`src/utils/preprocessing.py` では A～M のブロックに分けて特徴量を生成します。
主なブロックと次元数は以下の通りです【F:doc/preprocessing_pipeline.md†L5-L27】。

| ブロック | 関数 | 次元 | 説明 |
|---------|------|-----|------|
| A. 基本統計量 | `compute_basic_statistics` | 56 | mean, std, range など |
| B. ピーク&周期 | `compute_peak_features` | 18 | ピーク数の集計 |
| C. FFTエネルギー | `compute_fft_band_energy` | 10 | 0.5–20Hz の周波数帯エネルギー |
| D. ワールド線形加速度 | `linear_acceleration` など | 21 | 姿勢を除いた加速度 |
| E. 欠損フラグ | `add_missing_sensor_flags` | 3 | センサ欠測をフラグ化 |
| F. Demographics | (窓生成時に結合) | 4 | 性別・利き手等 |
| G. TDA統計量 | `compute_persistence_image_features` | 8 | 位相的特徴 |
| H. AE再構成誤差 | `compute_autoencoder_reconstruction_error` | 4 | 異常度スコア |
| I. 合成Tabular | (A～H 結合) | 120 | LightGBM用特徴量 |
| J. IMUウィンドウテンソル | `create_sliding_windows_with_demographics` | 1792 | LSTM用入力(7×256) |
| K. ToF 3D-Voxel | `tof_to_voxel_tensor` | 20480 | ToF CNN用入力 |
| L. 利き手反転 | `handedness_correction_v2` | - | 左利きサンプルのY/Z反転 |
| M. Wavelet特徴量 | `compute_wavelet_features` | 変動 | DWTエネルギー |

## 欠損値処理

- `clean_sensor_missing_values` ではセンサ種別ごとに欠損値の扱いを定めています。例えば ToF センサでは `-1` を欠損として `NaN` に置き換えます【F:src/utils/preprocessing.py†L355-L375】。
- `clean_missing_sensor_data_parallel_disk` を利用することで、長い欠損は前後補完や0埋めを行いながら並列処理できます【F:src/utils/preprocessing.py†L256-L330】。
- 欠損の有無は `add_missing_sensor_flags` によりフラグ化され、モデル入力に利用できます【F:src/utils/preprocessing.py†L109-L120】。

## データ次元の例

前処理後、ウィンドウ化されたセンサデータは `(サンプル数, ウィンドウ長, センサ特徴量数)` の形をとります。`load_w128_s32_data` 関数では読み込み時に以下の情報が表示されます【F:src/data/data_loader_fix.py†L66-L112】。

```
📊 全サンプル数: <n_samples>
🪟 ウィンドウサイズ: <window_size>
🔢 特徴量数: <n_features>
👥 Demographics特徴量数: <n_demographics>
🏷️ クラス数: <n_classes>
```

## 参考

詳細な実装は `src/utils/preprocessing.py` と `notebooks/lstm_v2_preprocessing_fixed.py` を参照してください。EDA結果や欠損統計の例は `doc/EDA.html` にまとめられています【F:doc/EDA.html†L8566-L8590】。
