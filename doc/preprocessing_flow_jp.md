# 前処理フローまとめ

本プロジェクトで用いる前処理ステップを日本語で整理します。センサ時系列データと年齢・性別などの **Demographics** 情報をどの順番で処理し、各モデルに入力するかをまとめました。

## 1. 生データのロード

- `train.csv` / `test.csv` … センサ時系列 (加速度, 回転, Thermal, ToF など)
- `train_demographics.csv` / `test_demographics.csv` … 年齢・性別・身長・利き手等

これらを `subject` と `sequence_id` で結合して一つのデータフレームとします。

## 2. 欠損値処理

`src/utils/preprocessing.py` の `clean_sensor_missing_values` などを利用し、センサー種別ごとのルールで欠損値を補完します。ToF の `-1` や加速度の極端値を NaN に置き換え、短い欠損は線形補間、長い欠損は前後値で埋めます。

## 3. IMU ワールド座標変換

- `quaternion_to_rotation_matrix`
- `rotate_acceleration`
- `linear_acceleration`

これらの関数で IMU をワールド座標系に変換し、重力成分を除いた線形加速度 (Block D) を求めます。

## 4. スライディングウィンドウ生成 + Demographics 結合

`create_sliding_windows_with_demographics` を用いて以下を実施します。

1. 時系列を `window_size` と `stride` で分割
2. 各ウィンドウに `sex` `handedness` `height` `arm_len` などの静的特徴量を付与

得られる IMU ウィンドウテンソル (Block J) は LSTM/GRU 系モデルで使用します。

## 5. 正規化

- `normalize_sensor_data` でセンサーデータを z-score 標準化
- `normalize_tabular_data` で Demographics を正規化

## 6. 利き手補正

左利きサンプルのみ `handedness_correction_v2` を適用し、IMU の Y/Z 軸や ToF チャンネルを反転します (Block L)。

## 7. 特徴量抽出

各ウィンドウに対し以下の関数を実行して特徴量を得ます。

- `compute_basic_statistics` … 平均・標準偏差・RMS など (Block A)
- `compute_peak_features` … ピーク数・周期情報 (Block B)
- `compute_fft_band_energy` … 0.5–20 Hz の周波数帯エネルギー (Block C)
- `compute_persistence_image_features` … 位相的特徴量 (Block G)　tda?
- `compute_autoencoder_reconstruction_error` … AE再構成誤差 (Block H)
- `compute_wavelet_features` … DWTバンドエネルギー (Block M)

## 8. 欠損フラグ付与

`add_missing_sensor_flags` で ToF/Thermal/IMU の欠測有無を one-hot で付けます (Block E)。

## 9. Tabular 特徴量への統合

上記 A–H,M と Demographics をまとめて **120次元** の tabular 特徴量 (Block I) を作成します。LightGBM や CatBoost といった勾配ブースティング系モデルに入力します。

## 10. ToF 3D ボクセル変換

`tof_to_voxel_tensor` を使い、ToF ピクセル列から (T, depth, H, W) の 3D テンソル (Block K) を生成します。ToF 3D CNN モデル向けの入力です。

---

### モデル別の入力まとめ

| モデル                 | 入力特徴量                                                    |
|-----------------------|-----------------------------------------------------------|
| **LightGBM/CatBoost** | Tabular 120次元特徴量 (Block I)                            |
| **GRU / CNN-GRU**     | IMU ウィンドウテンソル + Demographics (Block J + F)        |
| **ToF-3D-CNN**        | ToF 3D ボクセルテンソル (Block K)                           |

ウィンドウ化前後の正規化や利き手補正の順序は、`doc/preprocessing_pipeline.md` の「主な処理の流れ」に準拠しています【F:doc/preprocessing_pipeline.md†L21-L37】。

