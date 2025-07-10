# 前処理パイプライン概要

`src/utils/preprocessing.py` では、**センサ時系列 + Demographics** を  
A–M ブロックに分けて特徴量化するユーティリティをまとめています  
（デザインドキュメント 4-2 と 1 : 1 対応）。

| Block | 関数 | 次元 | 主な用途 / 備考 |
|-------|------|------|-----------------|
| **A 基本統計** | `compute_basic_statistics` | 56 | 広域量的特徴 (mean / std / range / RMS / energy) |
| **B ピーク&周期** | `compute_peak_features` | 18 | BFRB 反復性をカウント |
| **C FFT エネルギー** | `compute_fft_band_energy` | 10 | 0.5–20 Hz のリズム・速度指標 |
| **D ワールド線形加速度** | `quaternion_to_rotation_matrix`<br>`linear_acceleration` | 21 | 姿勢差を除去した純粋な動き |
| **E 欠損フラグ** | `add_missing_sensor_flags` | 3 | ToF / Thermal / IMU 欠測を one-hot |
| **F Demographics** | （窓生成時に結合） | 4 | sex / handedness / height / arm_len |
| **G TDA Stats** | `compute_persistence_image_features` | 8 | 位相的周期性 (Persistence Image) |
| **H AE 再構成誤差** | `compute_autoencoder_reconstruction_error` | 4 | 異常度スコア |
| **I 合成 Tabular** | A–H を結合 | 120 | LightGBM / CatBoost 用 |
| **J IMU Window Tensor** | `create_sliding_windows_with_demographics` | 1 792 | GRU / CNN-GRU 入力 |
| **K ToF 3D-Voxel** | `tof_to_voxel_tensor` | 20 480 | ToF-3D-CNN 入力 |
| **L Handedness Normalization** | `handedness_normalization` | — | 左利きサンプルの Y/Z 反転 |
| **M Wavelet Features** | `compute_wavelet_features` | 可変 (level×axis) | DWT バンドエネルギー |

> **Utility**  
> - `normalize_sensor_data`, `normalize_tabular_data` : z-score 正規化

---

## 主な処理の流れ

1. **D: IMU ワールド座標変換**  
2. **J: スライディングウィンドウ生成 (+F Demographics)**  
3. **Utility: 正規化**  
4. **L: 利き手反転**（左利きのみ Y/Z 反転）  
5. **A–C, G, H, M の特徴抽出**  
6. **E 欠損フラグ付与**  
7. **I: Tabular 120 dim へ統合**  
8. **K: ToF 3D-Voxel 変換**

---

## ファイル出力規約

| 保存対象 | ファイル例 | 形式 |
|----------|-----------|------|
| Tabular 120 dim | `features_tab120.parquet` | Parquet |
| IMU Tensor | `imu_windows.npy` | NumPy |
| ToF Tensor | `tof_voxel.npy` | NumPy |

---

## ハイパーパラメータ & 可視化

- すべて `config.yaml > preprocessing` で集中管理  
  (`window_size=256`, `stride=128`, `fft_bands`, `wavelet` など)
- Notebook: `notebooks/preprocessing_visualization.ipynb`  
  で各ブロックの挙動を確認できます。


