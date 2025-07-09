# 前処理パイプライン概要

このプロジェクトではセンサーデータと Demographics 情報を統合する前処理モジュール `src/utils/preprocessing.py` を作成しました。

## 主な処理内容

1. **IMU ワールド座標変換**
   - クォータニオンを回転行列に変換し、加速度ベクトルをワールド座標へ変換します。
   - 重力成分を除外して線形加速度を算出する関数 `linear_acceleration` を実装しました。
2. **スライディングウィンドウ生成**
   - `create_sliding_windows_with_demographics` で時系列データを固定長ウィンドウに変換します。
   - 各ウィンドウには静的な Demographics 特徴量を付与します。
3. **正規化処理**
   - センサーデータおよび Demographics データを `StandardScaler` で正規化するユーティリティを提供します。
4. **ピーク数特徴量**
   - 各ウィンドウから軸ごとのピーク数を抽出する簡易な特徴量計算を追加しました。
5. **欠損センサ検知フラグ**
   - センサ列がすべて欠損している場合に `missing_flag_*` を付与します。
6. **基本統計量抽出**
   - 各ウィンドウで mean, std, range, RMS, energy と合成加速度の
     mean/std を計算する `compute_basic_statistics` を追加しました。
7. **FFT バンドエネルギー**
   - `compute_fft_band_energy` で 0.5〜20Hz のバンド別エネルギーを算出します。
8. **利き手反転正規化**
   - `handedness_normalization` により左利きデータの Y/Z 軸を反転させます。
9. **Wavelet 周波数特徴**
   - `compute_wavelet_features` により離散 Wavelet 変換の各バンドエネルギーを抽出します。
10. **TDA 特徴量**
    - `compute_persistence_image_features` で位相的特徴を画像化します（`giotto-tda` 使用）。
11. **Auto‑Encoder 誤差**
    - 事前学習済み AE モデルを渡して `compute_autoencoder_reconstruction_error` で再構成誤差を取得します。
12. **ToF 3D Voxel 化**
    - `tof_to_voxel_tensor` で ToF センサ 5 層 × 8×8 グリッドを時系列テンソルに整形します。

これらの関数を学習・推論時に共通利用することで、前処理の再現性を高めています。

## ハイパーパラメータ管理と可視化

前処理に関するパラメータは `config.yaml` の `preprocessing` セクションでまとめて管理しています。ウィンドウサイズや FFT バンドなどを変更すると各スクリプトに自動反映されます。
デフォルトでは `window_size=128`、`stride=64` を使用しています。
実装した処理の挙動は `notebooks/preprocessing_visualization.ipynb` で確認できます。サンプルデータを用いて線形加速度計算や FFT バンドエネルギーの取得を行い、Matplotlib によるグラフ表示例を掲載しています。
