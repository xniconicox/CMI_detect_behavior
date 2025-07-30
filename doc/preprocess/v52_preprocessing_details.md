# v52 前処理パイプライン

このドキュメントでは `scripts/run_preprocessing_v52_no_windows.py` で実装されている前処理の流れを解説します。

## スクリプト構成
- `build_sensor_sequences` 関数では、各センサ値の有効データのみを使って平均と標準偏差を計算し、正規化した系列を生成します。T_mask と V_mask もここで作成されます【F:scripts/run_preprocessing_v52_no_windows.py†L32-L96】。
- `build_tof_sequences` 関数は ToF ボクセルを読み込み、反射あり (0〜254) を 1 とする `tof_mask` を生成します。反射なしは 254、欠損は -999 に置き換えた後、実値のみで正規化します【F:scripts/run_preprocessing_v52_no_windows.py†L99-L150】。
- `main` では `Preprocessor` を使ってクリーニングを行った後、上記関数で前処理を実施し、NumPy ファイルとして保存します【F:scripts/run_preprocessing_v52_no_windows.py†L160-L226】。

## 設定ファイル
`config/config_v52.yaml` ではセンサ列や ToF の形状、処理オプションを定義しています。`use_windows: false` として窓分割を無効化し、ToF ボクセルサイズは `tof_depth: 5`, `tof_height: 8`, `tof_width: 8` を指定しています【F:config/config_v52.yaml†L1-L36】。
