# v52 前処理パイプライン

このドキュメントでは `scripts/run_preprocessing_v52_no_windows.py` で実装されている前処理の流れを解説します。

## スクリプト構成
- `build_sensor_sequences` 関数では、各センサ値の有効データのみを使って平均と標準偏差を計算し、正規化した系列をメモリマップ形式で保存します。T_mask と V_mask もここで作成されます【F:scripts/run_preprocessing_v52_no_windows.py†L34-L110】。
- `build_tof_sequences` 関数は ToF ボクセルを読み込み、ストリーム処理で統計量を計算しつつ `open_memmap` で書き出します。反射あり (0〜254) を 1 とする `tof_mask` を生成します【F:scripts/run_preprocessing_v52_no_windows.py†L113-L181】。
- `main` では `Preprocessor` によるクリーニング後、上記関数を呼び出して `train_*` / `predict_*` の各 `.npy` ファイルを生成します【F:scripts/run_preprocessing_v52_no_windows.py†L188-L255】。

## 設定ファイル
`config/config_v52.yaml` ではセンサ列や ToF の形状、チャンクサイズ等を定義しています。`use_windows: false` として窓分割を無効化し、ToF ボクセルサイズは `tof_depth: 5`, `tof_height: 8`, `tof_width: 8` を指定しています【F:config/config_v52.yaml†L1-L36】。

## モデルへの入力
- `train_sequences.npy` : 正規化済みセンサ系列 (N×T×F)
- `train_t_mask.npy` / `train_v_mask.npy` : 時間・値マスク
- `train_tof.npy` : 正規化済み ToF ボクセル (N×T×D×H×W)
- `train_tof_mask.npy` : ToF マスク
- `train_labels.npy` : ジェスチャラベル

## モデルについて
`src/trainers/multimodal_trainer_v52.py` では LSTM ベースのセンサ分岐と 3D CNN を用いた ToF 分岐を結合した軽量モデルを構築します【F:src/trainers/multimodal_trainer_v52.py†L24-L67】。
