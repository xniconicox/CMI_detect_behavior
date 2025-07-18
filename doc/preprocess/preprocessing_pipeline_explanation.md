# CMI前処理パイプライン詳細説明書

## 実行コマンド
```bash
python -m scripts.run_preprocessing \
  --experiment-name 20250713_preproc_pipeline_first_try \
  --config config/config_v2.yaml \
  --use-cache \
  --mode train
```

## 概要

このコマンドは、CMI（Child Mind Institute）競技データセットに対して包括的な前処理を実行し、機械学習モデルで使用可能な形式にデータを変換します。

---

## 📋 実行パラメータ

| パラメータ | 値 | 説明 |
|------------|----|----|
| `--experiment-name` | `20250713_preproc_pipeline_first_try` | 実験名（出力ディレクトリ名） |
| `--config` | `config/config_v2.yaml` | 設定ファイルパス |
| `--use-cache` | 有効 | 中間結果のキャッシュを使用 |
| `--mode` | `train` | 訓練モード（fitとtransformを実行） |

---

## 🗂️ 入力データ

### 必要なファイル
- `data/train.csv` - 訓練データ（センサー時系列データ）
- `data/test.csv` - テストデータ（センサー時系列データ）
- `data/train_demographics.csv` - 訓練用人口統計データ
- `data/test_demographics.csv` - テスト用人口統計データ

### データ結合
人口統計データは`subject`列をキーとして結合され、統合データセットが作成されます。

---

## ⚙️ 設定パラメータ（config_v2.yaml）

### 基本設定
```yaml
data_dir: "data"
cache_dir: "cache"
output_dir: "output/experiments"
```

### 前処理パラメータ
| パラメータ | 値 | 説明 |
|------------|----|----|
| `window_size` | 128 | ウィンドウサイズ（時間ステップ数） |
| `stride` | 64 | ウィンドウのストライド |
| `min_sequence_length` | 20 | 最小シーケンス長 |
| `padding_value` | 0.0 | パディング値 |
| `sampling_rate` | 50.0 | サンプリングレート（Hz） |

### センサー構成
- **加速度**: `acc_x`, `acc_y`, `acc_z`
- **回転**: `rot_w`, `rot_x`, `rot_y`, `rot_z`（クォータニオン）
- **温度**: `thm_1`〜`thm_5`
- **ToF**: `tof_1_v0`〜`tof_5_v63`（5層×64ピクセル）

### 人口統計項目
- `adult_child` - 大人/子供分類
- `age` - 年齢
- `sex` - 性別
- `handedness` - 利き手
- `height_cm` - 身長（cm）
- `shoulder_to_wrist_cm` - 肩から手首までの長さ
- `elbow_to_wrist_cm` - 肘から手首までの長さ

---

## 🔄 前処理パイプライン

### Phase 1: データクリーニング

#### 1.1 利き手補正（handedness_correction_v2）
- **対象**: 左利き（`handedness==0`）のサンプル
- **処理内容**:
  - IMU（加速度、ジャイロ、回転、磁力）のY・Z軸を符号反転
  - `rot_w`は反転しない（クォータニオンの性質を保持）
  - ToFセンサーは水平ミラー（左右ピクセルを鏡像反転）

#### 1.2 基本センサークリーニング（clean_sensor_missing_values）
- **加速度センサー**: 極端値（`|value| > 40.0`）をNaN化
- **回転センサー**: NaN値のみ欠損として処理
- **ToFセンサー**: 
  - 上限超え（255以上）をNaN化
  - ハードエラー（-2以下）をNaN化
  - -1は"反射なし"として有効値で残す
- **温度センサー**: NaN値のみ欠損として処理

#### 1.3 高度補間処理（clean_missing_sensor_data_parallel_disk）
**並列処理**: 各subject+sequence_idのグループごとに独立して処理

- **加速度センサー**: 線形補間（最大3フレーム）
- **回転センサー**: SLERP補間（最大120フレーム）
  - クォータニオンの球面線形補間
  - 正規化して単位クォータニオンを維持
- **温度センサー**: 線形補間（最大120フレーム）
- **ToFセンサー**: 線形補間（最大4フレーム）
  - -1値は一時的にNaN化→補間→-1に復元

#### 1.4 世界座標系加速度特徴量（add_world_acc_features）
- 重力補正された世界座標系加速度を計算
- `acc_w_x`, `acc_w_y`, `acc_w_z`, `lin_acc_x`, `lin_acc_y`, `lin_acc_z`を追加

---

### Phase 2: 中間データ構築（キャッシュ可能）

#### 2.1 ウィンドウテンソル構築（WindowTensorBuilder）
**出力**: `cache/windows.pkl`

- **スライディングウィンドウ**: サイズ128、ストライド64
- **パディング**: 短いシーケンスは0.0でパディング
- **出力形式**:
  - `X_sensor`: (N_windows, 128, N_sensor_features)
  - `X_demo`: (N_windows, N_demo_features)
  - `y`: (N_windows,) - ジェスチャーラベル
  - `info`: ウィンドウメタ情報のリスト

#### 2.2 表形式特徴量構築（TabularFeatureBuilder）
**出力**: `cache/tabular_features.pkl`

**基本統計特徴量（各センサー軸ごと）**:
- 平均値、標準偏差、最小値、最大値
- 範囲、歪度、尖度
- パーセンタイル（25%, 50%, 75%）

**ピーク特徴量**:
- ピーク数、平均ピーク間隔
- 最大ピーク値、ピーク比率

**周波数領域特徴量**:
```yaml
fft_bands:
  - [0.5, 2]   # 低周波
  - [2, 5]     # 中低周波  
  - [5, 10]    # 中周波
  - [10, 20]   # 高周波
```
各バンドのパワースペクトル密度を計算

**ウェーブレット特徴量**（オプション）:
- デフォルト無効（`use_wavelet_features: false`）
- Daubechies 4ウェーブレット、3レベル分解

**位相的データ解析（TDA）特徴量**（オプション）:
- デフォルト無効（`use_tda_features: false`）
- Persistence Image特徴量

#### 2.3 ToFボクセル構築（ToFVoxelBuilder）
**出力**: `cache/tof_voxel.pkl`

- **入力**: 320次元ToFピクセルデータ（5層×64ピクセル）
- **出力**: (N_samples, 5, 8, 8) ボクセルテンソル
- **変換**: 1次元ピクセル配列を3次元ボクセルに再形成

#### 2.4 ToFウィンドウ構築（ToFWindowBuilder）
**出力**: `cache/tof_windows.pkl`

- **入力**: ToFボクセルテンソル
- **処理**: スライディングウィンドウ（サイズ128、ストライド64）
- **出力**: (N_windows, 128, 5, 8, 8) 時系列ToFテンソル

---

### Phase 3: 正規化とスケーリング

#### 3.1 センサー別欠損値処理
- **加速度**: 物理的適切値で置換
  - Z軸: -9.81 m/s²（重力）
  - X,Y軸: 0.0
- **回転**: 単位クォータニオンで置換
  - w成分: 1.0、x,y,z成分: 0.0
- **温度**: 体温（36.5°C）で置換
- **その他**: 0.0で置換

#### 3.2 StandardScaler適用
- **sensor_scaler**: センサーウィンドウデータ
- **demo_scaler**: 人口統計データ  
- **tab_scaler**: 表形式特徴量データ

各スケーラーで平均0、標準偏差1に正規化

---

### Phase 4: データ変換と出力

#### 4.1 訓練データ変換
**出力ディレクトリ**: `output/experiments/20250713_preproc_pipeline_first_try/preprocessed/`

**出力ファイル**:
- `train_sensor.pkl` - 正規化済みセンサーウィンドウ
- `train_demo.pkl` - 正規化済み人口統計データ
- `train_tabular.pkl` - 正規化済み表形式特徴量
- `train_labels.pkl` - ジェスチャーラベル
- `train_info.pkl` - メタ情報

#### 4.2 テストデータ変換
- `test_sensor.pkl` - 正規化済みセンサーウィンドウ
- `test_demo.pkl` - 正規化済み人口統計データ
- `test_tabular.pkl` - 正規化済み表形式特徴量
- `test_info.pkl` - メタ情報

#### 4.3 モデル保存
- `preprocessor.pkl` - 学習済みPreprocessorオブジェクト
  - 全スケーラーの状態を保持
  - 予測時の再現性を保証

---

## 📊 データフロー図

```
Raw Data
├── train.csv + train_demographics.csv
└── test.csv + test_demographics.csv
    ↓
Data Cleaning
├── 利き手補正
├── センサー別クリーニング 
├── 高度補間処理
└── 世界座標系変換
    ↓
Intermediate Processing (Cached)
├── ウィンドウテンソル → cache/windows.pkl
├── 表形式特徴量 → cache/tabular_features.pkl
├── ToFボクセル → cache/tof_voxel.pkl
└── ToFウィンドウ → cache/tof_windows.pkl
    ↓
Normalization & Scaling
├── 欠損値処理（センサー別）
└── StandardScaler適用
    ↓
Final Output
├── train_*.pkl (正規化済み訓練データ)
├── test_*.pkl (正規化済みテストデータ)
└── preprocessor.pkl (学習済みモデル)
```

---

## ⚡ パフォーマンス最適化

### キャッシュ機能（--use-cache）
- MD5ハッシュによるデータ整合性チェック
- 中間結果の再利用で処理時間短縮
- キャッシュヒット時は該当処理をスキップ

### 並列処理
- `joblib`による並列グループ処理
- メモリ効率的な一時ファイル管理
- ディスク結合による大容量データ対応

---

## 🎯 出力データの用途

### モデル学習用データ
1. **LSTM/GRU系**: `train_sensor.pkl` + `train_demo.pkl`
2. **CNN系**: ToFデータ（時系列畳み込み）
3. **表形式モデル**: `train_tabular.pkl`（LightGBM、CatBoost等）
4. **マルチモーダル**: 複数データタイプの組み合わせ

### 品質保証
- 全データにNaN値が含まれないことを保証
- 一貫した正規化（平均0、標準偏差1）
- 再現可能な前処理パイプライン

---

## 📝 ログと監視

### ログファイル
- `preprocess.log` - 詳細な処理ログ
- 各ステップの実行時間と統計情報
- エラーや警告の記録

### 進行状況表示
- tqdmによるプログレスバー
- 処理中のリアルタイム統計更新
- メモリ使用量の監視

---

この前処理パイプラインにより、生の時系列センサーデータが機械学習モデルで使用可能な高品質な特徴量セットに変換されます。 