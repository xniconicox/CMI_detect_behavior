# LSTM版モデル開発・提出フロー

このドキュメントは、LightGBM ベースラインから発展させた **LSTM 版ベースライン** を開発し、Kaggle へ提出するまでの一連の手順をまとめた備忘録です。

---
## 1. ディレクトリ構成
```
CMI_comp/
├─ data/                       # 生データ配置
├─ output/
│   └─ experiments/
│       └─ lstm_v1/
│           ├─ models/         # 学習済み重み・エンコーダ等を保存
│           ├─ preprocessed/   # 前処理済みデータ
│           └─ results/        # 学習結果・メトリクス
├─ src/
│   ├─ lstm_model.py          # LSTMモデルクラス
│   ├─ lstm_trainer.py        # 学習用スクリプト
│   └─ lstm_utils.py          # LSTM用ユーティリティ
├─ notebooks/
│   └─ lstm_train_and_evaluate.ipynb  # 学習実行ノートブック
└─ submissions/
    └─ lstm_v1/                # 推論スクリプト & 提出ファイル
        ├─ model_inference.py
        └─ submission.parquet
```

---
## 2. データ前処理

### 2.1 前処理済みデータの活用
- 既存の`notebooks/lstm_preprocessing_and_visualization.ipynb`で生成された前処理済みデータを使用
- スライディングウィンドウ（WINDOW_SIZE=128, STRIDE=32）で時系列分割済み
- StandardScalerで正規化済み
- LabelEncoderでラベル数値化済み

### 2.2 データ形式
```python
# 学習データ形状
X_train: (n_samples, WINDOW_SIZE, n_features)  # 3次元配列
y_train: (n_samples,)                          # 1次元配列（エンコード済み）

# 特徴量数: 332（センサー列）
# クラス数: 18（ジェスチャー種類）
```

---
## 3. 純粋なLSTMモデルの設計

### 3.1 モデルアーキテクチャ
```python
# 基本的なLSTM構成
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(WINDOW_SIZE, n_features)),
    BatchNormalization(),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])
```

### 3.2 学習設定
- **オプティマイザー**: Adam (learning_rate=0.001)
- **損失関数**: sparse_categorical_crossentropy
- **評価指標**: accuracy, macro_f1
- **バッチサイズ**: 32
- **エポック数**: 100（Early Stopping付き）

### 3.3 コールバック設定
```python
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ModelCheckpoint('lstm_best.h5', save_best_only=True),
    ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6)
]
```

---
## 4. 学習パイプライン

### 4.1 ベースラインアプローチの参考
- `src/baseline_model.py`の構造を参考にLSTMモデルを実装
- クロスバリデーション（5-fold）による性能評価
- モデル保存とメタデータ管理

### 4.2 実装方針
1. **モジュール化**: `src/lstm_model.py`にLSTMModelクラスを実装
2. **学習スクリプト**: `src/lstm_trainer.py`で学習ロジックを実装
3. **ノートブック実行**: `notebooks/lstm_train_and_evaluate.ipynb`で学習を実行

### 4.3 データ分割戦略
- 時系列リークを避けるため、シーケンスIDベースでtrain/validation分割
- 比率: 8:2
- StratifiedKFoldではなく、時間順分割を使用

---
## 5. 保存ファイル

### 5.1 モデル関連
| ファイル | 役割 |
|----------|------|
| `lstm_best.h5` | 学習済みKerasモデル |
| `model_architecture.json` | モデル構造定義 |
| `model_weights.h5` | モデル重みのみ |

### 5.2 前処理関連
| ファイル | 役割 |
|----------|------|
| `label_encoder.pkl` | LabelEncoderオブジェクト |
| `scaler.pkl` | StandardScalerオブジェクト |
| `feature_columns.pkl` | 特徴量カラム情報 |

### 5.3 メタデータ
| ファイル | 役割 |
|----------|------|
| `meta.json` | ウィンドウ長・stride・特徴数等 |
| `training_history.json` | 学習履歴 |
| `cv_results.json` | クロスバリデーション結果 |

保存先: `output/experiments/lstm_v1/models/`

---
## 6. 推論パイプライン (`submissions/lstm_v1/model_inference.py`)

### 6.1 推論フロー
1. **モデル読み込み**: `tf.keras.models.load_model()`
2. **前処理器読み込み**: `pickle.load()`でscaler, label_encoder
3. **データ前処理**: ウィンドウ分割 → 正規化
4. **予測実行**: `model.predict()`
5. **後処理**: 確率→クラス変換、アンサンブル

### 6.2 アンサンブル戦略
- 複数のウィンドウからの予測を平均化
- シーケンス単位での最終予測決定

---
## 7. 評価とメトリクス

### 7.1 評価指標
- **主要指標**: Macro F1-Score（コンペティション評価指標）
- **補助指標**: Accuracy, Weighted F1-Score
- **クラス別評価**: Precision, Recall, F1-Score

### 7.2 可視化
- 学習曲線（Loss, Accuracy）
- 混同行列
- クラス別性能
- 特徴量重要度（SHAP等）

---
## 8. 提出ファイル生成
* 形式: `submission.parquet`
```
sequence_id | gesture
```
* 生成スクリプトを `local_test.ipynb` に実装し、エンドツーエンドで確認後 `kaggle_evaluation` ラッパーで最終提出。

---
## 9. チェックリスト
- [ ] `WINDOW_SIZE / STRIDE / scaler` が学習・推論で一致
- [ ] GPU & CPU 双方で学習コード動作確認
- [ ] ラベル不均衡への `class_weight` 設定検討
- [ ] 依存ライブラリ版本固定 (`requirements.txt`)
- [ ] 小規模 subset で end-to-end 動作確認後、全量学習
- [ ] クロスバリデーション結果の記録
- [ ] 学習時間とメモリ使用量の監視

---
## 10. 実装の詳細

### 10.1 LSTMModelクラス設計
```python
class LSTMModel:
    def __init__(self, input_shape, num_classes, **kwargs):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
    def build_model(self):
        # モデル構築
        
    def train(self, X_train, y_train, X_val, y_val, **kwargs):
        # 学習実行
        
    def predict(self, X):
        # 予測実行
        
    def evaluate(self, X_test, y_test):
        # 評価実行
        
    def save_model(self, filepath):
        # モデル保存
        
    def load_model(self, filepath):
        # モデル読み込み
```

### 10.2 学習スクリプト構成
- データ読み込み・前処理
- モデル初期化・学習
- クロスバリデーション
- 結果保存・可視化

---
## ToDo（将来改良用）

- demographics.csv（年齢・性別・体格情報）の活用を検討
  * sequence_id ベースで時系列特徴と結合し、マルチモーダルモデル化
  * 方針候補：
    - LSTM最終層の出力に demographics を連結し Dense 層で統合
    - LightGBM に demographics の統計特徴を追加
  * 使用有無をエクスペリメント管理（version区別）で比較

- 高度な特徴量エンジニアリング
  * FFT特徴量の追加
  * 統計的特徴量の組み合わせ
  * センサー間の相関特徴量

- モデルアーキテクチャの改良
  * Bidirectional LSTM
  * Attention機構の追加
  * CNN+LSTM ハイブリッド

---
最終提出までこのフローを参考し、都度更新すること。 