
ensemble baseline
- IMU: LSTM(64)（単方向）
- ToF: Conv3D→MaxPooling3D→Conv3D→GlobalAveragePooling3D
- Tabular: Dense(64, relu)
- 残差接続なし、ResNetなし
    CMI 0.55

ensamble v20

- IMU: Bidirectional LSTM(64)（双方向）
- ToF: 3D ResNetブロック（残差接続付きConv3D×2）
- Tabular: Dense(64, relu)＋残差接続
- 各タワーの表現力・学習安定性が向上
    CMI 0.52

ensemble v30

- models/multimodal_model_v30_fold*.keras を全て読み込みアンサンブル
- Preprocessor.load() で前処理を共有
- 各モデルの predict() を平均し最終ラベルを決定
- 推論例: `python submissions/ensemble_v31/src/inference_pipeline.py`
    CMI 0.56

### v31 変更点

- **クラス重み導入**：`compute_class_weight()`で計算し `model.fit()` に適用
- **学習率減衰**：`ExponentialDecay` を Adam に設定
- **IMU ↔ ToF Attention Fusion**：MultiHeadAttentionで特徴結合
- **SpatialDropout3D(0.2) 追加**：ToF ResNetブロック末尾に挿入
- **Tabular を float32 変換**：`load_all_data` 内で `astype(np.float32)`
- **学習スクリプト追加**：`scripts/run_multimodal_training_v31.sh`
- **Ensemble v31 提出環境**：`submissions/ensemble_v31` に推論パイプライン配置
- **ユニットテスト拡充**：Attention・Dropout・型変換の検証を追加

    CMI 0.53