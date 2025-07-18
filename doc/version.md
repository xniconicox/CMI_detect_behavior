
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
- 推論例: `python submissions/ensemble_v30/src/inference_pipeline.py`
