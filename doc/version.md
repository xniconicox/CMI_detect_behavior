
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