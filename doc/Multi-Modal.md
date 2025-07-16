graph TD
    T1[ToF Input (128,5,8,8)] --> C1[Conv3D → MaxPool → Conv3D]
    C1 --> GAP[GlobalAvgPool3D (→ 32)]

    S1[Sensor Input (128,18)] --> NE[NotEqual → Any]
    S1 --> M1[Masking] --> L1[LSTM (→ 64)]

    D1[Demo Input (7,)] --> D_Dense[Dense(32)]
    T2[Tabular Input (389,)] --> T_Dense[Dense(64)]

    L1 --> CAT
    D_Dense --> CAT
    T_Dense --> CAT
    GAP --> CAT

    CAT[Concatenate (192)] --> FC1[Dense(64) → Dropout(0.5)]
    FC1 --> OUT[Dense(18) (Softmax)]
