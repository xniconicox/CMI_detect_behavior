# モデルフローまとめ (BFRB 二段階版)

このドキュメントでは、[前処理フローまとめ](preprocessing_flow_jp.md) で整理した特徴量をどのように各モデルへ投入し、学習・推論を行うかを概説します。BFRB (body focused repetitive behavior) と Non-BFRB を判定するバイナリステージを追加した二段階構成を想定します。

## 1. ステージ構成

1. **Binaryステージ**: BFRB vs Non-BFRB を判定
2. **Multi-classステージ**: BFRBと判定されたサンプルについて18クラス分類

## 2. モデル一覧と入力

| ステージ | モデル | 主な入力特徴量 | 備考 |
|----------|-------|----------------|------|
| Binary | LightGBM/CatBoost | Tabular 120次元特徴量 (Block I) | 迅速に推論可能 |
| Binary | **Binary-GRU** | IMUウィンドウテンソル + Demographics (Block J + F) | `preprocessing.py` のコメントで推奨【F:src/utils/preprocessing.py†L166-L190】 |
| Multi | GRU / CNN-GRU | IMUウィンドウテンソル + Demographics (Block J + F) | 18クラス分類向け |
| Multi | ToF-3D-CNN | ToF 3Dボクセルテンソル (Block K) | 18クラス分類向け |

入力対応は前処理ドキュメントの「モデル別の入力まとめ」【F:doc/preprocessing_flow_jp.md†L67-L73】と共通です。

## 3. Binaryステージ
- **目的**: まずBFRBかNon-BFRBかを高速に判定し、誤検知を減らす
- **LightGBM/CatBoost**: Tabular特徴のみで二値分類
- **Binary-GRU**: 時系列テンソルを用いた二値分類
- **出力**: BFRB確率 (0〜1)

BFRBに特化した特徴として `compute_peak_features` は反復性をカウントします【F:doc/preprocessing_pipeline.md†L9-L11】。また、AE再構成誤差でBFRB境界を補強する記述があります【F:src/utils/preprocessing.py†L162-L168】。

## 4. Multi-classステージ
BinaryステージでBFRBと判定されたサンプルに対し、従来通り以下のモデルで18クラス分類を行います。
- **GRU / CNN-GRU**: `LSTMTrainer` 等で学習【F:results/comprehensive_optimization/production_model/README.md†L1-L19】
- **ToF 3D CNN**: ToFボクセルテンソルからCNNで分類

推論フローの詳細は `doc/lstm_v1_submission_flow.md` の「推論パイプライン」で説明されています【F:doc/lstm_v1_submission_flow.md†L127-L139】。

## 5. まとめ
二段階化によりまずBFRBかどうかを絞り込むことで、18クラスモデルの誤検出を抑制しつつCMI評価指標【F:src/utils/cmi_evaluation.py†L70-L115】におけるBinary F1も向上させることが期待できます。前処理の順序は [前処理パイプライン概要](preprocessing_pipeline.md) の「主な処理の流れ」【F:doc/preprocessing_pipeline.md†L21-L37】を共通基盤とします。
