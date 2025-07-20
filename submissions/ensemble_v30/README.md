# CMI Ensemble V30 Submission

5-fold cross-validation で学習したマルチモーダルモデル（IMU、人口統計、表形式特徴、ToFボクセル）のアンサンブル推論を行います。

## 構成

```
submissions/ensemble_v30/
├── src/
│   ├── inference_pipeline.py      # 5-foldアンサンブル推論パイプライン
│   ├── trainers/
│   │   └── multimodal_trainer_v30.py  # モデル定義
│   └── utils/                     # 共通ユーティリティ
├── models/                        # 学習済みモデル（実行時にコピー）
├── output/                        # 推論結果
├── cache/                         # キャッシュ
├── cmi-2025-demo-submission.ipynb # Kaggle提出用ノートブック
├── prepare.sh                     # データ準備スクリプト
├── run_local.sh                   # ローカル実行スクリプト
└── README.md                      # このファイル
```

## 使用方法

### 1. 学習の実行

まず、プロジェクトルートで5-foldクロスバリデーション学習を実行：

```bash
# プロジェクトルートから
./scripts/run_multimodal_training_v30.sh
```

### 2. ローカルでの推論実行

```bash
cd submissions/ensemble_v30
./run_local.sh
```

このスクリプトは以下を実行します：
- 学習済みモデルと前処理器をコピー
- モデル読み込みテスト
- ノートブック実行（papermillが利用可能な場合）

### 3. Kaggleへの提出

1. `submissions/ensemble_v30`をzip化
2. Kaggleにデータセットとしてアップロード
3. 新しいNotebookで以下を実行：

```python
# Kaggle Notebook内
import zipfile
zipfile.ZipFile('/kaggle/input/your-uploaded-zip/ensemble_v30.zip').extractall('.')

# ノートブックを実行
exec(open('ensemble_v30/cmi-2025-demo-submission.ipynb').read())
```

## 特徴

- **5-foldアンサンブル**: 各foldのモデル予測を平均化して最終予測を決定
- **マルチモーダル**: IMU、人口統計、表形式特徴、ToFボクセルを統合
- **簡潔なコード**: ノートブック内のコードは最小限に抑制
- **ローカルテスト**: Kaggle環境と同様の推論をローカルで実行可能

## 依存関係

- TensorFlow 2.x
- polars
- pandas
- numpy
- scikit-learn
- kaggle_evaluation (Kaggle環境で自動提供)

## トラブルシューティング

### モデルが見つからない場合

```bash
# 学習が完了しているか確認
ls -la output/experiments/multimodal_v30/models/

# 期待されるファイル:
# multimodal_model_v30_fold1.keras
# multimodal_model_v30_fold2.keras
# multimodal_model_v30_fold3.keras
# multimodal_model_v30_fold4.keras
# multimodal_model_v30_fold5.keras
```

### インポートエラーの場合

```bash
# sys.pathが正しく設定されているか確認
cd submissions/ensemble_v30
python -c "
import sys
sys.path.insert(0, 'src')
from src.inference_pipeline import models
print('OK')
"
```

### 推論が遅い場合

- GPU使用を確認
- バッチサイズの調整（大きなデータセットの場合）
- モデル量子化の検討 