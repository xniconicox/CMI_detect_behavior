# CMI競技テスト実行ガイド

## 📋 概要

このプロジェクトには、コードの品質と信頼性を確保するための包括的なテストスイートが含まれています。テストは自動化されており、CI/CDパイプラインと統合されています。

## 🚀 クイックスタート

### 基本的なテスト実行

```bash
# 環境準備
source .venv/bin/activate
pip install pytest scikit-learn tensorflow matplotlib

# 全テスト実行
pytest tests/ -v

# 特定のテストファイル実行
pytest tests/test_handedness_correction.py -v
```

## 📁 テストファイル構成

```
tests/
├── test_handedness_correction.py      # 利き手補正関数
├── test_preprocessing.py              # 前処理パイプライン
├── test_pipeline.py                   # パイプラインコンポーネント
├── test_multimodal_trainer.py         # マルチモーダルトレーナー
├── test_multimodal_trainer_v30.py     # バージョン30トレーナー
├── test_multimodal_trainer_v31.py     # バージョン31トレーナー
└── test_tof_3d_cnn_trainer.py         # ToF 3D CNNトレーナー
```

## 🧪 テスト内容

### 1. 利き手補正テスト
- **ファイル**: `test_handedness_correction.py`
- **内容**: 左利き→右利き変換、右利き→左利き変換、データ拡張
- **実行時間**: ~5秒

### 2. 前処理パイプラインテスト
- **ファイル**: `test_preprocessing.py`
- **内容**: Preprocessorクラス、キャッシュ機能、欠損値処理
- **実行時間**: ~30秒

### 3. パイプラインコンポーネントテスト
- **ファイル**: `test_pipeline.py`
- **内容**: ウィンドウ化、特徴量計算、ToF処理
- **実行時間**: ~20秒

### 4. トレーナーテスト
- **ファイル**: `test_multimodal_trainer*.py`
- **内容**: モデル学習、評価、保存・読み込み
- **実行時間**: ~60秒

## 🔧 実行方法

### 全テスト実行
```bash
pytest tests/ -v
```

### 特定テスト実行
```bash
# ファイル指定
pytest tests/test_handedness_correction.py -v

# 関数指定
pytest tests/test_handedness_correction.py::test_handedness_correction -v

# パターンマッチング
pytest tests/ -k "handedness" -v
```

### 段階的実行（推奨）
```bash
# 1. 軽量テスト
pytest tests/test_handedness_correction.py -v

# 2. パイプラインテスト
pytest tests/test_pipeline.py -v

# 3. 前処理テスト
pytest tests/test_preprocessing.py -v

# 4. 全テスト
pytest tests/ -v
```

## ⚠️ トラブルシューティング

### よくある問題

1. **モジュールが見つからない**
   ```bash
   export PYTHONPATH=$PYTHONPATH:$(pwd)
   pytest tests/ -v
   ```

2. **依存関係エラー**
   ```bash
   pip install scikit-learn tensorflow matplotlib
   ```

3. **メモリ不足**
   ```bash
   pytest tests/test_handedness_correction.py -v
   ```

詳細なトラブルシューティングは `docs/testing_troubleshooting.md` を参照してください。

## 📊 テスト結果の解釈

### 成功ケース
```
============================== 15 passed in 45.32s ==============================
```

### 失敗ケース
```
================================= FAILURES ===================================
_________________ test_handedness_correction ________________
    assert 1.0 == -1.0
E   +1.0
E   -1.0
```

## 🔄 CI/CD統合

GitHub Actionsでプッシュ時に自動実行されます：

```yaml
# .github/workflows/test.yml
name: Test
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install numpy pandas scikit-learn scipy pyyaml joblib tqdm pytest tensorflow matplotlib
      - name: Run tests
        run: |
          pytest -q
```

## 📝 新しいテストの追加

### テストファイル作成
```python
# tests/test_my_feature.py
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.my_feature import my_function

def test_my_function():
    """新しい機能のテスト"""
    result = my_function()
    assert result is not None
```

### テスト実行確認
```bash
pytest tests/test_my_feature.py -v
```

## 📚 詳細ドキュメント

- **詳細ガイド**: `docs/testing_guide.md`
- **クイックスタート**: `docs/testing_quick_start.md`
- **トラブルシューティング**: `docs/testing_troubleshooting.md`

## 🎯 ベストプラクティス

1. **テストは小さく保つ** - 1つの機能につき1つのテスト
2. **明確な名前** - テスト関数名で何をテストするか明確に
3. **独立したテスト** - テスト間の依存関係を避ける
4. **適切なアサーション** - 期待される結果を具体的に記述

## 📞 サポート

問題が発生した場合は、以下の情報を収集してください：

1. エラーメッセージとトレースバック
2. 環境情報（OS、Pythonバージョン、パッケージバージョン）
3. 実行コマンド
4. 関連するログファイル

---

このガイドに従ってテストを実行することで、コードの品質と信頼性を確保できます。 