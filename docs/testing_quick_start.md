# テスト実行クイックスタート

## 🚀 5分でテストを実行

### Step 1: 環境準備

```bash
# 仮想環境の有効化
source .venv/bin/activate

# 必要なパッケージのインストール
pip install pytest scikit-learn tensorflow matplotlib
```

### Step 2: 基本的なテスト実行

```bash
# 全テストの実行
pytest tests/ -v

# 特定のテストファイルのみ実行
pytest tests/test_handedness_correction.py -v
```

### Step 3: 結果確認

```
============================= test session starts ==============================
platform linux -- Python 3.11.5, pytest-8.4.1, pluggy-1.6.0
collected 15 items

tests/test_handedness_correction.py::test_handedness_correction PASSED [ 20%]
tests/test_handedness_correction.py::test_augmentation PASSED        [ 40%]
...

============================== 15 passed in 45.32s ==============================
```

## 🔧 よくある問題と解決方法

### 問題1: モジュールが見つからない
```bash
ModuleNotFoundError: No module named 'src'
```

**解決方法:**
```bash
# プロジェクトルートで実行
cd /path/to/CMI_comp
export PYTHONPATH=$PYTHONPATH:$(pwd)
pytest tests/ -v
```

### 問題2: 依存関係エラー
```bash
ModuleNotFoundError: No module named 'sklearn'
```

**解決方法:**
```bash
pip install scikit-learn tensorflow matplotlib numpy pandas
```

### 問題3: メモリ不足
```bash
MemoryError: Unable to allocate array
```

**解決方法:**
```bash
# 軽量テストのみ実行
pytest tests/test_handedness_correction.py -v
```

## 📋 テスト一覧

| テストファイル | 内容 | 実行時間 |
|---------------|------|----------|
| `test_handedness_correction.py` | 利き手補正 | ~5秒 |
| `test_preprocessing.py` | 前処理パイプライン | ~30秒 |
| `test_pipeline.py` | パイプラインコンポーネント | ~20秒 |
| `test_multimodal_trainer.py` | マルチモーダルトレーナー | ~60秒 |

## 🎯 推奨実行順序

1. **軽量テストから開始**
   ```bash
   pytest tests/test_handedness_correction.py -v
   ```

2. **パイプラインテスト**
   ```bash
   pytest tests/test_pipeline.py -v
   ```

3. **前処理テスト**
   ```bash
   pytest tests/test_preprocessing.py -v
   ```

4. **全テスト実行**
   ```bash
   pytest tests/ -v
   ```

## 📊 テスト結果の解釈

### ✅ 成功
- すべてのテストが `PASSED`
- エラーや警告なし

### ❌ 失敗
- 一部のテストが `FAILED`
- エラーメッセージを確認
- 期待値と実際の値を比較

### ⚠️ 警告
- テストは成功だが警告あり
- 非推奨機能の使用など

## 🔄 CI/CDでの自動実行

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

### 1. テストファイル作成
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

### 2. テスト実行
```bash
pytest tests/test_my_feature.py -v
```

## 🎯 ベストプラクティス

1. **テストは小さく保つ** - 1つの機能につき1つのテスト
2. **明確な名前** - テスト関数名で何をテストするか明確に
3. **独立したテスト** - テスト間の依存関係を避ける
4. **適切なアサーション** - 期待される結果を具体的に記述

## 📚 詳細情報

詳細なテスト方法については `docs/testing_guide.md` を参照してください。 