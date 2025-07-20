# CMI競技テストガイド

## 📋 概要

このドキュメントでは、CMI競技プロジェクトのテスト実行方法について説明します。プロジェクトには複数のテストファイルが含まれており、コードの品質と信頼性を確保するために自動化されたテストが実装されています。

## 🏗️ テスト構造

### テストファイル一覧

```
tests/
├── test_handedness_correction.py      # 利き手補正関数のテスト
├── test_preprocessing.py              # 前処理パイプラインのテスト
├── test_pipeline.py                   # パイプラインコンポーネントのテスト
├── test_multimodal_trainer.py         # マルチモーダルトレーナーのテスト
├── test_multimodal_trainer_v30.py     # バージョン30のトレーナーテスト
├── test_multimodal_trainer_v31.py     # バージョン31のトレーナーテスト
└── test_tof_3d_cnn_trainer.py         # ToF 3D CNNトレーナーのテスト
```

## 🚀 セットアップ

### 1. 依存関係のインストール

```bash
# 仮想環境の有効化
source .venv/bin/activate

# 必要なパッケージのインストール
pip install pytest scikit-learn tensorflow matplotlib numpy pandas scipy pyyaml joblib tqdm
```

### 2. 環境確認

```bash
# pytestのバージョン確認
pytest --version

# Pythonパスの確認
python -c "import sys; print('\n'.join(sys.path))"
```

## 🧪 テスト実行方法

### 1. 全テストの実行

```bash
# 詳細出力付きで全テスト実行
pytest tests/ -v

# 簡潔な出力で全テスト実行
pytest tests/ -q

# テスト実行時間の表示
pytest tests/ --durations=10
```

### 2. 特定のテストファイルの実行

```bash
# 利き手補正テストのみ実行
pytest tests/test_handedness_correction.py -v

# 前処理テストのみ実行
pytest tests/test_preprocessing.py -v

# パイプラインテストのみ実行
pytest tests/test_pipeline.py -v
```

### 3. 特定のテスト関数の実行

```bash
# 特定のテスト関数のみ実行
pytest tests/test_handedness_correction.py::test_handedness_correction -v

# 複数のテスト関数を実行
pytest tests/test_preprocessing.py::test_preprocessor_cache_and_transform -v
```

### 4. テストパターンマッチング

```bash
# 関数名に "handedness" を含むテストを実行
pytest tests/ -k "handedness" -v

# ファイル名に "preprocessing" を含むテストを実行
pytest tests/ -k "preprocessing" -v
```

## 📊 テスト内容詳細

### 1. 利き手補正テスト (`test_handedness_correction.py`)

#### テスト対象関数
- `handedness_correction_v2()` - 左利き→右利き変換
- `handedness_correction_v2_reverse()` - 右利き→左利き変換
- `augment_by_handedness_flip()` - データ拡張

#### テスト内容
```python
def test_handedness_correction():
    """利き手補正関数のテスト"""
    # 1. テストデータの作成
    # 2. 左利き→右利き変換の検証
    # 3. 右利き→左利き変換の検証
    # 4. 符号反転の確認（acc_y, acc_z, rot_y, rot_z）

def test_augmentation():
    """データ拡張関数のテスト"""
    # 1. 元データの作成
    # 2. データ拡張の実行
    # 3. サンプル数の倍増確認
    # 4. 新しいsubject IDの割り当て確認
    # 5. handedness分布の確認
```

#### 期待される結果
- 左利きデータのY/Z軸が符号反転
- 右利きデータのY/Z軸が符号反転
- データ拡張でサンプル数が2倍
- 新しいsubject IDが正しく割り当て

### 2. 前処理パイプラインテスト (`test_preprocessing.py`)

#### テスト対象クラス
- `Preprocessor` - メイン前処理クラス

#### テスト内容
```python
def test_preprocessor_cache_and_transform(tmp_path: Path):
    """Preprocessorのキャッシュと変換機能のテスト"""
    # 1. 設定ファイルの作成
    # 2. ダミーデータの生成
    # 3. Preprocessorの初期化とfit_transform
    # 4. 出力形状の確認
    # 5. キャッシュファイルの存在確認
    # 6. 保存・読み込み機能の確認

def test_handle_missing_values_sensor_type(tmp_path: Path):
    """欠損値処理のテスト"""
    # 1. 欠損値を含むデータの作成
    # 2. センサータイプ別の欠損値処理
    # 3. 期待値との比較
```

#### 期待される結果
- ウィンドウサイズ: (1, 16, 15)
- デモグラフィクス: (1, 7)
- 表形式特徴量: (1, 159)
- ToFボクセル: (len(df), 5, 8, 8)
- ToFウィンドウ: (1, 16, 5, 8, 8)

### 3. パイプラインコンポーネントテスト (`test_pipeline.py`)

#### テスト対象クラス
- `WindowTensorBuilder` - ウィンドウテンソル作成
- `TabularFeatureBuilder` - 表形式特徴量作成
- `ToFVoxelBuilder` - ToFボクセル作成
- `ToFWindowBuilder` - ToFウィンドウ作成

#### テスト内容
```python
def test_window_tensor_builder(tmp_path: Path):
    """ウィンドウテンソルビルダーのテスト"""
    # 1. 設定とデータの準備
    # 2. ウィンドウ化処理の実行
    # 3. 出力形状の確認
    # 4. キャッシュ機能の確認

def test_tabular_feature_builder(tmp_path: Path):
    """表形式特徴量ビルダーのテスト"""
    # 1. 特徴量計算の実行
    # 2. 出力形状の確認
    # 3. キャッシュ機能の確認
```

## 🔧 トラブルシューティング

### 1. インポートエラー

#### 問題
```
ModuleNotFoundError: No module named 'src'
```

#### 解決方法
```python
# テストファイルの先頭に追加
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
```

### 2. 依存関係エラー

#### 問題
```
ModuleNotFoundError: No module named 'sklearn'
```

#### 解決方法
```bash
# 必要なパッケージをインストール
pip install scikit-learn tensorflow matplotlib

# または requirements.txt からインストール
pip install -r requirements.txt
```

### 3. メモリ不足エラー

#### 問題
```
MemoryError: Unable to allocate array
```

#### 解決方法
```bash
# メモリ使用量を制限してテスト実行
pytest tests/ --maxfail=1 -x

# 特定の軽量テストのみ実行
pytest tests/test_handedness_correction.py -v
```

## 📈 テスト結果の解釈

### 1. 成功ケース
```
============================= test session starts ==============================
platform linux -- Python 3.11.5, pytest-8.4.1, pluggy-1.6.0
collected 15 items

tests/test_handedness_correction.py::test_handedness_correction PASSED [ 20%]
tests/test_handedness_correction.py::test_augmentation PASSED        [ 40%]
tests/test_preprocessing.py::test_preprocessor_cache_and_transform PASSED [ 60%]
...

============================== 15 passed in 45.32s ==============================
```

### 2. 失敗ケース
```
============================= test session starts ==============================
collected 15 items

tests/test_handedness_correction.py::test_handedness_correction FAILED [ 20%]

================================= FAILURES ===================================
_________________ test_handedness_correction ________________

    def test_handedness_correction():
        # テストコード
>       assert corrected_left_data['acc_y'].iloc[0] == -1.0
E       assert 1.0 == -1.0
E         +1.0
E         -1.0

tests/test_handedness_correction.py:45: AssertionError
```

## 🚀 CI/CD統合

### GitHub Actions設定

```yaml
# .github/workflows/test.yml
name: Test
on:
  push:
    branches: ["**"]
  pull_request:

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
          python -m pip install --upgrade pip
          pip install numpy pandas scikit-learn scipy pyyaml joblib tqdm pytest tensorflow matplotlib
      - name: Run tests
        run: |
          pytest -q
```

### ローカルでのCI/CD実行

```bash
# テスト実行（CI/CDと同じ環境）
pytest -q

# カバレッジ付きテスト実行
pytest --cov=src tests/

# テストレポート生成
pytest --html=report.html tests/
```

## 📝 新しいテストの追加

### 1. テストファイルの作成

```python
# tests/test_new_feature.py
import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.new_feature import new_function

def test_new_function():
    """新しい機能のテスト"""
    # テストデータの準備
    test_data = pd.DataFrame({
        'column1': [1, 2, 3],
        'column2': [4, 5, 6]
    })
    
    # 関数の実行
    result = new_function(test_data)
    
    # 結果の検証
    assert result.shape == (3, 2)
    assert result.iloc[0, 0] == 1
```

### 2. テストの実行確認

```bash
# 新しいテストの実行
pytest tests/test_new_feature.py -v

# 全テストに含まれることを確認
pytest tests/ --collect-only
```

## 🎯 ベストプラクティス

### 1. テスト設計
- **単一責任**: 各テストは1つの機能のみをテスト
- **独立性**: テスト間の依存関係を避ける
- **再現性**: 同じ入力で同じ結果が得られることを確認

### 2. テストデータ
- **最小限**: 必要最小限のテストデータを使用
- **代表性**: 実際の使用ケースを反映したデータ
- **エッジケース**: 境界値や異常値を含める

### 3. アサーション
- **明確性**: 期待される結果を明確に記述
- **具体的**: 数値や形状を具体的に指定
- **包括的**: 重要な側面をすべてテスト

## 📚 参考資料

- [pytest公式ドキュメント](https://docs.pytest.org/)
- [Pythonテストガイド](https://docs.python.org/3/library/unittest.html)
- [CI/CDベストプラクティス](https://docs.github.com/en/actions)

---

このガイドに従ってテストを実行することで、コードの品質と信頼性を確保できます。 