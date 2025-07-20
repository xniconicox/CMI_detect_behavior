# テスト実行トラブルシューティング

## 🚨 よくあるエラーと解決方法

### 1. インポートエラー

#### エラー: `ModuleNotFoundError: No module named 'src'`

**原因**: Pythonがプロジェクトのルートディレクトリを認識していない

**解決方法1: PYTHONPATH設定**
```bash
# プロジェクトルートで実行
export PYTHONPATH=$PYTHONPATH:$(pwd)
pytest tests/ -v
```

**解決方法2: テストファイル内でパス設定**
```python
# テストファイルの先頭に追加
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
```

**解決方法3: pytest.ini設定**
```ini
# pytest.ini
[tool:pytest]
pythonpath = .
```

### 2. 依存関係エラー

#### エラー: `ModuleNotFoundError: No module named 'sklearn'`

**原因**: 必要なパッケージがインストールされていない

**解決方法:**
```bash
# 基本的な依存関係
pip install scikit-learn tensorflow matplotlib numpy pandas

# または全依存関係
pip install -r requirements.txt

# 開発用依存関係
pip install pytest pytest-cov pytest-html
```

#### エラー: `ModuleNotFoundError: No module named 'tensorflow'`

**解決方法:**
```bash
# TensorFlowのインストール
pip install tensorflow

# GPU版が必要な場合
pip install tensorflow-gpu
```

### 3. メモリ不足エラー

#### エラー: `MemoryError: Unable to allocate array`

**原因**: テストで大量のメモリを使用している

**解決方法1: 軽量テストのみ実行**
```bash
# メモリ使用量の少ないテストのみ実行
pytest tests/test_handedness_correction.py -v
```

**解決方法2: メモリ制限付き実行**
```bash
# 一度に1つのテストのみ実行
pytest tests/ --maxfail=1 -x

# 並列度を制限
pytest tests/ -n 1
```

**解決方法3: システムメモリの確認**
```bash
# メモリ使用量確認
free -h

# プロセス別メモリ使用量
ps aux --sort=-%mem | head -10
```

### 4. 権限エラー

#### エラー: `PermissionError: [Errno 13] Permission denied`

**原因**: ファイルやディレクトリへの書き込み権限がない

**解決方法:**
```bash
# 権限の確認
ls -la tests/

# 権限の変更
chmod +x tests/
chmod 644 tests/*.py

# 一時ディレクトリの権限確認
ls -la /tmp/
```

### 5. バージョン互換性エラー

#### エラー: `ImportError: cannot import name 'X' from 'Y'`

**原因**: パッケージのバージョンが互換性がない

**解決方法:**
```bash
# パッケージバージョンの確認
pip list | grep -E "(numpy|pandas|sklearn|tensorflow)"

# 特定バージョンのインストール
pip install numpy==1.24.3 pandas==2.0.3 scikit-learn==1.3.0

# 仮想環境の再作成
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 🔧 デバッグ方法

### 1. 詳細なエラー情報の表示

```bash
# 詳細なトレースバック表示
pytest tests/ -v --tb=long

# ローカル変数の表示
pytest tests/ -v --tb=long --showlocals

# デバッグモード
pytest tests/ -v --pdb
```

### 2. テストの段階的実行

```bash
# テスト収集のみ
pytest tests/ --collect-only

# 特定のテストのみ実行
pytest tests/test_handedness_correction.py::test_handedness_correction -v

# 失敗したテストのみ再実行
pytest tests/ --lf
```

### 3. 環境情報の確認

```bash
# Pythonバージョン確認
python --version

# インストール済みパッケージ確認
pip list

# システム情報確認
uname -a
free -h
df -h
```

## 📊 パフォーマンス問題

### 1. テスト実行時間の最適化

```bash
# 実行時間の表示
pytest tests/ --durations=10

# 並列実行
pytest tests/ -n auto

# キャッシュの活用
pytest tests/ --cache-clear
```

### 2. メモリ使用量の最適化

```bash
# メモリ使用量の監視
watch -n 1 'free -h'

# プロセス別メモリ使用量
ps aux --sort=-%mem | head -5

# ガベージコレクションの強制実行
python -c "import gc; gc.collect()"
```

## 🐛 特定のテストの問題

### 1. 前処理テストの問題

**問題**: `test_preprocessing.py`でメモリ不足

**解決方法:**
```bash
# 設定を軽量化
export PYTEST_ADDOPTS="--maxfail=1 -x"

# 個別テスト実行
pytest tests/test_preprocessing.py::test_preprocessor_cache_and_transform -v
```

### 2. パイプラインテストの問題

**問題**: `test_pipeline.py`でタイムアウト

**解決方法:**
```bash
# タイムアウト時間の延長
pytest tests/test_pipeline.py --timeout=300 -v

# 個別コンポーネントテスト
pytest tests/test_pipeline.py::test_window_tensor_builder -v
```

### 3. トレーナーテストの問題

**問題**: `test_multimodal_trainer.py`でGPUエラー

**解決方法:**
```bash
# CPU環境での実行
export CUDA_VISIBLE_DEVICES=""

# 軽量設定での実行
pytest tests/test_multimodal_trainer.py -v --tb=short
```

## 🔄 CI/CD環境での問題

### 1. GitHub Actionsでの失敗

**問題**: CI/CDでテストが失敗するがローカルでは成功

**解決方法:**
```bash
# CI/CDと同じ環境での実行
docker run --rm -v $(pwd):/workspace -w /workspace python:3.10 bash -c "
  pip install -r requirements.txt
  pytest tests/ -v
"
```

### 2. 環境の違いによる問題

**問題**: 開発環境とCI/CD環境の違い

**解決方法:**
```yaml
# .github/workflows/test.yml の修正
- name: Debug environment
  run: |
    python --version
    pip list
    free -h
    df -h
```

## 📝 ログとレポート

### 1. テストレポートの生成

```bash
# HTMLレポート生成
pytest tests/ --html=report.html --self-contained-html

# JUnit XMLレポート
pytest tests/ --junitxml=report.xml

# カバレッジレポート
pytest tests/ --cov=src --cov-report=html
```

### 2. ログファイルの確認

```bash
# ログファイルの場所確認
find . -name "*.log" -type f

# 最新のログファイル確認
tail -f output/experiments/*/preprocessed/preprocess.log
```

## 🎯 予防的対策

### 1. 定期的なテスト実行

```bash
# 毎日のテスト実行
crontab -e
# 0 9 * * * cd /path/to/CMI_comp && source .venv/bin/activate && pytest tests/ -v > test_log.txt 2>&1
```

### 2. 環境の定期更新

```bash
# 依存関係の更新
pip install --upgrade pip
pip install --upgrade -r requirements.txt

# 仮想環境の再作成
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. テストの保守

```bash
# 非推奨警告の確認
pytest tests/ -W error::DeprecationWarning

# コードスタイルの確認
pip install flake8 black
flake8 src/ tests/
black --check src/ tests/
```

## 📞 サポート

問題が解決しない場合は、以下の情報を収集してサポートに連絡してください：

1. **エラーメッセージ**: 完全なエラーメッセージとトレースバック
2. **環境情報**: OS、Pythonバージョン、パッケージバージョン
3. **実行コマンド**: 実行したコマンドの詳細
4. **ログファイル**: 関連するログファイルの内容

```bash
# 環境情報の収集
python -c "
import sys
import platform
import numpy as np
import pandas as pd
print(f'Python: {sys.version}')
print(f'Platform: {platform.platform()}')
print(f'NumPy: {np.__version__}')
print(f'Pandas: {pd.__version__}')
" 