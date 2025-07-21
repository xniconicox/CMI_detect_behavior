# 前処理結果確認ガイド

このガイドでは、前処理結果の妥当性を確認する方法を説明します。

## 📋 前処理の状況

ログを見る限り、前処理は正常に動作しています：

- ✅ センサーデータの処理完了
- ✅ タブラーデータの処理完了  
- ✅ ToFデータの処理完了
- ✅ ラベルエンコーディング完了
- ✅ キャッシュ機能が正常に動作

ただし、以下の警告が発生しています：
- ⚠️ DataFrameの断片化警告（パフォーマンスに影響する可能性）

## 🔧 修正された問題

### 1. スクリプトのバグ修正
`scripts/run_preprocessing.sh`の`EXPERIMENT_NAME`変数が未定義だった問題を修正しました。

## 📊 前処理結果の確認方法

### 1. 簡単確認（推奨）

```bash
# コマンドラインでの簡単確認
python scripts/quick_check.py --experiment preprocess_v2_ws64
python scripts/quick_check.py --experiment preprocess_v2_ws128
```

### 2. Jupyter Notebookでの確認

```bash
# notebooksディレクトリに移動
cd notebooks

# Jupyterを起動
jupyter notebook
```

以下のノートブックを実行：

#### 簡単確認用
- `quick_check.ipynb` - 基本的な統計情報とファイル確認

#### 詳細分析用  
- `preprocessing_analysis.ipynb` - 詳細な可視化と分析

### 3. 詳細な妥当性チェック

```bash
# データの整合性、異常値、欠損値などをチェック
python scripts/validate_preprocessing.py --experiment preprocess_v2_ws64
python scripts/validate_preprocessing.py --experiment preprocess_v2_ws128
```

### 4. 可視化

```bash
# データの分布や特徴を可視化
python scripts/visualize_preprocessing.py --experiment preprocess_v2_ws64
python scripts/visualize_preprocessing.py --experiment preprocess_v2_ws128
```

## 📈 確認すべき項目

### 1. データ整合性
- [ ] サンプル数が全てのデータで一致しているか
- [ ] ラベル数とデータ数が一致しているか
- [ ] ファイルサイズが妥当か

### 2. データ品質
- [ ] 欠損値がないか
- [ ] 異常値が多すぎないか
- [ ] データの範囲が妥当か

### 3. ラベル分布
- [ ] クラス不均衡が大きすぎないか
- [ ] 全てのラベルが含まれているか

### 4. データ統計
- [ ] センサーデータの統計が妥当か
- [ ] ToFデータの統計が妥当か
- [ ] タブラーデータの統計が妥当か

## 🎯 期待される結果

### データ形状
- **windows**: (10147, 64, 18) または (10147, 128, 18)
- **demographics**: (10147, 7)
- **tabular**: (10147, 392)
- **tof_windows**: (10147, 64, 5, 8, 8) または (10147, 128, 5, 8, 8)
- **labels**: (10147,)

### ラベル分布
- 18クラス
- 各クラスに数百〜数千のサンプル
- クラス不均衡比 < 10

### データ統計
- センサーデータ: 平均0付近、標準偏差1付近
- ToFデータ: 正の値、平均50-100程度
- 欠損値: なし

## 🚨 問題が見つかった場合

### 1. データ整合性の問題
```bash
# キャッシュをクリアして再実行
./scripts/run_preprocessing.sh preprocess_v2 --no-cache
```

### 2. メモリ不足
```bash
# バッチサイズを小さくするか、データを分割して処理
```

### 3. パフォーマンス警告
```bash
# DataFrameの断片化を避けるため、feature_engineering.pyを修正
```

## 📝 次のステップ

前処理結果が正常であることを確認したら：

1. **学習データの準備完了**
2. **モデル学習の実行**
3. **評価とチューニング**

## 🔍 トラブルシューティング

### よくある問題

1. **ファイルが見つからない**
   - パスが正しいか確認
   - 前処理が完了しているか確認

2. **メモリエラー**
   - データサイズが大きすぎる場合、サンプリングして確認

3. **データ型エラー**
   - pickleファイルの読み込みエラー、バージョン互換性を確認

### サポート

問題が解決しない場合は、以下を確認してください：
- ログファイルの詳細
- システムリソース（メモリ、ディスク容量）
- Python環境とライブラリのバージョン 