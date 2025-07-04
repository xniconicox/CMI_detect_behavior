#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CMIコンペ ベースライン実行スクリプト
前処理からモデル訓練、提出ファイル作成まで一括実行
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_notebook():
    """前処理ノートブックの実行"""
    print("="*60)
    print("Step 1: 前処理ノートブックの実行")
    print("="*60)
    
    notebook_path = "notebooks/preprocess.ipynb"
    
    if not os.path.exists(notebook_path):
        print(f"エラー: ノートブックが見つかりません: {notebook_path}")
        return False
    
    try:
        # jupyter nbconvertでノートブックを実行
        cmd = [
            "jupyter", "nbconvert", 
            "--to", "notebook", 
            "--execute", 
            "--output", "preprocess_executed.ipynb",
            notebook_path
        ]
        
        print(f"実行コマンド: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 前処理ノートブック実行完了")
            return True
        else:
            print(f"❌ 前処理ノートブック実行エラー:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ 前処理ノートブック実行エラー: {e}")
        return False

def run_baseline_model():
    """ベースラインモデルの実行"""
    print("\n" + "="*60)
    print("Step 2: ベースラインモデルの実行")
    print("="*60)
    
    model_script = "src/baseline_model.py"
    
    if not os.path.exists(model_script):
        print(f"エラー: モデルスクリプトが見つかりません: {model_script}")
        return False
    
    try:
        # Pythonスクリプトを実行
        cmd = [sys.executable, model_script]
        
        print(f"実行コマンド: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ ベースラインモデル実行完了")
            print("\n実行結果:")
            print(result.stdout)
            return True
        else:
            print(f"❌ ベースラインモデル実行エラー:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ ベースラインモデル実行エラー: {e}")
        return False

def check_output_files():
    """出力ファイルの確認"""
    print("\n" + "="*60)
    print("Step 3: 出力ファイルの確認")
    print("="*60)
    
    output_dir = Path("output")
    required_files = [
        "train_features.csv",
        "test_features.csv", 
        "submission.csv"
    ]
    
    all_exist = True
    for file_name in required_files:
        file_path = output_dir / file_name
        if file_path.exists():
            size = file_path.stat().st_size
            print(f"✅ {file_name}: {size:,} bytes")
        else:
            print(f"❌ {file_name}: ファイルが見つかりません")
            all_exist = False
    
    if all_exist:
        print("\n🎉 全ての出力ファイルが正常に作成されました！")
        
        # 提出ファイルの内容確認
        submission_path = output_dir / "submission.csv"
        if submission_path.exists():
            import pandas as pd
            submission = pd.read_csv(submission_path)
            print(f"\n提出ファイル情報:")
            print(f"- 行数: {len(submission):,}")
            print(f"- 列数: {len(submission.columns)}")
            print(f"- 予測ジェスチャー数: {submission['gesture'].nunique()}")
            print(f"- 予測分布:")
            print(submission['gesture'].value_counts())
    
    return all_exist

def main():
    """メイン実行関数"""
    start_time = time.time()
    
    print("🚀 CMIコンペ ベースライン実行開始")
    print(f"開始時刻: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: 前処理ノートブック実行
    if not run_notebook():
        print("❌ 前処理でエラーが発生しました。手動でノートブックを実行してください。")
        return
    
    # Step 2: ベースラインモデル実行
    if not run_baseline_model():
        print("❌ モデル実行でエラーが発生しました。")
        return
    
    # Step 3: 出力ファイル確認
    check_output_files()
    
    # 実行時間の表示
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n" + "="*60)
    print("🏁 ベースライン実行完了")
    print(f"実行時間: {elapsed_time:.1f}秒 ({elapsed_time/60:.1f}分)")
    print(f"終了時刻: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    print("\n📋 次のステップ:")
    print("1. output/submission.csv をKaggleに提出")
    print("2. スコアを確認して改善点を検討")
    print("3. 必要に応じてハイパーパラメータチューニングを実施")

if __name__ == "__main__":
    main() 