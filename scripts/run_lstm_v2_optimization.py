#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM v2最適化実行スクリプト

分析結果に基づいて以下の最適化を実行します：
1. Attention融合方式のエラー修正
2. Optunaによるハイパーパラメータ最適化
3. Demographics特徴量エンジニアリング最適化
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import argparse

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def run_demographics_optimization():
    """
    Demographics特徴量最適化を実行
    """
    print("="*60)
    print("Demographics特徴量最適化を実行中...")
    print("="*60)
    
    try:
        # demographics_optimizer.pyを実行
        result = subprocess.run([
            sys.executable, 
            str(project_root / "src" / "demographics_optimizer.py")
        ], capture_output=True, text=True, cwd=str(project_root))
        
        if result.returncode == 0:
            print("✓ Demographics特徴量最適化が完了しました")
            print(result.stdout)
        else:
            print("✗ Demographics特徴量最適化でエラーが発生しました")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"✗ Demographics特徴量最適化の実行中にエラー: {e}")
        return False
    
    return True

def run_hyperparameter_optimization():
    """
    ハイパーパラメータ最適化を実行
    """
    print("="*60)
    print("ハイパーパラメータ最適化を実行中...")
    print("="*60)
    
    try:
        # Jupyter notebookを実行
        result = subprocess.run([
            "jupyter", "nbconvert", 
            "--to", "notebook",
            "--execute",
            str(project_root / "notebooks" / "lstm_v2_hyperparameter_optimization.ipynb"),
            "--output", str(project_root / "notebooks" / "lstm_v2_hyperparameter_optimization_executed.ipynb")
        ], capture_output=True, text=True, cwd=str(project_root))
        
        if result.returncode == 0:
            print("✓ ハイパーパラメータ最適化が完了しました")
        else:
            print("✗ ハイパーパラメータ最適化でエラーが発生しました")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"✗ ハイパーパラメータ最適化の実行中にエラー: {e}")
        return False
    
    return True

def check_prerequisites():
    """
    前提条件をチェック
    """
    print("前提条件をチェック中...")
    
    # 必要なディレクトリの存在確認
    required_dirs = [
        project_root / "data" / "processed",
        project_root / "src",
        project_root / "notebooks"
    ]
    
    for dir_path in required_dirs:
        if not dir_path.exists():
            print(f"✗ 必要なディレクトリが見つかりません: {dir_path}")
            return False
    
    # 必要なファイルの存在確認
    required_files = [
        project_root / "src" / "lstm_v2_model.py",
        project_root / "src" / "demographics_optimizer.py",
        project_root / "notebooks" / "lstm_v2_hyperparameter_optimization.ipynb"
    ]
    
    for file_path in required_files:
        if not file_path.exists():
            print(f"✗ 必要なファイルが見つかりません: {file_path}")
            return False
    
    print("✓ 前提条件チェック完了")
    return True

def print_optimization_summary():
    """
    最適化の概要を表示
    """
    print("\n" + "="*60)
    print("LSTM v2最適化の概要")
    print("="*60)
    
    print("【実行内容】")
    print("1. ✓ Attention融合方式のエラー修正")
    print("   - 次元不一致エラーの修正")
    print("   - Demographics特徴量の投影層追加")
    
    print("\n2. Demographics特徴量エンジニアリング最適化")
    print("   - 特徴量重要度分析")
    print("   - 特徴量選択最適化")
    print("   - 多項式特徴量・PCA変換テスト")
    
    print("\n3. ハイパーパラメータ最適化")
    print("   - Optuna による30試行最適化")
    print("   - 融合方式の比較選択")
    print("   - アーキテクチャ・訓練パラメータ最適化")
    
    print("\n【期待される結果】")
    print("- LSTM v1性能（F1-macro: 0.5200）を上回る")
    print("- Demographics情報の効果的な活用")
    print("- 最適な融合方式の特定")
    
    print("\n【問題の解決】")
    print("✓ 主要因：ハイパーパラメータ最適化の欠如")
    print("✓ 副次因：Attention融合方式のエラー")
    print("✓ 検証項目：Demographics特徴量エンジニアリング")
    
    print("="*60)

def main():
    """
    メイン実行関数
    """
    parser = argparse.ArgumentParser(description="LSTM v2最適化実行スクリプト")
    parser.add_argument("--skip-demographics", action="store_true", 
                       help="Demographics特徴量最適化をスキップ")
    parser.add_argument("--skip-hyperopt", action="store_true", 
                       help="ハイパーパラメータ最適化をスキップ")
    parser.add_argument("--summary-only", action="store_true", 
                       help="概要のみ表示して終了")
    
    args = parser.parse_args()
    
    print("LSTM v2最適化実行スクリプト")
    print(f"実行開始: {datetime.now()}")
    print("="*60)
    
    # 概要のみ表示
    if args.summary_only:
        print_optimization_summary()
        return
    
    # 前提条件チェック
    if not check_prerequisites():
        print("前提条件が満たされていません。実行を中止します。")
        return
    
    # 最適化概要表示
    print_optimization_summary()
    
    # 実行確認
    response = input("\n最適化を実行しますか？ (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("実行をキャンセルしました。")
        return
    
    success_count = 0
    total_steps = 0
    
    # Demographics特徴量最適化
    if not args.skip_demographics:
        total_steps += 1
        if run_demographics_optimization():
            success_count += 1
    
    # ハイパーパラメータ最適化
    if not args.skip_hyperopt:
        total_steps += 1
        if run_hyperparameter_optimization():
            success_count += 1
    
    # 結果サマリー
    print("\n" + "="*60)
    print("最適化実行結果")
    print("="*60)
    print(f"成功: {success_count}/{total_steps} ステップ")
    
    if success_count == total_steps:
        print("✓ 全ての最適化が正常に完了しました！")
        print("\n【次のステップ】")
        print("1. 最適化結果の確認:")
        print("   - results/lstm_v2_optimization/optimization_results.json")
        print("   - results/demographics_optimization/demographics_optimization_results.json")
        print("2. 最適化されたモデルでの推論:")
        print("   - models/lstm_v2_optimized/best_model.h5")
        print("3. 性能比較:")
        print("   - LSTM v1 vs LSTM v2 (最適化後)")
    else:
        print("⚠ 一部の最適化でエラーが発生しました。")
        print("ログを確認して問題を解決してください。")
    
    print(f"\n実行完了: {datetime.now()}")
    print("="*60)

if __name__ == "__main__":
    main() 