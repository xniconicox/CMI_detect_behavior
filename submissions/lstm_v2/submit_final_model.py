#!/usr/bin/env python3
"""
最終モデル提出プロセス管理スクリプト
最適パラメータでの最終モデル学習から提出ファイル作成まで全自動実行
"""

import os
import sys
import subprocess
import json
import time
from datetime import datetime

def run_command(command, description):
    """コマンドを実行し、結果を表示"""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    print(f"📝 実行コマンド: {command}")
    print()
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {description} 完了")
            if result.stdout:
                print("📄 出力:")
                print(result.stdout)
        else:
            print(f"❌ {description} エラー")
            if result.stderr:
                print("⚠️ エラー詳細:")
                print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ {description} 実行中にエラー: {e}")
        return False
    
    return True

def check_prerequisites():
    """前提条件チェック"""
    print("🔍 前提条件チェック")
    print("=" * 40)
    
    # 最適設定ファイルの存在確認
    config_path = 'results/final_model/best_config.json'
    if not os.path.exists(config_path):
        print(f"❌ 最適設定ファイルが見つかりません: {config_path}")
        return False
    
    # データファイルの存在確認
    data_files = [
        'data/processed/sensor_data_w64_s16.npy',
        'data/processed/demographics_data_w64_s16.npy',
        'data/processed/labels_w64_s16.npy'
    ]
    
    for file_path in data_files:
        if not os.path.exists(file_path):
            print(f"❌ データファイルが見つかりません: {file_path}")
            return False
    
    print("✅ 前提条件チェック完了")
    return True

def display_optimization_summary():
    """最適化結果サマリーを表示"""
    print("\n🏆 最適化結果サマリー")
    print("=" * 40)
    
    with open('results/final_model/best_config.json', 'r') as f:
        config = json.load(f)
    
    print(f"🎯 最高CMIスコア: {config['best_score']:.4f}")
    print(f"🔢 最適試行: Trial {config['best_trial']}")
    print(f"📊 完了試行数: {config['completed_trials']}")
    print(f"🔧 ウィンドウ設定: {config['window_config']}")
    
    print(f"\n📋 最適パラメータ:")
    best_params = config['best_params']
    fusion_types = ['concatenate', 'attention', 'gated']
    
    for param, value in best_params.items():
        if param == 'fusion_type':
            fusion_name = fusion_types[int(value)]
            print(f"   {param}: {fusion_name} ({value})")
        else:
            print(f"   {param}: {value}")

def main():
    """メイン提出プロセス"""
    print("🎉 最終モデル提出プロセス開始")
    print("=" * 80)
    print(f"⏰ 開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 前提条件チェック
    if not check_prerequisites():
        print("❌ 前提条件チェックに失敗しました")
        return
    
    # 最適化結果表示
    display_optimization_summary()
    
    # 仮想環境有効化コマンド
    venv_activate = "source .venv/bin/activate"
    
    # ステップ1: 最終モデル学習
    print(f"\n📚 ステップ1: 最終モデル学習 (150エポック)")
    print("予想時間: 2-3時間")
    print("⚠️ 学習を開始しますか？ (y/n): ", end="")
    
    user_input = input()
    if user_input.lower() != 'y':
        print("❌ 学習をキャンセルしました")
        return
    
    start_time = time.time()
    
    if not run_command(
        f"{venv_activate} && python train_final_model.py",
        "最終モデル学習 (150エポック)"
    ):
        print("❌ 最終モデル学習に失敗しました")
        return
    
    training_time = time.time() - start_time
    print(f"⏱️ 学習時間: {training_time/3600:.1f}時間")
    
    # ステップ2: 提出ファイル生成
    print(f"\n📄 ステップ2: 提出ファイル生成")
    
    if not run_command(
        f"{venv_activate} && python generate_submission.py",
        "提出ファイル生成"
    ):
        print("❌ 提出ファイル生成に失敗しました")
        return
    
    # 完了報告
    total_time = time.time() - start_time
    print(f"\n🎉 提出プロセス完了!")
    print("=" * 80)
    print(f"⏰ 完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏱️ 総実行時間: {total_time/3600:.1f}時間")
    print(f"📁 提出ファイル場所: results/submission/")
    print(f"🏆 モデル保存場所: results/final_model/")
    
    # 次のステップガイド
    print(f"\n📋 次のステップ:")
    print("1. 📊 results/submission/ のCSVファイルを確認")
    print("2. 🔍 CMIスコアと予測値分布を確認")
    print("3. 📤 CSVファイルを提出プラットフォームにアップロード")
    print("4. 🎯 リーダーボードでの結果を確認")

if __name__ == "__main__":
    main() 