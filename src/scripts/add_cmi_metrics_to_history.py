#!/usr/bin/env python3
"""
学習履歴にCMI評価指標を追加するスクリプト

既存の学習履歴JSONファイルにCMI評価指標（CMI Score、Binary F1、Macro F1）を追加します。
"""

import json
import numpy as np
import argparse
from pathlib import Path
import sys
import os

# CMI評価モジュールをインポート
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.cmi_evaluation import add_cmi_metrics_to_history, create_cmi_metrics_history, calculate_cmi_score

def load_training_history(history_path):
    """学習履歴JSONファイルを読み込み"""
    with open(history_path, 'r') as f:
        history = json.load(f)
    return history

def save_training_history(history, save_path):
    """学習履歴JSONファイルを保存"""
    with open(save_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"学習履歴を保存しました: {save_path}")

def add_cmi_metrics_to_existing_history(history_path, y_true, y_pred, label_encoder=None, 
                                       output_path=None, verbose=False):
    """
    既存の学習履歴にCMI評価指標を追加
    
    Parameters:
    -----------
    history_path : str or Path
        学習履歴JSONファイルのパス
    y_true : array-like
        真のラベル
    y_pred : array-like
        予測ラベル
    label_encoder : LabelEncoder, optional
        ラベルエンコーダー
    output_path : str or Path, optional
        出力先パス（Noneの場合は元のファイルを上書き）
    verbose : bool, default=False
        詳細ログの出力フラグ
    """
    try:
        # 学習履歴を読み込み
        history = load_training_history(history_path)
        
        if verbose:
            print(f"学習履歴を読み込みました: {history_path}")
            print(f"エポック数: {len(history.get('training_metrics', {}).get('loss', history.get('loss', [])))}")
        
        # CMI評価指標を追加
        updated_history = add_cmi_metrics_to_history(
            history, y_true, y_pred, label_encoder, verbose
        )
        
        # 出力先パスを決定
        if output_path is None:
            output_path = history_path
        
        # 学習履歴を保存
        save_training_history(updated_history, output_path)
        
        return updated_history
        
    except Exception as e:
        print(f"CMI評価指標の追加でエラー: {str(e)}")
        return None

def create_demo_cmi_history(output_path, num_epochs=50, verbose=False):
    """
    デモ用のCMI評価指標付き学習履歴を作成
    
    Parameters:
    -----------
    output_path : str or Path
        出力先パス
    num_epochs : int, default=50
        エポック数
    verbose : bool, default=False
        詳細ログの出力フラグ
    """
    try:
        # デモデータを作成（18クラス分類）
        np.random.seed(42)
        y_true = np.random.randint(0, 18, 1000)
        y_pred = np.random.randint(0, 18, 1000)
        
        # CMI評価指標を計算
        cmi_score, binary_f1, macro_f1, accuracy = calculate_cmi_score(
            y_pred, y_true, None, verbose
        )
        
        # 学習履歴辞書を作成
        history_dict = {
            'epochs_completed': num_epochs,
            'training_metrics': {
                'loss': [2.0 - i * 0.02 for i in range(num_epochs)],
                'accuracy': [0.1 + i * 0.015 for i in range(num_epochs)],
                'cmi_score': [cmi_score] * num_epochs,
                'binary_f1': [binary_f1] * num_epochs,
                'macro_f1': [macro_f1] * num_epochs,
                'learning_rate': [0.001] * num_epochs
            },
            'validation_metrics': {
                'loss': [2.1 - i * 0.018 for i in range(num_epochs)],
                'accuracy': [0.08 + i * 0.014 for i in range(num_epochs)],
                'cmi_score': [cmi_score] * num_epochs,
                'binary_f1': [binary_f1] * num_epochs,
                'macro_f1': [macro_f1] * num_epochs
            },
            'metadata': {
                'timestamp': '2025-01-01 00:00:00',
                'model_config': {
                    'num_classes': 18,
                    'label_encoder_used': False
                },
                'best_epoch': 45,
                'best_val_loss': 1.29,
                'best_val_accuracy': 0.78
            },
            'summary': {
                'final_train_loss': 1.0,
                'final_train_accuracy': 0.85,
                'final_val_loss': 1.29,
                'final_val_accuracy': 0.78,
                'final_cmi_score': cmi_score,
                'final_binary_f1': binary_f1,
                'final_macro_f1': macro_f1
            }
        }
        
        # 学習履歴を保存
        save_training_history(history_dict, output_path)
        
        if verbose:
            print(f"デモ用のCMI評価指標付き学習履歴を作成しました:")
            print(f"  CMI Score: {cmi_score:.4f}")
            print(f"  Binary F1: {binary_f1:.4f}")
            print(f"  Macro F1: {macro_f1:.4f}")
            print(f"  Accuracy: {accuracy:.4f}")
        
        return history_dict
        
    except Exception as e:
        print(f"デモ用学習履歴の作成でエラー: {str(e)}")
        return None

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='学習履歴にCMI評価指標を追加するスクリプト')
    parser.add_argument('--history', type=str, help='学習履歴JSONファイルのパス')
    parser.add_argument('--output', type=str, help='出力先パス')
    parser.add_argument('--demo', action='store_true', help='デモ用の学習履歴を作成')
    parser.add_argument('--epochs', type=int, default=50, help='デモ用のエポック数')
    parser.add_argument('--verbose', action='store_true', help='詳細ログを出力')
    
    args = parser.parse_args()
    
    if args.demo:
        # デモ用の学習履歴を作成
        output_path = args.output or "demo_training_history_with_cmi.json"
        create_demo_cmi_history(output_path, args.epochs, args.verbose)
    elif args.history:
        # 既存の学習履歴にCMI評価指標を追加
        print("既存の学習履歴にCMI評価指標を追加するには、y_trueとy_predのデータが必要です。")
        print("現在はデモ用のデータを使用します。")
        
        # デモデータを作成
        np.random.seed(42)
        y_true = np.random.randint(0, 18, 1000)
        y_pred = np.random.randint(0, 18, 1000)
        
        add_cmi_metrics_to_existing_history(
            args.history, y_true, y_pred, None, args.output, args.verbose
        )
    else:
        print("使用方法:")
        print("  # デモ用の学習履歴を作成")
        print("  python add_cmi_metrics_to_history.py --demo --output demo_history.json")
        print("")
        print("  # 既存の学習履歴にCMI評価指標を追加")
        print("  python add_cmi_metrics_to_history.py --history path/to/history.json")

if __name__ == "__main__":
    # コマンドライン引数がない場合のテスト実行
    import sys
    if len(sys.argv) == 1:
        print("デモ用のCMI評価指標付き学習履歴を作成します...")
        create_demo_cmi_history("demo_training_history_with_cmi.json", 50, True)
        
        print("\n" + "="*60)
        print("CMI評価指標追加スクリプトの使用方法")
        print("="*60)
        
        print("""
【基本的な使用方法】

1. デモ用のCMI評価指標付き学習履歴を作成:
   python src/scripts/add_cmi_metrics_to_history.py --demo --output demo_history.json

2. 既存の学習履歴にCMI評価指標を追加:
   python src/scripts/add_cmi_metrics_to_history.py --history path/to/training_history.json --output updated_history.json

3. 詳細ログ付きで実行:
   python src/scripts/add_cmi_metrics_to_history.py --demo --output demo_history.json --verbose

4. カスタムエポック数でデモ作成:
   python src/scripts/add_cmi_metrics_to_history.py --demo --output demo_history.json --epochs 100

【CMI評価指標の内容】

追加されるCMI評価指標:
- cmi_score: CMIスコア（Binary F1 + Macro F1の平均）
- binary_f1: ターゲット vs 非ターゲットの2値分類F1スコア
- macro_f1: 全ジェスチャーのマクロF1スコア

【学習履歴の形式】

新しい形式（推奨）:
{
  "training_metrics": {
    "loss": [...],
    "accuracy": [...],
    "cmi_score": [...],
    "binary_f1": [...],
    "macro_f1": [...]
  },
  "validation_metrics": {
    "loss": [...],
    "accuracy": [...],
    "cmi_score": [...],
    "binary_f1": [...],
    "macro_f1": [...]
  }
}

古い形式:
{
  "loss": [...],
  "accuracy": [...],
  "cmi_score": [...],
  "binary_f1": [...],
  "macro_f1": [...]
}

【関連スクリプト】

学習履歴の可視化:
python src/scripts/visualize_training_history.py --history demo_history.json --title "CMI評価指標付き学習履歴"

【例】

# デモ用のCMI評価指標付き学習履歴を作成
python src/scripts/add_cmi_metrics_to_history.py --demo --output demo_history.json

# CMI評価指標付き学習履歴を可視化
python src/scripts/visualize_training_history.py --history demo_history.json --title "CMI評価指標付き学習履歴"

# 既存の学習履歴にCMI評価指標を追加
python src/scripts/add_cmi_metrics_to_history.py --history path/to/training_history.json --output updated_history.json
""")
    else:
        main() 