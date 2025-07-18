#!/usr/bin/env python3
"""
学習履歴可視化スクリプト
保存された学習履歴から学習曲線を生成
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import sys
import os

# CMI評価モジュールをインポート
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.cmi_evaluation import calculate_cmi_score, get_target_gestures, get_non_target_gestures

def load_training_history(history_path):
    """
    学習履歴JSONファイルを読み込み
    
    Parameters:
    -----------
    history_path : str or Path
        学習履歴JSONファイルのパス
    
    Returns:
    --------
    dict : 学習履歴データ
    """
    with open(history_path, 'r') as f:
        history = json.load(f)
    return history

def plot_training_curves(history, save_path=None, title="Training History"):
    """
    学習曲線をプロット（CMI評価指標対応版）
    
    Parameters:
    -----------
    history : dict
        学習履歴データ
    save_path : str, optional
        保存先パス
    title : str
        グラフのタイトル
    """
    # CMI評価指標が含まれているかチェック
    has_cmi_metrics = False
    if 'training_metrics' in history and 'validation_metrics' in history:
        has_cmi_metrics = (
            'cmi_score' in history['training_metrics'] or 
            'binary_f1' in history['training_metrics'] or
            'macro_f1' in history['training_metrics']
        )
    
    # サブプロット数を決定
    if has_cmi_metrics:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
    
    fig.suptitle(title, fontsize=16)
    
    # 新しい形式と古い形式の両方に対応
    if 'training_metrics' in history and 'validation_metrics' in history:
        # 新しい詳細形式
        train_loss = history['training_metrics']['loss']
        val_loss = history['validation_metrics']['loss']
        train_accuracy = history['training_metrics'].get('accuracy', [])
        val_accuracy = history['validation_metrics'].get('accuracy', [])
        lr_history = history.get('learning_rate_history', [])
        
        # CMI評価指標
        train_cmi_score = history['training_metrics'].get('cmi_score', [])
        val_cmi_score = history['validation_metrics'].get('cmi_score', [])
        train_binary_f1 = history['training_metrics'].get('binary_f1', [])
        val_binary_f1 = history['validation_metrics'].get('binary_f1', [])
        train_macro_f1 = history['training_metrics'].get('macro_f1', [])
        val_macro_f1 = history['validation_metrics'].get('macro_f1', [])
        
        # メタデータから最適エポック情報を取得
        best_epoch = history['metadata'].get('best_epoch', None)
        best_val_loss = history['metadata'].get('best_val_loss', None)
        
        # サマリー情報
        summary = history.get('summary', {})
        
    else:
        # 古い形式
        train_loss = history['loss']
        val_loss = history['val_loss']
        train_accuracy = history.get('accuracy', [])
        val_accuracy = history.get('val_accuracy', [])
        lr_history = history.get('lr', history.get('learning_rate', []))
        
        # CMI評価指標（古い形式では通常含まれていない）
        train_cmi_score = []
        val_cmi_score = []
        train_binary_f1 = []
        val_binary_f1 = []
        train_macro_f1 = []
        val_macro_f1 = []
        
        # 最適エポックを計算
        best_epoch = np.argmin(val_loss) if val_loss else None
        best_val_loss = min(val_loss) if val_loss else None
        
        # サマリー情報を作成
        summary = {
            'final_train_loss': train_loss[-1] if train_loss else 0,
            'final_val_loss': val_loss[-1] if val_loss else 0,
            'final_train_accuracy': train_accuracy[-1] if train_accuracy else 0,
            'final_val_accuracy': val_accuracy[-1] if val_accuracy else 0
        }
    
    epochs = range(1, len(train_loss) + 1)
    
    # Loss曲線
    axes[0].plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    if val_loss:
        axes[0].plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy曲線
    if train_accuracy:
        axes[1].plot(epochs, train_accuracy, 'b-', label='Training Accuracy', linewidth=2)
    if val_accuracy:
        axes[1].plot(epochs, val_accuracy, 'r-', label='Validation Accuracy', linewidth=2)
    axes[1].set_title('Model Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning Rate曲線
    if lr_history:
        axes[2].plot(epochs, lr_history, 'g-', label='Learning Rate', linewidth=2)
        axes[2].set_title('Learning Rate')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_yscale('log')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'Learning Rate\ndata not available', 
                   ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title('Learning Rate')
    
    # Loss差分（過学習の指標）
    if val_loss:
        loss_diff = np.array(val_loss) - np.array(train_loss)
        axes[3].plot(epochs, loss_diff, 'purple', label='Val Loss - Train Loss', linewidth=2)
        axes[3].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[3].set_title('Overfitting Indicator')
        axes[3].set_xlabel('Epoch')
        axes[3].set_ylabel('Loss Difference')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
    else:
        axes[3].text(0.5, 0.5, 'Validation Loss\nnot available', 
                   ha='center', va='center', transform=axes[3].transAxes)
        axes[3].set_title('Overfitting Indicator')
    
    # CMI評価指標の可視化（利用可能な場合）
    if has_cmi_metrics:
        # CMI Score
        if train_cmi_score and val_cmi_score:
            axes[4].plot(epochs, train_cmi_score, 'b-', label='Training CMI Score', linewidth=2)
            axes[4].plot(epochs, val_cmi_score, 'r-', label='Validation CMI Score', linewidth=2)
            axes[4].set_title('CMI Score')
            axes[4].set_xlabel('Epoch')
            axes[4].set_ylabel('CMI Score')
            axes[4].legend()
            axes[4].grid(True, alpha=0.3)
        else:
            axes[4].text(0.5, 0.5, 'CMI Score\ndata not available', 
                       ha='center', va='center', transform=axes[4].transAxes)
            axes[4].set_title('CMI Score')
        
        # Binary F1 vs Macro F1
        if (train_binary_f1 and val_binary_f1) or (train_macro_f1 and val_macro_f1):
            if train_binary_f1 and val_binary_f1:
                axes[5].plot(epochs, train_binary_f1, 'b-', label='Training Binary F1', linewidth=2)
                axes[5].plot(epochs, val_binary_f1, 'r-', label='Validation Binary F1', linewidth=2)
            if train_macro_f1 and val_macro_f1:
                axes[5].plot(epochs, train_macro_f1, 'b--', label='Training Macro F1', linewidth=2)
                axes[5].plot(epochs, val_macro_f1, 'r--', label='Validation Macro F1', linewidth=2)
            axes[5].set_title('Binary F1 vs Macro F1')
            axes[5].set_xlabel('Epoch')
            axes[5].set_ylabel('F1 Score')
            axes[5].legend()
            axes[5].grid(True, alpha=0.3)
        else:
            axes[5].text(0.5, 0.5, 'F1 Scores\ndata not available', 
                       ha='center', va='center', transform=axes[5].transAxes)
            axes[5].set_title('Binary F1 vs Macro F1')
    
    # 最適なエポックをマーク
    if best_epoch is not None:
        axes[0].axvline(x=best_epoch + 1, color='green', linestyle='--', alpha=0.7, 
                      label=f'Best Epoch: {best_epoch + 1}')
        axes[0].legend()
    
    # 統計情報を表示
    stats_text = f"""
    Epochs Completed: {len(epochs)}
    Final Training Loss: {summary.get('final_train_loss', 0):.4f}
    Final Validation Loss: {summary.get('final_val_loss', 0):.4f}
    Final Training Accuracy: {summary.get('final_train_accuracy', 0):.4f}
    Final Validation Accuracy: {summary.get('final_val_accuracy', 0):.4f}
    """
    
    if best_epoch is not None and best_val_loss is not None:
        stats_text += f"""
    Best Validation Loss: {best_val_loss:.4f}
    Best Epoch: {best_epoch + 1}
    """
    
    # CMI評価指標の統計情報
    if has_cmi_metrics:
        if train_cmi_score:
            stats_text += f"""
    Final Training CMI Score: {train_cmi_score[-1]:.4f}
    """
        if val_cmi_score:
            stats_text += f"""
    Final Validation CMI Score: {val_cmi_score[-1]:.4f}
    """
        if train_binary_f1:
            stats_text += f"""
    Final Training Binary F1: {train_binary_f1[-1]:.4f}
    """
        if val_binary_f1:
            stats_text += f"""
    Final Validation Binary F1: {val_binary_f1[-1]:.4f}
    """
        if train_macro_f1:
            stats_text += f"""
    Final Training Macro F1: {train_macro_f1[-1]:.4f}
    """
        if val_macro_f1:
            stats_text += f"""
    Final Validation Macro F1: {val_macro_f1[-1]:.4f}
    """
    
    plt.figtext(0.02, 0.02, stats_text, fontsize=10, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"学習曲線を保存しました: {save_path}")
    
    plt.show()

def plot_comparison(history_paths, labels=None, save_path=None):
    """
    複数の学習履歴を比較（CMI評価指標対応版）
    
    Parameters:
    -----------
    history_paths : list
        学習履歴JSONファイルのパスのリスト
    labels : list, optional
        各履歴のラベル
    save_path : str, optional
        保存先パス
    """
    if labels is None:
        labels = [f"Model {i+1}" for i in range(len(history_paths))]
    
    # CMI評価指標が含まれているかチェック
    has_cmi_metrics = False
    for path in history_paths:
        history = load_training_history(path)
        if 'training_metrics' in history and 'validation_metrics' in history:
            if ('cmi_score' in history['training_metrics'] or 
                'binary_f1' in history['training_metrics'] or
                'macro_f1' in history['training_metrics']):
                has_cmi_metrics = True
                break
    
    # サブプロット数を決定
    if has_cmi_metrics:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        axes = axes.flatten()
    
    fig.suptitle('Training History Comparison', fontsize=16)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    for i, (path, label) in enumerate(zip(history_paths, labels)):
        history = load_training_history(path)
        color = colors[i % len(colors)]
        
        # 新しい形式と古い形式の両方に対応
        if 'training_metrics' in history and 'validation_metrics' in history:
            # 新しい詳細形式
            train_loss = history['training_metrics']['loss']
            val_loss = history['validation_metrics']['loss']
            train_accuracy = history['training_metrics'].get('accuracy', [])
            val_accuracy = history['validation_metrics'].get('accuracy', [])
            
            # CMI評価指標
            train_cmi_score = history['training_metrics'].get('cmi_score', [])
            val_cmi_score = history['validation_metrics'].get('cmi_score', [])
            train_binary_f1 = history['training_metrics'].get('binary_f1', [])
            val_binary_f1 = history['validation_metrics'].get('binary_f1', [])
            train_macro_f1 = history['training_metrics'].get('macro_f1', [])
            val_macro_f1 = history['validation_metrics'].get('macro_f1', [])
        else:
            # 古い形式
            train_loss = history['loss']
            val_loss = history['val_loss']
            train_accuracy = history.get('accuracy', [])
            val_accuracy = history.get('val_accuracy', [])
            
            # CMI評価指標（古い形式では通常含まれていない）
            train_cmi_score = []
            val_cmi_score = []
            train_binary_f1 = []
            val_binary_f1 = []
            train_macro_f1 = []
            val_macro_f1 = []
        
        epochs = range(1, len(train_loss) + 1)
        
        # Loss比較
        if val_loss:
            axes[0].plot(epochs, val_loss, color=color, label=f'{label} (Val)', linewidth=2)
        axes[0].plot(epochs, train_loss, color=color, linestyle='--', alpha=0.7, label=f'{label} (Train)')
        
        # Accuracy比較
        if val_accuracy:
            axes[1].plot(epochs, val_accuracy, color=color, label=f'{label} (Val)', linewidth=2)
        if train_accuracy:
            axes[1].plot(epochs, train_accuracy, color=color, linestyle='--', alpha=0.7, label=f'{label} (Train)')
        
        # CMI評価指標の比較（利用可能な場合）
        if has_cmi_metrics:
            # CMI Score比較
            if train_cmi_score and val_cmi_score:
                axes[2].plot(epochs, train_cmi_score, color=color, linestyle='--', alpha=0.7, 
                           label=f'{label} (Train)', linewidth=2)
                axes[2].plot(epochs, val_cmi_score, color=color, label=f'{label} (Val)', linewidth=2)
            
            # Binary F1 vs Macro F1比較
            if train_binary_f1 and val_binary_f1:
                axes[3].plot(epochs, train_binary_f1, color=color, linestyle='--', alpha=0.7, 
                           label=f'{label} Binary F1 (Train)', linewidth=2)
                axes[3].plot(epochs, val_binary_f1, color=color, label=f'{label} Binary F1 (Val)', linewidth=2)
            if train_macro_f1 and val_macro_f1:
                axes[3].plot(epochs, train_macro_f1, color=color, linestyle=':', alpha=0.7, 
                           label=f'{label} Macro F1 (Train)', linewidth=2)
                axes[3].plot(epochs, val_macro_f1, color=color, linestyle='-.', alpha=0.7, 
                           label=f'{label} Macro F1 (Val)', linewidth=2)
    
    axes[0].set_title('Loss Comparison')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title('Accuracy Comparison')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    if has_cmi_metrics:
        axes[2].set_title('CMI Score Comparison')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('CMI Score')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        axes[3].set_title('F1 Score Comparison')
        axes[3].set_xlabel('Epoch')
        axes[3].set_ylabel('F1 Score')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"比較グラフを保存しました: {save_path}")
    
    plt.show()

def print_usage():
    """使用方法を表示"""
    print("""
=== CMI評価指標対応学習履歴可視化スクリプト ===

【基本的な使用方法】

1. 単一の学習履歴を可視化:
   python src/scripts/visualize_training_history.py --history path/to/training_history.json

2. カスタムタイトルで保存:
   python src/scripts/visualize_training_history.py \\
     --history path/to/training_history.json \\
     --title "Final Model Training" \\
     --save results/training_curves.png

3. 複数モデルの比較:
   python src/scripts/visualize_training_history.py \\
     --compare model1_history.json model2_history.json \\
     --save comparison.png

【CMI評価指標の表示】

学習履歴にCMI評価指標が含まれている場合、以下の6つのグラフが表示されます:
- Model Loss: 損失の推移
- Model Accuracy: 精度の推移
- Learning Rate: 学習率の推移
- Overfitting Indicator: 過学習の指標
- CMI Score: CMIスコアの推移
- Binary F1 vs Macro F1: バイナリF1とマクロF1の比較

【CMI評価指標を追加するには】

学習履歴にCMI評価指標を追加するには、以下のスクリプトを使用してください:
python src/scripts/add_cmi_metrics_to_history.py --demo --output demo_history.json

【例】

# デモ用のCMI評価指標付き学習履歴を作成
python src/scripts/add_cmi_metrics_to_history.py --demo --output demo_history.json

# CMI評価指標付き学習履歴を可視化
python src/scripts/visualize_training_history.py --history demo_history.json --title "CMI評価指標付き学習履歴"

# 既存の学習履歴にCMI評価指標を追加
python src/scripts/add_cmi_metrics_to_history.py --history path/to/training_history.json --output updated_history.json
""")


def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='学習履歴可視化スクリプト')
    parser.add_argument('--history', type=str, required=True, help='学習履歴JSONファイルのパス')
    parser.add_argument('--save', type=str, help='保存先パス')
    parser.add_argument('--title', type=str, default='Training History', help='グラフのタイトル')
    parser.add_argument('--compare', type=str, nargs='+', help='比較する複数の学習履歴ファイル')
    
    args = parser.parse_args()
    
    if args.compare:
        # 複数ファイルの比較
        plot_comparison(args.compare, save_path=args.save)
    else:
        # 単一ファイルの可視化
        history = load_training_history(args.history)
        plot_training_curves(history, save_path=args.save, title=args.title)

if __name__ == "__main__":
    # コマンドライン引数がない場合のテスト実行
    import sys
    if len(sys.argv) == 1:
        # デフォルトパスでテスト
        default_path = "../output/experiments/final_model_w64_s16/results/training_history_attention.json"
        if Path(default_path).exists():
            print(f"テスト実行: {default_path}")
            history = load_training_history(default_path)
            plot_training_curves(history, title="Final Model Training History")
        else:
            print(f"学習履歴ファイルが見つかりません: {default_path}")
            print("\n=== 使用方法 ===")
            print_usage()
    else:
        main()


# 使用方法の例（コメントアウト）
"""
# 基本的な可視化
python src/scripts/visualize_training_history.py --history path/to/training_history.json

# カスタムタイトルで保存
python src/scripts/visualize_training_history.py \
  --history path/to/training_history.json \
  --title "Final Model Training" \
  --save results/training_curves.png

# 複数モデルの比較
python src/scripts/visualize_training_history.py \
  --compare model1_history.json model2_history.json \
  --save comparison.png

# CMI評価指標付き学習履歴の可視化
python src/scripts/visualize_training_history.py \
  --history demo_training_history_with_cmi.json \
  --title "CMI評価指標付き学習履歴"
"""