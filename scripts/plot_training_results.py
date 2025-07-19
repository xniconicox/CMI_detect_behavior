#!/usr/bin/env python3
"""
学習結果の可視化スクリプト

学習完了後に、保存された学習履歴から学習曲線とクロスバリデーション結果を
プロットして保存する。

使用方法:
    python scripts/plot_training_results.py --experiment-name 20250717_preproc_train_v30 --trainer-name multimodal_v30
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent))


def load_training_histories(result_dir: Path) -> list:
    """保存された学習履歴を読み込む"""
    histories = []
    fold = 1
    
    while True:
        history_file = result_dir / f"training_history_fold{fold}.json"
        if not history_file.exists():
            break
        
        with open(history_file, 'r') as f:
            history = json.load(f)
        histories.append(history)
        fold += 1
    
    print(f"読み込んだ学習履歴: {len(histories)} folds")
    return histories


def load_cv_results(result_dir: Path) -> Dict[str, Any]:
    """クロスバリデーション結果を読み込む"""
    cv_file = result_dir / "cv_results.json"
    if cv_file.exists():
        with open(cv_file, 'r') as f:
            return json.load(f)
    
    # cv_results.jsonがない場合は、evaluation_results.jsonから基本情報を取得
    eval_file = result_dir / "evaluation_results.json"
    if eval_file.exists():
        with open(eval_file, 'r') as f:
            eval_results = json.load(f)
        
        # 基本的な結果を返す
        return {
            "fold_scores": [eval_results.get("macro_f1", 0)],
            "fold_cmi_scores": [eval_results.get("cmi_score", 0)],
            "mean_f1": eval_results.get("macro_f1", 0),
            "std_f1": 0.0,
            "mean_cmi": eval_results.get("cmi_score", 0),
            "std_cmi": 0.0,
        }
    
    return {}


def plot_training_curves(histories: list, save_path: Path) -> None:
    """学習曲線をプロットして保存する"""
    if not histories:
        print("学習履歴が見つかりません")
        return

    # プロット設定
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Curves (Multimodal Model V30)', fontsize=16)

    metrics = ['loss', 'accuracy']
    for i, metric in enumerate(metrics):
        # 訓練データ
        ax = axes[i, 0]
        for fold, history in enumerate(histories, 1):
            if metric in history:
                ax.plot(history[metric], label=f'Fold {fold}', alpha=0.7)
        
        ax.set_title(f'Training {metric.capitalize()}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 検証データ
        ax = axes[i, 1]
        for fold, history in enumerate(histories, 1):
            val_metric = f'val_{metric}'
            if val_metric in history:
                ax.plot(history[val_metric], label=f'Fold {fold}', alpha=0.7)
        
        ax.set_title(f'Validation {metric.capitalize()}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.capitalize())
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"学習曲線保存: {save_path}")
    plt.close()


def plot_cross_validation_results(cv_results: Dict[str, Any], save_path: Path) -> None:
    """クロスバリデーション結果をプロットする"""
    if not cv_results:
        print("クロスバリデーション結果が見つかりません")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Cross-Validation Results', fontsize=16)

    # Fold別F1スコア
    fold_scores = cv_results.get('fold_scores', [])
    if fold_scores:
        ax1.bar(range(1, len(fold_scores) + 1), fold_scores, alpha=0.7, color='blue')
        ax1.axhline(y=cv_results.get('mean_f1', 0), color='red', linestyle='--', 
                   label=f'Mean: {cv_results.get("mean_f1", 0):.4f}')
        ax1.set_xlabel('Fold')
        ax1.set_ylabel('F1 Score')
        ax1.set_title('F1 Score by Fold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # Fold別CMIスコア
    fold_cmi_scores = cv_results.get('fold_cmi_scores', [])
    if fold_cmi_scores:
        ax2.bar(range(1, len(fold_cmi_scores) + 1), fold_cmi_scores, alpha=0.7, color='green')
        ax2.axhline(y=cv_results.get('mean_cmi', 0), color='red', linestyle='--', 
                   label=f'Mean: {cv_results.get("mean_cmi", 0):.4f}')
        ax2.set_xlabel('Fold')
        ax2.set_ylabel('CMI Score')
        ax2.set_title('CMI Score by Fold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # F1スコア統計情報
    mean_f1 = cv_results.get('mean_f1', 0)
    std_f1 = cv_results.get('std_f1', 0)
    
    ax3.text(0.1, 0.8, f'Mean F1: {mean_f1:.4f}', fontsize=12, transform=ax3.transAxes)
    ax3.text(0.1, 0.7, f'Std F1: {std_f1:.4f}', fontsize=12, transform=ax3.transAxes)
    if fold_scores:
        ax3.text(0.1, 0.6, f'Min F1: {min(fold_scores):.4f}', fontsize=12, transform=ax3.transAxes)
        ax3.text(0.1, 0.5, f'Max F1: {max(fold_scores):.4f}', fontsize=12, transform=ax3.transAxes)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_title('F1 Score Statistics')
    ax3.axis('off')

    # CMIスコア統計情報
    mean_cmi = cv_results.get('mean_cmi', 0)
    std_cmi = cv_results.get('std_cmi', 0)
    
    ax4.text(0.1, 0.8, f'Mean CMI: {mean_cmi:.4f}', fontsize=12, transform=ax4.transAxes)
    ax4.text(0.1, 0.7, f'Std CMI: {std_cmi:.4f}', fontsize=12, transform=ax4.transAxes)
    if fold_cmi_scores:
        ax4.text(0.1, 0.6, f'Min CMI: {min(fold_cmi_scores):.4f}', fontsize=12, transform=ax4.transAxes)
        ax4.text(0.1, 0.5, f'Max CMI: {max(fold_cmi_scores):.4f}', fontsize=12, transform=ax4.transAxes)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_title('CMI Score Statistics')
    ax4.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"クロスバリデーション結果保存: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='学習結果の可視化')
    parser.add_argument('--experiment-name', type=str, required=True,
                       help='実験名（例: 20250717_preproc_train_v30）')
    parser.add_argument('--trainer-name', type=str, default='multimodal_v30',
                       help='トレーナー名（デフォルト: multimodal_v30）')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='出力ディレクトリ（デフォルト: 元の結果ディレクトリ）')
    
    args = parser.parse_args()
    
    # パス設定
    base_dir = Path("output/experiments") / args.trainer_name
    result_dir = base_dir / "results"
    
    if not result_dir.exists():
        print(f"❌ 結果ディレクトリが見つかりません: {result_dir}")
        print("学習を先に実行してください")
        sys.exit(1)
    
    # 出力ディレクトリ設定
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = result_dir
    
    print(f"📊 学習結果の可視化開始")
    print(f"実験名: {args.experiment_name}")
    print(f"トレーナー: {args.trainer_name}")
    print(f"結果ディレクトリ: {result_dir}")
    print(f"出力ディレクトリ: {output_dir}")
    print("=" * 50)
    
    # 学習履歴読み込み
    print("📖 学習履歴を読み込み中...")
    histories = load_training_histories(result_dir)
    
    # クロスバリデーション結果読み込み
    print("📊 クロスバリデーション結果を読み込み中...")
    cv_results = load_cv_results(result_dir)
    
    # 学習曲線プロット
    print("📈 学習曲線を作成中...")
    plot_training_curves(histories, output_dir / "training_curves.png")
    
    # クロスバリデーション結果プロット
    print("📊 クロスバリデーション結果を作成中...")
    plot_cross_validation_results(cv_results, output_dir / "cv_results.png")
    
    print("✅ 可視化完了！")
    print(f"生成されたファイル:")
    print(f"  - {output_dir / 'training_curves.png'}")
    print(f"  - {output_dir / 'cv_results.png'}")


if __name__ == "__main__":
    main() 