#!/usr/bin/env python3
"""
前処理結果の可視化スクリプト
使用例:
    python scripts/visualize_preprocessing.py --experiment preprocess_v2_ws64
    python scripts/visualize_preprocessing.py --experiment preprocess_v2_ws128 --output-dir plots
"""

import argparse
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_preprocessed_data(experiment_dir):
    """前処理済みデータを読み込み"""
    data = {}
    files = {
        'windows': 'train_windows.pkl',
        'demographics': 'train_demographics.pkl', 
        'tabular': 'train_tabular.pkl',
        'tof_voxel': 'train_tof_voxel.pkl',
        'tof_windows': 'train_tof_windows.pkl',
        'labels': 'train_labels.pkl',
        'info': 'train_info.pkl'
    }
    
    for key, filename in files.items():
        filepath = experiment_dir / 'preprocessed' / filename
        if filepath.exists():
            with open(filepath, 'rb') as f:
                data[key] = pickle.load(f)
            logger.info(f"Loaded {key}: {type(data[key])} - {getattr(data[key], 'shape', 'N/A')}")
        else:
            logger.warning(f"File not found: {filepath}")
    
    return data

def plot_data_overview(data, output_dir):
    """データ概要の可視化"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('前処理データ概要', fontsize=16)
    
    # 1. ラベル分布
    if 'labels' in data:
        labels = data['labels']
        unique, counts = np.unique(labels, return_counts=True)
        axes[0, 0].bar(unique, counts)
        axes[0, 0].set_title('ラベル分布')
        axes[0, 0].set_xlabel('ラベルID')
        axes[0, 0].set_ylabel('サンプル数')
    
    # 2. センサーデータの統計
    if 'windows' in data:
        windows = data['windows']
        axes[0, 1].hist(windows.flatten(), bins=50, alpha=0.7)
        axes[0, 1].set_title('センサーデータ分布')
        axes[0, 1].set_xlabel('値')
        axes[0, 1].set_ylabel('頻度')
        
        # 統計情報をテキストで表示
        stats_text = f"平均: {windows.mean():.3f}\n標準偏差: {windows.std():.3f}\n最小値: {windows.min():.3f}\n最大値: {windows.max():.3f}"
        axes[0, 1].text(0.02, 0.98, stats_text, transform=axes[0, 1].transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. タブラーデータの統計
    if 'tabular' in data:
        tabular = data['tabular']
        if hasattr(tabular, 'shape'):
            axes[0, 2].hist(tabular.flatten(), bins=50, alpha=0.7)
            axes[0, 2].set_title('タブラーデータ分布')
            axes[0, 2].set_xlabel('値')
            axes[0, 2].set_ylabel('頻度')
    
    # 4. ToFデータの統計
    if 'tof_windows' in data:
        tof_windows = data['tof_windows']
        axes[1, 0].hist(tof_windows.flatten(), bins=50, alpha=0.7)
        axes[1, 0].set_title('ToFデータ分布')
        axes[1, 0].set_xlabel('値')
        axes[1, 0].set_ylabel('頻度')
    
    # 5. デモグラフィックデータ
    if 'demographics' in data:
        demo = data['demographics']
        if hasattr(demo, 'columns'):
            # 数値列のみを選択
            numeric_cols = demo.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                demo[numeric_cols].boxplot(ax=axes[1, 1])
                axes[1, 1].set_title('デモグラフィックデータ')
                axes[1, 1].tick_params(axis='x', rotation=45)
    
    # 6. データ形状のサマリー
    shapes_text = "データ形状:\n"
    for key, value in data.items():
        if hasattr(value, 'shape'):
            shapes_text += f"{key}: {value.shape}\n"
        elif isinstance(value, (list, np.ndarray)):
            shapes_text += f"{key}: {np.array(value).shape}\n"
    
    axes[1, 2].text(0.1, 0.5, shapes_text, transform=axes[1, 2].transAxes, 
                   verticalalignment='center', fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    axes[1, 2].set_title('データ形状サマリー')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'data_overview.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_sensor_analysis(data, output_dir):
    """センサーデータの詳細分析"""
    if 'windows' not in data:
        return
    
    windows = data['windows']
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('センサーデータ詳細分析', fontsize=16)
    
    # 1. 時系列プロット（最初の数サンプル）
    n_samples = min(5, windows.shape[0])
    for i in range(n_samples):
        axes[0, 0].plot(windows[i, :, 0], alpha=0.7, label=f'サンプル{i+1}')
    axes[0, 0].set_title('センサーデータ時系列（最初のチャンネル）')
    axes[0, 0].set_xlabel('時間ステップ')
    axes[0, 0].set_ylabel('値')
    axes[0, 0].legend()
    
    # 2. チャンネル間の相関
    if windows.shape[2] > 1:
        # 各サンプルの平均を取ってチャンネル間相関を計算
        sample_means = windows.mean(axis=1)  # (n_samples, n_channels)
        corr_matrix = np.corrcoef(sample_means.T)
        im = axes[0, 1].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        axes[0, 1].set_title('チャンネル間相関')
        plt.colorbar(im, ax=axes[0, 1])
    
    # 3. ラベル別のセンサーデータ分布
    if 'labels' in data:
        labels = data['labels']
        unique_labels = np.unique(labels)
        n_labels = min(5, len(unique_labels))  # 最初の5つのラベル
        
        for i, label in enumerate(unique_labels[:n_labels]):
            mask = labels == label
            if mask.sum() > 0:
                label_data = windows[mask].flatten()
                axes[1, 0].hist(label_data, bins=30, alpha=0.5, label=f'ラベル{label}')
        
        axes[1, 0].set_title('ラベル別センサーデータ分布')
        axes[1, 0].set_xlabel('値')
        axes[1, 0].set_ylabel('頻度')
        axes[1, 0].legend()
    
    # 4. 統計情報
    stats_text = f"サンプル数: {windows.shape[0]}\n"
    stats_text += f"時間ステップ: {windows.shape[1]}\n"
    stats_text += f"チャンネル数: {windows.shape[2]}\n"
    stats_text += f"平均値: {windows.mean():.3f}\n"
    stats_text += f"標準偏差: {windows.std():.3f}\n"
    stats_text += f"最小値: {windows.min():.3f}\n"
    stats_text += f"最大値: {windows.max():.3f}"
    
    axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                   verticalalignment='center', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    axes[1, 1].set_title('統計情報')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sensor_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_tof_analysis(data, output_dir):
    """ToFデータの詳細分析"""
    if 'tof_windows' not in data:
        return
    
    tof_windows = data['tof_windows']
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('ToFデータ詳細分析', fontsize=16)
    
    # 1. ToFデータの時系列（最初のサンプル）
    if tof_windows.shape[0] > 0:
        sample_tof = tof_windows[0]  # (time_steps, channels, height, width)
        # 中央のスライスを表示
        mid_time = sample_tof.shape[0] // 2
        mid_channel = sample_tof.shape[1] // 2
        
        im = axes[0, 0].imshow(sample_tof[mid_time, mid_channel], cmap='viridis')
        axes[0, 0].set_title(f'ToFデータ（時間{mid_time}, チャンネル{mid_channel}）')
        plt.colorbar(im, ax=axes[0, 0])
    
    # 2. ToFデータの統計
    axes[0, 1].hist(tof_windows.flatten(), bins=50, alpha=0.7)
    axes[0, 1].set_title('ToFデータ分布')
    axes[0, 1].set_xlabel('値')
    axes[0, 1].set_ylabel('頻度')
    
    # 3. ラベル別ToF分布
    if 'labels' in data:
        labels = data['labels']
        unique_labels = np.unique(labels)
        n_labels = min(3, len(unique_labels))
        
        for i, label in enumerate(unique_labels[:n_labels]):
            mask = labels == label
            if mask.sum() > 0:
                label_tof = tof_windows[mask].flatten()
                axes[1, 0].hist(label_tof, bins=30, alpha=0.5, label=f'ラベル{label}')
        
        axes[1, 0].set_title('ラベル別ToF分布')
        axes[1, 0].set_xlabel('値')
        axes[1, 0].set_ylabel('頻度')
        axes[1, 0].legend()
    
    # 4. ToF統計情報
    stats_text = f"サンプル数: {tof_windows.shape[0]}\n"
    stats_text += f"時間ステップ: {tof_windows.shape[1]}\n"
    stats_text += f"チャンネル数: {tof_windows.shape[2]}\n"
    stats_text += f"高さ: {tof_windows.shape[3]}\n"
    stats_text += f"幅: {tof_windows.shape[4]}\n"
    stats_text += f"平均値: {tof_windows.mean():.3f}\n"
    stats_text += f"標準偏差: {tof_windows.std():.3f}"
    
    axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes,
                   verticalalignment='center', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    axes[1, 1].set_title('ToF統計情報')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'tof_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_summary_report(data, output_dir):
    """サマリーレポートの生成"""
    report = []
    report.append("# 前処理結果サマリーレポート")
    report.append("")
    
    # データ概要
    report.append("## データ概要")
    for key, value in data.items():
        if hasattr(value, 'shape'):
            report.append(f"- **{key}**: {value.shape}")
        elif isinstance(value, (list, np.ndarray)):
            report.append(f"- **{key}**: {np.array(value).shape}")
        else:
            report.append(f"- **{key}**: {type(value)}")
    
    report.append("")
    
    # ラベル情報
    if 'labels' in data:
        labels = data['labels']
        unique, counts = np.unique(labels, return_counts=True)
        report.append("## ラベル分布")
        for label, count in zip(unique, counts):
            report.append(f"- ラベル {label}: {count} サンプル")
        report.append("")
    
    # 統計情報
    report.append("## 統計情報")
    if 'windows' in data:
        windows = data['windows']
        report.append(f"- センサーデータ平均: {windows.mean():.3f}")
        report.append(f"- センサーデータ標準偏差: {windows.std():.3f}")
        report.append(f"- センサーデータ範囲: [{windows.min():.3f}, {windows.max():.3f}]")
    
    if 'tof_windows' in data:
        tof_windows = data['tof_windows']
        report.append(f"- ToFデータ平均: {tof_windows.mean():.3f}")
        report.append(f"- ToFデータ標準偏差: {tof_windows.std():.3f}")
        report.append(f"- ToFデータ範囲: [{tof_windows.min():.3f}, {tof_windows.max():.3f}]")
    
    # ファイルに保存
    with open(output_dir / 'summary_report.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    logger.info(f"サマリーレポートを保存: {output_dir / 'summary_report.md'}")

def main():
    parser = argparse.ArgumentParser(description='前処理結果の可視化')
    parser.add_argument('--experiment', required=True, help='実験名（例: preprocess_v2_ws64）')
    parser.add_argument('--output-dir', default='plots', help='出力ディレクトリ')
    parser.add_argument('--data-dir', default='output/experiments', help='データディレクトリ')
    
    args = parser.parse_args()
    
    # 出力ディレクトリ作成
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # データディレクトリ
    data_dir = Path(args.data_dir) / args.experiment
    
    if not data_dir.exists():
        logger.error(f"データディレクトリが見つかりません: {data_dir}")
        return
    
    logger.info(f"データを読み込み中: {data_dir}")
    data = load_preprocessed_data(data_dir)
    
    if not data:
        logger.error("データの読み込みに失敗しました")
        return
    
    logger.info("可視化を開始...")
    
    # 各種可視化を実行
    plot_data_overview(data, output_dir)
    plot_sensor_analysis(data, output_dir)
    plot_tof_analysis(data, output_dir)
    generate_summary_report(data, output_dir)
    
    logger.info(f"可視化完了: {output_dir}")

if __name__ == '__main__':
    main() 