#!/usr/bin/env python3
"""
前処理結果の簡単確認スクリプト
使用例:
    python scripts/quick_check.py --experiment preprocess_v2_ws64
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_check(experiment_dir: Path):
    """前処理結果の簡単確認"""
    print(f"\n{'='*60}")
    print(f"前処理結果確認: {experiment_dir.name}")
    print(f"{'='*60}")
    
    # ファイル一覧
    files = {
        'windows': 'train_windows.pkl',
        'demographics': 'train_demographics.pkl', 
        'tabular': 'train_tabular.pkl',
        'tof_voxel': 'train_tof_voxel.pkl',
        'tof_windows': 'train_tof_windows.pkl',
        'labels': 'train_labels.pkl',
        'info': 'train_info.pkl'
    }
    
    print("\n📁 ファイル確認:")
    for key, filename in files.items():
        filepath = experiment_dir / 'preprocessed' / filename
        if filepath.exists():
            file_size = filepath.stat().st_size / (1024 * 1024)  # MB
            print(f"  ✅ {filename}: {file_size:.1f} MB")
        else:
            print(f"  ❌ {filename}: 見つかりません")
    
    print("\n📊 データ統計:")
    
    # 主要データの読み込みと統計
    for key, filename in files.items():
        filepath = experiment_dir / 'preprocessed' / filename
        if filepath.exists():
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                
                if hasattr(data, 'shape'):
                    print(f"\n  {key}:")
                    print(f"    形状: {data.shape}")
                    if np.issubdtype(data.dtype, np.number):
                        print(f"    データ型: {data.dtype}")
                        print(f"    範囲: [{data.min():.3f}, {data.max():.3f}]")
                        print(f"    平均: {data.mean():.3f}")
                        print(f"    標準偏差: {data.std():.3f}")
                        
                        # 欠損値チェック
                        if np.isnan(data).any():
                            missing_count = np.isnan(data).sum()
                            print(f"    ⚠️ 欠損値: {missing_count}個")
                        else:
                            print(f"    ✅ 欠損値: なし")
                
                elif isinstance(data, (list, np.ndarray)):
                    print(f"\n  {key}:")
                    print(f"    形状: {np.array(data).shape}")
                    if isinstance(data, list) and len(data) > 0:
                        print(f"    要素数: {len(data)}")
                        print(f"    最初の要素: {type(data[0])}")
                
                else:
                    print(f"\n  {key}: {type(data)}")
                    
            except Exception as e:
                print(f"\n  {key}: 読み込みエラー - {e}")
    
    # ラベル分布の詳細確認
    labels_file = experiment_dir / 'preprocessed' / 'train_labels.pkl'
    if labels_file.exists():
        try:
            with open(labels_file, 'rb') as f:
                labels = pickle.load(f)
            
            print(f"\n🏷️ ラベル分布:")
            unique, counts = np.unique(labels, return_counts=True)
            print(f"    総サンプル数: {len(labels)}")
            print(f"    ラベル数: {len(unique)}")
            
            for label, count in zip(unique, counts):
                percentage = count / len(labels) * 100
                print(f"    ラベル {label}: {count} サンプル ({percentage:.1f}%)")
            
            # クラス不均衡チェック
            min_count = counts.min()
            max_count = counts.max()
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            print(f"    クラス不均衡比: {imbalance_ratio:.2f}")
            
            if imbalance_ratio > 10:
                print(f"    ⚠️ クラス不均衡が大きいです")
            else:
                print(f"    ✅ クラス分布は比較的均等です")
                
        except Exception as e:
            print(f"\n🏷️ ラベル読み込みエラー: {e}")
    
    print(f"\n{'='*60}")

def main():
    parser = argparse.ArgumentParser(description='前処理結果の簡単確認')
    parser.add_argument('--experiment', required=True, help='実験名（例: preprocess_v2_ws64）')
    parser.add_argument('--data-dir', default='output/experiments', help='データディレクトリ')
    
    args = parser.parse_args()
    
    # データディレクトリ
    data_dir = Path(args.data_dir) / args.experiment
    
    if not data_dir.exists():
        logger.error(f"データディレクトリが見つかりません: {data_dir}")
        return
    
    quick_check(data_dir)

if __name__ == '__main__':
    main() 