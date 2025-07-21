#!/usr/bin/env python3
"""
前処理結果の妥当性チェックスクリプト
使用例:
    python scripts/validate_preprocessing.py --experiment preprocess_v2_ws64
    python scripts/validate_preprocessing.py --experiment preprocess_v2_ws128 --output-dir validation_reports
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Any, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PreprocessingValidator:
    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.issues = []
        self.warnings = []
        
    def check_data_consistency(self) -> bool:
        """データの整合性をチェック"""
        logger.info("データ整合性チェック開始...")
        
        # サンプル数の整合性チェック
        sample_counts = {}
        for key, value in self.data.items():
            if hasattr(value, 'shape'):
                sample_counts[key] = value.shape[0]
            elif isinstance(value, (list, np.ndarray)):
                sample_counts[key] = len(value)
        
        # 最初のサンプル数を基準とする
        if sample_counts:
            base_count = list(sample_counts.values())[0]
            for key, count in sample_counts.items():
                if count != base_count:
                    issue = f"サンプル数不一致: {key}={count}, 基準={base_count}"
                    self.issues.append(issue)
                    logger.error(issue)
        
        # ラベルとデータの整合性
        if 'labels' in self.data and 'windows' in self.data:
            label_count = len(self.data['labels'])
            window_count = self.data['windows'].shape[0]
            if label_count != window_count:
                issue = f"ラベル数({label_count})とウィンドウ数({window_count})が一致しません"
                self.issues.append(issue)
                logger.error(issue)
        
        return len(self.issues) == 0
    
    def check_missing_values(self) -> bool:
        """欠損値チェック"""
        logger.info("欠損値チェック開始...")
        
        for key, value in self.data.items():
            if hasattr(value, 'shape'):
                if np.isnan(value).any():
                    missing_count = np.isnan(value).sum()
                    issue = f"{key}に欠損値が{missing_count}個存在します"
                    self.issues.append(issue)
                    logger.error(issue)
                else:
                    logger.info(f"{key}: 欠損値なし")
        
        return len([i for i in self.issues if '欠損値' in i]) == 0
    
    def check_outliers(self, threshold: float = 3.0) -> bool:
        """異常値チェック（Z-score法）"""
        logger.info("異常値チェック開始...")
        
        for key, value in self.data.items():
            if hasattr(value, 'shape') and value.size > 0:
                # 数値データのみチェック
                if np.issubdtype(value.dtype, np.number):
                    z_scores = np.abs((value - value.mean()) / value.std())
                    outlier_count = (z_scores > threshold).sum()
                    
                    if outlier_count > 0:
                        outlier_ratio = outlier_count / value.size * 100
                        warning = f"{key}: 異常値{outlier_count}個 ({outlier_ratio:.2f}%)"
                        self.warnings.append(warning)
                        logger.warning(warning)
                    else:
                        logger.info(f"{key}: 異常値なし")
        
        return True
    
    def check_data_ranges(self) -> bool:
        """データ範囲チェック"""
        logger.info("データ範囲チェック開始...")
        
        for key, value in self.data.items():
            if hasattr(value, 'shape') and value.size > 0:
                if np.issubdtype(value.dtype, np.number):
                    min_val = value.min()
                    max_val = value.max()
                    mean_val = value.mean()
                    std_val = value.std()
                    
                    logger.info(f"{key}: 範囲[{min_val:.3f}, {max_val:.3f}], 平均{mean_val:.3f}, 標準偏差{std_val:.3f}")
                    
                    # 極端な値のチェック
                    if abs(min_val) > 1e6 or abs(max_val) > 1e6:
                        warning = f"{key}: 極端に大きな値が存在します (範囲: [{min_val:.3e}, {max_val:.3e}])"
                        self.warnings.append(warning)
                        logger.warning(warning)
        
        return True
    
    def check_label_distribution(self) -> bool:
        """ラベル分布チェック"""
        logger.info("ラベル分布チェック開始...")
        
        if 'labels' not in self.data:
            logger.warning("ラベルデータが見つかりません")
            return True
        
        labels = self.data['labels']
        unique, counts = np.unique(labels, return_counts=True)
        
        logger.info(f"ラベル数: {len(unique)}")
        for label, count in zip(unique, counts):
            logger.info(f"  ラベル {label}: {count} サンプル")
        
        # クラス不均衡チェック
        total_samples = len(labels)
        min_samples = counts.min()
        max_samples = counts.max()
        
        imbalance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')
        logger.info(f"クラス不均衡比: {imbalance_ratio:.2f}")
        
        if imbalance_ratio > 10:
            warning = f"クラス不均衡が大きいです (比率: {imbalance_ratio:.2f})"
            self.warnings.append(warning)
            logger.warning(warning)
        
        return True
    
    def check_sensor_data_quality(self) -> bool:
        """センサーデータ品質チェック"""
        logger.info("センサーデータ品質チェック開始...")
        
        if 'windows' not in self.data:
            logger.warning("センサーデータが見つかりません")
            return True
        
        windows = self.data['windows']
        
        # ゼロ分散チェック
        for i in range(windows.shape[2]):  # チャンネルごと
            channel_data = windows[:, :, i]
            if channel_data.std() == 0:
                issue = f"チャンネル{i}の分散がゼロです"
                self.issues.append(issue)
                logger.error(issue)
        
        # 定数値チェック
        for i in range(windows.shape[2]):
            channel_data = windows[:, :, i]
            if channel_data.std() < 1e-6:
                warning = f"チャンネル{i}の分散が非常に小さいです ({channel_data.std():.2e})"
                self.warnings.append(warning)
                logger.warning(warning)
        
        return len([i for i in self.issues if 'チャンネル' in i]) == 0
    
    def check_tof_data_quality(self) -> bool:
        """ToFデータ品質チェック"""
        logger.info("ToFデータ品質チェック開始...")
        
        if 'tof_windows' not in self.data:
            logger.warning("ToFデータが見つかりません")
            return True
        
        tof_windows = self.data['tof_windows']
        
        # 負の値チェック（ToFは通常正の値）
        if (tof_windows < 0).any():
            negative_count = (tof_windows < 0).sum()
            warning = f"ToFデータに負の値が{negative_count}個存在します"
            self.warnings.append(warning)
            logger.warning(warning)
        
        # 極端に大きな値チェック
        if (tof_windows > 1000).any():
            large_count = (tof_windows > 1000).sum()
            warning = f"ToFデータに極端に大きな値(>1000)が{large_count}個存在します"
            self.warnings.append(warning)
            logger.warning(warning)
        
        return True
    
    def generate_report(self, output_dir: Path) -> None:
        """検証レポート生成"""
        report = []
        report.append("# 前処理妥当性検証レポート")
        report.append("")
        
        # サマリー
        total_issues = len(self.issues)
        total_warnings = len(self.warnings)
        
        if total_issues == 0 and total_warnings == 0:
            report.append("## ✅ 検証結果: 問題なし")
        elif total_issues == 0:
            report.append(f"## ⚠️ 検証結果: 警告{total_warnings}件")
        else:
            report.append(f"## ❌ 検証結果: 問題{total_issues}件, 警告{total_warnings}件")
        
        report.append("")
        
        # 問題の詳細
        if self.issues:
            report.append("## 問題")
            for issue in self.issues:
                report.append(f"- ❌ {issue}")
            report.append("")
        
        # 警告の詳細
        if self.warnings:
            report.append("## 警告")
            for warning in self.warnings:
                report.append(f"- ⚠️ {warning}")
            report.append("")
        
        # データ統計
        report.append("## データ統計")
        for key, value in self.data.items():
            if hasattr(value, 'shape'):
                report.append(f"- **{key}**: {value.shape}")
                if np.issubdtype(value.dtype, np.number):
                    report.append(f"  - 範囲: [{value.min():.3f}, {value.max():.3f}]")
                    report.append(f"  - 平均: {value.mean():.3f}")
                    report.append(f"  - 標準偏差: {value.std():.3f}")
            elif isinstance(value, (list, np.ndarray)):
                report.append(f"- **{key}**: {np.array(value).shape}")
        
        # ファイルに保存
        with open(output_dir / 'validation_report.md', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        logger.info(f"検証レポートを保存: {output_dir / 'validation_report.md'}")
        
        # 結果をコンソールにも表示
        print("\n" + "="*50)
        print("前処理妥当性検証結果")
        print("="*50)
        if total_issues == 0 and total_warnings == 0:
            print("✅ すべての検証項目が正常です")
        else:
            if total_issues > 0:
                print(f"❌ 問題: {total_issues}件")
                for issue in self.issues:
                    print(f"  - {issue}")
            if total_warnings > 0:
                print(f"⚠️ 警告: {total_warnings}件")
                for warning in self.warnings:
                    print(f"  - {warning}")
        print("="*50)

def load_preprocessed_data(experiment_dir: Path) -> Dict[str, Any]:
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

def main():
    parser = argparse.ArgumentParser(description='前処理結果の妥当性チェック')
    parser.add_argument('--experiment', required=True, help='実験名（例: preprocess_v2_ws64）')
    parser.add_argument('--output-dir', default='validation_reports', help='出力ディレクトリ')
    parser.add_argument('--data-dir', default='output/experiments', help='データディレクトリ')
    parser.add_argument('--outlier-threshold', type=float, default=3.0, help='異常値検出の閾値（Z-score）')
    
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
    
    # 検証実行
    validator = PreprocessingValidator(data)
    
    logger.info("前処理妥当性検証開始...")
    
    # 各種チェックを実行
    validator.check_data_consistency()
    validator.check_missing_values()
    validator.check_outliers(args.outlier_threshold)
    validator.check_data_ranges()
    validator.check_label_distribution()
    validator.check_sensor_data_quality()
    validator.check_tof_data_quality()
    
    # レポート生成
    validator.generate_report(output_dir)
    
    logger.info("検証完了")

if __name__ == '__main__':
    main() 