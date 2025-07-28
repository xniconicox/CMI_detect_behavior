#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Preprocess CMI dataset without window segmentation for anomaly detection.

Usage:
    python scripts/run_preprocessing_no_windows.py --experiment-name exp1 [--config CONFIG]
                                                  [--use-cache] [--mode train|predict]

This script processes sequences as a whole without window segmentation,
extracting tabular features for anomaly detection.
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Any
import pickle
import logging
import json
import numpy as np

import pandas as pd
import yaml

from src.utils.pipeline import Preprocessor
from src.utils.logging_utils import setup_logging


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class NoWindowPreprocessor(Preprocessor):
    """Preprocessor that skips window segmentation for sequence-level processing."""
    
    def __init__(self, config: dict | None = None, **kwargs):
        super().__init__(config, **kwargs)
        # 窓分割をスキップするため、WindowTensorBuilderを無効化
        self.win_builder = None
        self.tof_win_builder = None
    
    def fit(
        self,
        df: pd.DataFrame,
        use_cache: bool = True,
        *,
        autoencoder_model=None,
    ) -> "NoWindowPreprocessor":
        """Fit preprocessor without window segmentation."""
        logger.info("Fitting NoWindowPreprocessor without window segmentation")
        logger.info("Dataframe shape: %s", df.shape)
        
        # データクリーニング
        df_proc = self._maybe_clean(df)
        if self.use_handedness_augmentation:
            df_proc = augment_handedness_flip(df_proc)
        
        # シーケンス単位で特徴量抽出
        sensor_cols = (
            self.config.get("sensor_acc_cols", [])
            + self.config.get("sensor_rot_cols", [])
            + self.config.get("sensor_thm_cols", [])
        )
        if self.use_world_acc:
            world_acc_cols = [f"acc_w_{ax}" for ax in "xyz"] + [f"lin_acc_{ax}" for ax in "xyz"]
            sensor_cols.extend(world_acc_cols)
        
        demographics_cols = self.config.get("demographics_cols", [])
        
        # シーケンス単位のセンサーデータを抽出
        sequences = []
        demographics = []
        labels = []
        
        for subject in df_proc['subject'].unique():
            subject_data = df_proc[df_proc['subject'] == subject]
            
            for sequence_id in subject_data['sequence_id'].unique():
                seq_data = subject_data[subject_data['sequence_id'] == sequence_id]
                
                # センサーデータを抽出（時系列）
                sensor_data = seq_data[sensor_cols].values
                
                # 人口統計データ（シーケンス単位で平均）
                demo_data = seq_data[demographics_cols].mean().values
                
                # ラベル
                label = seq_data['gesture'].iloc[0] if 'gesture' in seq_data.columns else -1
                
                sequences.append(sensor_data)
                demographics.append(demo_data)
                labels.append(label)
        
        # センサーデータの正規化（シーケンス単位）
        X_sensor_clean = []
        for seq in sequences:
            # 欠損値処理
            seq_clean = np.nan_to_num(seq, nan=0.0)
            # 外れ値処理（シーケンス単位）
            seq_clean = self._handle_outliers_in_sensor_data(seq_clean)
            X_sensor_clean.append(seq_clean)
        
        # センサーデータの正規化パラメータを学習
        all_sensor_data = np.vstack(X_sensor_clean)
        self.sensor_scaler.fit(all_sensor_data)
        
        # 人口統計データの正規化パラメータを学習
        X_demo_clean = np.nan_to_num(np.array(demographics), nan=0.0)
        self.demo_scaler.fit(X_demo_clean)
        
        # 表形式特徴量の抽出（シーケンス単位）
        tabular_features = []
        for i, seq in enumerate(sequences):
            # 基本統計量
            seq_features = []
            for col_idx, col_name in enumerate(sensor_cols):
                values = seq[:, col_idx]
                seq_features.extend([
                    np.mean(values),  # mean
                    np.std(values),   # std
                    np.max(values) - np.min(values),  # range
                    np.sqrt(np.mean(values**2)),  # rms
                    np.sum(values**2),  # energy
                ])
            
            # ピーク特徴量
            for col_idx in range(len(sensor_cols)):
                values = seq[:, col_idx]
                # ピーク検出（簡易版）
                peaks = 0
                for j in range(1, len(values) - 1):
                    if values[j] > values[j-1] and values[j] > values[j+1]:
                        peaks += 1
                seq_features.append(peaks)
            
            # FFT特徴量（オプション）
            pp_config = self.config.get("preprocessing", {})
            fft_bands = pp_config.get("fft_bands", [])
            if fft_bands:
                for col_idx in range(len(sensor_cols)):
                    values = seq[:, col_idx]
                    # FFT計算
                    fft_vals = np.fft.fft(values)
                    fft_power = np.abs(fft_vals)**2
                    
                    for low, high in fft_bands:
                        # 周波数帯域のエネルギー
                        freq_bins = np.fft.fftfreq(len(values))
                        mask = (freq_bins >= low) & (freq_bins <= high)
                        energy = np.sum(fft_power[mask])
                        seq_features.append(energy)
            
            tabular_features.append(seq_features)
        
        # 表形式特徴量の正規化パラメータを学習
        tabular_features = np.array(tabular_features)
        tab_clean = np.nan_to_num(tabular_features, nan=0.0)
        
        # クリップ処理の閾値を設定
        self.tab_clip_low = np.percentile(tab_clean, 0.5)
        self.tab_clip_high = np.percentile(tab_clean, 99.5)
        
        self.tab_scaler.fit(tab_clean)
        
        # ラベルエンコーダーの学習
        labels = np.array(labels)
        if len(np.unique(labels)) > 1 and not (len(np.unique(labels)) == 1 and labels[0] == -1):
            from sklearn.preprocessing import LabelEncoder
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(labels)
            logger.info(f"ラベルエンコーダー作成: {list(self.label_encoder.classes_)}")
        
        self._fitted = True
        logger.info("NoWindowPreprocessor fitting completed")
        
        return self
    
    def transform(
        self,
        df: pd.DataFrame,
        use_cache: bool = True,
        *,
        autoencoder_model=None,
    ) -> dict:
        """Transform dataframe without window segmentation.
        
        Returns sequence-level features instead of window-level features.
        """
        if not self._fitted:
            raise RuntimeError("Preprocessor is not fitted")
        
        logger.info("Transforming dataframe without window segmentation")
        logger.info("Dataframe shape: %s", df.shape)
        
        # データクリーニング
        df_proc = self._maybe_clean(df)
        if self.use_handedness_augmentation:
            df_proc = augment_handedness_flip(df_proc)
        
        # シーケンス単位で特徴量抽出
        sensor_cols = (
            self.config.get("sensor_acc_cols", [])
            + self.config.get("sensor_rot_cols", [])
            + self.config.get("sensor_thm_cols", [])
        )
        if self.use_world_acc:
            world_acc_cols = [f"acc_w_{ax}" for ax in "xyz"] + [f"lin_acc_{ax}" for ax in "xyz"]
            sensor_cols.extend(world_acc_cols)
        
        demographics_cols = self.config.get("demographics_cols", [])
        
        # シーケンス単位のセンサーデータを抽出
        sequences = []
        demographics = []
        labels = []
        sequence_info = []
        
        for subject in df_proc['subject'].unique():
            subject_data = df_proc[df_proc['subject'] == subject]
            
            for sequence_id in subject_data['sequence_id'].unique():
                seq_data = subject_data[subject_data['sequence_id'] == sequence_id]
                
                # センサーデータを抽出（時系列）
                sensor_data = seq_data[sensor_cols].values
                
                # 人口統計データ（シーケンス単位で平均）
                demo_data = seq_data[demographics_cols].mean().values
                
                # ラベル
                label = seq_data['gesture'].iloc[0] if 'gesture' in seq_data.columns else -1
                
                # シーケンス情報
                info = {
                    'subject': subject,
                    'sequence_id': sequence_id,
                    'length': len(seq_data)
                }
                
                sequences.append(sensor_data)
                demographics.append(demo_data)
                labels.append(label)
                sequence_info.append(info)
        
        # センサーデータの正規化（シーケンス単位）
        X_sensor_clean = []
        for seq in sequences:
            # 欠損値処理
            seq_clean = np.nan_to_num(seq, nan=0.0)
            # 外れ値処理（シーケンス単位）
            seq_clean = self._handle_outliers_in_sensor_data(seq_clean)
            X_sensor_clean.append(seq_clean)
        
        # 正規化（シーケンス単位で適用）
        X_sensor_normalized = []
        for seq in X_sensor_clean:
            seq_norm = self.sensor_scaler.transform(seq)
            X_sensor_normalized.append(seq_norm)
        
        # 人口統計データの正規化
        X_demo_clean = np.nan_to_num(np.array(demographics), nan=0.0)
        X_demo_normalized = self.demo_scaler.transform(X_demo_clean)
        
        # 表形式特徴量の抽出（シーケンス単位）
        tabular_features = []
        for i, seq in enumerate(sequences):
            # 基本統計量
            seq_features = []
            for col_idx, col_name in enumerate(sensor_cols):
                values = seq[:, col_idx]
                seq_features.extend([
                    np.mean(values),  # mean
                    np.std(values),   # std
                    np.max(values) - np.min(values),  # range
                    np.sqrt(np.mean(values**2)),  # rms
                    np.sum(values**2),  # energy
                ])
            
            # ピーク特徴量
            for col_idx in range(len(sensor_cols)):
                values = seq[:, col_idx]
                # ピーク検出（簡易版）
                peaks = 0
                for j in range(1, len(values) - 1):
                    if values[j] > values[j-1] and values[j] > values[j+1]:
                        peaks += 1
                seq_features.append(peaks)
            
            # FFT特徴量（オプション）
            pp_config = self.config.get("preprocessing", {})
            fft_bands = pp_config.get("fft_bands", [])
            if fft_bands:
                for col_idx in range(len(sensor_cols)):
                    values = seq[:, col_idx]
                    # FFT計算
                    fft_vals = np.fft.fft(values)
                    fft_power = np.abs(fft_vals)**2
                    
                    for low, high in fft_bands:
                        # 周波数帯域のエネルギー
                        freq_bins = np.fft.fftfreq(len(values))
                        mask = (freq_bins >= low) & (freq_bins <= high)
                        energy = np.sum(fft_power[mask])
                        seq_features.append(energy)
            
            tabular_features.append(seq_features)
        
        # 表形式特徴量の正規化
        tabular_features = np.array(tabular_features)
        tab_clean = np.nan_to_num(tabular_features, nan=0.0)
        
        # クリップ処理（fit時と同じ閾値）
        if hasattr(self, "tab_clip_low") and hasattr(self, "tab_clip_high"):
            tab_clean = np.clip(tab_clean, self.tab_clip_low, self.tab_clip_high)
        
        tab_normalized = self.tab_scaler.transform(tab_clean)
        
        # ToF特徴量（シーケンス単位）
        tof_features = []
        for i, info in enumerate(sequence_info):
            subject = info['subject']
            sequence_id = info['sequence_id']
            
            # 該当シーケンスのToFデータを抽出
            seq_tof_data = df_proc[
                (df_proc['subject'] == subject) & 
                (df_proc['sequence_id'] == sequence_id)
            ]
            
            # ToF特徴量の計算（簡易版）
            tof_cols = [col for col in seq_tof_data.columns if col.startswith('tof_')]
            if tof_cols:
                tof_values = seq_tof_data[tof_cols].values
                # ToFの基本統計量
                tof_stats = []
                for col_idx in range(tof_values.shape[1]):
                    values = tof_values[:, col_idx]
                    valid_values = values[values > 0]  # 有効値のみ
                    if len(valid_values) > 0:
                        tof_stats.extend([
                            np.mean(valid_values),
                            np.std(valid_values),
                            np.max(valid_values),
                            len(valid_values)  # 有効値の数
                        ])
                    else:
                        tof_stats.extend([0, 0, 0, 0])
                tof_features.append(tof_stats)
            else:
                # ToFデータがない場合
                tof_features.append([0] * 4)  # ダミー値
        
        tof_features = np.array(tof_features)
        tof_normalized = tof_features  # 簡易版のため正規化はスキップ
        
        # ラベルエンコーディング
        labels = np.array(labels)
        if hasattr(self, "label_encoder") and len(np.unique(labels)) > 1:
            if len(np.unique(labels)) == 1 and labels[0] == -1:
                logger.info("testデータのため、ラベルエンコーディングをスキップ")
                y_encoded = labels
            else:
                y_encoded = self.label_encoder.transform(labels)
                logger.info(f"ラベルエンコーディング適用: {np.unique(y_encoded)}")
        else:
            y_encoded = labels
        
        logger.info(
            "Output shapes: sequences=%d, demographics=%s, tabular=%s, tof=%s",
            len(X_sensor_normalized),
            X_demo_normalized.shape,
            tab_normalized.shape,
            tof_normalized.shape,
        )
        
        return {
            "sequences": X_sensor_normalized,  # シーケンス単位のセンサーデータ
            "demographics": X_demo_normalized,
            "tabular": tab_normalized,
            "tof_features": tof_normalized,
            "labels": y_encoded,
            "info": sequence_info,
        }


def generate_feature_names_no_windows(config: dict) -> dict[str, list[str]]:
    """Generate feature names for sequence-level processing."""
    feature_names = {}
    
    # Demographics feature names
    demographics_cols = config.get("demographics_cols", [])
    feature_names["demographics"] = demographics_cols
    
    # Sensor feature names
    sensor_cols = (
        config.get("sensor_acc_cols", [])
        + config.get("sensor_rot_cols", [])
        + config.get("sensor_thm_cols", [])
    )
    if config.get("preprocessing", {}).get("use_world_acc", False):
        world_acc_cols = [f"acc_w_{ax}" for ax in "xyz"] + [f"lin_acc_{ax}" for ax in "xyz"]
        sensor_cols.extend(world_acc_cols)
    feature_names["sensor"] = sensor_cols
    
    # Tabular feature names (sequence-level)
    tabular_features = []
    
    # Basic statistics features (mean, std, range, rms, energy for each sensor)
    for col in sensor_cols:
        tabular_features.extend([
            f"{col}_mean", f"{col}_std", f"{col}_range", 
            f"{col}_rms", f"{col}_energy"
        ])
    
    # Peak features (one per sensor)
    for col in sensor_cols:
        tabular_features.append(f"{col}_peaks")
    
    # FFT band energy features
    fft_bands = config.get("preprocessing", {}).get("fft_bands", [])
    for i, (low, high) in enumerate(fft_bands):
        for col in sensor_cols:
            tabular_features.append(f"{col}_fft_{low}_{high}Hz")
    
    feature_names["tabular"] = tabular_features
    
    # ToF feature names (simplified)
    tof_features = []
    pp_config = config.get("preprocessing", {})
    tof_depth = pp_config.get("tof_depth", 5)
    tof_height = pp_config.get("tof_height", 8)
    tof_width = pp_config.get("tof_width", 8)
    
    for d in range(1, tof_depth + 1):
        for h in range(tof_height):
            for w in range(tof_width):
                tof_features.extend([
                    f"tof_{d}_v{h*tof_width+w}_mean",
                    f"tof_{d}_v{h*tof_width+w}_std",
                    f"tof_{d}_v{h*tof_width+w}_max",
                    f"tof_{d}_v{h*tof_width+w}_count"
                ])
    
    feature_names["tof_features"] = tof_features
    
    return feature_names


def create_metadata_no_windows(data: dict, config: dict, prefix: str) -> dict[str, Any]:
    """Create metadata for sequence-level preprocessed data."""
    metadata = {
        "prefix": prefix,
        "shapes": {},
        "feature_names": generate_feature_names_no_windows(config),
        "config": config,
        "processing_type": "sequence_level_no_windows"
    }
    
    # Extract shapes
    for key, value in data.items():
        if isinstance(value, list):
            metadata["shapes"][key] = f"list of {len(value)} sequences"
        elif isinstance(value, np.ndarray):
            metadata["shapes"][key] = list(value.shape)
        else:
            metadata["shapes"][key] = str(type(value))
    
    return metadata


def save_dict_no_windows(data: dict, prefix: str, out_dir: Path, config: dict) -> None:
    """Save sequence-level preprocessed data with metadata."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save data files
    for key, value in data.items():
        file = out_dir / f"{prefix}_{key}.pkl"
        with open(file, "wb") as f:
            pickle.dump(value, f)
        logging.getLogger(__name__).info("Saved %s", file)
    
    # Save metadata
    metadata = create_metadata_no_windows(data, config, prefix)
    metadata_file = out_dir / f"{prefix}_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logging.getLogger(__name__).info("Saved metadata to %s", metadata_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run preprocessing pipeline without window segmentation")
    parser.add_argument(
        "--config",
        default="config/config_v2.yaml",
        help="YAML config path",
    )
    parser.add_argument("--experiment-name", required=True, help="experiment name")
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="use cached intermediate files if available",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "predict"],
        default="train",
        help="train: fit and transform train/test, predict: transform only",
    )
    parser.add_argument(
        "--log-file",
        help=(
            "log file path (default: <output_dir>/<experiment>/preprocessed/preprocess.log)"
        ),
    )
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    data_dir = Path(config.get("data_dir", "data"))
    out_root = Path(config.get("output_dir", "output/experiments"))
    pre_dir = out_root / args.experiment_name / "preprocessed"
    pre_dir.mkdir(parents=True, exist_ok=True)

    log_path = Path(args.log_file) if args.log_file else pre_dir / "preprocess.log"
    setup_logging(log_path)
    logger = logging.getLogger(__name__)

    if args.mode == "train":
        logger.info("Loading train.csv and test.csv")
        train_df = pd.read_csv(data_dir / "train.csv")
        test_df = pd.read_csv(data_dir / "test.csv")
        logger.info("Fitting NoWindowPreprocessor")
        
        # Load demographics data and merge
        train_demo = pd.read_csv(data_dir / "train_demographics.csv")
        test_demo = pd.read_csv(data_dir / "test_demographics.csv")
        
        # Merge demographics data
        train_df = train_df.merge(train_demo, on="subject", how="left")
        test_df = test_df.merge(test_demo, on="subject", how="left")
        
        pp = NoWindowPreprocessor(config)
        pp.fit(train_df, use_cache=args.use_cache)
        train_data = pp.transform(train_df, use_cache=args.use_cache)
        # test_data = pp.transform(test_df, use_cache=args.use_cache)

        # save results with metadata
        save_dict_no_windows(train_data, "train", pre_dir, config)
        # save_dict_no_windows(test_data, "test", pre_dir, config)
        logger.info("Saved train/test outputs to %s", pre_dir)
        pp.save(pre_dir / "preprocessor.pkl")
        logger.info("Saved preprocessor to %s", pre_dir / "preprocessor.pkl")

    else:  # predict
        # Load main data
        logger.info("Loading test.csv for prediction")
        df = pd.read_csv(data_dir / "test.csv")
        logger.info("Loading preprocessor from %s", pre_dir / "preprocessor.pkl")
        
        # Load demographics data and merge
        test_demo = pd.read_csv(data_dir / "test_demographics.csv")
        df = df.merge(test_demo, on="subject", how="left")
        
        pp = NoWindowPreprocessor.load(pre_dir / "preprocessor.pkl")
        data = pp.transform(df, use_cache=args.use_cache)
        save_dict_no_windows(data, "predict", pre_dir, config)
        logger.info("Saved prediction outputs to %s", pre_dir)


if __name__ == "__main__":
    main() 