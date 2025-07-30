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
import os
import sys

# プロジェクトルートディレクトリをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.pipeline import Preprocessor
from src.utils.logging_utils import setup_logging
from src.utils.preprocessing import augment_handedness_flip
import logging

logger = logging.getLogger(__name__)


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
        
        # シーケンス単位のセンサーデータを抽出（メモリ効率化）
        logger.info("Extracting sequences...")
        sequences = []
        demographics = []
        labels = []
        
        # メモリ効率化のため、グループ化して処理
        grouped = df_proc.groupby(['subject', 'sequence_id'])
        total_groups = len(grouped)
        logger.info(f"Processing {total_groups} sequences...")
        
        for i, ((subject, sequence_id), seq_data) in enumerate(grouped):
            if i % 1000 == 0:
                logger.info(f"Processing sequence {i}/{total_groups}")
            
            # センサーデータを抽出（時系列）
            sensor_data = seq_data[sensor_cols].values
            
            # 人口統計データ（シーケンス単位で平均）
            demo_data = seq_data[demographics_cols].mean().values
            
            # ラベル
            label = seq_data['gesture'].iloc[0] if 'gesture' in seq_data.columns else -1
            
            sequences.append(sensor_data)
            demographics.append(demo_data)
            labels.append(label)
        
        # センサーデータの正規化（シーケンス単位、メモリ効率化）
        logger.info("Processing sensor data...")
        X_sensor_clean = []
        all_sensor_data_chunks = []
        
        for i, seq in enumerate(sequences):
            if i % 1000 == 0:
                logger.info(f"Processing sensor data {i}/{len(sequences)}")
            
            # 欠損値処理
            seq_clean = np.nan_to_num(seq, nan=0.0)
            # 外れ値処理（シーケンス単位）
            seq_clean = self._handle_outliers_in_sensor_data(seq_clean)
            X_sensor_clean.append(seq_clean)
            
            # メモリ効率化のため、チャンク単位で正規化パラメータを計算
            if len(all_sensor_data_chunks) < 1000:  # 1000シーケンスごとにチャンク
                all_sensor_data_chunks.append(seq_clean)
            else:
                # チャンクを結合して正規化パラメータを更新
                chunk_data = np.vstack(all_sensor_data_chunks)
                if not hasattr(self, '_partial_sensor_scaler'):
                    from sklearn.preprocessing import StandardScaler
                    self._partial_sensor_scaler = StandardScaler()
                    self._partial_sensor_scaler.partial_fit(chunk_data)
                else:
                    self._partial_sensor_scaler.partial_fit(chunk_data)
                all_sensor_data_chunks = [seq_clean]  # リセット
        
        # 最後のチャンクも処理
        if all_sensor_data_chunks:
            chunk_data = np.vstack(all_sensor_data_chunks)
            if hasattr(self, '_partial_sensor_scaler'):
                self._partial_sensor_scaler.partial_fit(chunk_data)
                self.sensor_scaler = self._partial_sensor_scaler
            else:
                self.sensor_scaler.fit(chunk_data)
        else:
            # 部分的な正規化ができない場合は全データで学習
            all_sensor_data = np.vstack(X_sensor_clean)
            self.sensor_scaler.fit(all_sensor_data)
        
        # 人口統計データの正規化パラメータを学習
        X_demo_clean = np.nan_to_num(np.array(demographics), nan=0.0)
        self.demo_scaler.fit(X_demo_clean)
        
        # feature_engineering.pyの関数を使用して特徴量を抽出
        from src.utils.feature_engineering import (
            compute_basic_statistics,
            compute_peak_features,
            compute_fft_band_energy,
            compute_wavelet_features,
            compute_persistence_image_features_batch
        )
        
        # 設定を取得
        pp_config = self.config.get("preprocessing", {})
        sampling_rate = pp_config.get("sampling_rate", 50.0)
        fft_bands = pp_config.get("fft_bands", [])
        use_wavelet = pp_config.get("use_wavelet_features", False)
        use_tda = pp_config.get("use_tda_features", False)
        
        # シーケンスごとに個別に特徴量を計算（可変長対応）
        tabular_features = []
        
        for i, seq in enumerate(X_sensor_clean):
            if i % 1000 == 0:
                logger.info(f"Processing features for sequence {i}/{len(X_sensor_clean)}")
            
            # シーケンスを3次元配列に変換（バッチ×時間×特徴量）
            seq_3d = seq[np.newaxis, :, :]  # (1, T, F)
            
            # 基本統計量
            stats_features = compute_basic_statistics(seq_3d)[0]  # バッチ次元を除去
            
            # ピーク特徴量
            peak_features = compute_peak_features(seq_3d)[0]  # バッチ次元を除去
            
            # FFT特徴量
            fft_features = compute_fft_band_energy(
                seq_3d, 
                fs=sampling_rate, 
                bands=fft_bands
            )[0]  # バッチ次元を除去
            
            # 特徴量を結合
            seq_features = np.hstack([stats_features, peak_features, fft_features])
            
            # ウェーブレット特徴量（オプション）
            if use_wavelet:
                wavelet = pp_config.get("wavelet", "db4")
                wavelet_level = pp_config.get("wavelet_level", 3)
                wave_features = compute_wavelet_features(
                    seq_3d,
                    wavelet=wavelet,
                    level=wavelet_level,
                )[0]  # バッチ次元を除去
                seq_features = np.hstack([seq_features, wave_features])
            
            # TDA特徴量（オプション）
            if use_tda:
                tda_dimension = pp_config.get("tda_dimension", 1)
                tda_bins = pp_config.get("tda_bins", 8)
                tda_sigma = pp_config.get("tda_sigma", 0.1)
                tda_features = compute_persistence_image_features_batch(
                    seq_3d,
                    dimension=tda_dimension,
                    n_bins=tda_bins,
                    sigma=tda_sigma,
                )[0]  # バッチ次元を除去
                seq_features = np.hstack([seq_features, tda_features])
            
            tabular_features.append(seq_features)
        
        # 特徴量を配列に変換
        tabular_features = np.array(tabular_features)
        
        # 表形式特徴量の正規化パラメータを学習
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
        
        # feature_engineering.pyの関数を使用して特徴量を抽出
        from src.utils.feature_engineering import (
            compute_basic_statistics,
            compute_peak_features,
            compute_fft_band_energy,
            compute_wavelet_features,
            compute_persistence_image_features_batch
        )
        
        # 設定を取得
        pp_config = self.config.get("preprocessing", {})
        sampling_rate = pp_config.get("sampling_rate", 50.0)
        fft_bands = pp_config.get("fft_bands", [])
        use_wavelet = pp_config.get("use_wavelet_features", False)
        use_tda = pp_config.get("use_tda_features", False)
        
        # シーケンスごとに個別に特徴量を計算（可変長対応）
        tabular_features = []
        
        for i, seq in enumerate(X_sensor_clean):
            if i % 1000 == 0:
                logger.info(f"Processing features for sequence {i}/{len(X_sensor_clean)}")
            
            # シーケンスを3次元配列に変換（バッチ×時間×特徴量）
            seq_3d = seq[np.newaxis, :, :]  # (1, T, F)
            
            # 基本統計量
            stats_features = compute_basic_statistics(seq_3d)[0]  # バッチ次元を除去
            
            # ピーク特徴量
            peak_features = compute_peak_features(seq_3d)[0]  # バッチ次元を除去
            
            # FFT特徴量
            fft_features = compute_fft_band_energy(
                seq_3d, 
                fs=sampling_rate, 
                bands=fft_bands
            )[0]  # バッチ次元を除去
            
            # 特徴量を結合
            seq_features = np.hstack([stats_features, peak_features, fft_features])
            
            # ウェーブレット特徴量（オプション）
            if use_wavelet:
                wavelet = pp_config.get("wavelet", "db4")
                wavelet_level = pp_config.get("wavelet_level", 3)
                wave_features = compute_wavelet_features(
                    seq_3d,
                    wavelet=wavelet,
                    level=wavelet_level,
                )[0]  # バッチ次元を除去
                seq_features = np.hstack([seq_features, wave_features])
            
            # TDA特徴量（オプション）
            if use_tda:
                tda_dimension = pp_config.get("tda_dimension", 1)
                tda_bins = pp_config.get("tda_bins", 8)
                tda_sigma = pp_config.get("tda_sigma", 0.1)
                tda_features = compute_persistence_image_features_batch(
                    seq_3d,
                    dimension=tda_dimension,
                    n_bins=tda_bins,
                    sigma=tda_sigma,
                )[0]  # バッチ次元を除去
                seq_features = np.hstack([seq_features, tda_features])
            
            tabular_features.append(seq_features)
        
        # 特徴量を配列に変換
        tabular_features = np.array(tabular_features)
        
        # 表形式特徴量の正規化
        tab_clean = np.nan_to_num(tabular_features, nan=0.0)
        
        # クリップ処理（fit時と同じ閾値）
        if hasattr(self, "tab_clip_low") and hasattr(self, "tab_clip_high"):
            tab_clean = np.clip(tab_clean, self.tab_clip_low, self.tab_clip_high)
        
        tab_normalized = self.tab_scaler.transform(tab_clean)
        
        # ToF特徴量（シーケンス単位）- 3Dボクセル形式で生成（メモリ効率化版）
        logger.info("Processing ToF features (3D voxel format) - memory efficient...")
        
        # 設定からToFの形状を取得
        pp_config = self.config.get("preprocessing", {})
        tof_depth = pp_config.get("tof_depth", 5)
        tof_height = pp_config.get("tof_height", 8)
        tof_width = pp_config.get("tof_width", 8)
        padding_value = pp_config.get("padding_value", -1.0)
        
        # 最大シーケンス長を計算
        max_seq_length = max(info['length'] for info in sequence_info)
        
        # メモリ制御のため最大長を制限（オプション）
        max_length_limit = pp_config.get("max_sequence_length_limit", None)
        if max_length_limit and max_seq_length > max_length_limit:
            logger.warning(f"最大シーケンス長 {max_seq_length} を {max_length_limit} に制限します（メモリ制御）")
            max_seq_length = max_length_limit
        
        logger.info(f"最大シーケンス長: {max_seq_length}")
        
        # メモリ効率化：チャンク処理とオンザフライ保存
        chunk_size = 1000  # 1000シーケンスごとに処理
        tof_chunks = []
        
        for chunk_start in range(0, len(sequence_info), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(sequence_info))
            logger.info(f"Processing ToF chunk {chunk_start//chunk_size + 1}/{(len(sequence_info)-1)//chunk_size + 1} (sequences {chunk_start}-{chunk_end})")
            
            chunk_tof_features = []
            
            for i in range(chunk_start, chunk_end):
                if i % 500 == 0:
                    logger.info(f"  Processing ToF sequence {i}/{len(sequence_info)}")
                
                info = sequence_info[i]
                subject = info['subject']
                sequence_id = info['sequence_id']
                seq_length = info['length']
                
                # 該当シーケンスのToFデータを抽出
                seq_tof_data = df_proc[
                    (df_proc['subject'] == subject) & 
                    (df_proc['sequence_id'] == sequence_id)
                ]
                
                # ToFデータを3Dボクセル形式に変換（float16でメモリ節約）
                tof_cols = [col for col in seq_tof_data.columns if col.startswith('tof_')]
                if tof_cols:
                    tof_values = seq_tof_data[tof_cols].values
                    
                    # 3Dボクセル配列を初期化（最大長でパディング、float16使用）
                    tof_3d = np.full((max_seq_length, tof_depth, tof_height, tof_width), 
                                    padding_value, dtype=np.float16)
                    
                    # 実際のデータを配置
                    for t in range(min(seq_length, tof_values.shape[0])):  # 各時間ステップ
                        for d in range(1, tof_depth + 1):  # 各深度
                            for h in range(tof_height):  # 各高さ
                                for w in range(tof_width):  # 各幅
                                    # ToFセンサーのインデックスを計算
                                    sensor_idx = (d - 1) * tof_height * tof_width + h * tof_width + w
                                    if sensor_idx < tof_values.shape[1]:
                                        value = tof_values[t, sensor_idx]
                                        # 有効値のみ使用
                                        if value > 0:
                                            tof_3d[t, d-1, h, w] = value
                                        else:
                                            tof_3d[t, d-1, h, w] = padding_value
                                    else:
                                        tof_3d[t, d-1, h, w] = padding_value
                    
                    chunk_tof_features.append(tof_3d)
                else:
                    # ToFデータがない場合、パディング値で埋めた3D形状を生成
                    tof_3d = np.full((max_seq_length, tof_depth, tof_height, tof_width), 
                                    padding_value, dtype=np.float16)
                    chunk_tof_features.append(tof_3d)
            
            # チャンクを配列化して保存
            chunk_array = np.array(chunk_tof_features, dtype=np.float16)
            tof_chunks.append(chunk_array)
            
            # メモリ解放
            del chunk_tof_features
            del chunk_array
        
        # チャンクを結合（メモリ効率的）
        logger.info("Combining ToF chunks...")
        tof_features = np.concatenate(tof_chunks, axis=0)
        del tof_chunks  # メモリ解放
        
        logger.info(f"ToF features shape: {tof_features.shape}")
        
        # ToF特徴量の正規化（3D形状、-1パディング対応、メモリ効率化）
        logger.info("Normalizing ToF features...")
        tof_clean = np.nan_to_num(tof_features, nan=padding_value)
        
        if hasattr(self, "tof_scaler"):
            # パディング値以外のデータのみで正規化（チャンク処理）
            valid_mask = tof_clean != padding_value
            valid_data = tof_clean[valid_mask]
            
            if len(valid_data) > 0:
                # 有効データのみでスケーラーを更新
                self.tof_scaler.partial_fit(valid_data.reshape(-1, 1))
            
            # 全データを正規化（パディング値はそのまま）
            original_shape = tof_clean.shape
            tof_flat = tof_clean.reshape(original_shape[0], -1)
            tof_normalized_flat = self.tof_scaler.transform(tof_flat.reshape(-1, 1)).reshape(tof_flat.shape)
            tof_normalized = tof_normalized_flat.reshape(original_shape)
            
            # パディング値を元に戻す
            tof_normalized[tof_clean == padding_value] = padding_value
        else:
            # 初回実行時はスケーラーを作成
            from sklearn.preprocessing import StandardScaler
            self.tof_scaler = StandardScaler()
            
            # パディング値以外のデータのみで学習
            valid_mask = tof_clean != padding_value
            valid_data = tof_clean[valid_mask]
            
            if len(valid_data) > 0:
                self.tof_scaler.fit(valid_data.reshape(-1, 1))
            
            # 全データを正規化
            original_shape = tof_clean.shape
            tof_flat = tof_clean.reshape(original_shape[0], -1)
            tof_normalized_flat = self.tof_scaler.transform(tof_flat.reshape(-1, 1)).reshape(tof_flat.shape)
            tof_normalized = tof_normalized_flat.reshape(original_shape)
            
            # パディング値を元に戻す
            tof_normalized[tof_clean == padding_value] = padding_value
        
        logger.info(f"ToF normalized shape: {tof_normalized.shape}")
        logger.info(f"パディング値: {padding_value}")
        
        # メモリ解放
        del tof_features
        del tof_clean
        
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
            "tof_voxels": tof_normalized,  # 3Dボクセル形式
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
    
    logger = logging.getLogger(__name__)
    logger.info(f"Saving {len(data)} data files...")
    
    # Save data files
    for i, (key, value) in enumerate(data.items()):
        file = out_dir / f"{prefix}_{key}.pkl"
        logger.info(f"Saving {key} ({i+1}/{len(data)})...")
        with open(file, "wb") as f:
            pickle.dump(value, f)
        logger.info("Saved %s", file)
    
    # Save metadata
    logger.info("Creating metadata...")
    metadata = create_metadata_no_windows(data, config, prefix)
    metadata_file = out_dir / f"{prefix}_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info("Saved metadata to %s", metadata_file)


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