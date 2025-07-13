# -*- coding: utf-8 -*-
"""Preprocessing pipeline utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config_utils import load_config
from .preprocessing import (
    create_sliding_windows_with_demographics,
    tof_to_voxel_tensor,
)
from .feature_engineering import (
    compute_basic_statistics,
    compute_peak_features,
    compute_fft_band_energy,
)


class WindowTensorBuilder:
    """Generate IMU window tensors with demographics."""

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or load_config()
        pp = self.config.get("preprocessing", {})
        self.window_size = pp.get("window_size", 128)
        self.stride = pp.get("stride", 64)
        self.min_len = pp.get("min_sequence_length", 10)
        self.padding_value = pp.get("padding_value", 0.0)
        self.sensor_cols = (
            self.config.get("sensor_acc_cols", [])
            + self.config.get("sensor_rot_cols", [])
            + self.config.get("sensor_thm_cols", [])
        )
        self.demographics_cols = self.config.get("demographics_cols", [])

    def build(self, df: pd.DataFrame):
        return create_sliding_windows_with_demographics(
            df,
            window_size=self.window_size,
            stride=self.stride,
            sensor_cols=self.sensor_cols,
            demographics_cols=self.demographics_cols,
            min_sequence_length=self.min_len,
            padding_value=self.padding_value,
        )


class TabularFeatureBuilder:
    """Build tabular features from IMU windows."""

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or load_config()
        pp = self.config.get("preprocessing", {})
        self.sampling_rate = pp.get("sampling_rate", 50.0)
        self.fft_bands = pp.get("fft_bands", [])
        self.window_builder = WindowTensorBuilder(self.config)

    def build(self, df: pd.DataFrame):
        X_sensor, X_demo, y, info = self.window_builder.build(df)
        stats = compute_basic_statistics(X_sensor)
        peaks = compute_peak_features(X_sensor)
        fft = compute_fft_band_energy(
            X_sensor, fs=self.sampling_rate, bands=self.fft_bands
        )
        features = np.hstack([stats, peaks, fft, X_demo])
        return features, y, info


class ToFVoxelBuilder:
    """Convert ToF pixel columns to voxel tensor."""

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or load_config()
        pp = self.config.get("preprocessing", {})
        self.fill_value = pp.get("padding_value", 0.0)
        depth = pp.get("tof_depth", 5)
        h = pp.get("tof_height", 8)
        w = pp.get("tof_width", 8)
        self.tof_cols = [f"tof_{d}_v{i}" for d in range(1, depth + 1) for i in range(h * w)]

    def build(self, df: pd.DataFrame):
        sub = df[self.tof_cols] if all(c in df.columns for c in self.tof_cols) else df
        return tof_to_voxel_tensor(sub, fill_value=self.fill_value)


class Preprocessor:
    """Manage preprocessing statistics and transformations."""

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or load_config()
        self.win_builder = WindowTensorBuilder(self.config)
        self.tab_builder = TabularFeatureBuilder(self.config)
        self.tof_builder = ToFVoxelBuilder(self.config)

        self.sensor_scaler = StandardScaler()
        self.demo_scaler = StandardScaler()
        self.tab_scaler = StandardScaler()
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        X_sensor, X_demo, _, _ = self.win_builder.build(df)
        self.sensor_scaler.fit(
            np.nan_to_num(X_sensor.reshape(-1, X_sensor.shape[-1]), nan=0.0)
        )
        self.demo_scaler.fit(X_demo)
        tab, _, _ = self.tab_builder.build(df)
        self.tab_scaler.fit(tab)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> dict:
        if not self._fitted:
            raise RuntimeError("Preprocessor is not fitted")
        X_sensor, X_demo, y, info = self.win_builder.build(df)
        X_sensor = self.sensor_scaler.transform(
            X_sensor.reshape(-1, X_sensor.shape[-1])
        ).reshape(X_sensor.shape)
        X_demo = self.demo_scaler.transform(X_demo)
        tab, _, _ = self.tab_builder.build(df)
        tab = self.tab_scaler.transform(tab)
        tof_tensor = self.tof_builder.build(df)
        return {
            "windows": X_sensor,
            "demographics": X_demo,
            "tabular": tab,
            "tof_voxel": tof_tensor,
            "labels": y,
            "info": info,
        }

    def fit_transform(self, df: pd.DataFrame) -> dict:
        self.fit(df)
        return self.transform(df)
