# -*- coding: utf-8 -*-
"""Preprocessing pipeline utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import json
import pickle
import hashlib
from sklearn.preprocessing import StandardScaler

from .io_utils import CACHE_DIR, df_md5

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
        self.cache_file = CACHE_DIR / "windows.pkl"
        self.meta_file = CACHE_DIR / "windows_meta.json"
        CACHE_DIR.mkdir(exist_ok=True)

    def build(self, df: pd.DataFrame, use_cache: bool = True):
        md5 = df_md5(df)
        if use_cache and self.cache_file.exists() and self.meta_file.exists():
            meta = json.loads(self.meta_file.read_text())
            if meta.get("md5") == md5:
                with open(self.cache_file, "rb") as f:
                    return pickle.load(f)

        result = create_sliding_windows_with_demographics(
            df,
            window_size=self.window_size,
            stride=self.stride,
            sensor_cols=self.sensor_cols,
            demographics_cols=self.demographics_cols,
            min_sequence_length=self.min_len,
            padding_value=self.padding_value,
        )
        if use_cache:
            with open(self.cache_file, "wb") as f:
                pickle.dump(result, f)
            self.meta_file.write_text(json.dumps({"md5": md5}))
        return result


class TabularFeatureBuilder:
    """Build tabular features from IMU windows."""

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or load_config()
        pp = self.config.get("preprocessing", {})
        self.sampling_rate = pp.get("sampling_rate", 50.0)
        self.fft_bands = pp.get("fft_bands", [])
        self.window_builder = WindowTensorBuilder(self.config)
        self.cache_file = CACHE_DIR / "tabular_features.pkl"
        self.meta_file = CACHE_DIR / "tabular_meta.json"
        CACHE_DIR.mkdir(exist_ok=True)

    def build(self, df: pd.DataFrame, use_cache: bool = True):
        md5 = df_md5(df)
        if (
            use_cache
            and self.cache_file.exists()
            and self.meta_file.exists()
        ):
            meta = json.loads(self.meta_file.read_text())
            if meta.get("md5") == md5:
                with open(self.cache_file, "rb") as f:
                    return pickle.load(f)

        X_sensor, X_demo, y, info = self.window_builder.build(df, use_cache=use_cache)
        stats = compute_basic_statistics(X_sensor)
        peaks = compute_peak_features(X_sensor)
        fft = compute_fft_band_energy(
            X_sensor, fs=self.sampling_rate, bands=self.fft_bands
        )
        features = np.hstack([stats, peaks, fft, X_demo])
        result = (features, y, info)
        if use_cache:
            with open(self.cache_file, "wb") as f:
                pickle.dump(result, f)
            self.meta_file.write_text(json.dumps({"md5": md5}))
        return result


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
        self.cache_file = CACHE_DIR / "tof_voxel.pkl"
        self.meta_file = CACHE_DIR / "tof_meta.json"
        CACHE_DIR.mkdir(exist_ok=True)

    def build(self, df: pd.DataFrame, use_cache: bool = True):
        md5 = df_md5(df)
        if use_cache and self.cache_file.exists() and self.meta_file.exists():
            meta = json.loads(self.meta_file.read_text())
            if meta.get("md5") == md5:
                with open(self.cache_file, "rb") as f:
                    return pickle.load(f)

        sub = df[self.tof_cols] if all(c in df.columns for c in self.tof_cols) else df
        result = tof_to_voxel_tensor(sub, fill_value=self.fill_value)
        if use_cache:
            with open(self.cache_file, "wb") as f:
                pickle.dump(result, f)
            self.meta_file.write_text(json.dumps({"md5": md5}))
        return result


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

    def fit(self, df: pd.DataFrame, use_cache: bool = True) -> "Preprocessor":
        X_sensor, X_demo, _, _ = self.win_builder.build(df, use_cache=use_cache)
        self.sensor_scaler.fit(
            np.nan_to_num(X_sensor.reshape(-1, X_sensor.shape[-1]), nan=0.0)
        )
        self.demo_scaler.fit(X_demo)
        tab, _, _ = self.tab_builder.build(df, use_cache=use_cache)
        self.tab_scaler.fit(tab)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame, use_cache: bool = True) -> dict:
        if not self._fitted:
            raise RuntimeError("Preprocessor is not fitted")
        X_sensor, X_demo, y, info = self.win_builder.build(df, use_cache=use_cache)
        X_sensor = self.sensor_scaler.transform(
            X_sensor.reshape(-1, X_sensor.shape[-1])
        ).reshape(X_sensor.shape)
        X_demo = self.demo_scaler.transform(X_demo)
        tab, _, _ = self.tab_builder.build(df, use_cache=use_cache)
        tab = self.tab_scaler.transform(tab)
        tof_tensor = self.tof_builder.build(df, use_cache=use_cache)
        return {
            "windows": X_sensor,
            "demographics": X_demo,
            "tabular": tab,
            "tof_voxel": tof_tensor,
            "labels": y,
            "info": info,
        }

    def fit_transform(self, df: pd.DataFrame, use_cache: bool = True) -> dict:
        self.fit(df, use_cache=use_cache)
        return self.transform(df, use_cache=use_cache)
