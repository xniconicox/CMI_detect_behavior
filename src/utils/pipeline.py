# -*- coding: utf-8 -*-
"""Preprocessing pipeline utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import json
import pickle
from sklearn.preprocessing import StandardScaler

from .io_utils import df_md5

from pathlib import Path
import yaml

from .preprocessing import (
    create_sliding_windows_with_demographics,
    handedness_correction_v2,
    clean_sensor_missing_values,
    clean_missing_sensor_data_parallel_disk,
)
from .config_utils import load_config, get_cache_dir
from .tof import tof_to_voxel_tensor
from .feature_engineering import (
    compute_basic_statistics,
    compute_peak_features,
    compute_fft_band_energy,
    compute_wavelet_features,
    compute_persistence_image_features_batch,
)
from .imu import add_world_acc_features


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
        self.cache_dir = get_cache_dir(self.config)
        self.cache_file = self.cache_dir / "windows.pkl"
        self.meta_file = self.cache_dir / "windows_meta.json"
        self.cache_dir.mkdir(exist_ok=True)

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
        self.use_wavelet = pp.get("use_wavelet_features", False)
        self.use_tda = pp.get("use_tda_features", False)
        self.wavelet = pp.get("wavelet", "db4")
        self.wavelet_level = pp.get("wavelet_level", 3)
        self.tda_dimension = pp.get("tda_dimension", 1)
        self.tda_bins = pp.get("tda_bins", 20)
        self.tda_sigma = pp.get("tda_sigma", 0.1)
        self.window_builder = WindowTensorBuilder(self.config)
        self.cache_dir = get_cache_dir(self.config)
        self.cache_file = self.cache_dir / "tabular_features.pkl"
        self.meta_file = self.cache_dir / "tabular_meta.json"
        self.cache_dir.mkdir(exist_ok=True)

    def build(self, df: pd.DataFrame, windows=None, use_cache: bool = True):
        md5 = df_md5(df)
        if use_cache and self.cache_file.exists() and self.meta_file.exists():
            meta = json.loads(self.meta_file.read_text())
            if meta.get("md5") == md5:
                with open(self.cache_file, "rb") as f:
                    return pickle.load(f)

        if windows is None:
            X_sensor, X_demo, y, info = self.window_builder.build(df, use_cache=use_cache)
        else:
            X_sensor, X_demo, y, info = windows
        stats = compute_basic_statistics(X_sensor)
        peaks = compute_peak_features(X_sensor)
        fft = compute_fft_band_energy(
            X_sensor, fs=self.sampling_rate, bands=self.fft_bands
        )
        feats = [stats, peaks, fft]
        if self.use_wavelet:
            wave = compute_wavelet_features(
                X_sensor, wavelet=self.wavelet, level=self.wavelet_level
            )
            feats.append(wave)
        if self.use_tda:
            tda = compute_persistence_image_features_batch(
                X_sensor,
                dimension=self.tda_dimension,
                n_bins=self.tda_bins,
                sigma=self.tda_sigma,
            )
            feats.append(tda)
        features = np.hstack(feats + [X_demo])
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
        self.cache_dir = get_cache_dir(self.config)
        self.cache_file = self.cache_dir / "tof_voxel.pkl"
        self.meta_file = self.cache_dir / "tof_meta.json"
        self.cache_dir.mkdir(exist_ok=True)

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

    def __init__(
        self,
        config: dict | None = None,
        *,
        use_handedness: bool = True,
        use_basic_cleaning: bool = True,
        use_interp_cleaning: bool = True,
        use_world_acc: bool | None = None,
    ) -> None:
        self.config = config or load_config()
        pp = self.config.get("preprocessing", {})
        if use_world_acc is None:
            use_world_acc = pp.get("use_world_acc", False)
        self.win_builder = WindowTensorBuilder(self.config)
        self.tab_builder = TabularFeatureBuilder(self.config)
        self.tof_builder = ToFVoxelBuilder(self.config)

        self.sensor_scaler = StandardScaler()
        self.demo_scaler = StandardScaler()
        self.tab_scaler = StandardScaler()

        self.use_handedness = use_handedness
        self.use_basic_cleaning = use_basic_cleaning
        self.use_interp_cleaning = use_interp_cleaning
        self.use_world_acc = use_world_acc

        pp = self.config.get("preprocessing", {})
        depth = pp.get("tof_depth", 5)
        h = pp.get("tof_height", 8)
        w = pp.get("tof_width", 8)
        self.sensor_type_groups = {
            "Accelerometer": list(self.config.get("sensor_acc_cols", [])),
            "Rotation": self.config.get("sensor_rot_cols", []),
            "Thermal": self.config.get("sensor_thm_cols", []),
            "ToF_Sensor": [f"tof_{d}_v{i}" for d in range(1, depth + 1) for i in range(h * w)],
        }
        if self.use_world_acc:
            self.sensor_type_groups["Accelerometer"].extend(
                [f"acc_w_{ax}" for ax in "xyz"]
                + [f"lin_acc_{ax}" for ax in "xyz"]
            )

        interp_path = Path(__file__).resolve().parents[2] / "config" / "interp_params.yaml"
        if interp_path.exists():
            with open(interp_path, "r") as f:
                self.interp_params = yaml.safe_load(f)
        else:
            self.interp_params = None

        self._fitted = False

    def _maybe_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        processed = df.copy()
        if self.use_handedness:
            processed = handedness_correction_v2(processed)
        if self.use_basic_cleaning:
            processed = clean_sensor_missing_values(
                processed, self.sensor_type_groups
            )
        if self.use_interp_cleaning:
            keep = self.config.get("demographics_cols", []) + ["gesture"]
            processed = clean_missing_sensor_data_parallel_disk(
                processed,
                sensor_type_groups=self.sensor_type_groups,
                interp_params=self.interp_params,
                keep_cols=keep,
            )
        if self.use_world_acc:
            processed = add_world_acc_features(processed)
        return processed

    def fit(self, df: pd.DataFrame, use_cache: bool = True) -> "Preprocessor":
        df_proc = self._maybe_clean(df)
        windows = self.win_builder.build(df_proc, use_cache=use_cache)
        X_sensor, X_demo, _, _ = windows
        self.sensor_scaler.fit(
            np.nan_to_num(X_sensor.reshape(-1, X_sensor.shape[-1]), nan=0.0)
        )
        self.demo_scaler.fit(X_demo)
        tab, _, _ = self.tab_builder.build(df_proc, windows=windows, use_cache=use_cache)
        self.tab_scaler.fit(tab)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame, use_cache: bool = True) -> dict:
        if not self._fitted:
            raise RuntimeError("Preprocessor is not fitted")
        df_proc = self._maybe_clean(df)
        windows = self.win_builder.build(df_proc, use_cache=use_cache)
        X_sensor, X_demo, y, info = windows
        X_sensor = self.sensor_scaler.transform(
            X_sensor.reshape(-1, X_sensor.shape[-1])
        ).reshape(X_sensor.shape)
        X_demo = self.demo_scaler.transform(X_demo)
        tab, _, _ = self.tab_builder.build(df_proc, windows=windows, use_cache=use_cache)
        tab = self.tab_scaler.transform(tab)
        tof_tensor = self.tof_builder.build(df_proc, use_cache=use_cache)
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

    def save(self, path: Path) -> None:
        """Save scaler objects and settings to a pickle file."""
        data = {
            "config": self.config,
            "use_handedness": self.use_handedness,
            "use_basic_cleaning": self.use_basic_cleaning,
            "use_interp_cleaning": self.use_interp_cleaning,
            "use_world_acc": self.use_world_acc,
            "sensor_scaler": self.sensor_scaler,
            "demo_scaler": self.demo_scaler,
            "tab_scaler": self.tab_scaler,
            "_fitted": self._fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    @classmethod
    def load(cls, path: Path) -> "Preprocessor":
        """Load scalers and settings from a pickle file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(
            data.get("config"),
            use_handedness=data.get("use_handedness", True),
            use_basic_cleaning=data.get("use_basic_cleaning", True),
            use_interp_cleaning=data.get("use_interp_cleaning", True),
            use_world_acc=data.get("use_world_acc"),
        )
        obj.sensor_scaler = data["sensor_scaler"]
        obj.demo_scaler = data["demo_scaler"]
        obj.tab_scaler = data["tab_scaler"]
        obj._fitted = data.get("_fitted", False)
        return obj
