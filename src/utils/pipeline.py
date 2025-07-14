# -*- coding: utf-8 -*-
"""Preprocessing pipeline utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import json
import pickle
import logging
from sklearn.preprocessing import StandardScaler

from .io_utils import df_md5

from pathlib import Path
import yaml

from .preprocessing import (
    create_sliding_windows_with_demographics,
    create_tof_windows_with_info,
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


logger = logging.getLogger(__name__)


def validate_no_nan(array: np.ndarray, name: str) -> None:
    """ArrayにNaNが含まれていないか検証する."""
    if array is None:
        return
    if np.isnan(array).any():
        msg = f"NaN detected in {name}"
        logger.error(msg)
        raise ValueError(msg)


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
        logger.info(
            "Building windows (size=%d, stride=%d, min_len=%d)",
            self.window_size,
            self.stride,
            self.min_len,
        )
        md5 = df_md5(df)
        if use_cache and self.cache_file.exists() and self.meta_file.exists():
            meta = json.loads(self.meta_file.read_text())
            if meta.get("md5") == md5:
                logger.info("Reusing cached windows from %s", self.cache_file)
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
        logger.info(
            "Windows shapes: X_sensor=%s, X_demo=%s, y=%s",
            result[0].shape,
            result[1].shape,
            result[2].shape,
        )
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

    def build(
        self,
        df: pd.DataFrame,
        windows=None,
        use_cache: bool = True,
        *,
        use_wavelet: bool | None = None,
        use_tda: bool | None = None,
    ):
        """Return tabular features for each window.

        Parameters
        ----------
        df : pd.DataFrame
            Raw sensor dataframe.
        windows : tuple or None
            Precomputed window tensors from :class:`WindowTensorBuilder`.
        use_cache : bool, default True
            If True, reuse cached results when available.
        use_wavelet : bool | None, optional
            Override config to compute wavelet features.
        use_tda : bool | None, optional
            Override config to compute TDA features.
        """

        md5 = df_md5(df)
        use_wavelet = self.use_wavelet if use_wavelet is None else use_wavelet
        use_tda = self.use_tda if use_tda is None else use_tda
        logger.info(
            "Building tabular features (wavelet=%s, tda=%s)",
            use_wavelet,
            use_tda,
        )
        if use_cache and self.cache_file.exists() and self.meta_file.exists():
            meta = json.loads(self.meta_file.read_text())
            if meta.get("md5") == md5:
                logger.info("Reusing cached tabular features from %s", self.cache_file)
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

        # decide whether to compute optional features
        feats = [stats, peaks, fft]
        if use_wavelet:
            wave = compute_wavelet_features(
                X_sensor, wavelet=self.wavelet, level=self.wavelet_level
            )
            feats.append(wave)
        if use_tda:
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
        logger.info("Tabular features shape %s", result[0].shape)
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
        logger.info("Building ToF voxel tensor")
        md5 = df_md5(df)
        if use_cache and self.cache_file.exists() and self.meta_file.exists():
            meta = json.loads(self.meta_file.read_text())
            if meta.get("md5") == md5:
                logger.info("Reusing cached ToF voxel from %s", self.cache_file)
                with open(self.cache_file, "rb") as f:
                    return pickle.load(f)

        sub = df[self.tof_cols] if all(c in df.columns for c in self.tof_cols) else df
        result = tof_to_voxel_tensor(sub, fill_value=self.fill_value)
        if use_cache:
            with open(self.cache_file, "wb") as f:
                pickle.dump(result, f)
            self.meta_file.write_text(json.dumps({"md5": md5}))
        logger.info("ToF voxel shape %s", result.shape)
        return result


class ToFWindowBuilder:
    """Create sliding windows from ToF voxel tensor."""

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or load_config()
        pp = self.config.get("preprocessing", {})
        self.window_size = pp.get("window_size", 128)
        self.stride = pp.get("stride", 64)
        self.min_len = pp.get("min_sequence_length", 10)
        self.fill_value = pp.get("padding_value", 0.0)
        depth = pp.get("tof_depth", 5)
        h = pp.get("tof_height", 8)
        w = pp.get("tof_width", 8)
        self.tof_cols = [f"tof_{d}_v{i}" for d in range(1, depth + 1) for i in range(h * w)]
        self.cache_dir = get_cache_dir(self.config)
        self.cache_file = self.cache_dir / "tof_windows.pkl"
        self.meta_file = self.cache_dir / "tof_windows_meta.json"
        self.cache_dir.mkdir(exist_ok=True)

    def build(self, df: pd.DataFrame, use_cache: bool = True):
        logger.info(
            "Building ToF windows (size=%d, stride=%d, min_len=%d)",
            self.window_size,
            self.stride,
            self.min_len,
        )
        md5 = df_md5(df)
        if use_cache and self.cache_file.exists() and self.meta_file.exists():
            meta = json.loads(self.meta_file.read_text())
            if meta.get("md5") == md5:
                logger.info("Reusing cached ToF windows from %s", self.cache_file)
                with open(self.cache_file, "rb") as f:
                    return pickle.load(f)

        result = create_tof_windows_with_info(
            df,
            window_size=self.window_size,
            stride=self.stride,
            tof_cols=self.tof_cols,
            min_sequence_length=self.min_len,
            fill_value=self.fill_value,
        )
        if use_cache:
            with open(self.cache_file, "wb") as f:
                pickle.dump(result, f)
            self.meta_file.write_text(json.dumps({"md5": md5}))
        logger.info("ToF windows shape %s", result[0].shape)
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
        self.tof_win_builder = ToFWindowBuilder(self.config)

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
            base_keep = self.config.get("demographics_cols", [])
            # Only include columns that exist in the dataframe
            keep = [col for col in base_keep + ["gesture"] if col in df.columns]
            processed = clean_missing_sensor_data_parallel_disk(
                processed,
                sensor_type_groups=self.sensor_type_groups,
                interp_params=self.interp_params,
                keep_cols=keep,
            )
        if self.use_world_acc:
            processed = add_world_acc_features(processed)
        return processed

    def _handle_missing_values_by_sensor_type(self, X_sensor: np.ndarray) -> np.ndarray:
        """センサー別に適切な欠損値処理を行う"""
        if X_sensor is None:
            return X_sensor
            
        X_clean = X_sensor.copy()
        
        # センサー別の処理
        sensor_config = self.win_builder.sensor_cols
        acc_cols = self.config.get("sensor_acc_cols", [])
        rot_cols = self.config.get("sensor_rot_cols", [])
        thm_cols = self.config.get("sensor_thm_cols", [])

        acc_indices = []
        rot_indices = []
        thm_indices = []
        
        # Accelerometer: 0で置換（物理的に正当）
        if acc_cols:
            acc_indices = [sensor_config.index(col) for col in acc_cols if col in sensor_config]
            for idx in acc_indices:
                if idx < X_clean.shape[-1]:
                    X_clean[..., idx] = np.nan_to_num(X_clean[..., idx], nan=0.0)
        
        # Rotation: 単位クォータニオンで置換
        if rot_cols:
            rot_indices = [sensor_config.index(col) for col in rot_cols if col in sensor_config]
            for i, idx in enumerate(rot_indices):
                if idx < X_clean.shape[-1]:
                    if i == 0:  # w成分
                        X_clean[..., idx] = np.nan_to_num(X_clean[..., idx], nan=1.0)
                    else:  # x, y, z成分
                        X_clean[..., idx] = np.nan_to_num(X_clean[..., idx], nan=0.0)
        
        # Thermal: 前後の値で補間（簡易版）
        if thm_cols:
            thm_indices = [sensor_config.index(col) for col in thm_cols if col in sensor_config]
            for idx in thm_indices:
                if idx < X_clean.shape[-1]:
                    # 時系列方向で補間
                    for window_idx in range(X_clean.shape[0]):
                        series = X_clean[window_idx, :, idx]
                        if np.isnan(series).any():
                            # 前後の値で補間
                            series_clean = np.nan_to_num(series, nan=np.nanmean(series))
                            X_clean[window_idx, :, idx] = series_clean
        
        # その他のセンサー: 0で置換
        all_processed_indices = set(acc_indices + rot_indices + thm_indices)
        for i in range(X_clean.shape[-1]):
            if i not in all_processed_indices:
                X_clean[..., i] = np.nan_to_num(X_clean[..., i], nan=0.0)
        
        return X_clean

    def fit(self, df: pd.DataFrame, use_cache: bool = True) -> "Preprocessor":
        logger.info("Fitting Preprocessor")
        df_proc = self._maybe_clean(df)
        windows = self.win_builder.build(df_proc, use_cache=use_cache)
        X_sensor, X_demo, _, _ = windows
        
        # センサー別の適切な欠損値処理
        X_sensor_clean = self._handle_missing_values_by_sensor_type(X_sensor)
        logger.info(
            "Window tensor shape %s, demographics shape %s", X_sensor.shape, X_demo.shape
        )
        self.sensor_scaler.fit(X_sensor_clean.reshape(-1, X_sensor_clean.shape[-1]))
        
        # 人口統計データの正規化
        X_demo_clean = np.nan_to_num(X_demo, nan=0.0)
        self.demo_scaler.fit(X_demo_clean)
        
        # 表形式特徴量の正規化
        tab, _, _ = self.tab_builder.build(df_proc, windows=windows, use_cache=use_cache)
        tab_clean = np.nan_to_num(tab, nan=0.0)
        self.tab_scaler.fit(tab_clean)
        logger.info("Tabular features shape %s", tab.shape)
        
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame, use_cache: bool = True) -> dict:
        if not self._fitted:
            raise RuntimeError("Preprocessor is not fitted")
        logger.info("Transforming dataframe of shape %s", df.shape)
        df_proc = self._maybe_clean(df)
        windows = self.win_builder.build(df_proc, use_cache=use_cache)
        X_sensor, X_demo, y, info = windows
        
        # センサー別の適切な欠損値処理
        logger.info("Window tensor shape %s", X_sensor.shape)
        X_sensor_clean = self._handle_missing_values_by_sensor_type(X_sensor)
        X_sensor_normalized = self.sensor_scaler.transform(
            X_sensor_clean.reshape(-1, X_sensor_clean.shape[-1])
        ).reshape(X_sensor_clean.shape)
        
        # 人口統計データの正規化
        X_demo_clean = np.nan_to_num(X_demo, nan=0.0)
        X_demo_normalized = self.demo_scaler.transform(X_demo_clean)
        
        # 表形式特徴量の正規化
        tab, _, _ = self.tab_builder.build(df_proc, windows=windows, use_cache=use_cache)
        tab_clean = np.nan_to_num(tab, nan=0.0)
        tab_normalized = self.tab_scaler.transform(tab_clean)
        
        tof_tensor = self.tof_builder.build(df_proc, use_cache=use_cache)
        tof_windows, _ = self.tof_win_builder.build(df_proc, use_cache=use_cache)
        logger.info(
            "Output shapes: windows=%s, demographics=%s, tabular=%s, tof=%s, tof_win=%s",
            X_sensor.shape,
            X_demo.shape,
            tab.shape,
            tof_tensor.shape,
            tof_windows.shape,
        )

        validate_no_nan(X_sensor_normalized, "windows")
        validate_no_nan(X_demo_normalized, "demographics")
        validate_no_nan(tab_normalized, "tabular")
        validate_no_nan(tof_tensor, "tof_voxel")
        validate_no_nan(tof_windows, "tof_windows")

        return {
            "windows": X_sensor_normalized,
            "demographics": X_demo_normalized,
            "tabular": tab_normalized,
            "tof_voxel": tof_tensor,
            "tof_windows": tof_windows,
            "labels": y,
            "info": info,
        }

    def fit_transform(self, df: pd.DataFrame, use_cache: bool = True) -> dict:
        self.fit(df, use_cache=use_cache)
        return self.transform(df, use_cache=use_cache)

    def save(self, path: Path) -> None:
        """Save scaler objects and settings to a pickle file."""
        logger.info("Saving Preprocessor to %s", path)
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
        logger.info("Loading Preprocessor from %s", path)
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
