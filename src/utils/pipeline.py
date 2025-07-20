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
    compute_autoencoder_reconstruction_error,
    add_missing_sensor_flags,
    compute_tof_rate_of_change,
    compute_temperature_change_features
)
from .imu import add_world_acc_features


logger = logging.getLogger(__name__)

class WindowTensorBuilder:
    """Generate IMU window tensors with demographics for multiple sizes."""

    def __init__(self, config: dict | None = None, mode: str = "train") -> None:
        self.config = config or load_config()
        self.mode = mode
        pp = self.config.get("preprocessing", {})
        self.window_lens = pp.get("window_lens") or [pp.get("window_size", 128)]
        self.stride = pp.get("stride", 64)
        self.min_len = pp.get("min_sequence_length", 10)
        self.padding_value = pp.get("padding_value", 0.0)
        
        self.sensor_cols = (
            self.config.get("sensor_acc_cols", [])
            + self.config.get("sensor_rot_cols", [])
            + self.config.get("sensor_thm_cols", [])
        )
        if pp.get("use_world_acc", False):
            world_acc_cols = [f"acc_w_{ax}" for ax in "xyz"] + [f"lin_acc_{ax}" for ax in "xyz"]
            self.sensor_cols.extend(world_acc_cols)
        
        self.demographics_cols = self.config.get("demographics_cols", [])
        self.cache_dir = get_cache_dir(self.config)
        self.cache_files = {
            l: self.cache_dir / f"windows_{l}.pkl" for l in self.window_lens
        }
        self.meta_files = {
            l: self.cache_dir / f"windows_{l}_meta.json" for l in self.window_lens
        }
        self.cache_dir.mkdir(exist_ok=True)

    def build(self, df: pd.DataFrame, use_cache: bool = True):
        logger.info(
            "Building windows (lens=%s, stride=%d, min_len=%d)",
            self.window_lens,
            self.stride,
            self.min_len,
        )
        md5 = df_md5(df)
        results: dict[int, tuple] = {}
        for l in self.window_lens:
            c_file = self.cache_files[l]
            m_file = self.meta_files[l]
            if use_cache and c_file.exists() and m_file.exists():
                meta = json.loads(m_file.read_text())
                if meta.get("md5") == md5:
                    logger.info("Reusing cached windows from %s", c_file)
                    with open(c_file, "rb") as f:
                        results[l] = pickle.load(f)
                    continue

            result = create_sliding_windows_with_demographics(
                df,
                window_size=l,
                stride=self.stride,
                sensor_cols=self.sensor_cols,
                demographics_cols=self.demographics_cols,
                min_sequence_length=self.min_len,
                padding_value=self.padding_value,
            )
            if use_cache:
                with open(c_file, "wb") as f:
                    pickle.dump(result, f)
                m_file.write_text(json.dumps({"md5": md5}))
            results[l] = result
            logger.info(
                "Windows length %d shape %s",
                l,
                result[0].shape,
            )
        return results


class TabularFeatureBuilder:
    """Build tabular features from IMU windows."""

    def __init__(self, config: dict | None = None) -> None:
        self.config = config or load_config()
        pp = self.config.get("preprocessing", {})
        self.sampling_rate = pp.get("sampling_rate", 50.0)
        self.fft_bands = pp.get("fft_bands", [])
        self.use_wavelet = pp.get("use_wavelet_features", False)
        self.use_tda = pp.get("use_tda_features", False)
        self.use_tof_rate = pp.get("use_tof_rate_features", False)
        self.use_temp_change = pp.get("use_temperature_change_features", False)
        self.wavelet = pp.get("wavelet", "db4")
        self.wavelet_level = pp.get("wavelet_level", 3)
        self.tda_dimension = pp.get("tda_dimension", 1)
        self.tda_bins = pp.get("tda_bins", 20)
        self.tda_sigma = pp.get("tda_sigma", 0.1)
        self.ae_model_path = self.config.get("ae_model_path")
        self._ae_model = None
        self.window_builder = WindowTensorBuilder(self.config)
        self.tof_window_builder = ToFWindowBuilder(self.config)

        acc_cols = self.config.get("sensor_acc_cols", [])
        rot_cols = self.config.get("sensor_rot_cols", [])
        thm_cols = self.config.get("sensor_thm_cols", [])
        start = len(acc_cols) + len(rot_cols)
        self._thm_slice = slice(start, start + len(thm_cols))
        self.cache_dir = get_cache_dir(self.config)
        self.cache_file = self.cache_dir / "tabular_features.pkl"
        self.meta_file = self.cache_dir / "tabular_meta.json"
        self.cache_dir.mkdir(exist_ok=True)

    def _load_ae_model(self):
        if self.ae_model_path and self._ae_model is None:
            try:
                from tensorflow.keras.models import load_model
            except Exception as e:  # pragma: no cover
                raise RuntimeError("TensorFlow is required for AE features") from e
            self._ae_model = load_model(self.ae_model_path)
        return self._ae_model

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
        if self.ae_model_path and Path(self.ae_model_path).exists():
            md5 = md5 + str(Path(self.ae_model_path).stat().st_mtime_ns)
        use_wavelet = self.use_wavelet if use_wavelet is None else use_wavelet
        use_tda = self.use_tda if use_tda is None else use_tda
        logger.info(
            "Building tabular features (wavelet=%s, tda=%s)",
            use_wavelet,
            use_tda,
        )
        if use_cache and self.cache_file.exists() and self.meta_file.exists():
            meta = json.loads(self.meta_file.read_text())
            if meta.get("md5") == md5 and meta.get("ae_model_path") == self.ae_model_path:
                logger.info("Reusing cached tabular features from %s", self.cache_file)
                with open(self.cache_file, "rb") as f:
                    return pickle.load(f)

        if windows is None:
            window_dict = self.window_builder.build(df, use_cache=use_cache)
        else:
            window_dict = windows

        results = {}
        for win_len, win_data in window_dict.items():
            X_sensor, X_demo, y, info = win_data

            stats = compute_basic_statistics(X_sensor)
            peaks = compute_peak_features(X_sensor)
            fft = compute_fft_band_energy(
                X_sensor, fs=self.sampling_rate, bands=self.fft_bands
            )

            feats = [stats, peaks, fft]
            if self.use_temp_change and self._thm_slice.stop > self._thm_slice.start:
                temp = compute_temperature_change_features(
                    X_sensor[:, :, self._thm_slice]
                )
                feats.append(temp)
            if self.use_tof_rate:
                tof_dict = self.tof_window_builder.build(df, use_cache=use_cache)
                tof_windows, _ = tof_dict.get(win_len, (None, None))
                if tof_windows is not None:
                    tof_chg = compute_tof_rate_of_change(tof_windows)
                    feats.append(tof_chg)
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
            if self.ae_model_path:
                model = self._load_ae_model()
                ae_err = compute_autoencoder_reconstruction_error(X_sensor, model)
                feats.append(ae_err)

            features = np.hstack(feats + [X_demo])
            final_nan_count = np.isnan(features).sum()
            if final_nan_count > 0:
                logger.warning(
                    f"TabularFeatureBuilder: 最終特徴量にNaN値 {final_nan_count} 個を検出、0.0に置換"
                )
                features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            results[win_len] = (features, y, info)

        if use_cache:
            with open(self.cache_file, "wb") as f:
                pickle.dump(results, f)
            meta = {"md5": md5, "ae_model_path": self.ae_model_path}
            self.meta_file.write_text(json.dumps(meta))
        for l, r in results.items():
            logger.info("Tabular features len %d shape %s", l, r[0].shape)
        return results


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
        self.window_lens = pp.get("window_lens") or [pp.get("window_size", 128)]
        self.stride = pp.get("stride", 64)
        self.min_len = pp.get("min_sequence_length", 10)
        self.fill_value = pp.get("padding_value", 0.0)
        depth = pp.get("tof_depth", 5)
        h = pp.get("tof_height", 8)
        w = pp.get("tof_width", 8)
        self.tof_cols = [f"tof_{d}_v{i}" for d in range(1, depth + 1) for i in range(h * w)]
        self.cache_dir = get_cache_dir(self.config)
        self.cache_files = {l: self.cache_dir / f"tof_windows_{l}.pkl" for l in self.window_lens}
        self.meta_files = {l: self.cache_dir / f"tof_windows_{l}_meta.json" for l in self.window_lens}
        self.cache_dir.mkdir(exist_ok=True)

    def build(self, df: pd.DataFrame, use_cache: bool = True):
        logger.info(
            "Building ToF windows (lens=%s, stride=%d, min_len=%d)",
            self.window_lens,
            self.stride,
            self.min_len,
        )
        md5 = df_md5(df)
        results = {}
        for l in self.window_lens:
            c_file = self.cache_files[l]
            m_file = self.meta_files[l]
            if use_cache and c_file.exists() and m_file.exists():
                meta = json.loads(m_file.read_text())
                if meta.get("md5") == md5:
                    logger.info("Reusing cached ToF windows from %s", c_file)
                    with open(c_file, "rb") as f:
                        results[l] = pickle.load(f)
                    continue

            result = create_tof_windows_with_info(
                df,
                window_size=l,
                stride=self.stride,
                tof_cols=self.tof_cols,
                min_sequence_length=self.min_len,
                fill_value=self.fill_value,
            )
            if use_cache:
                with open(c_file, "wb") as f:
                    pickle.dump(result, f)
                m_file.write_text(json.dumps({"md5": md5}))
            results[l] = result
            logger.info("ToF windows len %d shape %s", l, result[0].shape)

        return results


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
            "World_Accelerometer": [],  # ワールド座標系専用
            "Rotation": self.config.get("sensor_rot_cols", []),
            "Thermal": self.config.get("sensor_thm_cols", []),
            "ToF_Sensor": [f"tof_{d}_v{i}" for d in range(1, depth + 1) for i in range(h * w)],
        }
        if self.use_world_acc:
            self.sensor_type_groups["World_Accelerometer"].extend(
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
        # 欠損フラグ列を追加
        sensor_groups = {
            "missing_flag_imu": self.sensor_type_groups.get("Accelerometer", [])
            + self.sensor_type_groups.get("Rotation", []),
            "missing_flag_thermal": self.sensor_type_groups.get("Thermal", []),
            "missing_flag_tof": self.sensor_type_groups.get("ToF_Sensor", []),
        }
        processed = add_missing_sensor_flags(processed, sensor_groups)
        # WindowTensorBuilder にもフラグ列を含める
        for flag in sensor_groups.keys():
            for builder in [self.win_builder, self.tab_builder.window_builder]:
                if flag not in builder.sensor_cols:
                    builder.sensor_cols.append(flag)

        if self.use_world_acc:
            processed = add_world_acc_features(processed)
        return processed
    
    def _debug_nan_values(self, X_sensor: np.ndarray, stage: str = ""):
        """NaN値のデバッグ用関数"""
        if X_sensor is None:
            return
            
        nan_count = np.isnan(X_sensor).sum()
        if nan_count > 0:
            print(f"⚠️  {stage}: NaN値 {nan_count} 個を発見")
            
            # どの次元にNaNがあるかを確認
            nan_mask = np.isnan(X_sensor)
            if nan_mask.any():
                # ウィンドウごとのNaN数
                nan_per_window = nan_mask.sum(axis=(1, 2))
                windows_with_nan = (nan_per_window > 0).sum()
                print(f"   NaNを含むウィンドウ数: {windows_with_nan}/{len(X_sensor)}")
                
                # 特徴量次元ごとのNaN数
                nan_per_feature = nan_mask.sum(axis=(0, 1))
                features_with_nan = (nan_per_feature > 0).sum()
                print(f"   NaNを含む特徴量数: {features_with_nan}/{X_sensor.shape[-1]}")
                
                # 各特徴量のNaN数を詳細表示
                sensor_config = self.win_builder.sensor_cols
                print("   各特徴量のNaN数:")
                for i, count in enumerate(nan_per_feature):
                    if count > 0:
                        col_name = sensor_config[i] if i < len(sensor_config) else f"feature_{i}"
                        print(f"     {col_name} (index {i}): {count} 個")
                
                # 最初のNaNの位置を表示
                nan_indices = np.where(nan_mask)
                if len(nan_indices[0]) > 0:
                    first_nan = (nan_indices[0][0], nan_indices[1][0], nan_indices[2][0])
                    col_name = sensor_config[first_nan[2]] if first_nan[2] < len(sensor_config) else f"feature_{first_nan[2]}"
                    print(f"   最初のNaN位置: ウィンドウ{first_nan[0]}, 時間{first_nan[1]}, {col_name}(index {first_nan[2]})")
        else:
            print(f"✅ {stage}: NaN値なし")
            
    def _handle_missing_values_by_sensor_type(self, X_sensor: np.ndarray) -> np.ndarray:
        """センサー別に適切な欠損値処理を行う"""
        if X_sensor is None:
            return X_sensor
            
        X_clean = X_sensor.copy()
        # # デバッグ: 処理前のNaN値確認
        # self._debug_nan_values(X_sensor, "処理前")
        
        # センサー別の処理
        sensor_config = self.win_builder.sensor_cols
        acc_cols = self.config.get("sensor_acc_cols", [])
        rot_cols = self.config.get("sensor_rot_cols", [])
        thm_cols = self.config.get("sensor_thm_cols", [])

        # ワールド座標系加速度の列を追加
        world_acc_cols = []
        if self.use_world_acc:
            # ワールド座標系加速度の列を追加
            world_acc_cols = self.sensor_type_groups.get("World_Accelerometer", [])
            
        acc_indices = []
        rot_indices = []
        thm_indices = []
        world_acc_indices = []
        
        # Accelerometer: 0で置換（物理的に正当）
        if acc_cols:
            acc_indices = [sensor_config.index(col) for col in acc_cols if col in sensor_config]
            for idx in acc_indices:
                if idx < X_clean.shape[-1]:
                    X_clean[..., idx] = np.nan_to_num(X_clean[..., idx], nan=0.0)
        
        # World Accelerometer: 0で置換（加速度と同様）
        if world_acc_cols:
            world_acc_indices = [sensor_config.index(col) for col in world_acc_cols if col in sensor_config]
            for idx in world_acc_indices:
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
                            # 前後の値で補間（空のスライス対策）
                            if np.isnan(series).all():
                                # すべてNaNの場合は0で置換
                                series_clean = np.zeros_like(series)
                            else:
                                # 一部NaNの場合は平均で補間
                                series_clean = np.nan_to_num(series, nan=np.nanmean(series))
                            X_clean[window_idx, :, idx] = series_clean
        
        # その他のセンサー: 0で置換
        all_processed_indices = set(acc_indices + rot_indices + thm_indices + world_acc_indices)
        for i in range(X_clean.shape[-1]):
            if i not in all_processed_indices:
                X_clean[..., i] = np.nan_to_num(X_clean[..., i], nan=0.0)
        
        # # デバッグ: 処理後のNaN値確認
        # self._debug_nan_values(X_clean, "処理後")
        return X_clean

    def fit(self, df: pd.DataFrame, use_cache: bool = True) -> "Preprocessor":
        logger.info("Fitting Preprocessor")
        df_proc = self._maybe_clean(df)
        window_dict = self.win_builder.build(df_proc, use_cache=use_cache)
        first_len = sorted(window_dict.keys())[0]
        X_sensor, X_demo, y, _ = window_dict[first_len]

        cleaned_windows = {}
        for l, (xs, xd, yl, info) in window_dict.items():
            xs_clean = self._handle_missing_values_by_sensor_type(xs)
            cleaned_windows[l] = (xs_clean, xd, yl, info)
            if l == first_len:
                X_sensor_clean = xs_clean
                X_demo_first = xd
        
        # ラベルエンコーディング
        if "gesture" in df.columns:
            from sklearn.preprocessing import LabelEncoder
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(df["gesture"])
            logger.info(f"ラベルエンコーダー作成: {self.label_encoder.classes_}")
            logger.info(f"ラベルマッピング: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
        # センサー別の適切な欠損値処理
        logger.info("Window tensor shape %s", X_sensor_clean.shape)
        self.sensor_scaler.fit(X_sensor_clean.reshape(-1, X_sensor_clean.shape[-1]))
    
        
        # 人口統計データの正規化
        X_demo_clean = np.nan_to_num(X_demo_first, nan=0.0)
        self.demo_scaler.fit(X_demo_clean)
        
        # 表形式特徴量の正規化
        processed_windows = cleaned_windows
        tab_dict = self.tab_builder.build(df_proc, windows=processed_windows, use_cache=use_cache)
        tab_first = tab_dict[first_len][0]
        tab_clean = np.nan_to_num(tab_first, nan=0.0)
        self.tab_scaler.fit(tab_clean)
        logger.info("Tabular features shape %s", tab_first.shape)
        
        # === ToF正規化パラメータ計算 ===
        tof_tensor = self.tof_builder.build(df_proc, use_cache=use_cache)
        mask = (tof_tensor != -1)
        self.tof_mean = float(tof_tensor[mask].mean())
        self.tof_std = float(tof_tensor[mask].std())
        logger.info(f"ToF mean: {self.tof_mean:.3f}, std: {self.tof_std:.3f}")

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame, use_cache: bool = True) -> dict:
        if not self._fitted:
            raise RuntimeError("Preprocessor is not fitted")
        logger.info("Transforming dataframe of shape %s", df.shape)
        df_proc = self._maybe_clean(df)
        window_dict = self.win_builder.build(df_proc, use_cache=use_cache)
        first_len = sorted(window_dict.keys())[0]
        X_sensor, X_demo, y, info = window_dict[first_len]
        cleaned_windows = {}
        X_sensor_norm_dict = {}
        X_demo_norm_dict = {}
        for l, (xs, xd, yl, inf) in window_dict.items():
            xs_clean = self._handle_missing_values_by_sensor_type(xs)
            xs_norm = self.sensor_scaler.transform(xs_clean.reshape(-1, xs_clean.shape[-1])).reshape(xs_clean.shape)
            xd_clean = np.nan_to_num(xd, nan=0.0)
            xd_norm = self.demo_scaler.transform(xd_clean)
            X_sensor_norm_dict[l] = xs_norm
            X_demo_norm_dict[l] = xd_norm
            cleaned_windows[l] = (xs_clean, xd, yl, inf)
        
        # センサー別の適切な欠損値処理、正規化
        logger.info("Window tensor shape %s", X_sensor.shape)
        # 表形式特徴量の正規化
        tab_dict = self.tab_builder.build(df_proc, windows=cleaned_windows, use_cache=use_cache)
        tab_normalized = {l: self.tab_scaler.transform(t[0]) for l, t in tab_dict.items()}
        
        # === ToF正規化 ===
        tof_tensor = self.tof_builder.build(df_proc, use_cache=use_cache)
        mask = (tof_tensor != -1)
        tof_tensor_norm = np.zeros_like(tof_tensor)
        if hasattr(self, "tof_mean") and hasattr(self, "tof_std") and self.tof_std > 0:
            tof_tensor_norm[mask] = (tof_tensor[mask] - self.tof_mean) / self.tof_std
        else:
            tof_tensor_norm = tof_tensor  # 正規化できない場合はそのまま
        tof_win_dict = self.tof_win_builder.build(df_proc, use_cache=use_cache)
        tof_windows_norm = {}
        for l, (tw, _) in tof_win_dict.items():
            mask_win = (tw != -1)
            tw_norm = np.zeros_like(tw)
            if hasattr(self, "tof_mean") and hasattr(self, "tof_std") and self.tof_std > 0:
                tw_norm[mask_win] = (tw[mask_win] - self.tof_mean) / self.tof_std
            else:
                tw_norm = tw
            tof_windows_norm[l] = tw_norm
            
        logger.info(
            "Output lens: %s", list(window_dict.keys())
        )
        # ラベルエンコーディングの適用
        if hasattr(self, "label_encoder") and y is not None:
            # testデータかどうかを判定（-1のみの場合はtestデータ）
            if len(np.unique(y)) == 1 and y[0] == -1:
                logger.info("testデータのため、ラベルエンコーディングをスキップ")
                y_encoded = y  # -1のまま保持
            else:
                # trainデータの場合のみラベルエンコーディングを適用
                y_encoded = self.label_encoder.transform(y)
                logger.info(f"ラベルエンコーディング適用: {np.unique(y_encoded)}")
        else:
            y_encoded = y
        
        return {
            "windows": X_sensor_norm_dict,
            "demographics": X_demo_norm_dict,
            "tabular": tab_normalized,
            "tof_voxel": tof_tensor_norm,
            "tof_windows": tof_windows_norm,
            "labels": y_encoded,
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
            "tof_mean": getattr(self, "tof_mean", None),
            "tof_std": getattr(self, "tof_std", None),
            "label_encoder": getattr(self, "label_encoder", None),
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
        obj.tof_mean = data.get("tof_mean", None)
        obj.tof_std = data.get("tof_std", None)
        obj.label_encoder = data.get("label_encoder", None)
        return obj
