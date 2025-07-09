# -*- coding: utf-8 -*-
"""Preprocessing utilities for the CMI competition.

This module provides functions for:
- IMU world coordinate transformation
- sliding window generation with demographics
- normalization utilities
- simple peak feature extraction
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion q=[w, x, y, z] to a rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)],
    ])


def rotate_acceleration(acc: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Rotate accelerometer vector to world coordinates."""
    return quaternion_to_rotation_matrix(quat) @ acc


def linear_acceleration(acc: np.ndarray, quat: np.ndarray, gravity: float = 9.81) -> np.ndarray:
    """Return linear acceleration by removing gravity after rotation."""
    acc_world = rotate_acceleration(acc, quat)
    gravity_vec = np.array([0, 0, gravity])
    return acc_world - gravity_vec


def create_sliding_windows_with_demographics(
    df: pd.DataFrame,
    window_size: int,
    stride: int,
    sensor_cols: list,
    demographics_cols: list,
    min_sequence_length: int = 10,
    padding_value: float = 0.0,
):
    """Create sliding windows for sequences with static demographics features."""
    X_sensor_windows = []
    X_demographics_windows = []
    y_windows = []
    sequence_info = []

    grouped = df.groupby(["subject", "sequence_id"])
    for (subject, sequence_id), group in grouped:
        sequence_length = len(group)
        gesture = group["gesture"].iloc[0]
        if sequence_length < min_sequence_length:
            continue
        sensor_data = group[sensor_cols].values
        demographics_data = group[demographics_cols].iloc[0].values
        need_padding = sequence_length < window_size
        if need_padding:
            padding_size = window_size - sequence_length
            padding = np.full((padding_size, len(sensor_cols)), padding_value)
            sensor_data = np.vstack([sensor_data, padding])
        for start_idx in range(0, len(sensor_data) - window_size + 1, stride):
            end_idx = start_idx + window_size
            window = sensor_data[start_idx:end_idx]
            X_sensor_windows.append(window)
            X_demographics_windows.append(demographics_data)
            y_windows.append(gesture)
            sequence_info.append(
                {
                    "subject": subject,
                    "sequence_id": sequence_id,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "original_length": sequence_length,
                    "padded": need_padding,
                }
            )

    X_sensor_windows = np.array(X_sensor_windows, dtype=np.float32)
    X_demographics_windows = np.array(X_demographics_windows, dtype=np.float32)
    y_windows = np.array(y_windows)

    return X_sensor_windows, X_demographics_windows, y_windows, sequence_info


def normalize_sensor_data(X: np.ndarray):
    """Normalize sensor windows and return normalized data and scaler."""
    scaler = StandardScaler()
    n_samples, n_timesteps, n_features = X.shape
    flat = X.reshape(-1, n_features)
    flat = np.nan_to_num(flat, nan=0.0)
    normalized = scaler.fit_transform(flat).reshape(n_samples, n_timesteps, n_features)
    return normalized, scaler


def normalize_tabular_data(X: np.ndarray):
    """Normalize tabular demographics data."""
    scaler = StandardScaler()
    normalized = scaler.fit_transform(X)
    return normalized, scaler


def extract_peak_features(window: np.ndarray) -> np.ndarray:
    """Return peak counts for each axis within a window."""
    features = []
    for i in range(window.shape[1]):
        peaks, _ = find_peaks(window[:, i])
        features.append(len(peaks))
    return np.array(features, dtype=np.float32)


def compute_peak_features(X_windows: np.ndarray) -> np.ndarray:
    """Compute peak count features for multiple windows."""
    return np.vstack([extract_peak_features(w) for w in X_windows])


def add_missing_sensor_flags(df: pd.DataFrame, sensor_groups: dict) -> pd.DataFrame:
    """Add missing sensor flag columns for each sensor group.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing raw sensor columns.
    sensor_groups : dict
        Dictionary mapping flag column names to a list of column names that
        belong to the sensor group.

    Returns
    -------
    pd.DataFrame
        Dataframe with additional boolean flag columns.
    """
    for flag, cols in sensor_groups.items():
        df[flag] = df[cols].isna().all(axis=1)
    return df


def compute_basic_statistics(X_windows: np.ndarray) -> np.ndarray:
    """Compute simple statistics for each window.

    Statistics include mean, std, range, RMS, energy for each feature and
    mean/std of 3D vector magnitude for the first three axes (acceleration).
    """
    means = X_windows.mean(axis=1)
    stds = X_windows.std(axis=1)
    ranges = X_windows.max(axis=1) - X_windows.min(axis=1)
    rms = np.sqrt((X_windows ** 2).mean(axis=1))
    energy = (X_windows ** 2).sum(axis=1)

    # magnitude assuming the first three columns represent a vector
    if X_windows.shape[2] >= 3:
        mag = np.linalg.norm(X_windows[:, :, :3], axis=2)
        mag_mean = mag.mean(axis=1, keepdims=True)
        mag_std = mag.std(axis=1, keepdims=True)
        stats = np.hstack([means, stds, ranges, rms, energy, mag_mean, mag_std])
    else:
        stats = np.hstack([means, stds, ranges, rms, energy])
    return stats


def compute_fft_band_energy(
    X_windows: np.ndarray,
    fs: float = 50.0,
    bands: list | None = None,
) -> np.ndarray:
    """Compute FFT band energies for each window and feature.

    Parameters
    ----------
    X_windows : np.ndarray
        Shape (n_windows, window_size, n_features).
    fs : float, optional
        Sampling frequency used to compute the FFT. Defaults to 50Hz.
    bands : list of tuple, optional
        List of (low, high) frequency pairs specifying the bands.
        Defaults to [(0.5, 2), (2, 5), (5, 10), (10, 20)].

    Returns
    -------
    np.ndarray
        Concatenated band energies for all features.
    """
    if bands is None:
        bands = [(0.5, 2), (2, 5), (5, 10), (10, 20)]

    n_windows, window_size, n_features = X_windows.shape
    freqs = np.fft.rfftfreq(window_size, d=1.0 / fs)
    fft_vals = np.fft.rfft(X_windows, axis=1)
    power = np.abs(fft_vals) ** 2

    band_energy_list = []
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        band_energy = power[:, mask, :].sum(axis=1)
        band_energy_list.append(band_energy)

    return np.concatenate(band_energy_list, axis=1)


def handedness_normalization(
    df: pd.DataFrame,
    axis_cols: list,
    handedness_col: str = "handedness",
) -> pd.DataFrame:
    """Flip Y/Z axes for left handed subjects.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing IMU axes.
    axis_cols : list
        Columns corresponding to [x, y, z] axes.
    handedness_col : str, optional
        Column indicating handedness (0 left, 1 right).

    Returns
    -------
    pd.DataFrame
        Dataframe with axes flipped for left handed samples.
    """
    y_col, z_col = axis_cols[1], axis_cols[2]
    left_mask = df[handedness_col] == 0
    df.loc[left_mask, [y_col, z_col]] *= -1
    return df


def compute_wavelet_features(
    X_windows: np.ndarray,
    wavelet: str = "db4",
    level: int = 3,
) -> np.ndarray:
    """Extract wavelet band energies using discrete wavelet transform."""
    import pywt  # local import to avoid hard dependency

    features = []
    for window in X_windows:
        axis_feats = []
        for i in range(window.shape[1]):
            coeffs = pywt.wavedec(window[:, i], wavelet=wavelet, level=level)
            energies = [np.sum(c ** 2) for c in coeffs]
            axis_feats.extend(energies)
        features.append(axis_feats)
    return np.array(features, dtype=np.float32)


def compute_persistence_image_features(
    X_windows: np.ndarray,
    dimension: int = 1,
    n_bins: int = 20,
    sigma: float = 0.1,
) -> np.ndarray:
    """Compute persistence image features using giotto-tda.

    This function requires `giotto-tda` to be installed. If the package is not
    available, an ImportError is raised when calling this function.
    """
    try:
        from gtda.time_series import TakensEmbedding
        from gtda.homology import VietorisRipsPersistence
        from gtda.diagrams import PersistenceImage
    except Exception as e:  # pragma: no cover - library may not be installed
        raise ImportError(
            "giotto-tda is required for persistence image features"
        ) from e

    embedding = TakensEmbedding(time_delay=1, dimension=dimension)
    persistence = VietorisRipsPersistence(homology_dimensions=[0, 1])
    pimage = PersistenceImage(bandwidth=sigma, n_bins=(n_bins, n_bins))

    features = []
    for window in X_windows:
        emb = embedding.fit_transform(window)
        diag = persistence.fit_transform(emb[np.newaxis, ...])
        img = pimage.fit_transform(diag)
        features.append(img.reshape(-1))
    return np.array(features, dtype=np.float32)


def compute_autoencoder_reconstruction_error(
    X_windows: np.ndarray,
    model,
) -> np.ndarray:
    """Compute per-window MSE reconstruction error using a trained AE model."""
    reconstructed = model.predict(X_windows, verbose=0)
    mse = ((X_windows - reconstructed) ** 2).mean(axis=(1, 2))
    return mse.reshape(-1, 1)


def tof_to_voxel_tensor(
    df: pd.DataFrame,
    fill_value: float = 0.0,
    prefix: str = "tof_",
) -> np.ndarray:
    """Convert ToF sensor columns to a 3D voxel tensor.

    The dataframe is expected to contain columns in the form
    ``tof_{sensor}_v{index}`` where ``index`` ranges from 0 to
    ``height * width - 1``. Sensor numbers are used as the depth
    dimension of the resulting tensor.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with ToF columns.
    fill_value : float, optional
        Value used to replace NaN or -1 readings. Defaults to 0.0.
    prefix : str, optional
        Prefix for ToF column names. Defaults to "tof_".

    Returns
    -------
    np.ndarray
        Tensor with shape ``(len(df), depth, height, width)``.
    """
    import re

    pattern = re.compile(fr"^{prefix}(\d+)_v(\d+)$")
    matches = [(col, pattern.match(col)) for col in df.columns]
    sensor_nums = sorted({int(m.group(1)) for col, m in matches if m})
    if not sensor_nums:
        raise ValueError("No ToF columns found in dataframe")

    # Determine grid size from the largest voxel index
    voxel_indices = [int(m.group(2)) for col, m in matches if m]
    max_index = max(voxel_indices)
    width = height = int(np.sqrt(max_index + 1))
    depth = len(sensor_nums)

    n_timesteps = len(df)
    tensor = np.full((n_timesteps, depth, height, width), fill_value, dtype=np.float32)

    for d, sensor_num in enumerate(sensor_nums):
        for idx in range(height * width):
            col = f"{prefix}{sensor_num}_v{idx}"
            if col in df.columns:
                values = df[col].replace(-1, fill_value).to_numpy(dtype=np.float32)
                r, c = divmod(idx, width)
                tensor[:, d, r, c] = values

    return tensor
