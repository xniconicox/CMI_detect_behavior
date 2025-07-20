import json
from pathlib import Path

import numpy as np
import pandas as pd
import sys
from pathlib import Path as _Path

# src を import できるようにパスを追加
ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.pipeline import Preprocessor
from src.utils.io_utils import df_md5


def make_dummy_df(n_rows: int = 10) -> pd.DataFrame:
    data = {
        "subject": [0] * n_rows,
        "sequence_id": [0] * n_rows,
        "sequence_counter": list(range(n_rows)),
        "gesture": [1] * n_rows,
        "adult_child": [1] * n_rows,
        "age": [30] * n_rows,
        "sex": [1] * n_rows,
        "handedness": [1] * n_rows,
        "height_cm": [170] * n_rows,
        "shoulder_to_wrist_cm": [60] * n_rows,
        "elbow_to_wrist_cm": [30] * n_rows,
    }
    sensor_cols = [
        "acc_x",
        "acc_y",
        "acc_z",
        "rot_w",
        "rot_x",
        "rot_y",
        "rot_z",
        "thm_1",
        "thm_2",
        "thm_3",
        "thm_4",
        "thm_5",
    ]
    for c in sensor_cols:
        data[c] = np.random.randn(n_rows)
    for d in range(1, 6):
        for i in range(64):
            data[f"tof_{d}_v{i}"] = np.random.rand(n_rows)
    return pd.DataFrame(data)


def test_preprocessor_cache_and_transform(tmp_path: Path):
    cfg = {
        "cache_dir": str(tmp_path / "cache"),
        "preprocessing": {
            "window_size": 16,
            "stride": 8,
            "min_sequence_length": 4,
            "padding_value": 0.0,
            "sampling_rate": 50.0,
            "fft_bands": [(0.5, 2), (2, 5), (5, 10), (10, 20)],
            "wavelet": "db4",
            "wavelet_level": 3,
            "tda_dimension": 1,
            "tda_bins": 20,
            "tda_sigma": 0.1,
            "use_wavelet_features": False,
            "use_tda_features": False,
            "tof_depth": 5,
            "tof_height": 8,
            "tof_width": 8,
            "use_world_acc": False,
        },
        "sensor_acc_cols": ["acc_x", "acc_y", "acc_z"],
        "sensor_rot_cols": ["rot_w", "rot_x", "rot_y", "rot_z"],
        "sensor_thm_cols": ["thm_1", "thm_2", "thm_3", "thm_4", "thm_5"],
        "demographics_cols": [
            "adult_child",
            "age",
            "sex",
            "handedness",
            "height_cm",
            "shoulder_to_wrist_cm",
            "elbow_to_wrist_cm",
        ],
    }

    df = make_dummy_df()
    pp = Preprocessor(cfg, use_interp_cleaning=False)
    result = pp.fit_transform(df)

    assert result["windows"].shape == (1, 16, 12)
    assert result["demographics"].shape == (1, 7)
    assert result["tabular"].shape == (1, 132)
    assert result["tof_voxel"].shape == (len(df), 5, 8, 8)
    assert result["tof_windows"].shape == (1, 16, 5, 8, 8)
    assert result["labels"].shape == (1,)
    assert len(result["info"]) == 1

    caches = [
        (pp.win_builder.cache_file, pp.win_builder.meta_file),
        (pp.tab_builder.cache_file, pp.tab_builder.meta_file),
        (pp.tof_builder.cache_file, pp.tof_builder.meta_file),
        (pp.tof_win_builder.cache_file, pp.tof_win_builder.meta_file),
    ]
    cleaned = pp._maybe_clean(df)
    md5 = df_md5(cleaned)
    for cache, meta in caches:
        assert cache.exists()
        assert meta.exists()
        meta_md5 = json.loads(meta.read_text())["md5"]
        assert meta_md5 == md5

    pkl = tmp_path / "preproc.pkl"
    pp.save(pkl)
    loaded = Preprocessor.load(pkl)
    out = loaded.transform(df)
    for key in ["windows", "demographics", "tabular", "tof_voxel", "tof_windows"]:
        np.testing.assert_allclose(result[key], out[key])


def test_handle_missing_values_sensor_type(tmp_path: Path):
    cfg = {
        "cache_dir": str(tmp_path / "cache"),
        "preprocessing": {
            "window_size": 16,
            "stride": 8,
            "min_sequence_length": 4,
            "padding_value": 0.0,
            "sampling_rate": 50.0,
            "fft_bands": [(0.5, 2), (2, 5), (5, 10), (10, 20)],
            "wavelet": "db4",
            "wavelet_level": 3,
            "tda_dimension": 1,
            "tda_bins": 20,
            "tda_sigma": 0.1,
            "use_wavelet_features": False,
            "use_tda_features": False,
            "tof_depth": 5,
            "tof_height": 8,
            "tof_width": 8,
            "use_world_acc": False,
        },
        "sensor_acc_cols": ["acc_x", "acc_y", "acc_z"],
        "sensor_rot_cols": ["rot_w", "rot_x", "rot_y", "rot_z"],
        "sensor_thm_cols": ["thm_1", "thm_2", "thm_3", "thm_4", "thm_5"],
        "demographics_cols": [
            "adult_child",
            "age",
            "sex",
            "handedness",
            "height_cm",
            "shoulder_to_wrist_cm",
            "elbow_to_wrist_cm",
        ],
    }

    df = make_dummy_df(20)
    df.loc[0, "acc_x"] = np.nan
    df.loc[0, "rot_w"] = np.nan
    df.loc[0, "rot_x"] = np.nan
    df.loc[1, "thm_1"] = np.nan

    pp = Preprocessor(
        cfg,
        use_handedness=False,
        use_basic_cleaning=False,
        use_interp_cleaning=False,
    )

    windows = pp.win_builder.build(df)
    X_sensor = windows[0]
    sensor_config = pp.win_builder.sensor_cols
    acc_idx = sensor_config.index("acc_x")
    rotw_idx = sensor_config.index("rot_w")
    rotx_idx = sensor_config.index("rot_x")
    thm_idx = sensor_config.index("thm_1")

    assert np.isnan(X_sensor[0, 0, acc_idx])
    assert np.isnan(X_sensor[0, 0, rotw_idx])
    assert np.isnan(X_sensor[0, 0, rotx_idx])
    assert np.isnan(X_sensor[0, 1, thm_idx])

    X_clean = pp._handle_missing_values_by_sensor_type(X_sensor)

    assert X_clean[0, 0, acc_idx] == 0.0
    assert X_clean[0, 0, rotw_idx] == 1.0
    assert X_clean[0, 0, rotx_idx] == 0.0

    thm_series = X_sensor[0, :, thm_idx]
    expected = np.nan_to_num(thm_series, nan=np.nanmean(thm_series))
    np.testing.assert_allclose(X_clean[0, :, thm_idx], expected)


def test_missing_sensor_flags(tmp_path: Path):
    cfg = {
        "cache_dir": str(tmp_path / "cache"),
        "preprocessing": {
            "window_size": 16,
            "stride": 8,
            "min_sequence_length": 4,
            "padding_value": 0.0,
            "sampling_rate": 50.0,
            "fft_bands": [(0.5, 2), (2, 5), (5, 10), (10, 20)],
            "wavelet": "db4",
            "wavelet_level": 3,
            "tda_dimension": 1,
            "tda_bins": 20,
            "tda_sigma": 0.1,
            "use_wavelet_features": False,
            "use_tda_features": False,
            "tof_depth": 5,
            "tof_height": 8,
            "tof_width": 8,
            "use_world_acc": False,
        },
        "sensor_acc_cols": ["acc_x", "acc_y", "acc_z"],
        "sensor_rot_cols": ["rot_w", "rot_x", "rot_y", "rot_z"],
        "sensor_thm_cols": ["thm_1", "thm_2", "thm_3", "thm_4", "thm_5"],
        "demographics_cols": [
            "adult_child",
            "age",
            "sex",
            "handedness",
            "height_cm",
            "shoulder_to_wrist_cm",
            "elbow_to_wrist_cm",
        ],
    }

    df = make_dummy_df()
    imu_cols = cfg["sensor_acc_cols"] + cfg["sensor_rot_cols"]
    thm_cols = cfg["sensor_thm_cols"]
    tof_cols = [f"tof_{d}_v{i}" for d in range(1, 6) for i in range(64)]
    df.loc[0, imu_cols] = np.nan
    df.loc[0, thm_cols] = np.nan
    df.loc[0, tof_cols] = np.nan

    pp = Preprocessor(cfg, use_interp_cleaning=False)
    cleaned = pp._maybe_clean(df)

    for col in ["missing_flag_imu", "missing_flag_thermal", "missing_flag_tof"]:
        assert col in cleaned.columns
        assert cleaned.loc[0, col]

    result = pp.fit_transform(df)
    assert result["tabular"].shape[1] == 132
