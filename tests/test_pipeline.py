import json
from pathlib import Path
import numpy as np
import pandas as pd
import sys
from pathlib import Path as _Path

# src を import できるようにパスを追加
ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.pipeline import (
    WindowTensorBuilder,
    TabularFeatureBuilder,
    ToFVoxelBuilder,
    ToFWindowBuilder,
    Preprocessor,
)


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


def make_config(tmp_path: Path, ae_model_path: str | None = None) -> dict:
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
            "use_tof_rate_features": False,
            "use_temperature_change_features": False,
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
    if ae_model_path is not None:
        cfg["ae_model_path"] = ae_model_path
    return cfg


def test_window_tensor_builder(tmp_path: Path):
    cfg = make_config(tmp_path)
    df = make_dummy_df()
    builder = WindowTensorBuilder(cfg)

    result = builder.build(df)
    assert isinstance(result, tuple) and len(result) == 4

    windows, demos, labels, info = result
    assert windows.shape == (1, cfg["preprocessing"]["window_size"], 12)
    assert demos.shape == (1, 7)
    assert labels.shape == (1,)
    assert len(info) == 1

    assert builder.cache_file.exists()
    assert builder.meta_file.exists()
    md5_first = json.loads(builder.meta_file.read_text())["md5"]

    result_cached = builder.build(df)
    md5_second = json.loads(builder.meta_file.read_text())["md5"]
    assert md5_first == md5_second
    np.testing.assert_allclose(result[0], result_cached[0])


def test_tabular_feature_builder(tmp_path: Path):
    cfg = make_config(tmp_path)
    cfg["preprocessing"]["use_tof_rate_features"] = True
    cfg["preprocessing"]["use_temperature_change_features"] = True
    df = make_dummy_df()
    win_builder = WindowTensorBuilder(cfg)
    windows = win_builder.build(df)
    tab_builder = TabularFeatureBuilder(cfg)
    tab, labels, info = tab_builder.build(df, windows=windows)

    assert tab.shape == (1, 139)
    assert labels.shape == (1,)
    assert len(info) == 1

    assert tab_builder.cache_file.exists()
    assert tab_builder.meta_file.exists()
    md5_first = json.loads(tab_builder.meta_file.read_text())["md5"]
    tab_cached, _, _ = tab_builder.build(df, windows=windows)
    md5_second = json.loads(tab_builder.meta_file.read_text())["md5"]
    assert md5_first == md5_second
    np.testing.assert_allclose(tab, tab_cached)


def test_tabular_feature_builder_with_ae(tmp_path: Path, monkeypatch):
    class DummyAE:
        def predict(self, X, verbose=0):
            return np.zeros_like(X)

    import types
    tf_mod = types.ModuleType("tensorflow")
    keras_mod = types.ModuleType("keras")
    keras_mod.models = types.ModuleType("models")
    keras_mod.models.load_model = lambda path: DummyAE()
    tf_mod.keras = keras_mod
    monkeypatch.setitem(sys.modules, "tensorflow", tf_mod)
    monkeypatch.setitem(sys.modules, "tensorflow.keras", keras_mod)
    monkeypatch.setitem(sys.modules, "tensorflow.keras.models", keras_mod.models)

    cfg = make_config(tmp_path, ae_model_path=str(tmp_path / "dummy.h5"))
    df = make_dummy_df()
    win_builder = WindowTensorBuilder(cfg)
    windows = win_builder.build(df)
    tab_builder = TabularFeatureBuilder(cfg)
    tab, _, _ = tab_builder.build(df, windows=windows)

    assert tab.shape == (1, 130)


def test_tof_voxel_builder(tmp_path: Path):
    cfg = make_config(tmp_path)
    df = make_dummy_df()
    builder = ToFVoxelBuilder(cfg)
    voxel = builder.build(df)

    depth = cfg["preprocessing"]["tof_depth"]
    h = cfg["preprocessing"]["tof_height"]
    w = cfg["preprocessing"]["tof_width"]
    assert voxel.shape == (len(df), depth, h, w)

    assert builder.cache_file.exists()
    assert builder.meta_file.exists()


def test_tof_window_builder(tmp_path: Path):
    cfg = make_config(tmp_path)
    df = make_dummy_df()
    builder = ToFWindowBuilder(cfg)

    windows, info = builder.build(df)

    depth = cfg["preprocessing"]["tof_depth"]
    h = cfg["preprocessing"]["tof_height"]
    w = cfg["preprocessing"]["tof_width"]
    assert windows.shape == (1, cfg["preprocessing"]["window_size"], depth, h, w)
    assert len(info) == 1

    assert builder.cache_file.exists()
    assert builder.meta_file.exists()
    md5_first = json.loads(builder.meta_file.read_text())["md5"]
    win_cached, info_cached = builder.build(df)
    md5_second = json.loads(builder.meta_file.read_text())["md5"]
    assert md5_first == md5_second
    np.testing.assert_allclose(windows, win_cached)


def test_preprocessor_save_load(tmp_path: Path):
    cfg = make_config(tmp_path)
    df = make_dummy_df()
    pp = Preprocessor(cfg, use_interp_cleaning=False)
    pp.fit(df)
    orig = pp.transform(df)

    pkl = tmp_path / "preproc.pkl"
    pp.save(pkl)
    loaded = Preprocessor.load(pkl)
    out = loaded.transform(df)

    for key in ["windows", "demographics", "tabular", "tof_voxel"]:
        np.testing.assert_allclose(orig[key], out[key])
