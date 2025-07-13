from pathlib import Path
import pickle
import numpy as np
import pandas as pd
import sys
from pathlib import Path as _Path

ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.pipeline import Preprocessor
from src.trainers.tof_3d_cnn_trainer import ToF3DCNNTrainer


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


def make_config(tmp_path: Path) -> dict:
    return {
        "output_dir": str(tmp_path / "out"),
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


def test_tof3dcnn_trainer_load_data(tmp_path: Path):
    cfg = make_config(tmp_path)
    df = make_dummy_df()
    pp = Preprocessor(cfg, use_interp_cleaning=False)
    result = pp.fit_transform(df)

    pre_dir = Path(cfg["output_dir"]) / "exp" / "preprocessed"
    pre_dir.mkdir(parents=True, exist_ok=True)
    with open(pre_dir / "train_tof_windows.pkl", "wb") as f:
        pickle.dump(result["tof_windows"], f)
    with open(pre_dir / "train_labels.pkl", "wb") as f:
        pickle.dump(result["labels"], f)

    trainer = ToF3DCNNTrainer(experiment_name="exp")
    trainer.config = cfg
    trainer.preprocessed_dir = pre_dir
    trainer.window_size = cfg["preprocessing"]["window_size"]
    trainer.fill_value = cfg["preprocessing"]["padding_value"]

    X, y = trainer.load_data()
    assert X.shape == result["tof_windows"].shape
    assert y.shape == result["labels"].shape
