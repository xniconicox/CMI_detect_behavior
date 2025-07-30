import pickle
import sys
import types
from pathlib import Path as _Path
import numpy as np
import pytest

# minimal tensorflow stub
tf_mod = types.ModuleType("tensorflow")
keras_mod = types.ModuleType("keras")
keras_mod.layers = types.ModuleType("layers")
keras_mod.models = types.ModuleType("models")
tf_mod.keras = keras_mod
sys.modules.setdefault("tensorflow", tf_mod)
sys.modules.setdefault("tensorflow.keras", keras_mod)
sys.modules.setdefault("tensorflow.keras.layers", keras_mod.layers)
sys.modules.setdefault("tensorflow.keras.models", keras_mod.models)

ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

mm = pytest.importorskip("src.trainers.multimodal_trainer_v51")
from src.trainers.multimodal_trainer_v51 import MultimodalTrainerV51


def test_load_all_data_casts_features(tmp_path):
    pre_dir = tmp_path
    arrays = {
        "train_sequences": np.random.randn(2, 3, 4).astype(np.float32),
        "train_features": np.random.randn(2, 5).astype(np.float32),
        "train_tof_voxels": np.random.randn(2, 3, 2, 2, 2).astype(np.float32),
        "train_labels": np.array([0, 1]),
        "train_info": [
            {"length": 3, "subject": 0},
            {"length": 3, "subject": 1},
        ],
    }
    for name, arr in arrays.items():
        with open(pre_dir / f"{name}.pkl", "wb") as f:
            pickle.dump(arr, f)

    trainer = MultimodalTrainerV51()
    trainer.data_dir = pre_dir
    data = trainer.load_all_data()

    np.testing.assert_array_equal(data["sensor"], arrays["train_sequences"])
    np.testing.assert_array_equal(data["features"], arrays["train_features"])
    np.testing.assert_array_equal(data["tof"], arrays["train_tof_voxels"])
    np.testing.assert_array_equal(data["labels"], arrays["train_labels"])
    assert data["sensor_mask"].shape == (2, 3)

