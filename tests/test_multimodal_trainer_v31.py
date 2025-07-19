import pickle
import sys
import types
from pathlib import Path as _Path
import numpy as np

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

from src.trainers.multimodal_trainer_v31 import MultimodalTrainerV31


def test_load_all_data_casts_tabular(tmp_path):
    pre_dir = tmp_path
    arrays = {
        "train_windows": np.random.randn(2, 3, 4),
        "train_demographics": np.random.randn(2, 2),
        "train_tabular": np.random.randn(2, 5),
        "train_tof_windows": np.random.randn(2, 1, 1, 1, 1),
        "train_labels": np.array([0, 1]),
        "train_info": np.array([0, 1]),
    }
    for name, arr in arrays.items():
        with open(pre_dir / f"{name}.pkl", "wb") as f:
            pickle.dump(arr, f)

    trainer = MultimodalTrainerV31()
    trainer.data_dir = pre_dir
    data = trainer.load_all_data()

    assert data["tabular"].dtype == np.float32
    np.testing.assert_allclose(data["tabular"], arrays["train_tabular"].astype(np.float32))

    key_map = {
        "sensor": "train_windows",
        "demographics": "train_demographics",
        "tof": "train_tof_windows",
    }
    for key, src in key_map.items():
        np.testing.assert_array_equal(data[key], arrays[src])

    np.testing.assert_array_equal(data["labels"], arrays["train_labels"])
    np.testing.assert_array_equal(data["groups"], arrays["train_info"])

