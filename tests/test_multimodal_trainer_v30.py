import pickle
import sys
import types
from pathlib import Path as _Path
import numpy as np

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

from src.trainers.multimodal_trainer_v30 import MultimodalTrainer


class DummyModel:
    def save(self, path):
        _Path(path).touch()


def test_group_based_cv_split(tmp_path):
    pre_dir = tmp_path
    arrays = {
        "train_windows": np.random.randn(4, 1, 1).astype(np.float32),
        "train_demographics": np.random.randn(4, 1).astype(np.float32),
        "train_tabular": np.random.randn(4, 1).astype(np.float32),
        "train_tof_windows": np.random.randn(4, 1, 1, 1, 1).astype(np.float32),
        "train_labels": np.array([0, 1, 0, 1]),
        "train_info": [
            {"subject": 0, "sequence_id": 0},
            {"subject": 1, "sequence_id": 1},
            {"subject": 0, "sequence_id": 2},
            {"subject": 1, "sequence_id": 3},
        ],
    }
    for name, arr in arrays.items():
        with open(pre_dir / f"{name}.pkl", "wb") as f:
            pickle.dump(arr, f)

    trainer = MultimodalTrainer()
    trainer.data_dir = pre_dir
    trainer.model_dir = tmp_path
    data = trainer.load_all_data()

    splits = []

    def fake_train_fold(X_s, X_d, X_t, X_f, y, train_idx, val_idx, **kw):
        splits.append((train_idx, val_idx))
        return DummyModel(), None, 0.0

    trainer._train_fold = fake_train_fold
    trainer.train_cross_validation(data, epochs=1, batch_size=1, n_splits=2, groups=data["groups"])

    assert len(splits) == 2
    for tr_idx, val_idx in splits:
        assert len(set(data["groups"][tr_idx]).intersection(data["groups"][val_idx])) == 0
