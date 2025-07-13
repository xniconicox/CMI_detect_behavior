import pickle
from pathlib import Path as _Path
import numpy as np
import sys
import types

sys.modules.setdefault("tensorflow", types.ModuleType("tensorflow"))

ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.trainers.multimodal_trainer import MultimodalTrainer

def test_multimodal_trainer_load_all_data(tmp_path):
    pre_dir = tmp_path
    arrays = {
        "train_windows": np.random.randn(3, 4, 5).astype(np.float32),
        "train_demographics": np.random.randn(3, 2).astype(np.float32),
        "train_tabular": np.random.randn(3, 6).astype(np.float32),
        "train_tof_windows": np.random.randn(3, 4, 2, 2, 2).astype(np.float32),
        "train_labels": np.array([0, 1, 0]),
    }
    for name, arr in arrays.items():
        with open(pre_dir / f"{name}.pkl", "wb") as f:
            pickle.dump(arr, f)
    trainer = MultimodalTrainer()
    trainer.data_dir = pre_dir
    data = trainer.load_all_data()
    assert data["sensor"].shape == (3, 4, 5)
    assert data["demographics"].shape == (3, 2)
    assert data["tabular"].shape == (3, 6)
    assert data["tof"].shape == (3, 4, 2, 2, 2)
    assert data["labels"].shape == (3,)
