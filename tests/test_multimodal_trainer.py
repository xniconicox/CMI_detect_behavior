import pickle
import json
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


class DummyHistory:
    def __init__(self, history: dict) -> None:
        self.history = history


def test_save_training_history(tmp_path):
    trainer = MultimodalTrainer()
    trainer.result_dir = tmp_path
    trainer.history = DummyHistory({"loss": [1.0, 0.5], "val_loss": [1.2, 0.8]})
    trainer.save_training_history()

    hist_path = tmp_path / "training_history.json"
    assert hist_path.exists()
    with open(hist_path) as f:
        data = json.load(f)
    assert isinstance(data, dict)


def test_save_evaluation_results(tmp_path):
    trainer = MultimodalTrainer()
    trainer.result_dir = tmp_path
    results = {"f1_macro": 0.9, "accuracy": 0.95}
    trainer.save_evaluation_results(results)

    eval_path = tmp_path / "evaluation_results.json"
    assert eval_path.exists()
    with open(eval_path) as f:
        data = json.load(f)
    assert isinstance(data, dict)
