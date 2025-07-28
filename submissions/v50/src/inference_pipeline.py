import sys
from pathlib import Path
import numpy as np
import polars as pl
import pandas as pd
import tensorflow as tf
import logging

# submission src path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.pipeline import Preprocessor
from utils.config_utils import load_config
from utils.kaggle import is_kaggle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_preprocessor_and_models():
    """Load v50 preprocessor and models."""
    if is_kaggle():
        base = Path("/kaggle/input/cmi-v50")
        model_dir = base / "models"
        preproc_path = base / "preprocessor_v50.pkl"
        config_path = base / "config_v50.yaml"
    else:
        model_dir = Path("models")
        preproc_path = Path("preprocessor_v50.pkl")
        config_path = Path("config_v50.yaml")

    cfg = load_config(config_path) if config_path.exists() else None
    pre = Preprocessor.load(preproc_path)
    if cfg:
        pre.config.update(cfg)

    models = []
    for fold in range(1, 6):
        mpath = model_dir / f"multimodal_model_v50_fold{fold}.keras"
        if mpath.exists():
            models.append(tf.keras.models.load_model(mpath))
    if not models:
        raise FileNotFoundError("No v50 models found")
    logger.info("Loaded %d models", len(models))
    return pre, models


preprocessor, models = _load_preprocessor_and_models()


def _pad_with_mask(arr: np.ndarray, max_len: int, value: float = 0.0):
    """Pad 2D array to max_len and return mask."""
    seq_len = arr.shape[0]
    if seq_len >= max_len:
        padded = arr[:max_len]
        mask = np.ones(max_len, dtype=np.float32)
    else:
        pad_len = max_len - seq_len
        padded = np.pad(arr, ((0, pad_len), (0, 0)), constant_values=value)
        mask = np.concatenate([np.ones(seq_len, dtype=np.float32), np.zeros(pad_len, dtype=np.float32)])
    return padded, mask


def predict_one(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """Run inference on a single sequence."""
    seq_df = sequence.to_pandas()
    demo_df = demographics.to_pandas()
    df = seq_df.merge(demo_df, on="subject", how="left")

    seq_len = len(df)
    # temporarily adjust window builders to use full sequence
    orig_ws = preprocessor.win_builder.window_size
    orig_st = preprocessor.win_builder.stride
    orig_tws = preprocessor.tof_win_builder.window_size
    orig_tst = preprocessor.tof_win_builder.stride

    preprocessor.win_builder.window_size = seq_len
    preprocessor.win_builder.stride = seq_len
    preprocessor.tof_win_builder.window_size = seq_len
    preprocessor.tof_win_builder.stride = seq_len

    data = preprocessor.transform(df, use_cache=False)

    # restore settings
    preprocessor.win_builder.window_size = orig_ws
    preprocessor.win_builder.stride = orig_st
    preprocessor.tof_win_builder.window_size = orig_tws
    preprocessor.tof_win_builder.stride = orig_tst

    X_sensor = data["windows"][0]
    X_demo = data["demographics"][0]
    X_tab = data["tabular"][0]
    X_tof = data["tof_windows"][0]

    max_len = preprocessor.config.get("preprocessing", {}).get("max_sequence_length", X_sensor.shape[0])
    X_sensor_pad, mask = _pad_with_mask(X_sensor, max_len, preprocessor.win_builder.padding_value)
    X_tof_pad, _ = _pad_with_mask(X_tof, max_len, preprocessor.tof_win_builder.fill_value)

    inputs = [
        np.expand_dims(X_sensor_pad, 0),
        np.expand_dims(X_demo, 0),
        np.expand_dims(X_tab, 0),
        np.expand_dims(X_tof_pad, 0),
        np.expand_dims(mask, 0),
    ]

    preds = [model.predict(inputs, verbose=0)[0] for model in models]
    avg_pred = np.mean(preds, axis=0)

    label_idx = int(np.argmax(avg_pred))
    if hasattr(preprocessor, "label_encoder") and preprocessor.label_encoder is not None:
        return preprocessor.label_encoder.inverse_transform([label_idx])[0]
    return f"gesture_{label_idx}"
