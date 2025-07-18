import numpy as np
import polars as pl
from pathlib import Path
import tensorflow as tf
import logging

from src.utils.pipeline import Preprocessor
from src.utils.config_utils import load_config
from src.utils.kaggle import is_kaggle

# Logging settings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = load_config()

if is_kaggle():
    MODEL_DIR = Path("/kaggle/input/cmi-ensemble-v30/models")
    PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
else:
    MODEL_DIR = Path("models")
    PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"

# Load all fold models
model_paths = sorted(MODEL_DIR.glob("multimodal_model_v30_fold*.keras"))
models = [tf.keras.models.load_model(p) for p in model_paths]
logger.info("Loaded %d models", len(models))

# Load preprocessor
preprocessor = Preprocessor.load(PREPROCESSOR_PATH)


def predict_one(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """Perform inference on one sample using ensemble averaging."""
    logger.info("Starting inference for one sample")

    seq_df = sequence.to_pandas()
    demo_df = demographics.to_pandas()

    combined_df = seq_df.merge(demo_df, on="subject", how="left")
    data = preprocessor.transform(combined_df, use_cache=False)

    X_sensor = data["windows"]
    X_demo = data["demographics"]
    X_tabular = data["tabular"]
    X_tof = data["tof_windows"]

    logger.info(
        "Input shapes - Sensor: %s, Demo: %s, Tabular: %s, ToF: %s",
        X_sensor.shape,
        X_demo.shape,
        X_tabular.shape,
        X_tof.shape,
    )

    preds = [m.predict([X_sensor, X_demo, X_tabular, X_tof], verbose=0) for m in models]
    pred_avg = np.mean(preds, axis=0)
    label_indices = np.argmax(pred_avg, axis=1)

    if hasattr(preprocessor, "label_encoder") and preprocessor.label_encoder is not None:
        labels = preprocessor.label_encoder.inverse_transform(label_indices)
        predicted_label = labels[0] if len(labels) > 0 else "unknown"
    else:
        predicted_label = f"gesture_{label_indices[0]}" if len(label_indices) > 0 else "unknown"

    logger.info("Predicted label: %s", predicted_label)
    return predicted_label
