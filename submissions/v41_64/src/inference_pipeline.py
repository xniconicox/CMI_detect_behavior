import sys
from pathlib import Path
import numpy as np
import polars as pl
import pandas as pd
import tensorflow as tf
import logging

# Add the submission-specific src directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.pipeline import Preprocessor
from utils.config_utils import load_config
from utils.kaggle import is_kaggle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_and_preprocessor():
    """Load ws64 model and preprocessor."""
    if is_kaggle():
        base_path = Path("/kaggle/input/cmi-v41-ws64")
        models_dir = base_path / "models_64"
        preprocessor_path = base_path / "preprocessor_64.pkl"
        config_path = base_path / "config_v40_ws64.yaml"
    else:
        models_dir = Path("models_64")
        preprocessor_path = Path("preprocessor_64.pkl")
        config_path = Path("config_v40_ws64.yaml")

    logger.info("Loading configuration...")
    config = load_config(config_path) if config_path.exists() else None

    logger.info("Loading preprocessor...")
    preprocessor = Preprocessor.load(preprocessor_path)
    if config:
        preprocessor.config.update(config)

    logger.info("Loading models...")
    models = []
    for fold in range(1, 6):
        model_path = models_dir / f"multimodal_model_v41_fold{fold}.keras"
        if model_path.exists():
            models.append(tf.keras.models.load_model(model_path))
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")
    logger.info(f"Loaded {len(models)} ws64 models.")
    return preprocessor, models


preprocessor, models = load_model_and_preprocessor()


def predict_one(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """Run inference using ws64 model set."""
    seq_df = sequence.to_pandas()
    demo_df = demographics.to_pandas()
    combined_df = seq_df.merge(demo_df, on="subject", how="left")
    print("DEBUG: combined_df columns", combined_df.columns.tolist())
    data = preprocessor.transform(combined_df.copy(), use_cache=False)
    inputs = [
        data["windows"],
        data["demographics"],
        data["tabular"],
        data["tof_windows"],
    ]

    preds = [model.predict(inputs, verbose=0) for model in models]
    avg_pred = np.mean(preds, axis=0)

    label_index = np.argmax(avg_pred, axis=1)[0]
    if hasattr(preprocessor, "label_encoder") and preprocessor.label_encoder is not None:
        return preprocessor.label_encoder.inverse_transform([label_index])[0]
    return f"gesture_{label_index}"
