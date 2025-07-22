import sys
from pathlib import Path

# Add the submission-specific src directory to sys.path
# This allows importing modules from the src directory
# when running from the submission directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import polars as pl
import pandas as pd
import pickle
import tensorflow as tf
import logging

from utils.pipeline import Preprocessor
from trainers.multimodal_trainer_v40 import MultimodalTrainerV40
from utils.kaggle import is_kaggle

# Logging settings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if is_kaggle():
    # On Kaggle, the models and preprocessor are in the input directory
    MODEL_DIR = Path("/kaggle/input/cmi-v40/models")
    PREPROCESSOR_PATH = Path("/kaggle/input/cmi-v40/preprocessed/preprocessor.pkl")
else:
    # Local execution: models are in the models subdirectory
    MODEL_DIR = Path("models")
    PREPROCESSOR_PATH = Path("preprocessed/preprocessor.pkl")

# 1. Preprocessor and model loading
logger.info("Loading preprocessor and models...")
preprocessor = Preprocessor.load(PREPROCESSOR_PATH)

# Load 5-fold models
models = []
for fold in range(1, 6):
    model_path = MODEL_DIR / f"multimodal_model_v31_fold{fold}.keras"
    if model_path.exists():
        logger.info(f"Loading model: {model_path}")
        model = tf.keras.models.load_model(model_path)
        models.append(model)
    else:
        logger.error(f"Model not found: {model_path}")
        raise FileNotFoundError(f"Model not found: {model_path}")

if not models:
    raise ValueError("No models were loaded. Check model paths.")

logger.info(f"Loaded {len(models)} models successfully.")

# Check label mapping
if hasattr(preprocessor, 'label_encoder') and preprocessor.label_encoder is not None:
    label_mapping = dict(zip(preprocessor.label_encoder.classes_, range(len(preprocessor.label_encoder.classes_))))
    logger.info(f"Label mapping: {label_mapping}")
    logger.info(f"Number of classes: {len(preprocessor.label_encoder.classes_)}")
else:
    logger.warning("No label encoder found in preprocessor")


# 2. 1 sample inference function
def predict_one(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """
    Perform inference on one sample from raw data
    
    Args:
        sequence: Sequence data (polars DataFrame)
        demographics: Demographic data (polars DataFrame)
    
    Returns:
        str: Predicted gesture label
    """
    logger.info("Starting inference for one sample")
    
    # Convert polars to pandas
    seq_df = sequence.to_pandas()
    demo_df = demographics.to_pandas()
    
    # Merge demographic data into sequence data (subject is the common column)
    combined_df = seq_df.merge(demo_df, on="subject", how="left")
    logger.info(f"Combined data shape: {combined_df.shape}")
    
    # Apply the same preprocessing pipeline as during training
    data = preprocessor.transform(combined_df, use_cache=False)
    
    # Extract inputs for batch inference
    X_sensor = data["windows"]
    X_demo = data["demographics"] 
    X_tabular = data["tabular"]
    X_tof = data["tof_windows"]
    
    logger.info(f"Input shapes - Sensor: {X_sensor.shape}, Demo: {X_demo.shape}, "
                f"Tabular: {X_tabular.shape}, ToF: {X_tof.shape}")
    
    # Model inference (ensemble of 5 folds)
    all_preds = []
    for model in models:
        pred = model.predict([X_sensor, X_demo, X_tabular, X_tof], verbose=0)
        all_preds.append(pred)
    
    # Average predictions
    avg_pred = np.mean(all_preds, axis=0)
    
    # Convert prediction to label
    label_indices = np.argmax(avg_pred, axis=1)
    
    # Convert prediction to string label
    logger.info(f"Predicted {len(label_indices)} labels")
    
    if hasattr(preprocessor, 'label_encoder') and preprocessor.label_encoder is not None:
        labels = preprocessor.label_encoder.inverse_transform(label_indices)
        predicted_label = labels[0] if len(labels) > 0 else "unknown"
    else:
        predicted_label = f"gesture_{label_indices[0]}" if len(label_indices) > 0 else "unknown"
    
    logger.info(f"Predicted label: {predicted_label}")
    return predicted_label
