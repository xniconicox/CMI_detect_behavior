import numpy as np
import polars as pl
import pandas as pd
import pickle
from pathlib import Path
import tensorflow as tf
import logging

from src.utils.pipeline import Preprocessor
from src.trainers.multimodal_trainer import MultimodalTrainer
from src.utils.config_utils import load_config
from src.utils.kaggle import is_kaggle

# Logging settings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = load_config()

if is_kaggle():
    MODEL_DIR = Path("/kaggle/input/cmi-ensemble-v1/models")
    MODEL_PATH = MODEL_DIR / "multimodal_model.keras"
    PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
else:
    MODEL_DIR = Path("models")
    MODEL_PATH = MODEL_DIR / "multimodal_model.keras"
    PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"

# 1. Preprocessor and model loading (global once)
logger.info("Loading preprocessor and model...")
preprocessor = Preprocessor.load(PREPROCESSOR_PATH)
trainer = MultimodalTrainer()
trainer.model = tf.keras.models.load_model(MODEL_PATH)

# Check label mapping
if hasattr(preprocessor, 'label_encoder') and preprocessor.label_encoder is not None:
    label_mapping = dict(zip(preprocessor.label_encoder.classes_, range(len(preprocessor.label_encoder.classes_))))
    logger.info(f"Label mapping: {label_mapping}")
    logger.info(f"Number of classes: {len(preprocessor.label_encoder.classes_)}")
else:
    logger.warning("No label encoder found in preprocessor")

logger.info("Model and preprocessor loaded successfully")

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
    data = preprocessor.transform(combined_df, use_cache=False)  # Do not use cache
    
    # Extract inputs for batch inference of all test data
    X_sensor = data["windows"]
    X_demo = data["demographics"] 
    X_tabular = data["tabular"]
    X_tof = data["tof_windows"]  # Use tof_windows instead
    
    logger.info(f"Input shapes - Sensor: {X_sensor.shape}, Demo: {X_demo.shape}, "
                f"Tabular: {X_tabular.shape}, ToF: {X_tof.shape}")
    
    # Model inference (batch)
    pred = trainer.model.predict([X_sensor, X_demo, X_tabular, X_tof], verbose=0)
    
    # Convert prediction to label (all samples)
    label_indices = np.argmax(pred, axis=1)
    
    # Convert prediction to string label (CMIInferenceServer expects string)
    logger.info(f"Predicted {len(label_indices)} labels")
    
    # Convert label indices to string labels using label encoder
    if hasattr(preprocessor, 'label_encoder') and preprocessor.label_encoder is not None:
        labels = preprocessor.label_encoder.inverse_transform(label_indices)
        predicted_label = labels[0] if len(labels) > 0 else "unknown"
    else:
        # If no label encoder, return the first label index as string
        predicted_label = f"gesture_{label_indices[0]}" if len(label_indices) > 0 else "unknown"
    
    logger.info(f"Predicted label: {predicted_label}")
    return predicted_label