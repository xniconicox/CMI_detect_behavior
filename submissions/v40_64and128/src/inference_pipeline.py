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

# Logging settings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Model and Preprocessor Loading ---
def load_models_and_preprocessors():
    """Load all models and preprocessors for ensemble."""
    
    if is_kaggle():
        # Kaggle environment paths
        base_path_64 = Path("/kaggle/input/cmi-v40-ws64")
        base_path_128 = Path("/kaggle/input/cmi-v40-ws128")
        models_dir_64 = base_path_64 / "models_64"
        preprocessor_path_64 = base_path_64 / "preprocessor_64.pkl"
        config_path_64 = base_path_64 / "config_ws64.yaml"
        models_dir_128 = base_path_128 / "models_128"
        preprocessor_path_128 = base_path_128 / "preprocessor_128.pkl"
        config_path_128 = base_path_128 / "config_ws128.yaml"
    else:
        # Local environment paths
        models_dir_64 = Path("models_64")
        preprocessor_path_64 = Path("preprocessor_64.pkl")
        config_path_64 = Path("config_ws64.yaml")
        models_dir_128 = Path("models_128")
        preprocessor_path_128 = Path("preprocessor_128.pkl")
        config_path_128 = Path("config_ws128.yaml")

    # Load configs
    logger.info("Loading configurations...")
    config_64 = load_config(config_path_64) if config_path_64.exists() else None
    config_128 = load_config(config_path_128) if config_path_128.exists() else None
    
    # Load preprocessors
    logger.info("Loading preprocessors...")
    preprocessor_64 = Preprocessor.load(preprocessor_path_64)
    preprocessor_128 = Preprocessor.load(preprocessor_path_128)
    
    # Update preprocessor configs if available
    if config_64:
        logger.info("Applying ws64 config to preprocessor_64")
        preprocessor_64.config.update(config_64)
    if config_128:
        logger.info("Applying ws128 config to preprocessor_128")
        preprocessor_128.config.update(config_128)
    
    logger.info("Preprocessors loaded successfully.")
    
    # Load models
    models_64 = []
    models_128 = []

    logger.info("Loading ws64 models...")
    for fold in range(1, 6):
        model_path = models_dir_64 / f"multimodal_model_v40_fold{fold}.keras"
        if model_path.exists():
            models_64.append(tf.keras.models.load_model(model_path))
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")

    logger.info("Loading ws128 models...")
    for fold in range(1, 6):
        model_path = models_dir_128 / f"multimodal_model_v40_fold{fold}.keras"
        if model_path.exists():
            models_128.append(tf.keras.models.load_model(model_path))
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")

    if not models_64 or not models_128:
        raise ValueError("Models for ws64 or ws128 could not be loaded.")

    logger.info(f"Loaded {len(models_64)} ws64 models and {len(models_128)} ws128 models.")
    
    return preprocessor_64, models_64, preprocessor_128, models_128

preprocessor_64, models_64, preprocessor_128, models_128 = load_models_and_preprocessors()

# --- Inference Function ---
def predict_one(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """
    Perform ensemble inference on one sample using ws64 and ws128 models.
    """
    logger.info("Starting ensemble inference for one sample.")
    
    # Convert polars to pandas
    seq_df = sequence.to_pandas()
    demo_df = demographics.to_pandas()
    combined_df = seq_df.merge(demo_df, on="subject", how="left")
    
    all_final_preds = []

    # --- Inference with ws64 ---
    logger.info("Processing with ws64 preprocessor...")
    try:
        data_64 = preprocessor_64.transform(combined_df.copy(), use_cache=False)
        inputs_64 = [
            data_64["windows"],
            data_64["demographics"],
            data_64["tabular"],
            data_64["tof_windows"],
        ]
        
        preds_64 = [model.predict(inputs_64, verbose=0).mean(axis=0) for model in models_64]
        avg_pred_64 = np.mean(preds_64, axis=0)
        all_final_preds.append(avg_pred_64)
        logger.info(f"ws64 prediction completed. Shape: {avg_pred_64.shape}")
    except Exception as e:
        logger.error(f"Error in ws64 processing: {e}")
        raise

    # --- Inference with ws128 ---
    logger.info("Processing with ws128 preprocessor...")
    try:
        data_128 = preprocessor_128.transform(combined_df.copy(), use_cache=False)
        inputs_128 = [
            data_128["windows"],
            data_128["demographics"],
            data_128["tabular"],
            data_128["tof_windows"],
        ]

        preds_128 = [model.predict(inputs_128, verbose=0).mean(axis=0) for model in models_128]
        avg_pred_128 = np.mean(preds_128, axis=0)
        all_final_preds.append(avg_pred_128)
        logger.info(f"ws128 prediction completed. Shape: {avg_pred_128.shape}")
    except Exception as e:
        logger.error(f"Error in ws128 processing: {e}")
        raise

    # --- Ensemble predictions ---
    final_avg_pred = np.mean(all_final_preds, axis=0)
    logger.info(f"Ensemble prediction completed. Shape: {final_avg_pred.shape}")
    
    # Convert prediction to label
    label_index = np.argmax(final_avg_pred)
    
    # Use the label encoder from one of the preprocessors (they should be identical)
    if hasattr(preprocessor_64, 'label_encoder') and preprocessor_64.label_encoder is not None:
        predicted_label = preprocessor_64.label_encoder.inverse_transform([label_index])[0]
    else:
        predicted_label = f"gesture_{label_index}"
        
    logger.info(f"Final predicted label: {predicted_label}")
    return predicted_label 