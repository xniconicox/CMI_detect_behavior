"""
Core inference logic for CMI Competition
This module contains the model loading and inference functions
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings
warnings.filterwarnings('ignore')

# Global variables for models and encoders
global_models = None
global_le = None
global_feature_cols = None

def extract_features(df):
    """
    Extract statistical features from sensor data sequence
    
    Args:
        df: DataFrame containing sensor data sequence
        
    Returns:
        dict: Dictionary of extracted features
    """
    # Identify sensor columns
    sensor_cols = [col for col in df.columns if any(sensor in col for sensor in ['acc_', 'rot_', 'tof_', 'thm_'])]
    
    # Calculate statistics for each sensor column
    features = {}
    
    for col in sensor_cols:
        if df[col].dtype in ['float64', 'int64']:
            # Basic statistics
            features[f'{col}_mean'] = df[col].mean()
            features[f'{col}_std'] = df[col].std()
            features[f'{col}_min'] = df[col].min()
            features[f'{col}_max'] = df[col].max()
            features[f'{col}_median'] = df[col].median()
            
            # Quantiles
            features[f'{col}_q25'] = df[col].quantile(0.25)
            features[f'{col}_q75'] = df[col].quantile(0.75)
            
            # Range
            features[f'{col}_range'] = df[col].max() - df[col].min()
            
            # Skewness and kurtosis
            features[f'{col}_skew'] = df[col].skew()
            features[f'{col}_kurtosis'] = df[col].kurtosis()
    
    return features

def load_saved_model():
    """
    Load pre-trained models and encoders for CMI Competition
    """
    global global_models, global_le, global_feature_cols

    # Kaggle環境とローカル環境のパス分岐
    if os.path.exists('/kaggle/input/cmi-baseline-models-v1/'):
        model_dir = '/kaggle/input/cmi-baseline-models-v1/cmi-baseline-models/'
    else:
        model_dir = '../../output/experiments/baseline_lightgbm_v1/models/'


    try:
        with open(os.path.join(model_dir, 'trained_models.pkl'), 'rb') as f:
            global_models = pickle.load(f)

        with open(os.path.join(model_dir, 'label_encoder.pkl'), 'rb') as f:
            global_le = pickle.load(f)

        feature_cols_file = os.path.join(model_dir, 'feature_cols.pkl')
        if os.path.exists(feature_cols_file):
            with open(feature_cols_file, 'rb') as f:
                global_feature_cols = pickle.load(f)
        else:
            print("Feature columns file not found. Using dynamic feature construction...")
            global_feature_cols = None

        print("Pre-trained models loaded successfully")
        print(f"Number of models: {len(global_models)}")
        print(f"Number of classes: {len(global_le.classes_)}")
        if global_feature_cols:
            print(f"Number of features: {len(global_feature_cols)}")

    except FileNotFoundError as e:
        print(f"Model files not found: {e}")
        print("Please ensure model files are properly added as Kaggle dataset")
        raise e


def predict_gesture(sequence_df):
    """
    Predict gesture from sensor data sequence
    
    Args:
        sequence_df: pandas DataFrame containing sensor data sequence
        
    Returns:
        str: Predicted gesture name
    """
    global global_models, global_le, global_feature_cols
    
    # Load models if not already loaded
    if global_models is None:
        load_saved_model()
    
    # Verify models are loaded correctly
    if global_models is None or global_le is None:
        raise ValueError("Failed to load models")
    
    # Extract features from sequence
    features = extract_features(sequence_df)
    
    # Convert to DataFrame
    feature_df = pd.DataFrame([features])
    
    # Handle dynamic feature construction if needed
    if global_feature_cols is None:
        global_feature_cols = list(features.keys())
        print(f"Feature columns constructed dynamically: {len(global_feature_cols)} features")
    
    # Select required feature columns
    feature_df = feature_df[global_feature_cols]
    
    # Make predictions using ensemble
    predictions = []
    for model in global_models:
        pred = model.predict(feature_df, num_iteration=model.best_iteration)
        predictions.append(pred)
    
    # Average predictions from all models
    avg_pred = np.mean(predictions, axis=0)
    
    # Handle 2D array case (fix for LightGBM prediction shape)
    if avg_pred.ndim == 2 and avg_pred.shape[0] == 1:
        avg_pred = avg_pred.flatten()
    
    # Get predicted class
    predicted_class = np.argmax(avg_pred)
    
    # Convert to gesture name
    predicted_gesture = global_le.inverse_transform([predicted_class])[0]
    
    return predicted_gesture 