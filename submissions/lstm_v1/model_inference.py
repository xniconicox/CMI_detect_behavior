"""
LSTM Model Inference for CMI Competition
This module contains the LSTM model loading and inference functions
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# Disable TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Global variables for model and preprocessors
global_model = None
global_scaler = None
global_le = None
global_config = None

def create_sliding_windows(sequence_df, window_size=64, stride=16):
    """
    Create sliding windows from sensor data sequence
    
    Args:
        sequence_df: DataFrame containing sensor data sequence
        window_size: Size of each window
        stride: Step size between windows
        
    Returns:
        np.array: Array of windowed sequences
    """
    # Identify sensor columns
    sensor_cols = [col for col in sequence_df.columns 
                   if any(sensor in col for sensor in ['acc_', 'rot_', 'tof_', 'thm_'])]
    
    # Extract sensor data
    sensor_data = sequence_df[sensor_cols].values
    
    # Handle missing values
    sensor_data = np.nan_to_num(sensor_data, nan=0.0)
    
    # Create sliding windows
    windows = []
    for i in range(0, len(sensor_data) - window_size + 1, stride):
        window = sensor_data[i:i + window_size]
        windows.append(window)
    
    if len(windows) == 0:
        # If sequence is shorter than window_size, pad with zeros
        padded_sequence = np.zeros((window_size, len(sensor_cols)))
        padded_sequence[:len(sensor_data)] = sensor_data
        windows.append(padded_sequence)
    
    return np.array(windows)

def load_saved_model():
    """
    Load pre-trained LSTM model and preprocessors for CMI Competition
    """
    global global_model, global_scaler, global_le, global_config

    # Kaggle環境とローカル環境のパス分岐
    if os.path.exists('/kaggle/input/cmi-lstm-models-v1/'):
        model_dir = '/kaggle/input/cmi-lstm-models-v1/cmi-lstm-models-v1/models'
    else:
        model_dir = '../../output/experiments/lstm_w64_s16_final_model/models/'

    try:

        # Load model configuration
        with open(os.path.join(model_dir, 'config.json'), 'r') as f:
            global_config = json.load(f)
        
        # Load LSTM model
        model_path = os.path.join(model_dir, 'lstm_best.h5')
        global_model = tf.keras.models.load_model(model_path)  # type: ignore
        
        # Load scaler
        with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
            global_scaler = pickle.load(f)
        
        # Load label encoder
        with open(os.path.join(model_dir, 'label_encoder.pkl'), 'rb') as f:
            global_le = pickle.load(f)

        print("Pre-trained LSTM model loaded successfully")
        print(f"Model input shape: {global_model.input_shape}")
        print(f"Model output shape: {global_model.output_shape}")
        print(f"Number of classes: {len(global_le.classes_)}")
        print(f"Window size: {global_config.get('window_size', 64)}")

    except FileNotFoundError as e:
        print(f"Model files not found: {e}")
        print("Please ensure model files are properly added as Kaggle dataset")
        raise e
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

def predict_gesture(sequence_df):
    """
    Predict gesture from sensor data sequence using LSTM model
    
    Args:
        sequence_df: pandas DataFrame containing sensor data sequence
        
    Returns:
        str: Predicted gesture name
    """
    global global_model, global_scaler, global_le, global_config
    
    # Load model if not already loaded
    if global_model is None:
        load_saved_model()
    
    # Verify models are loaded correctly
    if global_model is None or global_scaler is None or global_le is None:
        raise ValueError("Failed to load model or preprocessors")
    
    # Get window parameters
    window_size = global_config.get('window_size', 64)
    stride = global_config.get('stride', 16)
    
    # Create sliding windows
    windows = create_sliding_windows(sequence_df, window_size, stride)
    
    if len(windows) == 0:
        raise ValueError("No valid windows created from sequence")
    
    # Reshape for model input: (n_windows, window_size, n_features)
    n_windows, window_len, n_features = windows.shape
    
    # Normalize features
    windows_reshaped = windows.reshape(-1, n_features)
    windows_normalized = global_scaler.transform(windows_reshaped)
    windows_normalized = windows_normalized.reshape(n_windows, window_len, n_features)
    
    # Make predictions for all windows
    predictions = global_model.predict(windows_normalized, verbose=0)
    
    # Average predictions across all windows
    avg_prediction = np.mean(predictions, axis=0)
    
    # Get predicted class
    predicted_class = np.argmax(avg_prediction)
    
    # Convert to gesture name
    predicted_gesture = global_le.inverse_transform([predicted_class])[0]
    
    return predicted_gesture

def get_model_info():
    """
    Get information about the loaded model
    
    Returns:
        dict: Model information
    """
    global global_model, global_config, global_le
    
    if global_model is None:
        load_saved_model()
    
    return {
        'model_type': 'LSTM',
        'window_size': global_config.get('window_size', 64),
        'stride': global_config.get('stride', 16),
        'n_classes': len(global_le.classes_),
        'classes': global_le.classes_.tolist(),
        'input_shape': global_model.input_shape,
        'output_shape': global_model.output_shape,
        'total_params': global_model.count_params()
    } 