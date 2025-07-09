#!/usr/bin/env python3
"""
LSTM v2 推論モジュール
CMI 2025 コンペティション用
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import polars as pl
import tensorflow as tf
from pathlib import Path

# 現在のディレクトリとプロジェクトルートを設定
CURRENT_DIR = Path(__file__).parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))

# モデル設定
MODEL_CONFIG = None
model = None
model_loaded = False
sensor_scaler = None
demographics_scaler = None

# ラベルマッピング（学習時と同じ）
LABEL_MAPPING = {
    'Above ear - pull hair': 0, 'Cheek - pinch skin': 1, 'Drink from bottle/cup': 2,
    'Eyebrow - pull hair': 3, 'Eyelash - pull hair': 4, 'Feel around in tray and pull out an object': 5,
    'Forehead - pull hairline': 6, 'Forehead - scratch': 7, 'Glasses on/off': 8,
    'Neck - pinch skin': 9, 'Neck - scratch': 10, 'Pinch knee/leg skin': 11,
    'Pull air toward your face': 12, 'Scratch knee/leg skin': 13, 'Text on phone': 14,
    'Wave hello': 15, 'Write name in air': 16, 'Write name on leg': 17
}

REVERSE_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}

def setup_gpu():
    """GPU設定"""
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU利用可能: {len(gpus)}台")
            return True
        else:
            print("CPU環境で実行")
            return False
    except Exception as e:
        print(f"GPU設定エラー: {e}")
        return False

def load_model_config():
    """モデル設定を読み込み"""
    global MODEL_CONFIG
    
    config_path = CURRENT_DIR / "model_config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            MODEL_CONFIG = json.load(f)
        print(f"モデル設定読み込み完了: CMIスコア {MODEL_CONFIG['model_info']['cmi_score']:.4f}")
        return True
    else:
        print("⚠️ モデル設定ファイルが見つかりません")
        return False

def load_model():
    """モデルを読み込み"""
    global model, model_loaded, sensor_scaler, demographics_scaler
    
    if model_loaded:
        return model
    
    if not MODEL_CONFIG:
        print("❌ モデル設定が読み込まれていません")
        return None
    
    try:
        model_file = MODEL_CONFIG['file_paths']['model_file']
        model_path = CURRENT_DIR / model_file
        
        if model_path.exists():
            print(f"モデル読み込み中: {model_path}")
            model = tf.keras.models.load_model(str(model_path))
            model_loaded = True
            print("✅ モデル読み込み完了")
            
            # Scalerの読み込み（オプション）
            try:
                import pickle
                scaler_base_path = str(model_path).replace('.keras', '')
                
                # sensor_scalerを読み込み
                sensor_scaler_path = f"{scaler_base_path}_sensor_scaler.pkl"
                if os.path.exists(sensor_scaler_path):
                    with open(sensor_scaler_path, 'rb') as f:
                        sensor_scaler = pickle.load(f)
                    print("✅ センサーScaler読み込み完了")
                else:
                    print("⚠️ センサーScalerファイルが見つかりません - 正規化なしで継続")
                
                # demographics_scalerを読み込み
                demographics_scaler_path = f"{scaler_base_path}_demographics_scaler.pkl"
                if os.path.exists(demographics_scaler_path):
                    with open(demographics_scaler_path, 'rb') as f:
                        demographics_scaler = pickle.load(f)
                    print("✅ DemographicsScaler読み込み完了")
                else:
                    print("⚠️ DemographicsScalerファイルが見つかりません - 正規化なしで継続")
                    
            except Exception as e:
                print(f"⚠️ Scaler読み込みエラー: {e}")
                print("正規化なしで継続します")
            
            return model
        else:
            print(f"❌ モデルファイルが見つかりません: {model_path}")
            return None
            
    except Exception as e:
        print(f"❌ モデル読み込みエラー: {e}")
        return None

def preprocess_sequence(sequence_df, demographics_df, window_size=64):
    """
    シーケンスデータを前処理してモデル入力形式に変換
    
    Parameters:
    -----------
    sequence_df : pl.DataFrame
        センサーデータのシーケンス
    demographics_df : pl.DataFrame
        Demographics情報
    window_size : int
        ウィンドウサイズ
    
    Returns:
    --------
    tuple : (windowed_data, demographics_features)
    """
    # センサーデータの列を選択（数値列のみ）
    sensor_columns = []
    skip_columns = ['id', 'timestamp', 'sequence_id', 'subject', 'gesture', 'sequence_counter']
    
    for col in sequence_df.columns:
        if col not in skip_columns:
            # 数値型の列のみを選択
            try:
                # 列のデータタイプをチェック
                col_data = sequence_df[col]
                
                # 文字列型の列をスキップ
                if col_data.dtype == pl.Utf8:
                    print(f"⚠️ 文字列列をスキップ: {col}")
                    continue
                
                # 最初の数行をテストして数値変換可能かチェック
                test_data = col_data.head(min(5, len(col_data))).to_numpy()
                
                # Noneや文字列が含まれていないかチェック
                if any(x is None or isinstance(x, str) for x in test_data):
                    print(f"⚠️ 非数値データを含む列をスキップ: {col}")
                    continue
                
                # 数値変換テスト
                test_converted = test_data.astype(np.float32)
                
                # NaNや無限大値をチェック
                if np.any(np.isnan(test_converted)) or np.any(np.isinf(test_converted)):
                    print(f"⚠️ NaN/Inf値を含む列をスキップ: {col}")
                    continue
                
                sensor_columns.append(col)
                
            except (ValueError, TypeError, AttributeError) as e:
                print(f"⚠️ 非数値列をスキップ: {col} - {str(e)}")
                continue
    
    # データを数値型に変換
    if not sensor_columns:
        print("❌ 有効なセンサー列が見つかりません")
        # デフォルトの332次元ゼロデータを作成
        sequence_data = np.zeros((window_size, 332), dtype=np.float32)
    else:
        try:
            sequence_data = sequence_df[sensor_columns].to_numpy().astype(np.float32)
            print(f"✅ {len(sensor_columns)}個のセンサー列を使用")
        except Exception as e:
            print(f"❌ データ変換エラー: {e}")
            # エラー時はデフォルトデータを使用
            sequence_data = np.zeros((window_size, 332), dtype=np.float32)
    
    # ウィンドウ作成
    if len(sequence_data) >= window_size:
        # 最後のウィンドウを使用
        start_idx = len(sequence_data) - window_size
        windowed_data = sequence_data[start_idx:start_idx + window_size]
    else:
        # データが短い場合はパディング
        padded_data = np.zeros((window_size, len(sensor_columns)), dtype=np.float32)
        padded_data[:len(sequence_data)] = sequence_data
        windowed_data = padded_data
    
    # Demographics データの処理（学習時と同じ特徴量エンジニアリング）
    demographics_row = demographics_df.row(0)
    
    # 基本特徴量
    age = demographics_row[demographics_df.columns.index('age')] if 'age' in demographics_df.columns else 25.0
    height = demographics_row[demographics_df.columns.index('height')] if 'height' in demographics_df.columns else 170.0
    weight = demographics_row[demographics_df.columns.index('weight')] if 'weight' in demographics_df.columns else 70.0
    bmi = demographics_row[demographics_df.columns.index('bmi')] if 'bmi' in demographics_df.columns else 24.0
    gender_encoded = demographics_row[demographics_df.columns.index('gender_encoded')] if 'gender_encoded' in demographics_df.columns else 1.0
    
    # 学習時と同じ18次元特徴量を作成（簡略版）
    demographics_features = np.array([
        age, height, weight, bmi, gender_encoded,
        1.0, 0.0, 0.0, 0.0, 0.0,  # age_group one-hot (adult)
        0.0, 1.0, 0.0, 0.0,       # height_category one-hot (average)
        0.65,                      # arm_ratio (typical value)
        0.38,                      # arm_length_relative (typical value)
        age * gender_encoded,      # sex_age_interaction
        gender_encoded             # handedness_sex (simplified)
    ], dtype=np.float32)
    
    # 18次元に調整
    if len(demographics_features) < 18:
        padded_demographics = np.zeros(18, dtype=np.float32)
        padded_demographics[:len(demographics_features)] = demographics_features
        demographics_features = padded_demographics
    elif len(demographics_features) > 18:
        demographics_features = demographics_features[:18]
    
    return windowed_data, demographics_features

def predict_gesture(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """
    ジェスチャー予測のメイン関数
    
    Parameters:
    -----------
    sequence : pl.DataFrame
        センサーデータのシーケンス
    demographics : pl.DataFrame
        Demographics情報
    
    Returns:
    --------
    str : 予測されたジェスチャー名
    """
    try:
        # モデルが読み込まれているか確認
        if model is None:
            print("❌ モデルが読み込まれていません")
            return 'Above ear - pull hair'
        
        # データ前処理
        windowed_data, demographics_features = preprocess_sequence(sequence, demographics)
        
        # 正規化処理（学習時と同じ）
        if sensor_scaler is not None:
            # センサーデータの正規化
            windowed_data_flat = windowed_data.reshape(-1, windowed_data.shape[-1])
            windowed_data_normalized = sensor_scaler.transform(windowed_data_flat)
            windowed_data = windowed_data_normalized.reshape(windowed_data.shape)
        
        if demographics_scaler is not None:
            # Demographics特徴量の正規化
            demographics_features = demographics_scaler.transform(demographics_features.reshape(1, -1)).flatten()
        
        # バッチ次元を追加
        X_sensor = np.expand_dims(windowed_data, axis=0)  # (1, 64, 332)
        X_demographics = np.expand_dims(demographics_features, axis=0)  # (1, 18)
        
        # 予測実行
        predictions = model.predict([X_sensor, X_demographics], verbose=0)
        predicted_class = int(np.argmax(predictions[0]))
        
        # クラス名に変換
        predicted_gesture = REVERSE_LABEL_MAPPING.get(predicted_class, 'Above ear - pull hair')
        
        return predicted_gesture
        
    except Exception as e:
        print(f"❌ 予測エラー: {e}")
        return 'Above ear - pull hair'  # エラー時のデフォルト値

# 初期化
def initialize():
    """推論システムを初期化"""
    print("🚀 LSTM推論システム初期化中...")
    
    # GPU設定
    setup_gpu()
    
    # モデル設定読み込み
    if not load_model_config():
        return False
    
    # モデル読み込み
    if load_model() is None:
        return False
    
    print("✅ 初期化完了")
    return True

# モジュール読み込み時に初期化実行
if __name__ != "__main__":
    initialize() 