"""
データローダーモジュール
w64_s16とw128_s32の前処理済みデータを読み込む
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from src.utils.config_utils import load_config

def load_w64_s16_data():
    """
    w64_s16の前処理済みデータを読み込む
    
    Returns:
        tuple: (sensor_data, demographics_data, labels)
    """
    config = load_config()
    data_dir = Path(config["output_dir"]) / "lstm_v2_w64_s16" / "preprocessed"
    
    print(f"📁 データ読み込み中...")
    
    try:
        # センサーデータ読み込み
        with open(data_dir / "X_sensor_windows.pkl", "rb") as f:
            sensor_data = pickle.load(f)
        print(f"✅ センサーデータ: {sensor_data.shape}")
        
        # Demographicsデータ読み込み
        with open(data_dir / "X_demographics_windows.pkl", "rb") as f:
            demographics_data = pickle.load(f)
        print(f"✅ Demographics: {demographics_data.shape}")
        
        # ラベルデータ読み込み
        with open(data_dir / "y_windows.pkl", "rb") as f:
            labels = pickle.load(f)
        print(f"✅ ラベル: {labels.shape}")
        
        # データ整合性チェック
        assert sensor_data.shape[0] == demographics_data.shape[0] == labels.shape[0], \
            f"データサイズ不一致: sensor={sensor_data.shape[0]}, demographics={demographics_data.shape[0]}, labels={labels.shape[0]}"
        
        print(f"✅ データ整合性チェック完了")
        print(f"📊 全サンプル数: {sensor_data.shape[0]}")
        print(f"🪟 ウィンドウサイズ: {sensor_data.shape[1]}")
        print(f"🔢 特徴量数: {sensor_data.shape[2]}")
        print(f"👥 Demographics特徴量数: {demographics_data.shape[1]}")
        
        return sensor_data, demographics_data, labels
        
    except Exception as e:
        print(f"❌ データ読み込みエラー: {e}")
        print(f"🔍 パス確認: {data_dir.exists()}")
        if data_dir.exists():
            files = list(data_dir.glob("*.pkl"))
            print(f"📁 利用可能ファイル: {[f.name for f in files]}")
        raise

"""
w128_s32データサイズエラー修正用のデータローダー
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

def load_w128_s32_data():
    """
    w128_s32の正しい前処理済みデータを読み込む
    
    Returns:
        tuple: (X_sensor, X_demographics, y, meta_info)
    """
    config = load_config()
    preprocessed_path = Path(config["output_dir"]) / "lstm_v2_w128_s32" / "preprocessed"
    
    print(f"📁 前処理済みデータパス: {preprocessed_path}")
    
    try:
        # センサーデータ読み込み
        with open(preprocessed_path / "X_sensor_windows.pkl", "rb") as f:
            X_sensor = pickle.load(f)
        print(f"✅ センサーデータ: {X_sensor.shape}")
        
        # Demographicsデータ読み込み
        with open(preprocessed_path / "X_demographics_windows.pkl", "rb") as f:
            X_demographics = pickle.load(f)
        print(f"✅ Demographics: {X_demographics.shape}")
        
        # ラベルデータ読み込み
        with open(preprocessed_path / "y_windows.pkl", "rb") as f:
            y = pickle.load(f)
        print(f"✅ ラベル: {y.shape}")
        
        # 設定情報読み込み
        with open(preprocessed_path / "config.pkl", "rb") as f:
            config = pickle.load(f)
        
        # スケーラー読み込み
        with open(preprocessed_path / "sensor_scaler.pkl", "rb") as f:
            sensor_scaler = pickle.load(f)
            
        with open(preprocessed_path / "demographics_scaler.pkl", "rb") as f:
            demographics_scaler = pickle.load(f)
            
        # ラベルエンコーダー読み込み
        with open(preprocessed_path / "label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        
        # データ整合性チェック
        assert X_sensor.shape[0] == X_demographics.shape[0] == y.shape[0], \
            f"データサイズ不一致: sensor={X_sensor.shape[0]}, demographics={X_demographics.shape[0]}, labels={y.shape[0]}"
        
        print(f"✅ データ整合性チェック完了")
        print(f"📊 全サンプル数: {X_sensor.shape[0]}")
        print(f"🪟 ウィンドウサイズ: {X_sensor.shape[1]}")
        print(f"🔢 特徴量数: {X_sensor.shape[2]}")
        print(f"👥 Demographics特徴量数: {X_demographics.shape[1]}")
        print(f"🏷️ クラス数: {len(label_encoder.classes_)}")
        
        meta_info = {
            'config': config,
            'sensor_scaler': sensor_scaler,
            'demographics_scaler': demographics_scaler,
            'label_encoder': label_encoder,
            'window_size': X_sensor.shape[1],
            'n_features': X_sensor.shape[2],
            'n_demographics': X_demographics.shape[1],
            'n_classes': len(label_encoder.classes_),
            'n_samples': X_sensor.shape[0]
        }
        
        return X_sensor, X_demographics, y, meta_info
        
    except Exception as e:
        print(f"❌ データ読み込みエラー: {e}")
        print(f"🔍 パス確認: {preprocessed_path.exists()}")
        if preprocessed_path.exists():
            files = list(preprocessed_path.glob("*.pkl"))
            print(f"📁 利用可能ファイル: {[f.name for f in files]}")
        raise

def validate_data_consistency(X_sensor, X_demographics, y):
    """データの整合性を検証"""
    print("🔍 データ整合性検証中...")
    
    # サンプル数チェック
    n_sensor = X_sensor.shape[0]
    n_demo = X_demographics.shape[0] 
    n_labels = y.shape[0]
    
    print(f"センサーサンプル数: {n_sensor}")
    print(f"Demographicsサンプル数: {n_demo}")
    print(f"ラベルサンプル数: {n_labels}")
    
    if n_sensor == n_demo == n_labels:
        print("✅ 全データのサンプル数が一致")
        return True
    else:
        print("❌ データサンプル数が不一致")
        return False

def create_data_dict(X_sensor, X_demographics, y, meta_info=None):
    """学習用のデータ辞書を作成（LSTMv2Trainer.train_model互換）"""
    data_dict = {
        'X_sensor_windows': X_sensor,  # 修正: 正しいキー名を使用
        'X_demographics_windows': X_demographics,  # 修正: 正しいキー名を使用
        'y_windows': y
    }
    
    # meta_infoが提供された場合は追加情報を含める
    if meta_info is not None:
        data_dict.update({
            'sensor_scaler': meta_info['sensor_scaler'],
            'demographics_scaler': meta_info['demographics_scaler'],
            'label_encoder': meta_info['label_encoder'],
            'config': meta_info['config']
        })
    
    return data_dict

if __name__ == "__main__":
    # テスト実行
    try:
        # w128_s32データテスト
        print("\n🔍 w128_s32データテスト:")
        X_sensor, X_demographics, y, meta_info = load_w128_s32_data()
        if validate_data_consistency(X_sensor, X_demographics, y):
            data_dict = create_data_dict(X_sensor, X_demographics, y, meta_info)
            print("✅ w128_s32データ読み込み成功")
        else:
            print("❌ データ整合性エラー")
            
        # w64_s16データテスト
        print("\n🔍 w64_s16データテスト:")
        sensor_data, demographics_data, labels = load_w64_s16_data()
        if validate_data_consistency(sensor_data, demographics_data, labels):
            print("✅ w64_s16データ読み込み成功")
        else:
            print("❌ データ整合性エラー")
            
    except Exception as e:
        print(f"❌ テスト失敗: {e}") 