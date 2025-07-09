#!/usr/bin/env python3
"""
提出用予測スクリプト
最終モデルを使用してテストデータに対して予測を行い、提出用CSVファイルを作成
"""

import os
import json
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import sys

# プロジェクトルートを追加
sys.path.append('src')

from data_loader_fix import load_w64_s16_data, validate_data_consistency, create_data_dict
from models.lstm_v2_trainer import LSTMv2Trainer

def load_latest_model():
    """最新の最終モデルを読み込み"""
    model_dir = 'results/final_model'
    
    # 最新の結果ファイルを取得
    result_files = glob.glob(f"{model_dir}/final_results_*.json")
    if not result_files:
        raise FileNotFoundError("最終結果ファイルが見つかりません")
    
    latest_result_file = max(result_files, key=os.path.getctime)
    
    with open(latest_result_file, 'r') as f:
        results = json.load(f)
    
    print(f"🏆 最新モデル読み込み: {latest_result_file}")
    print(f"📊 CMIスコア: {results['final_cmi_score']:.4f}")
    print(f"🗓️ 作成日時: {results['timestamp']}")
    
    return results

def load_test_data():
    """テストデータを読み込み"""
    print("\n📊 テストデータ読み込み中...")
    
    # テストデータパスを確認
    test_sensor_path = 'data/processed/test_sensor_data_w64_s16.npy'
    test_demographics_path = 'data/processed/test_demographics_data_w64_s16.npy'
    
    if not os.path.exists(test_sensor_path) or not os.path.exists(test_demographics_path):
        print("❌ テストデータが見つかりません")
        print("📝 テストデータパス:")
        print(f"   センサーデータ: {test_sensor_path}")
        print(f"   人口統計データ: {test_demographics_path}")
        return None, None
    
    test_sensor_data = np.load(test_sensor_path)
    test_demographics_data = np.load(test_demographics_path)
    
    print(f"✅ テストデータ読み込み完了")
    print(f"   センサーデータ: {test_sensor_data.shape}")
    print(f"   人口統計データ: {test_demographics_data.shape}")
    
    return test_sensor_data, test_demographics_data

def create_test_data_dict(sensor_data, demographics_data):
    """テストデータ辞書を作成"""
    return {
        'sensor_data': sensor_data,
        'demographics_data': demographics_data,
        'labels': None  # テストデータにはラベルがない
    }

def generate_predictions(model_results):
    """予測を生成"""
    print("\n🔮 予測生成中...")
    
    # モデル読み込み
    model_path = model_results['model_path']
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")
    
    model = tf.keras.models.load_model(model_path)
    print(f"🏗️ モデル読み込み完了: {model_path}")
    
    # テストデータ読み込み
    test_sensor_data, test_demographics_data = load_test_data()
    
    if test_sensor_data is None or test_demographics_data is None:
        print("❌ テストデータを読み込めませんでした")
        return None
    
    # 予測実行
    print("\n🚀 予測実行中...")
    
    # バッチサイズ設定
    batch_size = model_results['model_params']['batch_size']
    
    # 予測
    predictions = model.predict(
        [test_sensor_data, test_demographics_data],
        batch_size=batch_size,
        verbose=1
    )
    
    # 予測結果を0-1の範囲に変換（シグモイド出力の場合）
    predictions = predictions.flatten()
    
    print(f"✅ 予測完了")
    print(f"   予測数: {len(predictions)}")
    print(f"   予測値範囲: [{predictions.min():.4f}, {predictions.max():.4f}]")
    
    return predictions

def create_submission_file(predictions, model_results):
    """提出用CSVファイルを作成"""
    print("\n📝 提出ファイル作成中...")
    
    # 提出用ディレクトリ作成
    submission_dir = 'results/submission'
    os.makedirs(submission_dir, exist_ok=True)
    
    # テストデータのIDを取得（仮定：連番）
    test_ids = np.arange(len(predictions))
    
    # DataFrame作成
    submission_df = pd.DataFrame({
        'id': test_ids,
        'prediction': predictions
    })
    
    # ファイル名生成
    timestamp = model_results['timestamp']
    cmi_score = model_results['final_cmi_score']
    submission_filename = f"submission_{timestamp}_cmi{cmi_score:.4f}.csv"
    submission_path = os.path.join(submission_dir, submission_filename)
    
    # CSV保存
    submission_df.to_csv(submission_path, index=False)
    
    print(f"✅ 提出ファイル作成完了: {submission_path}")
    print(f"📊 提出データ:")
    print(f"   データ数: {len(submission_df)}")
    print(f"   予測値統計:")
    print(f"     平均: {predictions.mean():.4f}")
    print(f"     標準偏差: {predictions.std():.4f}")
    print(f"     最小値: {predictions.min():.4f}")
    print(f"     最大値: {predictions.max():.4f}")
    
    # 提出情報を保存
    submission_info = {
        'submission_file': submission_path,
        'model_path': model_results['model_path'],
        'cmi_score': cmi_score,
        'timestamp': timestamp,
        'prediction_stats': {
            'count': len(predictions),
            'mean': float(predictions.mean()),
            'std': float(predictions.std()),
            'min': float(predictions.min()),
            'max': float(predictions.max())
        }
    }
    
    info_path = os.path.join(submission_dir, f"submission_info_{timestamp}.json")
    with open(info_path, 'w') as f:
        json.dump(submission_info, f, indent=2)
    
    print(f"📋 提出情報保存: {info_path}")
    
    return submission_path

def main():
    """メイン処理"""
    print("🚀 提出用予測生成開始")
    print("=" * 80)
    
    try:
        # 最新モデル読み込み
        model_results = load_latest_model()
        
        # 予測生成
        predictions = generate_predictions(model_results)
        
        if predictions is None:
            print("❌ 予測生成に失敗しました")
            return
        
        # 提出ファイル作成
        submission_path = create_submission_file(predictions, model_results)
        
        print(f"\n🎉 提出準備完了!")
        print(f"📁 提出ファイル: {submission_path}")
        print(f"🏆 モデルスコア: {model_results['final_cmi_score']:.4f}")
        
    except Exception as e:
        print(f"\n❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 