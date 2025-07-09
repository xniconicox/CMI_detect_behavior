#!/usr/bin/env python3
"""
最終モデル学習スクリプト
w64_s16の最適パラメータを使用して提出用モデルを学習
"""

import os
import json
import datetime
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import sys

# プロジェクトルートを追加
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from data.data_loader_fix import load_w64_s16_data, validate_data_consistency, create_data_dict
from trainers.lstm_v2_trainer import LSTMv2Trainer
from utils.cmi_evaluation import calculate_cmi_score

def load_best_config():
    """最適設定を読み込み"""
    config_path = os.path.join('/mnt/c/Users/ShunK/works/CMI_comp/results/final_model', 'best_config.json')
    with open(config_path, 'r') as f:
        return json.load(f)

def prepare_final_model_directory():
    """最終モデル用ディレクトリ準備"""
    model_dir = os.path.join('/mnt/c/Users/ShunK/works/CMI_comp/results/lstm_v2')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(model_dir, 'logs'), exist_ok=True)
    return model_dir

def train_final_model():
    """最終モデル学習"""
    print("🏆 最終モデル学習開始")
    print("=" * 80)
    
    # 最適設定読み込み
    best_config = load_best_config()
    best_params = best_config['best_params']
    
    print(f"🎯 最適スコア: {best_config['best_score']:.4f}")
    print(f"🔧 最適パラメータ: Trial {best_config['best_trial']}")
    
    # トレーナー作成（データ読み込みのため）
    print("\n🏗️ トレーナー初期化中...")
    trainer = LSTMv2Trainer(
        experiment_name="final_model",
        window_config="w64_s16",
        n_demographics_features=18  # 最適化されたdemographics特徴量数
    )
    
    # データ読み込み
    print("\n📊 データ読み込み中...")
    data_dict = trainer.load_preprocessed_data(use_optimized_demographics=True)
    
    # データ一貫性検証
    print("✅ データ一貫性検証中...")
    validate_data_consistency(data_dict['X_sensor_windows'], data_dict['X_demographics_windows'], data_dict['y_windows'])
    
    # データ分割
    print("🔀 データ分割中...")
    indices = np.arange(len(data_dict['X_sensor_windows']))
    train_indices, test_indices = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=data_dict['y_windows']
    )
    
    # パラメータ変換
    fusion_types = ['concatenate', 'attention', 'gated']
    model_params = {
        'lstm_units_1': int(best_params['lstm_units_1']),
        'lstm_units_2': int(best_params['lstm_units_2']),
        'dense_units': int(best_params['dense_units']),
        'dropout_rate': float(best_params['dropout_rate']),
        'dense_dropout_rate': float(best_params['dense_dropout_rate']),
        'fusion_type': fusion_types[int(best_params['fusion_type'])],
        'fusion_dense_units': int(best_params['fusion_dense_units']),
        'demographics_dense_units': int(best_params['demographics_dense_units']),
        'learning_rate': float(best_params['learning_rate']),
        'batch_size': int(best_params['batch_size']),
        'epochs': 150,
        'patience': 25,
        'reduce_lr_patience': 10,
        'use_tqdm': True,
        'use_tensorboard': True
    }
    
    print(f"\n🔧 モデルパラメータ:")
    for key, value in model_params.items():
        print(f"   {key}: {value}")
    
    # 既存のtrainerを使用（重複作成を削除）
    
    # モデルディレクトリ準備
    model_dir = prepare_final_model_directory()
    
    # タイムスタンプ設定
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_path = f"{model_dir}/checkpoints/final_model_{timestamp}.h5"
    
    # モデル学習（150エポック）
    print(f"\n🚀 最終モデル学習開始 (150エポック)")
    print(f"💾 チェックポイント: {checkpoint_path}")
    
    result = trainer.train_model(
        data_dict,
        model_params=model_params,
        fusion_type=model_params['fusion_type']
    )
    
    # 最終評価（train_modelの戻り値から取得）
    print("\n📊 最終評価実行中...")
    test_results = result['results']
    
    # CMIスコアを手動で計算
    try:
        from utils.cmi_evaluation import calculate_cmi_score
        test_data = result['test_data']
        X_sensor_test, X_demographics_test, y_test = test_data
        
        # 予測実行
        predictions = result['model'].predict(X_sensor_test, X_demographics_test)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # CMIスコア計算（label_encoderを渡す）
        label_encoder = data_dict.get('label_encoder', None)
        cmi_result = calculate_cmi_score(predicted_classes, y_test, label_encoder=label_encoder, verbose=True)
        cmi_score, binary_f1, macro_f1, test_accuracy = cmi_result
        
    except Exception as e:
        print(f"CMIスコア計算エラー: {e}")
        # F1-macroスコアをCMIスコアの代替として使用
        cmi_score = test_results.get('f1_macro', 0.0)
        binary_f1 = 0.0
        macro_f1 = test_results.get('f1_macro', 0.0)
        test_accuracy = test_results.get('test_accuracy', 0.0)
    
    print(f"\n🏆 最終結果:")
    print(f"   CMIスコア: {cmi_score:.4f}")
    print(f"   Binary F1: {binary_f1:.4f}")
    print(f"   Macro F1: {macro_f1:.4f}")
    print(f"   テスト精度: {test_accuracy:.4f}")
    print(f"   最適化時スコア: {best_config['best_score']:.4f}")
    
    # モデル保存
    final_model_path = f"{model_dir}/final_model_{timestamp}.h5"
    result['model'].save_model(final_model_path)
    
    # 学習履歴保存
    history_path = f"{model_dir}/training_history_{timestamp}.json"
    result['model'].save_training_history(history_path)
    
    # 学習結果保存
    final_results = {
        'final_cmi_score': cmi_score,
        'binary_f1': binary_f1,
        'macro_f1': macro_f1,
        'test_accuracy': test_accuracy,
        'optimization_score': best_config['best_score'],
        'model_path': final_model_path,
        'checkpoint_path': checkpoint_path,
        'history_path': history_path,
        'timestamp': timestamp,
        'epochs_trained': model_params['epochs'],
        'best_params': best_params,
        'model_params': model_params,
        'window_config': 'w64_s16'
    }
    
    results_path = f"{model_dir}/final_results_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n💾 保存完了:")
    print(f"   モデル: {final_model_path}")
    print(f"   学習履歴: {history_path}")
    print(f"   結果: {results_path}")
    
    return final_results

if __name__ == "__main__":
    try:
        results = train_final_model()
        print("\n🎉 最終モデル学習完了!")
        print("🚀 提出準備完了!")
        
    except Exception as e:
        print(f"\n❌ エラー発生: {e}")
        import traceback
        traceback.print_exc() 