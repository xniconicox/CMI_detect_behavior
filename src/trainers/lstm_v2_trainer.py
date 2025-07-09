#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CMIコンペ LSTM v2ハイブリッドモデル学習スクリプト
時系列データ + demographics情報を統合したモデルの学習
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from src.utils.config_utils import load_config
warnings.filterwarnings('ignore')

# TensorFlow GPU設定
import tensorflow as tf

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.lstm_v2_model import LSTMv2HybridModel, create_lstm_v2_hybrid_model, setup_gpu

class LSTMv2Trainer:
    """
    LSTM v2ハイブリッドモデル学習管理クラス
    """
    
    def __init__(self, experiment_name="lstm_v2", window_config="w64_s16", n_demographics_features=None):
        """
        初期化
        
        Parameters:
        -----------
        experiment_name : str
            実験名
        window_config : str
            ウィンドウ設定 ("w64_s16" or "w128_s32")
        n_demographics_features : int, optional
            Demographics特徴量数。指定しない場合は20（デフォルト）
        """
        self.experiment_name = experiment_name
        self.window_config = window_config
        config = load_config()
        base_output = Path(config["output_dir"])
        self.output_dir = base_output / f"{experiment_name}_{window_config}"
        self.models_dir = self.output_dir / "models"
        self.results_dir = self.output_dir / "results"

        # 前処理済みデータのパス
        self.preprocessed_dir = base_output / f"lstm_v2_{window_config}" / "preprocessed"
        
        # ディレクトリ作成
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 設定
        if window_config == "w64_s16":
            self.window_size = 64
            self.stride = 16
        elif window_config == "w128_s32":
            self.window_size = 128
            self.stride = 32
        else:
            raise ValueError(f"Unknown window_config: {window_config}")
        
        self.n_sensor_features = 332
        self.n_demographics_features = n_demographics_features if n_demographics_features is not None else 20
        self.n_classes = 18
        self.random_state = 42
        
        # GPU設定を実行
        self.gpu_available = setup_gpu()
        
        print(f"LSTM v2学習環境初期化完了")
        print(f"実験名: {self.experiment_name}")
        print(f"ウィンドウ設定: {self.window_config}")
        print(f"出力ディレクトリ: {self.output_dir}")
        print(f"前処理済みデータ: {self.preprocessed_dir}")
        print(f"GPU利用可能: {self.gpu_available}")
    
    def load_preprocessed_data(self, use_optimized_demographics=False):
        """
        前処理済みデータの読み込み
        
        Parameters:
        -----------
        use_optimized_demographics : bool
            最適化されたdemographics特徴量を使用するかどうか
        
        Returns:
        --------
        data : dict
            読み込んだデータ
        """
        print("前処理済みデータを読み込み中...")
        
        # データファイルの確認
        required_files = [
            'X_sensor_windows.pkl',
            'X_demographics_windows.pkl',
            'y_windows.pkl',
            'sensor_scaler.pkl',
            'demographics_scaler.pkl',
            'label_encoder.pkl',
            'config.json'
        ]
        
        for file in required_files:
            file_path = self.preprocessed_dir / file
            if not file_path.exists():
                raise FileNotFoundError(f"Required file not found: {file_path}")
        
        # データ読み込み
        with open(self.preprocessed_dir / 'X_sensor_windows.pkl', 'rb') as f:
            X_sensor_windows = pickle.load(f)
        
        # Demographics特徴量の読み込み（最適化版 or オリジナル）
        if use_optimized_demographics:
            # 最適化されたdemographics特徴量を使用
            # notebookから実行する場合のパス調整
            optimized_demo_dir = Path("../results/demographics_optimization")
            train_demo_path = optimized_demo_dir / "X_demographics_train_optimized.npy"
            val_demo_path = optimized_demo_dir / "X_demographics_val_optimized.npy"
            
            if train_demo_path.exists() and val_demo_path.exists():
                print("最適化されたdemographics特徴量を使用します")
                X_demographics_train = np.load(train_demo_path)
                X_demographics_val = np.load(val_demo_path)
                X_demographics_windows = np.concatenate([X_demographics_train, X_demographics_val], axis=0)
                print(f"最適化されたdemographics形状: {X_demographics_windows.shape}")
            else:
                print("最適化されたdemographics特徴量が見つかりません。オリジナルを使用します")
                print(f"探索パス: {train_demo_path}, {val_demo_path}")
                with open(self.preprocessed_dir / 'X_demographics_windows.pkl', 'rb') as f:
                    X_demographics_windows = pickle.load(f)
        else:
            with open(self.preprocessed_dir / 'X_demographics_windows.pkl', 'rb') as f:
                X_demographics_windows = pickle.load(f)
        
        with open(self.preprocessed_dir / 'y_windows.pkl', 'rb') as f:
            y_windows = pickle.load(f)
        
        with open(self.preprocessed_dir / 'sensor_scaler.pkl', 'rb') as f:
            sensor_scaler = pickle.load(f)
        
        with open(self.preprocessed_dir / 'demographics_scaler.pkl', 'rb') as f:
            demographics_scaler = pickle.load(f)
        
        with open(self.preprocessed_dir / 'label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        with open(self.preprocessed_dir / 'config.json', 'r') as f:
            config = json.load(f)
        
        print(f"データ読み込み完了:")
        print(f"  センサーデータ: {X_sensor_windows.shape}")
        print(f"  Demographics: {X_demographics_windows.shape}")
        print(f"  ラベル: {y_windows.shape}")
        print(f"  クラス数: {len(label_encoder.classes_)}")
        
        # データの整合性チェック
        assert X_sensor_windows.shape[0] == X_demographics_windows.shape[0] == y_windows.shape[0], \
            "データサイズが一致しません"
        
        assert X_sensor_windows.shape[1] == self.window_size, \
            f"ウィンドウサイズが期待値と異なります: {X_sensor_windows.shape[1]} != {self.window_size}"
        
        assert X_sensor_windows.shape[2] == self.n_sensor_features, \
            f"センサー特徴量数が期待値と異なります: {X_sensor_windows.shape[2]} != {self.n_sensor_features}"
        
        assert X_demographics_windows.shape[1] == self.n_demographics_features, \
            f"Demographics特徴量数が期待値と異なります: {X_demographics_windows.shape[1]} != {self.n_demographics_features}"
        
        data = {
            'X_sensor_windows': X_sensor_windows,
            'X_demographics_windows': X_demographics_windows,
            'y_windows': y_windows,
            'sensor_scaler': sensor_scaler,
            'demographics_scaler': demographics_scaler,
            'label_encoder': label_encoder,
            'config': config
        }
        
        return data
    
    def split_data(self, X_sensor, X_demographics, y, test_size=0.2, val_size=0.2):
        """
        データ分割
        
        Parameters:
        -----------
        X_sensor : np.ndarray
            センサーデータ
        X_demographics : np.ndarray
            Demographics データ
        y : np.ndarray
            ラベル
        test_size : float
            テストデータの割合
        val_size : float
            検証データの割合（訓練データに対する割合）
        
        Returns:
        --------
        tuple
            分割されたデータ
        """
        print(f"データ分割中...")
        print(f"テストサイズ: {test_size}")
        print(f"検証サイズ: {val_size}")
        
        # まず訓練+検証とテストに分割
        X_sensor_temp, X_sensor_test, X_demographics_temp, X_demographics_test, y_temp, y_test = train_test_split(
            X_sensor, X_demographics, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        # 次に訓練と検証に分割
        X_sensor_train, X_sensor_val, X_demographics_train, X_demographics_val, y_train, y_val = train_test_split(
            X_sensor_temp, X_demographics_temp, y_temp,
            test_size=val_size,
            random_state=self.random_state,
            stratify=y_temp
        )
        
        print(f"データ分割完了:")
        print(f"  訓練 - センサー: {X_sensor_train.shape}, Demographics: {X_demographics_train.shape}, ラベル: {y_train.shape}")
        print(f"  検証 - センサー: {X_sensor_val.shape}, Demographics: {X_demographics_val.shape}, ラベル: {y_val.shape}")
        print(f"  テスト - センサー: {X_sensor_test.shape}, Demographics: {X_demographics_test.shape}, ラベル: {y_test.shape}")
        
        return (X_sensor_train, X_sensor_val, X_sensor_test, 
                X_demographics_train, X_demographics_val, X_demographics_test,
                y_train, y_val, y_test)
    
    def train_model(self, data, model_params=None, fusion_type='concatenate'):
        """
        ハイブリッドモデル学習
        
        Parameters:
        -----------
        data : dict
            学習データ
        model_params : dict, optional
            モデルパラメータ
        fusion_type : str
            融合方式 ('concatenate', 'attention', 'gated')
        
        Returns:
        --------
        results : dict
            学習結果
        """
        print(f"ハイブリッドモデル学習開始 (融合方式: {fusion_type})...")
        
        # データの取得
        X_sensor_windows = data['X_sensor_windows']
        X_demographics_windows = data['X_demographics_windows']
        y_windows = data['y_windows']
        
        # データ分割
        (X_sensor_train, X_sensor_val, X_sensor_test, 
         X_demographics_train, X_demographics_val, X_demographics_test,
         y_train, y_val, y_test) = self.split_data(X_sensor_windows, X_demographics_windows, y_windows)
        
        # モデルパラメータの設定
        if model_params is None:
            model_params = {
                'lstm_units_1': 64,
                'lstm_units_2': 32,
                'dense_units': 32,
                'demographics_dense_units': 16,
                'fusion_dense_units': 24,
                'dropout_rate': 0.3,
                'dense_dropout_rate': 0.2,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 50,
                'patience': 15,
                'reduce_lr_patience': 8,
                'use_tqdm': True,
                'use_tensorboard': True,
                'log_dir': str(self.results_dir / 'logs'),
                'fusion_type': fusion_type
            }
        else:
            model_params['fusion_type'] = fusion_type
        
        # 入力形状の設定
        sensor_input_shape = (X_sensor_train.shape[1], X_sensor_train.shape[2])
        demographics_input_shape = (X_demographics_train.shape[1],)
        num_classes = len(np.unique(y_windows))
        
        print(f"入力形状:")
        print(f"  センサー: {sensor_input_shape}")
        print(f"  Demographics: {demographics_input_shape}")
        print(f"  クラス数: {num_classes}")
        
        # GPUを使用してモデル作成と学習を実行
        if self.gpu_available:
            with tf.device('/GPU:0'):
                print("GPU上でモデルを作成・学習します")
                
                # モデル作成
                model = LSTMv2HybridModel(
                    sensor_input_shape, demographics_input_shape, num_classes, 
                    **model_params
                )
                model.build_model()
                
                # モデルサマリー表示
                print("\n=== モデルサマリー ===")
                model.get_model_summary()
                
                # 学習実行
                model_save_path = str(self.models_dir / f"lstm_v2_hybrid_{fusion_type}_best.h5")
                
                history = model.train(
                    X_sensor_train, X_demographics_train, y_train,
                    X_sensor_val, X_demographics_val, y_val,
                    model_save_path=model_save_path
                )
                
                # 評価実行
                results = model.evaluate(X_sensor_test, X_demographics_test, y_test)
        else:
            print("CPU上でモデルを作成・学習します")
            
            # モデル作成
            model = LSTMv2HybridModel(
                sensor_input_shape, demographics_input_shape, num_classes, 
                **model_params
            )
            model.build_model()
            
            # モデルサマリー表示
            print("\n=== モデルサマリー ===")
            model.get_model_summary()
            
            # 学習実行
            model_save_path = str(self.models_dir / f"lstm_v2_hybrid_{fusion_type}_best.h5")
            
            history = model.train(
                X_sensor_train, X_demographics_train, y_train,
                X_sensor_val, X_demographics_val, y_val,
                model_save_path=model_save_path
            )
            
            # 評価実行
            results = model.evaluate(X_sensor_test, X_demographics_test, y_test)
        
        # 結果保存
        self.save_results(model, results, history, model_params, fusion_type)
        
        return {
            'model': model,
            'history': history,
            'results': results,
            'test_data': (X_sensor_test, X_demographics_test, y_test)
        }
    
    def save_results(self, model, results, history, model_params, fusion_type):
        """
        結果保存
        
        Parameters:
        -----------
        model : LSTMv2HybridModel
            学習済みモデル
        results : dict
            評価結果
        history : History
            学習履歴
        model_params : dict
            モデルパラメータ
        fusion_type : str
            融合方式
        """
        print("結果保存中...")
        
        # 学習履歴保存
        history_path = self.results_dir / f"training_history_{fusion_type}.json"
        model.save_training_history(str(history_path))
        
        # 評価結果保存
        results_path = self.results_dir / f"evaluation_results_{fusion_type}.json"
        
        # numpy配列をリストに変換
        results_to_save = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                results_to_save[key] = value.tolist()
            elif isinstance(value, dict):
                results_to_save[key] = value
            else:
                results_to_save[key] = value
        
        with open(results_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        # モデル設定保存
        config_path = self.models_dir / f"config_{fusion_type}.json"
        config = model.get_config()
        config.update(model_params)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # 学習曲線の可視化
        self.plot_training_history(history, fusion_type)
        
        # 混同行列の可視化
        self.plot_confusion_matrix(results['confusion_matrix'], fusion_type)
        
        print(f"結果保存完了: {self.results_dir}")
    
    def plot_training_history(self, history, fusion_type):
        """
        学習曲線の可視化
        """
        plt.figure(figsize=(12, 4))
        
        # Loss曲線
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title(f'Model Loss ({fusion_type})')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Accuracy曲線
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title(f'Model Accuracy ({fusion_type})')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / f'training_history_{fusion_type}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, cm, fusion_type):
        """
        混同行列の可視化
        """
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix ({fusion_type})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(self.results_dir / f'confusion_matrix_{fusion_type}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def compare_fusion_methods(self, data, model_params=None):
        """
        異なる融合方式の比較
        
        Parameters:
        -----------
        data : dict
            学習データ
        model_params : dict, optional
            モデルパラメータ
        
        Returns:
        --------
        comparison_results : dict
            比較結果
        """
        print("異なる融合方式の比較実験開始...")
        
        fusion_types = ['concatenate', 'attention', 'gated']
        comparison_results = {}
        
        for fusion_type in fusion_types:
            print(f"\n=== {fusion_type.upper()} 融合方式 ===")
            
            try:
                result = self.train_model(data, model_params, fusion_type)
                comparison_results[fusion_type] = {
                    'test_accuracy': result['results']['test_accuracy'],
                    'test_loss': result['results']['test_loss'],
                    'f1_macro': result['results']['f1_macro'],
                    'f1_weighted': result['results']['f1_weighted']
                }
                
                print(f"{fusion_type} 完了:")
                print(f"  Accuracy: {result['results']['test_accuracy']:.4f}")
                print(f"  F1-macro: {result['results']['f1_macro']:.4f}")
                
            except Exception as e:
                print(f"{fusion_type} でエラーが発生: {e}")
                comparison_results[fusion_type] = None
        
        # 比較結果の保存
        comparison_path = self.results_dir / "fusion_comparison.json"
        with open(comparison_path, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        
        # 比較結果の可視化
        self.plot_fusion_comparison(comparison_results)
        
        return comparison_results
    
    def plot_fusion_comparison(self, comparison_results):
        """
        融合方式の比較結果可視化
        """
        # 有効な結果のみ抽出
        valid_results = {k: v for k, v in comparison_results.items() if v is not None}
        
        if not valid_results:
            print("比較可能な結果がありません")
            return
        
        metrics = ['test_accuracy', 'f1_macro', 'f1_weighted']
        fusion_types = list(valid_results.keys())
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
        
        for i, metric in enumerate(metrics):
            values = [valid_results[fusion_type][metric] for fusion_type in fusion_types]
            
            axes[i].bar(fusion_types, values)
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_ylabel('Score')
            
            # 値をバーの上に表示
            for j, value in enumerate(values):
                axes[i].text(j, value + 0.01, f'{value:.3f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'fusion_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"比較結果可視化完了: {self.results_dir}/fusion_comparison.png")


def main():
    """
    メイン実行関数
    """
    print("LSTM v2ハイブリッドモデル学習開始")
    
    # 設定
    experiment_name = "lstm_v2_hybrid"
    window_config = "w64_s16"  # または "w128_s32"
    
    # トレーナー初期化
    trainer = LSTMv2Trainer(experiment_name, window_config)
    
    # データ読み込み
    data = trainer.load_preprocessed_data()
    
    # 異なる融合方式の比較実験
    comparison_results = trainer.compare_fusion_methods(data)
    
    # 結果サマリー
    print("\n=== 実験結果サマリー ===")
    for fusion_type, results in comparison_results.items():
        if results is not None:
            print(f"{fusion_type}:")
            print(f"  Accuracy: {results['test_accuracy']:.4f}")
            print(f"  F1-macro: {results['f1_macro']:.4f}")
            print(f"  F1-weighted: {results['f1_weighted']:.4f}")
        else:
            print(f"{fusion_type}: エラー")
    
    print("\nLSTM v2ハイブリッドモデル学習完了")


if __name__ == "__main__":
    main() 