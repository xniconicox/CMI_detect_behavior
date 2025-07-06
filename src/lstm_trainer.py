#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CMIコンペ LSTM学習スクリプト
前処理済みデータを使用してLSTMモデルを学習する
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
warnings.filterwarnings('ignore')

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.lstm_model import LSTMModel

class LSTMTrainer:
    """
    LSTM学習管理クラス
    """
    
    def __init__(self, experiment_name="lstm_v1"):
        """
        初期化
        
        Parameters:
        -----------
        experiment_name : str
            実験名
        """
        self.experiment_name = experiment_name
        self.output_dir = Path(f"../output/experiments/{experiment_name}")
        self.models_dir = self.output_dir / "models"
        self.results_dir = self.output_dir / "results"
        
        # ディレクトリ作成
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 設定
        self.window_size = 128
        self.n_features = 332
        self.n_classes = 18
        self.random_state = 42
        
        print(f"LSTM学習環境初期化完了")
        print(f"実験名: {self.experiment_name}")
        print(f"出力ディレクトリ: {self.output_dir}")
        
    def load_preprocessed_data(self, data_path=None):
        """
        前処理済みデータの読み込み
        
        Parameters:
        -----------
        data_path : str, optional
            データパス（指定しない場合はデフォルトパス）
        
        Returns:
        --------
        data : dict
            読み込んだデータ
        """
        print("前処理済みデータを読み込み中...")
        
        if data_path is None:
            # デフォルトのデータパス
            data_path = self.output_dir / "preprocessed"
        
        data_path = Path(data_path)
        
        try:
            # 単一のpickleファイルかディレクトリかを判定
            if data_path.is_file() and data_path.suffix == '.pkl':
                # 単一のpickleファイルの場合
                print(f"単一ファイルからデータを読み込み: {data_path}")
                with open(data_path, "rb") as f:
                    data = pickle.load(f)
                
                print(f"データ読み込み完了")
                print(f"X_windows形状: {data['X_windows'].shape}")
                print(f"y_windows形状: {data['y_windows'].shape}")
                print(f"ウィンドウサイズ: {data['meta'].get('window_size', 'N/A')}")
                print(f"特徴量数: {data['meta'].get('n_features', 'N/A')}")
                print(f"クラス数: {data['meta'].get('n_classes', 'N/A')}")
                
                return data
            
            else:
                # ディレクトリ構造の場合（従来の方法）
                print(f"ディレクトリからデータを読み込み: {data_path}")
                
                # 前処理済みデータの読み込み
                with open(data_path / "X_windows.pkl", "rb") as f:
                    X_windows = pickle.load(f)
                
                with open(data_path / "y_windows.pkl", "rb") as f:
                    y_windows = pickle.load(f)
                
                with open(data_path / "scaler.pkl", "rb") as f:
                    scaler = pickle.load(f)
                
                with open(data_path / "label_encoder.pkl", "rb") as f:
                    label_encoder = pickle.load(f)
                
                # メタデータの読み込み
                with open(data_path / "meta.json", "r") as f:
                    meta = json.load(f)
                
                print(f"データ読み込み完了")
                print(f"X_windows形状: {X_windows.shape}")
                print(f"y_windows形状: {y_windows.shape}")
                print(f"ウィンドウサイズ: {meta.get('window_size', 'N/A')}")
                print(f"特徴量数: {meta.get('n_features', 'N/A')}")
                print(f"クラス数: {meta.get('n_classes', 'N/A')}")
                
                data = {
                    'X_windows': X_windows,
                    'y_windows': y_windows,
                    'scaler': scaler,
                    'label_encoder': label_encoder,
                    'meta': meta
                }
                
                return data
            
        except FileNotFoundError as e:
            print(f"前処理済みデータが見つかりません: {e}")
            print("前処理ノートブックを実行してデータを準備してください")
            return None
        except Exception as e:
            print(f"データ読み込みエラー: {e}")
            return None
    
    def create_alternative_data(self):
        """
        代替データの作成（前処理済みデータが無い場合）
        """
        print("代替データを作成中...")
        
        try:
            # 生データの読み込み
            train_df = pd.read_csv("../data/train.csv")
            
            # センサー列の特定
            sensor_cols = [col for col in train_df.columns 
                          if col.startswith(('acc_', 'rot_', 'tof_', 'thm_'))]
            
            print(f"センサー列数: {len(sensor_cols)}")
            
            # 簡単なサンプリング（デモ用）
            sample_sequences = train_df['sequence_id'].unique()[:100]  # 100シーケンスのみ
            sample_df = train_df[train_df['sequence_id'].isin(sample_sequences)]
            
            # 簡単なウィンドウ作成
            X_list = []
            y_list = []
            
            for seq_id in sample_sequences:
                seq_data = sample_df[sample_df['sequence_id'] == seq_id]
                if len(seq_data) >= self.window_size:
                    # 最初のウィンドウのみ取得
                    window_data = seq_data[sensor_cols].iloc[:self.window_size].values
                    X_list.append(window_data)
                    y_list.append(seq_data['gesture'].iloc[0])
            
            X_windows = np.array(X_list)
            y_windows = np.array(y_list)
            
            # ラベルエンコーディング
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y_windows)
            
            # 正規化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_windows.reshape(-1, X_windows.shape[-1]))
            X_scaled = X_scaled.reshape(X_windows.shape)
            
            print(f"代替データ作成完了")
            print(f"X_windows形状: {X_scaled.shape}")
            print(f"y_windows形状: {y_encoded.shape}")
            
            data = {
                'X_windows': X_scaled,
                'y_windows': y_encoded,
                'scaler': scaler,
                'label_encoder': label_encoder,
                'meta': {
                    'window_size': self.window_size,
                    'n_features': len(sensor_cols),
                    'n_classes': len(label_encoder.classes_),
                    'n_samples': len(X_scaled)
                }
            }
            
            return data
            
        except Exception as e:
            print(f"代替データ作成エラー: {e}")
            return None
    
    def split_data(self, X, y, test_size=0.2):
        """
        データ分割
        
        Parameters:
        -----------
        X : np.ndarray
            特徴量データ
        y : np.ndarray
            ラベルデータ
        test_size : float
            テストサイズ
        
        Returns:
        --------
        tuple
            分割されたデータ
        """
        print(f"データ分割中... (test_size={test_size})")
        
        # ラベルの分布を確認
        unique_labels, counts = np.unique(y, return_counts=True)
        min_count = np.min(counts)
        
        print(f"ラベル数: {len(unique_labels)}")
        print(f"最小ラベル数: {min_count}")
        
        # 層化抽出が可能かチェック（各クラスに最低2つのサンプルが必要）
        if min_count >= 2 and len(y) >= 10:  # 十分なサンプル数がある場合
            try:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, 
                    test_size=test_size, 
                    random_state=self.random_state,
                    stratify=y
                )
                print("層化抽出を使用してデータを分割しました")
            except ValueError as e:
                print(f"層化抽出に失敗: {e}")
                print("通常の分割を使用します")
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, 
                    test_size=test_size, 
                    random_state=self.random_state
                )
        else:
            print("サンプル数が少ないため、通常の分割を使用します")
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, 
                test_size=test_size, 
                random_state=self.random_state
            )
        
        print(f"訓練データ: {X_train.shape}")
        print(f"検証データ: {X_val.shape}")
        print(f"訓練ラベル: {y_train.shape}")
        print(f"検証ラベル: {y_val.shape}")
        
        return X_train, X_val, y_train, y_val
    
    def train_model(self, data, model_params=None):
        """
        モデル学習
        
        Parameters:
        -----------
        data : dict
            学習データ
        model_params : dict, optional
            モデルパラメータ
        
        Returns:
        --------
        results : dict
            学習結果
        """
        print("モデル学習開始...")
        
        # データの取得
        X_windows = data['X_windows']
        y_windows = data['y_windows']
        meta = data['meta']
        
        # データ分割
        X_train, X_val, y_train, y_val = self.split_data(X_windows, y_windows)
        
        # モデルパラメータの設定
        if model_params is None:
            model_params = {
                'lstm_units_1': 64,
                'lstm_units_2': 64,
                'dense_units': 32,
                'dropout_rate': 0.3,
                'dense_dropout_rate': 0.2,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 50,  # デモ用に短縮
                'patience': 10,
                'reduce_lr_patience': 5,
                'use_tqdm': True,  # 進捗表示を有効化
                'use_tensorboard': True,  # TensorBoard を有効化
                'log_dir': str(self.results_dir / 'logs')  # ログディレクトリ
            }
        
        # 入力形状の設定
        input_shape = (X_train.shape[1], X_train.shape[2])  # (timesteps, features)
        num_classes = meta['n_classes']
        
        print(f"入力形状: {input_shape}")
        print(f"クラス数: {num_classes}")
        
        # モデル作成
        model = LSTMModel(input_shape, num_classes, **model_params)
        
        # 学習実行
        model_save_path = str(self.models_dir / "lstm_best.h5")
        history = model.train(
            X_train, y_train, 
            X_val, y_val,
            model_save_path=model_save_path
        )
        
        # 評価実行
        eval_results = model.evaluate(X_val, y_val)
        
        # 結果の保存
        results = {
            'model': model,
            'history': history,
            'eval_results': eval_results,
            'model_params': model_params,
            'meta': meta
        }
        
        return results
    
    def save_results(self, results, data):
        """
        結果保存
        
        Parameters:
        -----------
        results : dict
            学習結果
        data : dict
            データ
        """
        print("結果保存中...")
        
        model = results['model']
        history = results['history']
        eval_results = results['eval_results']
        model_params = results['model_params']
        meta = results['meta']
        
        # モデル保存
        model_path = self.models_dir / "lstm_best.h5"
        model.save_model(str(model_path))
        
        # 学習履歴保存
        history_path = self.results_dir / "training_history.json"
        model.save_training_history(str(history_path))
        
        # 前処理器保存
        scaler_path = self.models_dir / "scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(data['scaler'], f)
        
        label_encoder_path = self.models_dir / "label_encoder.pkl"
        with open(label_encoder_path, "wb") as f:
            pickle.dump(data['label_encoder'], f)
        
        # メタデータ保存
        meta_path = self.models_dir / "meta.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        
        # 設定保存
        config_path = self.models_dir / "config.json"
        config = {
            'model_params': model_params,
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat()
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        # 評価結果保存
        eval_path = self.results_dir / "evaluation_results.json"
        eval_save = {
            'test_loss': float(eval_results['test_loss']),
            'test_accuracy': float(eval_results['test_accuracy']),
            'f1_macro': float(eval_results['f1_macro']),
            'f1_weighted': float(eval_results['f1_weighted'])
        }
        with open(eval_path, "w") as f:
            json.dump(eval_save, f, indent=2)
        
        print(f"結果保存完了")
        print(f"モデル: {model_path}")
        print(f"学習履歴: {history_path}")
        print(f"評価結果: {eval_path}")
    
    def plot_training_history(self, history, save_path=None):
        """
        学習履歴の可視化
        
        Parameters:
        -----------
        history : tf.keras.callbacks.History
            学習履歴
        save_path : str, optional
            保存パス
        """
        print("学習履歴可視化中...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss
        ax1.plot(history.history['loss'], label='Training Loss')
        ax1.plot(history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy
        ax2.plot(history.history['accuracy'], label='Training Accuracy')
        ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.results_dir / "training_history.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"学習履歴保存: {save_path}")
    
    def run_experiment(self, data_path=None, model_params=None):
        """
        実験実行
        
        Parameters:
        -----------
        data_path : str, optional
            データパス
        model_params : dict, optional
            モデルパラメータ
        
        Returns:
        --------
        results : dict
            実験結果
        """
        print(f"実験開始: {self.experiment_name}")
        print("="*50)
        
        # データ読み込み
        data = self.load_preprocessed_data(data_path)
        
        if data is None:
            print("前処理済みデータが見つからないため、代替データを作成します")
            data = self.create_alternative_data()
            
            if data is None:
                print("データの準備に失敗しました")
                return None
        
        # モデル学習
        results = self.train_model(data, model_params)
        
        # 結果保存
        self.save_results(results, data)
        
        # 可視化
        self.plot_training_history(results['history'])
        
        print("="*50)
        print(f"実験完了: {self.experiment_name}")
        print(f"最終F1スコア (macro): {results['eval_results']['f1_macro']:.4f}")
        print(f"最終Accuracy: {results['eval_results']['test_accuracy']:.4f}")
        print("="*50)
        
        return results


def main():
    """
    メイン関数
    """
    print("LSTM学習スクリプト実行開始")
    
    # 学習実行
    trainer = LSTMTrainer("lstm_v1")
    results = trainer.run_experiment()
    
    if results is not None:
        print("学習完了！")
    else:
        print("学習に失敗しました")


if __name__ == "__main__":
    main() 