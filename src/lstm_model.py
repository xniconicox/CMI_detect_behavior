#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CMIコンペ LSTMモデル
純粋なLSTMを使用したジェスチャー認識
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import json
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 進捗表示用
try:
    from tqdm.keras import TqdmCallback
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("tqdm not available. Install with: pip install tqdm")

# TensorFlowのログレベルを設定
tf.get_logger().setLevel('ERROR')

# GPU設定関数
def setup_gpu():
    """GPU設定を行う"""
    print("GPU設定を確認中...")
    
    # 利用可能なGPUを確認
    gpus = tf.config.list_physical_devices('GPU')
    print(f"利用可能なGPU: {len(gpus)}台")
    
    if gpus:
        try:
            # メモリ成長を有効化（必要に応じてメモリを確保）
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU メモリ成長設定: 有効")
            
            # 現在のGPUデバイスを表示
            print(f"使用予定GPU: {gpus[0].name}")
            
            # GPU使用可能かテスト
            with tf.device('/GPU:0'):
                test_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                result = tf.matmul(test_tensor, test_tensor)
            print("GPU動作テスト: 成功")
            
        except RuntimeError as e:
            print(f"GPU設定エラー: {e}")
            print("CPU で実行します")
    else:
        print("GPU が見つかりません。CPU で実行します")
        print("GPU を使用したい場合は、tensorflow-gpu をインストールしてください:")
        print("pip install tensorflow[and-cuda]")
    
    return len(gpus) > 0

# 推奨ハイパーパラメータ
recommended_params = {
    'lstm_units_1': 32,  # より小さく
    'lstm_units_2': 16,  # より小さく
    'dense_units': 16,   # より小さく
    'dropout_rate': 0.2, # より小さく
    'dense_dropout_rate': 0.1,
    'learning_rate': 0.0001,  # より小さく
    'batch_size': 16,    # より小さく
    'epochs': 50,
    'patience': 15,
    'reduce_lr_patience': 8
}

class LSTMModel:
    """
    純粋なLSTMモデルクラス
    """
    
    def __init__(self, input_shape, num_classes, **kwargs):
        """
        初期化
        
        Parameters:
        -----------
        input_shape : tuple
            入力データの形状 (timesteps, features)
        num_classes : int
            クラス数
        **kwargs : dict
            追加のハイパーパラメータ
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.history = None
        
        # ハイパーパラメータの設定（保守的な値）
        self.lstm_units_1 = kwargs.get('lstm_units_1', 32)  # より小さく
        self.lstm_units_2 = kwargs.get('lstm_units_2', 16)  # より小さく
        self.dense_units = kwargs.get('dense_units', 16)    # より小さく
        self.dropout_rate = kwargs.get('dropout_rate', 0.2) # より小さく
        self.dense_dropout_rate = kwargs.get('dense_dropout_rate', 0.1) # より小さく
        self.learning_rate = kwargs.get('learning_rate', 0.0001)  # より小さく
        self.batch_size = kwargs.get('batch_size', 16)      # より小さく
        self.epochs = kwargs.get('epochs', 50)              # より少なく
        self.patience = kwargs.get('patience', 15)          # より長く
        self.reduce_lr_patience = kwargs.get('reduce_lr_patience', 8) # より長く
        
        # 進捗表示とログ設定
        self.use_tqdm = kwargs.get('use_tqdm', TQDM_AVAILABLE)
        self.use_tensorboard = kwargs.get('use_tensorboard', True)
        self.log_dir = kwargs.get('log_dir', 'logs')
        
        # GPU設定を実行
        self.gpu_available = setup_gpu()
        
        print(f"LSTMモデル初期化完了")
        print(f"入力形状: {self.input_shape}")
        print(f"クラス数: {self.num_classes}")
        print(f"GPU利用可能: {self.gpu_available}")
        print(f"進捗表示: tqdm={self.use_tqdm}, TensorBoard={self.use_tensorboard}")
        
    def build_model(self):
        """
        モデル構築（改善版）
        """
        print("LSTMモデルを構築中...")
        
        model = Sequential([
            # 第1LSTM層（正則化追加）
            LSTM(
                self.lstm_units_1,
                return_sequences=True,
                input_shape=self.input_shape,
                name='lstm_1',
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                recurrent_regularizer=tf.keras.regularizers.l2(1e-4)
            ),
            BatchNormalization(name='bn_1'),
            Dropout(self.dropout_rate, name='dropout_1'),
            
            # 第2LSTM層（正則化追加）
            LSTM(
                self.lstm_units_2,
                return_sequences=False,
                name='lstm_2',
                kernel_regularizer=tf.keras.regularizers.l2(1e-4),
                recurrent_regularizer=tf.keras.regularizers.l2(1e-4)
            ),
            BatchNormalization(name='bn_2'),
            Dropout(self.dropout_rate, name='dropout_2'),
            
            # Dense層（正則化追加）
            Dense(
                self.dense_units,
                activation='relu',
                name='dense_1',
                kernel_regularizer=tf.keras.regularizers.l2(1e-4)
            ),
            Dropout(self.dense_dropout_rate, name='dropout_3'),
            
            # 出力層
            Dense(
                self.num_classes,
                activation='softmax',
                name='output'
            )
        ])
        
        # 勾配クリッピング付きオプティマイザー
        optimizer = Adam(
            learning_rate=self.learning_rate,
            clipvalue=0.5   # 勾配値クリッピング
        )
        
        # コンパイル
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        
        print("モデル構築完了")
        print(f"総パラメータ数: {model.count_params():,}")
        
        return model
    
    def get_callbacks(self, model_save_path):
        """
        コールバック設定
        
        Parameters:
        -----------
        model_save_path : str
            モデル保存パス
        """
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.patience,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.reduce_lr_patience,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # TensorBoard コールバック追加
        if self.use_tensorboard:
            tensorboard_callback = TensorBoard(
                log_dir=self.log_dir,
                histogram_freq=1,
                write_graph=True,
                write_images=True,
                update_freq='epoch'
            )
            callbacks.append(tensorboard_callback)
            print(f"TensorBoard ログ: {self.log_dir}")
        
        # tqdm コールバック追加
        if self.use_tqdm:
            tqdm_callback = TqdmCallback(verbose=1)
            callbacks.append(tqdm_callback)
        
        return callbacks
    
    def train(self, X_train, y_train, X_val, y_val, model_save_path=None, **kwargs):
        """
        モデル学習
        
        Parameters:
        -----------
        X_train : np.ndarray
            訓練データ (samples, timesteps, features)
        y_train : np.ndarray
            訓練ラベル (samples,)
        X_val : np.ndarray
            検証データ (samples, timesteps, features)
        y_val : np.ndarray
            検証ラベル (samples,)
        model_save_path : str
            モデル保存パス
        """
        print("モデル学習開始...")
        print(f"訓練データ形状: {X_train.shape}")
        print(f"検証データ形状: {X_val.shape}")
        print(f"訓練ラベル形状: {y_train.shape}")
        print(f"検証ラベル形状: {y_val.shape}")
        
        # モデルが構築されていない場合は構築
        if self.model is None:
            self.build_model()
        
        # デフォルトの保存パス
        if model_save_path is None:
            model_save_path = 'lstm_best.h5'
        
        # コールバック設定
        callbacks = self.get_callbacks(model_save_path)
        
        # 学習実行
        start_time = datetime.now()
        
        # tqdm使用時はverboseを0に設定
        verbose_level = 0 if self.use_tqdm else 1
        
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=verbose_level
        )
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        print(f"学習完了！学習時間: {training_time:.2f}秒")
        
        # 最良スコアの表示
        best_val_loss = min(self.history.history['val_loss'])
        best_val_acc = max(self.history.history['val_accuracy'])
        
        print(f"最良検証Loss: {best_val_loss:.4f}")
        print(f"最良検証Accuracy: {best_val_acc:.4f}")
        
        return self.history
    
    def predict(self, X):
        """
        予測実行
        
        Parameters:
        -----------
        X : np.ndarray
            予測データ (samples, timesteps, features)
        
        Returns:
        --------
        predictions : np.ndarray
            予測確率 (samples, num_classes)
        """
        if self.model is None:
            raise ValueError("モデルが学習されていません")
        
        predictions = self.model.predict(X, verbose=0)
        return predictions
    
    def evaluate(self, X_test, y_test):
        """
        評価実行
        
        Parameters:
        -----------
        X_test : np.ndarray
            テストデータ (samples, timesteps, features)
        y_test : np.ndarray
            テストラベル (samples,)
        
        Returns:
        --------
        results : dict
            評価結果
        """
        print("モデル評価中...")
        
        # 予測実行
        predictions = self.predict(X_test)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # 評価指標計算
        test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        
        # F1スコア計算
        f1_macro = f1_score(y_test, predicted_classes, average='macro')
        f1_weighted = f1_score(y_test, predicted_classes, average='weighted')
        
        # 混同行列
        cm = confusion_matrix(y_test, predicted_classes)
        
        # 分類レポート
        report = classification_report(y_test, predicted_classes, output_dict=True)
        
        results = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'predictions': predicted_classes,
            'probabilities': predictions,
            'confusion_matrix': cm,
            'classification_report': report
        }
        
        print(f"テストLoss: {test_loss:.4f}")
        print(f"テストAccuracy: {test_accuracy:.4f}")
        print(f"F1-Score (macro): {f1_macro:.4f}")
        print(f"F1-Score (weighted): {f1_weighted:.4f}")
        
        return results
    
    def save_model(self, filepath):
        """
        モデル保存
        
        Parameters:
        -----------
        filepath : str
            保存パス
        """
        if self.model is None:
            raise ValueError("保存するモデルがありません")
        
        try:
            # モデル全体を保存
            self.model.save(filepath)
            
            # アーキテクチャのみ保存
            architecture_path = filepath.replace('.h5', '_architecture.json')
            with open(architecture_path, 'w') as f:
                f.write(self.model.to_json())
            
            # 重みのみ保存（正しい拡張子を使用）
            weights_path = filepath.replace('.h5', '_weights.h5')
            # ファイル名の最後が_weights.h5になるように修正
            if not weights_path.endswith('_weights.h5'):
                weights_path = weights_path.replace('.h5', '') + '_weights.h5'
            
            self.model.save_weights(weights_path)
            
            print(f"モデル保存完了: {filepath}")
            
        except Exception as e:
            print(f"モデル保存エラー: {e}")
            print("基本的なモデル保存のみ実行します")
            # エラーが発生した場合は基本的な保存のみ
            self.model.save(filepath)
            print(f"基本モデル保存完了: {filepath}")
    
    def load_model(self, filepath):
        """
        モデル読み込み
        
        Parameters:
        -----------
        filepath : str
            読み込みパス
        """
        self.model = tf.keras.models.load_model(filepath)
        print(f"モデル読み込み完了: {filepath}")
        
        return self.model
    
    def get_model_summary(self):
        """
        モデルサマリー取得
        """
        if self.model is None:
            raise ValueError("モデルが構築されていません")
        
        return self.model.summary()
    
    def save_training_history(self, filepath):
        """
        学習履歴保存
        
        Parameters:
        -----------
        filepath : str
            保存パス
        """
        if self.history is None:
            raise ValueError("学習履歴がありません")
        
        history_dict = {
            'loss': self.history.history['loss'],
            'accuracy': self.history.history['accuracy'],
            'val_loss': self.history.history['val_loss'],
            'val_accuracy': self.history.history['val_accuracy'],
            'epochs': len(self.history.history['loss'])
        }
        
        with open(filepath, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        print(f"学習履歴保存完了: {filepath}")
    
    def get_config(self):
        """
        設定情報取得
        """
        config = {
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'lstm_units_1': self.lstm_units_1,
            'lstm_units_2': self.lstm_units_2,
            'dense_units': self.dense_units,
            'dropout_rate': self.dropout_rate,
            'dense_dropout_rate': self.dense_dropout_rate,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'patience': self.patience,
            'reduce_lr_patience': self.reduce_lr_patience
        }
        
        return config

    def train_with_warmup(self, X_train, y_train, X_val, y_val, model_save_path=None):
        """
        ウォームアップ学習
        """
        # 最初に小さな学習率で学習
        original_lr = self.learning_rate
        self.learning_rate = 0.00001  # 非常に小さな学習率
        
        # モデルを再コンパイル
        optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 短いエポックでウォームアップ
        warmup_history = self.model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=5,
            validation_data=(X_val, y_val),
            verbose=1
        )
        
        # 学習率を元に戻して本格学習
        self.learning_rate = original_lr
        optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 通常の学習を継続
        return self.train(X_train, y_train, X_val, y_val, model_save_path)


def create_lstm_model(input_shape, num_classes, **kwargs):
    """
    LSTMモデル作成のヘルパー関数
    
    Parameters:
    -----------
    input_shape : tuple
        入力データの形状
    num_classes : int
        クラス数
    **kwargs : dict
        追加のハイパーパラメータ
    
    Returns:
    --------
    model : LSTMModel
        LSTMモデルインスタンス
    """
    model = LSTMModel(input_shape, num_classes, **kwargs)
    model.build_model()
    
    return model


def normalize_data(X_windows):
    """
    時系列データの適切な正規化
    """
    # 全体のデータで正規化（サンプル間で統一）
    n_samples, n_timesteps, n_features = X_windows.shape
    X_flat = X_windows.reshape(-1, n_features)
    
    # 各特徴量の統計量を計算
    feature_means = np.mean(X_flat, axis=0)
    feature_stds = np.std(X_flat, axis=0)
    
    # ゼロ除算を防ぐ
    feature_stds = np.where(feature_stds == 0, 1.0, feature_stds)
    
    # 正規化
    X_normalized = (X_flat - feature_means) / feature_stds
    
    # 無限大やNaNを処理
    X_normalized = np.nan_to_num(X_normalized, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # 元の形状に戻す
    X_normalized = X_normalized.reshape(n_samples, n_timesteps, n_features)
    
    return X_normalized, feature_means, feature_stds


def check_data_quality(X_windows, y_windows):
    """
    データ品質のチェック
    """
    print("データ品質チェック中...")
    
    # NaNチェック
    nan_count = np.isnan(X_windows).sum()
    print(f"NaNの数: {nan_count}")
    
    # 無限大チェック
    inf_count = np.isinf(X_windows).sum()
    print(f"無限大の数: {inf_count}")
    
    # 値の範囲チェック
    print(f"データの最小値: {np.min(X_windows)}")
    print(f"データの最大値: {np.max(X_windows)}")
    print(f"データの平均値: {np.mean(X_windows)}")
    print(f"データの標準偏差: {np.std(X_windows)}")
    
    # ラベルの分布
    unique_labels, counts = np.unique(y_windows, return_counts=True)
    print(f"ラベル分布: {dict(zip(unique_labels, counts))}")
    
    return nan_count == 0 and inf_count == 0


if __name__ == "__main__":
    # テスト用のコード
    print("LSTMモデルクラスのテスト")
    
    # テスト用の設定
    input_shape = (128, 332)  # (timesteps, features)
    num_classes = 18
    
    # モデル作成
    model = create_lstm_model(input_shape, num_classes)
    
    # サマリー表示
    print("\n=== モデルサマリー ===")
    model.get_model_summary()
    
    print("\n=== 設定情報 ===")
    config = model.get_config()
    for key, value in config.items():
        print(f"{key}: {value}")
    
    print("\nLSTMモデルクラステスト完了") 