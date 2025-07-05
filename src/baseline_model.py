#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CMIコンペ ベースラインモデル
Kaggle提出用
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# グローバル変数
global_models = None
global_le = None
global_feature_cols = None

def extract_features(df):
    """
    シーケンス単位の特徴量抽出
    ノートブックの特徴量抽出ロジックを移植
    """
    # センサーデータのカラムを特定
    sensor_cols = [col for col in df.columns if any(sensor in col for sensor in ['acc_', 'rot_', 'tof_', 'thm_'])]
    
    # シーケンス単位で統計量を計算
    features = {}
    
    for col in sensor_cols:
        if df[col].dtype in ['float64', 'int64']:
            # 基本統計量
            features[f'{col}_mean'] = df[col].mean()
            features[f'{col}_std'] = df[col].std()
            features[f'{col}_min'] = df[col].min()
            features[f'{col}_max'] = df[col].max()
            features[f'{col}_median'] = df[col].median()
            
            # 分位数
            features[f'{col}_q25'] = df[col].quantile(0.25)
            features[f'{col}_q75'] = df[col].quantile(0.75)
            
            # その他の統計量
            features[f'{col}_range'] = df[col].max() - df[col].min()
            features[f'{col}_var'] = df[col].var()
            
            # 欠損値の割合
            features[f'{col}_missing_ratio'] = df[col].isnull().sum() / len(df)
    
    # シーケンス情報
    if 'sequence_id' in df.columns:
        features['sequence_id'] = df['sequence_id'].iloc[0]
    
    return pd.DataFrame([features])

def train_model():
    """モデルの学習"""
    global global_models, global_le, global_feature_cols
    
    print("モデル学習開始...")
    
    # 前処理済みデータの読み込み
    train_features = pd.read_csv('../../output/train_features.csv')
    
    # 特徴量とラベルの分離
    feature_cols = [col for col in train_features.columns if col not in ['sequence_id', 'gesture']]
    
    X_train = train_features[feature_cols]
    y_train = train_features['gesture']
    
    # ラベルエンコーディング
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    
    # モデルパラメータ
    params = {
        'objective': 'multiclass',
        'num_class': len(np.unique(y_train_encoded)),
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42
    }
    
    # 単一モデルで学習（時間短縮のため）
    train_data = lgb.Dataset(X_train, label=y_train_encoded)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(50)]
    )
    
    global_models = [model]  # リスト形式で保持
    global_le = le
    global_feature_cols = feature_cols
    
    print("モデル学習完了")

def predict_gesture(sequence_data):
    """シーケンスデータからジェスチャーを予測"""
    global global_models, global_le, global_feature_cols
    
    if global_models is None:
        train_model()
    
    # 特徴量抽出
    features = extract_features(sequence_data)
    
    # 予測に必要な特徴量のみ選択
    if 'sequence_id' in features.columns:
        features = features.drop('sequence_id', axis=1)
    
    # 特徴量の順序を統一
    features = features[global_feature_cols]
    
    # 予測実行
    predictions = np.zeros((1, len(global_le.classes_)))
    
    for model in global_models:
        pred = model.predict(features)
        predictions += pred
    
    predictions /= len(global_models)
    
    # ラベルに変換
    pred_label = predictions.argmax(axis=1)[0]
    pred_gesture = global_le.inverse_transform([pred_label])[0]
    
    return pred_gesture

# 初期化時にモデルを学習
if __name__ == "__main__":
    train_model() 