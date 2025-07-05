#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CMIコンペ モデル保存スクリプト
学習済みモデルを保存して、推論用ノートブックで使用できるようにします。
"""

import os
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import pickle
import warnings
warnings.filterwarnings('ignore')

def extract_features(df):
    """
    シーケンス単位の特徴量抽出
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
            
            # 範囲
            features[f'{col}_range'] = df[col].max() - df[col].min()
            
            # 歪度と尖度
            features[f'{col}_skew'] = df[col].skew()
            features[f'{col}_kurtosis'] = df[col].kurtosis()
    
    return features

def train_and_save_model():
    """
    モデルを学習して保存
    """
    print("モデル学習・保存開始")
    
    # 学習データの読み込み
    # ローカル環境とKaggle環境の両方に対応
    if os.path.exists('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv'):
        # Kaggle環境
        train_data = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
        train_demographics = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv')
    else:
        # ローカル環境
        train_data = pd.read_csv('../../data/train.csv')
        train_demographics = pd.read_csv('../../data/train_demographics.csv')
    
    print("学習データ読み込み完了")
    print(f"学習データサイズ: {len(train_data)}")
    
    # 特徴量抽出
    print("特徴量抽出中...")
    feature_list = []
    
    # シーケンス単位で特徴量を抽出
    for sequence_id in train_data['sequence_id'].unique():
        seq_data = train_data[train_data['sequence_id'] == sequence_id]
        features = extract_features(seq_data)
        features['sequence_id'] = sequence_id
        features['gesture'] = seq_data['gesture'].iloc[0]  # 同じシーケンス内は同じジェスチャー
        feature_list.append(features)
    
    # 特徴量データフレームを作成
    train_features = pd.DataFrame(feature_list)
    
    # 特徴量カラムを特定（sequence_idとgesture以外）
    feature_cols = [col for col in train_features.columns if col not in ['sequence_id', 'gesture']]
    
    print(f"特徴量数: {len(feature_cols)}")
    
    # ラベルエンコーダーの準備
    le = LabelEncoder()
    y_train = le.fit_transform(train_features['gesture'])
    
    X_train = train_features[feature_cols]
    
    # LightGBMモデルの学習
    print("LightGBMモデル学習中...")
    
    params = {
        'objective': 'multiclass',
        'num_class': len(np.unique(y_train)),
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
    
    # クロスバリデーションで5つのモデルを作成
    models = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"Fold {fold + 1}/5 学習中...")
        
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        # LightGBMデータセット作成
        train_data_lgb = lgb.Dataset(X_fold_train, label=y_fold_train)
        val_data_lgb = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data_lgb)
        
        # モデル学習
        model = lgb.train(
            params,
            train_data_lgb,
            valid_sets=[val_data_lgb],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
            num_boost_round=1000
        )
        
        models.append(model)
        print(f"Fold {fold + 1} 完了")
    
    print("モデル学習完了！")
    
    # モデルを保存
    print("モデル保存中...")
    
    # 保存ディレクトリを作成
    save_dir = '../output/experiments/baseline_lightgbm_v1/models'
    os.makedirs(save_dir, exist_ok=True)
    
    # モデルを保存
    with open(os.path.join(save_dir, 'trained_models.pkl'), 'wb') as f:
        pickle.dump(models, f)
    
    # ラベルエンコーダーを保存
    with open(os.path.join(save_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(le, f)
    
    # 特徴量カラムを保存
    with open(os.path.join(save_dir, 'feature_cols.pkl'), 'wb') as f:
        pickle.dump(feature_cols, f)
    
    print(f"モデル保存完了: {save_dir}/")
    print(f"保存されたファイル:")
    print(f"  - trained_models.pkl ({len(models)}個のモデル)")
    print(f"  - label_encoder.pkl ({len(le.classes_)}クラス)")
    print(f"  - feature_cols.pkl ({len(feature_cols)}特徴量)")
    
    return models, le, feature_cols

if __name__ == "__main__":
    train_and_save_model() 