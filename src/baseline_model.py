#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CMIコンペ ベースラインモデル
LightGBMを使用したマルチクラス分類
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """前処理済みデータの読み込み"""
    print("データ読み込み中...")
    train_features = pd.read_csv('../output/train_features.csv')
    test_features = pd.read_csv('../output/test_features.csv')
    
    print(f"Train shape: {train_features.shape}")
    print(f"Test shape: {test_features.shape}")
    
    return train_features, test_features

def prepare_data(train_features, test_features):
    """モデル用データの準備"""
    # 特徴量とラベルの分離
    feature_cols = [col for col in train_features.columns if col not in ['sequence_id', 'gesture']]
    
    X_train = train_features[feature_cols]
    y_train = train_features['gesture']
    X_test = test_features[feature_cols]
    
    # ラベルエンコーディング
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    
    print(f"Features: {len(feature_cols)}")
    print(f"Classes: {len(le.classes_)}")
    print(f"Class distribution: {np.bincount(y_train_encoded)}")
    
    return X_train, y_train_encoded, X_test, le

def train_model(X_train, y_train, n_folds=5):
    """LightGBMモデルの訓練"""
    print(f"\n{'-'*50}")
    print("LightGBMモデル訓練開始")
    print(f"{'-'*50}")
    
    # クロスバリデーション設定
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # モデルパラメータ
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
    
    # クロスバリデーション
    fold_scores = []
    models = []
    
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X_train, y_train), 1):
        print(f"\nFold {fold}/{n_folds}")
        
        X_fold_train, X_fold_valid = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_fold_train, y_fold_valid = y_train[train_idx], y_train[valid_idx]
        
        # データセット作成
        train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
        valid_data = lgb.Dataset(X_fold_valid, label=y_fold_valid, reference=train_data)
        
        # モデル訓練
        model = lgb.train(
            params,
            train_data,
            valid_sets=[valid_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )
        
        # 予測と評価
        valid_preds = model.predict(X_fold_valid)
        valid_preds_labels = valid_preds.argmax(axis=1)
        
        fold_score = f1_score(y_fold_valid, valid_preds_labels, average='macro')
        fold_scores.append(fold_score)
        
        print(f"Fold {fold} F1 Score: {fold_score:.4f}")
        
        models.append(model)
    
    # 全体のスコア
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    
    print(f"\n{'='*50}")
    print(f"Cross-validation Results:")
    print(f"Mean F1 Score: {mean_score:.4f} (+/- {std_score:.4f})")
    print(f"Individual scores: {[f'{score:.4f}' for score in fold_scores]}")
    print(f"{'='*50}")
    
    return models, mean_score, std_score

def predict_test(models, X_test, le):
    """テストデータでの予測"""
    print("\nテストデータでの予測中...")
    
    # 全モデルの予測を平均
    test_preds = np.zeros((X_test.shape[0], len(le.classes_)))
    
    for model in models:
        pred = model.predict(X_test)
        test_preds += pred
    
    test_preds /= len(models)
    
    # ラベルに変換
    test_preds_labels = test_preds.argmax(axis=1)
    test_preds_gestures = le.inverse_transform(test_preds_labels)
    
    return test_preds_gestures

def create_submission(test_features, predictions):
    """提出ファイルの作成"""
    print("\n提出ファイル作成中...")
    
    submission = pd.DataFrame({
        'sequence_id': test_features['sequence_id'],
        'gesture': predictions
    })
    
    submission_path = '../output/submission.csv'
    submission.to_csv(submission_path, index=False)
    
    print(f"提出ファイル保存: {submission_path}")
    print(f"Submission shape: {submission.shape}")
    
    # 予測分布の確認
    print("\n予測分布:")
    print(submission['gesture'].value_counts())
    
    return submission

def main():
    """メイン関数"""
    print("CMIコンペ ベースラインモデル実行開始")
    print("="*50)
    
    # データ読み込み
    train_features, test_features = load_data()
    
    # データ準備
    X_train, y_train, X_test, le = prepare_data(train_features, test_features)
    
    # モデル訓練
    models, mean_score, std_score = train_model(X_train, y_train)
    
    # テスト予測
    predictions = predict_test(models, X_test, le)
    
    # 提出ファイル作成
    submission = create_submission(test_features, predictions)
    
    print("\n" + "="*50)
    print("ベースラインモデル実行完了")
    print(f"CV Score: {mean_score:.4f} (+/- {std_score:.4f})")
    print("="*50)
    
    return submission, mean_score

if __name__ == "__main__":
    submission, score = main() 