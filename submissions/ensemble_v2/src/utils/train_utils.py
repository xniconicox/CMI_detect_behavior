#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CMIコンペ モデル学習・推論・可視化用ユーティリティ
baseline_model.pyの拡張版
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report, confusion_matrix
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
    
    return X_train, y_train_encoded, X_test, le, feature_cols

def train_model_detailed(X_train, y_train, feature_cols, n_folds=5):
    """LightGBMモデルの訓練（詳細版）"""
    print(f"\n{'-'*50}")
    print("LightGBMモデル訓練開始（詳細版）")
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
    fold_predictions = []
    fold_true_labels = []
    fold_importance = []
    
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
        
        # 詳細情報を保存
        fold_predictions.append(valid_preds_labels)
        fold_true_labels.append(y_fold_valid)
        fold_importance.append(model.feature_importance(importance_type='gain'))
        
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
    
    # 詳細結果を辞書で返す
    detailed_results = {
        'models': models,
        'mean_score': mean_score,
        'std_score': std_score,
        'fold_scores': fold_scores,
        'fold_predictions': fold_predictions,
        'fold_true_labels': fold_true_labels,
        'fold_importance': fold_importance,
        'feature_cols': feature_cols
    }
    
    return detailed_results

def analyze_feature_importance(detailed_results):
    """特徴量重要度の詳細分析"""
    feature_cols = detailed_results['feature_cols']
    fold_importance = detailed_results['fold_importance']
    
    # 全foldの重要度を平均
    avg_importance = np.mean(fold_importance, axis=0)
    
    # 重要度のDataFrame作成
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': avg_importance,
        'std_importance': np.std(fold_importance, axis=0)
    }).sort_values('importance', ascending=False)
    
    # センサー別の重要度集計
    sensor_importance = {}
    for feature, importance in zip(feature_cols, avg_importance):
        if 'acc_' in feature:
            sensor = 'Accelerometer'
        elif 'rot_' in feature:
            sensor = 'Rotation'
        elif 'tof_' in feature:
            sensor = 'ToF'
        elif 'thm_' in feature:
            sensor = 'Thermal'
        else:
            sensor = 'Other'
        
        if sensor not in sensor_importance:
            sensor_importance[sensor] = []
        sensor_importance[sensor].append(importance)
    
    sensor_avg_importance = {sensor: np.mean(importances) for sensor, importances in sensor_importance.items()}
    
    return importance_df, sensor_avg_importance

def create_confusion_matrix_analysis(detailed_results, le):
    """混同行列の詳細分析"""
    fold_predictions = detailed_results['fold_predictions']
    fold_true_labels = detailed_results['fold_true_labels']
    
    # 全foldの予測と真のラベルを結合
    all_predictions = np.concatenate(fold_predictions)
    all_true_labels = np.concatenate(fold_true_labels)
    
    # 混同行列作成
    cm = confusion_matrix(all_true_labels, all_predictions)
    
    # クラス別F1スコア
    class_f1_scores = f1_score(all_true_labels, all_predictions, average=None)
    
    # クラス別性能のDataFrame
    class_performance = pd.DataFrame({
        'class': le.classes_,
        'f1_score': class_f1_scores,
        'support': np.bincount(all_true_labels)
    })
    
    return cm, class_performance

def predict_test_detailed(models, X_test, le):
    """テストデータでの予測（詳細版）"""
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
    
    # 予測確率も返す
    return test_preds_gestures, test_preds

def create_submission_detailed(test_features, predictions, probabilities=None):
    """提出ファイルの作成（詳細版）"""
    print("\n提出ファイル作成中...")
    
    submission = pd.DataFrame({
        'sequence_id': test_features['sequence_id'],
        'gesture': predictions
    })
    
    # 予測確率も含める場合
    if probabilities is not None:
        for i, class_name in enumerate(['class_' + str(i) for i in range(probabilities.shape[1])]):
            submission[class_name] = probabilities[:, i]
    
    submission_path = '../output/submission.csv'
    submission.to_csv(submission_path, index=False)
    
    print(f"提出ファイル保存: {submission_path}")
    print(f"Submission shape: {submission.shape}")
    
    # 予測分布の確認
    print("\n予測分布:")
    print(submission['gesture'].value_counts())
    
    return submission

def generate_performance_report(detailed_results, le, test_predictions=None):
    """詳細な性能レポート生成"""
    print("\n" + "="*60)
    print("詳細性能レポート")
    print("="*60)
    
    # クロスバリデーション結果
    print(f"\nクロスバリデーション結果:")
    print(f"  平均F1スコア: {detailed_results['mean_score']:.4f}")
    print(f"  標準偏差: {detailed_results['std_score']:.4f}")
    print(f"  各foldスコア: {[f'{score:.4f}' for score in detailed_results['fold_scores']]}")
    
    # 特徴量重要度分析
    importance_df, sensor_importance = analyze_feature_importance(detailed_results)
    print(f"\n特徴量重要度 (Top 10):")
    for i, (_, row) in enumerate(importance_df.head(10).iterrows()):
        print(f"  {i+1}. {row['feature']}: {row['importance']:.4f} (±{row['std_importance']:.4f})")
    
    print(f"\nセンサー別重要度:")
    for sensor, importance in sensor_importance.items():
        print(f"  {sensor}: {importance:.4f}")
    
    # 混同行列分析
    cm, class_performance = create_confusion_matrix_analysis(detailed_results, le)
    print(f"\nクラス別性能:")
    for _, row in class_performance.iterrows():
        print(f"  {row['class']}: F1={row['f1_score']:.4f}, Support={row['support']}")
    
    # テスト予測の分析
    if test_predictions is not None:
        print(f"\nテスト予測分析:")
        pred_counts = pd.Series(test_predictions).value_counts()
        print(f"  予測サンプル数: {len(test_predictions)}")
        print(f"  予測クラス数: {len(pred_counts)}")
        print(f"  最も多い予測: {pred_counts.index[0]} ({pred_counts.iloc[0]}件)")
    
    print("\n" + "="*60)
    
    return {
        'importance_df': importance_df,
        'sensor_importance': sensor_importance,
        'confusion_matrix': cm,
        'class_performance': class_performance
    }

def main_detailed():
    """メイン関数（詳細版）"""
    print("CMIコンペ 詳細モデル実行開始")
    print("="*50)
    
    # データ読み込み
    train_features, test_features = load_data()
    
    # データ準備
    X_train, y_train, X_test, le, feature_cols = prepare_data(train_features, test_features)
    
    # モデル訓練（詳細版）
    detailed_results = train_model_detailed(X_train, y_train, feature_cols)
    
    # テスト予測
    predictions, probabilities = predict_test_detailed(detailed_results['models'], X_test, le)
    
    # 提出ファイル作成
    submission = create_submission_detailed(test_features, predictions, probabilities)
    
    # 詳細性能レポート生成
    performance_report = generate_performance_report(detailed_results, le, predictions)
    
    print("\n" + "="*50)
    print("詳細モデル実行完了")
    print(f"CV Score: {detailed_results['mean_score']:.4f} (+/- {detailed_results['std_score']:.4f})")
    print("="*50)
    
    return submission, detailed_results, performance_report

if __name__ == "__main__":
    submission, detailed_results, performance_report = main_detailed() 