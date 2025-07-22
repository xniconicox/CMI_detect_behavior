#!/usr/bin/env python3
"""
Fold-wise CMI Evaluation Script

このスクリプトは、クロスバリデーションの各foldで保存されたモデルを使って、
検証データに対するCMI値を計算します。

目的:
- 各foldのモデルの性能を個別に評価
- foldごとのCMIスコアのばらつきを確認
- 可視化で使用するcv_results.jsonの生成
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# プロジェクトのルートディレクトリをPATHに追加
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.cmi_evaluation import calculate_cmi_score
from src.trainers.multimodal_trainer_v40 import MultimodalTrainerV40


def main():
    parser = argparse.ArgumentParser(description="Fold-wise CMI Evaluation")
    parser.add_argument("--experiment-name", default="preprocess_v40_ws64", 
                        help="実験名 (default: preprocess_v40_ws64)")
    parser.add_argument("--n-splits", type=int, default=5, 
                        help="CVのfold数 (default: 5)")
    parser.add_argument("--random-state", type=int, default=42, 
                        help="ランダムシード (default: 42)")
    parser.add_argument("--save-confusion-matrix", action="store_true",
                        help="混同行列を保存するかどうか")
    
    args = parser.parse_args()
    
    # 設定
    EXPERIMENT_NAME = args.experiment_name
    N_SPLITS = args.n_splits
    RANDOM_STATE = args.random_state
    
    # パス設定
    BASE_DIR = Path(f"output/experiments/{EXPERIMENT_NAME}")
    MODELS_DIR = BASE_DIR / "models"
    RESULTS_DIR = BASE_DIR / "results"
    PREPROCESSED_DIR = BASE_DIR / "preprocessed"
    
    print(f"🔍 Fold-wise CMI Evaluation")
    print(f"実験名: {EXPERIMENT_NAME}")
    print(f"ベースディレクトリ: {BASE_DIR}")
    print(f"CVのfold数: {N_SPLITS}")
    print("=" * 50)
    
    # データの読み込み
    print("📊 データを読み込み中...")
    
    try:
        # MultimodalTrainerV40を使用してデータを読み込み
        trainer = MultimodalTrainerV40(EXPERIMENT_NAME)
        data = trainer.load_all_data()
        
        X_sensor = data['sensor']
        X_demo = data['demographics']
        X_tab = data['tabular']
        y = data['labels']
        groups = data['groups']
        
        print(f"✅ データ読み込み完了")
        print(f"  - センサーデータ: {X_sensor.shape}")
        print(f"  - デモグラフィックデータ: {X_demo.shape}")
        print(f"  - タブラーデータ: {X_tab.shape}")
        print(f"  - ラベル: {y.shape}")
        print(f"  - グループ数: {len(np.unique(groups))}")
        
    except Exception as e:
        print(f"❌ データ読み込みエラー: {e}")
        return
    
    # クロスバリデーションの分割を再現
    print("\n🔄 クロスバリデーション分割を再現中...")
    
    skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    folds = list(skf.split(X_sensor, y, groups))
    
    print(f"✅ {N_SPLITS}つのfoldを生成")
    
    # 各foldの情報を表示
    for i, (train_idx, val_idx) in enumerate(folds):
        train_size = len(train_idx)
        val_size = len(val_idx)
        print(f"  Fold {i+1}: 訓練={train_size:,}, 検証={val_size:,}")
    
    # 各foldのモデルを読み込んで評価
    print("\n🎯 各foldのモデルを評価中...")
    
    fold_results = []
    all_predictions = []
    all_true_labels = []
    
    for fold, (train_idx, val_idx) in enumerate(folds):
        print(f"\n--- Fold {fold+1} ---")
        
        # モデルの読み込み
        model_path = MODELS_DIR / f"model_fold_{fold+1}.keras"
        
        if not model_path.exists():
            print(f"⚠️  モデルファイルが見つかりません: {model_path}")
            continue
        
        try:
            model = tf.keras.models.load_model(model_path)
            print(f"✅ モデル読み込み完了: {model_path.name}")
        except Exception as e:
            print(f"❌ モデル読み込みエラー: {e}")
            continue
        
        # 検証データの準備
        X_val_sensor = X_sensor[val_idx]
        X_val_demo = X_demo[val_idx]
        X_val_tab = X_tab[val_idx]
        y_val = y[val_idx]
        
        # 予測
        print(f"📊 予測実行中... (検証データ数: {len(val_idx):,})")
        y_pred_proba = model.predict({
            'sensor_input': X_val_sensor,
            'demo_input': X_val_demo,
            'tabular_input': X_val_tab
        }, verbose=0)
        
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # メトリクス計算
        cmi_score = calculate_cmi_score(y_val, y_pred)
        binary_f1 = f1_score(y_val, y_pred, average='binary', pos_label=1) if len(np.unique(y_val)) == 2 else 0
        macro_f1 = f1_score(y_val, y_pred, average='macro')
        accuracy = np.mean(y_val == y_pred)
        
        # 結果を保存
        fold_result = {
            'fold': fold + 1,
            'cmi_score': float(cmi_score),
            'binary_f1': float(binary_f1),
            'macro_f1': float(macro_f1),
            'accuracy': float(accuracy),
            'val_size': len(val_idx)
        }
        
        fold_results.append(fold_result)
        all_predictions.extend(y_pred)
        all_true_labels.extend(y_val)
        
        print(f"  CMI Score: {cmi_score:.4f}")
        print(f"  Binary F1: {binary_f1:.4f}")
        print(f"  Macro F1: {macro_f1:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
    
    print(f"\n✅ 全{len(fold_results)}フォルドの評価完了")
    
    if not fold_results:
        print("❌ 評価結果がありません")
        return
    
    # 結果の集計と統計
    print("\n📈 結果の集計中...")
    
    df_results = pd.DataFrame(fold_results)
    
    # 統計情報
    stats = {
        'cmi_score': {
            'mean': df_results['cmi_score'].mean(),
            'std': df_results['cmi_score'].std(),
            'min': df_results['cmi_score'].min(),
            'max': df_results['cmi_score'].max()
        },
        'macro_f1': {
            'mean': df_results['macro_f1'].mean(),
            'std': df_results['macro_f1'].std(),
            'min': df_results['macro_f1'].min(),
            'max': df_results['macro_f1'].max()
        },
        'accuracy': {
            'mean': df_results['accuracy'].mean(),
            'std': df_results['accuracy'].std(),
            'min': df_results['accuracy'].min(),
            'max': df_results['accuracy'].max()
        }
    }
    
    print("\n📊 統計情報:")
    for metric, values in stats.items():
        print(f"  {metric.upper()}:")
        print(f"    平均: {values['mean']:.4f} ± {values['std']:.4f}")
        print(f"    範囲: [{values['min']:.4f}, {values['max']:.4f}]")
    
    # 詳細結果の表示
    print("\n📋 詳細結果:")
    print(df_results.round(4).to_string(index=False))
    
    # 結果の可視化
    print("\n📊 結果を可視化中...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Fold-wise Performance Metrics ({EXPERIMENT_NAME})', fontsize=16)
    
    # CMI Score
    axes[0, 0].bar(df_results['fold'], df_results['cmi_score'], color='skyblue', alpha=0.7)
    axes[0, 0].axhline(y=df_results['cmi_score'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {df_results["cmi_score"].mean():.4f}')
    axes[0, 0].set_title('CMI Score by Fold')
    axes[0, 0].set_xlabel('Fold')
    axes[0, 0].set_ylabel('CMI Score')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Macro F1
    axes[0, 1].bar(df_results['fold'], df_results['macro_f1'], color='lightgreen', alpha=0.7)
    axes[0, 1].axhline(y=df_results['macro_f1'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {df_results["macro_f1"].mean():.4f}')
    axes[0, 1].set_title('Macro F1 Score by Fold')
    axes[0, 1].set_xlabel('Fold')
    axes[0, 1].set_ylabel('Macro F1 Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1, 0].bar(df_results['fold'], df_results['accuracy'], color='orange', alpha=0.7)
    axes[1, 0].axhline(y=df_results['accuracy'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {df_results["accuracy"].mean():.4f}')
    axes[1, 0].set_title('Accuracy by Fold')
    axes[1, 0].set_xlabel('Fold')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # メトリクスの分布
    metrics_data = [df_results['cmi_score'], df_results['macro_f1'], df_results['accuracy']]
    metrics_labels = ['CMI Score', 'Macro F1', 'Accuracy']
    
    bp = axes[1, 1].boxplot(metrics_data, labels=metrics_labels, patch_artist=True)
    colors = ['skyblue', 'lightgreen', 'orange']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[1, 1].set_title('Metrics Distribution Across Folds')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 図の保存
    output_path = RESULTS_DIR / "fold_wise_evaluation.png"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 可視化結果を保存: {output_path}")
    
    plt.show()
    
    # cv_results.jsonの生成と保存
    print("\n💾 cv_results.jsonを生成中...")
    
    # 全体の平均を計算
    overall_cmi = calculate_cmi_score(all_true_labels, all_predictions)
    overall_macro_f1 = f1_score(all_true_labels, all_predictions, average='macro')
    overall_accuracy = np.mean(np.array(all_true_labels) == np.array(all_predictions))
    
    cv_results = {
        'fold_cmi_scores': [result['cmi_score'] for result in fold_results],
        'fold_macro_f1_scores': [result['macro_f1'] for result in fold_results],
        'fold_accuracies': [result['accuracy'] for result in fold_results],
        'mean_cmi_score': float(np.mean([result['cmi_score'] for result in fold_results])),
        'std_cmi_score': float(np.std([result['cmi_score'] for result in fold_results])),
        'mean_macro_f1': float(np.mean([result['macro_f1'] for result in fold_results])),
        'std_macro_f1': float(np.std([result['macro_f1'] for result in fold_results])),
        'mean_accuracy': float(np.mean([result['accuracy'] for result in fold_results])),
        'std_accuracy': float(np.std([result['accuracy'] for result in fold_results])),
        'overall_cmi_score': float(overall_cmi),
        'overall_macro_f1': float(overall_macro_f1),
        'overall_accuracy': float(overall_accuracy),
        'n_folds': len(fold_results),
        'experiment_name': EXPERIMENT_NAME
    }
    
    # ファイルに保存
    cv_results_path = RESULTS_DIR / "cv_results.json"
    with open(cv_results_path, 'w') as f:
        json.dump(cv_results, f, indent=2)
    
    print(f"✅ cv_results.jsonを保存: {cv_results_path}")
    
    # 詳細結果も保存
    detailed_results_path = RESULTS_DIR / "fold_wise_detailed_results.json"
    with open(detailed_results_path, 'w') as f:
        json.dump(fold_results, f, indent=2)
    
    print(f"✅ 詳細結果を保存: {detailed_results_path}")
    
    # サマリーの表示
    print("\n📋 生成されたcv_results.jsonのサマリー:")
    print(f"  - foldごとのCMIスコア: {cv_results['fold_cmi_scores']}")
    print(f"  - 平均CMIスコア: {cv_results['mean_cmi_score']:.4f} ± {cv_results['std_cmi_score']:.4f}")
    print(f"  - 全体CMIスコア: {cv_results['overall_cmi_score']:.4f}")
    
    # 混動行列の可視化（オプション）
    if args.save_confusion_matrix:
        print("\n🎯 混同行列を生成中（Fold 1の例）...")
        
        # 最初のfoldのデータを使用
        train_idx, val_idx = folds[0]
        
        # モデルの読み込み
        model_path = MODELS_DIR / "model_fold_1.keras"
        if model_path.exists():
            model = tf.keras.models.load_model(model_path)
            
            # 検証データの準備
            X_val_sensor = X_sensor[val_idx]
            X_val_demo = X_demo[val_idx]
            X_val_tab = X_tab[val_idx]
            y_val = y[val_idx]
            
            # 予測
            y_pred_proba = model.predict({
                'sensor_input': X_val_sensor,
                'demo_input': X_val_demo,
                'tabular_input': X_val_tab
            }, verbose=0)
            
            y_pred = np.argmax(y_pred_proba, axis=1)
            
            # 混同行列の作成
            cm = confusion_matrix(y_val, y_pred)
            
            # 可視化
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - Fold 1 ({EXPERIMENT_NAME})')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            
            # 保存
            cm_path = RESULTS_DIR / "confusion_matrix_fold1.png"
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            print(f"✅ 混同行列を保存: {cm_path}")
            
            plt.show()
            
            # 分類レポート
            print("\n📊 分類レポート（Fold 1）:")
            print(classification_report(y_val, y_pred))
        else:
            print(f"❌ モデルファイルが見つかりません: {model_path}")
    
    print("\n🎉 処理完了！")
    print(f"生成されたファイル:")
    print(f"  - {cv_results_path}")
    print(f"  - {detailed_results_path}")
    print(f"  - {output_path}")
    if args.save_confusion_matrix and (RESULTS_DIR / "confusion_matrix_fold1.png").exists():
        print(f"  - {RESULTS_DIR / 'confusion_matrix_fold1.png'}")


if __name__ == "__main__":
    main() 