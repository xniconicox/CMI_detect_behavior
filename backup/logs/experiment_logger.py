#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CMIコンペ 実験結果記録・保存システム
学習結果、パラメータ、性能指標などを体系的に管理
"""

import json
import os
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ExperimentLogger:
    """実験結果を記録・保存するクラス"""
    
    def __init__(self, experiment_name=None, base_dir="../output/experiments"):
        """
        初期化
        
        Args:
            experiment_name (str): 実験名（Noneの場合は自動生成）
            base_dir (str): 実験結果保存ディレクトリ
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # 実験名の生成
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"experiment_{timestamp}"
        else:
            self.experiment_name = experiment_name
        
        # 実験ディレクトリの作成
        self.experiment_dir = self.base_dir / self.experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # サブディレクトリの作成
        (self.experiment_dir / "models").mkdir(exist_ok=True)
        (self.experiment_dir / "plots").mkdir(exist_ok=True)
        (self.experiment_dir / "results").mkdir(exist_ok=True)
        
        # 実験情報の初期化
        self.experiment_info = {
            "experiment_name": self.experiment_name,
            "created_at": datetime.now().isoformat(),
            "status": "running"
        }
        
        print(f"実験ロガー初期化完了: {self.experiment_dir}")
    
    def log_parameters(self, params):
        """モデルパラメータを記録"""
        self.experiment_info["model_parameters"] = params
        print(f"パラメータ記録完了: {len(params)}個のパラメータ")
    
    def log_data_info(self, train_shape, test_shape, feature_count, class_count):
        """データ情報を記録"""
        self.experiment_info["data_info"] = {
            "train_shape": train_shape,
            "test_shape": test_shape,
            "feature_count": feature_count,
            "class_count": class_count
        }
        print(f"データ情報記録完了: {feature_count}特徴量, {class_count}クラス")
    
    def log_training_results(self, detailed_results):
        """学習結果を記録"""
        # 基本統計
        self.experiment_info["training_results"] = {
            "mean_score": float(detailed_results["mean_score"]),
            "std_score": float(detailed_results["std_score"]),
            "fold_scores": [float(score) for score in detailed_results["fold_scores"]],
            "best_fold": int(np.argmax(detailed_results["fold_scores"]) + 1),
            "worst_fold": int(np.argmin(detailed_results["fold_scores"]) + 1),
            "score_range": float(max(detailed_results["fold_scores"]) - min(detailed_results["fold_scores"]))
        }
        
        # 詳細結果をファイルに保存
        results_file = self.experiment_dir / "results" / "training_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.experiment_info["training_results"], f, indent=2, ensure_ascii=False)
        
        print(f"学習結果記録完了: CV Score = {detailed_results['mean_score']:.4f} (±{detailed_results['std_score']:.4f})")
    
    def log_feature_importance(self, importance_df, sensor_importance):
        """特徴量重要度を記録"""
        # 重要度データを保存
        importance_file = self.experiment_dir / "results" / "feature_importance.csv"
        importance_df.to_csv(importance_file, index=False)
        
        # センサー別重要度を保存
        sensor_importance_file = self.experiment_dir / "results" / "sensor_importance.json"
        with open(sensor_importance_file, 'w', encoding='utf-8') as f:
            json.dump(sensor_importance, f, indent=2, ensure_ascii=False)
        
        # 実験情報に追加
        self.experiment_info["feature_importance"] = {
            "top_5_features": importance_df.head(5)[['feature', 'importance']].to_dict('records'),
            "sensor_importance": sensor_importance,
            "total_features": len(importance_df)
        }
        
        print(f"特徴量重要度記録完了: {len(importance_df)}特徴量")
    
    def log_confusion_matrix(self, confusion_matrix, class_performance, le):
        """混同行列とクラス別性能を記録"""
        # 混同行列を保存
        cm_file = self.experiment_dir / "results" / "confusion_matrix.csv"
        cm_df = pd.DataFrame(confusion_matrix, 
                           index=le.classes_, 
                           columns=le.classes_)
        cm_df.to_csv(cm_file)
        
        # クラス別性能を保存
        class_perf_file = self.experiment_dir / "results" / "class_performance.csv"
        class_performance.to_csv(class_perf_file, index=False)
        
        # 実験情報に追加
        self.experiment_info["classification_results"] = {
            "confusion_matrix_shape": confusion_matrix.shape,
            "class_performance_summary": {
                "best_class": class_performance.loc[class_performance['f1_score'].idxmax(), 'class'],
                "worst_class": class_performance.loc[class_performance['f1_score'].idxmin(), 'class'],
                "mean_f1_per_class": float(class_performance['f1_score'].mean()),
                "std_f1_per_class": float(class_performance['f1_score'].std())
            }
        }
        
        print(f"分類結果記録完了: {len(le.classes_)}クラス")
    
    def log_prediction_results(self, submission, predictions, probabilities=None):
        """予測結果を記録"""
        # 提出ファイルをコピー
        submission_file = self.experiment_dir / "results" / "submission.csv"
        submission.to_csv(submission_file, index=False)
        
        # 予測分布を記録
        pred_counts = submission['gesture'].value_counts()
        pred_dist_file = self.experiment_dir / "results" / "prediction_distribution.json"
        with open(pred_dist_file, 'w', encoding='utf-8') as f:
            json.dump(pred_counts.to_dict(), f, indent=2, ensure_ascii=False)
        
        # 実験情報に追加
        self.experiment_info["prediction_results"] = {
            "prediction_count": len(predictions),
            "unique_predictions": len(pred_counts),
            "most_common_prediction": pred_counts.index[0],
            "most_common_count": int(pred_counts.iloc[0])
        }
        
        print(f"予測結果記録完了: {len(predictions)}サンプル")
    
    def save_models(self, models, le):
        """学習済みモデルを保存"""
        # モデルをpickleで保存
        models_file = self.experiment_dir / "models" / "trained_models.pkl"
        with open(models_file, 'wb') as f:
            pickle.dump(models, f)
        
        # ラベルエンコーダーを保存
        le_file = self.experiment_dir / "models" / "label_encoder.pkl"
        with open(le_file, 'wb') as f:
            pickle.dump(le, f)
        
        print(f"モデル保存完了: {len(models)}個のモデル")
    
    def finalize_experiment(self, status="completed"):
        """実験を完了"""
        self.experiment_info["status"] = status
        self.experiment_info["completed_at"] = datetime.now().isoformat()
        
        # 実験情報をJSONで保存
        info_file = self.experiment_dir / "experiment_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(self.experiment_info, f, indent=2, ensure_ascii=False)
        
        # 実験サマリーを作成
        self._create_summary()
        
        print(f"実験完了: {self.experiment_dir}")
        print(f"ステータス: {status}")
    
    def _create_summary(self):
        """実験サマリーを作成"""
        summary = f"""
# 実験サマリー: {self.experiment_name}

## 基本情報
- 作成日時: {self.experiment_info.get('created_at', 'N/A')}
- 完了日時: {self.experiment_info.get('completed_at', 'N/A')}
- ステータス: {self.experiment_info.get('status', 'N/A')}

## データ情報
"""
        
        if "data_info" in self.experiment_info:
            data_info = self.experiment_info["data_info"]
            summary += f"""
- 訓練データ: {data_info.get('train_shape', 'N/A')}
- テストデータ: {data_info.get('test_shape', 'N/A')}
- 特徴量数: {data_info.get('feature_count', 'N/A')}
- クラス数: {data_info.get('class_count', 'N/A')}
"""
        
        if "training_results" in self.experiment_info:
            train_results = self.experiment_info["training_results"]
            summary += f"""
## 学習結果
- 平均F1スコア: {train_results.get('mean_score', 'N/A'):.4f}
- 標準偏差: {train_results.get('std_score', 'N/A'):.4f}
- 最高fold: {train_results.get('best_fold', 'N/A')} ({train_results.get('fold_scores', [])[train_results.get('best_fold', 1)-1]:.4f})
- 最低fold: {train_results.get('worst_fold', 'N/A')} ({train_results.get('fold_scores', [])[train_results.get('worst_fold', 1)-1]:.4f})
"""
        
        if "feature_importance" in self.experiment_info:
            feat_imp = self.experiment_info["feature_importance"]
            summary += f"""
## 特徴量重要度 (Top 5)
"""
            for i, feat in enumerate(feat_imp.get('top_5_features', []), 1):
                summary += f"{i}. {feat['feature']}: {feat['importance']:.4f}\n"
        
        # サマリーファイルを保存
        summary_file = self.experiment_dir / "README.md"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)
    
    @classmethod
    def list_experiments(cls, base_dir="../output/experiments"):
        """実験一覧を取得"""
        base_path = Path(base_dir)
        if not base_path.exists():
            return []
        
        experiments = []
        for exp_dir in base_path.iterdir():
            if exp_dir.is_dir():
                info_file = exp_dir / "experiment_info.json"
                if info_file.exists():
                    with open(info_file, 'r', encoding='utf-8') as f:
                        info = json.load(f)
                    experiments.append({
                        "name": exp_dir.name,
                        "created_at": info.get("created_at", "N/A"),
                        "status": info.get("status", "N/A"),
                        "mean_score": info.get("training_results", {}).get("mean_score", "N/A")
                    })
        
        return sorted(experiments, key=lambda x: x["created_at"], reverse=True)

def log_experiment_results(experiment_name, detailed_results, performance_report, 
                          submission, models, le, params, train_shape, test_shape):
    """実験結果を一括で記録（便利関数）"""
    logger = ExperimentLogger(experiment_name)
    
    # 基本情報
    logger.log_parameters(params)
    logger.log_data_info(train_shape, test_shape, 
                        len(detailed_results['feature_cols']), 
                        len(le.classes_))
    
    # 学習結果
    logger.log_training_results(detailed_results)
    
    # 特徴量重要度
    importance_df = performance_report['importance_df']
    sensor_importance = performance_report['sensor_importance']
    logger.log_feature_importance(importance_df, sensor_importance)
    
    # 分類結果
    confusion_matrix = performance_report['confusion_matrix']
    class_performance = performance_report['class_performance']
    logger.log_confusion_matrix(confusion_matrix, class_performance, le)
    
    # 予測結果
    predictions = submission['gesture'].values
    logger.log_prediction_results(submission, predictions)
    
    # モデル保存
    logger.save_models(models, le)
    
    # 実験完了
    logger.finalize_experiment()
    
    return logger

