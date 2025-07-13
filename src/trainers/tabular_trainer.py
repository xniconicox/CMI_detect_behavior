#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tabularモデル学習用トレーナー

LightGBM と CatBoost を用いた Tabular 特徴量の学習管理を行う。
前処理済みデータは ``output/experiments/<experiment_name>/preprocessed/``
に配置された ``train_tabular.pkl`` と ``train_labels.pkl`` を読み込む。
"""

from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

import lightgbm as lgb
from catboost import CatBoostClassifier


class TabularModelTrainer:
    """Tabular 特徴量用モデルの学習クラス"""

    def __init__(self, experiment_name: str = "tabular_baseline") -> None:
        """出力ディレクトリなどを設定する"""
        self.experiment_name = experiment_name
        self.output_dir = Path(f"../output/experiments/{experiment_name}")
        self.preprocessed_dir = self.output_dir / "preprocessed"
        self.models_dir = self.output_dir / "models"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.n_classes: int | None = None

        print("TabularModelTrainer 初期化完了")
        print(f"実験名: {self.experiment_name}")
        print(f"前処理データ: {self.preprocessed_dir}")
        print(f"モデル保存先: {self.models_dir}")

    def load_data(self):
        """前処理済みの Tabular 特徴量とラベルを読み込む"""
        print("Tabular データ読み込み中...")
        with open(self.preprocessed_dir / "train_tabular.pkl", "rb") as f:
            X = pickle.load(f)
        with open(self.preprocessed_dir / "train_labels.pkl", "rb") as f:
            y = pickle.load(f)

        label_path = self.preprocessed_dir / "label_encoder.pkl"
        label_encoder = None
        if label_path.exists():
            with open(label_path, "rb") as f:
                label_encoder = pickle.load(f)
            self.n_classes = len(label_encoder.classes_)
        else:
            self.n_classes = len(np.unique(y))

        print(f"特徴量形状: {X.shape}")
        print(f"ラベル形状: {y.shape}")
        print(f"クラス数: {self.n_classes}")
        return X, y, label_encoder

    def train_lightgbm(self, n_folds: int = 5):
        """LightGBM を用いた学習とモデル保存"""
        X, y, le = self.load_data()
        print("LightGBM 学習開始")
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        models = []
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
            print(f"Fold {fold}/{n_folds}")
            train_ds = lgb.Dataset(X[tr_idx], label=y[tr_idx])
            val_ds = lgb.Dataset(X[val_idx], label=y[val_idx])
            params = {
                "objective": "multiclass",
                "num_class": self.n_classes,
                "metric": "multi_logloss",
                "learning_rate": 0.05,
                "num_leaves": 31,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbose": -1,
                "seed": 42,
            }
            model = lgb.train(
                params,
                train_ds,
                valid_sets=[val_ds],
                num_boost_round=1000,
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
            )
            preds = model.predict(X[val_idx])
            score = f1_score(y[val_idx], preds.argmax(axis=1), average="macro")
            print(f"Fold {fold} F1: {score:.4f}")
            model_path = self.models_dir / f"lightgbm_fold{fold}.txt"
            model.save_model(str(model_path))
            print(f"モデル保存: {model_path}")
            models.append(model)

        with open(self.models_dir / "lightgbm_models.pkl", "wb") as f:
            pickle.dump(models, f)
        if le is not None:
            with open(self.models_dir / "label_encoder.pkl", "wb") as f:
                pickle.dump(le, f)
        print("LightGBM 学習完了")
        return models

    def train_catboost(self, n_folds: int = 5):
        """CatBoost を用いた学習とモデル保存"""
        X, y, le = self.load_data()
        print("CatBoost 学習開始")
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        models = []
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
            print(f"Fold {fold}/{n_folds}")
            model = CatBoostClassifier(
                loss_function="MultiClass",
                depth=6,
                learning_rate=0.05,
                iterations=1000,
                random_seed=42,
                verbose=False,
            )
            model.fit(X[tr_idx], y[tr_idx], eval_set=(X[val_idx], y[val_idx]), use_best_model=True)
            preds = model.predict(X[val_idx])
            score = f1_score(y[val_idx], preds, average="macro")
            print(f"Fold {fold} F1: {score:.4f}")
            model_path = self.models_dir / f"catboost_fold{fold}.cbm"
            model.save_model(model_path)
            print(f"モデル保存: {model_path}")
            models.append(model)

        with open(self.models_dir / "catboost_models.pkl", "wb") as f:
            pickle.dump(models, f)
        if le is not None:
            with open(self.models_dir / "label_encoder.pkl", "wb") as f:
                pickle.dump(le, f)
        print("CatBoost 学習完了")
        return models


if __name__ == "__main__":
    trainer = TabularModelTrainer()
    trainer.train_lightgbm()
