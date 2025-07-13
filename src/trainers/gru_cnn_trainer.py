#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GRU/CNN-GRU モデル用トレーナー."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import GRU, Conv1D, MaxPooling1D, Dense, Concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam

# プロジェクトルートへのパスを設定
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class GRUCNNTrainer:
    """GRU または CNN-GRU モデルの学習を管理するクラス。"""

    def __init__(self, experiment_name: str = "gru_cnn_example") -> None:
        self.experiment_name = experiment_name
        self.exp_dir = PROJECT_ROOT / "output" / "experiments" / experiment_name
        self.pre_dir = self.exp_dir / "preprocessed"
        self.models_dir = self.exp_dir / "models"
        self.logs_dir = self.exp_dir / "logs"

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """IMU ウィンドウと demographics を読み込む。"""
        x_path = self.pre_dir / "train_windows.pkl"
        demo_path = self.pre_dir / "train_demographics.pkl"
        y_path = self.pre_dir / "train_labels.pkl"

        with open(x_path, "rb") as f:
            X_windows = pickle.load(f)
        with open(demo_path, "rb") as f:
            demographics = pickle.load(f)
        with open(y_path, "rb") as f:
            y = pickle.load(f)

        return X_windows, demographics, y

    # ------------------------------------------------------------------
    def build_gru_model(
        self, input_shape: Tuple[int, int], demo_dim: int, num_classes: int
    ) -> Model:
        """シンプルな GRU モデルを構築"""
        sensor_input = Input(shape=input_shape, name="sensor_input")
        x = GRU(64, return_sequences=False)(sensor_input)

        demo_input = Input(shape=(demo_dim,), name="demo_input")
        concat = Concatenate()([x, demo_input])
        x = Dense(64, activation="relu")(concat)
        output = Dense(num_classes, activation="softmax")(x)

        model = Model([sensor_input, demo_input], output)
        return model

    # ------------------------------------------------------------------
    def build_cnn_gru_model(
        self, input_shape: Tuple[int, int], demo_dim: int, num_classes: int
    ) -> Model:
        """1D CNN を前段に入れた GRU モデルを構築"""
        sensor_input = Input(shape=input_shape, name="sensor_input")
        x = Conv1D(32, 3, padding="same", activation="relu")(sensor_input)
        x = MaxPooling1D()(x)
        x = GRU(64)(x)

        demo_input = Input(shape=(demo_dim,), name="demo_input")
        concat = Concatenate()([x, demo_input])
        x = Dense(64, activation="relu")(concat)
        output = Dense(num_classes, activation="softmax")(x)

        model = Model([sensor_input, demo_input], output)
        return model

    # ------------------------------------------------------------------
    def train(
        self,
        model_type: str = "gru",
        batch_size: int = 32,
        epochs: int = 50,
        test_size: float = 0.2,
    ) -> Model:
        """モデルを学習し、結果を保存する"""
        X, demo, y = self.load_data()

        X_train, X_val, demo_train, demo_val, y_train, y_val = train_test_split(
            X, demo, y, test_size=test_size, random_state=42, stratify=y
        )

        input_shape = X_train.shape[1:]
        demo_dim = demo_train.shape[1]
        num_classes = int(np.max(y) + 1)

        if model_type == "gru":
            model = self.build_gru_model(input_shape, demo_dim, num_classes)
            model_name = "gru"
        else:
            model = self.build_cnn_gru_model(input_shape, demo_dim, num_classes)
            model_name = "cnn_gru"

        model.compile(
            optimizer=Adam(),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        ckpt_path = self.models_dir / f"{model_name}_best.h5"
        callbacks = [
            ModelCheckpoint(str(ckpt_path), save_best_only=True, monitor="val_loss"),
            TensorBoard(log_dir=str(self.logs_dir)),
        ]

        history = model.fit(
            [X_train, demo_train],
            y_train,
            validation_data=([X_val, demo_val], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=2,
        )

        # 最終モデル保存
        final_path = self.models_dir / f"{model_name}_last.h5"
        model.save(str(final_path))

        # 履歴保存
        history_path = self.logs_dir / f"history_{model_name}.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history.history, f, ensure_ascii=False, indent=2)

        return model

