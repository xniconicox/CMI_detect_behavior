"""Minimal multimodal trainer for v52 data."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


class MultimodalTrainerV52:
    def __init__(self, experiment_name: str = "preprocess_v52") -> None:
        self.experiment_name = experiment_name
        self.base_dir = Path("output/experiments") / experiment_name
        self.data_dir = self.base_dir / "preprocessed"
        self.model_dir = self.base_dir / "models_v52"
        self.result_dir = self.base_dir / "results_v52"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)

    def load_all_data(self) -> Dict[str, np.ndarray]:
        def _load(name: str) -> np.ndarray:
            path = self.data_dir / name
            return np.load(path)

        X_seq = _load("train_sequences.npy")
        T_mask = _load("train_t_mask.npy")
        V_mask = _load("train_v_mask.npy")
        X_tof = _load("train_tof.npy")
        tof_mask = _load("train_tof_mask.npy")
        y = _load("train_labels.npy")

        return {
            "sensor": X_seq,
            "t_mask": T_mask,
            "v_mask": V_mask,
            "tof": X_tof,
            "tof_mask": tof_mask,
            "labels": y,
        }

    def build_model(self, sensor_shape: tuple, tof_shape: tuple, num_classes: int) -> keras.Model:
        sensor_input = keras.Input(shape=sensor_shape, name="sensor")
        t_mask_input = keras.Input(shape=(sensor_shape[0],), name="t_mask")
        v_mask_input = keras.Input(shape=sensor_shape, name="v_mask")
        x_s = sensor_input * v_mask_input
        x_s = keras.layers.Masking()(x_s)
        x_s = keras.layers.Bidirectional(
            keras.layers.LSTM(32)
        )(x_s, mask=tf.cast(t_mask_input, tf.bool))

        tof_input = keras.Input(shape=tof_shape, name="tof")
        tof_mask_input = keras.Input(shape=tof_shape, name="tof_mask")
        x_t = tof_input
        x_t = keras.layers.Lambda(lambda x: tf.where(tf.equal(tof_mask_input, 1), x, tf.zeros_like(x)))(x_t)
        x_t = keras.layers.Conv3D(16, 3, strides=2, padding="same", activation="relu")(x_t)
        x_t = keras.layers.Conv3D(32, 3, strides=2, padding="same", activation="relu")(x_t)
        x_t = keras.layers.GlobalAveragePooling3D()(x_t)

        merged = keras.layers.concatenate([x_s, x_t])
        out = keras.layers.Dense(num_classes, activation="softmax")(merged)

        model = keras.Model(inputs=[sensor_input, t_mask_input, v_mask_input, tof_input, tof_mask_input], outputs=out)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        return model

    def train(self, data: Dict[str, np.ndarray], epochs: int = 10, batch_size: int = 32) -> Dict[str, Any]:
        X_s = data["sensor"]
        T = data["t_mask"]
        V = data["v_mask"]
        X_tof = data["tof"]
        tof_m = data["tof_mask"]
        y = data["labels"]

        idx_train, idx_val = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=42)

        model = self.build_model(sensor_shape=X_s.shape[1:], tof_shape=X_tof.shape[1:], num_classes=len(np.unique(y)))
        history = model.fit(
            [X_s[idx_train], T[idx_train], V[idx_train], X_tof[idx_train], tof_m[idx_train]],
            y[idx_train],
            validation_data=([X_s[idx_val], T[idx_val], V[idx_val], X_tof[idx_val], tof_m[idx_val]], y[idx_val]),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
        )
        self.model = model
        self.history = history

        preds = model.predict([X_s[idx_val], T[idx_val], V[idx_val], X_tof[idx_val], tof_m[idx_val]])
        f1 = f1_score(y[idx_val], preds.argmax(axis=1), average="macro")
        return {"f1": float(f1)}

    def save_model(self) -> None:
        path = self.model_dir / "model_v52.keras"
        self.model.save(path)

    def save_history(self) -> None:
        path = self.result_dir / "history.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.history.history, f, indent=2)

