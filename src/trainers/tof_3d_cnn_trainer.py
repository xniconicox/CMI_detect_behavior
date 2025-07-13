"""ToF 3D CNN trainer."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from src.utils.config_utils import load_config


class ToF3DCNNTrainer:
    """Trainer for ToF 3D CNN model."""

    def __init__(self, experiment_name: str = "tof_3d_cnn") -> None:
        self.config = load_config()
        self.experiment_name = experiment_name
        self.preprocessed_dir = (
            Path(self.config["output_dir"]) / experiment_name / "preprocessed"
        )
        self.models_dir = (
            Path(self.config["output_dir"]) / experiment_name / "models"
        )
        self.models_dir.mkdir(parents=True, exist_ok=True)

        pp = self.config.get("preprocessing", {})
        self.window_size = pp.get("window_size", 128)
        self.fill_value = pp.get("padding_value", 0.0)

    def load_data(self):
        """Load ToF windows and labels."""
        tof_path = self.preprocessed_dir / "train_tof_voxel.pkl"
        label_path = self.preprocessed_dir / "train_labels.pkl"
        info_path = self.preprocessed_dir / "train_info.pkl"

        with open(tof_path, "rb") as f:
            tof_tensor = pickle.load(f)
        with open(label_path, "rb") as f:
            labels = pickle.load(f)
        with open(info_path, "rb") as f:
            info = pickle.load(f)

        windows = []
        for meta in info:
            start = meta.get("start_idx", 0)
            end = meta.get("end_idx", start + self.window_size)
            window = tof_tensor[start:end]
            if window.shape[0] < self.window_size:
                pad = np.full(
                    (self.window_size - window.shape[0],) + window.shape[1:],
                    self.fill_value,
                    dtype=np.float32,
                )
                window = np.concatenate([window, pad], axis=0)
            windows.append(window)

        X = np.stack(windows).astype(np.float32)
        y = np.asarray(labels)
        return X, y

    def build_tof_3d_cnn(
        self,
        input_shape: tuple[int, int, int, int],
        num_classes: int,
    ) -> models.Model:
        """Build simple 3D CNN network."""
        inputs = layers.Input(
            shape=(
                input_shape[0],
                input_shape[2],
                input_shape[3],
                input_shape[1],
            )
        )
        x = layers.Conv3D(
            16,
            (3, 3, 3),
            activation="relu",
            padding="same",
        )(inputs)
        x = layers.MaxPool3D((2, 2, 2))(x)
        x = layers.Conv3D(32, (3, 3, 3), activation="relu", padding="same")(x)
        x = layers.MaxPool3D((2, 2, 2))(x)
        x = layers.Conv3D(64, (3, 3, 3), activation="relu", padding="same")(x)
        x = layers.GlobalAveragePooling3D()(x)
        x = layers.Dense(64, activation="relu")(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        model = models.Model(inputs, outputs)
        return model

    def train(self, epochs: int = 20, batch_size: int = 32) -> Path:
        """Train model and save weights."""
        X, y = self.load_data()
        X = X.transpose(0, 1, 3, 4, 2)
        num_classes = int(np.max(y) + 1)
        y_cat = tf.keras.utils.to_categorical(y, num_classes)

        model = self.build_tof_3d_cnn(X.shape[1:], num_classes)
        model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.fit(
            X,
            y_cat,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=2,
        )
        save_path = self.models_dir / "tof_3d_cnn.h5"
        model.save(save_path)
        return save_path


__all__ = ["ToF3DCNNTrainer"]
