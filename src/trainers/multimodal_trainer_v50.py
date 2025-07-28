"""Simplified multimodal trainer for version 50.

This trainer demonstrates handling of variable length
sequences with a mask input.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import tensorflow as tf
from tensorflow import keras


class MultimodalTrainerV50:
    """Minimal trainer that builds a model with mask input."""

    def __init__(self, experiment_name: str = "multimodal_v50") -> None:
        self.experiment_name = experiment_name
        self.base_dir = Path("output/experiments") / experiment_name
        self.data_dir = self.base_dir / "preprocessed"
        self.model_dir = self.base_dir / "models"
        self.result_dir = self.base_dir / "results"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)

    def build_model(
        self,
        sensor_shape: tuple,
        demo_shape: int,
        num_classes: int,
    ) -> keras.Model:
        """Build small model using sequence mask."""
        sensor_input = keras.Input(shape=sensor_shape, name="sensor")
        mask_input = keras.Input(shape=(sensor_shape[0],), name="mask", dtype="bool")
        x = keras.layers.LSTM(8)(sensor_input, mask=mask_input)

        demo_input = keras.Input(shape=(demo_shape,), name="demo")
        merged = keras.layers.concatenate([x, demo_input])
        output = keras.layers.Dense(num_classes, activation="softmax")(merged)

        model = keras.Model(inputs=[sensor_input, mask_input, demo_input], outputs=output)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        return model

    def forward(self, model: keras.Model, X_s: np.ndarray, mask: np.ndarray, X_d: np.ndarray) -> np.ndarray:
        """Run a forward pass."""
        return model.predict([X_s, mask, X_d])
