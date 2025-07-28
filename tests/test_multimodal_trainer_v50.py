import sys
import types
import numpy as np
import pytest

# minimal tensorflow stub
tf_mod = types.ModuleType("tensorflow")
keras_mod = types.ModuleType("keras")
keras_mod.layers = types.ModuleType("layers")
keras_mod.models = types.ModuleType("models")
tf_mod.keras = keras_mod
sys.modules.setdefault("tensorflow", tf_mod)
sys.modules.setdefault("tensorflow.keras", keras_mod)
sys.modules.setdefault("tensorflow.keras.layers", keras_mod.layers)
sys.modules.setdefault("tensorflow.keras.models", keras_mod.models)

mm = pytest.importorskip("src.trainers.multimodal_trainer_v50")
from src.trainers.multimodal_trainer_v50 import MultimodalTrainerV50


def test_forward_pass_with_mask(monkeypatch):
    class DummyTensor:
        pass

    class BaseLayer:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, *args, **kwargs):
            return DummyTensor()

    def make_layer(name):
        return type(name, (BaseLayer,), {})

    layers = {
        "Input": lambda *a, **k: DummyTensor(),
        "LSTM": make_layer("LSTM"),
        "Dense": make_layer("Dense"),
    }

    def concatenate(inputs):
        return DummyTensor()

    for name, cls in layers.items():
        setattr(keras_mod.layers, name, cls)
    keras_mod.layers.concatenate = concatenate
    keras_mod.Input = keras_mod.layers.Input

    class Model:
        def __init__(self, inputs=None, outputs=None):
            pass

        def compile(self, *args, **kwargs):
            pass

        def predict(self, inputs):
            batch = inputs[0].shape[0]
            return np.zeros((batch, 18))

    keras_mod.models.Model = Model
    keras_mod.Model = Model
    mm.keras.Model = Model
    mm.keras.models.Model = Model

    trainer = MultimodalTrainerV50()
    model = trainer.build_model(sensor_shape=(4, 3), demo_shape=2, num_classes=18)

    X = np.random.randn(2, 4, 3).astype(np.float32)
    mask = np.array([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=bool)
    demo = np.random.randn(2, 2).astype(np.float32)

    out = trainer.forward(model, X, mask, demo)
    assert out.shape == (2, 18)
