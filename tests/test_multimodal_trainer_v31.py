import pickle
import sys
import types
from pathlib import Path as _Path
import numpy as np

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

ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.trainers.multimodal_trainer_v31 import MultimodalTrainerV31
import src.trainers.multimodal_trainer_v31 as mm


def test_load_all_data_casts_tabular(tmp_path):
    pre_dir = tmp_path
    arrays = {
        "train_windows": np.random.randn(2, 3, 4),
        "train_demographics": np.random.randn(2, 2),
        "train_tabular": np.random.randn(2, 5),
        "train_tof_windows": np.random.randn(2, 1, 1, 1, 1),
        "train_labels": np.array([0, 1]),
        "train_info": np.array([0, 1]),
    }
    for name, arr in arrays.items():
        with open(pre_dir / f"{name}.pkl", "wb") as f:
            pickle.dump(arr, f)

    trainer = MultimodalTrainerV31()
    trainer.data_dir = pre_dir
    data = trainer.load_all_data()

    assert data["tabular"].dtype == np.float32
    np.testing.assert_allclose(data["tabular"], arrays["train_tabular"].astype(np.float32))

    key_map = {
        "sensor": "train_windows",
        "demographics": "train_demographics",
        "tof": "train_tof_windows",
    }
    for key, src in key_map.items():
        np.testing.assert_array_equal(data[key], arrays[src])

    np.testing.assert_array_equal(data["labels"], arrays["train_labels"])
    np.testing.assert_array_equal(data["groups"], arrays["train_info"])


def test_build_model_with_attention(monkeypatch):
    created = []

    class DummyTensor:
        pass

    class BaseLayer:
        def __init__(self, *args, **kwargs):
            created.append(self.__class__.__name__)

        def __call__(self, *args, **kwargs):
            return DummyTensor()

    def make_layer(name):
        return type(name, (BaseLayer,), {})

    layers = {
        "Masking": make_layer("Masking"),
        "LSTM": make_layer("LSTM"),
        "Bidirectional": make_layer("Bidirectional"),
        "Dense": make_layer("Dense"),
        "Add": make_layer("Add"),
        "Conv3D": make_layer("Conv3D"),
        "BatchNormalization": make_layer("BatchNormalization"),
        "ReLU": make_layer("ReLU"),
        "GlobalAveragePooling3D": make_layer("GlobalAveragePooling3D"),
        "Reshape": make_layer("Reshape"),
        "Flatten": make_layer("Flatten"),
        "MultiHeadAttention": make_layer("MultiHeadAttention"),
        "Dropout": make_layer("Dropout"),
        "SpatialDropout3D": make_layer("SpatialDropout3D"),
    }

    def Input(*args, **kwargs):
        created.append("Input")
        return DummyTensor()

    def concatenate(inputs):
        created.append("concatenate")
        return DummyTensor()

    for name, cls in layers.items():
        setattr(keras_mod.layers, name, cls)
    keras_mod.layers.Input = Input
    keras_mod.Input = Input
    mm.keras.Input = Input
    mm.keras.layers = keras_mod.layers
    mm.keras.layers.concatenate = concatenate
    keras_mod.layers.concatenate = concatenate

    class Model:
        def __init__(self, inputs=None, outputs=None):
            self.compiled = False

        def compile(self, *args, **kwargs):
            self.compiled = True

        def summary(self):
            return "summary"

    keras_mod.models.Model = Model
    keras_mod.Model = Model
    mm.keras.models.Model = Model
    mm.keras.Model = Model

    keras_mod.optimizers = types.ModuleType("optimizers")
    keras_mod.optimizers.schedules = types.ModuleType("schedules")
    mm.keras.optimizers = keras_mod.optimizers

    class ExponentialDecay:
        def __init__(self, *args, **kwargs):
            pass

    class Adam:
        def __init__(self, *args, **kwargs):
            pass

    keras_mod.optimizers.schedules.ExponentialDecay = ExponentialDecay
    keras_mod.optimizers.Adam = Adam
    mm.keras.optimizers.schedules = keras_mod.optimizers.schedules
    mm.keras.optimizers.schedules.ExponentialDecay = ExponentialDecay
    mm.keras.optimizers.Adam = Adam

    trainer = MultimodalTrainerV31()
    model = trainer.build_multimodal_model(
        sensor_shape=(2, 3),
        demo_shape=2,
        tab_shape=4,
        tof_shape=(1, 1, 1, 1),
        num_classes=2,
        use_attention=True,
    )

    assert "MultiHeadAttention" in created
    assert "SpatialDropout3D" in created
    assert isinstance(model, Model)
    assert model.compiled


def test_build_model_multiwindow(monkeypatch):
    created = []

    class DummyTensor:
        pass

    class BaseLayer:
        def __init__(self, *args, **kwargs):
            created.append(self.__class__.__name__)

        def __call__(self, *args, **kwargs):
            return DummyTensor()

    def make_layer(name):
        return type(name, (BaseLayer,), {})

    layers = {"Masking": make_layer("Masking"), "LSTM": make_layer("LSTM"), "Bidirectional": make_layer("Bidirectional"), "Dense": make_layer("Dense"), "Add": make_layer("Add"), "Conv3D": make_layer("Conv3D"), "BatchNormalization": make_layer("BatchNormalization"), "ReLU": make_layer("ReLU"), "GlobalAveragePooling3D": make_layer("GlobalAveragePooling3D")}

    def Input(*args, **kwargs):
        created.append("Input")
        return DummyTensor()

    for name, cls in layers.items():
        setattr(keras_mod.layers, name, cls)
    keras_mod.layers.Input = Input
    keras_mod.Input = Input
    mm.keras.Input = Input
    mm.keras.layers = keras_mod.layers

    class Model:
        def __init__(self, inputs=None, outputs=None):
            self.compiled = False

        def compile(self, *args, **kwargs):
            self.compiled = True

        def summary(self):
            return "summary"

    keras_mod.models.Model = Model
    mm.keras.models.Model = Model

    keras_mod.optimizers = types.ModuleType("optimizers")
    keras_mod.optimizers.schedules = types.ModuleType("schedules")
    mm.keras.optimizers = keras_mod.optimizers
    keras_mod.optimizers.schedules.ExponentialDecay = lambda *a, **k: None
    keras_mod.optimizers.Adam = lambda *a, **k: None
    mm.keras.optimizers.schedules = keras_mod.optimizers.schedules
    mm.keras.optimizers.Adam = keras_mod.optimizers.Adam

    trainer = MultimodalTrainerV31()
    model = trainer.build_multimodal_model(
        sensor_shape={8: (2, 3), 16: (2, 3)},
        demo_shape=2,
        tab_shape=4,
        tof_shape={8: (1, 1, 1, 1), 16: (1, 1, 1, 1)},
        num_classes=2,
    )

    assert "Input" in created
    assert isinstance(model, Model)

