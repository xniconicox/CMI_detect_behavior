"""多モダリティ学習用トレーナー

IMUウィンドウ、人口統計、表形式特徴量、ToFボクセルの4種類の前処理済みデータを
読み込んでタワー型のニューラルネットワークで学習を行う。
"""

from __future__ import annotations

import os
import json
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

from src.utils.cmi_evaluation import calculate_cmi_score
from src.utils.pipeline import Preprocessor
import tensorflow as tf
from tensorflow import keras


class MultimodalTrainer:
    """多モダリティモデル学習管理クラス"""

    def __init__(self, experiment_name: str = "multimodal") -> None:
        self.experiment_name = experiment_name
        self.base_dir = Path("output/experiments") / experiment_name
        self.data_dir = self.base_dir / "preprocessed"
        self.model_dir = self.base_dir / "models"
        self.result_dir = self.base_dir / "results"

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)

        print(f"Experiment name: {self.experiment_name}")
        print(f"Data directory: {self.data_dir}")

    def _resnet_block_3d(self, x: tf.Tensor, filters: int, kernel_size: int = 3, stride: int = 1) -> tf.Tensor:
        """3D ResNet ブロック"""
        shortcut = x

        # メインパス
        y = keras.layers.Conv3D(filters, kernel_size, strides=stride, padding="same")(x)
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.ReLU()(y)

        y = keras.layers.Conv3D(filters, kernel_size, padding="same")(y)
        y = keras.layers.BatchNormalization()(y)

        # ショートカットパス
        if stride != 1 or x.shape[-1] != filters:
            shortcut = keras.layers.Conv3D(filters, 1, strides=stride, padding="same")(shortcut)
            shortcut = keras.layers.BatchNormalization()(shortcut)

        # Add & ReLU
        y = keras.layers.Add()([y, shortcut])
        y = keras.layers.ReLU()(y)
        return y

    def load_all_data(self) -> Dict[str, np.ndarray]:
        """各モダリティの前処理済みデータをすべて読み込む"""
        print("前処理済みデータを読み込み中...")

        def _load(name: str) -> np.ndarray:
            npy = self.data_dir / f"{name}.npy"
            pkl = self.data_dir / f"{name}.pkl"
            if npy.exists():
                return np.load(npy)
            if pkl.exists():
                with open(pkl, "rb") as f:
                    return pickle.load(f)
            raise FileNotFoundError(f"{name} ファイルが見つかりません")

        X_sensor = _load("train_windows")
        X_demo = _load("train_demographics")
        X_tab = _load("train_tabular")
        X_tof = _load("train_tof_windows")
        y = _load("train_labels")

        print(f"センサー: {X_sensor.shape}")
        print(f"人口統計: {X_demo.shape}")
        print(f"表形式: {X_tab.shape}")
        print(f"ToF: {X_tof.shape}")
        print(f"ラベル: {y.shape}")

        return {
            "sensor": X_sensor,
            "demographics": X_demo,
            "tabular": X_tab,
            "tof": X_tof,
            "labels": y,
        }

    def build_multimodal_model(
        self,
        sensor_shape: tuple,
        demo_shape: int,
        tab_shape: int,
        tof_shape: tuple,
        num_classes: int,
    ) -> keras.Model:
        """タワー型統合ネットワークを構築"""
        # 1. IMU Tower (Bidirectional LSTM)
        sensor_input = keras.Input(shape=sensor_shape, name="sensor")
        x1 = keras.layers.Masking()(sensor_input)
        x1 = keras.layers.Bidirectional(keras.layers.LSTM(64))(x1)

        # 2. Demographics Tower
        demo_input = keras.Input(shape=(demo_shape,), name="demo")
        x2 = keras.layers.Dense(32, activation="relu")(demo_input)

        # 3. Tabular Tower (Residual Connection)
        tab_input = keras.Input(shape=(tab_shape,), name="tabular")
        x3_base = keras.layers.Dense(64, activation="relu")(tab_input)
        x3_res = keras.layers.Dense(64, activation="relu")(x3_base)
        x3 = keras.layers.Add()([x3_base, x3_res])

        # 4. ToF Tower (3D ResNet)
        tof_input = keras.Input(shape=tof_shape, name="tof")
        x4 = self._resnet_block_3d(tof_input, filters=16, stride=2)
        x4 = self._resnet_block_3d(x4, filters=32, stride=2)
        x4 = keras.layers.GlobalAveragePooling3D()(x4)

        merged = keras.layers.concatenate([x1, x2, x3, x4])
        merged = keras.layers.Dense(64, activation="relu")(merged)
        merged = keras.layers.Dropout(0.3)(merged)
        output = keras.layers.Dense(num_classes, activation="softmax")(merged)

        model = keras.Model(
            inputs=[sensor_input, demo_input, tab_input, tof_input], outputs=output
        )
        model.compile(
            optimizer=keras.optimizers.Adam(),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        print(model.summary())
        return model

    def train(self, data: Dict[str, np.ndarray], epochs: int = 50, batch_size: int = 32) -> keras.callbacks.History:
        print("=== データ型確認 ===")
        print(f"X_sensor dtype: {data['sensor'].dtype}")
        print(f"X_demo dtype: {data['demographics'].dtype}")
        print(f"X_tabular dtype: {data['tabular'].dtype}")
        print(f"X_tof dtype: {data['tof'].dtype}")
        print(f"y dtype: {data['labels'].dtype}")
        print(f"y unique values: {np.unique(data['labels'])}")        

        X_s = data["sensor"]
        X_d = data["demographics"]
        X_t = data["tabular"]
        X_f = data["tof"]
        y = data["labels"]

        Xs_train, Xs_val, yd_train, yd_val = train_test_split(
            np.arange(len(y)), y, test_size=0.2, stratify=y, random_state=42
        )
        model = self.build_multimodal_model(
            sensor_shape=X_s.shape[1:],
            demo_shape=X_d.shape[1],
            tab_shape=X_t.shape[1],
            tof_shape=X_f.shape[1:],
            num_classes=len(np.unique(y)),
        )
        history = model.fit(
            [X_s[Xs_train], X_d[Xs_train], X_t[Xs_train], X_f[Xs_train]],
            yd_train,
            validation_data=(
                [X_s[Xs_val], X_d[Xs_val], X_t[Xs_val], X_f[Xs_val]],
                yd_val,
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
            verbose=1,
        )
        self.model = model
        self.history = history
        return history

    def evaluate(
        self,
        data: Dict[str, np.ndarray],
        *,
        preprocessor_path: str | Path | None = None,
    ) -> Dict[str, Any]:
        X_s = data["sensor"]
        X_d = data["demographics"]
        X_t = data["tabular"]
        X_f = data["tof"]
        y = data["labels"]

        preds = self.model.predict([X_s, X_d, X_t, X_f])
        pred_labels = preds.argmax(axis=1)

        label_encoder = None
        if preprocessor_path is None:
            preprocessor_path = self.data_dir / "preprocessor.pkl"
        try:
            pp_path = Path(preprocessor_path)
            if pp_path.exists():
                pp = Preprocessor.load(pp_path)
                label_encoder = getattr(pp, "label_encoder", None)
        except Exception as e:
            print(f"label_encoder 読み込み失敗: {e}")

        cmi_score, binary_f1, macro_f1, test_accuracy = calculate_cmi_score(
            pred_labels,
            y,
            label_encoder=label_encoder,
        )

        report = classification_report(y, pred_labels, output_dict=True)

        results = {
            "cmi_score": float(cmi_score),
            "binary_f1": float(binary_f1),
            "macro_f1": float(macro_f1),
            "test_accuracy": float(test_accuracy),
            "report": report,
        }

        print(
            f"CMI Score: {cmi_score:.4f} | Binary F1: {binary_f1:.4f} | "
            f"Macro F1: {macro_f1:.4f} | Acc: {test_accuracy:.4f}"
        )

        return results

    def save_model(self, path: str | Path | None = None) -> None:
        save_path: Path
        if path is None:
            save_path = self.model_dir / "multimodal_model.keras"
        else:
            save_path = Path(path)
        self.model.save(save_path)
        print(f"モデル保存: {save_path}")

    def save_training_history(self, path: str | Path | None = None) -> Path:
        save_path: Path
        if path is None:
            save_path = self.result_dir / "training_history.json"
        else:
            save_path = Path(path)

        if self.history is None:
            raise ValueError("history is not set")

        if hasattr(self.history, "history"):
            history_dict = self.history.history
        else:
            history_dict = self.history

        with open(save_path, "w") as f:
            json.dump(history_dict, f, indent=2)

        print(f"学習履歴保存: {save_path}")
        return save_path

    def save_evaluation_results(
        self, results: dict, path: str | Path | None = None
    ) -> None:
        save_path: Path
        if path is None:
            save_path = self.result_dir / "evaluation_results.json"
        else:
            save_path = Path(path)

        def _convert(obj: Any):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, float)):
                return float(obj)
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            return obj

        serializable = {k: _convert(v) for k, v in results.items()}
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        print(f"評価結果保存: {save_path}")


if __name__ == "__main__":
    trainer = MultimodalTrainer()
    try:
        data = trainer.load_all_data()
        trainer.train(data, epochs=5)
        trainer.save_model()
        results = trainer.evaluate(data)
        print(results)
    except Exception as e:
        print(f"エラー: {e}")