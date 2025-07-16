"""多モダリティ学習用トレーナー

IMUウィンドウ、人口統計、表形式特徴量、ToFボクセルの4種類の前処理済みデータを
読み込んでタワー型のニューラルネットワークで学習を行う。
"""

from __future__ import annotations

import os
import json
import pickle
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import tensorflow as tf


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

        print(f"実験名: {self.experiment_name}")
        print(f"データディレクトリ: {self.data_dir}")

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    def build_multimodal_model(
        self,
        sensor_shape: tuple[int, int],
        demo_shape: int,
        tab_shape: int,
        tof_shape: tuple[int, int, int],
        num_classes: int,
    ) -> tf.keras.Model:
        """タワー型統合ネットワークを構築"""
        # センサー時系列タワー
        sensor_input = tf.keras.Input(shape=sensor_shape, name="sensor")
        x1 = tf.keras.layers.Masking()(sensor_input)
        x1 = tf.keras.layers.LSTM(64)(x1)

        # 人口統計タワー
        demo_input = tf.keras.Input(shape=(demo_shape,), name="demo")
        x2 = tf.keras.layers.Dense(32, activation="relu")(demo_input)

        # 表形式タワー
        tab_input = tf.keras.Input(shape=(tab_shape,), name="tabular")
        x3 = tf.keras.layers.Dense(64, activation="relu")(tab_input)

        # ToFボクセルタワー
        tof_input = tf.keras.Input(shape=tof_shape, name="tof")
        x4 = tf.keras.layers.Conv3D(16, 3, activation="relu", padding="same")(tof_input)
        x4 = tf.keras.layers.MaxPooling3D()(x4)
        x4 = tf.keras.layers.Conv3D(32, 3, activation="relu", padding="same")(x4)
        x4 = tf.keras.layers.GlobalAveragePooling3D()(x4)

        # 統合
        merged = tf.keras.layers.concatenate([x1, x2, x3, x4])
        merged = tf.keras.layers.Dense(64, activation="relu")(merged)
        merged = tf.keras.layers.Dropout(0.3)(merged)
        output = tf.keras.layers.Dense(num_classes, activation="softmax")(merged)

        model = tf.keras.Model(
            inputs=[sensor_input, demo_input, tab_input, tof_input], outputs=output
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        print(model.summary())
        return model

    # ------------------------------------------------------------------
    def train(self, data: Dict[str, np.ndarray], epochs: int = 50, batch_size: int = 32) -> tf.keras.callbacks.History:

        # データ型の確認と修正
        print("=== データ型確認 ===")
        print(f"X_sensor dtype: {data['sensor'].dtype}")
        print(f"X_demo dtype: {data['demographics'].dtype}")
        print(f"X_tabular dtype: {data['tabular'].dtype}")
        print(f"X_tof dtype: {data['tof'].dtype}")
        print(f"y dtype: {data['labels'].dtype}")
        print(f"y unique values: {np.unique(data['labels'])}")        
        
        # """モデルを学習"""
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
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
            verbose=1,
        )
        self.model = model
        self.history = history
        return history

    # ------------------------------------------------------------------
    def evaluate(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """テストデータで評価"""
        X_s = data["sensor"]
        X_d = data["demographics"]
        X_t = data["tabular"]
        X_f = data["tof"]
        y = data["labels"]

        preds = self.model.predict([X_s, X_d, X_t, X_f])
        pred_labels = preds.argmax(axis=1)
        f1 = f1_score(y, pred_labels, average="macro")
        report = classification_report(y, pred_labels, output_dict=True)
        print(f"Macro F1: {f1:.4f}")
        return {"f1_macro": f1, "report": report}

    # ------------------------------------------------------------------
    def save_model(self, path: str | None = None) -> None:
        """モデルを保存"""
        if path is None:
            path = self.model_dir / "multimodal_model.keras"
        else:
            path = Path(path)
        self.model.save(path)
        print(f"モデル保存: {path}")

    def save_training_history(self, path: str | None = None) -> Path:
        """学習履歴をJSON形式で保存"""
        if path is None:
            path = self.result_dir / "training_history.json"
        else:
            path = Path(path)

        if self.history is None:
            raise ValueError("history is not set")

        if hasattr(self.history, "history"):
            history_dict = self.history.history
        else:
            history_dict = self.history

        with open(path, "w") as f:
            json.dump(history_dict, f, indent=2)

        print(f"学習履歴保存: {path}")
        return path

    # ------------------------------------------------------------------
    def save_evaluation_results(
        self, results: Dict[str, Any], path: str | None = None
    ) -> Path:
        """評価結果をJSON形式で保存"""
        if path is None:
            path = self.result_dir / "evaluation_results.json"
        else:
            path = Path(path)

        with open(path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"評価結果保存: {path}")
        return path


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

