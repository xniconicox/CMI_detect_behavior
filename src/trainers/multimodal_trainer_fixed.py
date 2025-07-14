"""修正版多モダリティ学習用トレーナー

IMUウィンドウ、人口統計、表形式特徴量、ToFボクセルの4種類の前処理済みデータを
読み込んでタワー型のニューラルネットワークで学習を行う。
NaN問題と学習率問題を修正。
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf


class MultimodalTrainerFixed:
    """修正版多モダリティモデル学習管理クラス"""

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

        # データの正規化とNaNチェック
        print("データ正規化とNaNチェック中...")
        
        # センサーデータの正規化
        sensor_shape = X_sensor.shape
        X_sensor_flat = X_sensor.reshape(-1, X_sensor.shape[-1])
        X_sensor_flat = (X_sensor_flat - np.mean(X_sensor_flat, axis=0)) / (np.std(X_sensor_flat, axis=0) + 1e-8)
        X_sensor = X_sensor_flat.reshape(sensor_shape)
        
        # 人口統計データの正規化
        X_demo = (X_demo - np.mean(X_demo, axis=0)) / (np.std(X_demo, axis=0) + 1e-8)
        
        # 表形式データの正規化
        X_tab = (X_tab - np.mean(X_tab, axis=0)) / (np.std(X_tab, axis=0) + 1e-8)
        
        # ToFデータの正規化
        tof_shape = X_tof.shape
        X_tof_flat = X_tof.reshape(-1, X_tof.shape[-1])
        X_tof_flat = (X_tof_flat - np.mean(X_tof_flat, axis=0)) / (np.std(X_tof_flat, axis=0) + 1e-8)
        X_tof = X_tof_flat.reshape(tof_shape)

        # NaNチェックと修正
        for name, data in [("センサー", X_sensor), ("人口統計", X_demo), ("表形式", X_tab), ("ToF", X_tof)]:
            if np.isnan(data).any():
                print(f"⚠️ {name}データにNaNを検出、0で置換")
                data = np.nan_to_num(data, nan=0.0)

        # ラベルエンコーディング（文字列ラベルを数値に変換）
        if y.dtype == object or y.dtype.kind in 'U':  # 文字列型の場合
            print("ラベルエンコーディング実行中...")
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            print(f"ラベルクラス: {label_encoder.classes_}")
            self.label_encoder = label_encoder
        else:
            print("ラベルは既に数値型です")
            self.label_encoder = None

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
        """タワー型統合ネットワークを構築（修正版）"""
        # センサー時系列タワー
        sensor_input = tf.keras.Input(shape=sensor_shape, name="sensor")
        x1 = tf.keras.layers.Masking()(sensor_input)
        x1 = tf.keras.layers.LSTM(64, return_sequences=True)(x1)
        x1 = tf.keras.layers.LSTM(32)(x1)

        # 人口統計タワー
        demo_input = tf.keras.Input(shape=(demo_shape,), name="demo")
        x2 = tf.keras.layers.Dense(32, activation="relu")(demo_input)
        x2 = tf.keras.layers.Dropout(0.2)(x2)

        # 表形式タワー
        tab_input = tf.keras.Input(shape=(tab_shape,), name="tabular")
        x3 = tf.keras.layers.Dense(64, activation="relu")(tab_input)
        x3 = tf.keras.layers.Dropout(0.2)(x3)

        # ToFボクセルタワー
        tof_input = tf.keras.Input(shape=tof_shape, name="tof")
        x4 = tf.keras.layers.Conv3D(16, 3, activation="relu", padding="same")(tof_input)
        x4 = tf.keras.layers.BatchNormalization()(x4)
        x4 = tf.keras.layers.MaxPooling3D()(x4)
        x4 = tf.keras.layers.Conv3D(32, 3, activation="relu", padding="same")(x4)
        x4 = tf.keras.layers.BatchNormalization()(x4)
        x4 = tf.keras.layers.GlobalAveragePooling3D()(x4)

        # 統合
        merged = tf.keras.layers.concatenate([x1, x2, x3, x4])
        merged = tf.keras.layers.Dense(128, activation="relu")(merged)
        merged = tf.keras.layers.Dropout(0.3)(merged)
        merged = tf.keras.layers.Dense(64, activation="relu")(merged)
        merged = tf.keras.layers.Dropout(0.3)(merged)
        output = tf.keras.layers.Dense(num_classes, activation="softmax")(merged)

        model = tf.keras.Model(
            inputs=[sensor_input, demo_input, tab_input, tof_input], outputs=output
        )
        
        # 学習率を下げて安定化
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        print(model.summary())
        return model

    # ------------------------------------------------------------------
    def train(self, data: Dict[str, np.ndarray], epochs: int = 50, batch_size: int = 32) -> tf.keras.callbacks.History:
        """モデルを学習（修正版）"""
        X_s = data["sensor"]
        X_d = data["demographics"]
        X_t = data["tabular"]
        X_f = data["tof"]
        y = data["labels"]

        # データ分割
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
        
        # コールバックを追加
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-6),
            tf.keras.callbacks.ModelCheckpoint(
                str(self.model_dir / "best_model.keras"),
                save_best_only=True,
                monitor='val_accuracy'
            )
        ]
        
        history = model.fit(
            [X_s[Xs_train], X_d[Xs_train], X_t[Xs_train], X_f[Xs_train]],
            yd_train,
            validation_data=(
                [X_s[Xs_val], X_d[Xs_val], X_t[Xs_val], X_f[Xs_val]],
                yd_val,
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
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
        
        # ラベルエンコーダーも保存
        if hasattr(self, 'label_encoder') and self.label_encoder is not None:
            encoder_path = self.model_dir / "label_encoder.pkl"
            with open(encoder_path, "wb") as f:
                pickle.dump(self.label_encoder, f)
            print(f"ラベルエンコーダー保存: {encoder_path}")


if __name__ == "__main__":
    trainer = MultimodalTrainerFixed()
    try:
        data = trainer.load_all_data()
        trainer.train(data, epochs=5)
        trainer.save_model()
        results = trainer.evaluate(data)
        print(results)
    except Exception as e:
        print(f"エラー: {e}") 