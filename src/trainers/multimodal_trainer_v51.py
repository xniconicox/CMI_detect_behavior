
"""多モダリティ学習用トレーナー

IMUウィンドウ、人口統計、表形式特徴量、ToFボクセルの4種類の前処理済みデータに加え、
ToFイベント特徴量や温度勾配特徴量などの拡張タブular特徴量を利用して学習を行う。
少数クラスを考慮するため、``_train_fold`` では
``sklearn.utils.class_weight.compute_class_weight`` を使ってクラス重みを計算し、
``model.fit()`` に ``class_weight`` として渡す。
"""

from __future__ import annotations
import re
import os

import json
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    train_test_split,
    StratifiedGroupKFold,
)
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight

from src.utils.cmi_evaluation import calculate_cmi_score
from src.utils.pipeline import Preprocessor
import tensorflow as tf
from tensorflow import keras


class MultimodalTrainerV51:
    """Trainer for v51 sequence-level data."""


    def __init__(self, experiment_name: str = "multimodal") -> None:
        self.experiment_name = experiment_name
        self.base_dir = Path("output/experiments") / experiment_name
        self.data_dir = self.base_dir / "preprocessed"
        self.model_dir = self.base_dir / "models_v51"
        self.result_dir = self.base_dir / "results_v51"

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)

        print(f"Experiment name: {self.experiment_name}")
        print(f"Data directory: {self.data_dir}")

    def _create_mask(self, x: np.ndarray, padding_value: float = 0.0) -> np.ndarray:
        """パディング位置を0、実データを1とするマスクを生成"""
        mask = np.any(x != padding_value, axis=-1).astype(np.float32)
        return mask

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
        y = keras.layers.SpatialDropout3D(0.2)(y)
        return y

    def load_all_data(self) -> Dict[str, np.ndarray]:
        """各モダリティの前処理済みデータをすべて読み込む

        追加で ``train_info.pkl`` を読み込み、``subject`` または ``sequence_id``
        の配列を ``groups`` キーで返す。
        """
        print("前処理済みデータを読み込み中...")

        def _remove_flip_suffix(subject):
            # 末尾の"_flip"を除去
            return re.sub(r"_flip$", "", str(subject))
        def _load(name: str) -> np.ndarray:
            npy = self.data_dir / f"{name}.npy"
            pkl = self.data_dir / f"{name}.pkl"
            if npy.exists():
                return np.load(npy)
            if pkl.exists():
                with open(pkl, "rb") as f:
                    return pickle.load(f)
            raise FileNotFoundError(f"{name} ファイルが見つかりません")

        X_sensor = _load("train_sequences")
        info = _load("train_info")
        sensor_mask = self._create_mask(X_sensor, padding_value=-1)
        X_feat = _load("train_features")
        X_feat = X_feat.astype(np.float32)
        X_tof = _load("train_tof_voxels")
        y = _load("train_labels")

        if isinstance(info, list) and len(info) > 0 and isinstance(info[0], dict):
            groups = np.array([
                _remove_flip_suffix(d.get("subject")) for d in info
            ])
        else:
            groups = np.asarray(info)

        print(f"センサー: {X_sensor.shape}")
        print(f"特徴量: {X_feat.shape}")
        print(f"ToF: {X_tof.shape}")
        print(f"ラベル: {y.shape}")
        print(f"グループ数: {len(groups)}")

        return {
            "sensor": X_sensor,
            "features": X_feat,
            "tof": X_tof,
            "labels": y,
            "groups": groups,
            "sensor_mask": sensor_mask,
        }

    def build_multimodal_model(
        self,
        sensor_shape: tuple,
        feat_shape: int,
        tof_shape: tuple,
        num_classes: int,
        *,
        use_attention: bool = False,
    ) -> keras.Model:
        """タワー型統合ネットワークを構築"""
        # 1. IMU Tower (Bidirectional LSTM)
        sensor_input = keras.Input(shape=sensor_shape, name="sensor")
        sensor_mask_input = keras.Input(shape=(None,), name="sensor_mask")
        x1 = keras.layers.Masking()(sensor_input)
        x1 = keras.layers.Bidirectional(keras.layers.LSTM(32))(x1, mask=sensor_mask_input)

        # 2. Integrated feature tower
        feat_input = keras.Input(shape=(feat_shape,), name="features")
        x2 = keras.layers.Dense(128, activation="relu")(feat_input)

        # 4. ToF Tower (3D ResNet)
        tof_input = keras.Input(shape=tof_shape, name="tof")
        x_toF = keras.layers.Lambda(lambda t: tf.where(t == -1, 0.0, t))(tof_input)
        x4 = self._resnet_block_3d(x_toF, filters=16, stride=2)
        x4 = self._resnet_block_3d(x4, filters=32, stride=2)
        if use_attention:
            x4 = keras.layers.SpatialDropout3D(0.2)(x4)
        x4 = keras.layers.GlobalAveragePooling3D()(x4)

        if use_attention:
            q = keras.layers.Reshape((1, 64))(keras.layers.Dense(64)(x1))
            v = keras.layers.Reshape((1, 64))(keras.layers.Dense(64)(x4))
            attn = keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)(q, v)
            attn = keras.layers.Flatten()(attn)
            merged = keras.layers.concatenate([attn, x2])
        else:
            merged = keras.layers.concatenate([x1, x2, x4])
        merged = keras.layers.Dense(64, activation="relu")(merged)
        merged = keras.layers.Dropout(0.3)(merged)
        output = keras.layers.Dense(num_classes, activation="softmax")(merged)

        model = keras.Model(
            inputs=[sensor_input, sensor_mask_input, feat_input, tof_input],
            outputs=output,
        )

        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3,
            decay_steps=1000,
            decay_rate=0.96,
            staircase=True,
        )
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        print(model.summary())
        return model

    def _train_fold(
        self,
        X_s: np.ndarray,
        X_feat: np.ndarray,
        X_tof: np.ndarray,
        m_s: np.ndarray,
        y: np.ndarray,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        *,
        epochs: int = 50,
        batch_size: int = 32,
        use_attention: bool = False,
    ) -> tuple[keras.Model, keras.callbacks.History, float, float]:
        """単一foldでモデルを学習しF1スコアを返す"""
        model = self.build_multimodal_model(
            sensor_shape=(None, X_s.shape[2]),
            feat_shape=X_feat.shape[1],
            tof_shape=X_tof.shape[1:],
            num_classes=len(np.unique(y)),
            use_attention=use_attention,
        )
        classes = np.unique(y)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y[train_idx])
        class_weight = {cls: w for cls, w in zip(classes, weights)}
        history = model.fit(
            [X_s[train_idx], m_s[train_idx], X_feat[train_idx], X_tof[train_idx]],
            y[train_idx],
            validation_data=(
                [X_s[val_idx], m_s[val_idx], X_feat[val_idx], X_tof[val_idx]],
                y[val_idx],
            ),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight,
            callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
            verbose=1,
        )
        preds = model.predict([X_s[val_idx], m_s[val_idx], X_feat[val_idx], X_tof[val_idx]])
        pred_labels = preds.argmax(axis=1)
        f1 = f1_score(y[val_idx], pred_labels, average="macro")
        
        # CMI-score計算
        cmi_score, _, _, _ = calculate_cmi_score(pred_labels, y[val_idx])
        
        return model, history, float(f1), float(cmi_score)

    def train(
        self,
        data: Dict[str, np.ndarray],
        epochs: int = 50,
        batch_size: int = 32,
        *,
        use_attention: bool = False,
    ) -> keras.callbacks.History:
        print("=== データ型確認 ===")
        print(f"X_sensor dtype: {data['sensor'].dtype}")
        print(f"X_features dtype: {data['features'].dtype}")
        print(f"X_tof dtype: {data['tof'].dtype}")
        print(f"y dtype: {data['labels'].dtype}")
        print(f"y unique values: {np.unique(data['labels'])}")        

        X_s = data["sensor"]
        m_s = data.get("sensor_mask")
        if m_s is None:
            m_s = self._create_mask(X_s)
        X_feat = data["features"]
        X_tof = data["tof"]
        y = data["labels"]

        Xs_train, Xs_val, yd_train, yd_val = train_test_split(
            np.arange(len(y)), y, test_size=0.2, stratify=y, random_state=42
        )
        model, history, _, _ = self._train_fold(
            X_s,
            X_feat,
            X_tof,
            m_s,
            y,
            Xs_train,
            Xs_val,
            epochs=epochs,
            batch_size=batch_size,
            use_attention=use_attention,
        )
        self.model = model
        self.history = history
        return history

    def train_cross_validation(
        self,
        data: Dict[str, np.ndarray],
        epochs: int = 50,
        batch_size: int = 32,
        n_splits: int = 5,
        groups: np.ndarray | None = None,
        *,
        use_attention: bool = False,
    ) -> Dict[str, Any]:
        """StratifiedGroupKFold を用いたクロスバリデーション学習"""
        print("=== データ型確認 ===")
        print(f"X_sensor dtype: {data['sensor'].dtype}")
        print(f"X_features dtype: {data['features'].dtype}")
        print(f"X_tof dtype: {data['tof'].dtype}")
        print(f"y dtype: {data['labels'].dtype}")
        print(f"y unique values: {np.unique(data['labels'])}")

        X_s = data["sensor"]
        m_s = data.get("sensor_mask")
        if m_s is None:
            m_s = self._create_mask(X_s)
        X_feat = data["features"]
        X_tof = data["tof"]
        y = data["labels"]

        if groups is None:
            groups = data.get("groups")

        skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_scores: list[float] = []
        fold_cmi_scores: list[float] = []

        fold_histories = []
        for fold, (tr_idx, val_idx) in enumerate(skf.split(np.arange(len(y)), y, groups), 1):
            print(f"Fold {fold}/{n_splits}")
            model, history, f1, cmi_score = self._train_fold(
                X_s,
                X_feat,
                X_tof,
                m_s,
                y,
                tr_idx,
                val_idx,
                epochs=epochs,
                batch_size=batch_size,
                use_attention=use_attention,
            )
            fold_scores.append(f1)
            fold_cmi_scores.append(cmi_score)
            fold_histories.append(history)
            
            # モデル保存
            model_path = self.model_dir / f"multimodal_model_v51_fold{fold}.keras"
            model.save(model_path)
            print(f"モデル保存: {model_path}")
            
            # 各foldの学習履歴保存
            history_path = self.result_dir / f"training_history_fold{fold}.json"
            self._save_history(history, history_path)

        mean_score = float(np.mean(fold_scores))
        std_score = float(np.std(fold_scores))
        mean_cmi = float(np.mean(fold_cmi_scores))
        std_cmi = float(np.std(fold_cmi_scores))
        cv_results = {
            "fold_scores": [float(s) for s in fold_scores],
            "fold_cmi_scores": [float(s) for s in fold_cmi_scores],
            "mean_f1": mean_score,
            "std_f1": std_score,
            "mean_cmi": mean_cmi,
            "std_cmi": std_cmi,
        }

        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        with open(results_dir / "cv_results.json", "w", encoding="utf-8") as f:
            json.dump(cv_results, f, ensure_ascii=False, indent=2)
        print(f"Cross-validation results saved: {results_dir / 'cv_results.json'}")

        self.cv_results = cv_results
        
        # 学習曲線とクロスバリデーション結果をプロット
        self.plot_training_curves(fold_histories)
        self.plot_cross_validation_results(cv_results)
        
        # 最後のfoldのモデルと履歴をselfにセット
        self.model = model
        self.history = fold_histories[-1] if fold_histories else None
        
        return cv_results

    def evaluate(
        self,
        data: Dict[str, np.ndarray],
        *,
        preprocessor_path: str | Path | None = None,
    ) -> Dict[str, Any]:
        X_s = data["sensor"]
        m_s = data.get("sensor_mask")
        if m_s is None:
            m_s = self._create_mask(X_s)
        X_feat = data["features"]
        X_tof = data["tof"]
        y = data["labels"]

        preds = self.model.predict([X_s, m_s, X_feat, X_tof])
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

    def _save_history(self, history, path: Path) -> None:
        """学習履歴を保存する内部メソッド"""
        if hasattr(history, "history"):
            history_dict = history.history
        else:
            history_dict = history

        with open(path, "w") as f:
            json.dump(history_dict, f, indent=2)
        print(f"学習履歴保存: {path}")

    def save_training_history(self, path: str | Path | None = None) -> Path:
        save_path: Path
        if path is None:
            save_path = self.result_dir / "training_history.json"
        else:
            save_path = Path(path)

        if self.history is None:
            print("⚠️  history is not set, skipping training history save")
            return save_path

        self._save_history(self.history, save_path)
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

    def plot_training_curves(self, fold_histories: list | None = None, save_path: str | Path | None = None) -> None:
        """学習曲線をプロットして保存する"""
        if fold_histories is None:
            if hasattr(self, 'history') and self.history is not None:
                fold_histories = [self.history]
            else:
                print("学習履歴が見つかりません")
                return

        if save_path is None:
            save_path = self.result_dir / "training_curves.png"
        else:
            save_path = Path(save_path)

        # プロット設定
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Curves (Multimodal Model V50)', fontsize=16)

        metrics = ['loss', 'accuracy']
        for i, metric in enumerate(metrics):
            # 訓練データ
            ax = axes[i, 0]
            for fold, history in enumerate(fold_histories, 1):
                if hasattr(history, 'history'):
                    hist = history.history
                else:
                    hist = history
                
                if metric in hist:
                    ax.plot(hist[metric], label=f'Fold {fold}', alpha=0.7)
            
            ax.set_title(f'Training {metric.capitalize()}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.legend()
            ax.grid(True, alpha=0.3)

            # 検証データ
            ax = axes[i, 1]
            for fold, history in enumerate(fold_histories, 1):
                if hasattr(history, 'history'):
                    hist = history.history
                else:
                    hist = history
                
                val_metric = f'val_{metric}'
                if val_metric in hist:
                    ax.plot(hist[val_metric], label=f'Fold {fold}', alpha=0.7)
            
            ax.set_title(f'Validation {metric.capitalize()}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.capitalize())
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"学習曲線保存: {save_path}")
        plt.close()

    def plot_cross_validation_results(self, cv_results: dict | None = None, save_path: str | Path | None = None) -> None:
        """クロスバリデーション結果をプロットする"""
        if cv_results is None:
            if hasattr(self, 'cv_results'):
                cv_results = self.cv_results
            else:
                print("クロスバリデーション結果が見つかりません")
                return

        if save_path is None:
            save_path = self.result_dir / "cv_results.png"
        else:
            save_path = Path(save_path)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Cross-Validation Results', fontsize=16)

        # Fold別F1スコア
        fold_scores = cv_results.get('fold_scores', [])
        if fold_scores:
            ax1.bar(range(1, len(fold_scores) + 1), fold_scores, alpha=0.7, color='blue')
            ax1.axhline(y=cv_results.get('mean_f1', 0), color='red', linestyle='--', 
                       label=f'Mean: {cv_results.get("mean_f1", 0):.4f}')
            ax1.set_xlabel('Fold')
            ax1.set_ylabel('F1 Score')
            ax1.set_title('F1 Score by Fold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        # Fold別CMIスコア
        fold_cmi_scores = cv_results.get('fold_cmi_scores', [])
        if fold_cmi_scores:
            ax2.bar(range(1, len(fold_cmi_scores) + 1), fold_cmi_scores, alpha=0.7, color='green')
            ax2.axhline(y=cv_results.get('mean_cmi', 0), color='red', linestyle='--', 
                       label=f'Mean: {cv_results.get("mean_cmi", 0):.4f}')
            ax2.set_xlabel('Fold')
            ax2.set_ylabel('CMI Score')
            ax2.set_title('CMI Score by Fold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # F1スコア統計情報
        mean_f1 = cv_results.get('mean_f1', 0)
        std_f1 = cv_results.get('std_f1', 0)
        
        ax3.text(0.1, 0.8, f'Mean F1: {mean_f1:.4f}', fontsize=12, transform=ax3.transAxes)
        ax3.text(0.1, 0.7, f'Std F1: {std_f1:.4f}', fontsize=12, transform=ax3.transAxes)
        ax3.text(0.1, 0.6, f'Min F1: {min(fold_scores):.4f}', fontsize=12, transform=ax3.transAxes)
        ax3.text(0.1, 0.5, f'Max F1: {max(fold_scores):.4f}', fontsize=12, transform=ax3.transAxes)
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.set_title('F1 Score Statistics')
        ax3.axis('off')

        # CMIスコア統計情報
        mean_cmi = cv_results.get('mean_cmi', 0)
        std_cmi = cv_results.get('std_cmi', 0)
        
        ax4.text(0.1, 0.8, f'Mean CMI: {mean_cmi:.4f}', fontsize=12, transform=ax4.transAxes)
        ax4.text(0.1, 0.7, f'Std CMI: {std_cmi:.4f}', fontsize=12, transform=ax4.transAxes)
        ax4.text(0.1, 0.6, f'Min CMI: {min(fold_cmi_scores):.4f}', fontsize=12, transform=ax4.transAxes)
        ax4.text(0.1, 0.5, f'Max CMI: {max(fold_cmi_scores):.4f}', fontsize=12, transform=ax4.transAxes)
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.set_title('CMI Score Statistics')
        ax4.axis('off')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"クロスバリデーション結果保存: {save_path}")
        plt.close()


if __name__ == "__main__":
    trainer = MultimodalTrainerV50()
    try:
        data = trainer.load_all_data()
        trainer.train(data, epochs=5)
        trainer.save_model()
        results = trainer.evaluate(data)
        print(results)
    except Exception as e:
        print(f"エラー: {e}")
