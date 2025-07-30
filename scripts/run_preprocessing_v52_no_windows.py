#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Preprocessing pipeline for v52 (no windows).

This script builds sensor sequences with T_mask/V_mask and ToF voxel tensors
following the rules described in the prompt. Basic cleaning utilities from
``src.utils.pipeline`` and ``src.utils.preprocessing`` are reused.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

from src.utils.pipeline import Preprocessor


# -------------------------------------------------------------
# Utility
# -------------------------------------------------------------

def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_sensor_sequences(
    df_raw: pd.DataFrame,
    df_interp: pd.DataFrame,
    sensor_cols: List[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[dict]]:
    """Return normalized sequences and masks."""
    groups_raw = list(df_raw.groupby(["subject", "sequence_id"], sort=False))
    groups_interp = list(df_interp.groupby(["subject", "sequence_id"], sort=False))
    assert [k for k, _ in groups_raw] == [k for k, _ in groups_interp]
    max_len = max(len(g) for _, g in groups_interp)

    # statistics using valid values only
    valid_by_feature: List[List[float]] = [[] for _ in sensor_cols]
    for (_, g_raw), (_, g_interp) in zip(groups_raw, groups_interp):
        arr_raw = g_raw[sensor_cols].to_numpy(np.float32)
        arr_interp = g_interp[sensor_cols].to_numpy(np.float32)
        for j in range(len(sensor_cols)):
            mask = ~np.isnan(arr_raw[:, j])
            if mask.any():
                valid_by_feature[j].append(arr_interp[mask, j])
    means = np.zeros(len(sensor_cols), dtype=np.float32)
    stds = np.ones(len(sensor_cols), dtype=np.float32)
    for j, vals in enumerate(valid_by_feature):
        if vals:
            v = np.concatenate(vals)
            means[j] = np.mean(v)
            std = np.std(v)
            stds[j] = std if std > 0 else 1.0

    sequences, t_masks, v_masks, labels, info = [], [], [], [], []
    for (key_raw, g_raw), (_, g_interp) in zip(groups_raw, groups_interp):
        subject, seq_id = key_raw
        arr_raw = g_raw[sensor_cols].to_numpy(np.float32)
        arr_interp = g_interp[sensor_cols].to_numpy(np.float32)
        label = g_raw["gesture"].iloc[0] if "gesture" in g_raw.columns else -1

        v_mask = ~np.isnan(arr_raw)
        norm = np.zeros_like(arr_interp)
        for j in range(len(sensor_cols)):
            valid = v_mask[:, j]
            norm[valid, j] = (arr_interp[valid, j] - means[j]) / stds[j]
        norm[~v_mask] = 0.0

        pad_len = max_len - len(arr_interp)
        if pad_len > 0:
            pad_arr = np.zeros((pad_len, len(sensor_cols)), dtype=np.float32)
            pad_mask = np.zeros((pad_len, len(sensor_cols)), dtype=bool)
            norm = np.vstack([norm, pad_arr])
            v_mask = np.vstack([v_mask, pad_mask])
        t_mask = np.zeros(max_len, dtype=np.float32)
        t_mask[: len(arr_interp)] = 1.0

        sequences.append(norm)
        t_masks.append(t_mask)
        v_masks.append(v_mask.astype(np.float32))
        labels.append(label)
        info.append({"subject": subject, "sequence_id": seq_id, "length": len(arr_interp)})

    return (
        np.stack(sequences),
        np.stack(t_masks),
        np.stack(v_masks),
        np.asarray(labels, dtype=np.int64),
        info,
    )


def build_tof_sequences(
    df: pd.DataFrame, depth: int, height: int, width: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized ToF voxel tensor and mask."""
    tof_cols = [f"tof_{d}_v{i}" for d in range(1, depth + 1) for i in range(height * width)]
    groups = list(df.groupby(["subject", "sequence_id"], sort=False))
    max_len = max(len(g) for _, g in groups)

    valid_vals = []
    for _, g in groups:
        arr = np.stack(
            [g[f"tof_{d}_v{i}"].to_numpy(np.float32) for d in range(1, depth + 1) for i in range(height * width)],
            axis=1,
        ).reshape(len(g), depth, height, width)
        refl = (arr >= 0) & (arr <= 254)
        valid_vals.append(arr[refl])
    if valid_vals:
        all_valid = np.concatenate(valid_vals)
        mean = float(np.mean(all_valid))
        std = float(np.std(all_valid))
    else:
        mean, std = 0.0, 1.0
    if std == 0:
        std = 1.0

    sequences, masks = [], []
    for (_, g) in groups:
        arr = np.stack(
            [g[f"tof_{d}_v{i}"].to_numpy(np.float32) for d in range(1, depth + 1) for i in range(height * width)],
            axis=1,
        ).reshape(len(g), depth, height, width)
        refl = (arr >= 0) & (arr <= 254)
        refl_none = arr == -1
        invalid = (arr <= -2) | np.isnan(arr)

        mask = refl.astype(np.float32)
        val = arr.copy()
        val[refl_none] = 254
        val[invalid] = -999
        norm = val.copy()
        norm[refl] = (val[refl] - mean) / std

        pad_len = max_len - len(arr)
        if pad_len > 0:
            pad_val = np.full((pad_len, depth, height, width), -999, dtype=np.float32)
            pad_mask = np.zeros((pad_len, depth, height, width), dtype=np.float32)
            norm = np.vstack([norm, pad_val])
            mask = np.vstack([mask, pad_mask])
        sequences.append(norm)
        masks.append(mask)

    return np.stack(sequences), np.stack(masks)


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--config", default="config/config_v52.yaml")
    parser.add_argument("--mode", choices=["train", "predict"], default="train")
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    data_dir = Path(config.get("data_dir", "data"))
    out_dir = Path(config.get("output_dir", "output/experiments")) / args.experiment_name / "preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Preprocessor instances for cleaning (with and without interpolation)
    pp_raw = Preprocessor(config, use_windows=False, use_interp_cleaning=False)
    pp_interp = Preprocessor(config, use_windows=False, use_interp_cleaning=True)

    if args.mode == "train":
        df = pd.read_csv(data_dir / "train.csv")
        demo = pd.read_csv(data_dir / "train_demographics.csv")
        df = df.merge(demo, on="subject", how="left")
        df_raw = pp_raw._maybe_clean(df)
        df_interp = pp_interp._maybe_clean(df)

        sensor_cols = (
            config.get("sensor_acc_cols", [])
            + config.get("sensor_rot_cols", [])
            + config.get("sensor_thm_cols", [])
        )
        depth = config.get("preprocessing", {}).get("tof_depth", 5)
        height = config.get("preprocessing", {}).get("tof_height", 8)
        width = config.get("preprocessing", {}).get("tof_width", 8)

        X_seq, T_mask, V_mask, y, info = build_sensor_sequences(df_raw, df_interp, sensor_cols)
        X_tof, tof_mask = build_tof_sequences(df_interp, depth, height, width)

        np.save(out_dir / "train_sequences.npy", X_seq)
        np.save(out_dir / "train_t_mask.npy", T_mask)
        np.save(out_dir / "train_v_mask.npy", V_mask)
        np.save(out_dir / "train_tof.npy", X_tof)
        np.save(out_dir / "train_tof_mask.npy", tof_mask)
        np.save(out_dir / "train_labels.npy", y)
        with open(out_dir / "train_info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
    else:
        df = pd.read_csv(data_dir / "test.csv")
        demo = pd.read_csv(data_dir / "test_demographics.csv")
        df = df.merge(demo, on="subject", how="left")
        df_raw = pp_raw._maybe_clean(df)
        df_interp = pp_interp._maybe_clean(df)

        sensor_cols = (
            config.get("sensor_acc_cols", [])
            + config.get("sensor_rot_cols", [])
            + config.get("sensor_thm_cols", [])
        )
        depth = config.get("preprocessing", {}).get("tof_depth", 5)
        height = config.get("preprocessing", {}).get("tof_height", 8)
        width = config.get("preprocessing", {}).get("tof_width", 8)

        X_seq, T_mask, V_mask, y, info = build_sensor_sequences(df_raw, df_interp, sensor_cols)
        X_tof, tof_mask = build_tof_sequences(df_interp, depth, height, width)

        np.save(out_dir / "predict_sequences.npy", X_seq)
        np.save(out_dir / "predict_t_mask.npy", T_mask)
        np.save(out_dir / "predict_v_mask.npy", V_mask)
        np.save(out_dir / "predict_tof.npy", X_tof)
        np.save(out_dir / "predict_tof_mask.npy", tof_mask)
        np.save(out_dir / "predict_labels.npy", y)
        with open(out_dir / "predict_info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
