#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simplified preprocessing pipeline for v52.

This version follows the masking rules described in the prompt.
It generates T_mask/V_mask for IMU sequences and ToF masks without
window segmentation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml


# -------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------

def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_sensor_sequences(df: pd.DataFrame, sensor_cols: List[str]):
    """Return padded sensor sequences with masks."""
    sequences = []
    t_masks = []
    v_masks = []
    labels = []
    info = []

    groups = list(df.groupby(["subject", "sequence_id"]))
    max_len = max(len(g) for _, g in groups)

    # gather all values for scaler
    all_values = []
    for _, g in groups:
        arr = g[sensor_cols].to_numpy(np.float32)
        all_values.append(arr)
    concat = np.vstack(all_values)
    mean = np.nanmean(concat, axis=0)
    std = np.nanstd(concat, axis=0)
    std[std == 0] = 1.0

    for (subject, seq_id), g in groups:
        arr = g[sensor_cols].to_numpy(np.float32)
        label = g["gesture"].iloc[0] if "gesture" in g.columns else -1

        v_mask = ~np.isnan(arr)
        norm = np.zeros_like(arr)
        for j in range(arr.shape[1]):
            valid = ~np.isnan(arr[:, j])
            norm[valid, j] = (arr[valid, j] - mean[j]) / std[j]
        norm[np.isnan(norm)] = 0.0

        pad_len = max_len - len(arr)
        if pad_len > 0:
            pad_arr = np.zeros((pad_len, arr.shape[1]), dtype=np.float32)
            pad_mask = np.zeros((pad_len, arr.shape[1]), dtype=bool)
            norm = np.vstack([norm, pad_arr])
            v_mask = np.vstack([v_mask, pad_mask])
        t_mask = np.zeros(max_len, dtype=np.float32)
        t_mask[: len(g)] = 1.0

        sequences.append(norm)
        t_masks.append(t_mask)
        v_masks.append(v_mask.astype(np.float32))
        labels.append(label)
        info.append({"subject": subject, "sequence_id": seq_id, "length": len(g)})

    return (
        np.stack(sequences),
        np.stack(t_masks),
        np.stack(v_masks),
        np.asarray(labels, dtype=np.int64),
        info,
    )


def build_tof_sequences(df: pd.DataFrame, depth: int, height: int, width: int):
    """Process ToF values and masks."""
    tof_cols = [f"tof_{d}_v{i}" for d in range(1, depth + 1) for i in range(height * width)]
    groups = list(df.groupby(["subject", "sequence_id"]))
    max_len = max(len(g) for _, g in groups)

    sequences = []
    masks = []

    # gather values for scaler
    valid_vals = []
    for _, g in groups:
        arr = np.stack([
            g[f"tof_{d}_v{i}"].to_numpy(np.float32)
            for d in range(1, depth + 1)
            for i in range(height * width)
        ], axis=1).reshape(len(g), depth, height, width)
        refl = (arr >= 0) & (arr <= 254)
        valid_vals.append(arr[refl])
    if valid_vals:
        all_valid = np.concatenate(valid_vals)
        mean = np.mean(all_valid)
        std = np.std(all_valid)
    else:
        mean, std = 0.0, 1.0
    if std == 0:
        std = 1.0

    for (subject, seq_id), g in groups:
        arr = np.stack([
            g[f"tof_{d}_v{i}"].to_numpy(np.float32)
            for d in range(1, depth + 1)
            for i in range(height * width)
        ], axis=1).reshape(len(g), depth, height, width)

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--config", default="config/config_v52.yaml")
    parser.add_argument("--mode", choices=["train", "predict"], default="train")
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    data_dir = Path(config.get("data_dir", "data"))
    out_dir = Path("output/experiments") / args.experiment_name / "preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "train":
        train_df = pd.read_csv(data_dir / "train.csv")
        train_demo = pd.read_csv(data_dir / "train_demographics.csv")
        train_df = train_df.merge(train_demo, on="subject", how="left")

        sensor_cols = (
            config.get("sensor_acc_cols", [])
            + config.get("sensor_rot_cols", [])
            + config.get("sensor_thm_cols", [])
        )
        depth = config.get("preprocessing", {}).get("tof_depth", 5)
        height = config.get("preprocessing", {}).get("tof_height", 8)
        width = config.get("preprocessing", {}).get("tof_width", 8)

        X_seq, T_mask, V_mask, y, info = build_sensor_sequences(train_df, sensor_cols)
        X_tof, tof_mask = build_tof_sequences(train_df, depth, height, width)

        np.save(out_dir / "train_sequences.npy", X_seq)
        np.save(out_dir / "train_t_mask.npy", T_mask)
        np.save(out_dir / "train_v_mask.npy", V_mask)
        np.save(out_dir / "train_tof.npy", X_tof)
        np.save(out_dir / "train_tof_mask.npy", tof_mask)
        np.save(out_dir / "train_labels.npy", y)
        with open(out_dir / "train_info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        print("Saved preprocessing outputs to", out_dir)
    else:
        df = pd.read_csv(data_dir / "test.csv")
        test_demo = pd.read_csv(data_dir / "test_demographics.csv")
        df = df.merge(test_demo, on="subject", how="left")

        sensor_cols = (
            config.get("sensor_acc_cols", [])
            + config.get("sensor_rot_cols", [])
            + config.get("sensor_thm_cols", [])
        )
        depth = config.get("preprocessing", {}).get("tof_depth", 5)
        height = config.get("preprocessing", {}).get("tof_height", 8)
        width = config.get("preprocessing", {}).get("tof_width", 8)

        X_seq, T_mask, V_mask, y, info = build_sensor_sequences(df, sensor_cols)
        X_tof, tof_mask = build_tof_sequences(df, depth, height, width)

        np.save(out_dir / "predict_sequences.npy", X_seq)
        np.save(out_dir / "predict_t_mask.npy", T_mask)
        np.save(out_dir / "predict_v_mask.npy", V_mask)
        np.save(out_dir / "predict_tof.npy", X_tof)
        np.save(out_dir / "predict_tof_mask.npy", tof_mask)
        np.save(out_dir / "predict_labels.npy", y)
        with open(out_dir / "predict_info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        print("Saved prediction outputs to", out_dir)


if __name__ == "__main__":
    main()
