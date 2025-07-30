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

from numpy.lib.format import open_memmap
import numpy as np
import pandas as pd
import yaml
from src.utils.pipeline import Preprocessor
from src.utils.feature_engineering import (
    compute_basic_statistics,
    compute_peak_features,
    compute_fft_band_energy,
    compute_wavelet_features,
    compute_persistence_image_features_batch,
)


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
    out_dir: Path,
    prefix: str,
) -> tuple[np.memmap, np.memmap, np.memmap, np.ndarray, List[dict]]:
    """Return normalized sequences and masks saved as memmap."""
    groups_raw = list(df_raw.groupby(["subject", "sequence_id"], sort=False))
    groups_interp = list(df_interp.groupby(["subject", "sequence_id"], sort=False))
    assert [k for k, _ in groups_raw] == [k for k, _ in groups_interp]
    max_len = max(len(g) for _, g in groups_interp)
    n_seq = len(groups_interp)

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

    seq_map = open_memmap(out_dir / f"{prefix}_sequences.npy", mode="w+", dtype=np.float32, shape=(n_seq, max_len, len(sensor_cols)))
    t_map = open_memmap(out_dir / f"{prefix}_t_mask.npy", mode="w+", dtype=np.float32, shape=(n_seq, max_len))
    v_map = open_memmap(out_dir / f"{prefix}_v_mask.npy", mode="w+", dtype=np.float32, shape=(n_seq, max_len, len(sensor_cols)))

    labels = np.zeros(n_seq, dtype=np.int64)
    info: List[dict] = []
    for idx, ((key_raw, g_raw), (_, g_interp)) in enumerate(zip(groups_raw, groups_interp)):
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

        seq_map[idx] = norm
        t_map[idx] = t_mask
        v_map[idx] = v_mask.astype(np.float32)
        labels[idx] = label
        info.append({"subject": subject, "sequence_id": seq_id, "length": len(arr_interp)})

    seq_map.flush()
    t_map.flush()
    v_map.flush()

    return (
        seq_map,
        t_map,
        v_map,
        labels,
        info,
    )


def build_tof_sequences(
    df: pd.DataFrame,
    depth: int,
    height: int,
    width: int,
    out_dir: Path,
    prefix: str,
) -> tuple[np.memmap, np.memmap]:
    """Return normalized ToF voxel tensor and mask saved as memmap."""
    tof_cols = [f"tof_{d}_v{i}" for d in range(1, depth + 1) for i in range(height * width)]
    groups = list(df.groupby(["subject", "sequence_id"], sort=False))
    max_len = max(len(g) for _, g in groups)
    n_seq = len(groups)

    # compute mean/std using valid values only (streaming)
    total = 0.0
    total_sq = 0.0
    count = 0
    for _, g in groups:
        arr = np.stack(
            [g[f"tof_{d}_v{i}"].to_numpy(np.float32) for d in range(1, depth + 1) for i in range(height * width)],
            axis=1,
        ).reshape(len(g), depth, height, width)
        mask = (arr >= 0) & (arr <= 254)
        vals = arr[mask]
        total += float(vals.sum())
        total_sq += float((vals ** 2).sum())
        count += vals.size
    if count > 0:
        mean = total / count
        std = (total_sq / count - mean ** 2) ** 0.5
    else:
        mean, std = 0.0, 1.0
    if std == 0:
        std = 1.0

    tof_map = open_memmap(out_dir / f"{prefix}_tof.npy", mode="w+", dtype=np.float32, shape=(n_seq, max_len, depth, height, width))
    mask_map = open_memmap(out_dir / f"{prefix}_tof_mask.npy", mode="w+", dtype=np.float32, shape=(n_seq, max_len, depth, height, width))

    for idx, (_, g) in enumerate(groups):
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

        tof_map[idx] = norm
        mask_map[idx] = mask

    tof_map.flush()
    mask_map.flush()

    return tof_map, mask_map


def build_tabular_features(
    groups: List[tuple[tuple, pd.DataFrame]],
    sensor_cols: List[str],
    demo_cols: List[str],
    stats: dict | None,
    config: Dict[str, Any],
    out_path: Path,
) -> dict:
    """Compute simple tabular features per sequence."""
    feats = []
    labels = []
    for _, g in groups:
        arr = g[sensor_cols].to_numpy(np.float32)
        seq = arr[np.newaxis, :, :]
        f = [
            compute_basic_statistics(seq)[0],
            compute_peak_features(seq)[0],
            compute_fft_band_energy(
                seq,
                fs=config.get("sampling_rate", 50.0),
                bands=config.get("fft_bands", []),
            )[0],
        ]
        if config.get("use_wavelet_features", False):
            f.append(
                compute_wavelet_features(
                    seq,
                    wavelet=config.get("wavelet", "db4"),
                    level=config.get("wavelet_level", 3),
                )[0]
            )
        if config.get("use_tda_features", False):
            f.append(
                compute_persistence_image_features_batch(
                    seq,
                    dimension=config.get("tda_dimension", 1),
                    n_bins=config.get("tda_bins", 8),
                    sigma=config.get("tda_sigma", 0.1),
                )[0]
            )
        feat_vec = np.hstack(f)
        if demo_cols:
            demo = g[demo_cols].mean().to_numpy(np.float32)
            feat_vec = np.hstack([demo, feat_vec])
        feats.append(feat_vec)
        labels.append(g["gesture"].iloc[0] if "gesture" in g.columns else -1)
    feats = np.asarray(feats, dtype=np.float32)
    if stats is None:
        mean = feats.mean(axis=0)
        std = feats.std(axis=0)
        std[std == 0] = 1.0
        stats = {"mean": mean, "std": std}
    norm = (feats - stats["mean"]) / stats["std"]
    np.save(out_path, norm)
    return {"features": norm, "labels": np.asarray(labels, dtype=np.int64), "stats": stats}


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

        X_seq, T_mask, V_mask, y, info = build_sensor_sequences(
            df_raw, df_interp, sensor_cols, out_dir, "train"
        )
        X_tof, tof_mask = build_tof_sequences(
            df_interp, depth, height, width, out_dir, "train"
        )

        groups_interp = list(df_interp.groupby(["subject", "sequence_id"], sort=False))
        demo_cols = config.get("demographics_cols", [])
        tab_cfg = config.get("preprocessing", {})
        tab_res = build_tabular_features(
            groups_interp,
            sensor_cols,
            demo_cols,
            None,
            tab_cfg,
            out_dir / "train_features.npy",
        )
        np.savez(out_dir / "tabular_stats.npz", mean=tab_res["stats"]["mean"], std=tab_res["stats"]["std"])

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

        X_seq, T_mask, V_mask, y, info = build_sensor_sequences(
            df_raw, df_interp, sensor_cols, out_dir, "predict"
        )
        X_tof, tof_mask = build_tof_sequences(
            df_interp, depth, height, width, out_dir, "predict"
        )

        stats_npz = np.load(out_dir / "tabular_stats.npz")
        stats = {"mean": stats_npz["mean"], "std": stats_npz["std"]}
        groups_interp = list(df_interp.groupby(["subject", "sequence_id"], sort=False))
        demo_cols = config.get("demographics_cols", [])
        tab_cfg = config.get("preprocessing", {})
        build_tabular_features(
            groups_interp,
            sensor_cols,
            demo_cols,
            stats,
            tab_cfg,
            out_dir / "predict_features.npy",
        )

        np.save(out_dir / "predict_labels.npy", y)
        with open(out_dir / "predict_info.json", "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
