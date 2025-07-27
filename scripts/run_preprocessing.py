#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Preprocess CMI dataset using the Preprocessor pipeline.

Usage:
    python scripts/run_preprocessing.py --experiment-name exp1 [--config CONFIG]
                                       [--use-cache] [--mode train|predict]

The script loads `train.csv` and `test.csv` from the directory
specified in the config file and saves processed data under
`output/experiments/<experiment_name>/preprocessed/`.
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Any
import pickle
import logging
import json
import numpy as np

import pandas as pd
import yaml

from src.utils.pipeline import Preprocessor
from src.utils.logging_utils import setup_logging


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_feature_names(config: dict) -> dict[str, list[str]]:
    """Generate feature names for each modality based on config."""
    feature_names = {}
    
    # Demographics feature names
    demographics_cols = config.get("demographics_cols", [])
    feature_names["demographics"] = demographics_cols
    
    # Sensor feature names (for windows tensor)
    sensor_cols = (
        config.get("sensor_acc_cols", [])
        + config.get("sensor_rot_cols", [])
        + config.get("sensor_thm_cols", [])
    )
    if config.get("preprocessing", {}).get("use_world_acc", False):
        world_acc_cols = [f"acc_w_{ax}" for ax in "xyz"] + [f"lin_acc_{ax}" for ax in "xyz"]
        sensor_cols.extend(world_acc_cols)
    feature_names["sensor"] = sensor_cols
    
    # Tabular feature names
    tabular_features = []
    
    # Basic statistics features (mean, std, range, rms, energy for each sensor)
    for col in sensor_cols:
        tabular_features.extend([
            f"{col}_mean", f"{col}_std", f"{col}_range", 
            f"{col}_rms", f"{col}_energy"
        ])
    
    # Magnitude features (if 3+ axes available)
    if len(sensor_cols) >= 3:
        tabular_features.extend(["mag_mean", "mag_std"])
    
    # Peak features (one per sensor)
    for col in sensor_cols:
        tabular_features.append(f"{col}_peaks")
    
    # FFT band energy features
    fft_bands = config.get("preprocessing", {}).get("fft_bands", [])
    for i, (low, high) in enumerate(fft_bands):
        for col in sensor_cols:
            tabular_features.append(f"{col}_fft_{low}_{high}Hz")
    
    # Optional features
    pp_config = config.get("preprocessing", {})
    if pp_config.get("use_wavelet_features", False):
        wavelet = pp_config.get("wavelet", "db4")
        level = pp_config.get("wavelet_level", 3)
        for col in sensor_cols:
            for l in range(1, level + 1):
                tabular_features.append(f"{col}_wavelet_{wavelet}_level_{l}")
    
    if pp_config.get("use_tda_features", False):
        tda_dim = pp_config.get("tda_dimension", 1)
        tda_bins = pp_config.get("tda_bins", 20)
        for col in sensor_cols:
            for i in range(tda_bins):
                tabular_features.append(f"{col}_tda_dim{tda_dim}_bin{i}")
    
    if pp_config.get("use_tof_event_features", False):
        tof_depth = pp_config.get("tof_depth", 5)
        tof_height = pp_config.get("tof_height", 8)
        tof_width = pp_config.get("tof_width", 8)
        for d in range(1, tof_depth + 1):
            for h in range(tof_height):
                for w in range(tof_width):
                    tabular_features.append(f"tof_{d}_v{h*tof_width+w}_events")
    
    feature_names["tabular"] = tabular_features
    
    # ToF feature names
    tof_depth = pp_config.get("tof_depth", 5)
    tof_height = pp_config.get("tof_height", 8)
    tof_width = pp_config.get("tof_width", 8)
    
    # ToF voxel features
    tof_voxel_features = []
    for d in range(1, tof_depth + 1):
        for h in range(tof_height):
            for w in range(tof_width):
                tof_voxel_features.append(f"tof_{d}_v{h*tof_width+w}")
    feature_names["tof_voxel"] = tof_voxel_features
    
    # ToF windows features (same as voxel but for time series)
    feature_names["tof_windows"] = tof_voxel_features
    
    return feature_names


def create_metadata(data: dict, config: dict, prefix: str) -> dict[str, Any]:
    """Create metadata for preprocessed data."""
    metadata = {
        "prefix": prefix,
        "shapes": {},
        "feature_names": generate_feature_names(config),
        "config": config
    }
    
    # Extract shapes
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            metadata["shapes"][key] = list(value.shape)
        else:
            metadata["shapes"][key] = str(type(value))
    
    return metadata


def save_dict(data: dict, prefix: str, out_dir: Path, config: dict) -> None:
    """Save components of preprocessed data with a prefix and metadata."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save data files
    for key, value in data.items():
        file = out_dir / f"{prefix}_{key}.pkl"
        with open(file, "wb") as f:
            pickle.dump(value, f)
        logging.getLogger(__name__).info("Saved %s", file)
    
    # Save metadata
    metadata = create_metadata(data, config, prefix)
    metadata_file = out_dir / f"{prefix}_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logging.getLogger(__name__).info("Saved metadata to %s", metadata_file)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run preprocessing pipeline")
    parser.add_argument(
        "--config",
        default="config/config_v2.yaml",
        help="YAML config path",
    )
    parser.add_argument("--experiment-name", required=True, help="experiment name")
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="use cached intermediate files if available",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "predict"],
        default="train",
        help="train: fit and transform train/test, predict: transform only",
    )
    parser.add_argument(
        "--log-file",
        help=(
            "log file path (default: <output_dir>/<experiment>/preprocessed/preprocess.log)"
        ),
    )
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    data_dir = Path(config.get("data_dir", "data"))
    out_root = Path(config.get("output_dir", "output/experiments"))
    pre_dir = out_root / args.experiment_name / "preprocessed"
    pre_dir.mkdir(parents=True, exist_ok=True)

    log_path = Path(args.log_file) if args.log_file else pre_dir / "preprocess.log"
    setup_logging(log_path)
    logger = logging.getLogger(__name__)

    if args.mode == "train":
        logger.info("Loading train.csv and test.csv")
        train_df = pd.read_csv(data_dir / "train.csv")
        test_df = pd.read_csv(data_dir / "test.csv")
        logger.info("Fitting Preprocessor")
        
        # Load demographics data and merge
        train_demo = pd.read_csv(data_dir / "train_demographics.csv")
        test_demo = pd.read_csv(data_dir / "test_demographics.csv")
        
        # Merge demographics data
        train_df = train_df.merge(train_demo, on="subject", how="left")
        test_df = test_df.merge(test_demo, on="subject", how="left")
        
        pp = Preprocessor(config)
        pp.fit(train_df, use_cache=args.use_cache)
        train_data = pp.transform(train_df, use_cache=args.use_cache)
        # test_data = pp.transform(test_df, use_cache=args.use_cache)

        # save results with metadata
        save_dict(train_data, "train", pre_dir, config)
        # save_dict(test_data, "test", pre_dir, config)
        logger.info("Saved train/test outputs to %s", pre_dir)
        pp.save(pre_dir / "preprocessor.pkl")
        logger.info("Saved preprocessor to %s", pre_dir / "preprocessor.pkl")

    else:  # predict
        # Load main data
        logger.info("Loading test.csv for prediction")
        df = pd.read_csv(data_dir / "test.csv")
        logger.info("Loading preprocessor from %s", pre_dir / "preprocessor.pkl")
        
        # Load demographics data and merge
        test_demo = pd.read_csv(data_dir / "test_demographics.csv")
        df = df.merge(test_demo, on="subject", how="left")
        
        pp = Preprocessor.load(pre_dir / "preprocessor.pkl")
        data = pp.transform(df, use_cache=args.use_cache)
        save_dict(data, "predict", pre_dir, config)
        logger.info("Saved prediction outputs to %s", pre_dir)


if __name__ == "__main__":
    main()
