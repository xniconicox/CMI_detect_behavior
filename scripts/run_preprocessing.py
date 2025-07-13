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

import argparse
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.utils.pipeline import Preprocessor


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_dict(data: dict, prefix: str, out_dir: Path) -> None:
    """Save components of preprocessed data with a prefix."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, value in data.items():
        file = out_dir / f"{prefix}_{key}.pkl"
        with open(file, "wb") as f:
            pickle.dump(value, f)


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
    args = parser.parse_args()

    config = load_yaml(Path(args.config))
    data_dir = Path(config.get("data_dir", "data"))
    out_root = Path(config.get("output_dir", "output/experiments"))
    pre_dir = out_root / args.experiment_name / "preprocessed"
    pre_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "train":
        train_df = pd.read_csv(data_dir / "train.csv")
        test_df = pd.read_csv(data_dir / "test.csv")
        pp = Preprocessor(config)
        pp.fit(train_df, use_cache=args.use_cache)
        train_data = pp.transform(train_df, use_cache=args.use_cache)
        test_data = pp.transform(test_df, use_cache=args.use_cache)

        # save results
        save_dict(train_data, "train", pre_dir)
        save_dict(test_data, "test", pre_dir)
        with open(pre_dir / "preprocessor.pkl", "wb") as f:
            pickle.dump(pp, f)

    else:  # predict
        df = pd.read_csv(data_dir / "test.csv")
        with open(pre_dir / "preprocessor.pkl", "rb") as f:
            pp = pickle.load(f)
        data = pp.transform(df, use_cache=args.use_cache)
        save_dict(data, "predict", pre_dir)


if __name__ == "__main__":
    main()
