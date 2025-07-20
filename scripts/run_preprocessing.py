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

import pandas as pd
import yaml

from src.utils.pipeline import Preprocessor
from src.utils.logging_utils import setup_logging
from src.utils.preprocessing import augment_by_handedness_flip


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
        logging.getLogger(__name__).info("Saved %s", file)


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
        "--augment-handedness",
        action="store_true",
        help="flip handedness and append augmented samples",
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

        # Fit on original data first
        pp = Preprocessor(config)
        logger.info("Fitting Preprocessor on original data")
        pp.fit(train_df, use_cache=args.use_cache)

        logger.info("Transforming original train data")
        train_data = pp.transform(train_df, use_cache=args.use_cache)

        if args.augment_handedness:
            logger.info("Augmenting train data by handedness flip")
            augmented_df = augment_by_handedness_flip(train_df)
            logger.info("Augmented train samples: %d", len(augmented_df))

            logger.info("Transforming augmented data in chunks")
            # Process augmented data in chunks to reduce memory usage
            chunk_size = 100000  # Adjust based on available memory
            num_chunks = (len(augmented_df) + chunk_size - 1) // chunk_size
            
            augmented_data_chunks = []
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, len(augmented_df))
                chunk_df = augmented_df.iloc[start_idx:end_idx]
                
                logger.info(f"Processing chunk {i+1}/{num_chunks} (samples {start_idx}-{end_idx})")
                chunk_data = pp.transform(chunk_df, use_cache=args.use_cache)
                augmented_data_chunks.append(chunk_data)
            
            # Merge all chunks
            logger.info("Merging augmented data chunks")
            augmented_data = {}
            for key in augmented_data_chunks[0]:
                if isinstance(augmented_data_chunks[0][key], pd.DataFrame):
                    augmented_data[key] = pd.concat(
                        [chunk[key] for chunk in augmented_data_chunks], ignore_index=True
                    )
                elif isinstance(augmented_data_chunks[0][key], (list, tuple)):
                    augmented_data[key] = []
                    for chunk in augmented_data_chunks:
                        augmented_data[key].extend(chunk[key])
                else:  # numpy array
                    augmented_data[key] = pd.concat(
                        [pd.DataFrame(chunk[key]) for chunk in augmented_data_chunks],
                        ignore_index=True,
                    ).values

            logger.info("Merging original and augmented data")
            for key in train_data:
                if key in augmented_data:
                    original_component = train_data[key]
                    augmented_component = augmented_data[key]
                    if isinstance(original_component, pd.DataFrame):
                        train_data[key] = pd.concat(
                            [original_component, augmented_component], ignore_index=True
                        )
                    elif isinstance(original_component, (list, tuple)):
                        train_data[key] = original_component + augmented_component
                    else:  # numpy array
                        train_data[key] = pd.concat(
                            [
                                pd.DataFrame(original_component),
                                pd.DataFrame(augmented_component),
                            ],
                            ignore_index=True,
                        ).values

        # save results
        save_dict(train_data, "train", pre_dir)
        # save_dict(test_data, "test", pre_dir)
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
        save_dict(data, "predict", pre_dir)
        logger.info("Saved prediction outputs to %s", pre_dir)


if __name__ == "__main__":
    main()
