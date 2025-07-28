#!/bin/bash

# Create directories
mkdir -p src/utils
mkdir -p src/trainers

# Copy required source files
cp -p ../../src/utils/pipeline.py ./src/utils/
cp -p ../../src/utils/kaggle.py ./src/utils/
cp -p ../../src/utils/cmi_evaluation.py ./src/utils/
cp -p ../../src/utils/config_utils.py ./src/utils/
cp -p ../../src/utils/preprocessing.py ./src/utils/
cp -p ../../src/utils/tof.py ./src/utils/
cp -p ../../src/utils/imu.py ./src/utils/
cp -p ../../src/utils/feature_engineering.py ./src/utils/
cp -p ../../src/utils/io_utils.py ./src/utils/
cp -p ../../src/trainers/multimodal_trainer_v40.py ./src/trainers/

# Link model directory and preprocessor
ln -s ../../output/experiments/preprocess_v50/models models
ln -s ../../output/experiments/preprocess_v50/preprocessed/preprocessor_v50.pkl preprocessor_v50.pkl

# Link config file if available
ln -s ../../config/config_v50.yaml config_v50.yaml

# Link kaggle evaluation data for local testing
if [ ! -d "kaggle_evaluation" ]; then
    ln -s ../../data/kaggle_evaluation kaggle_evaluation
fi
