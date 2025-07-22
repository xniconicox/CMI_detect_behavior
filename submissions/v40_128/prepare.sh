#!/bin/bash

# Create necessary directories
mkdir -p src/utils
mkdir -p src/trainers

# Copy essential source files
cp -p ../../src/utils/pipeline.py ./src/utils/
cp -p ../../src/utils/kaggle.py ./src/utils/
cp -p ../../src/utils/cmi_evaluation.py ./src/utils/
cp -p ../../src/utils/config_utils.py ./src/utils/
cp -p ../../src/trainers/multimodal_trainer_v40.py ./src/trainers/
cp -p ../../src/utils/preprocessing.py ./src/utils/
cp -p ../../src/utils/tof.py ./src/utils/
cp -p ../../src/utils/imu.py ./src/utils/
cp -p ../../src/utils/feature_engineering.py ./src/utils/
cp -p ../../src/utils/io_utils.py ./src/utils/

# Link ws128 model and preprocessor
ln -s ../../output/experiments/preprocess_v2_ws128/models models_128
ln -s ../../output/experiments/preprocess_v2_ws128/preprocessed/preprocessor.pkl preprocessor_128.pkl

# Link config file
ln -s ../../config/config_v40_ws128.yaml config_ws128.yaml

# Create a dummy kaggle_evaluation link if it doesn't exist, for local testing
if [ ! -d "kaggle_evaluation" ]; then
    ln -s ../../data/kaggle_evaluation kaggle_evaluation
fi
