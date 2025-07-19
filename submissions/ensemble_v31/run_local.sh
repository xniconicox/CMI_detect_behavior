#!/bin/bash
set -e

# This script is for local execution of the submission code.
# It simulates the environment of Kaggle notebooks.

# 1. Symlink to the evaluation data
if [ ! -d "kaggle_evaluation" ]; then
    echo "Creating symlink to kaggle_evaluation data..."
    ./prepare.sh
fi

# 2. Copy trained models
echo "Copying trained models..."
mkdir -p models
cp ../../output/experiments/20250717_preproc_train_v31/models/multimodal_model_v31_fold*.keras models/
cp ../../output/experiments/20250717_preproc_train_v31/preprocessed/preprocessor.pkl models/

# 3. Create a simple test to verify the setup
echo "Testing model loading..."
python -c "
import sys
sys.path.insert(0, 'src')
from src.inference_pipeline import models, preprocessor
print(f'Loaded {len(models)} models successfully')
print(f'Preprocessor classes: {len(preprocessor.label_encoder.classes_)}')
"

# 4. Copy notebook for local execution
echo "Copying notebook for local execution..."
cp ../../cmi-2025-demo-submission.ipynb ./original_notebook.ipynb 2>/dev/null || echo "Original notebook not found, using custom notebook"

# 5. Execute notebook using papermill (if available)
if command -v papermill &> /dev/null; then
    echo "Executing notebook with papermill..."
    papermill cmi-2025-demo-submission.ipynb output/submission.ipynb
else
    echo "Papermill not available. Please run the notebook manually."
    echo "Notebook: cmi-2025-demo-submission.ipynb"
fi

# 6. Show submission file (if exists)
if [ -f "submission.parquet" ]; then
    echo "Submission file created:"
    ls -l submission.parquet
    echo "First few rows:"
    python -c "import pandas as pd; df = pd.read_parquet('submission.parquet'); print(df.head()); print('Shape:', df.shape)"
else
    echo "No submission file found. Please run the notebook manually."
fi 