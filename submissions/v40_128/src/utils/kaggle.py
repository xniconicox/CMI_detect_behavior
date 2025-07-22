import os
from pathlib import Path

def is_kaggle():
    # Kaggle環境ではこの環境変数が必ず存在
    return "KAGGLE_URL_BASE" in os.environ or Path("/kaggle").exists()

if is_kaggle():
    # Kaggle環境
    PROJECT_ROOT = Path("/kaggle/input/cmi-ensemble-v20/")
    DATA_DIR = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data/")
    EVAL_MODULE_DIR = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data/")
else:
    # ローカル環境
    PROJECT_ROOT = Path("/mnt/c/Users/ShunK/works/CMI_comp/submissions/ensemble_v1")
    DATA_DIR = Path("../../data/")
    EVAL_MODULE_DIR = Path("/mnt/c/Users/ShunK/works/CMI_comp/submissions/ensemble_v1/kaggle_evaluation")

# # 例：使い方
# print("PROJECT_ROOT:", PROJECT_ROOT)
# print("DATA_DIR:", DATA_DIR)
# print("EVAL_MODULE_DIR:", EVAL_MODULE_DIR)