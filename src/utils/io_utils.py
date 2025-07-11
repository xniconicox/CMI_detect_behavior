# src/utils/io_utils.py
from pathlib import Path
import pandas as pd, json, hashlib

CACHE_DIR   = Path("cache")
TRAIN_PARQ  = CACHE_DIR / "train_clean.parquet"
TEST_PARQ   = CACHE_DIR / "test_clean.parquet"
META_JSON   = CACHE_DIR / "clean_meta.json"

def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()

def load_clean_data(raw_train_csv="data/train.csv",
                    raw_test_csv="data/test.csv",
                    strict=True):
    """
    前処理済み Parquet を読み込む。
    * strict=True なら MD5 がズレていた場合は例外を投げる
    """
    if not TRAIN_PARQ.exists() or not TEST_PARQ.exists():
        raise FileNotFoundError("cleaned parquet not found – 先に前処理セルを実行してください")

    # MD5 チェック
    if META_JSON.exists():
        meta = json.loads(META_JSON.read_text())
        if strict:
            if meta["raw_train_md5"] != _md5(Path(raw_train_csv)):
                raise RuntimeError("raw train CSV has changed – recalc needed")
            if meta["raw_test_md5"]  != _md5(Path(raw_test_csv)):
                raise RuntimeError("raw test CSV has changed – recalc needed")
    else:
        print("⚠️ meta file not found – skipping MD5 check")

    print("📥 loading cached parquet …", end="", flush=True)
    train_df = pd.read_parquet(TRAIN_PARQ)
    test_df  = pd.read_parquet(TEST_PARQ)
    print("done")
    return train_df, test_df
