"""
Preprocessing utilities for the CMI competition (commented version).

"""

import numpy as np
import pandas as pd


# ============================================================
# L. 利き手反転正規化 (Y/Z flip for left‑handed)
# ============================================================
from typing import Sequence

def handedness_correction_v2(
    df: pd.DataFrame,
    *,
    imu_prefixes: Sequence[str] = ("acc", "gyro", "rot", "mag"),
    apply_tof_mirror: bool = True,
) -> pd.DataFrame:
    """
    左利き (handedness==0) サンプルを右利き座標系に正規化する。

    1. IMU (acc/gyro/rot/mag) の Y・Z 軸を符号反転
       - rot_w は反転しない
    2. ToF センサは左右ピクセルを鏡像ミラー (値はそのまま)
       - 値の符号反転は行わない
    """
    df = df.copy()
    left = df["handedness"] == 0

    # --- 1) IMU の Y/Z 反転 ------------------------------------------
    axis_sign = {"x": 1, "y": -1, "z": -1, "w": 1}       # rot_w は 1
    for pre in imu_prefixes:
        axes = ("w", "x", "y", "z") if pre == "rot" else ("x", "y", "z")
        for ax in axes:
            col = f"{pre}_{ax}"
            if col in df.columns:
                df.loc[left, col] *= axis_sign[ax]

    # --- 2) ToF 水平ミラー -------------------------------------------
    if apply_tof_mirror:
        tof_cols = [c for c in df.columns if c.startswith("tof_")]
        if tof_cols:  # mirror_tof_rows は既存関数を流用
            df.loc[left, tof_cols] = mirror_tof_rows(df.loc[left, :], tof_cols)

    return df


import re
# ── 1. ToF の水平ミラー（左右反転）関数 ─────────────────────────
def mirror_tof_rows(df: pd.DataFrame, tof_cols: list[str]) -> pd.DataFrame:
    """
    各行の ToF ピクセル(8×8) を水平ミラーします。
    df[tof_cols] の形状は (N, 64×D) を想定。
    """
    pattern = re.compile(r"tof_(\d+)_v(\d+)")
    # センサIDごとのグループを取得
    sensor_ids = sorted({int(pattern.match(c).group(1)) for c in tof_cols})
    out = df.copy()
    for sid in sensor_ids:
        # 当該センサの 64 列を時系列順に並べて (N, H, W) にreshape
        cols = [f"tof_{sid}_v{i}" for i in range(64)]
        arr = df[cols].to_numpy().reshape(-1, 8, 8)
        # W方向を反転
        flipped = arr[:, :, ::-1]
        # 元に戻して DataFrame に代入
        out.loc[:, cols] = flipped.reshape(-1, 64)
    return out


# ============================================================
# Utility : Missing-value Cleaning
# =============================

# =========================================================
# D) センサータイプごとに欠損値を適切に処理する関数
# =========================================================

def clean_sensor_missing_values(
    df: pd.DataFrame,
    sensor_type_groups: dict,
    acc_clip: tuple = (-40.0, 40.0),
) -> pd.DataFrame:
    """
    センサータイプごとに欠損値を適切に処理する関数（調査結果に基づく）
    
    ── Rules (調査結果ベース) ────────────────────────────────
      • Accelerometer :   0 / −1 は正当値 → 極端値 |value|>acc_clip を NaN
      • Rotation      :   NaN のみ欠損値
      • ToF_Sensor    :   NaN, -1 および -1 未満 → 欠損値 (NaN 化)
      • Thermal       :   NaN のみ欠損値
    ─────────────────────────────────────────────────────────
    """
    df = df.copy()

    # ------------------ 1) Accelerometer ------------------
    acc_cols = sensor_type_groups.get("Accelerometer", [])
    if acc_cols:
        # |value| > clip 上限を欠測とみなす
        df[acc_cols] = df[acc_cols].where(
            df[acc_cols].abs().le(acc_clip[1]), np.nan
        )

    # ------------------ 2) ToF_Sensor ---------------------
    tof_cols = sensor_type_groups.get("ToF_Sensor", [])
    if tof_cols:
        # a) 上限超え (255 以上) は NaN
        df[tof_cols] = df[tof_cols].where(df[tof_cols] <= 254, np.nan)
        # b) -2 以下も NaN （ハードエラー）
        df[tof_cols] = df[tof_cols].where(df[tof_cols] >= -1, np.nan)
        # ※ -1 は “反射なし” → そのまま残す

    # ------------------ 3) Rotation / Thermal -------------
    # 既定では NaN のみ欠測 → 追加処理不要
    return df


# =========================================================
# A) 補間パラメータ（調査結果を反映）
# =========================================================
# 外部ファイルで定義

# =========================================================
# B) クォータニオン SLERP と ToF 用補間関数
# =========================================================
from scipy.spatial.transform import Rotation, Slerp
import numpy as np
import pandas as pd

def _interpolate_quaternion_block(df, cols, time_col, limit):
    q = df[cols].to_numpy(float)

    # 0-norm → 欠測扱い
    norm = np.linalg.norm(q, axis=1, keepdims=True)
    q[norm.squeeze() < 1e-6] = np.nan

    t = df[time_col].to_numpy()
    mask_nan = np.isnan(q).any(axis=1)

    i = 0
    while i < len(q):
        if mask_nan[i]:
            # --- 欠測 run の [start+1 : end-1] を補間 -----------------
            start = i - 1               # start は常に定義される
            j = i
            while j < len(q) and mask_nan[j]:
                j += 1
            end = j                      # first non-NaN idx or len(q)
            run_len = end - start - 1
            if start >= 0 and end < len(q) and run_len <= limit:
                key_times = np.array([t[start], t[end]])
                key_rots  = Rotation.from_quat(q[[start, end]])
                slerp = Slerp(key_times, key_rots)
                q[start+1:end] = slerp(t[start+1:end]).as_quat()
            i = end                      # ← 欠測 run の次フレームへ
        else:
            i += 1                       # 欠測でない→次へ

    # 補間後に正規化
    valid = ~np.isnan(q).any(axis=1)
    q[valid] /= np.linalg.norm(q[valid], axis=1, keepdims=True)
    # まだ NaN があれば identity に
    q[~valid] = np.array([1,0,0,0])

    return pd.DataFrame(q, columns=cols, index=df.index)



def _interpolate_tof_block(df, cols, limit, fill_value=-1):
    """-1 は反射なし → 一時 NaN → 短ラン補間 → -1 に戻す"""
    blk = df[cols].replace(fill_value, np.nan)
    blk = blk.interpolate(method="linear",
                          limit=limit,
                          limit_direction="both")
    return blk.fillna(fill_value)

# =========================================================
# C) 並列 & ディスク結合版クリーニング
# =========================================================
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import tempfile, shutil
from pathlib import Path

def clean_missing_sensor_data_parallel_disk(
    df: pd.DataFrame,
    sensor_type_groups: dict[str, list[str]],
    group_cols=("subject", "sequence_id"),
    time_col: str = "sequence_counter",
    interp_params: dict | None = None,
    n_jobs: int = -1,
    tmp_parent: str | Path = "/tmp/tmp_clean_chunks",
    keep_tmp: bool = False,
    keep_cols: list[str] | None = None,        # ★NEW
) -> pd.DataFrame:

    keep_cols = list(keep_cols or [])
    base_cols = list(group_cols) + [time_col] + keep_cols
    
    # 0) デフォルト
    default_params = {
        "Accelerometer": {"method": "linear", "limit": 3},
        "Rotation":      {"method": "slerp",  "limit": 120},
        "ToF_Sensor":    {"method": "linear", "limit": 4, "fill_value": -1},
        "Thermal":       {"method": "linear", "limit": 120},
    }
    interp_params = interp_params or default_params

    # 1) ソート
    df = df.sort_values(list(group_cols) + [time_col]).copy()

    # 2) 一時ディレクトリ
    tmp_root = Path(tmp_parent); tmp_root.mkdir(exist_ok=True)
    tmp_dir = tempfile.mkdtemp(dir=tmp_root)

    # 3) 各グループを並列補間 → parquet 保存
    def _interpolate_and_dump(key, g,
                            sensor_type_groups, interp_params,
                            time_col, tmp_dir):

        # --- 欠測補間 -------------------------------------------------
        for s_type, cols in sensor_type_groups.items():
            params = interp_params[s_type]
            method = params.get("method")
            limit  = params.get("limit", 3)

            if method is None:
                continue
            if s_type == "Rotation":
                g[cols] = _interpolate_quaternion_block(
                    g, cols, time_col, limit)
            elif s_type == "ToF_Sensor":
                g[cols] = _interpolate_tof_block(
                    g, cols, limit, params.get("fill_value", -1))
            else:  # Accelerometer / Thermal
                g[cols] = g[cols].interpolate(
                    method=method,
                    limit=limit,
                    limit_direction="both"
                )
        # -------------------------------------------------------------

        # Parquet 出力
        subject, seq_id = key
        # ❶ 保存したい列を計算
        save_cols = list(dict.fromkeys(
            base_cols + sum(sensor_type_groups.values(), [])))

        # ❷ サブセットを取ってから Parquet 書き出し
        out_path = Path(tmp_dir) / f"{subject}_{seq_id}.parquet"
        g.loc[:, save_cols].to_parquet(out_path)   # ← columns 引数を削除
        return out_path

    # ① (key, grp) を保持
    groups = [(key, grp) for key, grp in df.groupby(list(group_cols), sort=False)]

    # ② key を _interpolate_and_dump に渡す
    paths = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_interpolate_and_dump)(
            key, grp,              # ← 変更
            sensor_type_groups,
            interp_params,
            time_col,
            tmp_dir
        )
        for key, grp in tqdm(groups, desc="interp", unit="seq")
    )
    # 4) 結合
    cleaned = pd.concat(
        (pd.read_parquet(p) for p in tqdm(paths, desc="concat")),
        ignore_index=True          # ← sort_index 不要に
    )

    if not keep_tmp:
        # 削除はバックグラウンドでも可
        import threading, shutil
        threading.Thread(target=shutil.rmtree,
                        args=(tmp_dir,),
                        kwargs=dict(ignore_errors=True),
                        daemon=True).start()

    return cleaned 

# ============================================================
# I. 合成 Tabular (= A–H)
# ============================================================
### --- Block I Summary ---------------------------------------
# dims: 120 | 推奨モデル: LightGBM_all / CatBoost
# 木モデル用に Tabular 特徴を統合
# ------------------------------------------------------------
# （統合処理は学習パイプライン側で実施）


# ============================================================
# J. IMU Sliding Window Tensor
# ============================================================
### --- Block J Summary ---------------------------------------
# dims: 7×256 = 1 792 | 推奨モデル: Binary-GRU / CNN-GRU
# 局所パターン＋長期文脈を時系列テンソルで捕捉
# ------------------------------------------------------------
def create_sliding_windows_with_demographics(
    df: pd.DataFrame,
    window_size: int,
    stride: int,
    sensor_cols: list,
    demographics_cols: list,
    min_sequence_length: int = 10,
    padding_value: float = 0.0,
):
    """Block B: generate fixed‑length windows and attach static demographics."""
    X_sensor_windows, X_demographics_windows, y_windows, info = [], [], [], []

    for (subject, seq_id), g in df.groupby(["subject", "sequence_id"]):
        seq_len = len(g)
        if seq_len < min_sequence_length:
            continue
        sensor = g[sensor_cols].values
        demo = g[demographics_cols].iloc[0].values
        gesture = g["gesture"].iloc[0]
        need_pad = seq_len < window_size
        if need_pad:
            pad = np.full((window_size - seq_len, len(sensor_cols)), padding_value)
            sensor = np.vstack([sensor, pad])
        for s in range(0, len(sensor) - window_size + 1, stride):
            e = s + window_size
            X_sensor_windows.append(sensor[s:e])
            X_demographics_windows.append(demo)
            y_windows.append(gesture)
            info.append({"subject": subject, "sequence_id": seq_id, "start_idx": s, "end_idx": e, "padded": need_pad})

    return (
        np.asarray(X_sensor_windows, dtype=np.float32),
        np.asarray(X_demographics_windows, dtype=np.float32),
        np.asarray(y_windows),
        info,
    )





# ============================================================
# K. ToF 3D Voxel Tensor
# ============================================================
### --- Block K Summary ---------------------------------------
# dims: 20 480 | 推奨モデル: ToF-3D-CNN
# 顔・対象物との空間的接近パターンを 3D で表現
# ------------------------------------------------------------

# ============================================================
# C. 正規化ユーティリティ (sensor / tabular)
# ============================================================

def normalize_sensor_data(X: np.ndarray):
    """Block C‑1: z‑score normalisation for sensor windows."""
    scaler = StandardScaler()
    n, t, f = X.shape
    X_flat = np.nan_to_num(X.reshape(-1, f), nan=0.0)
    X_norm = scaler.fit_transform(X_flat).reshape(n, t, f)
    return X_norm, scaler


def normalize_tabular_data(X: np.ndarray):
    """Block C‑2: z‑score normalisation for demographics/tabular."""
    scaler = StandardScaler()
    return scaler.fit_transform(X), scaler