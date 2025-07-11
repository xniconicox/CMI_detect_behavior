"""
Preprocessing utilities for the CMI competition (commented version).

Added **A–L** section headers that map directly to the design document
(section 4‑2) so you can quickly trace which function produces which
feature block.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.signal import find_peaks

# ============================================================
# A. 基本統計量 (mean / std / range / RMS / energy)
# ============================================================
### --- Block A Summary ---------------------------------------
# dims: 56 | 推奨モデル: LightGBM_all / non
# 広域量的特徴――窓全体の強度・分布を圧縮
# ------------------------------------------------------------
def compute_basic_statistics(X_windows: np.ndarray) -> np.ndarray:
    """Compute Block F statistics per window."""
    means  = X_windows.mean(axis=1)
    stds   = X_windows.std(axis=1)
    ranges = X_windows.max(axis=1) - X_windows.min(axis=1)
    rms    = np.sqrt((X_windows ** 2).mean(axis=1))
    energy = (X_windows ** 2).sum(axis=1)
    if X_windows.shape[2] >= 3:
        mag = np.linalg.norm(X_windows[:, :, :3], axis=2)
        mag_mean = mag.mean(axis=1, keepdims=True)
        mag_std  = mag.std(axis=1, keepdims=True)
        return np.hstack([means, stds, ranges, rms, energy, mag_mean, mag_std])
    return np.hstack([means, stds, ranges, rms, energy])


# ============================================================
# B. ピーク & 周期特徴量
# ============================================================
### --- Block B Summary ---------------------------------------
# dims: 18 | 推奨モデル: LightGBM_all / non
# BFRB の “反復性” をカウントして数値化
# ------------------------------------------------------------

def extract_peak_features(window: np.ndarray) -> np.ndarray:
    """Return per‑axis peak counts (Block D)."""
    return np.array([len(find_peaks(window[:, i])[0]) for i in range(window.shape[1])], dtype=np.float32)


def compute_peak_features(X_windows: np.ndarray) -> np.ndarray:
    """Block D wrapper for many windows."""
    return np.vstack([extract_peak_features(w) for w in X_windows])

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

# ============================================================
# C. FFT バンドエネルギー (0.5–20 Hz)
# ============================================================
### --- Block C Summary ---------------------------------------
# dims: 10 | 推奨モデル: LightGBM_all / non
# 動作リズム・速度の周波数分布
# ------------------------------------------------------------

def compute_fft_band_energy(X_windows: np.ndarray, fs: float = 50.0, bands=None) -> np.ndarray:
    """Compute block G FFT band energies."""
    if bands is None:
        bands = [(0.5, 2), (2, 5), (5, 10), (10, 20)]
    n_win, win_len, n_feat = X_windows.shape
    freqs = np.fft.rfftfreq(win_len, d=1.0 / fs)
    power = np.abs(np.fft.rfft(X_windows, axis=1)) ** 2
    energies = []
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        energies.append(power[:, mask, :].sum(axis=1))
    return np.concatenate(energies, axis=1)

# ============================================================
# D. ワールド線形加速度 & IMU World-Frame 変換
# ============================================================
### --- Block D Summary ---------------------------------------
# dims: 21 | 推奨モデル: LightGBM_all / CNN-GRU
# 姿勢成分を除去した「純粋な動き」ベクトル
# ------------------------------------------------------------
def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion **q = [w, x, y, z]** to its 3×3 rotation matrix.

    Part of **Block A** in the feature table (IMU world transformation).
    """
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)],
    ])


# def rotate_acceleration(acc: np.ndarray, quat: np.ndarray) -> np.ndarray:
#     """Rotate raw accelerometer vector into the world frame (Block A)."""
#     return quaternion_to_rotation_matrix(quat) @ acc


# def linear_acceleration(acc: np.ndarray, quat: np.ndarray, gravity: float = 9.81) -> np.ndarray:
#     """Remove gravity after rotation, yielding linear acceleration (Block A)."""
#     acc_world = rotate_acceleration(acc, quat)
#     gravity_vec = np.array([0, 0, gravity])
#     return acc_world - gravity_vec
# セルD-1：IMU をワールド座標系に変換＋線形加速度算出を並列＆ディスクキャッシュで実装
import tempfile, shutil, uuid, gc
from pathlib import Path
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import pandas as pd, numpy as np

def _world_transform_and_dump(
    seq_df: pd.DataFrame,
    time_col: str,
    tmp_dir: str,
) -> str:
    """シーケンス毎に world-frame 変換＋線形加速度を計算して Parquet 出力"""
    seg = seq_df.sort_values(time_col).copy()
    accs = seg[['acc_x','acc_y','acc_z']].to_numpy()
    quats = seg[['rot_w','rot_x','rot_y','rot_z']].to_numpy()

    # 結果用配列
    acc_w    = np.empty_like(accs)
    lin_acc  = np.empty_like(accs)
    g_vec    = np.array([0,0,9.81])

    # 各行を変換
    for i, (acc, q) in enumerate(zip(accs, quats)):
        R = quaternion_to_rotation_matrix(q)
        aw = R @ acc
        acc_w[i]   = aw
        lin_acc[i] = aw - g_vec

    # DataFrame に列追加
    seg[['acc_w_x','acc_w_y','acc_w_z']] = acc_w
    seg[['lin_acc_x','lin_acc_y','lin_acc_z']] = lin_acc

    # Parquet 出力
    fname = f"seq_{seq_df['sequence_id'].iloc[0]}_{uuid.uuid4().hex}.parquet"
    out_path = Path(tmp_dir) / fname
    seg.to_parquet(out_path, compression='zstd')

    # メモリ解放
    del seg, accs, quats, acc_w, lin_acc
    gc.collect()
    return str(out_path)

# セルD-1：IMU→World変換＆線形加速度を subject 単位で処理＋キャッシュ
import tempfile, shutil, os, gc
from pathlib import Path
import pandas as pd, numpy as np
from tqdm.auto import tqdm
def transform_world_frame_by_subject(
    df: pd.DataFrame,
    cache_path: Path,
    group_col: str = "subject",
    time_col: str = "sequence_counter",
    tmp_parent: str = "tmp_world_subj",
    keep_tmp: bool = False,
) -> pd.DataFrame:
    """
    subject ごとに IMU をワールド座標系に変換 & 重力除去 → Parquet キャッシュ保存
    """
    # ソート＆コピー
    df = df.sort_values([group_col, "sequence_id", time_col]).copy()

    # 一時ディレクトリ
    tmp_root = Path(tmp_parent); tmp_root.mkdir(exist_ok=True)
    parts = []

    # subject ごとにループ
    for subj, sub_df in tqdm(df.groupby(group_col, sort=True),
                             desc="subject", unit="sub"):
        acc_arr  = sub_df[['acc_x','acc_y','acc_z']].to_numpy()
        quat_arr = sub_df[['rot_w','rot_x','rot_y','rot_z']].to_numpy()

        # 結果配列
        acc_w = np.empty_like(acc_arr)
        lin   = np.empty_like(acc_arr)
        g_vec = np.array([0,0,9.81])

        # 各フレームを変換
        for i in range(len(sub_df)):
            R = quaternion_to_rotation_matrix(quat_arr[i])
            aw = R.dot(acc_arr[i])
            acc_w[i] = aw
            lin[i]   = aw - g_vec

        # 列を追加
        sub_df[['acc_w_x','acc_w_y','acc_w_z']]   = acc_w
        sub_df[['lin_acc_x','lin_acc_y','lin_acc_z']] = lin

        # Parquet 出力
        out_path = tmp_root / f"subject_{subj}.parquet"
        sub_df.to_parquet(out_path, compression="zstd")
        parts.append(out_path)

        # メモリ解放
        del sub_df, acc_arr, quat_arr, acc_w, lin
        gc.collect()

    # すべて読み込み＆結合
    world_df = pd.concat(
        (pd.read_parquet(p) for p in tqdm(parts, desc="concat", unit="file")),
        ignore_index=True
    )

    # 一時ディレクトリ削除
    if not keep_tmp:
        shutil.rmtree(tmp_root, ignore_errors=True)

    # キャッシュ保存
    world_df.to_parquet(cache_path, compression="zstd")
    return world_df



# ============================================================
# E. 欠損フラグ (missing sensor flags)
# ============================================================
### --- Block E Summary ---------------------------------------
# dims: 3 | 推奨モデル: すべて
# 未接続センサを one-hot で明示
# ------------------------------------------------------------
def add_missing_sensor_flags(df: pd.DataFrame, sensor_groups: dict) -> pd.DataFrame:
    """Add boolean missing‑sensor flags per group (Block E)."""
    for flag, cols in sensor_groups.items():
        df[flag] = df[cols].isna().all(axis=1)
    return df

# ============================================================
# G. TDA Stats (Persistence Image)
# ============================================================
### --- Block G Summary ---------------------------------------
# dims: 8 | 推奨モデル: CNN-GRU concat / LightGBM
# 位相的周期性を独自にエンコード
# ------------------------------------------------------------
def compute_persistence_image_features(X_windows: np.ndarray, dimension:int=1, n_bins:int=20, sigma:float=0.1) -> np.ndarray:
    """Block J: persistence image features via giotto‑tda."""
    from gtda.time_series import TakensEmbedding
    from gtda.homology import VietorisRipsPersistence
    from gtda.diagrams import PersistenceImage

    emb = TakensEmbedding(time_delay=1, dimension=dimension)
    vrp = VietorisRipsPersistence(homology_dimensions=[0, 1])
    pim = PersistenceImage(bandwidth=sigma, n_bins=(n_bins, n_bins))
    feats = []
    for w in X_windows:
        e = emb.fit_transform(w)
        d = vrp.fit_transform(e[np.newaxis, ...])
        img = pim.fit_transform(d)
        feats.append(img.reshape(-1))
    return np.array(feats, dtype=np.float32)


# ============================================================
# H. Auto-Encoder 再構成誤差
# ============================================================
### --- Block H Summary ---------------------------------------
# dims: 4 | 推奨モデル: LightGBM / Binary-GRU (aux)
# 異常度スコアで BFRB 境界を補強
# ------------------------------------------------------------

def compute_autoencoder_reconstruction_error(X_windows: np.ndarray, model) -> np.ndarray:
    """Block K: per‑window MSE reconstruction error from a trained AE."""
    recon = model.predict(X_windows, verbose=0)
    return ((X_windows - recon) ** 2).mean(axis=(1, 2), keepdims=True)


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

def tof_to_voxel_tensor(df: pd.DataFrame, fill_value: float = 0.0, prefix: str = "tof_") -> np.ndarray:
    """Block L: convert ToF pixel columns → (T, depth, H, W) tensor."""
    import re
    pat = re.compile(fr"^{prefix}(\d+)_v(\d+)$")
    matches = [(c, pat.match(c)) for c in df.columns]
    sensors = sorted({int(m.group(1)) for _, m in matches if m})
    idxs = [int(m.group(2)) for _, m in matches if m]
    if not sensors:
        raise ValueError("No ToF columns found")
    W = H = int(np.sqrt(max(idxs) + 1))
    T = len(df)
    D = len(sensors)
    tensor = np.full((T, D, H, W), fill_value, dtype=np.float32)
    for d, sn in enumerate(sensors):
        for idx in range(H * W):
            col = f"{prefix}{sn}_v{idx}"
            if col in df.columns:
                vals = df[col].replace(-1, fill_value).to_numpy(dtype=np.float32)
                r, c = divmod(idx, W)
                tensor[:, d, r, c] = vals
    return tensor

# ============================================================
# L. 利き手反転正規化 (Y/Z flip for left‑handed)
# ============================================================

def handedness_correction(df):
    df = df.copy()
    left = df['handedness']==0
    # 加速度・ジャイロの Y/Z 軸反転
    for col in ['acc_y','acc_z','rot_y','rot_z']:
        df.loc[left, col] *= -1
    # ToF センサー X 軸反転 (例)
    for col in df.columns:
        if col.startswith('tof_') and col.endswith('_x'):
            df.loc[left, col] *= -1
    return df
# セル1：利き手補正 v2 定義
def handedness_correction_v2(df):
    """
    左利き(subject: handedness==0) に対し、
    ・加速度(acc_*), ジャイロ(gyro_*), 回転(rot_*) の Y/Z 軸を反転
    ・磁力計(mag_*) があれば同様に反転
    ・ToF 全チャンネルを反転（必要に応じて調整）
    """
    df = df.copy()
    left = df['handedness'] == 0

    # IMU 系センサー
    sensors = {
        'acc': ['x','y','z'],
        'gyro': ['x','y','z'],
        'rot':  ['w','x','y','z'],
        'mag':  ['x','y','z'],
    }
    for sensor, axes in sensors.items():
        for axis in axes:
            col = f"{sensor}_{axis}"
            if col in df.columns:
                # X/W 軸はそのまま、Y/Z 軸は符号反転
                factor = -1 if axis in ['y','z'] else 1
                df.loc[left, col] *= factor

    # ToF センサー全チャンネル反転
    for col in df.columns:
        if col.startswith('tof_'):
            df.loc[left, col] *= -1

    return df

# ============================================================
# M. Wavelet 周波数特徴 (DWT energies)
# ============================================================

def compute_wavelet_features(X_windows: np.ndarray, wavelet: str = "db4", level: int = 3) -> np.ndarray:
    """Block I: discrete wavelet band energies using PyWavelets."""
    import pywt
    feats = []
    for w in X_windows:
        ax_feats = []
        for i in range(w.shape[1]):
            coeffs = pywt.wavedec(w[:, i], wavelet=wavelet, level=level)
            ax_feats += [np.sum(c ** 2) for c in coeffs]
        feats.append(ax_feats)
    return np.array(feats, dtype=np.float32)


# ============================================================
# Utility : Missing-value Cleaning
# =============================

def clean_missing_sensor_data(
    df: pd.DataFrame,
    sensor_cols: list,
    group_cols: tuple | list = ("subject", "sequence_id"),
    time_col: str = "timestamp",
    short_gap: int = 5,
) -> pd.DataFrame:
    """Fill missing sensor values using interpolation and ffill/bfill.

    Short gaps of up to ``short_gap`` consecutive NaNs are linearly interpolated
    for each sequence.  Longer gaps are filled with forward/backward fill.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe containing sensor readings.
    sensor_cols : list
        Columns corresponding to sensor values.
    group_cols : tuple | list, optional
        Columns that define a sequence (default is ("subject", "sequence_id")).
    time_col : str, optional
        Timestamp column used to sort values before interpolation.
    short_gap : int, optional
        Maximum length of gap to interpolate linearly.

    Returns
    -------
    pd.DataFrame
        Dataframe with missing values filled.
    """

    df = df.sort_values(list(group_cols) + [time_col]).copy()
    df[sensor_cols] = df[sensor_cols].replace(-1, np.nan)

    grouped = df.groupby(list(group_cols))
    for _, idx in grouped.groups.items():
        segment = df.loc[idx, sensor_cols]
        segment = segment.interpolate(
            method="linear",
            limit=short_gap,
            limit_direction="both",
        )
        segment = segment.ffill().bfill()
        df.loc[idx, sensor_cols] = segment

    return df

# セル：前処理関数をセンサー種別ごとに最適化
import tempfile, shutil, uuid, os
from pathlib import Path
from joblib import Parallel, delayed
from tqdm.auto import tqdm
import pandas as pd, numpy as np
import gc
# Optional: quaternion補間にSLERPを使う場合
# from scipy.spatial.transform import Rotation, Slerp

def _interpolate_and_dump(
    seq_df: pd.DataFrame,
    sensor_type_groups: dict,
    interp_params: dict,
    time_col: str,
    tmp_dir: str,
) -> str:
    seg = seq_df.sort_values(time_col).copy()

    # --- ToF: -1→NaN  (Accelerometer の -1/0 は別関数で対処済とする) ---
    tof_cols = sensor_type_groups.get("ToF_Sensor", [])
    if tof_cols:
        seg[tof_cols] = seg[tof_cols].replace(-1, np.nan)

    # --- 各センサー種別ごとに補間処理 ----------
    for sensor_type, cols in sensor_type_groups.items():
        params = interp_params[sensor_type]
        if sensor_type == "Accelerometer":
            # 線形補間のみ、limitフレームまで
            seg[cols] = seg[cols].interpolate(
                limit=params["limit"],
                limit_direction="both",
                method="linear"
            )
            # 端点は埋めず NaN のまま
        elif sensor_type == "Rotation":
            # 現状は線形補間で代用（要SLERPアップデート）
            seg[cols] = seg[cols].interpolate(
                limit=params["limit"],
                limit_direction="both",
                method="linear"
            )
        elif sensor_type == "ToF_Sensor":
            if params.get("method") == "linear":
                seg[cols] = seg[cols].interpolate(
                    limit=params["limit"],
                    limit_direction="both",
                    method="linear"
                )
            # 長い欠損は 0 埋め
            seg[cols] = seg[cols].fillna(params["fill_value"])
        elif sensor_type == "Thermal":
            seg[cols] = (
                seg[cols]
                .interpolate(limit=params["limit"], limit_direction="both", method="linear")
                .ffill()
                .bfill()
            )

    # --- ファイル出力 ---
    fname = f"seq_{seq_df['sequence_id'].iloc[0]}_{uuid.uuid4().hex}.parquet"
    out_path = Path(tmp_dir) / fname
    seg.to_parquet(out_path, compression="zstd")

    # メモリ開放
    del seg; gc.collect()
    return str(out_path)


def clean_missing_sensor_data_parallel_disk(
    df: pd.DataFrame,
    sensor_type_groups: dict,
    group_cols=("subject", "sequence_id"),
    time_col: str = "sequence_counter",
    interp_params: dict | None = None,
    n_jobs: int = -1,
    tmp_parent: str | Path = "tmp_clean_chunks",
    keep_tmp: bool = False,
) -> pd.DataFrame:
    """
    センサー種別ごとの補間ルールを適用して並列処理＋ディスク結合。
    sensor_type_groups: {
      "Accelerometer": [...],
      "Rotation": [...],
      "ToF_Sensor": [...],
      "Thermal": [...],
    }
    interp_params: 各種別ごとの dict、例は下記デフォルト参照
    """
    # デフォルトパラメータ
    default_params = {
        "Accelerometer": {"limit": 3},
        "Rotation":      {"limit": 2},
        "ToF_Sensor":    {"method": None,   "limit": 2, "fill_value": 0},
        "Thermal":       {"limit": 5},
    }
    if interp_params is None:
        interp_params = default_params

    # ソート & コピー
    df = df.sort_values(list(group_cols) + [time_col]).copy()

    # --- 一時ディレクトリ準備 ---
    tmp_root = Path(tmp_parent)
    tmp_root.mkdir(exist_ok=True)
    tmp_dir = tempfile.mkdtemp(dir=tmp_root)

    # --- グループ化して並列実行 ---
    groups = [g for _, g in df.groupby(list(group_cols), sort=False)]
    paths = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_interpolate_and_dump)(
            g,
            sensor_type_groups,
            interp_params,
            time_col,
            tmp_dir
        )
        for g in tqdm(groups, desc="interp", unit="seq")
    )

    # --- 結合 & クリーンアップ ---
    cleaned = pd.concat(pd.read_parquet(p) for p in tqdm(paths, desc="concat"))
    cleaned = cleaned.sort_index().reset_index(drop=True)
    if not keep_tmp:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return cleaned


def clean_sensor_missing_values(
    df: pd.DataFrame,
    sensor_type_groups: dict,
    acc_clip: tuple = (-40.0, 40.0),
) -> pd.DataFrame:
    """
    センサータイプごに欠損値を適切に処理する関数（調査結果に基づく）
    
    ── Rules (調査結果ベース) ────────────────────────────────
      • Accelerometer :   0 / −1 は正当値 → NaN 化しない  
                          極端値 |value| > acc_clip → NaN
      • Rotation      :   NaN のみ欠損値
      • ToF_Sensor    :   NaN, -1 → 欠損値
      • Thermal       :   NaN のみ欠損値
    ─────────────────────────────────────────────────────────
    """
    df = df.copy()

    # --- Accelerometer: -1と0をNaN化 --------------------------------
    acc_cols = sensor_type_groups.get("Accelerometer", [])
    if acc_cols:
        df[acc_cols] = df[acc_cols].where(
            df[acc_cols].abs().le(acc_clip[1]),  # clip both sides
            np.nan
        )

    # --- ToF_Sensor: -1をNaN化 -------------------------------------
    tof_cols = sensor_type_groups.get("ToF_Sensor", [])
    if tof_cols:
        for col in tof_cols:
            df.loc[df[col] == -1, col] = np.nan

    # --- Rotation, Thermal: NaNのみ → 変更なし ----------------------
    # 何も処理しない

    return df

