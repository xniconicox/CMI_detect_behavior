# --- utils/imu.py -------------------------------------------------------
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

GRAVITY = 9.80665   # [m/s²]

def quat_normalize_safe(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat, axis=1, keepdims=True)
    mask_zero = norm.squeeze() < 1e-6
    quat[mask_zero] = np.array([1, 0, 0, 0])          # identity
    quat[~mask_zero] /= norm[~mask_zero]
    return quat

def accel_world_linear(acc: np.ndarray, quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    acc  : (N,3) body-frame raw accelerometer
    quat : (N,4) quaternion (w,x,y,z) body→world

    Returns
    -------
    acc_w  : (N,3) world-frame acceleration
    lin_acc: (N,3) world-frame linear acceleration (= acc_w – g)
    """
    quat = quat_normalize_safe(quat)
    rot  = R.from_quat(quat[:, [1,2,3,0]])            # (x,y,z,w)
    acc_w = rot.apply(acc)
    lin   = acc_w - np.array([0, 0, GRAVITY])
    return acc_w, lin

def add_world_acc_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    前提: 欠損補間 & handedness_correction_v2 後の DataFrame
    追加列:
        acc_w_[xyz], lin_acc_[xyz]
    """
    df = df.copy()
    acc  = df[['acc_x','acc_y','acc_z']].to_numpy(dtype=np.float32)
    quat = quat_normalize_safe(                          # ← ここを変更
        df[['rot_w','rot_x','rot_y','rot_z']].to_numpy(dtype=np.float32)
    )
    acc_w, lin = accel_world_linear(acc, quat)
    df[['acc_w_x','acc_w_y','acc_w_z']]     = acc_w
    df[['lin_acc_x','lin_acc_y','lin_acc_z']] = lin
    return df