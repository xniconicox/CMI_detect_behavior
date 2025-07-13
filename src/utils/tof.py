
"""ToF (Time-of-Flight) related utilities."""

import numpy as np
import pandas as pd


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
