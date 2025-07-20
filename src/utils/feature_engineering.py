import numpy as np
import pandas as pd
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
    pim = PersistenceImage(sigma=sigma, n_bins=n_bins)
    feats = []
    for w in X_windows:
        e = emb.fit_transform(w)
        d = vrp.fit_transform(e)
        img = pim.fit_transform(d)
        feats.append(img.reshape(-1))
    return np.array(feats, dtype=np.float32)

def compute_persistence_image_features_batch(X_windows: np.ndarray, dimension:int=2, n_bins:int=16, sigma:float=0.1) -> np.ndarray:
    from gtda.time_series import TakensEmbedding
    from gtda.homology import VietorisRipsPersistence
    from gtda.diagrams import PersistenceImage

    # デバッグ: TDA処理前のNaN値確認
    nan_count = np.isnan(X_windows).sum()
    if nan_count > 0:
        print(f"⚠️  TDA処理前: NaN値 {nan_count} 個を発見")
        print(f"   データ形状: {X_windows.shape}")
        
        # NaN値を処理
        X_windows_clean = np.nan_to_num(X_windows, nan=0.0, posinf=1.0, neginf=-1.0)
        print(f"✅ NaN値を0.0に置換しました")
    else:
        X_windows_clean = X_windows
        print(f"✅ TDA処理前: NaN値なし")

    # 1. Takens埋め込み（全ウィンドウまとめて）
    emb = TakensEmbedding(time_delay=1, dimension=dimension)
    embedded = emb.fit_transform(X_windows_clean)  # shape: (n_samples, new_len, dimension)

    # 2. パーシステンス図（全ウィンドウまとめて）
    vrp = VietorisRipsPersistence(homology_dimensions=[0, 1])
    diagrams = vrp.fit_transform(embedded)   # shape: (n_samples, n_points, 3)

    # 3. パーシステンス画像（全ウィンドウまとめて）
    pim = PersistenceImage(sigma=sigma, n_bins=n_bins)
    images = pim.fit_transform(diagrams)     # shape: (n_samples, n_bins, n_bins)

    # 4. ベクトル化
    feats = images.reshape(images.shape[0], -1)
    return feats.astype(np.float32)

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
    return ((X_windows - recon) ** 2).mean(axis=(1, 2), keepdims=True).reshape(len(X_windows), 1)



# ============================================================
# M. Wavelet 周波数特徴 (DWT energies)
# ============================================================

def compute_wavelet_features(X_windows: np.ndarray, wavelet: str = "db4", level: int = 3) -> np.ndarray:
    """Block I: discrete wavelet band energies using PyWavelets."""
    import pywt
    
    # NaN値処理
    nan_count = np.isnan(X_windows).sum()
    if nan_count > 0:
        print(f"⚠️  Wavelet処理前: NaN値 {nan_count} 個を発見")
        X_windows_clean = np.nan_to_num(X_windows, nan=0.0, posinf=1.0, neginf=-1.0)
        print(f"✅ NaN値を0.0に置換しました")
    else:
        X_windows_clean = X_windows
    
    feats = []
    for w in X_windows_clean:
        ax_feats = []
        for i in range(w.shape[1]):
            coeffs = pywt.wavedec(w[:, i], wavelet=wavelet, level=level)
            ax_feats += [np.sum(c ** 2) for c in coeffs]
        feats.append(ax_feats)
    return np.array(feats, dtype=np.float32)
