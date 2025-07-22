import numpy as np
import pandas as pd
import warnings
from scipy.signal import find_peaks

# ============================================================
# A. 基本統計量 (mean / std / range / RMS / energy)
# ============================================================
### --- Block A Summary ---------------------------------------
# dims: 56 | 推奨モデル: LightGBM_all / non
# 広域量的特徴――窓全体の強度・分布を圧縮
# ------------------------------------------------------------
def compute_basic_statistics(X_windows: np.ndarray) -> np.ndarray:
    """Compute Block F statistics per window."""
    # 外れ値にロバストな統計量を使用
    means  = np.nanmean(X_windows, axis=1)
    stds   = np.nanstd(X_windows, axis=1)
    
    # パーセンタイルベースの範囲（外れ値にロバスト）
    q25 = np.nanpercentile(X_windows, 25, axis=1)
    q75 = np.nanpercentile(X_windows, 75, axis=1)
    ranges = q75 - q25  # 四分位範囲（IQR）
    
    # 外れ値を除外したRMSとエネルギー
    # 各ウィンドウで外れ値を除外してから計算
    rms_values = []
    energy_values = []
    
    for i in range(X_windows.shape[0]):
        window_data = X_windows[i, :, :]
        
        # 外れ値を除外（各軸ごと）
        clean_data = []
        for axis in range(window_data.shape[1]):
            axis_data = window_data[:, axis]
            q1, q99 = np.nanpercentile(axis_data, [1, 99])
            clean_axis = axis_data[(axis_data >= q1) & (axis_data <= q99)]
            clean_data.append(clean_axis)
        
        # 全軸のデータを結合
        all_clean_data = np.concatenate(clean_data)
        
        if len(all_clean_data) > 0:
            rms = np.sqrt(np.nanmean(all_clean_data ** 2))
            energy = np.nansum(all_clean_data ** 2)
        else:
            rms = 0.0
            energy = 0.0
            
        rms_values.append(rms)
        energy_values.append(energy)
    
    rms = np.array(rms_values)
    energy = np.array(energy_values)
    
    if X_windows.shape[2] >= 3:
        mag = np.linalg.norm(X_windows[:, :, :3], axis=2)
        mag_mean = np.nanmean(mag, axis=1, keepdims=True)
        mag_std  = np.nanstd(mag, axis=1, keepdims=True)
        return np.hstack([means, stds, ranges, rms.reshape(-1, 1), energy.reshape(-1, 1), mag_mean, mag_std])
    else:
        return np.hstack([means, stds, ranges, rms.reshape(-1, 1), energy.reshape(-1, 1)])


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
    """Compute peak features per window."""
    def extract_peak_features(window):
        features = []
        for axis in range(window.shape[1]):
            signal = window[:, axis]
            
            # 外れ値を除外してからピーク検出
            q1, q99 = np.nanpercentile(signal, [1, 99])
            clean_signal = signal[(signal >= q1) & (signal <= q99)]
            
            if len(clean_signal) > 10:  # 十分なデータがある場合のみ
                peaks, _ = find_peaks(clean_signal, height=np.nanmean(clean_signal))
                n_peaks = len(peaks)
                
                if n_peaks > 0:
                    peak_heights = clean_signal[peaks]
                    avg_peak_height = np.nanmean(peak_heights)
                    max_peak_height = np.nanmax(peak_heights)
                    peak_ratio = n_peaks / len(clean_signal)
                else:
                    avg_peak_height = 0.0
                    max_peak_height = 0.0
                    peak_ratio = 0.0
            else:
                n_peaks = 0
                avg_peak_height = 0.0
                max_peak_height = 0.0
                peak_ratio = 0.0
            
            features.extend([n_peaks, avg_peak_height, max_peak_height, peak_ratio])
        return features
    
    return np.vstack([extract_peak_features(w) for w in X_windows])


# ============================================================
# C. FFT バンドエネルギー (0.5–20 Hz)
# ============================================================
### --- Block C Summary ---------------------------------------
# dims: 10 | 推奨モデル: LightGBM_all / non
# 動作リズム・速度の周波数分布
# ------------------------------------------------------------

def compute_fft_band_energy(X_windows: np.ndarray, fs: float = 50.0, bands=None) -> np.ndarray:
    """Compute block G FFT band energies."""
    if bands is None:
        bands = [(0.5, 2), (2, 5), (5, 10), (10, 20)]
    n_win, win_len, n_feat = X_windows.shape
    freqs = np.fft.rfftfreq(win_len, d=1.0 / fs)
    
    energies = []
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        band_energies = []
        
        for i in range(n_win):
            window_energies = []
            for j in range(n_feat):
                signal = X_windows[i, :, j]
                
                # 外れ値を除外してからFFT計算
                q1, q99 = np.nanpercentile(signal, [1, 99])
                clean_signal = signal[(signal >= q1) & (signal <= q99)]
                
                if len(clean_signal) > 10:
                    # パディングして元の長さに戻す
                    padded_signal = np.zeros(win_len)
                    padded_signal[:len(clean_signal)] = clean_signal
                    
                    power = np.abs(np.fft.rfft(padded_signal)) ** 2
                    energy = power[mask].sum()
                else:
                    energy = 0.0
                
                window_energies.append(energy)
            band_energies.append(window_energies)
        
        energies.append(np.array(band_energies))
    
    return np.hstack(energies)


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

def compute_persistence_image_features_batch(
    X_windows: np.ndarray, dimension: int = 2, n_bins: int = 16, sigma: float = 0.1
) -> np.ndarray:
    try:
        from gtda.time_series import TakensEmbedding
        from gtda.homology import VietorisRipsPersistence
        from gtda.diagrams import PersistenceImage
    except ImportError:
        warnings.warn(
            "giotto-tda がインストールされていないため、TDA特徴量をスキップします。"
        )
        return np.zeros((X_windows.shape[0], 0), dtype=np.float32)

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
    return ((X_windows - recon) ** 2).mean(axis=(1, 2), keepdims=True)



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


# ============================================================
# N. ToF イベント特徴量
# ============================================================

def compute_tof_event_features(X_windows: np.ndarray) -> np.ndarray:
    """Compute simple ToF event features.

    各ウィンドウについて以下を計算します。

    - 最小距離値
    - 時系列方向の絶対変化率平均
    """

    window_min = X_windows.min(axis=(1, 2, 3, 4))[:, None]
    diff = np.diff(X_windows, axis=1)
    rate = np.mean(np.abs(diff), axis=(1, 2, 3, 4))[:, None]
    return np.hstack([window_min, rate])


# ============================================================
# O. 温度勾配特徴量
# ============================================================

def compute_temperature_gradient_features(X_windows: np.ndarray) -> np.ndarray:
    """Compute temperature gradient based features.

    Parameters
    ----------
    X_windows : np.ndarray
        Shape ``(N, T, C)`` where ``C`` is the数 of thermal sensors.

    Returns
    -------
    np.ndarray
        ``(N, C*3)`` array containing mean/std of first differences and
        peak（max-min）for each channel.
    """

    diffs = np.diff(X_windows, axis=1)
    diff_mean = diffs.mean(axis=1)
    diff_std = diffs.std(axis=1)
    peak_width = X_windows.max(axis=1) - X_windows.min(axis=1)
    return np.hstack([diff_mean, diff_std, peak_width])
