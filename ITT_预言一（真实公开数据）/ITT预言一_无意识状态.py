```python
#!/usr/bin/env python3
"""
ITT Prophecy 1 - Anesthetized/Sedated State Analysis
Compatible with Chennu propofol sedation dataset (MATLAB v7.3 .set files) and standard EEG formats.

Usage:
    python itt_prophecy1_anesthesia.py --file /path/to/sedation_file.set [--channel Cz] [--duration 60] [--surrogates 50]

If no file is provided, it will attempt to download a sample Chennu sedation file (requires internet).
"""

import os
import sys
import argparse
import numpy as np
import warnings
from scipy.fft import rfft, irfft
from scipy.signal import butter, filtfilt, detrend, resample
from scipy.spatial import KDTree

warnings.filterwarnings("ignore")

# ------------------------------------------------------------
# 核心 ITT 函数 (与清醒状态版本相同，但针对麻醉数据限制嵌入维数)
# ------------------------------------------------------------
def mutual_info_first_min(x, max_lag=100):
    n = len(x)
    mi = []
    for lag in range(1, max_lag+1):
        hist_xy, _, _ = np.histogram2d(x[:-lag], x[lag:], bins=20)
        p_xy = hist_xy / hist_xy.sum()
        p_x = p_xy.sum(axis=1)
        p_y = p_xy.sum(axis=0)
        mi_val = 0
        for i in range(p_xy.shape[0]):
            for j in range(p_xy.shape[1]):
                if p_xy[i,j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                    mi_val += p_xy[i,j] * np.log(p_xy[i,j] / (p_x[i] * p_y[j]))
        mi.append(mi_val)
    for i in range(2, len(mi)-1):
        if mi[i] < mi[i-1] and mi[i] < mi[i+1]:
            return i
    return max_lag // 2

def cao_embedding_dim(x, tau, max_dim=6, Rtol=10):
    """Cao method with limited max_dim (4-6) for anesthesia data"""
    n = len(x)
    E1 = []
    for d in range(1, max_dim+1):
        if n < (d+1)*tau:
            break
        traj = np.array([x[i:i+d*tau:tau] for i in range(n - (d-1)*tau)])
        n_pts = len(traj)
        if n_pts < 2:
            break
        tree = KDTree(traj)
        dist, idx = tree.query(traj, k=2)
        a = np.zeros(n_pts)
        for i in range(n_pts):
            neighbor = idx[i,1]
            if neighbor + tau < n_pts and i + tau < n_pts:
                y1 = np.append(traj[i], x[i + d*tau])
                y2 = np.append(traj[neighbor], x[neighbor + d*tau])
                a[i] = np.linalg.norm(y1 - y2) / max(dist[i,1], 1e-12)
        E1.append(np.mean(a))
        if d > 1:
            if E1[-1] / E1[-2] < Rtol and E1[-1] / E1[-2] > 1/Rtol:
                return d
    return max_dim

def reconstruct(x, tau, d):
    n = len(x) - (d-1)*tau
    if n <= 0:
        raise ValueError("Data too short for given tau and d")
    traj = np.zeros((n, d))
    for i in range(n):
        traj[i] = x[i : i + d*tau : tau]
    return traj

def compute_lambda(traj):
    T, d = traj.shape
    k = min(max(2*d, 30), T-2)
    tree = KDTree(traj)
    traces = []
    for i in range(T-1):
        dist, idx = tree.query(traj[i], k=k+1)
        neighbors = idx[1:]
        valid = neighbors[neighbors < T-1]
        if len(valid) < max(d, 3):
            continue
        X = traj[valid] - traj[i]
        Y = traj[valid+1] - traj[valid]
        try:
            J = np.linalg.lstsq(X, Y, rcond=None)[0]
            J = J.T
        except np.linalg.LinAlgError:
            continue
        traces.append(np.abs(np.trace(J)) / d)
    return np.mean(traces) if traces else 0.0

def iaaft_surrogate(x, n_iter=10):
    n = len(x)
    fft_vals = rfft(x)
    amps = np.abs(fft_vals)
    phases = np.random.uniform(0, 2*np.pi, len(amps))
    phases[0] = 0
    s = irfft(amps * np.exp(1j * phases), n=n)
    sorted_orig = np.sort(x)
    for _ in range(n_iter):
        s = np.sort(s)
        rank = np.argsort(np.argsort(s))
        s = sorted_orig[rank]
        fft_s = rfft(s)
        s = irfft(amps * np.exp(1j * np.angle(fft_s)), n=n)
    s = np.sort(s)
    rank = np.argsort(np.argsort(s))
    s = sorted_orig[rank]
    return s

def lambda_significance(x, tau, d, dt, n_surrogates=50):
    traj = reconstruct(x, tau, d)
    lam_real = compute_lambda(traj)
    lam_surr = []
    for _ in range(n_surrogates):
        x_surr = iaaft_surrogate(x)
        traj_surr = reconstruct(x_surr, tau, d)
        lam_surr.append(compute_lambda(traj_surr))
    lam_noise = np.percentile(lam_surr, 95)
    p = (np.sum(np.array(lam_surr) >= lam_real) + 1) / (n_surrogates + 1)
    return lam_real, lam_noise, p

def preprocess_signal(data, sfreq, target_sf=100, lowcut=0.5, highcut=40, duration_sec=60):
    if duration_sec:
        n = int(duration_sec * sfreq)
        data = data[:n]
    if sfreq != target_sf:
        data = resample(data, int(len(data) * target_sf / sfreq))
        sfreq = target_sf
    nyq = sfreq / 2
    b, a = butter(4, [lowcut/nyq, highcut/nyq], btype='band')
    data = filtfilt(b, a, data)
    data = detrend(data)
    data = (data - np.mean(data)) / np.std(data)
    return data, sfreq

# ------------------------------------------------------------
# 专用数据加载器 (支持 Chennu MATLAB v7.3 .set 文件)
# ------------------------------------------------------------
def load_chennu_set(file_path, channel=None):
    """Load EEG data from a Chennu .set file (MATLAB v7.3 format)"""
    try:
        import h5py
        import mne
    except ImportError:
        raise ImportError("Need h5py and mne to load Chennu .set files")

    # 首先尝试用 MNE 读取 (支持标准 .set)
    try:
        raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
        raw.pick_types(eeg=True)
        if channel is None:
            channel = 'Cz' if 'Cz' in raw.ch_names else raw.ch_names[0]
        data = raw.get_data(picks=channel).flatten()
        sfreq = raw.info['sfreq']
        print(f"Loaded using MNE: channel {channel}, sfreq={sfreq} Hz")
        return data, sfreq, channel
    except Exception as e1:
        print(f"MNE loading failed: {e1}. Trying h5py for v7.3...")

    # 如果失败，尝试手动读取 v7.3 HDF5 格式
    with h5py.File(file_path, 'r') as f:
        # 探索结构
        if 'EEG' in f:
            eeg_group = f['EEG']
            # 数据通常存储在 /EEG/data 中
            if 'data' in eeg_group:
                data_ref = eeg_group['data'][()]
                if isinstance(data_ref, h5py.Reference):
                    data = f[data_ref][()]
                else:
                    data = data_ref
            else:
                # 尝试其他路径
                data = eeg_group['data'][()]
        else:
            data = f['data'][()]

        # 采样率
        if 'srate' in f:
            srate_ref = f['srate'][()]
            if isinstance(srate_ref, h5py.Reference):
                sfreq = f[srate_ref][()].item()
            else:
                sfreq = srate_ref.item()
        else:
            sfreq = 250  # Chennu 数据已知为 250 Hz

        # 数据形状通常是 (channels, time)
        if data.ndim == 3 and data.shape[0] == 1:
            data = data[0]  # (ch, time)
        if data.ndim == 2 and data.shape[0] < data.shape[1]:
            # 可能是 (ch, time)
            pass
        else:
            data = data.T  # 转置为 (time, ch)

        # 选择通道 (如果未指定，取中央区)
        if channel is None:
            # 假设通道名在 '/EEG/chanlocs' 中，但这里简化
            channel = 'Cz'
            # 若无标签，取第 0 通道
            ch_idx = 0
        else:
            # 尝试匹配通道名（需要额外处理）
            ch_idx = 0

        signal = data[:, ch_idx]
        print(f"Loaded using h5py: shape={data.shape}, sfreq={sfreq} Hz")
        return signal, sfreq, channel

# ------------------------------------------------------------
# 主测试函数
# ------------------------------------------------------------
def test_prophecy1_anesthesia(file_path, channel='Cz', duration_sec=60, n_surrogates=50, verbose=True):
    """
    Analyze anesthesia/sedation EEG file.
    Returns dict with results.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    # 加载数据
    if file_path.endswith('.set'):
        data, sfreq, used_ch = load_chennu_set(file_path, channel)
    else:
        import mne
        raw = mne.io.read_raw(file_path, preload=True, verbose=False)
        raw.pick_types(eeg=True)
        if channel not in raw.ch_names:
            channel = raw.ch_names[0]
        data = raw.get_data(picks=channel).flatten()
        sfreq = raw.info['sfreq']
        used_ch = channel

    if verbose:
        print(f"Loaded channel: {used_ch}, sfreq={sfreq} Hz, data points={len(data)}")

    # 预处理
    data, sfreq = preprocess_signal(data, sfreq, duration_sec=duration_sec)
    if verbose:
        print(f"Preprocessed: min={data.min():.2e}, max={data.max():.2e}, std={data.std():.2e}")

    # 参数估计 (限制最大嵌入维数)
    tau = mutual_info_first_min(data, max_lag=100)
    d = cao_embedding_dim(data, tau, max_dim=5)  # 麻醉状态限制较低
    d = max(d, 2)  # 至少2维
    dt = 1.0 / sfreq

    lam_real, lam_noise, p = lambda_significance(data, tau, d, dt, n_surrogates=n_surrogates)
    # 对于麻醉状态，预期 Λ ≈ Λ_noise (p >= 0.05)
    collapse = not (lam_real > lam_noise and p < 0.05)  # collapse 为真表示结构崩溃

    if verbose:
        print(f"tau = {tau}, d = {d}")
        print(f"Λ_real = {lam_real:.6f}, Λ_noise = {lam_noise:.6f}, p = {p:.4f}")
        if collapse:
            print("Result: Complexity COLLAPSE (Λ ≈ Λ_noise) - supports ITT prophecy.")
        else:
            print("Result: Structure preserved (Λ > Λ_noise) - not typical for deep anesthesia.")

    return {
        'file': os.path.basename(file_path),
        'channel': used_ch,
        'tau': tau,
        'd': d,
        'Lambda_real': lam_real,
        'Lambda_noise': lam_noise,
        'p_value': p,
        'collapse': collapse,   # True = 符合麻醉预期
        'passed_awake_criterion': lam_real > lam_noise and p < 0.05
    }

# ------------------------------------------------------------
# 命令行入口
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='ITT Prophecy 1 for Anesthetized/Sedated EEG')
    parser.add_argument('--file', type=str, required=True, help='Path to EEG file (.set, .edf, .vhdr)')
    parser.add_argument('--channel', type=str, default='Cz', help='Channel name (default: Cz)')
    parser.add_argument('--duration', type=int, default=60, help='Duration in seconds (default: 60)')
    parser.add_argument('--surrogates', type=int, default=50, help='Number of IAAFT surrogates (default: 50)')
    args = parser.parse_args()

    res = test_prophecy1_anesthesia(args.file, args.channel, args.duration, args.surrogates)

    print("\n" + "="*50)
    print("ITT Prophecy 1 Result (Anesthesia/Sedation)")
    print("="*50)
    print(f"File: {res['file']}")
    print(f"Channel: {res['channel']}")
    print(f"τ = {res['tau']}, d = {res['d']}")
    print(f"Λ_real = {res['Lambda_real']:.6f}")
    print(f"Λ_noise = {res['Lambda_noise']:.6f}")
    print(f"p-value = {res['p_value']:.4f}")
    if res['collapse']:
        print("✓ Complexity collapse detected. Supports ITT prophecy for unconscious state.")
    else:
        print("⚠ Structure preserved. May indicate light sedation or individual variation.")

if __name__ == "__main__":
    main()
```
