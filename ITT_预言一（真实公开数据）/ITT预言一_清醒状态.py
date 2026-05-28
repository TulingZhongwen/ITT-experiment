```python
#!/usr/bin/env python3
"""
ITT Prophecy 1: Closed-loop causal strength Λ > Λ_noise
Standardized pipeline for conscious (e.g., awake) EEG data, with state-space trajectory visualization.

Usage:
    python itt_prophecy1_awake.py --file <path_to_eeg_file> --channel <channel_name>
"""

import numpy as np
import mne
from scipy.fft import rfft, irfft
from scipy.signal import butter, filtfilt, detrend, resample
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
import argparse
import sys

warnings.filterwarnings("ignore")

# ========== 1. Parameter estimation ==========
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

def cao_embedding_dim(x, tau, max_dim=12, Rtol=10):
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
        raise ValueError(f"Not enough data: len={len(x)}, tau={tau}, d={d}")
    traj = np.zeros((n, d))
    for i in range(n):
        traj[i] = x[i : i + d*tau : tau]
    return traj

# ========== 2. Lambda computation ==========
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
    if traces:
        return np.mean(traces)
    else:
        return 0.0

# ========== 3. IAAFT surrogate ==========
def iaaft_surrogate(x, n_iter=10):
    n = len(x)
    fft_vals = rfft(x)
    amplitudes = np.abs(fft_vals)
    phases = np.random.uniform(0, 2*np.pi, len(amplitudes))
    phases[0] = 0
    new_fft = amplitudes * np.exp(1j * phases)
    s = irfft(new_fft, n=n)
    sorted_orig = np.sort(x)
    for _ in range(n_iter):
        s = np.sort(s)
        rank = np.argsort(np.argsort(s))
        s = sorted_orig[rank]
        fft_s = rfft(s)
        s = irfft(amplitudes * np.exp(1j * np.angle(fft_s)), n=n)
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

# ========== 4. Preprocessing ==========
def preprocess_signal(data, sfreq, target_sf=100, lowcut=0.5, highcut=40, duration_sec=60):
    if duration_sec is not None:
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

# ========== 5. Trajectory visualization ==========
def plot_trajectory(traj, tau, d, save_path='trajectory.png'):
    """Plot state-space trajectory (2D or 3D)."""
    if d >= 3:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], linewidth=0.5, color='blue', alpha=0.7)
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Dimension 3')
        ax.set_title(f'State Space Trajectory (τ={tau}, d={d})')
    else:
        plt.figure(figsize=(6, 5))
        plt.plot(traj[:, 0], traj[:, 1], linewidth=0.5, color='blue', alpha=0.7)
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title(f'State Space Trajectory (τ={tau}, d={d})')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Trajectory plot saved to {save_path}")

# ========== 6. Main test function ==========
def test_prophecy1_awake(data, sfreq, duration_sec=60, n_surrogates=50, verbose=True):
    data, sfreq = preprocess_signal(data, sfreq, duration_sec=duration_sec)
    if verbose:
        print(f"Preprocessed signal: min={data.min():.2e}, max={data.max():.2e}, std={data.std():.2e}")

    tau = mutual_info_first_min(data, max_lag=100)
    d = cao_embedding_dim(data, tau, max_dim=12)
    if verbose:
        print(f"Estimated tau = {tau}, d = {d}")

    # Reconstruct trajectory for visualization
    traj = reconstruct(data, tau, d)
    if verbose:
        plot_trajectory(traj, tau, d)

    dt = 1.0 / sfreq
    lam_real, lam_noise, p = lambda_significance(data, tau, d, dt, n_surrogates=n_surrogates)
    passed = lam_real > lam_noise and p < 0.05

    if verbose:
        print(f"Λ_real = {lam_real:.6f}, Λ_noise = {lam_noise:.6f}, p = {p:.4f}")
        print(f"Prophecy 1 passed: {passed}")

    return {
        'tau': tau,
        'd': d,
        'Lambda_real': lam_real,
        'Lambda_noise': lam_noise,
        'p_value': p,
        'passed': passed
    }

# ========== 7. Command-line interface ==========
def load_eeg_file(file_path, channel=None):
    if file_path.endswith('.fif'):
        raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)
    elif file_path.endswith('.set'):
        raw = mne.io.read_raw_eeglab(file_path, preload=True, verbose=False)
    elif file_path.endswith('.vhdr'):
        raw = mne.io.read_raw_brainvision(file_path, preload=True, verbose=False)
    else:
        raise ValueError("Unsupported file format. Use .fif, .set, or .vhdr")
    raw.pick_types(eeg=True)
    if channel is None:
        channel = raw.ch_names[0]
        print(f"Using first EEG channel: {channel}")
    else:
        if channel not in raw.ch_names:
            raise ValueError(f"Channel {channel} not found. Available: {raw.ch_names[:5]}...")
    data = raw.get_data(picks=channel).flatten()
    sfreq = raw.info['sfreq']
    return data, sfreq

def main():
    parser = argparse.ArgumentParser(description='ITT Prophecy 1 for conscious (awake) EEG')
    parser.add_argument('--file', type=str, help='Path to EEG file (.fif, .set, .vhdr)')
    parser.add_argument('--channel', type=str, default=None, help='Channel name (e.g., Cz, EEG001)')
    parser.add_argument('--duration', type=float, default=60, help='Duration in seconds (default 60)')
    parser.add_argument('--surrogates', type=int, default=50, help='Number of IAAFT surrogates (default 50)')
    args = parser.parse_args()

    if args.file:
        print(f"Loading EEG from {args.file}")
        data, sfreq = load_eeg_file(args.file, args.channel)
    else:
        print("No file provided. Using MNE sample data (awake EEG).")
        data_path = mne.datasets.sample.data_path()
        raw = mne.io.read_raw_fif(data_path + '/MEG/sample/sample_audvis_raw.fif', preload=True, verbose=False)
        raw.pick_types(eeg=True)
        channel = args.channel if args.channel else raw.ch_names[0]
        print(f"Using channel: {channel}")
        data = raw.get_data(picks=channel).flatten()
        sfreq = raw.info['sfreq']

    result = test_prophecy1_awake(data, sfreq, duration_sec=args.duration, n_surrogates=args.surrogates)

    print("\n" + "="*50)
    print("ITT Prophecy 1 Result (Awake/Conscious State)")
    print("="*50)
    print(f"τ = {result['tau']}, d = {result['d']}")
    print(f"Λ_real = {result['Lambda_real']:.6f}")
    print(f"Λ_noise = {result['Lambda_noise']:.6f}")
    print(f"p-value = {result['p_value']:.4f}")
    print(f"Prophecy 1 passed: {result['passed']}")

if __name__ == "__main__":
    main()
```
