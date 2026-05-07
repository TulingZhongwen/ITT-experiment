"""
ITT Core Module (Improved) – Inertia‑Tension Theory
Version: 1.1 (with numerical stability improvements)
Author: Tuling Zhongwen (图灵中文)
"""

import numpy as np
from scipy.spatial import KDTree
from scipy.ndimage import median_filter
from sklearn.metrics import mutual_info_score

# ---------- 1. 状态空间重构 ----------
def reconstruct(x, tau, d):
    """Delay embedding."""
    n = len(x) - (d-1)*tau
    if n <= 0:
        raise ValueError(f"Not enough data for tau={tau}, d={d}")
    indices = np.arange(d)[:, None] * tau + np.arange(n)
    return x[indices].T
    
# ---------- 2. 延迟 tau 估计 ----------
def mutual_info_first_min(x, max_lag=50):
    """Estimate tau as first minimum of mutual information."""
    mi = []
    for lag in range(1, max_lag+1):
        mi.append(mutual_info_score(x[:-lag], x[lag:]))
    for i in range(2, len(mi)-1):
        if mi[i] < mi[i-1] and mi[i] < mi[i+1]:
            return i
    return max_lag//2

# ---------- 3. 局部嵌入维数（滑动窗口） ----------
def local_embedding_dim(signal, tau, fs, window_sec=5, max_dim=30):
    """
    Time‑varying embedding dimension using sliding‑window FNN.
    Returns array of length len(signal) with estimated d at each sample.
    """
    T = len(signal)
    win_len = int(window_sec * fs)
    d_local = np.ones(T, dtype=int) * 2
    padding = (win_len // 2)
    # pad signal for edge handling
    padded = np.pad(signal, (padding, padding), mode='reflect')
    for i in range(T):
        start = i
        end = i + win_len
        win = padded[start:end]
        if len(win) < 100:
            d_local[i] = 2
            continue
        best_d = 2
        for d in range(2, max_dim+1):
            # reconstruct
            n_pts = len(win) - (d-1)*tau
            if n_pts < d+1:
                break
            traj = reconstruct(win, tau, d)
            # FNN ratio
            fnn = 0
            for j in range(traj.shape[0]):
                dist = np.linalg.norm(traj[j] - traj, axis=1)
                dist[j] = np.inf
                nn = np.argmin(dist)
                # test in next dimension
                if j+tau < len(win) and nn+tau < len(win):
                    x1 = np.append(traj[j], win[j + d*tau])
                    x2 = np.append(traj[nn], win[nn + d*tau])
                    dist_new = np.linalg.norm(x1 - x2)
                    if dist_new > 10 * dist[nn]:
                        fnn += 1
            if fnn / traj.shape[0] < 0.01:
                best_d = d
                break
        d_local[i] = best_d
    # median smooth
    d_local = median_filter(d_local, size=11)
    return d_local

# ---------- 4. 闭环因果强度 Λ（改进版） ----------
def compute_lambda(traj, dt, n_neighbors=None, ridge_alpha=0.01):
    """
    Compute Λ with condition number check and adaptive ridge regression.
    """
    T, d = traj.shape
    if n_neighbors is None:
        n_neighbors = max(2*d, 30)
    n_neighbors = min(n_neighbors, T-1)
    traces = []
    tree = KDTree(traj)
    for i in range(T-1):
        # find neighbors
        dist, idx = tree.query(traj[i], k=n_neighbors+1)
        neighbors = idx[1:]
        X = traj[neighbors] - traj[i]
        Y = traj[neighbors+1] - traj[neighbors]
        # condition number via SVD
        _, s, _ = np.linalg.svd(X, full_matrices=False)
        cond = s[0] / s[-1] if s[-1] > 0 else 1e16
        if cond > 1e6:
            # ridge regression
            XTX = X.T @ X
            reg = ridge_alpha * np.trace(XTX) / d * np.eye(d)
            J = np.linalg.inv(XTX + reg) @ X.T @ Y
            J = J.T
        else:
            J, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
            J = J.T
        traces.append(np.abs(np.trace(J)) / d)
    return np.mean(traces)

# ---------- 5. 噪声基准 Λ_noise ----------
def lambda_significance(signal, tau, d, dt, n_shuffle=100):
    """Compute Λ_real, 95% threshold, and p‑value via permutation test."""
    traj = reconstruct(signal, tau, d)
    lam_real = compute_lambda(traj, dt)
    lam_shuf = []
    for _ in range(n_shuffle):
        shuf = np.random.permutation(signal)
        traj_shuf = reconstruct(shuf, tau, d)
        lam_shuf.append(compute_lambda(traj_shuf, dt))
    thresh = np.percentile(lam_shuf, 95)
    p = (np.sum(np.array(lam_shuf) >= lam_real) + 1) / (n_shuffle + 1)
    return lam_real, thresh, p

# ---------- 6. 可访问自指距离 Θ（改进版） ----------
def compute_theta(traj, delta, c=2.0, time_window=5000, verify_steps=5):
    """
    Compute accessible self‑distance Θ(s) with time‑localized recurrence.
    """
    T, d = traj.shape
    R_indices = set()
    # Use a shorter window for large T
    window = min(time_window, T//10) if T > 10000 else time_window
    for i in range(T):
        start = max(0, i - window)
        end = min(T, i + window + 1)
        for j in range(start, end):
            if i == j:
                continue
            if np.linalg.norm(traj[i] - traj[j]) < c * delta:
                # double check by forward propagation
                ok = True
                for step in range(1, verify_steps+1):
                    if i+step >= T or j+step >= T:
                        ok = False
                        break
                    if np.linalg.norm(traj[i+step] - traj[j+step]) >= c * delta:
                        ok = False
                        break
                if ok:
                    R_indices.add(i)
                    R_indices.add(j)
    if len(R_indices) == 0:
        return 0.0
    R = traj[list(R_indices)]
    treeR = KDTree(R)
    dist, _ = treeR.query(traj, k=1)
    Theta = np.maximum(0, (dist - delta) / delta)
    return np.mean(Theta)

# ---------- 7. 辅助：全局维数（用于平稳数据） ----------
def false_nearest_neighbor(x, tau, max_dim=20, rtol=10):
    """Find global embedding dimension using FNN."""
    for dim in range(1, max_dim+1):
        if len(x) <= (dim+1)*tau:
            break
        y = reconstruct(x, tau, dim)
        y_next = reconstruct(x, tau, dim+1)
        dist = np.linalg.norm(y[:, :-1] - y[:, 1:], axis=1)
        dist_next = np.linalg.norm(y_next[:, :-1] - y_next[:, 1:], axis=1)
        ratio = np.sum(dist_next > rtol * dist) / len(dist)
        if ratio < 0.01:
            return dim
    return max_dim
