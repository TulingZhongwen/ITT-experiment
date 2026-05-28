"""
ITT 预言三：方向规则 R 独立复现
使用 Sleep-EDF 数据集（闭眼静息态清醒片段）
输出：R 值、95% Bootstrap 置信区间、群体统计
"""

# ============================================================
# 1. 安装依赖（如在本地运行，可注释掉）
# ============================================================
!pip install -q mne scipy scikit-learn matplotlib

import os
import numpy as np
import mne
from scipy.signal import resample, butter, filtfilt, detrend
from scipy.spatial import KDTree
from sklearn.neighbors import KernelDensity
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 2. 核心函数：参数估计、状态空间重构、方向规则计算
# ============================================================

def mutual_info_first_min(x, max_lag=100):
    """互信息第一极小值估计延迟 tau"""
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
    """Cao方法估计嵌入维数 d"""
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
    """Takens延迟嵌入"""
    n = len(x) - (d-1)*tau
    if n <= 0:
        raise ValueError("数据长度不足")
    traj = np.zeros((n, d))
    for i in range(n):
        traj[i] = x[i : i + d*tau : tau]
    return traj

def compute_direction_rule(traj, n_bootstrap=500):
    """
    计算方向规则一致性比例 R 及其 95% Bootstrap 置信区间
    使用核密度估计 + 局部线性回归估计梯度
    """
    T, d = traj.shape
    # 自适应带宽 (Silverman 规则)
    sigma = np.mean(np.std(traj, axis=0))
    bandwidth = 1.06 * sigma * T ** (-1/(d+4))
    
    # 核密度估计
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(traj)
    log_p = kde.score_samples(traj)
    
    # 局部线性回归估计 ∇U
    grad_U = np.zeros((T, d))
    n_neighbors = min(50, T-1)
    for i in range(T):
        dist = np.linalg.norm(traj - traj[i], axis=1)
        idx = np.argsort(dist)[1:n_neighbors+1]
        if len(idx) < d+1:
            continue
        X = traj[idx] - traj[i]
        y = log_p[idx] - log_p[i]
        weights = np.exp(-dist[idx]**2 / (2*bandwidth**2))
        W = np.diag(weights)
        XW = X.T @ W
        XWX = XW @ X
        if np.linalg.cond(XWX) < 1e12:
            beta = np.linalg.solve(XWX, XW @ y)
            grad_U[i] = -beta   # ∇U = -∇log p
        else:
            grad_U[i] = np.zeros(d)
    
    # 自指映射（最近邻）
    tree = KDTree(traj)
    _, idx = tree.query(traj, k=2)
    Pi = traj[idx[:, 1]]
    Delta = traj - Pi
    norm = np.linalg.norm(Delta, axis=1, keepdims=True)
    Delta_unit = Delta / (norm + 1e-12)
    
    # 方向规则符号
    dot = np.sum(Delta_unit * grad_U, axis=1)
    sigma_sign = np.sign(dot).astype(int)
    R_val = np.mean(sigma_sign > 0)
    
    # Bootstrap 置信区间
    boot_R = [np.mean(np.random.choice(sigma_sign, size=T, replace=True) > 0) for _ in range(n_bootstrap)]
    ci_low, ci_high = np.percentile(boot_R, [2.5, 97.5])
    return R_val, ci_low, ci_high

def preprocess_signal(data, sfreq, target_sf=250, lowcut=0.5, highcut=40, duration_sec=60):
    """预处理：降采样、滤波、去趋势、归一化"""
    if duration_sec:
        n = int(duration_sec * sfreq)
        if len(data) > n:
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

def get_wake_segment(raw, duration_sec=60):
    """从 Sleep-EDF Raw 对象中提取第一个足够长的清醒片段"""
    annot = raw.annotations
    for onset, dur, desc in zip(annot.onset, annot.duration, annot.description):
        if 'W' in str(desc) and dur >= duration_sec + 10:
            start = onset + 5
            end = start + duration_sec
            # 选择第一个 EEG 通道
            eeg_picks = mne.pick_types(raw.info, eeg=True)
            if len(eeg_picks) == 0:
                raise ValueError("未找到 EEG 通道")
            ch_name = raw.ch_names[eeg_picks[0]]
            data, _ = raw[ch_name, int(start * raw.info['sfreq']):int(end * raw.info['sfreq'])]
            sfreq = raw.info['sfreq']
            return data.flatten(), sfreq, ch_name
    raise ValueError("未找到足够长的清醒片段")

# ============================================================
# 3. 主程序：自动下载 Sleep-EDF 并分析单个受试者
# ============================================================
def run_single_subject(subject_id=0):
    """
    分析单个受试者 (subject_id: 0 = SC4001, 1 = SC4002, ...)
    返回 R, ci_low, ci_high
    """
    from mne.datasets.sleep_physionet.age import fetch_data
    raw_fnames, ann_fnames = fetch_data(subjects=[subject_id], verbose=False)
    raw = mne.io.read_raw_edf(raw_fnames[0], preload=True, verbose=False)
    annot = mne.read_annotations(ann_fnames[0])
    raw.set_annotations(annot)
    
    signal, sfreq_orig, ch_name = get_wake_segment(raw, duration_sec=60)
    print(f"受试者 {subject_id}: 通道 {ch_name}, 原始采样率 {sfreq_orig} Hz")
    data, sfreq = preprocess_signal(signal, sfreq_orig, duration_sec=60)
    tau = mutual_info_first_min(data, max_lag=100)
    d = cao_embedding_dim(data, tau, max_dim=6)
    d = max(d, 2)
    traj = reconstruct(data, tau, d)
    R, ci_low, ci_high = compute_direction_rule(traj, n_bootstrap=500)
    print(f"tau={tau}, d={d}, R={R:.3f}, 95% CI [{ci_low:.3f}, {ci_high:.3f}]")
    return R, ci_low, ci_high

def run_batch(subject_ids=[0,1,2,3,4]):
    """
    批量分析多个受试者，输出群体统计
    """
    from mne.datasets.sleep_physionet.age import fetch_data
    results = []
    for sid in subject_ids:
        try:
            raw_fnames, ann_fnames = fetch_data(subjects=[sid], verbose=False)
            raw = mne.io.read_raw_edf(raw_fnames[0], preload=True, verbose=False)
            annot = mne.read_annotations(ann_fnames[0])
            raw.set_annotations(annot)
            signal, sfreq_orig, ch_name = get_wake_segment(raw, duration_sec=60)
            data, sfreq = preprocess_signal(signal, sfreq_orig, duration_sec=60)
            tau = mutual_info_first_min(data, max_lag=100)
            d = cao_embedding_dim(data, tau, max_dim=6)
            d = max(d, 2)
            traj = reconstruct(data, tau, d)
            R, ci_low, ci_high = compute_direction_rule(traj, n_bootstrap=500)
            print(f"Subject {sid}: R={R:.3f} CI=[{ci_low:.3f},{ci_high:.3f}]")
            results.append(R)
        except Exception as e:
            print(f"Subject {sid} failed: {e}")
    if len(results) < 2:
        print("样本量不足，无法进行统计检验")
        return
    print("\n=== 群体统计 ===")
    print(f"样本量: {len(results)}")
    print(f"R 均值: {np.mean(results):.3f} ± {np.std(results):.3f}")
    t_stat, p_val = stats.ttest_1samp(results, 0.5)
    print(f"单样本 t 检验 (vs 0.5): t = {t_stat:.3f}, p = {p_val:.4f}")
    if p_val < 0.05:
        print("✓ 群体水平上 R 显著大于 0.5，支持预言三")
    else:
        print("⚠ R 未显著大于 0.5")

# ============================================================
# 4. 执行示例
# ============================================================
if __name__ == "__main__":
    print("="*60)
    print("ITT 预言三：方向规则 R 独立复现 (Sleep-EDF)")
    print("="*60)
    # 选项1：运行单个受试者
    # R, ci_low, ci_high = run_single_subject(subject_id=0)
    
    # 选项2：批量分析（推荐）
    run_batch(subject_ids=[0,1,2,3,4])
