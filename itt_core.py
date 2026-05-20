"""
ITT Core Module v2.0 — Inertia-Tension Theory
更新内容：
- IAAFT surrogate 替换简单洗牌
- 新增块状Bootstrap统计检验
- 新增Silverman核密度估计 + 局部线性回归∇U估计
- 新增互信息块长度计算
- 新增高维降级判断
- 统一效应量分层输出
- **修复 false_nearest_neighbor 函数（标准FNN实现）**

Author: 图灵中文
Version: 2.0
"""

import numpy as np
from scipy.spatial import KDTree
from scipy.ndimage import median_filter
from sklearn.metrics import mutual_info_score
from scipy.stats import norm

# ============================================================
# 1. 状态空间重构（不变）
# ============================================================

def reconstruct(x, tau, d):
    """Takens延迟嵌入"""
    n = len(x) - (d-1)*tau
    if n <= 0:
        raise ValueError(f"Not enough data for tau={tau}, d={d}")
    indices = np.arange(d)[:, None] * tau + np.arange(n)
    return x[indices].T

# ============================================================
# 2. 延迟τ估计（不变）
# ============================================================

def mutual_info_first_min(x, max_lag=50):
    """互信息第一极小值"""
    mi = []
    for lag in range(1, max_lag+1):
        mi.append(mutual_info_score(x[:-lag], x[lag:]))
    for i in range(2, len(mi)-1):
        if mi[i] < mi[i-1] and mi[i] < mi[i+1]:
            return i
    return max_lag//2

# ============================================================
# 3. 互信息块长度（新增，用于Bootstrap）
# ============================================================

def mutual_info_block_length(x, max_lag=200, bin_width_factor=0.3):
    """
    计算互信息第一极小值，作为块状Bootstrap的块长度。
    使用自适应分箱：箱宽 = bin_width_factor * std(x)
    """
    std_x = np.std(x)
    if std_x == 0:
        return 1

    bin_width = bin_width_factor * std_x
    n_bins = max(10, int((np.max(x) - np.min(x)) / bin_width))
    n_bins = min(n_bins, 100)  # 上限

    def discretize(a):
        bins = np.linspace(np.min(a), np.max(a), n_bins + 1)
        return np.digitize(a, bins) - 1

    mi = []
    for lag in range(1, max_lag+1):
        a, b = x[:-lag], x[lag:]
        a_d, b_d = discretize(a), discretize(b)
        # 联合分布
        joint = np.zeros((n_bins, n_bins))
        for i in range(len(a_d)):
            if 0 <= a_d[i] < n_bins and 0 <= b_d[i] < n_bins:
                joint[a_d[i], b_d[i]] += 1
        joint /= len(a_d)
        # 边缘分布
        pa = np.sum(joint, axis=1)
        pb = np.sum(joint, axis=0)
        # 互信息
        mi_val = 0
        for i in range(n_bins):
            for j in range(n_bins):
                if joint[i,j] > 0 and pa[i] > 0 and pb[j] > 0:
                    mi_val += joint[i,j] * np.log(joint[i,j] / (pa[i]*pb[j]))
        mi.append(mi_val)

    # 找第一极小值
    for i in range(2, len(mi)-1):
        if mi[i] < mi[i-1] and mi[i] < mi[i+1]:
            return max(1, i)

    # 备选：下降到初始值的0.3倍
    target = mi[0] * 0.3
    for i, m in enumerate(mi):
        if m < target:
            return max(1, i + 1)

    return max(1, max_lag // 10)

# ============================================================
# 4. 局部嵌入维数（不变）
# ============================================================

def local_embedding_dim(signal, tau, fs, window_sec=5, max_dim=30):
    """滑动窗口FNN估计局部嵌入维数"""
    T = len(signal)
    win_len = int(window_sec * fs)
    d_local = np.ones(T, dtype=int) * 2
    padding = (win_len // 2)
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
            n_pts = len(win) - (d-1)*tau
            if n_pts < d+1:
                break
            traj = reconstruct(win, tau, d)
            fnn = 0
            for j in range(traj.shape[0]):
                dist = np.linalg.norm(traj[j] - traj, axis=1)
                dist[j] = np.inf
                nn = np.argmin(dist)
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
    d_local = median_filter(d_local, size=11)
    return d_local

# ============================================================
# 5. 闭环因果强度Λ（不变）
# ============================================================

def compute_lambda(traj, dt, n_neighbors=None, ridge_alpha=0.01):
    """计算闭环因果强度Λ"""
    T, d = traj.shape
    if n_neighbors is None:
        n_neighbors = max(2*d, 30)
    n_neighbors = min(n_neighbors, T-1)
    traces = []
    tree = KDTree(traj)
    for i in range(T-1):
        dist, idx = tree.query(traj[i], k=n_neighbors+1)
        neighbors = idx[1:]
        # 过滤掉会导致越界的邻居
        valid_neighbors = neighbors[neighbors < T-1]
        if len(valid_neighbors) < d:
            continue
        X = traj[valid_neighbors] - traj[i]
        Y = traj[valid_neighbors+1] - traj[valid_neighbors]
        _, s, _ = np.linalg.svd(X, full_matrices=False)
        cond = s[0] / s[-1] if s[-1] > 0 else 1e16
        if cond > 1e6:
            XTX = X.T @ X
            reg = ridge_alpha * np.trace(XTX) / d * np.eye(d)
            J = np.linalg.inv(XTX + reg) @ X.T @ Y
            J = J.T
        else:
            J, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
            J = J.T
        traces.append(np.abs(np.trace(J)) / d)
    return np.mean(traces) if traces else 0.0

# ============================================================
# 6. IAAFT Surrogate（新增，替换简单洗牌）
# ============================================================

def iaaft_surrogate(x, n_iter=10):
    """
    迭代幅度调整傅里叶变换 (IAAFT)
    保持功率谱结构，破坏相位关系。

    参数:
    x: 一维时间序列
    n_iter: 迭代次数（默认10次）
    返回:
    surrogate: 与x长度相同的surrogate序列
    """
    n = len(x)

    # FFT of original
    fft_vals = np.fft.rfft(x)
    amplitudes = np.abs(fft_vals)

    # Random phases
    phases = np.random.uniform(0, 2*np.pi, len(amplitudes))
    phases[0] = 0  # DC component

    # Initial surrogate with random phases but original amplitudes
    new_fft = amplitudes * np.exp(1j * phases)
    surrogate = np.fft.irfft(new_fft, n=n)

    # Sort original for amplitude matching
    sorted_orig = np.sort(x)

    # Iterative refinement
    for _ in range(n_iter):
        # Rank-based amplitude adjustment
        rank = np.argsort(np.argsort(surrogate))
        surrogate = sorted_orig[rank]

        # Re-adjust phases to match original power spectrum
        fft_s = np.fft.rfft(surrogate)
        surrogate = np.fft.irfft(amplitudes * np.exp(1j * np.angle(fft_s)), n=n)

    # Final amplitude matching
    rank = np.argsort(np.argsort(surrogate))
    surrogate = sorted_orig[rank]

    return surrogate

# ============================================================
# 7. 噪声基准Λ_noise（修改：用IAAFT替换简单洗牌）
# ============================================================

def lambda_significance(signal, tau, d, dt, n_surrogates=100):
    """
    计算Λ_real, Λ_noise(95%分位数), 和p值。
    使用IAAFT生成surrogate数据（保持功率谱，破坏时间因果结构）。
    """
    traj = reconstruct(signal, tau, d)
    lam_real = compute_lambda(traj, dt)

    lam_surr = []
    for _ in range(n_surrogates):
        # IAAFT: 对每个维度分别处理
        x_surr = np.zeros_like(signal)
        x_surr = iaaft_surrogate(signal, n_iter=10)

        traj_surr = reconstruct(x_surr, tau, d)
        lam_surr.append(compute_lambda(traj_surr, dt))

    lam_noise = np.percentile(lam_surr, 95)
    p = (np.sum(np.array(lam_surr) >= lam_real) + 1) / (n_surrogates + 1)

    return lam_real, lam_noise, p, lam_surr

# ============================================================
# 8. 可访问自指距离Θ（不变）
# ============================================================

def compute_theta(traj, delta, c=2.0, time_window=5000, verify_steps=5):
    """计算可访问自指距离Θ(s)"""
    T, d = traj.shape
    window = min(time_window, T//10) if T > 10000 else time_window
    R_indices = set()
    for i in range(T):
        start = max(0, i - window)
        end = min(T, i + window + 1)
        for j in range(start, end):
            if i == j:
                continue
            if np.linalg.norm(traj[i] - traj[j]) < c * delta:
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

# ============================================================
# 9. 核密度估计 + Silverman带宽（新增）
# ============================================================

def kde_silverman(data):
    """
    各向同性高斯核密度估计。

    参数:
    data: (N, d) 状态空间轨迹
    返回:
    p: (N,) 每个点的概率密度估计
    h: float Silverman带宽
    """
    N, d = data.shape

    # Silverman带宽
    sigma = np.mean(np.std(data, axis=0))
    h = (4 / (d + 2)) ** (1 / (d + 4)) * sigma * N ** (-1 / (d + 4))

    # 高斯核密度估计
    p = np.zeros(N)
    for i in range(N):
        dists = np.linalg.norm(data - data[i], axis=1)
        p[i] = np.mean(np.exp(-0.5 * (dists / h) ** 2)) / ((2 * np.pi * h**2) ** (d/2))

    return p, h

def kde_silverman_fast(data, h=None):
    """
    快速版本：使用KDTree近似（大数据集）
    """
    N, d = data.shape

    if h is None:
        sigma = np.mean(np.std(data, axis=0))
        h = (4 / (d + 2)) ** (1 / (d + 4)) * sigma * N ** (-1 / (d + 4))

    tree = KDTree(data)
    p = np.zeros(N)
    for i in range(N):
        # 查询半径3h内的点
        idx = tree.query_ball_point(data[i], r=3*h)
        dists = np.linalg.norm(data[idx] - data[i], axis=1)
        p[i] = np.sum(np.exp(-0.5 * (dists / h) ** 2)) / (N * (2 * np.pi * h**2) ** (d/2))

    return p, h

# ============================================================
# 10. 局部线性回归估计∇U（新增）
# ============================================================

def gradient_u_local_linear(data, h=None, epsilon=1e-6, k_neighbors=None):
    """
    用局部线性回归估计张力势梯度∇U。

    参数:
    data: (N, d) 状态空间轨迹
    h: 核密度带宽（None则自动计算Silverman）
    epsilon: 岭回归正则化系数（相对于最大特征值的比例）
    k_neighbors: 近邻数（None则自动选择）
    返回:
    gradU: (N, d) 每个点的∇U估计
    p: (N,) 概率密度
    h_used: 实际使用的带宽
    """
    N, d = data.shape

    # 核密度估计
    p, h_used = kde_silverman_fast(data, h)

    # 对数密度
    log_p = np.log(p + 1e-12)

    # 确定近邻数
    if k_neighbors is None:
        k_neighbors = min(max(2*d, 30), N-1)

    tree = KDTree(data)
    gradU = np.zeros_like(data)

    for i in range(N):
        dist, idx = tree.query(data[i], k=k_neighbors+1)
        neighbors = idx[1:]  # 排除自身

        # 加权最小二乘
        X = data[neighbors] - data[i]  # (k, d)
        y = log_p[neighbors] - log_p[i]  # (k,)

        # 权重（高斯核）
        weights = np.exp(-0.5 * (dist[1:] / h_used) ** 2)
        W = np.diag(weights)

        # 加权最小二乘
        Xw = X.T @ W  # (d, k)
        XWX = Xw @ X  # (d, d)
        XWy = Xw @ y  # (d,)

        # 岭回归正则化
        lambda_max = np.max(np.linalg.eigvalsh(XWX))
        reg = epsilon * lambda_max * np.eye(d)

        try:
            grad = np.linalg.solve(XWX + reg, XWy)
            gradU[i] = -grad  # U = -log p, so ∇U = -∇log p
        except np.linalg.LinAlgError:
            gradU[i] = np.zeros(d)

    return gradU, p, h_used

# ============================================================
# 11. 块状Bootstrap（新增）
# ============================================================

def block_bootstrap(sigma_sequence, block_length, n_bootstrap=1000):
    """
    块状Bootstrap：对符号序列进行有放回块重抽样。

    参数:
    sigma_sequence: (T,) 方向规则符号序列（+1或-1）
    block_length: int 块长度
    n_bootstrap: int Bootstrap次数
    返回:
    R_star: (n_bootstrap,) Bootstrap一致性比例分布
    ci_lower, ci_upper: 95%置信区间
    """
    T = len(sigma_sequence)
    n_blocks = T // block_length

    if n_blocks < 2:
        # 块长度太大，退化为简单Bootstrap
        R_star = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(sigma_sequence, size=T, replace=True)
            R_star.append(np.mean(sample > 0))
        R_star = np.array(R_star)
        ci_lower, ci_upper = np.percentile(R_star, [2.5, 97.5])
        return R_star, ci_lower, ci_upper

    # 分块
    blocks = []
    for i in range(n_blocks):
        start = i * block_length
        end = min(start + block_length, T)
        blocks.append(sigma_sequence[start:end])

    R_star = []
    for _ in range(n_bootstrap):
        # 有放回抽取块
        sampled_blocks = []
        for _ in range(n_blocks):
            idx = np.random.randint(0, len(blocks))
            sampled_blocks.append(blocks[idx])

        # 拼接
        sample = np.concatenate(sampled_blocks)[:T]
        R_star.append(np.mean(sample > 0))

    R_star = np.array(R_star)
    ci_lower, ci_upper = np.percentile(R_star, [2.5, 97.5])

    return R_star, ci_lower, ci_upper

# ============================================================
# 12. 效应量分层（新增）
# ============================================================

def effect_size_tier(R):
    """
    方向规则一致性比例的效应量分层。

    参数:
    R: float 一致性比例
    返回:
    tier: str 分层标签
    description: str 描述
    """
    if R > 0.75:
        return "strong_support", "强支持（ITT核心预测）"
    elif R >= 0.60:
        return "weak_support", "弱支持（自反力存在，效应量低于预期）"
    elif R >= 0.50:
        return "marginal", "边缘（可能自反力微弱，或数据质量不足）"
    else:
        return "falsified", "明确证伪"

# ============================================================
# 13. 高维降级判断（新增）
# ============================================================

def check_high_dim_degradation(N, d, min_samples_per_dim=100):
    """
    检查主轨道密度估计是否应降级为简化轨道。

    参数:
    N: int 轨迹点数
    d: int 嵌入维数
    min_samples_per_dim: int 每维最小样本数
    返回:
    should_degrade: bool 是否降级
    N_eff: float 有效样本量
    N_min: float 最小要求
    """
    N_eff = N / (2 ** d)
    N_min = min_samples_per_dim * d

    should_degrade = N_eff < N_min

    return should_degrade, N_eff, N_min

# ============================================================
# 14. 修复：标准假近邻法（FNN）
# ============================================================

def false_nearest_neighbor(x, tau, max_dim=10, rtol=10, atol=1e-8):
    """
    标准假近邻法 (False Nearest Neighbors) 估计嵌入维数。
    对于每个点，在 d 维空间中找到最近邻（排除自身），
    检查在 d+1 维中该邻居的距离是否显著增大。
    
    参数:
    x: 一维时间序列
    tau: 延迟
    max_dim: 最大嵌入维数
    rtol: 距离比率阈值（典型值 10~15）
    atol: 绝对容差，避免分母过小
    
    返回:
    dim: 估计的嵌入维数
    """
    N = len(x)
    for dim in range(1, max_dim+1):
        # 重构 dim 维和 dim+1 维轨迹
        n_pts = N - (dim-1)*tau
        if n_pts < 2:
            return dim
        traj_dim = reconstruct(x, tau, dim)
        if dim == max_dim:
            return dim
        traj_dim1 = reconstruct(x, tau, dim+1)
        # 取相同点数
        n = min(len(traj_dim), len(traj_dim1))
        traj_dim = traj_dim[:n]
        traj_dim1 = traj_dim1[:n]
        
        false_count = 0
        for i in range(n):
            # 在 dim 维中找最近邻（排除自身）
            dists = np.linalg.norm(traj_dim - traj_dim[i], axis=1)
            dists[i] = np.inf
            if np.all(dists == np.inf):
                continue
            nearest = np.argmin(dists)
            R_d = dists[nearest]
            if R_d < atol:
                continue
            # 在 dim+1 维中计算与同一邻居的距离
            R_d1 = np.linalg.norm(traj_dim1[i] - traj_dim1[nearest])
            if R_d1 / R_d > rtol:
                false_count += 1
        fnn_ratio = false_count / n
        if fnn_ratio < 0.01:  # 假近邻比例低于1%
            return dim
    return max_dim
