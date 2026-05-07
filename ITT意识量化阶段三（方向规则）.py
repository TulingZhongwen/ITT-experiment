"""
ITT Stage 3: Self-force direction rule (Prediction 3)
Integrated version with robust gradient estimation, Savitzky-Golay velocity,
positive/negative mock data tests, permutation test, and real data placeholder.

Author: Tuling Zhongwen (图灵中文)
Dependencies: numpy, scipy, scikit-learn, matplotlib
"""

import numpy as np
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 导入 ITT 核心模块（请确保仓库中有 ITT核心模块改进.py）
try:
    from ITT核心模块改进 import reconstruct, mutual_info_first_min, false_nearest_neighbor
except ImportError:
    print("警告: 无法导入 ITT核心模块改进，将使用本地简化版本（仅用于测试）。")
    # 如果找不到核心模块，提供简化的重构函数（仅用于模拟数据）
    def reconstruct(x, tau, d):
        n = len(x) - (d-1)*tau
        if n <= 0:
            raise ValueError
        indices = np.arange(d)[:, None] * tau + np.arange(n)
        return x[indices].T

# ---------- 1. 模拟数据生成（正面对照）----------
def generate_mock_data(T=5000, d=10, seed=42, noise_level=0.05):
    """
    生成符合方向规则的人造状态空间轨迹。
    noise_level: 速度方向添加的噪声强度。
    返回: traj, Delta, gradU, v, dot
    """
    np.random.seed(seed)
    traj = np.cumsum(np.random.randn(T, d) * 0.01, axis=0)
    Delta = np.random.randn(T, d)
    Delta /= (np.linalg.norm(Delta, axis=1, keepdims=True) + 1e-12)
    gradU = np.random.randn(T, d)
    dot = np.sum(Delta * gradU, axis=1)
    v_ideal = np.zeros((T-1, d))
    for i in range(T-1):
        if dot[i] < 0:
            v_ideal[i] = -gradU[i] / (np.linalg.norm(gradU[i]) + 1e-12)
        else:
            v_ideal[i] = gradU[i] / (np.linalg.norm(gradU[i]) + 1e-12)
    v = v_ideal + noise_level * np.random.randn(T-1, d)
    return traj, Delta, gradU, v, dot

def generate_noise_data(T=5000, d=10):
    """生成纯随机高斯噪声数据（负面对照）"""
    traj = np.random.randn(T, d)
    Delta = np.random.randn(T, d)
    Delta /= (np.linalg.norm(Delta, axis=1, keepdims=True) + 1e-12)
    gradU = np.random.randn(T, d)
    v = np.diff(traj, axis=0)
    return traj, Delta, gradU, v

# ---------- 2. 核心计算（鲁棒实现）----------
def estimate_gradU_llr(traj, k=100, sigma=1.0):
    """
    使用局部线性回归估计势能梯度 ∇U。
    步骤：密度估计（k近邻体积）-> 负对数 -> 高斯平滑 -> 局部拟合。
    """
    T, d = traj.shape
    tree = KDTree(traj)
    densities = np.zeros(T)
    for i in range(T):
        dist, _ = tree.query(traj[i], k+1)
        r = dist[-1]
        densities[i] = (r ** d) + 1e-12
    U = -np.log(densities)
    U_smooth = gaussian_filter1d(U, sigma=sigma)
    gradU = np.zeros_like(traj)
    for i in range(T):
        dist, idx = tree.query(traj[i], k+1)
        neighbors = idx[1:]
        X = traj[neighbors] - traj[i]
        y = U_smooth[neighbors] - U_smooth[i]
        if len(neighbors) >= d:
            lr = LinearRegression(fit_intercept=False)
            lr.fit(X, y)
            gradU[i] = lr.coef_
        else:
            gradU[i] = np.zeros(d)
    return gradU

def compute_delta_unit(traj, delta, time_window=5000):
    """
    计算单位偏差向量 Δ̅ = (Π(s)-s)/||Π(s)-s||
    使用时间窗口搜索近似周期点。
    """
    T, d = traj.shape
    window = min(time_window, T//10)
    R_indices = set()
    for i in range(T):
        start = max(0, i - window)
        end = min(T, i + window + 1)
        for j in range(start, end):
            if i == j:
                continue
            if np.linalg.norm(traj[i] - traj[j]) < delta:
                R_indices.add(i)
                R_indices.add(j)
    if len(R_indices) == 0:
        tree = KDTree(traj)
        for i in range(T):
            dist, idx = tree.query(traj[i], k=2)
            R_indices.add(idx[1])
    R = traj[list(R_indices)]
    treeR = KDTree(R)
    distances, indices = treeR.query(traj, k=1)
    Pi = R[indices]
    Delta_vec = Pi - traj
    norm = np.linalg.norm(Delta_vec, axis=1, keepdims=True)
    Delta_unit = Delta_vec / (norm + 1e-12)
    return Delta_unit

def compute_velocity(traj, window_length=15, polyorder=3):
    """使用 Savitzky-Golay 滤波器计算平滑速度（一阶导数）"""
    return savgol_filter(traj, window_length=window_length, polyorder=polyorder, deriv=1, axis=0)

def test_direction_rule(traj, Delta_unit, gradU, v):
    """
    计算方向规则的一致性比例。
    返回: consistency (0~1), align_neg, align_pos, dot
    """
    dot = np.sum(Delta_unit * gradU, axis=1)
    v_unit = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    dot_aligned = dot[:-1]
    mask_neg = dot_aligned < 0
    mask_pos = dot_aligned > 0
    if np.sum(mask_neg) == 0 or np.sum(mask_pos) == 0:
        return 0.5, 0.5, 0.5, dot
    align_neg = np.mean(np.sum(v_unit[mask_neg] * (-gradU[:-1][mask_neg]), axis=1) > 0)
    align_pos = np.mean(np.sum(v_unit[mask_pos] * (gradU[:-1][mask_pos]), axis=1) > 0)
    consistency = (align_neg + align_pos) / 2
    return consistency, align_neg, align_pos, dot

def permutation_test(traj, Delta_unit, gradU, v, n_perm=1000):
    """
    置换检验：打乱速度的时间顺序，得到零分布和 p 值。
    """
    real_cons, _, _, _ = test_direction_rule(traj, Delta_unit, gradU, v)
    null_cons = []
    T = len(v)
    for _ in range(n_perm):
        v_shuff = v[np.random.permutation(T)]
        cons, _, _, _ = test_direction_rule(traj, Delta_unit, gradU, v_shuff)
        null_cons.append(cons)
    p = np.mean(np.array(null_cons) >= real_cons)
    return real_cons, null_cons, p

# ---------- 3. 模拟测试（正面对照 + 负面对照）----------
def test_on_mock():
    print("\n===== ITT 阶段三：模拟数据测试（算法验证）=====")

    # 正面对照
    traj, _, _, v, _ = generate_mock_data(T=3000, d=10, noise_level=0.05)
    gradU_est = estimate_gradU_llr(traj, k=100, sigma=1.0)
    delta = np.percentile(np.linalg.norm(np.diff(traj, axis=0), axis=1), 5)
    Delta_unit = compute_delta_unit(traj, delta, time_window=2000)
    v_smooth = compute_velocity(traj, window_length=15, polyorder=3)
    cons_pos, align_neg, align_pos, _ = test_direction_rule(traj, Delta_unit, gradU_est, v_smooth)
    _, null_pos, p_pos = permutation_test(traj, Delta_unit, gradU_est, v_smooth, n_perm=500)

    print("正面对照（符合规则的合成数据）:")
    print(f"  一致性 = {cons_pos:.3f} (目标 > 0.75)")
    print(f"  Δ̅·∇U < 0 时对齐比例 = {align_neg:.3f}")
    print(f"  Δ̅·∇U > 0 时对齐比例 = {align_pos:.3f}")
    print(f"  置换检验 p = {p_pos:.4f} (期望 < 0.05)")

    # 负面对照
    traj_n, _, _, v_n = generate_noise_data(T=3000, d=10)
    gradU_est_n = estimate_gradU_llr(traj_n, k=100, sigma=1.0)
    delta_n = np.percentile(np.linalg.norm(np.diff(traj_n, axis=0), axis=1), 5)
    Delta_unit_n = compute_delta_unit(traj_n, delta_n, time_window=2000)
    v_smooth_n = compute_velocity(traj_n, window_length=15, polyorder=3)
    cons_neg, align_neg_n, align_pos_n, _ = test_direction_rule(traj_n, Delta_unit_n, gradU_est_n, v_smooth_n)
    _, null_neg, p_neg = permutation_test(traj_n, Delta_unit_n, gradU_est_n, v_smooth_n, n_perm=500)

    print("\n负面对照（纯随机噪声）:")
    print(f"  一致性 = {cons_neg:.3f} (预期 ≈ 0.5)")
    print(f"  置换检验 p = {p_neg:.4f} (预期 > 0.05)")

    # 绘制零分布对比图
    plt.figure(figsize=(8,6))
    plt.hist(null_pos, bins=30, alpha=0.6, label='正面对照零分布', color='blue')
    plt.hist(null_neg, bins=30, alpha=0.6, label='负面对照零分布', color='gray')
    plt.axvline(x=cons_pos, color='blue', linestyle='--', label=f'正面对照观测值 = {cons_pos:.3f}')
    plt.axvline(x=cons_neg, color='red', linestyle='--', label=f'负面对照观测值 = {cons_neg:.3f}')
    plt.xlabel('一致性比例')
    plt.ylabel('频次')
    plt.title('方向规则置换检验（算法验证）')
    plt.legend()
    plt.savefig('stage3_permutation_test.png')
    plt.show()

# ---------- 4. 真实数据接口（预留）----------
def prepare_real_data(subject_id, data_path):
    """
    用户根据实际数据集（如 OpenNeuro ds002785）实现此函数。
    应返回 (traj, Delta_unit, gradU, v)
    """
    raise NotImplementedError("真实数据接口未实现，请根据数据集格式自行实现 prepare_real_data。")

def test_on_real_data():
    try:
        traj, Delta_unit, gradU, v = prepare_real_data('sub-01', '/path/to/dataset')
        cons, _, _, _ = test_direction_rule(traj, Delta_unit, gradU, v)
        _, _, p = permutation_test(traj, Delta_unit, gradU, v, n_perm=1000)
        print(f"真实数据一致性 = {cons:.3f}, 置换检验 p = {p:.4f}")
    except NotImplementedError as e:
        print(e)

if __name__ == "__main__":
    test_on_mock()
    # test_on_real_data()   # 取消注释并实现 prepare_real_data 后使用
