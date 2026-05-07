"""
ITT Stage 3: Self-force direction rule (Prediction 3) 
文件名: ITT意识量化阶段三（方向规则）.py
版本: 3.0

功能:
- 独立实现 ITT 预言三的验证 (无需 itt_core.py)
- 包含稳健的梯度估计 (局部线性回归) 和速度计算 (Savitzky-Golay)
- 包含正面对照 (符合规则) 和负面对照 (纯噪声)
- 执行置换检验并输出标准图表
- 预留真实数据接口 (prepare_real_data)，用户可扩展

Author: Tuling Zhongwen (图灵中文)
__version__: 3.0
"""

import numpy as np
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, List

__version__ = "3.0.1"

# ========== 1. 模拟数据生成 ==========
def generate_mock_data(T: int = 3000, d: int = 10, seed: int = 42, noise_level: float = 0.05) -> Tuple[np.ndarray, ...]:
    """生成符合方向规则的人造数据（正面对照）"""
    np.random.seed(seed)
    traj = np.cumsum(np.random.randn(T, d) * 0.01, axis=0)
    Delta = np.random.randn(T, d)
    Delta = Delta / (np.linalg.norm(Delta, axis=1, keepdims=True) + 1e-12)
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

def generate_noise_data(T: int = 3000, d: int = 10) -> Tuple[np.ndarray, ...]:
    """生成纯随机噪声数据（负面对照）"""
    traj = np.random.randn(T, d)
    Delta = np.random.randn(T, d)
    Delta = Delta / (np.linalg.norm(Delta, axis=1, keepdims=True) + 1e-12)
    gradU = np.random.randn(T, d)
    v = np.diff(traj, axis=0)
    return traj, Delta, gradU, v

# ========== 2. 核心算法（独立实现） ==========
def estimate_gradU_llr(traj: np.ndarray, k: int = 100, sigma: float = 1.0) -> np.ndarray:
    """局部线性回归估计梯度 ∇U"""
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
        if len(neighbors) >= d and X.shape[0] >= X.shape[1]:
            try:
                lr = LinearRegression(fit_intercept=False)
                lr.fit(X, y)
                gradU[i] = lr.coef_
            except:
                gradU[i] = np.zeros(d)
        else:
            gradU[i] = np.zeros(d)
    return gradU

def compute_delta_unit(traj: np.ndarray, delta: float, time_window: int = 5000) -> np.ndarray:
    """计算单位偏差向量 Δ̅ (自反性算子)"""
    T, d = traj.shape
    window = min(time_window, T//10)
    R_indices = set()
    for i in range(T):
        start = max(0, i - window)
        end = min(T, i + window + 1)
        for j in range(start, end):
            if i != j and np.linalg.norm(traj[i] - traj[j]) < delta:
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

def compute_velocity(traj: np.ndarray, window_length: int = 15, polyorder: int = 3) -> np.ndarray:
    """Savitzky-Golay 平滑速度计算 (F_self) 增强鲁棒性"""
    n = len(traj)
    if n < window_length:
        window_length = n if n % 2 == 1 else n - 1
    if window_length < polyorder + 1:
        return np.diff(traj, axis=0)
    return savgol_filter(traj, window_length, polyorder, deriv=1, axis=0)

def test_direction_rule(traj: np.ndarray, Delta_unit: np.ndarray, gradU: np.ndarray, v: np.ndarray) -> Tuple[float, float, float, np.ndarray]:
    """计算一致性比例 (验证预言三)"""
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

def permutation_test(traj: np.ndarray, Delta_unit: np.ndarray, gradU: np.ndarray, v: np.ndarray, n_perm: int = 1000) -> Tuple[float, List[float], float]:
    """置换检验 (统计显著性)"""
    real_cons, _, _, _ = test_direction_rule(traj, Delta_unit, gradU, v)
    null_cons = []
    T = len(v)
    for _ in range(n_perm):
        v_shuff = v[np.random.permutation(T)]
        cons, _, _, _ = test_direction_rule(traj, Delta_unit, gradU, v_shuff)
        null_cons.append(cons)
    p = np.mean(np.array(null_cons) >= real_cons)
    return real_cons, null_cons, p

# ========== 3. 真实数据接口（预留） ==========
def prepare_real_data(subject_id: str, data_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    用户根据实际数据集（如 OpenNeuro ds002785）实现此函数。
    应返回 (traj, Delta_unit, gradU, v)
    """
    # 示例：path = Path(data_path)
    # 用户需在此处加载真实 EEG/fMRI 数据并预处理
    raise NotImplementedError("真实数据接口未实现。请根据你的数据集格式实现此函数。")

def test_on_real_data():
    """调用真实数据进行检验"""
    try:
        traj, Delta_unit, gradU, v = prepare_real_data('sub-01', Path('/path/to/dataset'))
        cons, _, _, _ = test_direction_rule(traj, Delta_unit, gradU, v)
        _, _, p = permutation_test(traj, Delta_unit, gradU, v, n_perm=1000)
        print(f"真实数据一致性 = {cons:.3f}, 置换检验 p = {p:.4f}")
    except NotImplementedError as e:
        print(e)

# ========== 4. 主程序：模拟测试 ==========
def main():
    print(f"\n🚀 ITT 阶段三：自反力方向规则验证 (版本 {__version__})")
    print("本脚本可独立运行，无需依赖 itt_core.py")
    print("模拟测试将生成正面对照与负面对照结果。\n")

    T, d = 3000, 10

    # --- 正面数据 ---
    print("[1/3] 正面对照（符合 ITT 规则的合成数据）...")
    traj, _, _, v, _ = generate_mock_data(T=T, d=d, noise_level=0.05)
    gradU_est = estimate_gradU_llr(traj, k=100, sigma=1.0)
    delta_noise = np.percentile(np.linalg.norm(np.diff(traj, axis=0), axis=1), 5)
    Delta_unit = compute_delta_unit(traj, delta=delta_noise)
    v_smooth = compute_velocity(traj)
    cons_pos, _, _, _ = test_direction_rule(traj, Delta_unit, gradU_est, v_smooth)
    _, null_pos, p_pos = permutation_test(traj, Delta_unit, gradU_est, v_smooth, n_perm=500)
    print(f"   一致性 = {cons_pos:.3f} (目标 >0.75), p = {p_pos:.4f}")

    # --- 负面数据 ---
    print("[2/3] 负面对照（纯随机噪声）...")
    traj_n, _, _, v_n = generate_noise_data(T=T, d=d)
    gradU_n = estimate_gradU_llr(traj_n, k=100, sigma=1.0)
    delta_n = np.percentile(np.linalg.norm(np.diff(traj_n, axis=0), axis=1), 5)
    Delta_unit_n = compute_delta_unit(traj_n, delta=delta_n)
    v_smooth_n = compute_velocity(traj_n)
    cons_neg, _, _, _ = test_direction_rule(traj_n, Delta_unit_n, gradU_n, v_smooth_n)
    _, null_neg, p_neg = permutation_test(traj_n, Delta_unit_n, gradU_n, v_smooth_n, n_perm=500)
    print(f"   一致性 = {cons_neg:.3f} (预期 ≈0.5), p = {p_neg:.4f}")

    # --- 绘图 ---
    print("[3/3] 生成图表...")
    plt.figure(figsize=(10, 6))
    plt.hist(null_pos, bins=25, alpha=0.7, color='skyblue', label='正面对照零分布')
    plt.hist(null_neg, bins=25, alpha=0.7, color='lightcoral', label='负面对照零分布')
    plt.axvline(cons_pos, color='blue', linestyle='dashed', linewidth=2, label=f'正面对照观测 = {cons_pos:.2f}')
    plt.axvline(cons_neg, color='red', linestyle='dashed', linewidth=2, label=f'负面对照观测 = {cons_neg:.2f}')
    plt.title('ITT Stage 3: Permutation Test of Direction Rule')
    plt.xlabel('一致性比例')
    plt.ylabel('频次')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('stage3_permutation_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\n模拟测试完成。如需真实数据检验，请实现 prepare_real_data 函数并调用 test_on_real_data()。")

if __name__ == "__main__":
    main()
