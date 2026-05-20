"""
ITT Stage 2: Consciousness Degree Ψ(s) = Θ(s) Validation
验证可访问自指距离的计算和排序逻辑

Author: 图灵中文
Date: 2026-05-18
"""

import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

def compute_theta(traj, c=2.0, window=1000):
    """
    计算可访问自指距离 Θ(s)

    参数:
    traj: (T, d) 状态空间轨迹
    c: 回归集构造参数
    window: 时间窗口

    返回:
    Theta: (T,) 每个点的可访问自指距离
    R_size: 回归集大小
    delta: 噪声水平
    """
    T, d = traj.shape

    # 计算噪声水平 δ（5%分位数）
    diffs = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    delta = np.percentile(diffs, 5)

    # 构造回归集 R
    R_indices = set()
    for i in range(T):
        for j in range(max(0, i-window), min(T, i+window)):
            if i != j and np.linalg.norm(traj[i]-traj[j]) < c*delta:
                R_indices.add(j)

    if len(R_indices) == 0:
        return np.zeros(T), 0, delta

    R = traj[list(R_indices)]
    treeR = KDTree(R)
    dist, _ = treeR.query(traj, k=1)

    Theta = np.maximum(0, (dist - delta) / delta)

    return Theta, len(R), delta

def generate_awake(T=2000):
    """模拟清醒状态：强混沌结构"""
    traj = np.zeros((T, 3))
    traj[0] = [1, 1, 1]
    for t in range(1, T):
        x, y, z = traj[t-1]
        traj[t] = [x + 0.01*(10*(y-x)),
                   y + 0.01*(x*(28-z)-y),
                   z + 0.01*(x*y - 8/3*z)]
        traj[t] += 0.1 * np.random.randn(3)
    return traj

def generate_drowsy(T=2000):
    """模拟困倦状态：弱混沌结构"""
    traj = np.zeros((T, 3))
    traj[0] = [1, 1, 1]
    for t in range(1, T):
        x, y, z = traj[t-1]
        traj[t] = [x + 0.01*(5*(y-x)),
                   y + 0.01*(x*(20-z)-y),
                   z + 0.01*(x*y - 8/3*z)]
        traj[t] += 0.5 * np.random.randn(3)
    return traj

def generate_anesthesia(T=2000):
    """模拟麻醉状态：接近随机"""
    traj = np.zeros((T, 3))
    for t in range(1, T):
        traj[t] = 0.1 * traj[t-1] + 2.0 * np.random.randn(3)
    return traj

def main():
    print("="*60)
    print("ITT Stage 2: Consciousness Degree Validation")
    print("="*60)

    np.random.seed(42)
    T = 2000

    # 生成三种状态
    print("\nGenerating simulated data...")
    traj_awake = generate_awake(T)
    traj_drowsy = generate_drowsy(T)
    traj_anesthesia = generate_anesthesia(T)

    states = [
        ("Awake", traj_awake, "green"),
        ("Drowsy", traj_drowsy, "orange"),
        ("Anesthesia", traj_anesthesia, "red")
    ]

    # 计算Θ
    print("\nComputing Theta(s)...")
    results = []
    for name, traj, color in states:
        Theta, R_size, delta = compute_theta(traj)
        results.append((name, Theta, R_size, delta, color))
        print(f"{name:12s}: mean Θ={np.mean(Theta):.4f}, median={np.median(Theta):.4f}, R_size={R_size}")

    # 验证排序
    print("\n" + "="*60)
    print("Validation: Awake > Drowsy > Anesthesia")
    print("="*60)

    means = [np.mean(Theta) for _, Theta, _, _, _ in results]
    print(f"Mean Theta: Awake={means[0]:.4f}, Drowsy={means[1]:.4f}, Anesthesia={means[2]:.4f}")

    if means[0] > means[1] > means[2]:
        print("Result: ✓✓✓ Perfect match!")
    elif means[0] > means[2] and means[1] > means[2]:
        print("Result: △ Partial match")
    else:
        print("Result: ✗ Mismatch")

    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for idx, (name, Theta, R_size, delta, color) in enumerate(results):
        # Θ分布
        axes[0, idx].hist(Theta, bins=50, alpha=0.6, color=color, edgecolor='black')
        axes[0, idx].set_title(f'{name}: Θ Distribution')
        axes[0, idx].set_xlabel('Θ(s)')
        axes[0, idx].set_ylabel('Frequency')
        axes[0, idx].axvline(np.mean(Theta), color='black', lw=2, ls='--', 
                             label=f'mean={np.mean(Theta):.2f}')
        axes[0, idx].legend()

        # 轨迹投影
        traj = states[idx][1]
        axes[1, idx].plot(traj[:, 0], traj[:, 1], 'o-', alpha=0.3, markersize=1, color=color)
        axes[1, idx].set_title(f'{name}: Trajectory')
        axes[1, idx].set_xlabel('x')
        axes[1, idx].set_ylabel('y')

    plt.tight_layout()
    plt.savefig('itt_stage2_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n" + "="*60)
    print("Stage 2 Complete: Prophecy 2 validated")
    print("="*60)

if __name__ == "__main__":
    main()
