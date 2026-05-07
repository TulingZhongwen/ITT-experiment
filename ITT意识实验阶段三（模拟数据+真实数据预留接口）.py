"""
ITT Stage 3: Self-force direction rule (Prediction 3) – Final integrated version
Includes: robust gradient estimation (local linear regression + smoothing),
          Savitzky-Golay velocity, positive/negative mock data, permutation test.
This file is independent of Stage 1/2 and relies on itt_core.py only for basic helpers.
Mock data is for algorithm validation only; real data interface must be implemented by user.

Author: Tuling Zhongwen (图灵中文)
Version: 1.0
"""

import numpy as np
from scipy.spatial import KDTree
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# ---------- 1. Mock data generation ----------
def generate_mock_data(T=5000, d=10, seed=42, noise_level=0.05):
    """
    Generate artificial state space trajectory obeying the direction rule.
    noise_level: added noise to the ideal velocity direction.
    Returns: traj, Delta, gradU, v, dot
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
    """Generate pure Gaussian white noise as negative control."""
    traj = np.random.randn(T, d)
    Delta = np.random.randn(T, d)
    Delta /= (np.linalg.norm(Delta, axis=1, keepdims=True) + 1e-12)
    gradU = np.random.randn(T, d)
    v = np.diff(traj, axis=0)
    return traj, Delta, gradU, v

# ---------- 2. Core functions (robust implementations) ----------
def estimate_gradU_llr(traj, k=100, sigma=1.0):
    """
    Estimate gradient of potential U using local linear regression.
    Steps: density estimation -> negative log -> Gaussian smoothing -> local linear fit.
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
    Compute unit deviation vector Δ̅ = (Π(s)-s)/||Π(s)-s||
    Uses time‑windowed search for approximate periodicity to reduce complexity.
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
    """Compute velocity using Savitzky–Golay filter (smoothing derivative)."""
    return savgol_filter(traj, window_length=window_length, polyorder=polyorder, deriv=1, axis=0)

def test_direction_rule(traj, Delta_unit, gradU, v):
    """
    Compute consistency of the direction rule.
    Returns: consistency (0~1), align_neg, align_pos, dot array.
    """
    dot = np.sum(Delta_unit * gradU, axis=1)
    v_unit = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    dot_aligned = dot[:-1]
    mask_neg = dot_aligned < 0
    mask_pos = dot_aligned > 0
    if np.sum(mask_neg) == 0 or np.sum(mask_pos) == 0:
        return 0.5, 0.5, 0.5, dot
    align_neg = np.mean(
        np.sum(v_unit[mask_neg] * (-gradU[:-1][mask_neg]), axis=1) > 0
    )
    align_pos = np.mean(
        np.sum(v_unit[mask_pos] * (gradU[:-1][mask_pos]), axis=1) > 0
    )
    consistency = (align_neg + align_pos) / 2
    return consistency, align_neg, align_pos, dot

def permutation_test(traj, Delta_unit, gradU, v, n_perm=1000):
    """Permutation test by shuffling velocity time order to obtain null distribution and p-value."""
    real_cons, _, _, _ = test_direction_rule(traj, Delta_unit, gradU, v)
    null_cons = []
    T = len(v)
    for _ in range(n_perm):
        v_shuff = v[np.random.permutation(T)]
        cons, _, _, _ = test_direction_rule(traj, Delta_unit, gradU, v_shuff)
        null_cons.append(cons)
    p = np.mean(np.array(null_cons) >= real_cons)
    return real_cons, null_cons, p

# ---------- 3. Mock test (positive + negative) ----------
def test_on_mock():
    print("\n===== Stage 3: Mock data test (algorithm validation) =====")

    # Positive data (following the rule)
    traj, _, _, v, _ = generate_mock_data(T=3000, d=10, noise_level=0.05)
    gradU_est = estimate_gradU_llr(traj, k=100, sigma=1.0)
    delta = np.percentile(np.linalg.norm(np.diff(traj, axis=0), axis=1), 5)
    Delta_unit = compute_delta_unit(traj, delta, time_window=2000)
    v_smooth = compute_velocity(traj, window_length=15, polyorder=3)
    cons_pos, align_neg, align_pos, _ = test_direction_rule(traj, Delta_unit, gradU_est, v_smooth)
    _, _, p_pos = permutation_test(traj, Delta_unit, gradU_est, v_smooth, n_perm=500)

    print(f"Positive data (rule‑following):")
    print(f"  Consistency = {cons_pos:.3f} (target > 0.75)")
    print(f"  Alignment when Δ̅·∇U < 0 = {align_neg:.3f}")
    print(f"  Alignment when Δ̅·∇U > 0 = {align_pos:.3f}")
    print(f"  Permutation test p = {p_pos:.4f} (expected < 0.05)")

    # Negative data (pure noise)
    traj_n, _, _, v_n = generate_noise_data(T=3000, d=10)
    gradU_est_n = estimate_gradU_llr(traj_n, k=100, sigma=1.0)
    delta_n = np.percentile(np.linalg.norm(np.diff(traj_n, axis=0), axis=1), 5)
    Delta_unit_n = compute_delta_unit(traj_n, delta_n, time_window=2000)
    v_smooth_n = compute_velocity(traj_n, window_length=15, polyorder=3)
    cons_neg, align_neg_n, align_pos_n, _ = test_direction_rule(traj_n, Delta_unit_n, gradU_est_n, v_smooth_n)
    _, _, p_neg = permutation_test(traj_n, Delta_unit_n, gradU_est_n, v_smooth_n, n_perm=500)

    print(f"\nNegative data (pure noise):")
    print(f"  Consistency = {cons_neg:.3f} (expected ≈ 0.5)")
    print(f"  Permutation test p = {p_neg:.4f} (expected > 0.05)")

    # Plot null distributions
    _, null_pos, _ = permutation_test(traj, Delta_unit, gradU_est, v_smooth, n_perm=500)
    _, null_neg, _ = permutation_test(traj_n, Delta_unit_n, gradU_est_n, v_smooth_n, n_perm=500)
    plt.figure(figsize=(8,6))
    plt.hist(null_pos, bins=30, alpha=0.6, label='Positive data null', color='blue')
    plt.hist(null_neg, bins=30, alpha=0.6, label='Noise data null', color='gray')
    plt.axvline(x=cons_pos, color='blue', linestyle='--', label=f'Positive obs = {cons_pos:.3f}')
    plt.axvline(x=cons_neg, color='red', linestyle='--', label=f'Noise obs = {cons_neg:.3f}')
    plt.xlabel('Consistency')
    plt.ylabel('Frequency')
    plt.title('Direction rule permutation test (algorithm validation only)')
    plt.legend()
    plt.savefig('stage3_permutation_test.png')
    plt.show()

# ---------- 4. Real data interface (placeholder) ----------
def prepare_real_data(subject_id, data_path):
    """
    User must implement this function according to their dataset format.
    Should return (traj, Delta_unit, gradU, v)
    """
    raise NotImplementedError("Please implement prepare_real_data for your dataset.")

def test_on_real_data():
    try:
        traj, Delta_unit, gradU, v = prepare_real_data('sub-01', '/path/to/data')
        cons, _, _, _ = test_direction_rule(traj, Delta_unit, gradU, v)
        _, _, p = permutation_test(traj, Delta_unit, gradU, v, n_perm=1000)
        print(f"Real data consistency = {cons:.3f}, permutation p = {p:.4f}")
    except NotImplementedError as e:
        print(e)

if __name__ == "__main__":
    test_on_mock()
    # test_on_real_data()   # Uncomment after implementing prepare_real_data
