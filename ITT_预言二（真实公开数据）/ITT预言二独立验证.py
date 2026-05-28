"""
ITT Prophecy 2: PRIOS Dataset - Awake vs Propofol Anesthesia
独立复现代码，自动下载数据，个体配对分析
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import resample, butter, filtfilt, detrend
from scipy.spatial import KDTree
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ========== 1. 检查/下载 PRIOS 数据 ==========
def check_prios_data(data_dir="./ds004370"):
    """检查 PRIOS 数据是否存在，不存在则提示下载"""
    if os.path.exists(data_dir):
        print(f"数据目录已存在: {data_dir}")
        return True
    else:
        print("\n" + "="*60)
        print("PRIOS 数据集未找到。请先下载：")
        print("="*60)
        print("方法1 - 使用 openneuro-py:")
        print("  !pip install openneuro-py")
        print("  !openneuro download --dataset ds004370 --target-dir ./ds004370")
        print("\n方法2 - 手动下载:")
        print("  访问: https://openneuro.org/datasets/ds004370/versions/1.0.2")
        print("  下载 ZIP 文件并解压到 ./ds004370")
        print("="*60)
        return False

# ========== 2. 核心 ITT 函数 ==========
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
        raise ValueError("数据长度不足")
    traj = np.zeros((n, d))
    for i in range(n):
        traj[i] = x[i : i + d*tau : tau]
    return traj

def compute_theta(traj, delta, c=2.0, time_window=1000, verify_steps=5):
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
        tree = KDTree(traj)
        for i in range(T):
            _, idx = tree.query(traj[i], k=2)
            R_indices.add(idx[1])
    R = traj[list(R_indices)]
    treeR = KDTree(R)
    dist, _ = treeR.query(traj, k=1)
    Theta = np.maximum(0, (dist - delta) / delta)
    return np.mean(Theta)

def preprocess_signal(data, sfreq, target_sf=250, lowcut=0.5, highcut=80, duration_sec=30):
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

def analyze_theta(file_path, label, duration_sec=30, channel_idx=0):
    """分析单个文件的 Θ"""
    import mne
    print(f"  处理 {label}...")
    raw = mne.io.read_raw_brainvision(file_path, preload=True, verbose=False)
    raw.pick_types(ecog=True)
    if len(raw.ch_names) == 0:
        raw.pick_types(eeg=True)
    data = raw.get_data(picks=channel_idx).flatten()
    sfreq = raw.info['sfreq']
    data, sfreq = preprocess_signal(data, sfreq, duration_sec=duration_sec)
    tau = mutual_info_first_min(data, max_lag=100)
    d = cao_embedding_dim(data, tau, max_dim=6)
    d = max(d, 2)
    traj = reconstruct(data, tau, d)
    diffs = np.linalg.norm(np.diff(traj, axis=0), axis=1)
    delta = np.percentile(diffs, 5)
    theta = compute_theta(traj, delta)
    print(f"    tau={tau}, d={d}, Θ={theta:.6f}")
    return theta

# ========== 3. 批量分析主程序 ==========
def main():
    print("="*60)
    print("ITT 预言二：PRIOS 数据集独立复现")
    print("清醒 (SPESclin) vs 丙泊酚麻醉 (SPESprop)")
    print("="*60)

    # 检查数据
    data_dir = "./ds004370"
    if not check_prios_data(data_dir):
        return

    # 查找所有患者
    patients = []
    for d in os.listdir(data_dir):
        if d.startswith('sub-PRIOS') and os.path.isdir(os.path.join(data_dir, d)):
            patients.append(d)

    print(f"\n找到 {len(patients)} 名患者: {patients}")

    results = []
    for patient in patients:
        patient_dir = os.path.join(data_dir, patient, 'ses-1', 'ieeg')
        if not os.path.exists(patient_dir):
            continue
        
        # 查找清醒和麻醉文件
        awake_file = None
        propofol_file = None
        for f in os.listdir(patient_dir):
            if f.endswith('.vhdr'):
                if 'SPESclin' in f:
                    awake_file = os.path.join(patient_dir, f)
                elif 'SPESprop' in f:
                    propofol_file = os.path.join(patient_dir, f)
        
        if awake_file and propofol_file:
            print(f"\n===== {patient} =====")
            try:
                theta_awake = analyze_theta(awake_file, f"{patient} 清醒", duration_sec=30)
                theta_prop = analyze_theta(propofol_file, f"{patient} 麻醉", duration_sec=30)
                results.append({
                    'subject': patient,
                    'awake_theta': theta_awake,
                    'propofol_theta': theta_prop
                })
            except Exception as e:
                print(f"  分析失败: {e}")
        else:
            print(f"患者 {patient} 缺少清醒或麻醉文件")

    # 汇总结果
    print("\n" + "="*60)
    print("分析结果汇总")
    print("="*60)

    if results:
        df = pd.DataFrame(results)
        print(df.to_string(index=False))
        
        awake_vals = df['awake_theta']
        prop_vals = df['propofol_theta']
        
        print(f"\n清醒状态 Θ 均值: {np.mean(awake_vals):.6f} ± {np.std(awake_vals):.6f}")
        print(f"麻醉状态 Θ 均值: {np.mean(prop_vals):.6f} ± {np.std(prop_vals):.6f}")
        
        # 配对 t 检验
        t_stat, p_val = stats.ttest_rel(awake_vals, prop_vals)
        print(f"配对 t 检验: t = {t_stat:.4f}, p = {p_val:.4f}")
        
        # 个体下降比例
        decreased = (prop_vals < awake_vals).sum()
        print(f"个体下降比例: {decreased}/{len(results)} ({decreased/len(results)*100:.0f}%)")
        
        if p_val < 0.05:
            print("\n✓ 预言二获得支持：麻醉状态下 Θ 显著降低")
        else:
            print("\n⚠ 预言二未获得显著支持：p 值未达到 0.05")
        
        # 可视化
        plt.figure(figsize=(10, 6))
        x = np.arange(len(df))
        width = 0.35
        plt.bar(x - width/2, df['awake_theta'], width, label='Awake', color='green', alpha=0.7)
        plt.bar(x + width/2, df['propofol_theta'], width, label='Propofol', color='orange', alpha=0.7)
        plt.xticks(x, df['subject'], rotation=45)
        plt.ylabel('Θ (Self-reference distance)')
        plt.title('ITT 预言二：PRIOS 数据集 - 清醒 vs 丙泊酚麻醉')
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        # 添加个体连线
        for i, row in df.iterrows():
            plt.plot([i - width/2, i + width/2], [row['awake_theta'], row['propofol_theta']], 
                     color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('prios_theta_paired.png', dpi=150)
        plt.show()
        print("\n图表已保存: prios_theta_paired.png")
    else:
        print("没有成功分析任何患者")

if __name__ == "__main__":
    main()
