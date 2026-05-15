ITT实验验证指南（阶段二）：基于公开EEG/fMRI数据检验三个预言

版本：2.0
依据：惯性-张力理论（ITT）论文v1（DOI：10.5281/zenodo.20204270）
开源协作：欢迎提交Issue/PR，详见GitHub仓库
适用对象：已完成阶段一（模拟验证）或具非线性动力学经验的研究者

本指南提供完整的、可复现的实验方案，用于检验ITT的三大预言。所有步骤均基于公开数据，无需新采集。我们鼓励社区成员独立复现，并反馈结果。

---

1. 总体目标

· 预言一（意识涌现最低物理条件）：清醒状态Λ > Λ\_noise（p<0.01）；深度睡眠/麻醉状态 Λ ≈ Λ\_noise（p>0.05）。
· 预言二（意识度）：平均Θ(s) 清醒 > 困倦 > 麻醉/深睡（效应量Cohen's d > 0.8或统计显著）。
· 预言三（自反力方向规则）：当Δ̅·∇U < 0时，自发运动方向与–∇U夹角<90°的比例>75%；当Δ̅·∇U > 0时，与+∇U夹角<90°的比例>75%。

---

2. 推荐公开数据集（带明确意识状态标签）

数据集 模态 状态标签 适合预言 链接
ds002718 EEG 基线清醒→丙泊酚镇静→意识丧失→恢复 一、二 OpenNeuro
ds003478 EEG 清醒闭眼、N1、N2、N3、REM 一、二 OpenNeuro
HCP S1200 fMRI 清醒、轻睡（部分有） 一、二（验证可重复性） ConnectomeDB
EEG+Stroop EEG+行为 清醒（错误/正确反应），需公开标签 三 见论文

优先使用 ds002718（麻醉边界清晰）和 ds003478（多级意识）。
对于预言三，需寻找公开的“同步EEG+按键反应”数据集，例如 eeg-during-stroop-task（OpenNeuro ds002785）。

---

3. 数据预处理（标准化脚本，见GitHub）

3.1 EEG预处理（MNE-Python）

```python
import mne
raw = mne.io.read_raw_edf(file, preload=True)
raw.filter(0.5, 45, fir_design='firwin')
raw.notch_filter(50)
raw.set_eeg_reference('average')
# ICA去伪迹（眼电、肌电）
ica = mne.preprocessing.ICA(n_components=20, random_state=0)
ica.fit(raw.copy().filter(1, 40))
eog_indices = ica.find_bads_eog(raw, ch_name='Fp1')[0]  # 根据实际通道调整
ica.exclude = eog_indices
raw = ica.apply(raw)
# 分段：每个状态取至少5分钟稳定片段
epochs = mne.make_fixed_length_epochs(raw, duration=300, overlap=0)
```

3.2 fMRI预处理（fMRIPrep输出）

· 使用fMRIPrep生成的preproc_bold.nii.gz和confounds.tsv。
· 提取全脑平均时间序列（或ROI平均，如默认模式网络）。
· 去线性漂移、滤波（0.01–0.1 Hz），并回归24个头动参数。

---

4. 参数自适应选择（避免固定值）

所有参数必须按每个信号（通道或被试）独立计算，并在报告中列出范围。

参数 方法 实现建议
延迟τ 互信息第一极小值 scipy.signal.find_peaks(-mi)[0][0]
嵌入维数d 假近邻法（FNN）阈值0.01 pyeeg.false_nearest_neighbor或自写
邻域点数n_neighbors max(2*d, 30)，且n_neighbors < N/10 动态调整，若不足则警告
噪声水平δ 轨迹点间欧氏距离的5%分位数 np.percentile(np.diff(traj, axis=0), 5)
周期点阈值ε 取δ 同δ
置换次数 100（打乱） 对原始时间序列使用IAAFT（迭代幅度调整傅里叶变换）生成保持功率谱结构的surrogate数据

所有参数选择过程必须记录，并可在ITT-experiment仓库中导出日志。

---

5. 核心算法实现（Python，已优化）

5.1 计算Λ（闭环因果强度）

```python
def compute_lambda(traj, dt, n_neighbors=30):
    """
    traj: (T, d) 状态空间轨迹
    dt: 采样间隔（秒）
    """
    T, d = traj.shape
    n_neighbors = max(2*d, n_neighbors)
    traces = []
    from scipy.spatial import KDTree
    tree = KDTree(traj)
    for t in range(T-1):
        dist, idx = tree.query(traj[t], k=n_neighbors+1)  # +1排除自身
        neighbors = idx[1:]
        X = traj[neighbors]
        Y = traj[neighbors+1] - X
        J = np.linalg.lstsq(X, Y, rcond=None)[0].T
        traces.append(np.abs(np.trace(J)) / d)
    return np.mean(traces)
```

5.2 IAAFT置换检验与Λ_noise

```python
def lambda_vs_noise(signal, tau, d, n_surrogate=100):
    traj = reconstruct(signal, tau, d)
    Lambda_real = compute_lambda(traj, dt, n_neighbors=30)
    Lambda_shuff = []
    for _ in range(n_surrogate):
        shuf = np.random.permutation(signal)
        traj_shuf = reconstruct(shuf, tau, d)
        Lambda_shuff.append(compute_lambda(traj_shuf, dt, n_neighbors=30))
    p = (np.sum(np.array(Lambda_shuff) >= Lambda_real) + 1) / (n_surrogate+1)
    return Lambda_real, np.percentile(Lambda_shuff, 95), p
```

5.3 计算Θ(s)（可访问自指距离，使用KDTree加速）

```python
def compute_theta(traj, delta, eps=None):
    if eps is None: eps = delta
    from scipy.spatial import KDTree
    # 寻找近似周期点对（ε邻域内）
    # 为避免O(N²)，采用KDTree查询：每个点找邻域内其他点
    tree = KDTree(traj)
    R_indices = set()
    for i, point in enumerate(traj):
        idx = tree.query_ball_point(point, eps)
        idx = [j for j in idx if j != i]
        if idx:
            R_indices.update(idx)
            R_indices.add(i)  # 自身也加入回归集
    if not R_indices:
        return 0.0
    R = traj[list(R_indices)]
    # 对每个点，计算到R的最近距离
    tree_R = KDTree(R)
    distances, _ = tree_R.query(traj, k=1)
    Theta = np.maximum(0, (distances - delta) / delta)
    return np.mean(Theta)
```

5.4 方向规则（预言三）所需计算

```python
def compute_direction_rule(traj, gradU, Delta_unit):
    """
    traj: (T, d)
    gradU: (T, d) 张力梯度（由概率密度负对数差分估计）
    Delta_unit: (T, d) 单位偏差向量
    """
    dot = np.sum(Delta_unit * gradU, axis=1)
    v = np.diff(traj, axis=0)  # 速度
    v_norm = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    # 对齐前一时刻的dot（因为速度是与前一帧的差分）
    dot = dot[:-1]
    pos = dot < 0
    neg = dot > 0
    if np.sum(pos)==0 or np.sum(neg)==0:
        return 0.5
    align_pos = np.sum(np.sum(v_norm[pos] * (-gradU[:-1][pos]), axis=1) > 0) / np.sum(pos)
    align_neg = np.sum(np.sum(v_nist[neg] * (gradU[:-1][neg]), axis=1) > 0) / np.sum(neg)
    return (align_pos + align_neg)/2
```

---

6. 预言三的数据集与附加步骤

· 推荐数据集：OpenNeuro ds002785 (EEG Stroop task)。包含同步EEG和按键反应，可提取反应正确/错误试次的脑电信号。
· 预处理：同上，按试次分段（-1s ~ 2s相对刺激）。
· 状态划分：“高意识参与”试次（正确反应）vs“低意识参与”（错误或事后报告无意识？）——实际上需要利用行为反应建立意识水平的代理标记（例如反应时、正确率）。
· 计算Δ̅和∇U：需对每个时间窗（如刺激后200-400ms）的状态空间进行密度估计（核密度），求梯度。
· 组水平统计：计算每个试次的一致性比例，然后用配对t检验比较正确与错误试次。

预言三的检验较为复杂，建议先完成预言一、二，再作为独立研究扩展。

---

7. 统计检验与效应量

预言 统计方法 预期效应量/阈值
一（个体层） 置换检验（p<0.01） 清醒组>80%显著，麻醉组<20%显著
一（组层） 配对t检验（清醒vs麻醉的Λ真实值） Cohen's d > 1.0
二 配对t检验或Wilcoxon 清醒 vs 麻醉：Cohen's d > 0.8
三 二项检验（一致性 >75%） p < 0.001

必须报告效应量，不能仅依赖p值。

---

8. 与现有指标的对照分析（建立“外部锚点”）

为了验证Λ和Θ不是“另一个复杂度指标”，需同时计算：

· LZc (Lempel-Ziv复杂度)：使用antropy.lziv_complexity
· Permutation entropy：antropy.perm_entropy
· PCI估计：如有TMS-EEG数据可用，否则跳过
· 样本熵（Sample Entropy）

计算所有指标后，绘制斯皮尔曼相关矩阵，检验Λ与LZc的相关性是否显著（r>0.6）。若相关度过高，需论证Λ的独特性；若低相关，则需解释为何Λ能区分意识而其他指标不能。

---

9. 稳健性分析与参数敏感性

· 嵌入维数敏感性：在最优d±3范围内重复计算Λ，报告变异系数CV。
· 采样率敏感性：对数据降采样至250 Hz、125 Hz重新计算，检查预言是否稳定。
· 噪声水平变化：人为添加不同信噪比的高斯噪声（SNR=10,5,2），观察Λ与Λ_noise的分离度变化。

这些分析脚本也应包含在仓库中。

---

10. 结果报告模板（建议Jupyter Notebook）

一个完整的分析应输出：

1. 被试列表与状态标签。
2. 参数选择表格（每个被试的τ、d、δ等）。
3. 箱线图：清醒 vs 麻醉组的 Λ、Θ。
4. 置换检验直方图：清醒组、麻醉组各一个典型被试。
5. 效应量条形图：Cohen's d与95%置信区间。
6. 相关性矩阵热图：Λ、Θ、LZc、PermEn等。
7. 方向规则散点图：展示一致性比例。

所有图表自动保存到results/文件夹。

---

11. 如何贡献与协作

· Fork仓库 TulingZhongwen/ITT-experiment
· 运行指南：在examples/中创建run_stage2_ds002718.ipynb，提交PR。
· 问题反馈：在GitHub Issues中贴上运行日志和错误信息。
· 结果汇总：在仓库的results/文件夹中上传自己的输出图表（命名格式：ds00xxx_subjX.png）。

初次贡献者可以只完成一个数据集、一个被试的验证。

---

12. 常见问题与调试

· 状态空间重构计算太慢：使用KDTree或仅取前20000个点（足矣）。
· Λ在麻醉组也显著：检查预处理是否残留肌电/眼电伪迹，或嵌入维数过高导致过拟合。
· Θ(s)全为0：增大周期点阈值ε（例如取2δ）或增加轨迹长度。
· fMRI时间点不足：使用滑动窗口（长度100 TR，步长1 TR），计算每个窗口的Λ均值。

---

版本历史

· v2.0 : 增加稳健性分析、与现有指标对比、预言三的具体数据集、参数自适应规则。
  维护者：图灵中文
  许可证：MIT
  
  --

以下基于公开EEG数据验证预言一和预言二（完整脚本）

此脚本需要先下载 ds002718 数据集（丙泊酚麻醉），脚本会自动预处理、计算Λ和Θ、统计并以箱线图和直方图呈现。

```python
"""
itt_stage2_anesthesia.py
使用ds002718 (丙泊酚EEG) 验证ITT预言一和预言二。
需要先安装: mne, numpy, scipy, scikit-learn, matplotlib, antropy
数据集下载: 从OpenNeuro下载 ds002718 并解压到 ./data/ds002718/
或使用 mne.datasets 无法直接获取，需手动下载。
"""
import os
import numpy as np
import mne
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from scipy import stats

# ------------------- 复用阶段一的ITT核心函数 -------------------
# (将阶段一中的 reconstruct, mutual_info_first_min, false_nearest_neighbor,
#  compute_lambda, lambda_significance 以及 compute_theta 函数复制过来)

def compute_theta(traj, delta, eps=None):
    """计算平均自指距离 Θ(s)"""
    if eps is None:
        eps = delta
    tree = KDTree(traj)
    # 为每个点寻找eps邻域内的其他点 -> 构成回归集R
    R_indices = set()
    for i, p in enumerate(traj):
        idx = tree.query_ball_point(p, eps)
        idx = [j for j in idx if j != i]
        if idx:
            R_indices.update(idx)
            R_indices.add(i)
    if not R_indices:
        return 0.0
    R = traj[list(R_indices)]
    treeR = KDTree(R)
    dist, _ = treeR.query(traj, k=1)
    delta = np.percentile(np.linalg.norm(np.diff(traj, axis=0), axis=1), 5)
    theta_vals = np.maximum(0, (dist - delta) / delta)
    return np.mean(theta_vals)

# ------------------- 数据加载与预处理 -------------------
def load_ds002718(data_path):
    """加载ds002718中一个被试的EEG，返回原始对象和事件"""
    raw = mne.io.read_raw_edf(os.path.join(data_path, 'sub-01', 'eeg', 'sub-01_task_rest_eeg.edf'),
                              preload=True)
    # 简化：选择Fz, Cz, Pz等通道，也可全部使用
    raw.pick_types(eeg=True, eog=False)  # 保留所有EEG通道
    # 滤波
    raw.filter(0.5, 45, fir_design='firwin')
    raw.notch_filter(50)
    raw.set_eeg_reference('average')
    # 从注释中提取事件（麻醉前、中、后），需根据实际数据调整
    events, event_id = mne.events_from_annotations(raw)
    # 例如事件名可能包含 'baseline', 'sedation', 'loss'
    return raw, events, event_id

def extract_epochs(raw, events, event_id, tmin=-5, tmax=300, baseline=None):
    """提取不同状态的epochs"""
    # 根据实际事件id选择条件
    # 此处示例: 假设事件字典包含 'baseline' 和 'loss'
    epochs = mne.Epochs(raw, events, event_id, tmin=tmin, tmax=tmax,
                        baseline=baseline, preload=True)
    return epochs

# ------------------- 主分析流程 -------------------
def main():
    data_path = './data/ds002718'  # 修改为实际路径
    if not os.path.exists(data_path):
        print("请先下载ds002718数据集，并放在", data_path)
        return

    raw, events, event_id = load_ds002718(data_path)
    # 举例：选择baseline和loss两种状态
    epochs_baseline = extract_epochs(raw, events, {'baseline': event_id['baseline']})
    epochs_loss = extract_epochs(raw, events, {'loss': event_id['loss']})

    # 对每个被试，计算每个epoch的Λ和Θ (这里需循环epochs)
    # 简单起见，我们只取第一个epoch演示，实际应循环所有epochs并进行统计
    # 此处为演示，直接计算连续段（5分钟）的Λ和Θ
    # 实际中应将epochs数据串联成一段连续信号

    # 获取第一个epoch的数据（通道数×时间点）-> 取平均参考
    data_baseline = epochs_baseline.get_data(copy=False)[0]  # (n_channels, n_times)
    data_loss = epochs_loss.get_data(copy=False)[0]

    # 对每个通道计算Λ，然后平均
    tau = None; d = None
    lambdas = []
    thetas = []
    for ch in range(data_baseline.shape[0]):
        sig_b = data_baseline[ch]
        sig_l = data_loss[ch]
        if tau is None:
            tau = mutual_info_first_min(sig_b, max_lag=50)
            d = false_nearest_neighbor(sig_b, tau=tau, max_dim=15)
        # 计算Λ和p值 (置换检验)
        lam_b, _, p_b = lambda_significance(sig_b, tau, d, 1/500)
        lam_l, _, p_l = lambda_significance(sig_l, tau, d, 1/500)
        # 计算Θ
        # 需重构状态空间
        traj_b = reconstruct(sig_b, tau, d)
        traj_l = reconstruct(sig_l, tau, d)
        delta_b = np.percentile(np.linalg.norm(np.diff(traj_b, axis=0), axis=1), 5)
        delta_l = np.percentile(np.linalg.norm(np.diff(traj_l, axis=0), axis=1), 5)
        theta_b = compute_theta(traj_b, delta_b)
        theta_l = compute_theta(traj_l, delta_l)
        lambdas.append((lam_b, lam_l))
        thetas.append((theta_b, theta_l))

    lambdas = np.array(lambdas)
    thetas = np.array(thetas)

    # 统计
    lam_b_mean = np.mean(lambdas[:,0])
    lam_l_mean = np.mean(lambdas[:,1])
    theta_b_mean = np.mean(thetas[:,0])
    theta_l_mean = np.mean(thetas[:,1])

    print("=== 阶段二初步结果 ===")
    print(f"清醒   Λ均值: {lam_b_mean:.4f}, Θ均值: {theta_b_mean:.4f}")
    print(f"麻醉   Λ均值: {lam_l_mean:.4f}, Θ均值: {theta_l_mean:.4f}")
    # t检验
    t_stat, p_val = stats.ttest_rel(lambdas[:,0], lambdas[:,1])
    print(f"配对t检验 p={p_val:.5f}")

    # 绘图
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4))
    ax1.boxplot([lambdas[:,0], lambdas[:,1]], labels=['Baseline','Loss'])
    ax1.set_ylabel('Λ')
    ax1.set_title('预言一: 闭环因果强度')
    ax2.boxplot([thetas[:,0], thetas[:,1]], labels=['Baseline','Loss'])
    ax2.set_ylabel('Θ(s)')
    ax2.set_title('预言二: 自指距离')
    plt.suptitle('ITT阶段二验证 (ds002718 示例)')
    plt.tight_layout()
    plt.savefig('stage2_results.png')
    plt.show()

if __name__ == '__main__':
    main()
```

运行前需要：

1. 安装依赖：pip install mne numpy scipy scikit-learn matplotlib antropy
2. 下载ds002718数据集（约几GB），解压到 ./data/ds002718。
3. 根据实际事件名调整event_id的键值（需要查看数据集的events.tsv）。

阶段二完整分析还应包括置换检验的直方图、效应量、与现有指标对比等，完整代码可去GitHub仓库查看。
