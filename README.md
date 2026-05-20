# ITT（有bug，暂停修复中） 意识物理测量框架 — 真实数据复现指南

## 项目概述及定位

- 本脚本包用于复现论文《惯性-张力理论（ITT）：一个关于意识的公理化物理学框架》中的核心预言，使用公开的人类神经影像数据集。
- ITT 的三条公设各自对应意识的基本物理属性，这些属性在自然界中可能单独出现（如阻尼系统有惯性、梯度系统有张力、反馈系统有自反性）。但单一属性或任意两条属性的组合，不构成人类概念上的意识。意识作为涌现现象，要求三条公设在同一个物理载体上形成不可拆分的耦合结构（三体纠缠）。预言一、二、三的操作化定义，仅在确认三体纠缠的系统上具有意识判定意义；在单一公设或两两组合的系统上，这些量可能仍有数学定义，但不用于意识判定。ITT v1.0 的研究范围限定于宏观物理意识，即以人类神经系统尺度为标准参照的意识物理条件。三条公设在单一或两两组合时可能对应更微观尺度的物理过程（包括潜在的量子意识过程），但这超出 ITT v1.0 的检验范围。ITT 不否认、不研究、不证伪量子意识假说，但为其保留了理论兼容接口：如果未来量子意识研究能在微观尺度上独立验证单一公设的意识关联性，ITT 的宏观框架可与其实现层级衔接。

## 数据集

**推荐数据集**: OpenNeuro ds005620
- **标题**: A repeated awakening study exploring the capacity of complexity measures to capture dreaming during propofol sedation
- **被试**: 21名健康成年人（年龄18-35岁）
- **设备**: 62通道 EEG（BrainAmp MR plus）
- **采样率**: 5000 Hz
- **条件**: 
  - `awake`: 清醒闭眼静息态
  - `sed`: 丙泊酚镇静（轻度）
  - `sed2`: 丙泊酚镇静（深度）
- **大小**: ~77 GB（完整数据集）
- **下载**: `aws s3 sync --no-sign-request s3://openneuro.org/ds005620 ./ds005620`

## 文件清单

| 文件 | 说明 | 大小 |
|------|------|------|
| `itt_core.py` | ITT核心模块（IAAFT、Bootstrap、KDE） | ~15 KB |
| `ITT_stage1.py` | 阶段一：意识涌现条件 Λ > Λ_noise | ~3 KB |
| `ITT_stage2.py` | 阶段二：意识度 Ψ(s) = Θ(s) | ~5 KB |
| `ITT_stage3.py` | 阶段三：方向规则 | ~8 KB |
| `ITT_realdata_replication.py` | **完整真实数据复现脚本** | ~15 KB |

## 环境要求

### 硬件
- **存储**: >100 GB 可用空间（完整分析）
- **内存**: >16 GB RAM（推荐）
- **CPU**: 多核处理器（Bootstrap和KDE计算密集型）

### 软件
- Python 3.8+
- AWS CLI（用于从S3下载数据）

### Python依赖
```bash
pip install mne numpy scipy matplotlib boto3
```

## 快速开始

### 1. 环境准备

```bash
# 检查AWS CLI安装
aws --version

# 安装Python依赖
pip install mne numpy scipy matplotlib boto3
```

### 2. 下载数据（单个被试样本）

```bash
# 创建数据目录
mkdir ds005620

# 下载单个被试（约3GB，推荐用于测试）
aws s3 cp --no-sign-request s3://openneuro.org/ds005620/sub-1016 ./ds005620/sub-1016 --recursive

# 或下载完整数据集（约77GB，用于完整复现）
# aws s3 sync --no-sign-request s3://openneuro.org/ds005620 ./ds005620
```

### 3. 运行分析

```bash
python ITT_realdata_replication.py
```

### 4. 交互式流程

脚本将引导你完成以下步骤：
1. **环境检查**: 验证依赖和存储空间
2. **数据选择**: 选择被试和条件
3. **预处理**: 自动执行标准EEG预处理
4. **ITT分析**: 计算三个核心预言指标
5. **结果输出**: 生成对比图表和统计报告

## 分析流程详解

### 预处理步骤

```python
# 标准EEG预处理（基于文献最佳实践）
raw = mne.io.read_raw_brainvision(vhdr_file, preload=True)
raw.pick_types(eeg=True)                    # 选择EEG通道
raw.filter(0.5, 80, fir_design='firwin')   # 带通滤波
raw.notch_filter(50)                        # 工频陷波
raw.set_eeg_reference('average')            # 共同平均参考
raw.resample(100)                           # 降采样至100Hz
```

### ITT分析步骤

#### 预言一：意识涌现条件 Λ > Λ_noise

```python
# 1. 估计延迟和嵌入维数
tau = mutual_info_first_min(signal, max_lag=50)
d = false_nearest_neighbor(signal, tau=tau, max_dim=15)

# 2. 重构状态空间
traj = reconstruct(signal, tau, d)

# 3. 计算闭环因果强度 Λ
lam_real = compute_lambda(traj, dt)

# 4. IAAFT替代数据检验
lam_real, lam_noise, p_value, _ = lambda_significance(signal, tau, d, dt)

# 判断: 清醒时 Λ > Λ_noise (p < 0.05)
#       镇静时 Λ ≈ Λ_noise (p > 0.05)
```

#### 预言二：意识度 Ψ(s) = Θ(s)

```python
# 1. 构造回归集 R（历史重复状态）
R_indices = find_recurrent_points(traj, delta, window)

# 2. 计算自指映射 Π(s)
Pi = nearest_neighbor_in_R(traj, R)

# 3. 计算可访问自指距离
delta = noise_level(traj)
Theta = max(0, (d(s, Pi) - delta) / delta)

# 判断: 清醒 Θ > 困倦 Θ > 麻醉 Θ
```

#### 预言三：方向规则

```python
# 1. 计算有向自指偏差 Δ⃗(s)
Delta_vec = s - Pi(s)
Delta_unit = Delta_vec / ||Delta_vec||

# 2. 估计张力势梯度 ∇U（排除Pi附近点确保独立性）
gradU = gradient_u_local_linear(traj, exclusion_radius=...)

# 3. 计算方向规则一致性
dot = Delta_unit · gradU
sigma = sign(dot)
R = mean(sigma > 0)

# 判断: R > 0.50（零假设拒绝）
#       理想: R > 0.75（强支持）
```

## 预期结果

### 理想结果（ITT预言）

| 指标 | 清醒 (Awake) | 镇静 (Sedation) | 深镇静 (Deep Sed) |
|------|-------------|----------------|------------------|
| Λ | > Λ_noise | ≈ Λ_noise | ≈ Λ_noise |
| Θ | 高 | 中 | 低 |
| R | > 0.75 | > 0.50 | ≈ 0.50 |

### 证伪标准

若以下任一条件在 ≥2 个独立被试上成立，ITT 核心预言被证伪：

1. **预言一证伪**: 清醒状态 Λ ≤ Λ_noise（统计不显著）
2. **预言二证伪**: 清醒 Θ ≤ 镇静 Θ（排序错误）
3. **预言三证伪**: 清醒 R ≤ 0.50（方向规则不成立）

## 已知问题与限制

### 代码实现问题
- **原始代码偏差**: ∇U估计与Pi计算使用相同近邻结构导致人为正相关
- **修正方案**: 在∇U估计时排除Pi附近点（已在 `ITT_realdata_replication.py` 中修正）

### 数据限制
- **样本量**: 21名被试，统计功效有限
- **个体差异**: 丙泊酚反应存在个体差异
- **通道选择**: 分析结果可能依赖于所选通道

## 结果解读

### 强支持（ITT预言成立）
- 清醒: Λ > Λ_noise, Θ 高, R > 0.75
- 镇静: Λ ≈ Λ_noise, Θ 低, R ≈ 0.50

### 弱支持（需进一步研究）
- 清醒: Λ > Λ_noise, Θ 中, R > 0.50
- 镇静: Λ ≈ Λ_noise, Θ 中低, R ≈ 0.50

### 证伪（ITT预言不成立）
- 清醒与镇静的 Λ、Θ、R 无显著差异
- 或差异方向与预言相反

## 扩展分析建议

1. **多通道分析**: 不仅分析Oz，还分析Fz、Cz等关键通道
2. **频带特异性**: 分析不同频带（delta、theta、alpha、beta、gamma）的ITT指标
3. **时变分析**: 计算Θ(t)随时间变化，观察意识转换动态
4. **机器学习**: 使用Λ、Θ、R作为特征训练意识状态分类器
5. **跨数据集验证**: 在ds003768（睡眠）和ds004504（阿尔茨海默）上验证

## 引用

若使用本脚本包进行发表，请引用：

1. 原始论文: Tuling Zhongwen (2026). https://doi.org/10.5281/zenodo.20204270
2. 数据集: OpenNeuro ds005620. DOI: 10.18112/openneuro.ds005620.v1.0.0

## 联系方式

- 代码仓库: https://github.com/TulingZhongwen/ITT-experiment
- 论文预印本: 预印本 v1.0

## 许可

本脚本包遵循与原始代码仓库相同的许可协议。
