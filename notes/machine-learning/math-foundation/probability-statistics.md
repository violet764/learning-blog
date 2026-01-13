# 概率论与统计学基础

## 概述

概率论与统计学是机器学习的核心数学基础，为不确定性建模、参数估计和假设检验提供了理论支持。理解概率分布、统计推断和假设检验对于掌握机器学习算法至关重要。

## 概率论基础

### 概率空间

**定义**：概率空间(Ω, F, P)由三个要素组成：
- **样本空间Ω**：所有可能结果的集合
- **事件域F**：Ω的子集构成的σ-代数
- **概率测度P**：满足Kolmogorov公理的函数

### 概率公理

1. **非负性**：∀A∈F, P(A) ≥ 0
2. **规范性**：P(Ω) = 1
3. **可列可加性**：对于互不相容的事件序列{A₁, A₂, ...}，有P(∪ᵢAᵢ) = ∑ᵢP(Aᵢ)

### 条件概率与贝叶斯定理

**条件概率**：
$$
P(A|B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0
$$

**贝叶斯定理**：
$$
P(A|B) = \frac{P(B|A)P(A)}{P(B)}
$$

```python
import numpy as np

# 贝叶斯定理示例
# 假设疾病检测：患病率1%，检测准确率99%
P_disease = 0.01      # 先验概率：患病
P_positive_given_disease = 0.99    # 似然：患病时检测阳性
P_positive_given_no_disease = 0.01  # 假阳性率

# 计算全概率
P_positive = (P_positive_given_disease * P_disease + 
             P_positive_given_no_disease * (1 - P_disease))

# 贝叶斯更新
P_disease_given_positive = (P_positive_given_disease * P_disease) / P_positive

print(f"检测阳性后患病的后验概率: {P_disease_given_positive:.4f}")
```

## 随机变量与概率分布

### 离散随机变量

**定义**：取值可数的随机变量。

**概率质量函数(PMF)**：
$$
P(X = x_i) = p_i, \quad \sum_i p_i = 1
$$

#### 常见离散分布

1. **伯努利分布**：
$$
P(X = 1) = p, \quad P(X = 0) = 1-p
$$

2. **二项分布**：n次伯努利试验的成功次数
$$
P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}
$$

3. **泊松分布**：单位时间内事件发生次数
$$
P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}
$$

### 连续随机变量

**定义**：取值不可数的随机变量。

**概率密度函数(PDF)**：
$$
P(a \leq X \leq b) = \int_a^b f(x) dx, \quad \int_{-\infty}^{\infty} f(x) dx = 1
$$

#### 常见连续分布

1. **均匀分布**：
$$
f(x) = \begin{cases}
\frac{1}{b-a} & a \leq x \leq b \\
0 & \text{其他}
\end{cases}
$$

2. **正态分布（高斯分布）**：
$$
f(x) = \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{(x-\mu)^2}{2\sigma^2}}
$$

3. **指数分布**：
$$
f(x) = \lambda e^{-\lambda x}, \quad x \geq 0
$$

```python
import matplotlib.pyplot as plt
import scipy.stats as stats

# 概率分布可视化
x = np.linspace(-4, 4, 1000)

# 正态分布
normal_pdf = stats.norm.pdf(x, 0, 1)

# 均匀分布
uniform_pdf = stats.uniform.pdf(x, -2, 4)

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(x, normal_pdf, label='N(0,1)')
plt.title('正态分布')
plt.xlabel('x')
plt.ylabel('概率密度')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x, uniform_pdf, label='U(-2,2)')
plt.title('均匀分布')
plt.xlabel('x')
plt.ylabel('概率密度')
plt.legend()

plt.tight_layout()
plt.show()
```

## 数字特征

### 期望（均值）

**定义**：随机变量的平均值。

- **离散变量**：E[X] = ∑xᵢP(X=xᵢ)
- **连续变量**：E[X] = ∫xf(x)dx

### 方差与标准差

**方差**：衡量随机变量与其均值的偏离程度。
$$
\text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
$$

**标准差**：方差的平方根。
$$
\sigma = \sqrt{\text{Var}(X)}
$$

### 协方差与相关系数

**协方差**：衡量两个随机变量的线性相关性。
$$
\text{Cov}(X,Y) = E[(X - E[X])(Y - E[Y])]
$$

**相关系数**：标准化的协方差。
$$
\rho_{XY} = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y}
$$

## 大数定律与中心极限定理

### 大数定律

**弱大数定律**：样本均值依概率收敛于总体均值。
$$
\lim_{n \to \infty} P\left(\left|\frac{1}{n}\sum_{i=1}^n X_i - \mu\right| > \epsilon\right) = 0
$$

### 中心极限定理

**定理**：独立同分布随机变量的和近似服从正态分布。
$$
\frac{\sum_{i=1}^n X_i - n\mu}{\sigma\sqrt{n}} \xrightarrow{d} N(0,1)
$$

```python
# 中心极限定理演示
np.random.seed(42)
n_samples = 10000
sample_means = []

# 从均匀分布采样，计算样本均值
for _ in range(n_samples):
    sample = np.random.uniform(0, 1, 100)  # 100个均匀分布样本
    sample_means.append(np.mean(sample))

plt.figure(figsize=(10, 6))
plt.hist(sample_means, bins=50, density=True, alpha=0.7, label='样本均值分布')

# 理论正态分布
x = np.linspace(0.4, 0.6, 100)
theoretical_mean = 0.5  # 均匀分布的均值
theoretical_std = 1/np.sqrt(12*100)  # 均匀分布方差=1/12，样本均值方差=σ²/n
normal_pdf = stats.norm.pdf(x, theoretical_mean, theoretical_std)

plt.plot(x, normal_pdf, 'r-', linewidth=2, label='理论正态分布')
plt.title('中心极限定理演示')
plt.xlabel('样本均值')
plt.ylabel('概率密度')
plt.legend()
plt.show()
```

## 统计推断

### 参数估计

#### 点估计

**最大似然估计(MLE)**：选择使观测数据出现概率最大的参数值。
$$
\hat{\theta}_{MLE} = \arg\max_{\theta} L(\theta|X) = \arg\max_{\theta} \prod_{i=1}^n f(x_i|\theta)
$$

**矩估计**：使样本矩等于总体矩的参数值。

#### 区间估计

**置信区间**：参数真值以一定概率落入的区间。
$$
P(\theta_L \leq \theta \leq \theta_U) = 1 - \alpha
$$

### 假设检验

#### 基本概念

- **原假设H₀**：待检验的假设
- **备择假设H₁**：与原假设对立的假设
- **显著性水平α**：犯第一类错误的最大概率
- **p值**：在原假设下，观测到比样本更极端结果的概率

#### 常用检验方法

1. **t检验**：检验均值是否等于特定值
2. **卡方检验**：检验分类变量的独立性
3. **F检验**：检验方差齐性

```python
# 假设检验示例：t检验
from scipy.stats import ttest_1samp

# 生成样本数据
np.random.seed(42)
sample_data = np.random.normal(loc=5.2, scale=1.0, size=100)

# 单样本t检验：检验均值是否等于5
# H0: μ = 5, H1: μ ≠ 5
t_stat, p_value = ttest_1samp(sample_data, 5)

print(f"样本均值: {np.mean(sample_data):.4f}")
print(f"t统计量: {t_stat:.4f}")
print(f"p值: {p_value:.4f}")

if p_value < 0.05:
    print("拒绝原假设，样本均值显著不等于5")
else:
    print("无法拒绝原假设，样本均值可能等于5")
```

## 信息论基础

### 信息熵

**定义**：衡量随机变量的不确定性。
$$
H(X) = -\sum_{i=1}^n p(x_i) \log p(x_i)
$$

### 交叉熵

**定义**：衡量两个概率分布的差异。
$$
H(P,Q) = -\sum_{i=1}^n p(x_i) \log q(x_i)
$$

### KL散度

**定义**：衡量两个概率分布的非对称差异。
$$
D_{KL}(P||Q) = \sum_{i=1}^n p(x_i) \log \frac{p(x_i)}{q(x_i)}
$$

## 在机器学习中的应用

### 朴素贝叶斯分类器

基于贝叶斯定理和特征条件独立性假设。

### 逻辑回归

使用sigmoid函数将线性组合映射到概率空间。

### 高斯过程

基于多元正态分布的贝叶斯非参数方法。

### 模型评估

使用统计检验比较不同模型的性能。

## 数值计算考虑

### 数值稳定性

1. **对数概率**：使用log概率避免数值下溢
2. **softmax函数**：数值稳定的实现方式
3. **协方差矩阵**：防止矩阵奇异的正则化方法

### 计算复杂度

| 操作 | 复杂度 | 备注 |
|------|--------|------|
| 参数估计 | O(n) | 样本数量 |
| 假设检验 | O(n) | 样本数量 |
| 协方差矩阵 | O(nd²) | 特征维度平方 |
| 多元正态分布 | O(d³) | 特征维度立方 |

## 总结

概率论与统计学为机器学习提供了处理不确定性和进行统计推断的理论基础。从基本的概率概念到高级的统计方法，这些工具在各种机器学习算法中都有广泛应用。深入理解概率统计不仅有助于掌握算法原理，还能提高模型选择和评估的能力。

---

*下一章：优化理论*