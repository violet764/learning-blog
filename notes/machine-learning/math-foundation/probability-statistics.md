# 概率论与统计学基础

概率论与统计学是机器学习中处理不确定性的核心工具。从贝叶斯分类器到概率图模型，从参数估计到假设检验，概率统计的思想贯穿机器学习的方方面面。理解这些概念，能让你更好地理解模型的不确定性、做出更可靠的预测。

## 概率论基础

### 概率空间

📌 **概率空间**是概率论的数学基础，由三元组 $(\Omega, \mathcal{F}, P)$ 定义：

| 组成部分 | 描述 |
|----------|------|
| 样本空间 $\Omega$ | 所有可能结果的集合 |
| 事件域 $\mathcal{F}$ | $\Omega$ 的子集构成的 σ-代数 |
| 概率测度 $P$ | 满足 Kolmogorov 公理的函数 |

**Kolmogorov 公理**：
1. **非负性**：$\forall A \in \mathcal{F}, P(A) \geq 0$
2. **规范性**：$P(\Omega) = 1$
3. **可列可加性**：对于互不相容的事件序列，概率可相加

### 条件概率与贝叶斯定理

📌 **条件概率**描述在已知某事件发生的情况下，另一事件发生的概率：

$$P(A|B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0$$

📌 **贝叶斯定理**是概率论中最重要的定理之一，它描述了如何根据新证据更新概率：

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

| 术语 | 含义 |
|------|------|
| $P(A)$ | **先验概率**：在观察到证据之前的概率 |
| $P(B\|A)$ | **似然**：在假设 A 为真时观察到证据 B 的概率 |
| $P(A\|B)$ | **后验概率**：在观察到证据之后的概率 |
| $P(B)$ | **证据因子**：观察到证据的概率（归一化常数） |

```python
import numpy as np
import matplotlib.pyplot as plt

# 贝叶斯定理经典案例：疾病检测
# 已知：疾病患病率 1%，检测准确率 99%
P_disease = 0.01                    # 先验：患病概率
P_positive_given_disease = 0.99     # 似然：患病时检测阳性
P_negative_given_healthy = 0.99     # 健康时检测阴性

# 计算检测阳性的总概率（全概率公式）
P_positive = (P_positive_given_disease * P_disease + 
              (1 - P_negative_given_healthy) * (1 - P_disease))

# 贝叶斯更新：检测阳性后真正患病的概率
P_disease_given_positive = (P_positive_given_disease * P_disease) / P_positive

print("=" * 50)
print("贝叶斯定理：疾病检测示例")
print("=" * 50)
print(f"先验概率（患病率）: {P_disease:.2%}")
print(f"检测准确率: {P_positive_given_disease:.2%}")
print(f"检测阳性时真正患病的概率: {P_disease_given_positive:.2%}")
print("\n💡 这个结果可能出乎意料！即使检测准确率很高，")
print("   由于疾病本身很罕见，假阳性的数量会超过真阳性。")

# 可视化：不同先验下的后验概率
prior_range = np.logspace(-4, 0, 100)  # 从 0.01% 到 100%
posteriors = []

for prior in prior_range:
    pos = (P_positive_given_disease * prior) / (
        P_positive_given_disease * prior + 
        0.01 * (1 - prior)
    )
    posteriors.append(pos)

plt.figure(figsize=(10, 5))
plt.semilogx(prior_range * 100, posteriors, 'b-', linewidth=2)
plt.axvline(x=1, color='r', linestyle='--', label='先验=1%')
plt.axhline(y=P_disease_given_positive, color='r', linestyle=':', alpha=0.5)
plt.xlabel('先验概率（患病率）%')
plt.ylabel('后验概率 P(患病|阳性)')
plt.title('贝叶斯更新：先验对后验的影响')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
```

## 随机变量与概率分布

### 离散随机变量

📌 **离散随机变量**的取值是可数的。其概率分布用**概率质量函数（PMF）**描述：

$$P(X = x_i) = p_i, \quad \sum_i p_i = 1$$

#### 常见离散分布

| 分布 | 符号 | 概率质量函数 | 期望 | 方差 |
|------|------|--------------|------|------|
| 伯努利 | $\text{Bern}(p)$ | $P(X=1)=p$ | $p$ | $p(1-p)$ |
| 二项 | $\text{Bin}(n,p)$ | $\binom{n}{k}p^k(1-p)^{n-k}$ | $np$ | $np(1-p)$ |
| 泊松 | $\text{Poi}(\lambda)$ | $\frac{\lambda^k e^{-\lambda}}{k!}$ | $\lambda$ | $\lambda$ |

### 连续随机变量

📌 **连续随机变量**的取值是不可数的。其概率分布用**概率密度函数（PDF）**描述：

$$P(a \leq X \leq b) = \int_a^b f(x) \, dx$$

#### 常见连续分布

| 分布 | 符号 | 概率密度函数 | 期望 | 方差 |
|------|------|--------------|------|------|
| 均匀 | $U(a,b)$ | $\frac{1}{b-a}$, $x \in [a,b]$ | $\frac{a+b}{2}$ | $\frac{(b-a)^2}{12}$ |
| 正态 | $N(\mu,\sigma^2)$ | $\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | $\mu$ | $\sigma^2$ |
| 指数 | $\text{Exp}(\lambda)$ | $\lambda e^{-\lambda x}$, $x \geq 0$ | $\frac{1}{\lambda}$ | $\frac{1}{\lambda^2}$ |

```python
import scipy.stats as stats

# 常见分布可视化
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 二项分布
ax1 = axes[0, 0]
n, p = 20, 0.5
x_binom = np.arange(0, n + 1)
ax1.bar(x_binom, stats.binom.pmf(x_binom, n, p), alpha=0.7, color='steelblue')
ax1.set_title(f'二项分布 Bin({n}, {p})')
ax1.set_xlabel('k')
ax1.set_ylabel('P(X=k)')

# 泊松分布
ax2 = axes[0, 1]
lambda_param = 5
x_poisson = np.arange(0, 15)
ax2.bar(x_poisson, stats.poisson.pmf(x_poisson, lambda_param), alpha=0.7, color='coral')
ax2.set_title(f'泊松分布 Poi({lambda_param})')
ax2.set_xlabel('k')
ax2.set_ylabel('P(X=k)')

# 正态分布
ax3 = axes[1, 0]
x_norm = np.linspace(-4, 4, 100)
for mu, sigma, color in [(0, 1, 'blue'), (0, 2, 'green'), (1, 0.5, 'red')]:
    ax3.plot(x_norm, stats.norm.pdf(x_norm, mu, sigma), 
             color=color, label=f'μ={mu}, σ={sigma}')
ax3.set_title('正态分布')
ax3.set_xlabel('x')
ax3.set_ylabel('f(x)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 指数分布
ax4 = axes[1, 1]
x_exp = np.linspace(0, 5, 100)
for lam, color in [(0.5, 'blue'), (1, 'green'), (2, 'red')]:
    ax4.plot(x_exp, stats.expon.pdf(x_exp, scale=1/lam), 
             color=color, label=f'λ={lam}')
ax4.set_title('指数分布')
ax4.set_xlabel('x')
ax4.set_ylabel('f(x)')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 多元正态分布

📌 **多元正态分布**是机器学习中最重要的多维分布：

$$f(\mathbf{x}) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$

其中 $\boldsymbol{\mu}$ 是均值向量，$\Sigma$ 是协方差矩阵。

```python
# 二元正态分布可视化
from mpl_toolkits.mplot3d import Axes3D

# 定义参数
mu = np.array([0, 0])
cov = np.array([[1, 0.8], [0.8, 1]])

# 生成网格
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))

# 计算概率密度
rv = stats.multivariate_normal(mu, cov)
Z = rv.pdf(pos)

# 绘制 3D 图
fig = plt.figure(figsize=(14, 5))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax1.set_xlabel('X₁')
ax1.set_ylabel('X₂')
ax1.set_zlabel('密度')
ax1.set_title('二元正态分布 (3D)')

# 绘制等高线图
ax2 = fig.add_subplot(122)
contour = ax2.contourf(X, Y, Z, levels=20, cmap='viridis')
plt.colorbar(contour, ax=ax2, label='概率密度')
ax2.set_xlabel('X₁')
ax2.set_ylabel('X₂')
ax2.set_title('二元正态分布 (等高线)')
ax2.axis('equal')

plt.tight_layout()
plt.show()

# 从分布中采样
samples = rv.rvs(500)

plt.figure(figsize=(6, 6))
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
plt.contour(X, Y, Z, levels=[0.01, 0.05, 0.1, 0.15], colors='red', alpha=0.7)
plt.xlabel('X₁')
plt.ylabel('X₂')
plt.title('二元正态分布采样')
plt.axis('equal')
plt.grid(True, alpha=0.3)
plt.show()
```

## 数字特征

### 期望与方差

📌 **期望**是随机变量的平均值，反映其集中趋势：

- 离散变量：$E[X] = \sum_i x_i P(X=x_i)$
- 连续变量：$E[X] = \int_{-\infty}^{\infty} x f(x) \, dx$

📌 **方差**衡量随机变量与其均值的偏离程度：

$$\text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2$$

### 协方差与相关系数

📌 **协方差**衡量两个随机变量的线性相关性：

$$\text{Cov}(X, Y) = E[(X - E[X])(Y - E[Y])]$$

📌 **相关系数**是标准化的协方差，取值在 $[-1, 1]$ 之间：

$$\rho_{XY} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$$

```python
# 协方差与相关系数示例
np.random.seed(42)

# 生成不同相关性的数据
n = 500

# 正相关
X1 = np.random.randn(n)
Y1 = X1 + np.random.randn(n) * 0.5

# 负相关
X2 = np.random.randn(n)
Y2 = -X2 + np.random.randn(n) * 0.5

# 不相关
X3 = np.random.randn(n)
Y3 = np.random.randn(n)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, X, Y, title in [
    (axes[0], X1, Y1, '正相关'),
    (axes[1], X2, Y2, '负相关'),
    (axes[2], X3, Y3, '不相关')
]:
    ax.scatter(X, Y, alpha=0.5)
    cov = np.cov(X, Y)[0, 1]
    corr = np.corrcoef(X, Y)[0, 1]
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'{title}\n协方差={cov:.2f}, 相关系数={corr:.2f}')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## 大数定律与中心极限定理

### 大数定律

📌 **大数定律**说明样本均值会随着样本量增大而收敛于总体均值：

$$\lim_{n \to \infty} P\left(\left|\frac{1}{n}\sum_{i=1}^n X_i - \mu\right| > \epsilon\right) = 0$$

💡 **直观理解**：抛硬币次数越多，正面朝上的频率越接近 0.5。

### 中心极限定理

📌 **中心极限定理**是统计学中最重要的定理之一：

$$\frac{\sum_{i=1}^n X_i - n\mu}{\sigma\sqrt{n}} \xrightarrow{d} N(0, 1)$$

即：独立同分布随机变量的**标准化和**近似服从标准正态分布，无论原始分布是什么。

```python
# 中心极限定理演示
np.random.seed(42)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))

# 原始分布：均匀分布
original_dist = '均匀分布 U(0, 1)'
sample_sizes = [5, 30, 100, 500, 1000, 5000]

for idx, n in enumerate(sample_sizes):
    row, col = idx // 3, idx % 3
    ax = axes[row, col]
    
    # 重复采样，计算样本均值
    sample_means = []
    for _ in range(10000):
        samples = np.random.uniform(0, 1, n)
        sample_means.append(np.mean(samples))
    
    # 绘制直方图
    ax.hist(sample_means, bins=50, density=True, alpha=0.7, color='steelblue')
    
    # 理论正态分布
    mu_theory = 0.5  # 均匀分布的均值
    sigma_theory = np.sqrt(1/12 / n)  # 标准误差
    x = np.linspace(mu_theory - 4*sigma_theory, mu_theory + 4*sigma_theory, 100)
    ax.plot(x, stats.norm.pdf(x, mu_theory, sigma_theory), 'r-', linewidth=2)
    
    ax.set_title(f'n = {n}')
    ax.set_xlabel('样本均值')
    ax.set_ylabel('密度')

plt.suptitle(f'中心极限定理演示\n原始分布: {original_dist}', fontsize=14)
plt.tight_layout()
plt.show()

print("💡 观察：随着样本量增大，样本均值的分布越来越接近正态分布")
```

## 统计推断

### 最大似然估计

📌 **最大似然估计（MLE）**是一种参数估计方法，选择使观测数据出现概率最大的参数值：

$$\hat{\theta}_{MLE} = \arg\max_{\theta} L(\theta|\mathbf{X}) = \arg\max_{\theta} \prod_{i=1}^n f(x_i|\theta)$$

通常使用对数似然更方便：

$$\hat{\theta}_{MLE} = \arg\max_{\theta} \sum_{i=1}^n \log f(x_i|\theta)$$

```python
# MLE 示例：估计正态分布参数
np.random.seed(42)

# 真实参数
true_mu = 3.0
true_sigma = 1.5
n_samples = 100

# 生成数据
data = np.random.normal(true_mu, true_sigma, n_samples)

# MLE 估计
# 正态分布的 MLE 解有闭式解：
mu_mle = np.mean(data)
sigma_mle = np.std(data, ddof=0)  # ddof=0 表示 MLE 版本

print("=" * 50)
print("正态分布参数的最大似然估计")
print("=" * 50)
print(f"真实均值: {true_mu:.4f}")
print(f"MLE 估计均值: {mu_mle:.4f}")
print(f"估计误差: {abs(mu_mle - true_mu):.4f}")
print()
print(f"真实标准差: {true_sigma:.4f}")
print(f"MLE 估计标准差: {sigma_mle:.4f}")
print(f"估计误差: {abs(sigma_mle - true_sigma):.4f}")

# 可视化似然函数
mu_range = np.linspace(1, 5, 100)
sigma_range = np.linspace(0.5, 3, 100)

def log_likelihood(mu, sigma):
    """计算对数似然"""
    return np.sum(stats.norm.logpdf(data, mu, sigma))

# 计算似然曲面
LL = np.zeros((len(sigma_range), len(mu_range)))
for i, sigma in enumerate(sigma_range):
    for j, mu in enumerate(mu_range):
        LL[i, j] = log_likelihood(mu, sigma)

plt.figure(figsize=(10, 6))
contour = plt.contourf(mu_range, sigma_range, LL, levels=30, cmap='viridis')
plt.colorbar(contour, label='对数似然')
plt.scatter([mu_mle], [sigma_mle], c='red', s=100, marker='*', label='MLE 估计')
plt.scatter([true_mu], [true_sigma], c='white', s=100, marker='o', label='真实值')
plt.xlabel('μ')
plt.ylabel('σ')
plt.title('对数似然函数')
plt.legend()
plt.tight_layout()
plt.show()
```

### 假设检验

📌 **假设检验**用于判断样本证据是否足以拒绝某个假设。

**基本概念**：
- **原假设 $H_0$**：待检验的假设（通常是"无差异"或"无效果"）
- **备择假设 $H_1$**：与原假设对立的假设
- **显著性水平 $\alpha$**：犯第一类错误（假阳性）的最大允许概率
- **p 值**：在原假设下，观察到当前或更极端结果的概率

```python
from scipy.stats import ttest_1samp, ttest_ind, chi2_contingency

# 示例 1：单样本 t 检验
np.random.seed(42)
sample = np.random.normal(loc=5.2, scale=1.5, size=50)

print("=" * 50)
print("单样本 t 检验")
print("=" * 50)
print(f"H₀: μ = 5.0 (总体均值等于 5)")
print(f"H₁: μ ≠ 5.0 (总体均值不等于 5)")
print(f"样本均值: {np.mean(sample):.4f}")

t_stat, p_value = ttest_1samp(sample, 5.0)
print(f"t 统计量: {t_stat:.4f}")
print(f"p 值: {p_value:.4f}")
print(f"结论 (α=0.05): {'拒绝 H₀' if p_value < 0.05 else '无法拒绝 H₀'}")

# 示例 2：双样本 t 检验
print("\n" + "=" * 50)
print("双样本 t 检验")
print("=" * 50)

group1 = np.random.normal(loc=100, scale=15, size=30)
group2 = np.random.normal(loc=110, scale=15, size=30)

print(f"组1均值: {np.mean(group1):.2f}")
print(f"组2均值: {np.mean(group2):.2f}")

t_stat, p_value = ttest_ind(group1, group2)
print(f"t 统计量: {t_stat:.4f}")
print(f"p 值: {p_value:.4f}")
print(f"结论 (α=0.05): {'两组均值显著不同' if p_value < 0.05 else '两组均值无显著差异'}")

# 可视化
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(group1, alpha=0.6, label=f'组1 (μ={np.mean(group1):.1f})')
plt.hist(group2, alpha=0.6, label=f'组2 (μ={np.mean(group2):.1f})')
plt.xlabel('值')
plt.ylabel('频数')
plt.title('两组数据分布')
plt.legend()

plt.subplot(1, 2, 2)
# 绘制 t 分布和检验统计量
x = np.linspace(-4, 4, 100)
plt.plot(x, stats.t.pdf(x, df=58), 'b-', label='t 分布 (df=58)')
plt.axvline(t_stat, color='r', linestyle='--', label=f't 统计量={t_stat:.2f}')
plt.axvline(-t_stat, color='r', linestyle='--')
plt.xlabel('t 值')
plt.ylabel('密度')
plt.title('t 检验可视化')
plt.legend()

plt.tight_layout()
plt.show()
```

## 信息论基础

信息论为机器学习提供了量化不确定性的工具。

### 信息熵

📌 **信息熵**衡量随机变量的不确定性：

$$H(X) = -\sum_{i=1}^n p(x_i) \log p(x_i)$$

💡 **直观理解**：熵越大，不确定性越高。均匀分布的熵最大。

### 交叉熵与 KL 散度

📌 **交叉熵**衡量用分布 Q 来编码分布 P 所需的平均比特数：

$$H(P, Q) = -\sum_{i=1}^n p(x_i) \log q(x_i)$$

📌 **KL 散度**衡量两个分布的差异：

$$D_{KL}(P \| Q) = \sum_{i=1}^n p(x_i) \log \frac{p(x_i)}{q(x_i)} = H(P, Q) - H(P)$$

```python
# 信息论示例
def entropy(p):
    """计算熵"""
    # 避免数值问题
    p = np.array(p)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

def cross_entropy(p, q):
    """计算交叉熵"""
    p, q = np.array(p), np.array(q)
    # 避免 log(0)
    mask = p > 0
    return -np.sum(p[mask] * np.log2(q[mask]))

def kl_divergence(p, q):
    """计算 KL 散度"""
    return cross_entropy(p, q) - entropy(p)

print("=" * 50)
print("信息熵与 KL 散度")
print("=" * 50)

# 不同分布的熵
distributions = {
    "均匀分布": [0.25, 0.25, 0.25, 0.25],
    "确定性分布": [1.0, 0.0, 0.0, 0.0],
    "不平衡分布": [0.7, 0.2, 0.08, 0.02],
}

for name, dist in distributions.items():
    print(f"{name}: H = {entropy(dist):.4f} bits")

print("\n💡 均匀分布的熵最大，确定性分布的熵为 0")

# KL 散度示例
p = [0.4, 0.3, 0.2, 0.1]  # 真实分布
q1 = [0.4, 0.3, 0.2, 0.1]  # 完全匹配
q2 = [0.25, 0.25, 0.25, 0.25]  # 均匀分布
q3 = [0.5, 0.3, 0.15, 0.05]  # 稍有偏差

print(f"\nKL(P || Q完全匹配) = {kl_divergence(p, q1):.4f}")
print(f"KL(P || Q均匀分布) = {kl_divergence(p, q2):.4f}")
print(f"KL(P || Q稍有偏差) = {kl_divergence(p, q3):.4f}")

# 可视化熵随概率变化
plt.figure(figsize=(12, 4))

# 二元分布的熵
plt.subplot(1, 3, 1)
p_values = np.linspace(0.001, 0.999, 100)
entropies = [entropy([p, 1-p]) for p in p_values]
plt.plot(p_values, entropies, 'b-', linewidth=2)
plt.axvline(x=0.5, color='r', linestyle='--', alpha=0.7, label='p=0.5')
plt.xlabel('p(X=1)')
plt.ylabel('熵 (bits)')
plt.title('二元分布的熵')
plt.legend()
plt.grid(True, alpha=0.3)

# 不同分布的可视化
plt.subplot(1, 3, 2)
x = ['类别1', '类别2', '类别3', '类别4']
width = 0.25
x_pos = np.arange(len(x))

for i, (name, dist) in enumerate(list(distributions.items())[:3]):
    plt.bar(x_pos + i*width, dist, width, label=name, alpha=0.7)

plt.xlabel('类别')
plt.ylabel('概率')
plt.title('不同概率分布')
plt.xticks(x_pos + width, x)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# KL 散度随 Q 变化
plt.subplot(1, 3, 3)
p_fixed = [0.4, 0.3, 0.2, 0.1]
kl_values = []
q_range = np.linspace(0.1, 0.9, 50)

for q1 in q_range:
    q = [q1, (1-q1)/3, (1-q1)/3, (1-q1)/3]
    kl_values.append(kl_divergence(p_fixed, q))

plt.plot(q_range, kl_values, 'g-', linewidth=2)
plt.xlabel('Q(类别1)')
plt.ylabel('KL(P || Q)')
plt.title('KL 散度随 Q 变化')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## 机器学习中的应用

### 朴素贝叶斯分类器

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# 加载数据
iris = load_iris()
X, y = iris.data, iris.target

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练高斯朴素贝叶斯
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# 预测
y_pred = gnb.predict(X_test)

print("=" * 50)
print("高斯朴素贝叶斯分类器")
print("=" * 50)
print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
print("\n混淆矩阵:")
print(confusion_matrix(y_test, y_pred))

# 查看学习到的参数
print("\n各类别的均值:")
for i, name in enumerate(iris.target_names):
    print(f"  {name}: {gnb.theta_[i]}")
```

### 最大后验估计（MAP）

📌 **最大后验估计**在 MLE 的基础上加入先验信息：

$$\hat{\theta}_{MAP} = \arg\max_{\theta} P(\theta|\mathbf{X}) = \arg\max_{\theta} P(\mathbf{X}|\theta) P(\theta)$$

```python
# MAP vs MLE 对比：带先验的估计
np.random.seed(42)

# 真实参数
true_p = 0.7
n_trials = 10

# 生成数据
data = np.random.binomial(1, true_p, n_trials)
successes = np.sum(data)

print("=" * 50)
print("MLE vs MAP 估计")
print("=" * 50)
print(f"真实概率: {true_p}")
print(f"试验次数: {n_trials}")
print(f"成功次数: {successes}")

# MLE 估计
p_mle = successes / n_trials
print(f"\nMLE 估计: {p_mle:.4f}")

# MAP 估计（Beta 先验，相当于 L2 正则化）
# 使用 Beta(α, β) 作为先验，相当于事先观察到 α-1 次成功和 β-1 次失败
alpha, beta = 2, 2  # 先验参数
p_map = (successes + alpha - 1) / (n_trials + alpha + beta - 2)
print(f"MAP 估计 (Beta先验 α={alpha}, β={beta}): {p_map:.4f}")

# 可视化先验和后验
x = np.linspace(0, 1, 100)

# 先验分布
prior = stats.beta.pdf(x, alpha, beta)

# 后验分布（解析解）
posterior = stats.beta.pdf(x, successes + alpha, n_trials - successes + beta)

plt.figure(figsize=(10, 5))
plt.plot(x, prior, 'b-', label=f'先验 Beta({alpha}, {beta})', linewidth=2)
plt.plot(x, posterior, 'r-', label=f'后验 Beta({successes + alpha}, {n_trials - successes + beta})', linewidth=2)
plt.axvline(p_mle, color='g', linestyle='--', label=f'MLE = {p_mle:.2f}')
plt.axvline(p_map, color='orange', linestyle='--', label=f'MAP = {p_map:.2f}')
plt.axvline(true_p, color='black', linestyle=':', label=f'真实值 = {true_p}')
plt.xlabel('p')
plt.ylabel('密度')
plt.title('MLE vs MAP 估计')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n💡 MAP 估计通过先验信息进行了正则化，避免极端估计")
```

## 常见问题与注意事项

1. **贝叶斯 vs 频率学派**：两者对概率的解释不同，但都有其适用场景
2. **小样本问题**：样本量小时，MLE 可能不可靠，考虑使用 MAP 或贝叶斯方法
3. **数值稳定性**：计算概率时使用对数空间避免下溢
4. **独立性假设**：朴素贝叶斯假设特征独立，实际应用中需验证

## 参考资料

- Christopher Bishop, *Pattern Recognition and Machine Learning*（经典教材）
- Larry Wasserman, *All of Statistics*
- 3Blue1Brown, *Bayes theorem*（视频）
