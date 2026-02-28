# EM算法

期望最大化算法（Expectation-Maximization, EM）是处理含有**隐变量**概率模型的经典参数估计方法。当数据存在缺失或模型包含不可观测变量时，传统的最大似然估计往往无法直接求解，EM算法提供了一种优雅的迭代解决方案。

## 基本概念

### 什么是隐变量？

隐变量（Latent Variable）是指**不可直接观测**的变量，但会影响可观测变量的概率分布。

**典型例子**：
- 📊 **高斯混合模型**：样本来自哪个高斯分量是未知的
- 🧬 **隐马尔可夫模型**：隐状态序列不可观测
- 📝 **主题模型**：文档的主题分布是隐含的
- 🩺 **医学诊断**：疾病的真实状态可能未知

### 为什么需要EM算法？

对于完全数据的似然函数 $P(X, Z|\theta)$，最大似然估计可以直接求解。但当存在隐变量 $Z$ 时，我们只能观测到 $X$，需要最大化**边缘似然**：

$$ P(X|\theta) = \sum_{Z} P(X, Z|\theta) $$

这个求和（或积分）使得对数似然难以直接优化：

$$ \log P(X|\theta) = \log \sum_{Z} P(X, Z|\theta) $$

对数里面套求和，无法分解到各个隐变量状态上，传统优化方法失效。

### EM算法的核心思想

💡 **核心洞察**：既然直接优化 $\log P(X|\theta)$ 困难，不如构造一个易于优化的**下界函数**，通过交替迭代逼近最优解。

EM算法通过两步迭代：
1. **E步（期望步）**：基于当前参数，计算隐变量的后验分布
2. **M步（最大化步）**：基于后验分布，更新模型参数

## 数学推导

### Jensen不等式与下界函数

对于对数似然：

$$ \log P(X|\theta) = \log \sum_{Z} P(X, Z|\theta) $$

引入隐变量的分布 $Q(Z)$，由 Jensen 不等式：

$$ \log \sum_{Z} Q(Z) \frac{P(X, Z|\theta)}{Q(Z)} \geq \sum_{Z} Q(Z) \log \frac{P(X, Z|\theta)}{Q(Z)} $$

定义 **ELBO（Evidence Lower Bound）**：

$$ \mathcal{L}(Q, \theta) = \sum_{Z} Q(Z) \log P(X, Z|\theta) - \sum_{Z} Q(Z) \log Q(Z) $$

可以证明：

$$ \log P(X|\theta) = \mathcal{L}(Q, \theta) + D_{KL}(Q \| P(Z|X, \theta)) $$

其中 $D_{KL}$ 是 KL 散度，始终非负。当 $Q(Z) = P(Z|X, \theta)$ 时，KL 散度为 0，下界紧致。

### E步：计算隐变量后验

给定当前参数 $\theta^{(t)}$，计算隐变量的后验分布：

$$ Q^{(t)}(Z) = P(Z|X, \theta^{(t)}) $$

这一步使得 ELBO 等于对数似然（下界紧致）。

### M步：最大化期望

固定 $Q^{(t)}(Z)$，更新参数：

$$ \theta^{(t+1)} = \arg\max_{\theta} \sum_{Z} Q^{(t)}(Z) \log P(X, Z|\theta) $$

这等价于最大化完整数据对数似然的期望：

$$ \theta^{(t+1)} = \arg\max_{\theta} \mathbb{E}_{Z|X, \theta^{(t)}}[\log P(X, Z|\theta)] $$

### 收敛性证明

每次迭代，对数似然单调不减：

$$ \log P(X|\theta^{(t+1)}) \geq \log P(X|\theta^{(t)}) $$

**证明思路**：
1. E步后：$\log P(X|\theta^{(t)}) = \mathcal{L}(Q^{(t)}, \theta^{(t)})$
2. M步：$\mathcal{L}(Q^{(t)}, \theta^{(t+1)}) \geq \mathcal{L}(Q^{(t)}, \theta^{(t)})$
3. 新的E步：$\log P(X|\theta^{(t+1)}) \geq \mathcal{L}(Q^{(t)}, \theta^{(t+1)})$

因此：$\log P(X|\theta^{(t+1)}) \geq \log P(X|\theta^{(t)})$

⚠️ **注意**：EM算法保证收敛到**局部最优**或**鞍点**，不一定是全局最优。

## 高斯混合模型（GMM）

### 问题设定

假设数据由 $K$ 个高斯分布混合生成：

$$ P(x|\theta) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x|\mu_k, \Sigma_k) $$

其中：
- $\pi_k$：第 $k$ 个分量的混合系数，$\sum_k \pi_k = 1$
- $\mu_k, \Sigma_k$：第 $k$ 个高斯分量的均值和协方差

隐变量 $z_i \in \{1, 2, \ldots, K\}$ 表示样本 $x_i$ 来自哪个分量。

### E步：计算后验概率

计算样本 $x_i$ 属于第 $k$ 个分量的后验概率（责任度）：

$$ \gamma_{ik} = P(z_i = k | x_i, \theta) = \frac{\pi_k \mathcal{N}(x_i|\mu_k, \Sigma_k)}{\sum_{j=1}^{K} \pi_j \mathcal{N}(x_i|\mu_j, \Sigma_j)} $$

### M步：更新参数

$$ N_k = \sum_{i=1}^{n} \gamma_{ik} \quad \text{（有效样本数）} $$

$$ \mu_k^{new} = \frac{1}{N_k} \sum_{i=1}^{n} \gamma_{ik} x_i $$

$$ \Sigma_k^{new} = \frac{1}{N_k} \sum_{i=1}^{n} \gamma_{ik} (x_i - \mu_k^{new})(x_i - \mu_k^{new})^T $$

$$ \pi_k^{new} = \frac{N_k}{n} $$

## 代码示例

### 从零实现EM算法求解GMM

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.datasets import make_blobs

class MyGMM:
    """从零实现高斯混合模型"""
    
    def __init__(self, n_components=3, max_iter=100, tol=1e-6, random_state=None):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        
    def _initialize(self, X):
        """初始化参数"""
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)
        
        # 随机选择初始均值
        idx = rng.choice(n_samples, self.n_components, replace=False)
        self.means_ = X[idx].copy()
        
        # 初始化协方差为单位矩阵
        self.covariances_ = np.array([
            np.eye(n_features) for _ in range(self.n_components)
        ])
        
        # 初始化混合系数均匀分布
        self.weights_ = np.ones(self.n_components) / self.n_components
        
        # 记录对数似然
        self.log_likelihood_history_ = []
        
    def _e_step(self, X):
        """E步：计算后验概率（责任度）"""
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            responsibilities[:, k] = self.weights_[k] * multivariate_normal.pdf(
                X, self.means_[k], self.covariances_[k]
            )
        
        # 归一化
        responsibilities = responsibilities / responsibilities.sum(axis=1, keepdims=True)
        
        # 计算对数似然
        log_likelihood = np.sum(np.log(
            responsibilities.sum(axis=1) + 1e-300
        ))
        
        return responsibilities, log_likelihood
    
    def _m_step(self, X, responsibilities):
        """M步：更新参数"""
        n_samples, n_features = X.shape
        
        # 有效样本数
        N_k = responsibilities.sum(axis=0)
        
        # 更新混合系数
        self.weights_ = N_k / n_samples
        
        # 更新均值
        self.means_ = (responsibilities.T @ X) / N_k[:, np.newaxis]
        
        # 更新协方差
        for k in range(self.n_components):
            diff = X - self.means_[k]
            weighted_diff = responsibilities[:, k:k+1] * diff
            self.covariances_[k] = (weighted_diff.T @ diff) / N_k[k]
            # 添加正则化避免奇异矩阵
            self.covariances_[k] += 1e-6 * np.eye(n_features)
    
    def fit(self, X):
        """训练模型"""
        self._initialize(X)
        
        prev_log_likelihood = -np.inf
        
        for iteration in range(self.max_iter):
            # E步
            responsibilities, log_likelihood = self._e_step(X)
            self.log_likelihood_history_.append(log_likelihood)
            
            # 检查收敛
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                print(f"EM算法在第 {iteration + 1} 次迭代后收敛")
                break
            
            # M步
            self._m_step(X, responsibilities)
            
            prev_log_likelihood = log_likelihood
            
            if iteration == self.max_iter - 1:
                print(f"达到最大迭代次数 {self.max_iter}")
        
        return self
    
    def predict(self, X):
        """预测样本所属簇"""
        responsibilities, _ = self._e_step(X)
        return np.argmax(responsibilities, axis=1)
    
    def predict_proba(self, X):
        """预测样本属于各簇的概率"""
        responsibilities, _ = self._e_step(X)
        return responsibilities


# 生成模拟数据
np.random.seed(42)
X, y_true = make_blobs(
    n_samples=500, centers=3, cluster_std=[1.0, 1.5, 0.5],
    random_state=42
)

# 训练GMM模型
print("=" * 50)
print("高斯混合模型 - EM算法实现")
print("=" * 50)

gmm = MyGMM(n_components=3, max_iter=100, random_state=42)
gmm.fit(X)

print(f"\n学习到的参数:")
print(f"混合系数: {gmm.weights_}")
print(f"均值:\n{gmm.means_}")

# 预测
y_pred = gmm.predict(X)

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 真实标签
ax1 = axes[0, 0]
scatter1 = ax1.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.6)
ax1.set_title('真实标签')
ax1.set_xlabel('特征1')
ax1.set_ylabel('特征2')
plt.colorbar(scatter1, ax=ax1)

# 预测标签
ax2 = axes[0, 1]
scatter2 = ax2.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.6)
ax2.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', marker='x', s=200, linewidths=3)
ax2.set_title('GMM预测结果（红色×为聚类中心）')
ax2.set_xlabel('特征1')
ax2.set_ylabel('特征2')
plt.colorbar(scatter2, ax=ax2)

# 对数似然收敛曲线
ax3 = axes[1, 0]
ax3.plot(gmm.log_likelihood_history_, 'b-o')
ax3.set_xlabel('迭代次数')
ax3.set_ylabel('对数似然')
ax3.set_title('EM算法收敛过程')
ax3.grid(True)

# 软聚类概率可视化
ax4 = axes[1, 1]
probabilities = gmm.predict_proba(X)
# 使用最大概率表示置信度
max_probs = probabilities.max(axis=1)
scatter4 = ax4.scatter(X[:, 0], X[:, 1], c=max_probs, cmap='coolwarm', alpha=0.6)
ax4.set_title('预测置信度（颜色越亮表示越确信）')
ax4.set_xlabel('特征1')
ax4.set_ylabel('特征2')
plt.colorbar(scatter4, ax=ax4, label='最大概率')

plt.tight_layout()
plt.show()
```

### 使用sklearn的GMM

```python
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score

print("=" * 50)
print("sklearn GaussianMixture 使用示例")
print("=" * 50)

# sklearn实现
gmm_sklearn = GaussianMixture(
    n_components=3, 
    covariance_type='full',
    max_iter=100,
    random_state=42
)
gmm_sklearn.fit(X)
y_pred_sklearn = gmm_sklearn.predict(X)

print(f"收敛: {'是' if gmm_sklearn.converged_ else '否'}")
print(f"迭代次数: {gmm_sklearn.n_iter_}")
print(f"对数似然: {gmm_sklearn.lower_bound_:.2f}")
print(f"ARI分数: {adjusted_rand_score(y_true, y_pred_sklearn):.4f}")
print(f"轮廓系数: {silhouette_score(X, y_pred_sklearn):.4f}")

# 不同协方差类型比较
print("\n不同协方差类型比较:")
cov_types = ['full', 'tied', 'diag', 'spherical']

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, cov_type in zip(axes.flat, cov_types):
    gmm_temp = GaussianMixture(n_components=3, covariance_type=cov_type, random_state=42)
    gmm_temp.fit(X)
    y_temp = gmm_temp.predict(X)
    
    ax.scatter(X[:, 0], X[:, 1], c=y_temp, cmap='viridis', alpha=0.6)
    ax.scatter(gmm_temp.means_[:, 0], gmm_temp.means_[:, 1], 
               c='red', marker='x', s=200, linewidths=3)
    ax.set_title(f"协方差类型: {cov_type}\nBIC: {gmm_temp.bic(X):.1f}")
    ax.set_xlabel('特征1')
    ax.set_ylabel('特征2')

plt.tight_layout()
plt.show()
```

### 模型选择：确定最佳分量数

```python
print("=" * 50)
print("使用BIC/AIC选择最佳分量数")
print("=" * 50)

n_components_range = range(1, 11)
bics = []
aics = []

for n in n_components_range:
    gmm_temp = GaussianMixture(n_components=n, random_state=42)
    gmm_temp.fit(X)
    bics.append(gmm_temp.bic(X))
    aics.append(gmm_temp.aic(X))

best_n_bic = n_components_range[np.argmin(bics)]
best_n_aic = n_components_range[np.argmin(aics)]

print(f"BIC选择的最佳分量数: {best_n_bic}")
print(f"AIC选择的最佳分量数: {best_n_aic}")

plt.figure(figsize=(10, 5))
plt.plot(n_components_range, bics, 'b-o', label='BIC')
plt.plot(n_components_range, aics, 'r-o', label='AIC')
plt.axvline(best_n_bic, color='b', linestyle='--', alpha=0.5, label=f'BIC最佳: {best_n_bic}')
plt.axvline(best_n_aic, color='r', linestyle='--', alpha=0.5, label=f'AIC最佳: {best_n_aic}')
plt.xlabel('分量数')
plt.ylabel('信息准则值')
plt.title('BIC/AIC模型选择')
plt.legend()
plt.grid(True)
plt.show()
```

### 应用：异常检测

```python
print("=" * 50)
print("GMM用于异常检测")
print("=" * 50)

# 训练GMM（只使用正常数据）
gmm_ad = GaussianMixture(n_components=3, random_state=42)
gmm_ad.fit(X)

# 计算每个样本的对数概率
log_probs = gmm_ad.score_samples(X)

# 设置阈值（例如，低于5%分位数认为是异常）
threshold = np.percentile(log_probs, 5)

# 生成一些异常点
np.random.seed(42)
X_outliers = np.random.uniform(
    low=X.min(axis=0) - 3,
    high=X.max(axis=0) + 3,
    size=(50, 2)
)

# 预测异常点
log_probs_outliers = gmm_ad.score_samples(X_outliers)
is_outlier = log_probs_outliers < threshold

print(f"异常检测阈值: {threshold:.2f}")
print(f"检测到的异常点数: {is_outlier.sum()}/{len(X_outliers)}")

plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c='blue', alpha=0.5, label='正常数据')
plt.scatter(X_outliers[:, 0], X_outliers[:, 1], 
            c=['red' if o else 'green' for o in is_outlier],
            marker='x', s=100, label='测试点（红=异常）')
plt.xlabel('特征1')
plt.ylabel('特征2')
plt.title('GMM异常检测')
plt.legend()
plt.grid(True)
plt.show()
```

## EM算法的其他应用

### 隐马尔可夫模型（HMM）

HMM使用**Baum-Welch算法**（EM的特殊形式）学习转移概率和发射概率。

### 缺失数据填补

EM可以自然地处理缺失数据问题：
- E步：用期望值填补缺失数据
- M步：基于完整数据更新参数

### 因子分析

EM用于估计因子载荷矩阵和潜在因子。

## 实践建议

### 初始化策略

- **K-means初始化**：用K-means结果初始化GMM均值
- **多次随机初始化**：选择似然最高的结果
- **分层初始化**：逐步增加分量数

### 收敛判断

- 对数似然变化小于阈值
- 参数变化小于阈值
- 达到最大迭代次数

### 常见问题

| 问题 | 现象 | 解决方案 |
|------|------|----------|
| 奇异协方差 | 某分量坍缩到单点 | 添加正则化，限制最小方差 |
| 空分量 | 某分量无样本 | 删除该分量或重新初始化 |
| 局部最优 | 不同初始化结果差异大 | 多次运行取最优 |
| 收敛慢 | 迭代次数过多 | 改善初始化，调整学习率 |

---

[上一节：朴素贝叶斯](./naive-bayes.md) | [下一节：高斯过程](./gaussian-processes.md)
