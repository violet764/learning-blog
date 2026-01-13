# 线性判别分析

## 概述

监督学习的降维方法，最大化类间方差，最小化类内方差。

## 数学原理

### 目标函数

LDA的目标是找到投影方向$\mathbf{w}$，使得类间散度最大，类内散度最小：

$J(\mathbf{w}) = \frac{\mathbf{w}^T\mathbf{S}_B\mathbf{w}}{\mathbf{w}^T\mathbf{S}_W\mathbf{w}}$

其中：
- $\mathbf{S}_B$：类间散度矩阵
- $\mathbf{S}_W$：类内散度矩阵

### 散度矩阵定义

**类间散度矩阵**：
$\mathbf{S}_B = \sum_{c=1}^{C} n_c(\boldsymbol{\mu}_c - \boldsymbol{\mu})(\boldsymbol{\mu}_c - \boldsymbol{\mu})^T$

**类内散度矩阵**：
$\mathbf{S}_W = \sum_{c=1}^{C} \sum_{\mathbf{x} \in D_c} (\mathbf{x} - \boldsymbol{\mu}_c)(\mathbf{x} - \boldsymbol{\mu}_c)^T$

其中：
- $C$：类别数
- $n_c$：第c类的样本数
- $\boldsymbol{\mu}_c$：第c类的均值向量
- $\boldsymbol{\mu}$：总体均值向量

## Python实现

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class MyLDA:
    """
    手动实现LDA
    """
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.eigenvalues_ = None
        self.eigenvectors_ = None
    
    def fit(self, X, y):
        """
        拟合LDA模型
        """
        n_samples, n_features = X.shape
        classes = np.unique(y)
        n_classes = len(classes)
        
        # 计算总体均值
        overall_mean = np.mean(X, axis=0)
        
        # 初始化散度矩阵
        S_W = np.zeros((n_features, n_features))
        S_B = np.zeros((n_features, n_features))
        
        # 计算类内散度矩阵和类间散度矩阵
        for c in classes:
            X_c = X[y == c]
            n_c = X_c.shape[0]
            mean_c = np.mean(X_c, axis=0)
            
            # 类内散度
            S_W += (X_c - mean_c).T @ (X_c - mean_c)
            
            # 类间散度
            mean_diff = (mean_c - overall_mean).reshape(-1, 1)
            S_B += n_c * (mean_diff @ mean_diff.T)
        
        # 求解广义特征值问题：S_B * w = λ * S_W * w
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.pinv(S_W) @ S_B)
        
        # 排序特征值和特征向量
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 选择主成分
        if self.n_components is not None:
            eigenvectors = eigenvectors[:, :self.n_components]
        
        self.eigenvalues_ = eigenvalues
        self.eigenvectors_ = eigenvectors
        
        return self
    
    def transform(self, X):
        """
        数据降维
        """
        return X @ self.eigenvectors_
    
    def fit_transform(self, X, y):
        """
        拟合并转换数据
        """
        self.fit(X, y)
        return self.transform(X)

def lda_demo():
    """
    LDA演示
    """
    # 生成示例数据
    X, y = make_classification(n_samples=200, n_features=4, n_informative=4,
                              n_redundant=0, n_classes=3, n_clusters_per_class=1,
                              random_state=42)
    
    # 使用手动实现的LDA
    my_lda = MyLDA(n_components=2)
    X_lda_manual = my_lda.fit_transform(X, y)
    
    # 使用sklearn的LDA
    sklearn_lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda_sklearn = sklearn_lda.fit_transform(X, y)
    
    # 可视化结果
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 手动实现结果
    for i, color in zip(np.unique(y), ['red', 'blue', 'green']):
        ax1.scatter(X_lda_manual[y == i, 0], X_lda_manual[y == i, 1],
                   c=color, label=f'Class {i}', alpha=0.7)
    ax1.set_title('Manual LDA Implementation')
    ax1.legend()
    ax1.grid(True)
    
    # sklearn实现结果
    for i, color in zip(np.unique(y), ['red', 'blue', 'green']):
        ax2.scatter(X_lda_sklearn[y == i, 0], X_lda_sklearn[y == i, 1],
                   c=color, label=f'Class {i}', alpha=0.7)
    ax2.set_title('Sklearn LDA Implementation')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 输出特征值信息
    print("特征值:", my_lda.eigenvalues_[:2])
    print("特征值解释方差比例:", my_lda.eigenvalues_[:2] / np.sum(my_lda.eigenvalues_))

if __name__ == "__main__":
    lda_demo()
```

## 数学推导

### 最优化问题求解

最大化目标函数$J(\mathbf{w})$等价于求解广义特征值问题：

$\mathbf{S}_B\mathbf{w} = \lambda\mathbf{S}_W\mathbf{w}$

或者等价于：

$(\mathbf{S}_W^{-1}\mathbf{S}_B)\mathbf{w} = \lambda\mathbf{w}$

### Fisher判别准则

对于两类问题，最优投影方向为：

$\mathbf{w} = \mathbf{S}_W^{-1}(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)$

### 多类扩展

对于多类问题，LDA最多可以提取C-1个判别方向。

## 与PCA的比较

| 特性 | PCA | LDA |
|------|-----|-----|
| 监督性 | 无监督 | 有监督 |
| 目标 | 最大化方差 | 最大化类间方差/类内方差 |
| 应用 | 数据压缩、去噪 | 分类预处理、特征提取 |
| 投影方向 | 数据方差最大的方向 | 类别分离最佳的方向 |

## 优缺点

### 优点
- 考虑了类别信息
- 能有效提高分类性能
- 计算效率高

### 缺点
- 需要类别标签
- 对离群点敏感
- 假设各类数据服从高斯分布

## 应用场景
- 人脸识别
- 生物信息学
- 文本分类
- 医学诊断