# 特征变换

## 概述

特征变换是通过数学函数将原始特征映射到新的特征空间，旨在改善数据分布、增强模型表达能力、处理非线性关系。

## 数学基础

### 函数变换理论

**单调变换**：保持特征顺序关系的变换，如对数变换、平方根变换

**非线性变换**：引入非线性关系的变换，如多项式变换、核方法

### 特征空间理论

原始特征空间X通过变换函数φ映射到新特征空间：
$$\phi: X \rightarrow \mathbb{R}^d$$

## 数值特征变换

### 非线性变换方法

#### 1. 对数变换
**数学原理**：$x' = \log(x + c)$，其中c为常数避免log(0)

**适用场景**：
- 右偏分布数据
- 方差与均值相关的数据
- 乘法关系的数据

**Python实现**：
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def log_transformation_analysis(X):
    """对数变换分析与实现"""
    
    # 不同基数的对数变换
    log_transforms = {
        'log2': np.log2(X + 1),
        'log10': np.log10(X + 1),
        'natural_log': np.log(X + 1)
    }
    
    # 可视化变换效果
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 原始数据分布
    axes[0, 0].hist(X, bins=50, alpha=0.7)
    axes[0, 0].set_title('原始数据分布')
    axes[0, 0].set_xlabel('数值')
    axes[0, 0].set_ylabel('频数')
    
    # 不同对数变换效果
    for idx, (name, transformed) in enumerate(log_transforms.items()):
        row, col = (idx + 1) // 2, (idx + 1) % 2
        axes[row, col].hist(transformed, bins=50, alpha=0.7)
        axes[row, col].set_title(f'{name}变换')
        axes[row, col].set_xlabel('变换后数值')
    
    plt.tight_layout()
    plt.show()
    
    # 统计特性比较
    stats_comparison = []
    stats_comparison.append({
        'Method': 'Original',
        'Mean': np.mean(X),
        'Std': np.std(X),
        'Skewness': stats.skew(X),
        'Kurtosis': stats.kurtosis(X)
    })
    
    for name, transformed in log_transforms.items():
        stats_comparison.append({
            'Method': name,
            'Mean': np.mean(transformed),
            'Std': np.std(transformed),
            'Skewness': stats.skew(transformed),
            'Kurtosis': stats.kurtosis(transformed)
        })
    
    return pd.DataFrame(stats_comparison)
```

#### 2. 幂变换
**Box-Cox变换**：
$$x^{(\lambda)} = \begin{cases}
\frac{x^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0 \\
\log(x) & \text{if } \lambda = 0
\end{cases}$$

**Yeo-Johnson变换**：适用于包含零和负数的数据

**Python实现**：
```python
from sklearn.preprocessing import PowerTransformer

def power_transformation_demo(X):
    """幂变换演示"""
    
    # Box-Cox变换（要求数据为正）
    pt_boxcox = PowerTransformer(method='box-cox')
    X_boxcox = pt_boxcox.fit_transform(X[X > 0].reshape(-1, 1))
    
    # Yeo-Johnson变换
    pt_yeojohnson = PowerTransformer(method='yeo-johnson')
    X_yeojohnson = pt_yeojohnson.fit_transform(X.reshape(-1, 1))
    
    # 可视化比较
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].hist(X, bins=50, alpha=0.7)
    axes[0].set_title('原始数据')
    
    axes[1].hist(X_boxcox, bins=50, alpha=0.7)
    axes[1].set_title('Box-Cox变换')
    
    axes[2].hist(X_yeojohnson, bins=50, alpha=0.7)
    axes[2].set_title('Yeo-Johnson变换')
    
    plt.tight_layout()
    plt.show()
    
    # 最佳λ值
    print(f"Box-Cox最佳λ: {pt_boxcox.lambdas_[0]:.3f}")
    print(f"Yeo-Johnson最佳λ: {pt_yeojohnson.lambdas_[0]:.3f}")
```

#### 3. 分位数变换
**原理**：将特征分布映射到指定分布（通常是正态分布）

**数学公式**：
$$x' = F^{-1}(G(x))$$
其中F是目标分布的CDF，G是原始分布的CDF

### 多项式特征与交互项

#### 多项式特征生成
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

def polynomial_feature_analysis(X, y, degree=3):
    """多项式特征分析"""
    
    # 生成多项式特征
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    # 特征名称
    feature_names = poly.get_feature_names_out(['x1', 'x2'])  # 假设2个特征
    
    # 构建多项式回归模型
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # 特征重要性分析
    feature_importance = np.abs(model.coef_)
    
    # 可视化特征重要性
    plt.figure(figsize=(12, 6))
    indices = np.argsort(feature_importance)[::-1]
    
    plt.bar(range(len(indices)), feature_importance[indices])
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
    plt.title('多项式特征重要性')
    plt.tight_layout()
    plt.show()
    
    return X_poly, feature_names, feature_importance
```

#### 交互特征
**数学原理**：捕捉特征间的相互作用

**常用交互形式**：
- 乘积交互：$x_i \times x_j$
- 比率交互：$\frac{x_i}{x_j}$
- 差异交互：$|x_i - x_j|$

## 核方法特征变换

### 核函数理论

#### 核技巧原理
通过核函数隐式地将数据映射到高维特征空间：
$$K(x, x') = \langle \phi(x), \phi(x') \rangle$$

#### 常用核函数

**1. 多项式核**：
$$K(x, x') = (x \cdot x' + c)^d$$

**2. 高斯核（RBF）**：
$$K(x, x') = \exp\left(-\frac{\|x - x'\|^2}{2\sigma^2}\right)$$

**3. Sigmoid核**：
$$K(x, x') = \tanh(\gamma x \cdot x' + r)$$

### 核主成分分析（Kernel PCA）

```python
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_circles

def kernel_pca_demo():
    """核PCA演示"""
    
    # 生成非线性可分数据
    X, y = make_circles(n_samples=400, factor=0.3, noise=0.05, random_state=42)
    
    # 不同核函数的KPCA
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for idx, kernel in enumerate(kernels):
        kpca = KernelPCA(kernel=kernel, n_components=2, gamma=10)
        X_kpca = kpca.fit_transform(X)
        
        scatter = axes[idx].scatter(X_kpca[:, 0], X_kpca[:, 1], c=y, cmap='viridis')
        axes[idx].set_title(f'KPCA with {kernel} kernel')
        axes[idx].set_xlabel('First Principal Component')
        axes[idx].set_ylabel('Second Principal Component')
        plt.colorbar(scatter, ax=axes[idx])
    
    plt.tight_layout()
    plt.show()
    
    # 核函数选择分析
    return compare_kernel_performance(X, y)

def compare_kernel_performance(X, y):
    """比较不同核函数性能"""
    
    from sklearn.svm import SVC
    from sklearn.model_selection import cross_val_score
    
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    results = []
    
    for kernel in kernels:
        # 使用KPCA进行特征变换
        kpca = KernelPCA(kernel=kernel, n_components=10)
        X_transformed = kpca.fit_transform(X)
        
        # SVM分类性能
        svm = SVC(kernel='linear')
        scores = cross_val_score(svm, X_transformed, y, cv=5)
        
        results.append({
            'kernel': kernel,
            'mean_accuracy': np.mean(scores),
            'std_accuracy': np.std(scores),
            'n_components': X_transformed.shape[1]
        })
    
    return pd.DataFrame(results)
```

## 离散化与分箱技术

### 分箱方法分类

#### 1. 等宽分箱
**原理**：将值域等分为k个区间

**数学公式**：
$$bin_i = [min + \frac{i}{k}(max-min), min + \frac{i+1}{k}(max-min)]$$

#### 2. 等频分箱
**原理**：每个分箱包含大致相同数量的样本

**实现方法**：基于分位数进行划分

#### 3. 基于聚类的分箱
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import KBinsDiscretizer

def advanced_binning_methods(X, n_bins=5):
    """高级分箱方法"""
    
    # 1. 等宽分箱
    equal_width = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
    X_width = equal_width.fit_transform(X.reshape(-1, 1))
    
    # 2. 等频分箱
    equal_freq = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    X_freq = equal_freq.fit_transform(X.reshape(-1, 1))
    
    # 3. K-means分箱
    kmeans = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='kmeans')
    X_kmeans = kmeans.fit_transform(X.reshape(-1, 1))
    
    # 可视化比较
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].hist(X, bins=50, alpha=0.7)
    axes[0, 0].set_title('原始数据')
    
    axes[0, 1].hist(X_width, bins=n_bins, alpha=0.7)
    axes[0, 1].set_title('等宽分箱')
    
    axes[1, 0].hist(X_freq, bins=n_bins, alpha=0.7)
    axes[1, 0].set_title('等频分箱')
    
    axes[1, 1].hist(X_kmeans, bins=n_bins, alpha=0.7)
    axes[1, 1].set_title('K-means分箱')
    
    plt.tight_layout()
    plt.show()
    
    # 分箱边界分析
    bin_edges = {
        'equal_width': equal_width.bin_edges_[0],
        'equal_freq': equal_freq.bin_edges_[0],
        'kmeans': kmeans.bin_edges_[0]
    }
    
    return bin_edges
```

### 最优分箱数量确定

#### 信息增益最大化
```python
def optimal_binning_by_information_gain(X, y, max_bins=20):
    """基于信息增益的最优分箱"""
    
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import mutual_info_score
    
    information_gains = []
    
    for n_bins in range(2, max_bins + 1):
        # 分箱处理
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
        X_binned = discretizer.fit_transform(X.reshape(-1, 1)).ravel()
        
        # 计算信息增益
        mi_score = mutual_info_score(X_binned, y)
        information_gains.append(mi_score)
    
    # 找到最优分箱数
    optimal_bins = np.argmax(information_gains) + 2
    
    plt.plot(range(2, max_bins + 1), information_gains, 'bo-', linewidth=2)
    plt.axvline(x=optimal_bins, color='red', linestyle='--', label=f'最优分箱数: {optimal_bins}')
    plt.xlabel('分箱数量')
    plt.ylabel('互信息')
    plt.title('信息增益 vs 分箱数量')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return optimal_bins, information_gains
```

## 高级特征变换技术

### 小波变换

#### 连续小波变换（CWT）
$$W(a,b) = \frac{1}{\sqrt{|a|}} \int_{-\infty}^{\infty} f(t) \psi\left(\frac{t-b}{a}\right) dt$$

#### 离散小波变换（DWT）
```python
import pywt

def wavelet_feature_extraction(signal):
    """小波特征提取"""
    
    # 小波基函数选择
    wavelet_families = ['db4', 'sym5', 'coif3']
    
    features = {}
    
    for wavelet in wavelet_families:
        # 多级小波分解
        coeffs = pywt.wavedec(signal, wavelet, level=4)
        
        # 提取小波系数特征
        wavelet_features = {
            f'{wavelet}_energy': np.sum([np.sum(c**2) for c in coeffs]),
            f'{wavelet}_entropy': -np.sum([np.sum(c**2 * np.log(c**2 + 1e-8)) for c in coeffs]),
            f'{wavelet}_std': np.std([np.std(c) for c in coeffs])
        }
        
        features.update(wavelet_features)
    
    return features
```

### 傅里叶变换

#### 快速傅里叶变换（FFT）
```python
def frequency_domain_features(signal, sampling_rate=1000):
    """频域特征提取"""
    
    # FFT变换
    fft_result = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(signal), 1/sampling_rate)
    
    # 功率谱密度
    power_spectrum = np.abs(fft_result)**2
    
    # 频域特征
    features = {
        'dominant_frequency': frequencies[np.argmax(power_spectrum)],
        'spectral_centroid': np.sum(frequencies * power_spectrum) / np.sum(power_spectrum),
        'spectral_bandwidth': np.sqrt(np.sum((frequencies - features['spectral_centroid'])**2 * power_spectrum) / np.sum(power_spectrum)),
        'spectral_entropy': -np.sum((power_spectrum/np.sum(power_spectrum)) * np.log(power_spectrum/np.sum(power_spectrum) + 1e-8))
    }
    
    return features
```

## 特征变换的数学理论

### 函数逼近理论

#### 多项式逼近（Weierstrass定理）
任何连续函数都可以用多项式函数一致逼近。

#### 核方法的理论基础
**Mercer定理**：正定核函数对应一个特征空间的內积。

**表示定理**：在再生核希尔伯特空间（RKHS）中，优化问题的解可以表示为核函数的线性组合。

### 信息论视角

#### 特征变换的信息保持
**数据处理不等式**：
$$I(X;Y) \geq I(\phi(X);Y)$$
特征变换不会增加关于目标变量的信息。

#### 最优特征变换
寻找变换函数φ使得互信息最大化：
$$\phi^* = \arg\max_{\phi} I(\phi(X);Y)$$

## 实际应用案例

### 金融时间序列特征工程

```python
def financial_feature_engineering(price_series, volume_series):
    """金融时间序列特征工程"""
    
    features = {}
    
    # 价格相关特征
    returns = np.diff(np.log(price_series))
    
    features.update({
        'log_returns_mean': np.mean(returns),
        'log_returns_volatility': np.std(returns),
        'log_returns_skewness': stats.skew(returns),
        'log_returns_kurtosis': stats.kurtosis(returns)
    })
    
    # 技术指标特征
    # RSI（相对强弱指数）
    def calculate_rsi(prices, window=14):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.convolve(gain, np.ones(window)/window, mode='valid')
        avg_loss = np.convolve(loss, np.ones(window)/window, mode='valid')
        
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    features['rsi'] = calculate_rsi(price_series)[-1] if len(price_series) > 14 else 50
    
    # 成交量特征
    volume_features = frequency_domain_features(volume_series)
    features.update({f'volume_{k}': v for k, v in volume_features.items()})
    
    return features
```

### 图像特征变换

```python
def image_feature_transformation(image):
    """图像特征变换"""
    
    from skimage import filters, feature, transform
    
    features = {}
    
    # 边缘检测特征
    edges_sobel = filters.sobel(image)
    edges_canny = feature.canny(image)
    
    features.update({
        'sobel_energy': np.sum(edges_sobel**2),
        'canny_edge_density': np.mean(edges_canny)
    })
    
    # 纹理特征（LBP）
    lbp = feature.local_binary_pattern(image, 8, 1, method='uniform')
    features['lbp_entropy'] = stats.entropy(np.histogram(lbp, bins=10)[0])
    
    # 频域特征
    fft_image = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft_image)
    magnitude_spectrum = 20 * np.log(np.abs(fft_shift) + 1)
    
    features['spectral_energy'] = np.sum(magnitude_spectrum**2)
    
    return features
```

## 性能评估与最佳实践

### 变换方法评估框架

```python
def evaluate_transformation_methods(X, y, transformations):
    """特征变换方法评估"""
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    
    results = []
    
    for name, transformer in transformations.items():
        # 应用变换
        if hasattr(transformer, 'fit_transform'):
            X_transformed = transformer.fit_transform(X, y)
        else:
            X_transformed = transformer.fit_transform(X)
        
        # 模型性能评估
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        scores = cross_val_score(model, X_transformed, y, cv=5, scoring='accuracy')
        
        # 特征相关性分析
        if hasattr(X_transformed, 'shape'):
            n_features = X_transformed.shape[1]
        else:
            n_features = len(X_transformed)
        
        results.append({
            'transformation': name,
            'mean_accuracy': np.mean(scores),
            'std_accuracy': np.std(scores),
            'n_features': n_features,
            'feature_correlation': np.mean(np.corrcoef(X_transformed.T))
        })
    
    return pd.DataFrame(results)
```

### 最佳实践指南

1. **数据分布分析**：变换前先分析数据分布特性
2. **变换组合**：尝试多种变换方法的组合
3. **领域知识**：结合业务理解选择适当的变换
4. **稳定性验证**：通过交叉验证评估变换效果的稳定性
5. **可解释性**：选择易于解释的变换方法

## 总结与展望

特征变换是特征工程的核心环节，通过合适的数学变换可以：
- 改善数据分布特性
- 增强模型表达能力
- 处理非线性关系
- 提取更有信息量的特征

未来的发展方向包括：
- 自动化特征变换选择
- 深度学习驱动的特征学习
- 多模态数据融合变换
- 可解释性特征变换方法

选择合适的特征变换策略需要结合数据特性、模型需求和业务目标进行综合考量。