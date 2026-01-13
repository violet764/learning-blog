# 统计检验

# 统计检验

## 概述与数学基础

统计检验是评估模型性能、比较不同模型和验证假设的数学方法。在机器学习中，这些方法提供了理论保证，帮助我们区分真实的性能差异和随机波动。

### 概率论基础
设$X$为随机变量，$f(x)$为其概率密度函数，则：

**期望值**：$E[X] = \int_{-\infty}^{\infty} x f(x) dx$

**方差**：$Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2$

**协方差**：$Cov(X,Y) = E[(X - E[X])(Y - E[Y])]$

### 抽样分布理论
中心极限定理指出，当样本量足够大时，样本均值的分布近似正态分布：

$\bar{X} \sim N(\mu, \frac{\sigma^2}{n})$

其中$\mu$为总体均值，$\sigma^2$为总体方差，$n$为样本大小。

## 基础概念

### 假设检验
- **零假设（H₀）**：默认假设，通常表示无差异或无效应
- **备择假设（H₁）**：研究假设，表示有差异或有效应
- **显著性水平（α）**：拒绝零假设的阈值，通常为0.05
- **p值**：观察到的数据支持零假设的概率

### 错误类型
- **第一类错误（α错误）**：错误拒绝真零假设
- **第二类错误（β错误）**：错误接受假零假设
- **检验功效（1-β）**：正确拒绝假零假设的概率

## 常用检验方法

## t检验家族：理论与应用

### 单样本t检验
**数学推导**：检验样本均值是否等于特定值，基于t分布理论。

设$X_1, X_2, \dots, X_n$为独立同分布的随机变量，来自正态分布$N(\mu, \sigma^2)$。

**检验统计量**：

$t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}} \sim t(n-1)$

其中：
- $\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$为样本均值
- $s^2 = \frac{1}{n-1}\sum_{i=1}^n (x_i - \bar{x})^2$为样本方差
- $\mu_0$为假设的总体均值
- 自由度$df = n-1$

**拒绝域**：$|t| > t_{\alpha/2}(n-1)$，其中$t_{\alpha/2}(n-1)$为t分布的上$\alpha/2$分位数。

### 独立样本t检验
**数学基础**：检验两个独立样本的均值差异，假设方差齐性。

设$X_{11}, \dots, X_{1n_1} \sim N(\mu_1, \sigma^2)$，$X_{21}, \dots, X_{2n_2} \sim N(\mu_2, \sigma^2)$。

**检验统计量**：

$t = \frac{\bar{x}_1 - \bar{x}_2}{s_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}} \sim t(n_1 + n_2 - 2)$

其中合并标准差$s_p^2 = \frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1 + n_2 - 2}$

**Welch's t检验（方差不齐时）**：

$t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}}$

近似自由度$df \approx \frac{(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2})^2}{\frac{(s_1^2/n_1)^2}{n_1-1} + \frac{(s_2^2/n_2)^2}{n_2-1}}$

### 配对样本t检验
**数学原理**：检验配对样本的均值差异，消除个体间变异。

设配对差异$D_i = X_{i1} - X_{i2} \sim N(\mu_d, \sigma_d^2)$，$i = 1, \dots, n$。

**检验统计量**：

$t = \frac{\bar{d}}{s_d / \sqrt{n}} \sim t(n-1)$

其中$\bar{d} = \frac{1}{n}\sum_{i=1}^n d_i$，$s_d^2 = \frac{1}{n-1}\sum_{i=1}^n (d_i - \bar{d})^2$

**优势**：配对设计提高了检验功效，因为消除了个体间变异的影响。

## 方差分析（ANOVA）：数学模型与分解

### 单因素ANOVA
**数学模型**：检验多个独立组的均值差异，基于方差分解理论。

设$Y_{ij}$为第$i$组第$j$个观测值，$i = 1, \dots, k$组，$j = 1, \dots, n_i$个观测。

**线性模型**：$Y_{ij} = \mu + \alpha_i + \epsilon_{ij}$

其中：
- $\mu$为总体均值
- $\alpha_i$为第$i$组的效应，满足$\sum_{i=1}^k \alpha_i = 0$
- $\epsilon_{ij} \sim N(0, \sigma^2)$为随机误差

**方差分解**：
总平方和$SST = \sum_{i=1}^k \sum_{j=1}^{n_i} (Y_{ij} - \bar{Y}_{\cdot\cdot})^2$
组间平方和$SSB = \sum_{i=1}^k n_i (\bar{Y}_{i\cdot} - \bar{Y}_{\cdot\cdot})^2$
组内平方和$SSW = \sum_{i=1}^k \sum_{j=1}^{n_i} (Y_{ij} - \bar{Y}_{i\cdot})^2$

**检验统计量**：
$F = \frac{MSB}{MSW} = \frac{SSB/(k-1)}{SSW/(N-k)} \sim F(k-1, N-k)$

其中$N = \sum_{i=1}^k n_i$为总样本量。

### 多因素ANOVA
**数学模型**：检验多个因素及其交互作用对结果的影响。

设两因素模型：$Y_{ijk} = \mu + \alpha_i + \beta_j + (\alpha\beta)_{ij} + \epsilon_{ijk}$

其中：
- $\alpha_i$为因素A第$i$水平的效应
- $\beta_j$为因素B第$j$水平的效应
- $(\alpha\beta)_{ij}$为交互作用效应

**方差分解表**：
```
来源         | 自由度 | 平方和 | 均方 | F统计量
因素A        | a-1    | SSA    | MSA  | MSA/MSE
因素B        | b-1    | SSB    | MSB  | MSB/MSE
交互作用AB   | (a-1)(b-1) | SSAB | MSAB | MSAB/MSE
误差         | ab(n-1) | SSE   | MSE  | -
总计         | abn-1  | SST    | -    | -
```

**多重比较校正**：
- Tukey HSD：控制族错误率（Family-Wise Error Rate）
- Bonferroni校正：保守但简单
- Holm-Bonferroni：改进的逐步校正方法

### 卡方检验

#### 独立性检验
检验两个分类变量是否独立：

$\chi^2 = \sum \frac{(O - E)^2}{E}$

其中O是观察频数，E是期望频数。

#### 拟合优度检验
检验观察分布是否符合预期分布。

## Python实现：理论与实践结合

## Python实现：理论与实践结合

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind, ttest_rel, f_oneway, chi2_contingency, norm, t, f, chi2
from statsmodels.stats.power import TTestIndPower, FTestAnovaPower
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.proportion import proportions_ztest
from scipy.optimize import minimize
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子确保结果可复现
np.random.seed(42)

def t_test_examples():
    """
    t检验示例：理论与实现
    """
    print("=== t检验示例：理论与实践 ===")
    
    # 生成示例数据
    np.random.seed(42)
    
    # 单样本t检验：检验均值是否为0
    sample1 = np.random.normal(loc=0.5, scale=1, size=100)
    t_stat, p_value = stats.ttest_1samp(sample1, 0)
    
    # 手动计算验证
    sample_mean = np.mean(sample1)
    sample_std = np.std(sample1, ddof=1)  # 无偏估计
    n = len(sample1)
    manual_t = (sample_mean - 0) / (sample_std / np.sqrt(n))
    
    print(f"单样本t检验:")
    print(f"  样本均值: {sample_mean:.3f}, 样本标准差: {sample_std:.3f}")
    print(f"  手动计算t值: {manual_t:.3f}, scipy计算t值: {t_stat:.3f}")
    print(f"  p值: {p_value:.3f}")
    print(f"  临界t值 (α=0.05): ±{stats.t.ppf(0.975, n-1):.3f}")
    
    # 独立样本t检验：比较两个独立样本
    sample2 = np.random.normal(loc=1.0, scale=1, size=100)
    t_stat, p_value = ttest_ind(sample1, sample2)
    
    # 手动计算合并方差
    n1, n2 = len(sample1), len(sample2)
    var1 = np.var(sample1, ddof=1)
    var2 = np.var(sample2, ddof=1)
    pooled_var = ((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2)
    manual_t = (np.mean(sample1) - np.mean(sample2)) / np.sqrt(pooled_var * (1/n1 + 1/n2))
    
    print(f"\n独立样本t检验:")
    print(f"  样本1均值: {np.mean(sample1):.3f}, 样本2均值: {np.mean(sample2):.3f}")
    print(f"  合并方差: {pooled_var:.3f}")
    print(f"  手动计算t值: {manual_t:.3f}, scipy计算t值: {t_stat:.3f}")
    print(f"  p值: {p_value:.3f}")
    
    # 配对样本t检验：比较配对样本
    before = np.random.normal(loc=10, scale=2, size=50)
    after = before + np.random.normal(loc=1, scale=1, size=50)  # 处理后有所提高
    t_stat, p_value = ttest_rel(before, after)
    
    # 手动计算配对差异
    differences = after - before
    diff_mean = np.mean(differences)
    diff_std = np.std(differences, ddof=1)
    manual_t = diff_mean / (diff_std / np.sqrt(len(differences)))
    
    print(f"\n配对样本t检验:")
    print(f"  配对差异均值: {diff_mean:.3f}, 差异标准差: {diff_std:.3f}")
    print(f"  手动计算t值: {manual_t:.3f}, scipy计算t值: {t_stat:.3f}")
    print(f"  p值: {p_value:.3f}")
    
    # 可视化结果
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 单样本t检验可视化
    axes[0,0].hist(sample1, alpha=0.7, bins=20, density=True)
    x_range = np.linspace(sample1.min(), sample1.max(), 100)
    axes[0,0].plot(x_range, norm.pdf(x_range, sample_mean, sample_std), 'r-', lw=2)
    axes[0,0].axvline(sample_mean, color='red', linestyle='--', label=f'Mean: {sample_mean:.2f}')
    axes[0,0].axvline(0, color='blue', linestyle='--', label='H₀: μ=0')
    axes[0,0].set_title('单样本t检验')
    axes[0,0].legend()
    
    # 独立样本t检验可视化
    axes[0,1].hist(sample1, alpha=0.7, bins=20, density=True, label='Sample 1')
    axes[0,1].hist(sample2, alpha=0.7, bins=20, density=True, label='Sample 2')
    axes[0,1].axvline(np.mean(sample1), color='red', linestyle='--')
    axes[0,1].axvline(np.mean(sample2), color='blue', linestyle='--')
    axes[0,1].set_title('独立样本t检验')
    axes[0,1].legend()
    
    # 配对样本t检验可视化
    axes[1,0].scatter(before, after, alpha=0.6)
    axes[1,0].plot([before.min(), before.max()], [before.min(), before.max()], 'r--')
    axes[1,0].set_xlabel('Before')
    axes[1,0].set_ylabel('After')
    axes[1,0].set_title('配对样本t检验')
    
    # 差异分布可视化
    axes[1,1].hist(differences, alpha=0.7, bins=15, density=True)
    axes[1,1].axvline(diff_mean, color='red', linestyle='--', label=f'Diff Mean: {diff_mean:.2f}')
    axes[1,1].axvline(0, color='blue', linestyle='--', label='H₀: μ_diff=0')
    axes[1,1].set_title('配对差异分布')
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.show()

def anova_example():
    """
    方差分析示例
    """
    print("\n=== 方差分析示例 ===")
    
    # 生成三组数据
    np.random.seed(42)
    group1 = np.random.normal(loc=5, scale=1, size=30)
    group2 = np.random.normal(loc=6, scale=1, size=30)
    group3 = np.random.normal(loc=7, scale=1, size=30)
    
    # 单因素ANOVA
    f_stat, p_value = f_oneway(group1, group2, group3)
    print(f"单因素ANOVA: F={f_stat:.3f}, p={p_value:.3f}")
    
    # 事后检验（Tukey HSD）
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    
    # 准备数据
    data = np.concatenate([group1, group2, group3])
    groups = np.array(['Group1']*30 + ['Group2']*30 + ['Group3']*30)
    
    tukey_result = pairwise_tukeyhsd(data, groups, alpha=0.05)
    print("\nTukey HSD事后检验:")
    print(tukey_result)
    
    # 可视化
    plt.figure(figsize=(10, 6))
    
    plt.subplot(121)
    plt.boxplot([group1, group2, group3], labels=['Group1', 'Group2', 'Group3'])
    plt.title('箱线图')
    
    plt.subplot(122)
    for i, (group, color) in enumerate(zip([group1, group2, group3], ['red', 'blue', 'green'])):
        plt.scatter([i+1]*len(group), group, alpha=0.6, color=color)
    plt.xticks([1, 2, 3], ['Group1', 'Group2', 'Group3'])
    plt.title('散点图')
    
    plt.tight_layout()
    plt.show()

def chi_square_example():
    """
    卡方检验示例
    """
    print("\n=== 卡方检验示例 ===")
    
    # 创建列联表
    # 假设检验性别与产品偏好是否相关
    observed = np.array([[30, 10, 20],   # 男性选择A,B,C产品的人数
                        [15, 25, 30]])  # 女性选择A,B,C产品的人数
    
    # 卡方独立性检验
    chi2, p_value, dof, expected = chi2_contingency(observed)
    
    print(f"卡方统计量: {chi2:.3f}")
    print(f"p值: {p_value:.3f}")
    print(f"自由度: {dof}")
    print("\n期望频数:")
    print(expected)
    
    # 可视化列联表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 观察频数热图
    sns.heatmap(observed, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Product A', 'Product B', 'Product C'],
                yticklabels=['Male', 'Female'], ax=ax1)
    ax1.set_title('观察频数')
    
    # 期望频数热图
    sns.heatmap(expected, annot=True, fmt='.1f', cmap='Reds',
                xticklabels=['Product A', 'Product B', 'Product C'],
                yticklabels=['Male', 'Female'], ax=ax2)
    ax2.set_title('期望频数')
    
    plt.tight_layout()
    plt.show()

def model_comparison_statistical_test():
    """
    模型比较的统计检验
    """
    print("\n=== 模型比较统计检验 ===")
    
    # 生成分类数据
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                              n_redundant=5, random_state=42)
    
    # 多次运行模型以获得性能分布
    n_runs = 50
    rf_scores = []
    lr_scores = []
    
    for _ in range(n_runs):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                           random_state=None)
        
        # 随机森林
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_score = accuracy_score(y_test, rf.predict(X_test))
        rf_scores.append(rf_score)
        
        # 逻辑回归
        lr = LogisticRegression(random_state=42)
        lr.fit(X_train, y_train)
        lr_score = accuracy_score(y_test, lr.predict(X_test))
        lr_scores.append(lr_score)
    
    # 配对t检验比较模型性能
    t_stat, p_value = ttest_rel(rf_scores, lr_scores)
    
    print(f"随机森林平均准确率: {np.mean(rf_scores):.3f} ± {np.std(rf_scores):.3f}")
    print(f"逻辑回归平均准确率: {np.mean(lr_scores):.3f} ± {np.std(lr_scores):.3f}")
    print(f"配对t检验: t={t_stat:.3f}, p={p_value:.3f}")
    
    # 可视化性能分布
    plt.figure(figsize=(10, 6))
    
    plt.boxplot([rf_scores, lr_scores], labels=['Random Forest', 'Logistic Regression'])
    plt.ylabel('Accuracy')
    plt.title('模型性能比较')
    plt.grid(True, alpha=0.3)
    
    # 添加统计显著性标记
    y_max = max(max(rf_scores), max(lr_scores)) + 0.02
    plt.plot([1, 1, 2, 2], [y_max-0.01, y_max, y_max, y_max-0.01], 'k-')
    plt.text(1.5, y_max+0.005, f'p = {p_value:.3f}', ha='center', va='bottom')
    
    plt.show()

def statistical_power_analysis():
    """
    统计功效分析
    """
    print("\n=== 统计功效分析 ===")
    
    # 功效分析：确定所需样本大小
    effect_size = 0.5  # 中等效应大小
    alpha = 0.05      # 显著性水平
    power = 0.8       # 期望功效
    
    # t检验功效分析
    t_power_analysis = TTestIndPower()
    sample_size = t_power_analysis.solve_power(effect_size=effect_size, 
                                              alpha=alpha, power=power)
    
    print(f"t检验所需样本大小（每组）: {sample_size:.0f}")
    
    # 绘制功效曲线
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 样本大小与功效的关系
    sample_sizes = np.arange(10, 200, 10)
    powers = t_power_analysis.power(effect_size=effect_size, nobs1=sample_sizes, alpha=alpha)
    
    ax1.plot(sample_sizes, powers, 'b-o')
    ax1.axhline(y=0.8, color='red', linestyle='--', label='目标功效 (0.8)')
    ax1.set_xlabel('样本大小')
    ax1.set_ylabel('统计功效')
    ax1.set_title('样本大小与统计功效')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 效应大小与功效的关系
    effect_sizes = np.linspace(0.1, 1.0, 20)
    powers_effect = t_power_analysis.power(effect_size=effect_sizes, nobs1=50, alpha=alpha)
    
    ax2.plot(effect_sizes, powers_effect, 'g-o')
    ax2.set_xlabel('效应大小')
    ax2.set_ylabel('统计功效')
    ax2.set_title('效应大小与统计功效')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# 示例使用
if __name__ == "__main__":
    # 运行所有示例
    t_test_examples()
    anova_example()
    chi_square_example()
    model_comparison_statistical_test()
    statistical_power_analysis()
    
    # 实际应用：A/B测试
    print("\n=== A/B测试示例 ===")
    
    # 模拟A/B测试数据
    np.random.seed(42)
    
    # A组：现有版本
    group_a_conversions = np.random.binomial(1, 0.1, size=1000)  # 10%转化率
    
    # B组：新版本
    group_b_conversions = np.random.binomial(1, 0.12, size=1000)  # 12%转化率
    
    # 比例检验
    from statsmodels.stats.proportion import proportions_ztest
    
    count = [group_a_conversions.sum(), group_b_conversions.sum()]
    nobs = [len(group_a_conversions), len(group_b_conversions)]
    
    z_stat, p_value = proportions_ztest(count, nobs)
    
    print(f"A组转化率: {count[0]/nobs[0]:.3f}")
    print(f"B组转化率: {count[1]/nobs[1]:.3f}")
    print(f"比例检验: z={z_stat:.3f}, p={p_value:.3f}")
    
    if p_value < 0.05:
        print("结果显著：新版本效果更好")
    else:
        print("结果不显著：无法确定新版本更好")
```

## 机器学习专用检验：理论与算法

### McNemar检验
**数学基础**：基于配对设计的二分类数据检验，用于比较两个分类模型在相同测试集上的表现。

设$n_{00}$为两个模型都正确的样本数，$n_{01}$为模型1正确模型2错误的样本数，
$n_{10}$为模型1错误模型2正确的样本数，$n_{11}$为两个模型都错误的样本数。

**原假设**：两个模型的错误率无显著差异，即$p_{01} = p_{10}$

**检验统计量**（连续校正版本）：

$\chi^2 = \frac{(|n_{01} - n_{10}| - 1)^2}{n_{01} + n_{10}} \sim \chi^2(1)$

**精确检验**（小样本时）：
$p = 2 \times \sum_{i=0}^{\min(n_{01}, n_{10})} \binom{n_{01}+n_{10}}{i} (0.5)^{n_{01}+n_{10}}$

**优势**：仅考虑不一致的分类结果，提高了检验效率。

### Friedman检验
**数学理论**：非参数检验方法，比较多个算法在多个数据集上的性能排名。

设$k$个算法在$N$个数据集上进行测试，$r_{ij}$为算法$j$在数据集$i$上的排名。

**平均排名**：$R_j = \frac{1}{N}\sum_{i=1}^N r_{ij}$

**Friedman统计量**：

$\chi_F^2 = \frac{12N}{k(k+1)} \left[ \sum_{j=1}^k R_j^2 - \frac{k(k+1)^2}{4} \right] \sim \chi^2(k-1)$

**Iman-Davenport改进**（更精确）：

$F_F = \frac{(N-1)\chi_F^2}{N(k-1) - \chi_F^2} \sim F(k-1, (k-1)(N-1))$

**事后检验（Nemenyi检验）**：
算法$i$和$j$的临界差异：$CD = q_{\alpha} \sqrt{\frac{k(k+1)}{6N}}$
其中$q_{\alpha}$为Studentized range统计量的临界值。

### Cochran's Q检验
**应用场景**：比较多个分类器在多个数据集上的表现。

设$k$个分类器，$N$个数据集，$y_{ij}$表示分类器$j$在数据集$i$上的表现（1=正确，0=错误）。

**检验统计量**：

$Q = \frac{(k-1)\left[k\sum_{j=1}^k G_j^2 - (\sum_{j=1}^k G_j)^2\right]}{k\sum_{i=1}^N L_i - \sum_{i=1}^N L_i^2} \sim \chi^2(k-1)$

其中$G_j = \sum_{i=1}^N y_{ij}$为分类器$j$的总正确数，$L_i = \sum_{j=1}^k y_{ij}$为数据集$i$上正确分类的总数。

## 应用场景
- 模型性能比较
- 特征重要性检验
- 假设验证
- A/B测试分析
- 算法选择
- 超参数优化验证

## 注意事项
- 确保满足检验的前提假设
- 考虑多重比较问题
- 正确解释p值
- 结合效应大小分析