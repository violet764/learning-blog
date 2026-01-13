# 优化理论

## 概述

优化理论是机器学习的核心数学基础，为模型训练和参数调优提供了理论支持。理解凸优化、梯度方法和约束优化对于掌握机器学习算法至关重要。

## 优化问题基本形式

### 一般优化问题

**定义**：优化问题的一般形式为：
$$
\min_{x \in \mathbb{R}^n} f(x)
$$
其中f(x)是目标函数，x是决策变量。

### 约束优化问题

**定义**：带约束的优化问题：
$$
\begin{aligned}
\min_{x} & \quad f(x) \\
\text{s.t.} & \quad g_i(x) \leq 0, \quad i = 1, \ldots, m \\
& \quad h_j(x) = 0, \quad j = 1, \ldots, p
\end{aligned}
$$

## 凸优化基础

### 凸集

**定义**：集合C是凸集，如果对于任意x,y∈C和任意θ∈[0,1]，有：
$$
\theta x + (1-\theta)y \in C
$$

### 凸函数

**定义**：函数f是凸函数，如果对于任意x,y∈dom(f)和任意θ∈[0,1]，有：
$$
f(\theta x + (1-\theta)y) \leq \theta f(x) + (1-\theta) f(y)
$$

### 凸优化问题

**定义**：如果目标函数f是凸函数，且可行域是凸集，则优化问题是凸优化问题。

**性质**：凸优化问题的局部最优解就是全局最优解。

## 梯度下降方法

### 梯度概念

**定义**：函数f在点x的梯度是函数增长最快的方向。
$$
\nabla f(x) = \left[ \frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_n} \right]^T
$$

### 批量梯度下降

**算法**：
$$
x^{(k+1)} = x^{(k)} - \alpha \nabla f(x^{(k)})
$$
其中α是学习率。

```python
import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(f, grad_f, x0, alpha=0.1, max_iter=1000, tol=1e-6):
    """批量梯度下降算法"""
    x = x0.copy()
    history = [x.copy()]
    
    for i in range(max_iter):
        gradient = grad_f(x)
        x_new = x - alpha * gradient
        
        # 检查收敛
        if np.linalg.norm(x_new - x) < tol:
            break
            
        x = x_new
        history.append(x.copy())
    
    return x, history

# 示例：最小化二次函数 f(x) = x^2 + 2x + 1
f = lambda x: x**2 + 2*x + 1
grad_f = lambda x: 2*x + 2

x0 = np.array([5.0])  # 初始点
x_opt, history = gradient_descent(f, grad_f, x0, alpha=0.1)

print(f"最优解: x = {x_opt[0]:.6f}")
print(f"最优值: f(x) = {f(x_opt[0]):.6f}")
```

### 随机梯度下降(SGD)

**算法**：每次迭代使用一个样本计算梯度。
$$
x^{(k+1)} = x^{(k)} - \alpha \nabla f_i(x^{(k)})
$$
其中f_i是第i个样本的损失函数。

## 高级优化算法

### 动量法

**算法**：引入动量项加速收敛。
$$
\begin{aligned}
v^{(k+1)} &= \beta v^{(k)} + (1-\beta) \nabla f(x^{(k)}) \\
x^{(k+1)} &= x^{(k)} - \alpha v^{(k+1)}
\end{aligned}
$$

### Adam算法

**算法**：结合动量法和自适应学习率。
$$
\begin{aligned}
m^{(k+1)} &= \beta_1 m^{(k)} + (1-\beta_1) \nabla f(x^{(k)}) \\
v^{(k+1)} &= \beta_2 v^{(k)} + (1-\beta_2) (\nabla f(x^{(k)}))^2 \\
\hat{m} &= \frac{m^{(k+1)}}{1-\beta_1^{k+1}} \\
\hat{v} &= \frac{v^{(k+1)}}{1-\beta_2^{k+1}} \\
x^{(k+1)} &= x^{(k)} - \alpha \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}
\end{aligned}
$$

## 约束优化

### 拉格朗日乘子法

**定义**：对于等式约束优化问题：
$$
\min f(x) \quad \text{s.t.} \quad h(x) = 0
$$
构造拉格朗日函数：
$$
L(x, \lambda) = f(x) + \lambda^T h(x)
$$

### KKT条件

**定理**：对于凸优化问题，x*是最优解的必要条件是存在λ*, μ*满足：
1. **原始可行性**：g(x*) ≤ 0, h(x*) = 0
2. **对偶可行性**：μ* ≥ 0
3. **互补松弛**：μ*ᵢgᵢ(x*) = 0
4. **梯度为零**：∇f(x*) + ∑λ*ᵢ∇hᵢ(x*) + ∑μ*ᵢ∇gᵢ(x*) = 0

## 在机器学习中的应用

### 线性回归

**优化问题**：
$$
\min_w \frac{1}{2} \|Xw - y\|_2^2
$$
**闭式解**：w* = (XᵀX)⁻¹Xᵀy

### 逻辑回归

**优化问题**：
$$
\min_w \sum_{i=1}^n \log(1 + \exp(-y_i w^T x_i))
$$
使用梯度下降求解。

### 支持向量机

**优化问题**：
$$
\min_{w,b} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^n \max(0, 1 - y_i(w^T x_i + b))
$$
使用对偶问题求解。

## 总结

优化理论为机器学习提供了模型训练和参数调优的数学基础。从基本的梯度下降到高级的约束优化方法，这些工具在各种机器学习算法中都有广泛应用。