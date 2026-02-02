# Optimization Algorithms: Theory and Practice

## 1. 数学基础与优化理论

### 1.1 梯度下降的基本原理

给定损失函数$L(\theta)$，梯度下降更新规则：

$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

其中$\eta$是学习率，$\nabla L(\theta_t)$是梯度。

### 1.2 收敛性分析

**Lipschitz连续性假设**：存在常数$L$使得
$$\|\nabla L(\theta) - \nabla L(\theta')\| \leq L\|\theta - \theta'\|$$

在适当的学习率下（$\eta < \frac{2}{L}$），梯度下降保证收敛。

## 2. 一阶优化算法详析

### 2.1 随机梯度下降（SGD）

**更新公式**：
$$\theta_{t+1} = \theta_t - \eta \nabla L_i(\theta_t)$$

其中$L_i$是单个样本的损失。

**方差分析**：
$$\mathbb{V}[\nabla L_i(\theta)] = \mathbb{V}[\nabla L(\theta)] + \text{噪声项}$$

### 2.2 动量法（Momentum）

**物理启发**：模拟小球在损失曲面上的运动，考虑惯性。

**更新公式**：
$$v_t = \gamma v_{t-1} + \eta \nabla L(\theta_t)$$
$$\theta_{t+1} = \theta_t - v_t$$

其中$\gamma$是动量系数（通常0.9）。

### 2.3 Nesterov加速梯度（NAG）

**改进动量**：先看未来位置，再计算梯度。

**更新公式**：
$$v_t = \gamma v_{t-1} + \eta \nabla L(\theta_t - \gamma v_{t-1})$$
$$\theta_{t+1} = \theta_t - v_t$$

### 2.4 自适应学习率方法

#### AdaGrad
**累积梯度平方**：
$$G_t = G_{t-1} + (\nabla L(\theta_t))^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \odot \nabla L(\theta_t)$$

**问题**：$G_t$单调递增，学习率过早衰减。

#### RMSProp
**指数移动平均**：
$$E[g^2]_t = \beta E[g^2]_{t-1} + (1-\beta)(\nabla L(\theta_t))^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{E[g^2]_t + \epsilon}} \odot \nabla L(\theta_t)$$

#### Adam（自适应矩估计）
**一阶矩（均值）**：
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) \nabla L(\theta_t)$$

**二阶矩（方差）**：
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)(\nabla L(\theta_t))^2$$

**偏差校正**：
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

**最终更新**：
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

## 3. 二阶优化方法

### 3.1 牛顿法

**二阶泰勒展开**：
$$L(\theta) \approx L(\theta_t) + \nabla L(\theta_t)^T(\theta-\theta_t) + \frac{1}{2}(\theta-\theta_t)^T H(\theta_t)(\theta-\theta_t)$$

其中$H(\theta_t)$是Hessian矩阵。

**更新公式**：
$$\theta_{t+1} = \theta_t - H^{-1}(\theta_t) \nabla L(\theta_t)$$

### 3.2 拟牛顿法（L-BFGS）

**近似Hessian逆**：避免直接计算和存储Hessian矩阵。

## 4. 梯度下降的数学推导

### 4.1 学习率选择理论

**线搜索**：寻找最优步长
$$\eta_t = \arg\min_{\eta > 0} L(\theta_t - \eta \nabla L(\theta_t))$$

**Armijo条件**：保证充分下降
$$L(\theta_t - \eta \nabla L(\theta_t)) \leq L(\theta_t) - c\eta \|\nabla L(\theta_t)\|^2$$

### 4.2 收敛速率分析

**强凸函数**：线性收敛$O((1-\mu/L)^t)$

**非强凸函数**：次线性收敛$O(1/t)$

**随机优化**：$O(1/\sqrt{t})$收敛速率

## 5. PyTorch实现与对比

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Callable

class CustomOptimizers:
    """自定义优化器实现"""
    
    @staticmethod
    def sgd_momentum(parameters, lr=0.01, momentum=0.9):
        """手动实现带动量的SGD"""
        velocities = [torch.zeros_like(p) for p in parameters]
        
        def step(gradients):
            updates = []
            for i, (grad, vel) in enumerate(zip(gradients, velocities)):
                # 动量更新
                new_vel = momentum * vel + lr * grad
                velocities[i] = new_vel
                updates.append(-new_vel)
            return updates
        
        return step
    
    @staticmethod
    def adagrad(parameters, lr=0.01, epsilon=1e-8):
        """手动实现AdaGrad"""
        squared_grads = [torch.zeros_like(p) for p in parameters]
        
        def step(gradients):
            updates = []
            for i, (grad, sq_grad) in enumerate(zip(gradients, squared_grads)):
                # 累积梯度平方
                new_sq_grad = sq_grad + grad ** 2
                squared_grads[i] = new_sq_grad
                
                # 自适应学习率
                adaptive_lr = lr / (torch.sqrt(new_sq_grad) + epsilon)
                updates.append(-adaptive_lr * grad)
            return updates
        
        return step
    
    @staticmethod
    def rmsprop(parameters, lr=0.01, alpha=0.99, epsilon=1e-8):
        """手动实现RMSProp"""
        squared_avg = [torch.zeros_like(p) for p in parameters]
        
        def step(gradients):
            updates = []
            for i, (grad, avg) in enumerate(zip(gradients, squared_avg)):
                # 指数移动平均
                new_avg = alpha * avg + (1 - alpha) * grad ** 2
                squared_avg[i] = new_avg
                
                # 自适应学习率
                adaptive_lr = lr / (torch.sqrt(new_avg) + epsilon)
                updates.append(-adaptive_lr * grad)
            return updates
        
        return step
    
    @staticmethod
    def adam(parameters, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """手动实现Adam"""
        m = [torch.zeros_like(p) for p in parameters]  # 一阶矩
        v = [torch.zeros_like(p) for p in parameters]  # 二阶矩
        t = 0  # 时间步
        
        def step(gradients):
            nonlocal t
            t += 1
            updates = []
            
            for i, (grad, m_i, v_i) in enumerate(zip(gradients, m, v)):
                # 更新一阶矩估计
                m_i_new = beta1 * m_i + (1 - beta1) * grad
                m[i] = m_i_new
                
                # 更新二阶矩估计
                v_i_new = beta2 * v_i + (1 - beta2) * grad ** 2
                v[i] = v_i_new
                
                # 偏差校正
                m_hat = m_i_new / (1 - beta1 ** t)
                v_hat = v_i_new / (1 - beta2 ** t)
                
                # 参数更新
                update = -lr * m_hat / (torch.sqrt(v_hat) + epsilon)
                updates.append(update)
            
            return updates
        
        return step

class OptimizerBenchmark:
    """优化器性能对比基准测试"""
    
    def __init__(self):
        self.optimizers = {
            'SGD': optim.SGD,
            'SGD+Momentum': lambda params: optim.SGD(params, momentum=0.9),
            'Adam': optim.Adam,
            'AdamW': optim.AdamW,
            'RMSprop': optim.RMSprop,
            'Adagrad': optim.Adagrad,
            'Adadelta': optim.Adadelta
        }
    
    def test_convergence(self, model_fn, loss_fn, data_generator, 
                        num_iterations=1000, lr=0.01):
        """测试不同优化器的收敛性能"""
        results = {}
        
        for name, optimizer_class in self.optimizers.items():
            # 重置模型
            model = model_fn()
            optimizer = optimizer_class(model.parameters(), lr=lr)
            
            losses = []
            
            for i in range(num_iterations):
                # 生成数据
                x, y = data_generator()
                
                # 前向传播
                output = model(x)
                loss = loss_fn(output, y)
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
            
            results[name] = losses
        
        return results
    
    def plot_convergence(self, results, figsize=(12, 8)):
        """绘制收敛曲线"""
        plt.figure(figsize=figsize)
        
        for name, losses in results.items():
            plt.plot(losses, label=name, alpha=0.8)
        
        plt.yscale('log')  # 对数坐标更好地显示收敛
        plt.xlabel('Iteration')
        plt.ylabel('Loss (log scale)')
        plt.title('Optimizer Convergence Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def analyze_gradient_distribution(self, model, dataloader, optimizer_name, lr=0.01):
        """分析梯度分布特性"""
        optimizer_class = self.optimizers[optimizer_name]
        optimizer = optimizer_class(model.parameters(), lr=lr)
        
        gradient_stats = []
        
        model.train()
        for batch_idx, (data, target) in enumerate(dataloader):
            if batch_idx >= 10:  # 分析前10个batch
                break
            
            optimizer.zero_grad()
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            
            # 收集梯度统计信息
            batch_grads = []
            for param in model.parameters():
                if param.grad is not None:
                    batch_grads.extend(param.grad.view(-1).tolist())
            
            if batch_grads:
                grad_array = np.array(batch_grads)
                stats = {
                    'mean': np.mean(grad_array),
                    'std': np.std(grad_array),
                    'max': np.max(grad_array),
                    'min': np.min(grad_array),
                    'norm': np.linalg.norm(grad_array)
                }
                gradient_stats.append(stats)
            
            optimizer.step()
        
        return gradient_stats

class LearningRateScheduler:
    """学习率调度器实现"""
    
    @staticmethod
    def step_decay(initial_lr, decay_factor, decay_epochs):
        """阶梯式衰减"""
        def scheduler(epoch):
            return initial_lr * (decay_factor ** (epoch // decay_epochs))
        return scheduler
    
    @staticmethod
    def exponential_decay(initial_lr, decay_rate):
        """指数衰减"""
        def scheduler(epoch):
            return initial_lr * np.exp(-decay_rate * epoch)
        return scheduler
    
    @staticmethod
    def cosine_annealing(initial_lr, T_max, eta_min=0):
        """余弦退火"""
        def scheduler(epoch):
            return eta_min + 0.5 * (initial_lr - eta_min) * (
                1 + np.cos(np.pi * epoch / T_max))
        return scheduler
    
    @staticmethod
    def warmup_cosine(initial_lr, warmup_epochs, total_epochs, eta_min=0):
        """预热+余弦退火"""
        def scheduler(epoch):
            if epoch < warmup_epochs:
                # 线性预热
                return initial_lr * (epoch + 1) / warmup_epochs
            else:
                # 余弦退火
                cos_epoch = epoch - warmup_epochs
                cos_total = total_epochs - warmup_epochs
                return eta_min + 0.5 * (initial_lr - eta_min) * (
                    1 + np.cos(np.pi * cos_epoch / cos_total))
        return scheduler

# 测试函数
class TestFunctions:
    """优化测试函数"""
    
    @staticmethod
    def quadratic_function(x, A, b):
        """二次函数: 0.5 * x^T A x + b^T x"""
        return 0.5 * torch.matmul(x.t(), torch.matmul(A, x)) + torch.matmul(b.t(), x)
    
    @staticmethod
    def rosenbrock(x, a=1, b=100):
        """Rosenbrock函数（香蕉函数）"""
        return (a - x[0])**2 + b * (x[1] - x[0]**2)**2
    
    @staticmethod
    def rastrigin(x, A=10):
        """Rastrigin函数（多局部最小值）"""
        n = x.size(0)
        return A * n + torch.sum(x**2 - A * torch.cos(2 * torch.pi * x))

# 使用示例
if __name__ == "__main__":
    # 创建简单的测试模型
    class SimpleModel(nn.Module):
        def __init__(self, input_size=10, hidden_size=50, output_size=1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size)
            )
        
        def forward(self, x):
            return self.net(x)
    
    # 数据生成器
    def data_generator():
        x = torch.randn(32, 10)  # batch_size=32, input_size=10
        # 简单的线性关系加噪声
        y = x.sum(dim=1, keepdim=True) + 0.1 * torch.randn(32, 1)
        return x, y
    
    # 基准测试
    benchmark = OptimizerBenchmark()
    
    print("开始优化器收敛性能测试...")
    
    results = benchmark.test_convergence(
        model_fn=lambda: SimpleModel(),
        loss_fn=nn.MSELoss(),
        data_generator=data_generator,
        num_iterations=500,
        lr=0.01
    )
    
    # 绘制收敛曲线
    benchmark.plot_convergence(results)
    
    # 自定义优化器测试
    print("\n自定义优化器实现测试:")
    
    # 创建测试参数
    params = [torch.randn(10, requires_grad=True) for _ in range(3)]
    
    # 测试Adam实现
    custom_adam = CustomOptimizers.adam(params, lr=0.001)
    
    # 模拟几次更新
    for step in range(5):
        # 生成随机梯度（模拟反向传播）
        grads = [torch.randn_like(p) for p in params]
        
        # 自定义Adam更新
        updates = custom_adam(grads)
        
        # 应用更新（这里只是演示，实际中应该用真实梯度）
        for i, (param, update) in enumerate(zip(params, updates)):
            # 注意：这里只是演示，实际应该用 param.data += update
            print(f"Step {step}, Param {i}: update norm = {update.norm().item():.6f}")
    
    # 学习率调度测试
    print("\n学习率调度器测试:")
    
    scheduler = LearningRateScheduler.warmup_cosine(
        initial_lr=0.1, warmup_epochs=5, total_epochs=50
    )
    
    for epoch in range(0, 50, 5):
        lr = scheduler(epoch)
        print(f"Epoch {epoch:2d}: learning rate = {lr:.6f}")
```

## 6. 优化器选择指南

### 6.1 根据问题特性选择

| 问题类型 | 推荐优化器 | 理由 |
|---------|-----------|------|
| 凸优化问题 | SGD + Momentum | 理论保证好，收敛稳定 |
| 非凸深度学习 | Adam/AdamW | 自适应学习率，收敛快 |
| 小批量数据 | SGD | 避免过拟合，泛化好 |
| 大规模数据 | Adam | 计算高效，自适应 |
| 强化学习 | RMSprop | 处理非平稳目标 |

### 6.2 超参数调优建议

**学习率**：
- SGD：0.01-0.1
- Adam：0.001-0.0001
- 使用学习率查找器（LR Finder）

**动量系数**：
- 通常0.9
- Nesterov动量效果更好

**Adam参数**：
- β₁=0.9, β₂=0.999, ε=1e-8
- 偏差校正很重要

## 7. 高级优化技术

### 7.1 梯度裁剪
防止梯度爆炸：
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 7.2 权重衰减
L2正则化：
- SGD：直接在损失函数中添加
- AdamW：解耦权重衰减

### 7.3 Lookahead优化器
外层循环稳定训练：
```python
base_optimizer = optim.Adam(model.parameters(), lr=0.001)
lookahead = Lookahead(base_optimizer, k=5, alpha=0.5)
```

## 8. 延伸学习

### 8.1 核心论文
1. **[Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)** - Adam原始论文
2. **[Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)** - AdamW提出
3. **[On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ)** - Adam收敛性分析
4. **[An Overview of Gradient Descent Optimization Algorithms](https://ruder.io/optimizing-gradient-descent/)** - 综合综述

### 8.2 实践建议
1. **默认选择**：从Adam开始，观察训练动态
2. **精调策略**：后期切换到SGD提高精度
3. **监控工具**：使用TensorBoard监控梯度分布
4. **自动化调优**：尝试Optuna等超参数优化库

### 8.3 未来方向
1. **自适应优化**：根据数据特性自动调整优化策略
2. **二阶方法**： scalable的二阶优化方法
3. **理论突破**：更深入的非凸优化理论理解
4. **硬件优化**：针对特定硬件的优化算法设计