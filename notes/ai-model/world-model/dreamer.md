# Dreamer 系列模型详解

Dreamer 是世界模型领域的里程碑工作，由 Danijar Hafner 等人提出。本章详细介绍 Dreamer 系列的演进和核心技术。

---

## 系列概览

Dreamer 系列的发展历程：

```
World Models (2018)          Dreamer v1 (2020)          Dreamer v2 (2021)          Dreamer v3 (2023)
     │                            │                           │                           │
     │    VAE + MDN-RNN           │    RSSM                   │    离散潜在状态            │    固定超参数
     │    分离训练                │    端到端想象规划          │    Atari 突破              │    通用算法
     │                            │                           │                           │
     └────────────────────────────┴───────────────────────────┴───────────────────────────┘
```

| 版本 | 年份 | 核心贡献 | 论文 |
|------|------|----------|------|
| Dreamer v1 | 2020 | RSSM + 端到端想象规划 | [Dream to Control](https://arxiv.org/abs/1912.01603) |
| Dreamer v2 | 2021 | 离散潜在状态 | [Mastering Atari](https://arxiv.org/abs/2010.02193) |
| Dreamer v3 | 2023 | 固定超参数、通用性 | [Mastering Diverse Domains](https://arxiv.org/abs/2301.04104) |

---

## Dreamer v1

### 核心创新

Dreamer v1 的核心贡献是实现了**端到端的想象规划**：

1. **RSSM 架构**：结合确定性和随机性状态
2. **想象规划**：在潜在空间中进行规划
3. **端到端训练**：策略梯度通过世界模型反向传播

### 架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Dreamer v1 架构                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐                                                    │
│  │  编码器      │ o_t ──→ z_t                                       │
│  │  (CNN VAE)  │                                                    │
│  └─────────────┘                                                    │
│         │                                                           │
│         ↓                                                           │
│  ┌─────────────────────────────────────────────┐                    │
│  │                    RSSM                      │                    │
│  │  ┌───────┐   ┌───────┐   ┌───────┐          │                    │
│  │  │  GRU  │──→│ 混合  │──→│ 采样  │──→ s_t   │                    │
│  │  │       │   │ 密度  │   │       │          │                    │
│  │  └───────┘   └───────┘   └───────┘          │                    │
│  │      ↑                                      │                    │
│  │  h_{t-1}, s_{t-1}, a_{t-1}                  │                    │
│  └─────────────────────────────────────────────┘                    │
│         │                                                           │
│         ↓                                                           │
│  ┌─────────────┐                                                    │
│  │  想象规划    │ 在潜在空间模拟未来轨迹，学习策略                     │
│  └─────────────┘                                                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 想象规划 (Imagined Planning)

Dreamer 的关键创新是在学到的世界模型中进行规划：

```python
# 想象 H 步
h, s = current_state
imagined_states = []
imagined_rewards = []

for t in range(H):
    # 想象中的动作
    a = policy(h, s)
    # 想象中的状态转移
    h, s = rssm.transition(h, s, a)
    # 想象中的奖励
    r = reward_model(h, s)
    
    imagined_states.append((h, s))
    imagined_rewards.append(r)
```

### 损失函数

Dreamer v1 的训练目标包含三部分：

$$
\mathcal{L} = \mathcal{L}_{dyn} + \mathcal{L}_{rep} + \mathcal{L}_{actor}
$$

| 损失 | 目标 |
|------|------|
| $\mathcal{L}_{dyn}$ | 动态预测损失（先验vs后验） |
| $\mathcal{L}_{rep}$ | 表示学习损失（重构观测） |
| $\mathcal{L}_{actor}$ | 策略优化损失（想象中的价值） |

---

## Dreamer v2

### 核心改进

Dreamer v2 的关键改进是使用**离散潜在状态**：

1. **离散表示**：将连续潜在空间离散化
2. **Atari 突破**：首次在 Atari 100k 上超越人类
3. **更好的长期规划**：离散状态更适合建模多模态分布

### 离散潜在状态

```
Dreamer v1: z ∈ R^d (连续)
Dreamer v2: z ∈ {0, 1, ..., K-1}^M (离散)
```

将潜在状态分为 M 个类别，每个类别有 K 个离散值：

```python
class DiscreteLatent(nn.Module):
    def __init__(self, num_classes=32, num_categories=32):
        self.num_classes = num_classes
        self.num_categories = num_categories
        
    def forward(self, logits):
        # logits: [batch, num_categories, num_classes]
        # straight-through estimator for gradient
        if self.training:
            # Gumbel-Softmax 采样
            samples = gumbel_softmax(logits, hard=True)
        else:
            # 直接取 argmax
            samples = one_hot(logits.argmax(dim=-1))
        return samples
```

### 为什么离散更好？

| 方面 | 连续潜在状态 | 离散潜在状态 |
|------|--------------|--------------|
| 表示能力 | 单峰分布 | 多峰分布 |
| 长期预测 | 误差累积 | 更稳定 |
| 多模态场景 | 困难 | 自然处理 |
| 计算效率 | 较低 | 较高 |

### 性能表现

Dreamer v2 在 Atari 100k benchmark 上的表现：

```
Atari 100k (200M帧等价的100k交互):
├── Dreamer v2: 134% 人类水平
├── SimPLe: 79% 人类水平
├── OTRainbow: 20% 人类水平
└── DER: 14% 人类水平
```

---

## Dreamer v3

### 核心改进

Dreamer v3 的目标是实现**通用性**：

1. **固定超参数**：所有任务使用相同的超参数
2. **更广泛的适用性**：从 Atari 到机器人控制
3. **样本效率**：进一步提升

### 架构改进

```
Dreamer v3 主要改进:
├── 更大的模型容量
├── Symlog 预测 (预测 log(1+|x|) * sign(x))
├── Free bits (防止 KL 塌缩)
├── 分离的 actor-critic 学习率
└── 更长的想象深度 (H=15)
```

### Symlog 预测

处理不同尺度的奖励预测：

```python
def symlog(x):
    """Symmetric log transform"""
    return torch.sign(x) * torch.log(1 + torch.abs(x))

def symexp(x):
    """Inverse of symlog"""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)
```

### 通用性验证

Dreamer v3 在多种任务上验证了通用性：

| 域 | 任务 | 性能 |
|---|------|------|
| Atari 100k | 26 游戏 | 164% 人类水平 |
| Atari 200M | 55 游戏 | 超越人类中位数 |
| ProcGen | 16 关卡 | 竞争性能 |
| BSUITE | 基础能力 | 强泛化 |
| Minecraft | 钻石获取 | 首次解决 |
| DMC | 机器人控制 | SOTA |

---

## 关键技术总结

### RSSM 核心实现

```python
class RSSM:
    """
    Recurrent State-Space Model
    h: 确定性状态 (GRU hidden)
    s: 随机性状态 (离散或连续)
    """
    
    def __init__(self, config):
        self.gru = GRUCell(...)
        self.prior_net = MLP(...)  # h -> prior s
        self.posterior_net = MLP(...)  # h + z -> posterior s
        
    def observe(self, obs, action):
        """编码观测序列，得到后验状态序列"""
        h, s = self.init_state()
        states = []
        
        for o, a in zip(obs, action):
            z = self.encoder(o)
            h, s = self.posterior_step(h, s, a, z)
            states.append((h, s))
            
        return states
    
    def imagine(self, policy, horizon):
        """想象未来轨迹"""
        h, s = self.current_state
        trajectory = []
        
        for _ in range(horizon):
            a = policy(h, s)
            h, s = self.prior_step(h, s, a)
            r = self.reward_model(h, s)
            v = self.value_model(h, s)
            trajectory.append((h, s, a, r, v))
            
        return trajectory
```

### 想象规划策略学习

```python
def train_policy(world_model, policy, value, replay_buffer):
    # 1. 从世界模型想象轨迹
    trajectories = world_model.imagine(policy, horizon=H)
    
    # 2. 计算想象中的回报
    returns = compute_returns(trajectories, gamma)
    
    # 3. 更新策略 (policy gradient in imagination)
    policy_loss = -returns.mean()
    
    # 4. 更新价值函数
    value_loss = F.mse_loss(value_predictions, returns.detach())
    
    return policy_loss, value_loss
```

---

## Dreamer vs 其他方法

| 方法 | 模型基? | 样本效率 | 计算开销 | 长期规划 |
|------|---------|----------|----------|----------|
| PPO | 否 | 低 | 低 | 差 |
| SAC | 否 | 中 | 低 | 差 |
| MuZero | 是 | 高 | 高 | 好 |
| Dreamer | 是 | 高 | 中 | 好 |

---

## 实践建议

### 什么时候使用 Dreamer？

- 需要高样本效率
- 环境模拟代价高
- 需要进行规划
- 状态空间复杂（图像观测）

### 实现注意事项

1. **KL 平衡**：避免后验塌缩
2. **梯度传播**：正确处理离散采样的梯度
3. **想象深度**：根据任务调整 H
4. **奖励缩放**：使用 symlog 或其他归一化

---

## 小结

Dreamer 系列的关键贡献：

1. **RSSM**：优雅地结合确定性和随机性状态
2. **想象规划**：在潜在空间高效规划
3. **端到端训练**：无需分阶段训练
4. **通用性**：v3 实现了固定超参数的通用算法

---

## 参考资料

- [Dream to Control (Dreamer v1)](https://arxiv.org/abs/1912.01603)
- [Mastering Atari with Discrete World Models (Dreamer v2)](https://arxiv.org/abs/2010.02193)
- [Mastering Diverse Domains through World Models (Dreamer v3)](https://arxiv.org/abs/2301.04104)
- [Dreamer 官方代码](https://github.com/danijar/dreamer)
- [Dreamer v3 代码](https://github.com/danijar/dreamerv3)
