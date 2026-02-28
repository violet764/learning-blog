# 策略梯度方法详解

本章深入讲解策略梯度方法，这是强化学习中直接优化策略参数的重要技术。与价值函数方法不同，策略梯度方法直接在策略空间进行搜索，特别适用于连续动作空间和高维问题。

## 策略梯度基础

### 策略参数化

将策略表示为参数化函数：

$$
\pi_\theta(a|s) = P(A_t = a | S_t = s, \theta_t = \theta)
$$

其中 $\theta$ 是策略参数（神经网络权重等）。

### 目标函数

策略梯度方法的目标是最大化期望回报：

$$
J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{\infty} \gamma^t r_t\right] = \mathbb{E}_{\pi_\theta}[G_0]
$$

### 策略梯度定理

策略梯度定理给出了目标函数对参数的梯度：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot Q^{\pi_\theta}(s,a)\right]
$$

这是策略梯度方法的理论基础，表明可以通过采样估计梯度。

**直观理解**：
- 当 $Q(s,a)$ 较大（动作好）时，增加该动作的概率 $\pi_\theta(a|s)$
- 当 $Q(s,a)$ 较小（动作差）时，降低该动作的概率

## REINFORCE 算法

### 算法原理

REINFORCE（蒙特卡洛策略梯度）使用完整回合的累积回报作为 $Q(s,a)$ 的估计：

$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^{N} \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t^i|s_t^i) \cdot G_t^i
$$

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PolicyNetwork(nn.Module):
    """策略网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.network(x)


class REINFORCE:
    """REINFORCE 算法"""
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.gamma = gamma
        
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # 存储回合数据
        self.log_probs = []
        self.rewards = []
    
    def select_action(self, state):
        """根据策略选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy_net(state_tensor)
        
        # 采样动作
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        self.log_probs.append(log_prob)
        return action.item()
    
    def store_reward(self, reward):
        """存储奖励"""
        self.rewards.append(reward)
    
    def update(self):
        """更新策略"""
        # 计算回报（从后向前）
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        # 标准化（减少方差）
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # 计算策略梯度损失
        policy_loss = []
        for log_prob, G in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * G)
        
        # 优化
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()
        
        # 清空数据
        self.log_probs = []
        self.rewards = []
        
        return loss.item()
```

### 基线技巧

为减少方差，引入基线 $b(s)$：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot (Q(s,a) - b(s))\right]
$$

常用基线：状态价值函数 $V(s)$

## Actor-Critic 方法

### 基本思想

结合策略梯度（Actor）和价值函数（Critic）：

- **Actor**：负责选择动作，直接优化策略
- **Critic**：负责评估动作价值，减少方差

### 优势函数

使用优势函数 $A(s,a) = Q(s,a) - V(s)$ 作为 Critic 的评估：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot A(s,a)\right]
$$

```python
class ValueNetwork(nn.Module):
    """价值网络（Critic）"""
    def __init__(self, state_dim, hidden_dim=128):
        super(ValueNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.network(x)


class ActorCritic:
    """Actor-Critic 算法"""
    def __init__(self, state_dim, action_dim, lr_actor=1e-3, lr_critic=1e-3, gamma=0.99):
        self.gamma = gamma
        
        # Actor
        self.actor = PolicyNetwork(state_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        # Critic
        self.critic = ValueNetwork(state_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.saved_log_probs = []
        self.saved_values = []
        self.rewards = []
    
    def select_action(self, state):
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Actor 输出动作概率
        probs = self.actor(state_tensor)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Critic 输出状态价值
        value = self.critic(state_tensor)
        
        self.saved_log_probs.append(log_prob)
        self.saved_values.append(value)
        
        return action.item()
    
    def update(self):
        """更新网络"""
        # 计算回报
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.FloatTensor(returns)
        values = torch.cat(self.saved_values).squeeze()
        log_probs = torch.stack(self.saved_log_probs)
        
        # 计算优势
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Actor 损失
        actor_loss = -(log_probs * advantages).mean()
        
        # Critic 损失
        critic_loss = nn.MSELoss()(values, returns)
        
        # 更新 Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 更新 Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 清空数据
        self.saved_log_probs = []
        self.saved_values = []
        self.rewards = []
        
        return actor_loss.item(), critic_loss.item()
```

## 连续动作空间

### 高斯策略

对于连续动作空间，使用高斯分布：

$$
\pi_\theta(a|s) = \mathcal{N}(\mu_\theta(s), \sigma_\theta^2(s))
$$

```python
class GaussianPolicy(nn.Module):
    """高斯策略网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(GaussianPolicy, self).__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 均值输出
        self.mu = nn.Linear(hidden_dim, action_dim)
        
        # 标准差（使用参数或网络输出）
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, x):
        features = self.shared(x)
        mu = torch.tanh(self.mu(features))  # 限制在 [-1, 1]
        std = torch.exp(self.log_std)
        return mu, std
    
    def sample(self, x):
        """采样动作"""
        mu, std = self.forward(x)
        dist = torch.distributions.Normal(mu, std)
        action = dist.rsample()  # 重参数化
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        action = torch.tanh(action)  # 限制动作范围
        return action, log_prob
```

## 实战：CartPole 与 Pendulum

### CartPole（离散动作）

```python
import gym

def train_reinforce_cartpole():
    """在 CartPole 环境训练 REINFORCE"""
    env = gym.make('CartPole-v1')
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = REINFORCE(state_dim, action_dim)
    
    episodes = 1000
    scores = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.store_reward(reward)
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        agent.update()
        scores.append(total_reward)
        
        if episode % 50 == 0:
            avg = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
            print(f"Episode {episode}: 平均奖励 {avg:.1f}")
    
    env.close()
    return agent


def train_ac_cartpole():
    """在 CartPole 环境训练 Actor-Critic"""
    env = gym.make('CartPole-v1')
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = ActorCritic(state_dim, action_dim)
    
    episodes = 500
    scores = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.rewards.append(reward)
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        agent.update()
        scores.append(total_reward)
        
        if episode % 50 == 0:
            avg = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
            print(f"Episode {episode}: 平均奖励 {avg:.1f}")
    
    env.close()
    return agent
```

## 知识点关联

```
REINFORCE（蒙特卡洛策略梯度）
         ↓
    ┌────┴────┐
    ↓         ↓
  方差大    无偏
    ↓
 加入基线
    ↓
Actor-Critic
    ↓
┌───┼───┐
↓   ↓   ↓
A2C A3C TRPO/PPO
```

## 核心考点

### 必须掌握

| 概念 | 说明 |
|------|------|
| 策略梯度定理 | $\nabla_\theta J = \mathbb{E}[\nabla_\theta \log \pi_\theta(a\|s) \cdot Q(s,a)]$ |
| REINFORCE | 使用完整回合的蒙特卡洛策略梯度 |
| 基线技巧 | 减少方差，不引入偏差 |
| Actor-Critic | 结合策略梯度和价值函数 |

### 方法对比

| 方法 | 方差 | 偏差 | 样本效率 | 适用场景 |
|------|------|------|----------|----------|
| REINFORCE | 高 | 无 | 低 | 简单环境 |
| REINFORCE + 基线 | 中 | 无 | 中 | 一般环境 |
| Actor-Critic | 低 | 低 | 高 | 复杂环境 |

### 连续动作空间处理

| 技术 | 说明 |
|------|------|
| 高斯策略 | 输出均值和方差，采样动作 |
| 重参数化 | 使采样过程可微分 |
| tanh 限制 | 将动作限制在有效范围内 |

## 学习建议

### 实践要点

1. **学习率**：Actor 通常比 Critic 学习率小
2. **基线选择**：$V(s)$ 是最常用的基线
3. **方差减少**：标准化优势函数，使用 GAE

### 常见问题

| 问题 | 解决方案 |
|------|----------|
| 训练不稳定 | 降低学习率，使用熵正则化 |
| 收敛慢 | 检查网络架构，增加 hidden_dim |
| 方差过大 | 使用更好的基线，标准化回报 |

### 后续方向

- **PPO**：稳定的策略优化算法
- **TRPO**：信任域策略优化
- **SAC**：最大熵强化学习

---

**上一章**：[函数逼近与深度强化学习](./function-approximation.md)  
**下一章**：[现代强化学习算法](./modern-algorithms.md)
