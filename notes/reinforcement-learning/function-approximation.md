# 函数逼近与深度强化学习

本章介绍函数逼近技术在强化学习中的应用，重点讲解如何用神经网络等函数逼近器处理大规模状态空间问题。这是连接传统强化学习和深度强化学习的关键技术。

## 函数逼近基础

### 维数灾难

当状态空间过大时，表格方法面临严重挑战：

| 问题 | 说明 |
|------|------|
| 存储问题 | 状态数量指数增长，无法存储完整 Q 表 |
| 泛化问题 | 相似状态需要共享经验 |
| 计算效率 | 需要高效利用有限的经验数据 |

### 函数逼近原理

用参数化函数近似价值函数：

$$
V(s) \approx \hat{V}(s; \mathbf{w}), \quad Q(s,a) \approx \hat{Q}(s,a; \mathbf{w})
$$

其中 $\mathbf{w}$ 是参数向量（神经网络权重等）。

### 目标函数

最小化预测值与目标值的均方误差：

$$
J(\mathbf{w}) = \mathbb{E}_\pi \left[ (V_\pi(s) - \hat{V}(s;\mathbf{w}))^2 \right]
$$

**梯度下降更新**：

$$
\mathbf{w} \leftarrow \mathbf{w} - \alpha \nabla_{\mathbf{w}} J(\mathbf{w}) = \mathbf{w} + \alpha [V_\pi(s) - \hat{V}(s;\mathbf{w})] \nabla_{\mathbf{w}} \hat{V}(s;\mathbf{w})
$$

## 深度 Q 网络（DQN）

DQN 是 DeepMind 在 2015 年提出的突破性算法，首次实现了从原始像素学习控制策略。

### DQN 核心创新

| 技术 | 作用 |
|------|------|
| **经验回放** | 打破数据相关性，提高样本效率 |
| **目标网络** | 稳定训练过程，避免目标值快速变化 |
| **端到端学习** | 直接从原始输入（如像素）学习 |

### 网络架构

```python
import torch
import torch.nn as nn

class DQN(nn.Module):
    """深度 Q 网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)


class ConvDQN(nn.Module):
    """卷积 DQN，用于图像输入"""
    def __init__(self, input_shape, n_actions):
        super(ConvDQN, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # 计算卷积输出尺寸
        conv_out_size = self._get_conv_out(input_shape)
        
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    
    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(torch.prod(torch.tensor(o.size())))
    
    def forward(self, x):
        conv_out = self.conv(x).view(x.size(0), -1)
        return self.fc(conv_out)
```

### 经验回放

```python
import random
from collections import deque

class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """存储经验"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """随机采样"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)
```

### DQN 智能体

```python
import torch.optim as optim
import numpy as np

class DQNAgent:
    """DQN 智能体"""
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 memory_size=10000, batch_size=32, target_update=100):
        
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update = target_update
        
        # 主网络和目标网络
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(memory_size)
        self.steps = 0
    
    def select_action(self, state):
        """ε-贪心选择动作"""
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            return self.policy_net(state_tensor).argmax(1).item()
    
    def update(self):
        """更新网络"""
        if len(self.memory) < self.batch_size:
            return None
        
        # 采样批次
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        # 当前 Q 值
        current_q = self.policy_net(states).gather(1, actions)
        
        # 目标 Q 值
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # 计算损失并更新
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新目标网络
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # 衰减探索率
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
```

## DQN 变体

### Double DQN

解决 Q 值过高估计问题：

$$
Q_{\text{target}} = r + \gamma Q_{\theta^-}\left(s', \arg\max_{a'} Q_\theta(s', a')\right)
$$

```python
class DoubleDQNAgent(DQNAgent):
    """Double DQN 智能体"""
    def update(self):
        if len(self.memory) < self.batch_size:
            return None
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        # 当前 Q 值
        current_q = self.policy_net(states).gather(1, actions)
        
        # Double DQN：用 policy_net 选动作，target_net 评估
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        loss = nn.MSELoss()(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # ... 其他更新逻辑同父类
```

### Dueling DQN

分离状态价值和优势函数：

$$
Q(s,a) = V(s) + A(s,a) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s,a')
$$

```python
class DuelingDQN(nn.Module):
    """Dueling DQN 网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DuelingDQN, self).__init__()
        
        # 共享特征层
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 价值流
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 优势流
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # 合并：Q = V + A - mean(A)
        q_values = value + advantages - advantages.mean(dim=1, keepdim=True)
        return q_values
```

### 优先经验回放

根据 TD 误差优先回放重要经验：

```python
class PrioritizedReplayBuffer:
    """优先经验回放"""
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha  # 优先级指数
        self.beta = beta    # 重要性采样指数
        self.beta_increment = beta_increment
        
        self.buffer = []
        self.priorities = np.zeros(capacity)
        self.pos = 0
        self.max_priority = 1.0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]
        
        # 采样概率
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # 重要性采样权重
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        batch = [self.buffer[i] for i in indices]
        return indices, batch, weights
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
```

## 实战：CartPole

```python
import gym

def train_dqn_cartpole():
    """在 CartPole 环境训练 DQN"""
    env = gym.make('CartPole-v1')
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_dim, action_dim)
    
    episodes = 500
    scores = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.memory.push(state, action, reward, next_state, done)
            agent.update()
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        scores.append(total_reward)
        
        if episode % 50 == 0:
            avg = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
            print(f"Episode {episode}: 平均奖励 {avg:.1f}, ε={agent.epsilon:.3f}")
        
        # 提前停止
        if len(scores) >= 100 and np.mean(scores[-100:]) >= 195:
            print(f"环境解决于 Episode {episode}!")
            break
    
    env.close()
    return agent, scores

# 运行训练
agent, scores = train_dqn_cartpole()
```

## 知识点关联

```
线性函数逼近
       ↓
神经网络逼近
       ↓
    DQN
   ┌───┼───┐
   ↓   ↓   ↓
Double  Dueling  PER
  DQN     DQN
   └───┼───┘
       ↓
   Rainbow DQN
```

## 核心考点

### 必须掌握

| 技术点 | 核心作用 |
|--------|----------|
| 经验回放 | 打破数据相关性，提高样本效率 |
| 目标网络 | 稳定训练，避免目标振荡 |
| Double DQN | 解决 Q 值过高估计 |
| Dueling 架构 | 分离状态价值和优势函数 |

### DQN 变体对比

| 变体 | 解决问题 | 核心思想 |
|------|----------|----------|
| Double DQN | 过高估计 | 分离动作选择和价值评估 |
| Dueling DQN | 状态价值学习 | 分离 V 和 A 流 |
| PER | 样本效率 | 按 TD 误差优先采样 |
| Noisy Nets | 探索 | 参数噪声替代 ε-贪心 |

## 学习建议

### 实践要点

1. **调参建议**：学习率 `1e-4 ~ 1e-3`，目标网络更新频率 `100~1000` 步
2. **监控指标**：Q 值均值、TD 误差、探索率变化
3. **常见问题**：训练不稳定时检查目标网络更新、经验回放是否正常

### 后续方向

- **策略梯度方法**：直接优化策略参数
- **Actor-Critic**：结合价值和策略
- **分布式 DQN**：Categorical DQN（C51）、QR-DQN

---

**上一章**：[表格型方法详解](./tabular-methods.md)  
**下一章**：[策略梯度方法详解](./policy-gradient.md)
