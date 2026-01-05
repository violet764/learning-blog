# 函数逼近与深度强化学习

## 章节概述
本章将介绍函数逼近技术在强化学习中的应用，重点讲解如何用神经网络等函数逼近器处理大规模状态空间问题。这是连接传统强化学习和深度强化学习的关键技术，为后续的深度Q网络（DQN）等现代算法奠定基础。

## 核心知识点

### 1. 函数逼近基础

#### 1.1 维数灾难问题
当状态空间过大时，表格方法面临存储和计算挑战：

- **存储问题**：状态数量指数增长，无法存储完整Q表
- **泛化问题**：相似状态需要共享经验
- **计算效率**：需要高效利用有限的经验数据

#### 1.2 函数逼近原理
用参数化函数近似价值函数：

$$
V(s) ≈ \hat{V}(s; \mathbf{w})
Q(s,a) ≈ \hat{Q}(s,a; \mathbf{w})
$$

其中**w**是参数向量，可以是神经网络权重、线性系数等。

#### 1.3 逼近器类型

**线性函数逼近**：
$$
\hat{V}(s; \mathbf{w}) = \mathbf{w}^T \phi(s)
$$

**神经网络逼近**：
$$
\hat{Q}(s,a; \mathbf{w}) = f_{NN}(s,a; \mathbf{w})
$$

### 2. 梯度下降与价值函数逼近

#### 2.1 目标函数
最小化预测值与真实值的均方误差：

$$
J(\mathbf{w}) = E_π[(V_π(s) - \hat{V}(s;\mathbf{w}))^2]
$$

#### 2.2 随机梯度下降
参数更新规则：

$$
\mathbf{w} ← \mathbf{w} - \alpha \nabla_{\mathbf{w}} J(\mathbf{w})
$$

对于价值函数逼近：
$$
\mathbf{w} ← \mathbf{w} + \alpha [V_π(s) - \hat{V}(s;\mathbf{w})] \nabla_{\mathbf{w}} \hat{V}(s;\mathbf{w})
$$

#### 2.3 线性函数逼近实现
```python
import numpy as np

class LinearFunctionApproximator:
    """线性函数逼近器"""
    def __init__(self, n_features, learning_rate=0.01):
        self.w = np.zeros(n_features)
        self.alpha = learning_rate
    
    def predict(self, features):
        """预测价值"""
        return np.dot(self.w, features)
    
    def update(self, features, target):
        """更新参数"""
        prediction = self.predict(features)
        error = target - prediction
        self.w += self.alpha * error * features
    
    def get_weights(self):
        return self.w.copy()

# 特征工程示例
def tile_coding(state, n_tilings=8, n_tiles=8):
    """瓦片编码特征提取"""
    features = np.zeros(n_tilings * n_tiles)
    
    for tiling in range(n_tilings):
        # 每个瓦片网格有微小偏移
        offset = tiling / n_tilings
        scaled_state = state + offset
        
        # 离散化状态
        tile_idx = int(scaled_state * n_tiles) % n_tiles
        feature_idx = tiling * n_tiles + tile_idx
        features[feature_idx] = 1.0
    
    return features
```

### 3. 深度Q网络（DQN）

#### 3.1 DQN核心思想
DeepMind在2015年提出的突破性算法：

- **经验回放**：打破数据相关性
- **目标网络**：稳定学习过程
- **端到端学习**：直接从像素学习

#### 3.2 网络架构
```python
import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    """深度Q网络"""
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(DQN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class ConvDQN(nn.Module):
    """卷积DQN，用于图像输入"""
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
        return int(np.prod(o.size()))
    
    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)
```

#### 3.3 经验回放
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
        """随机采样批次"""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)
```

#### 3.4 完整DQN实现
```python
class DQNAgent:
    """DQN智能体"""
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, 
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 memory_size=10000, batch_size=32, target_update=100):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update = target_update
        
        # 网络
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # 经验回放
        self.memory = ReplayBuffer(memory_size)
        
        self.steps = 0
    
    def select_action(self, state):
        """选择动作"""
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储转移"""
        self.memory.push(state, action, reward, next_state, done)
    
    def update(self):
        """更新网络"""
        if len(self.memory) < self.batch_size:
            return
        
        # 采样批次
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones).unsqueeze(1)
        
        # 计算当前Q值
        current_q_values = self.policy_net(states).gather(1, actions)
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # 计算损失
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新目标网络
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.steps += 1
        
        return loss.item()
```

### 4. DQN变体与改进

#### 4.1 Double DQN
解决Q值过高估计问题：
```python
class DoubleDQNAgent(DQNAgent):
    def update(self):
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones).unsqueeze(1)
        
        # 当前Q值
        current_q_values = self.policy_net(states).gather(1, actions)
        
        # Double DQN: 用policy net选择动作，target net评估
        with torch.no_grad():
            next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.steps += 1
        return loss.item()
```

#### 4.2 Dueling DQN
分离状态价值和优势函数：
```python
class DuelingDQN(nn.Module):
    """Dueling DQN网络"""
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super(DuelingDQN, self).__init__()
        
        # 共享特征层
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 价值流
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 优势流
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # 合并价值和建议
        q_values = value + (advantages - advantages.mean())
        return q_values
```

#### 4.3 Prioritized Experience Replay
优先回放重要经验：
```python
import heapq

class PrioritizedReplayBuffer:
    """优先经验回放"""
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = np.zeros(capacity)
        self.pos = 0
        self.max_priority = 1.0
    
    def push(self, state, action, reward, next_state, done):
        """存储经验"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size):
        """根据优先级采样"""
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]
        
        # 计算采样概率
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # 采样索引
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        
        # 计算重要性采样权重
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        
        # 更新beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return indices, states, actions, rewards, next_states, dones, weights
    
    def update_priorities(self, indices, priorities):
        """更新优先级"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
```

### 5. 实战应用：CartPole环境

#### 5.1 CartPole问题描述
- **状态**：小车位置、速度、杆角度、角速度
- **动作**：向左或向右推车
- **目标**：保持杆平衡尽可能长时间

#### 5.2 DQN在CartPole中的实现
```python
import gym

def train_dqn_cartpole():
    """在CartPole环境中训练DQN"""
    env = gym.make('CartPole-v1')
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_dim, action_dim)
    
    episodes = 1000
    scores = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        while True:
            # 选择动作
            action = agent.select_action(state)
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            
            # 存储经验
            agent.store_transition(state, action, reward, next_state, done)
            
            # 更新网络
            loss = agent.update()
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        scores.append(total_reward)
        
        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"Episode {episode}, Average Score: {avg_score:.2f}, Epsilon: {agent.epsilon:.3f}")
        
        # 提前停止条件
        if np.mean(scores[-100:]) >= 195:
            print(f"Solved at episode {episode}!")
            break
    
    env.close()
    return agent, scores

# 运行训练
# agent, scores = train_dqn_cartpole()
```

## 知识点间关联逻辑

### 技术演进路径
```
线性函数逼近（基础）
    ↓
神经网络逼近（非线性）
    ↓
DQN（经验回放+目标网络）
    ↓
DQN变体（Double、Dueling、PER）
    ↓
深度强化学习家族
```

### 数学基础衔接
- **梯度下降**：参数优化基础
- **贝尔曼方程**：价值函数理论基础
- **近似动态规划**：函数逼近的理论支持

## 章节核心考点汇总

### 必须掌握的技术
1. **函数逼近原理**：处理维数灾难的核心思想
2. **DQN三大组件**：经验回放、目标网络、端到端学习
3. **DQN变体改进**：Double DQN、Dueling DQN、优先回放

### 重要概念
- **经验回放**：打破数据相关性，提高样本效率
- **目标网络**：稳定训练过程，避免振荡
- **价值函数逼近**：用神经网络近似Q函数

### 算法选择原则
- **状态空间大**：使用函数逼近
- **需要稳定训练**：DQN及其变体
- **样本效率重要**：优先经验回放

## 学习建议/后续延伸方向

### 学习建议
1. **理解逼近思想**：重点掌握函数逼近的核心价值
2. **实现完整DQN**：从零实现帮助深入理解
3. **比较不同变体**：理解每种改进的出发点和效果

### 常见错误避免
1. **忽视经验回放**：直接在线学习可能不稳定
2. **目标网络更新过快**：导致训练振荡
3. **特征工程不当**：影响逼近效果

### 后续学习方向
1. **策略梯度方法**：直接优化策略
2. **Actor-Critic架构**：结合价值和策略
3. **分布式强化学习**：多智能体协同
4. **大模型强化学习**：LLM中的RL应用

### 扩展阅读
- DeepMind DQN原始论文（2015 Nature）
- Double DQN、Dueling DQN改进论文
- OpenAI Spinning Up深度Q学习教程

---

**下一章预告**：在掌握了价值函数逼近后，下一章将介绍策略梯度方法，直接从策略空间进行优化，为更复杂的强化学习问题提供解决方案。