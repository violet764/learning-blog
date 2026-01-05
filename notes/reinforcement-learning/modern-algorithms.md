# 现代强化学习算法

## 章节概述
本章将深入讲解现代强化学习的前沿算法，这些算法在实际应用中表现出色，解决了传统方法的稳定性、样本效率和收敛性等问题。我们将重点介绍PPO、SAC、DDPG、TD3等算法，它们代表了当前强化学习研究的最高水平。

## 核心知识点

### 1. 近端策略优化（PPO）

#### 1.1 算法背景
PPO（Proximal Policy Optimization）是OpenAI在2017年提出的算法，解决了策略梯度方法中步长选择困难的问题。

#### 1.2 裁剪目标函数
PPO的核心创新是裁剪的目标函数：

$$
L^{CLIP}(θ) = E_t[\min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
$$

其中：
- **r_t(θ)** = π_θ(a_t|s_t) / π_θ_old(a_t|s_t) 策略比率
- **A_t** 优势函数
- **ε** 裁剪参数（通常0.1-0.2）

#### 1.3 PPO实现
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PPONetwork(nn.Module):
    """PPO网络（Actor-Critic共享特征）"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PPONetwork, self).__init__()
        
        # 共享特征层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Actor输出（动作概率）
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic输出（状态价值）
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        features = self.shared(x)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value

class PPO:
    """PPO算法"""
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, 
                 gae_lambda=0.95, clip_epsilon=0.2, epochs=10, batch_size=64):
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        
        # 网络
        self.network = PPONetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # 存储经验
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def select_action(self, state):
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs, value = self.network(state_tensor)
        
        # 采样动作
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def store_transition(self, state, action, log_prob, value, reward, done):
        """存储转移"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_advantages(self):
        """计算广义优势估计（GAE）"""
        advantages = []
        gae = 0
        
        # 反向计算优势
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_value = 0  # 终止状态价值为0
            else:
                next_value = self.values[t+1]
            
            delta = self.rewards[t] + self.gamma * next_value * (1 - self.dones[t]) - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
        
        returns = [adv + val for adv, val in zip(advantages, self.values)]
        
        return advantages, returns
    
    def update(self):
        """更新网络"""
        if len(self.states) < self.batch_size:
            return
        
        # 转换为张量
        states = torch.FloatTensor(self.states)
        actions = torch.LongTensor(self.actions)
        old_log_probs = torch.FloatTensor(self.log_probs)
        
        # 计算优势和回报
        advantages, returns = self.compute_advantages()
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 多轮优化
        for _ in range(self.epochs):
            # 随机打乱数据
            indices = np.random.permutation(len(states))
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 前向传播
                action_probs, values = self.network(batch_states)
                dist = torch.distributions.Categorical(action_probs)
                
                # 新策略的log概率
                new_log_probs = dist.log_prob(batch_actions)
                
                # 策略比率
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # 裁剪目标函数
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值函数损失
                value_loss = nn.MSELoss()(values.squeeze(), batch_returns)
                
                # 总损失
                loss = policy_loss + 0.5 * value_loss
                
                # 优化
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)  # 梯度裁剪
                self.optimizer.step()
        
        # 清空经验
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
```

### 2. 深度确定性策略梯度（DDPG）

#### 2.1 算法特点
DDPG适用于连续动作空间，结合了DQN和策略梯度的思想：
- **确定性策略**：输出具体动作值而非概率分布
- **经验回放**：借鉴DQN的经验存储
- **目标网络**：稳定训练过程

#### 2.2 DDPG实现
```python
class ActorNetwork(nn.Module):
    """DDPG Actor网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=256, max_action=1.0):
        super(ActorNetwork, self).__init__()
        self.max_action = max_action
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 输出在[-1,1]范围内
        )
    
    def forward(self, x):
        return self.max_action * self.network(x)

class CriticNetwork(nn.Module):
    """DDPG Critic网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(CriticNetwork, self).__init__()
        
        # 输入状态和动作
        self.layer1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)

class DDPG:
    """DDPG算法"""
    def __init__(self, state_dim, action_dim, max_action=1.0, 
                 actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=0.005):
        
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action
        
        # Actor网络
        self.actor = ActorNetwork(state_dim, action_dim, max_action=max_action)
        self.actor_target = ActorNetwork(state_dim, action_dim, max_action=max_action)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # Critic网络
        self.critic = CriticNetwork(state_dim, action_dim)
        self.critic_target = CriticNetwork(state_dim, action_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # 硬更新目标网络
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 经验回放
        self.memory = ReplayBuffer(100000)
    
    def select_action(self, state, noise_std=0.1):
        """选择动作（添加探索噪声）"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state_tensor).detach().numpy()[0]
        
        # 添加探索噪声
        noise = np.random.normal(0, noise_std, size=action.shape)
        action = np.clip(action + noise, -self.max_action, self.max_action)
        
        return action
    
    def update(self, batch_size=100):
        """更新网络"""
        if len(self.memory) < batch_size:
            return
        
        # 采样批次
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        # 更新Critic
        next_actions = self.actor_target(next_states)
        next_q_values = self.critic_target(next_states, next_actions)
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        current_q_values = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q_values, target_q_values.detach())
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 更新Actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 软更新目标网络
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return actor_loss.item(), critic_loss.item()
```

### 3. 双延迟深度确定性策略梯度（TD3）

#### 3.1 TD3改进
TD3解决了DDPG中的Q值过高估计问题：
- **双Q网络**：使用两个Critic网络取最小值
- **目标策略平滑**：添加噪声防止策略过拟合
- **延迟更新**：策略网络更新频率低于价值网络

#### 3.2 TD3实现
```python
class TD3:
    """TD3算法"""
    def __init__(self, state_dim, action_dim, max_action=1.0, 
                 actor_lr=1e-3, critic_lr=1e-3, gamma=0.99, tau=0.005,
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        
        # Actor网络
        self.actor = ActorNetwork(state_dim, action_dim, max_action=max_action)
        self.actor_target = ActorNetwork(state_dim, action_dim, max_action=max_action)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # 双Critic网络
        self.critic1 = CriticNetwork(state_dim, action_dim)
        self.critic2 = CriticNetwork(state_dim, action_dim)
        self.critic1_target = CriticNetwork(state_dim, action_dim)
        self.critic2_target = CriticNetwork(state_dim, action_dim)
        
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), 
            lr=critic_lr
        )
        
        # 硬更新目标网络
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        self.memory = ReplayBuffer(100000)
        self.total_it = 0
    
    def select_action(self, state, noise_std=0.1):
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state_tensor).detach().numpy()[0]
        
        noise = np.random.normal(0, noise_std, size=action.shape)
        action = np.clip(action + noise, -self.max_action, self.max_action)
        
        return action
    
    def update(self, batch_size=100):
        """更新网络"""
        self.total_it += 1
        
        if len(self.memory) < batch_size:
            return
        
        # 采样批次
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        with torch.no_grad():
            # 目标策略平滑
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-self.max_action, self.max_action)
            
            # 双Q学习
            target_q1 = self.critic1_target(next_states, next_actions)
            target_q2 = self.critic2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        # 更新Critic
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 延迟更新Actor
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic1(states, self.actor(states)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # 软更新目标网络
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
            for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return actor_loss.item() if self.total_it % self.policy_freq == 0 else 0, critic_loss.item()
```

### 4. 软演员-评论家（SAC）

#### 4.1 SAC特点
SAC结合了最大熵强化学习的优势：
- **最大熵目标**：鼓励探索，提高鲁棒性
- **自动温度调整**：自适应熵权重
- **离线策略学习**：高效利用经验

#### 4.2 SAC实现
```python
class SACActor(nn.Module):
    """SAC Actor网络（输出高斯分布的均值和标准差）"""
    def __init__(self, state_dim, action_dim, hidden_dim=256, max_action=1.0):
        super(SACActor, self).__init__()
        self.max_action = max_action
        self.log_std_max = 2
        self.log_std_min = -20
        
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        features = self.shared(x)
        mu = self.mu(features)
        log_std = self.log_std(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std
    
    def sample(self, x):
        """采样动作（使用重参数化技巧）"""
        mu, log_std = self.forward(x)
        std = log_std.exp()
        
        # 重参数化
        normal = torch.distributions.Normal(mu, std)
        x_t = normal.rsample()  # 重参数化采样
        
        # 应用tanh确保动作在有效范围内
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action * self.max_action, log_prob

class SAC:
    """SAC算法"""
    def __init__(self, state_dim, action_dim, max_action=1.0, 
                 actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4,
                 gamma=0.99, tau=0.005, alpha=0.2, automatic_entropy_tuning=True):
        
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action
        self.automatic_entropy_tuning = automatic_entropy_tuning
        
        # Actor网络
        self.actor = SACActor(state_dim, action_dim, max_action=max_action)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # 双Critic网络
        self.critic1 = CriticNetwork(state_dim, action_dim)
        self.critic2 = CriticNetwork(state_dim, action_dim)
        self.critic1_target = CriticNetwork(state_dim, action_dim)
        self.critic2_target = CriticNetwork(state_dim, action_dim)
        
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), 
            lr=critic_lr
        )
        
        # 硬更新目标网络
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # 自动熵调整
        if automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor([action_dim])).item()
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        else:
            self.alpha = alpha
        
        self.memory = ReplayBuffer(1000000)
    
    def select_action(self, state, evaluate=False):
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        if evaluate:
            # 评估时使用均值
            mu, _ = self.actor(state_tensor)
            return torch.tanh(mu).detach().numpy()[0] * self.max_action
        else:
            # 训练时采样
            action, _ = self.actor.sample(state_tensor)
            return action.detach().numpy()[0]
    
    def update(self, batch_size=256):
        """更新网络"""
        if len(self.memory) < batch_size:
            return
        
        # 采样批次
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        with torch.no_grad():
            # 下一个状态的动作和log概率
            next_actions, next_log_probs = self.actor.sample(next_states)
            
            # 目标Q值（包含熵奖励）
            next_q1 = self.critic1_target(next_states, next_actions)
            next_q2 = self.critic2_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # 更新Critic
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 更新Actor
        new_actions, log_probs = self.actor.sample(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 自动调整温度参数
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
        
        # 软更新目标网络
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        return actor_loss.item(), critic_loss.item()
```

## 知识点间关联逻辑

### 算法演进关系
```
DDPG（连续动作DQN）
    ↓
TD3（解决Q值过高估计）
    ↓
SAC（最大熵强化学习）
    ↓
PPO（离散和连续动作通用）
```

### 技术特点对比
| 算法 | 动作空间 | 策略类型 | 主要特点 |
|------|----------|----------|----------|
| PPO | 离散/连续 | 随机策略 | 稳定、易调参 |
| DDPG | 连续 | 确定性策略 | 经验回放、目标网络 |
| TD3 | 连续 | 确定性策略 | 双Q网络、延迟更新 |
| SAC | 连续 | 随机策略 | 最大熵、自动温度调整 |

## 章节核心考点汇总

### 必须掌握的算法
1. **PPO**：最流行的策略梯度算法
2. **DDPG/TD3**：连续动作空间的标准算法
3. **SAC**：最大熵强化学习的代表

### 重要改进技术
- **裁剪目标函数**：PPO的稳定性保证
- **双Q学习**：TD3解决过高估计
- **最大熵目标**：SAC的探索优势
- **自动温度调整**：SAC的自适应能力

### 算法选择原则
- **需要稳定性**：PPO
- **连续动作空间**：TD3或SAC
- **样本效率重要**：SAC（离线策略）
- **简单易用**：PPO

## 学习建议/后续延伸方向

### 学习建议
1. **理解算法思想**：掌握每种算法的核心创新点
2. **比较算法差异**：理解不同算法的适用场景
3. **动手调参实践**：通过实验掌握超参数影响

### 常见错误避免
1. **忽视算法适用性**：根据问题特点选择算法
2. **超参数设置不当**：参考论文推荐值
3. **实现细节错误**：注意梯度裁剪、目标网络更新等

### 后续学习方向
1. **分布式强化学习**：A3C、IMPALA等
2. **多智能体强化学习**：MADDPG、QMIX等
3. **离线强化学习**：CQL、BCQ等
4. **大模型强化学习**：RLHF、PPO在LLM中的应用

### 扩展阅读
- PPO原始论文（OpenAI, 2017）
- TD3原始论文（Fujimoto et al., 2018）
- SAC原始论文（Haarnoja et al., 2018）
- Spinning Up现代强化学习教程

---

**下一章预告**：在掌握了现代强化学习算法后，下一章将介绍多智能体强化学习，探讨多个智能体在复杂环境中的协同与竞争问题。