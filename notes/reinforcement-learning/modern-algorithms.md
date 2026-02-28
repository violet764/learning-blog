# 现代强化学习算法

本章深入讲解现代强化学习的前沿算法，这些算法在实际应用中表现出色，解决了传统方法的稳定性、样本效率和收敛性等问题。重点介绍 PPO、SAC、DDPG、TD3 等主流算法。

## 近端策略优化（PPO）

PPO（Proximal Policy Optimization）是 OpenAI 在 2017 年提出的算法，因其简单、稳定、高效而成为当前最流行的强化学习算法之一。

### 核心思想

PPO 通过限制策略更新幅度来保证训练稳定性。关键创新是**裁剪目标函数**：

$$
L^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]
$$

其中：
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$：策略比率
- $\hat{A}_t$：优势函数估计
- $\epsilon$：裁剪参数（通常 0.1~0.2）

### 广义优势估计（GAE）

$$
\hat{A}_t^{GAE(\gamma,\lambda)} = \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}
$$

其中 $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ 是 TD 残差。

### PPO 实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PPOActorCritic(nn.Module):
    """PPO Actor-Critic 网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PPOActorCritic, self).__init__()
        
        # 共享特征层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # Actor
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        features = self.shared(x)
        action_probs = self.actor(features)
        value = self.critic(features)
        return action_probs, value


class PPO:
    """PPO 算法"""
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99,
                 gae_lambda=0.95, clip_epsilon=0.2, epochs=10, batch_size=64):
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        
        self.network = PPOActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # 经验存储
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def select_action(self, state):
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            probs, value = self.network(state_tensor)
        
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        self.states.append(state)
        self.actions.append(action.item())
        self.log_probs.append(log_prob.item())
        self.values.append(value.item())
        
        return action.item()
    
    def store_transition(self, reward, done):
        """存储转移"""
        self.rewards.append(reward)
        self.dones.append(done)
    
    def compute_gae(self):
        """计算广义优势估计"""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_value = 0
            else:
                next_value = self.values[t + 1]
            
            delta = self.rewards[t] + self.gamma * next_value * (1 - self.dones[t]) - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
        
        returns = [a + v for a, v in zip(advantages, self.values)]
        return advantages, returns
    
    def update(self):
        """更新网络"""
        if len(self.states) < self.batch_size:
            return None
        
        # 计算优势和回报
        advantages, returns = self.compute_gae()
        
        # 转换为张量
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.LongTensor(self.actions)
        old_log_probs = torch.FloatTensor(self.log_probs)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 多轮优化
        total_loss = 0
        for _ in range(self.epochs):
            indices = np.random.permutation(len(states))
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]
                
                batch_states = states[idx]
                batch_actions = actions[idx]
                batch_old_log_probs = old_log_probs[idx]
                batch_advantages = advantages[idx]
                batch_returns = returns[idx]
                
                # 前向传播
                probs, values = self.network(batch_states)
                dist = torch.distributions.Categorical(probs)
                
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # 策略比率
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # PPO 裁剪目标
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                value_loss = 0.5 * nn.MSELoss()(values.squeeze(), batch_returns)
                
                # 熵奖励
                entropy_loss = -0.01 * entropy
                
                # 总损失
                loss = policy_loss + value_loss + entropy_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()
                
                total_loss += loss.item()
        
        # 清空经验
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
        return total_loss / self.epochs
```

## 深度确定性策略梯度（DDPG）

DDPG 适用于连续动作空间，结合了 DQN 和策略梯度的思想。

### 核心特点

| 特点 | 说明 |
|------|------|
| 确定性策略 | 输出具体动作值，而非概率分布 |
| 经验回放 | 打破数据相关性 |
| 目标网络 | 软更新，稳定训练 |

### DDPG 实现

```python
class DDPGActor(nn.Module):
    """DDPG Actor 网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=256, max_action=1.0):
        super(DDPGActor, self).__init__()
        self.max_action = max_action
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.max_action * self.network(x)


class DDPGCritic(nn.Module):
    """DDPG Critic 网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DDPGCritic, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        return self.network(torch.cat([state, action], dim=-1))


class DDPG:
    """DDPG 算法"""
    def __init__(self, state_dim, action_dim, max_action=1.0,
                 actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=0.005):
        
        self.gamma = gamma
        self.tau = tau
        
        # Actor 网络
        self.actor = DDPGActor(state_dim, action_dim, max_action=max_action)
        self.actor_target = DDPGActor(state_dim, action_dim, max_action=max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # Critic 网络
        self.critic = DDPGCritic(state_dim, action_dim)
        self.critic_target = DDPGCritic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.memory = []
        self.memory_capacity = 100000
    
    def select_action(self, state, noise_std=0.1):
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state_tensor).detach().numpy()[0]
        
        # 添加探索噪声
        noise = np.random.normal(0, noise_std, size=action.shape)
        action = np.clip(action + noise, -1, 1)
        
        return action
    
    def soft_update(self, target, source):
        """软更新目标网络"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def update(self, batch_size=100):
        """更新网络"""
        if len(self.memory) < batch_size:
            return None
        
        # 采样
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        # 更新 Critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + self.gamma * target_q * (1 - dones)
        
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 更新 Actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 软更新目标网络
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)
        
        return actor_loss.item(), critic_loss.item()
```

## TD3：双延迟 DDPG

TD3 解决了 DDPG 中的 Q 值过高估计问题。

### 核心改进

| 技术 | 说明 |
|------|------|
| 双 Q 网络 | 使用两个 Critic，取最小值 |
| 目标策略平滑 | 添加噪声防止过拟合 |
| 延迟更新 | Actor 更新频率低于 Critic |

### TD3 实现

```python
class TD3:
    """TD3 算法"""
    def __init__(self, state_dim, action_dim, max_action=1.0,
                 actor_lr=1e-3, critic_lr=1e-3, gamma=0.99, tau=0.005,
                 policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        
        self.gamma = gamma
        self.tau = tau
        self.max_action = max_action
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        
        # Actor
        self.actor = DDPGActor(state_dim, action_dim, max_action=max_action)
        self.actor_target = DDPGActor(state_dim, action_dim, max_action=max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        
        # 双 Critic
        self.critic1 = DDPGCritic(state_dim, action_dim)
        self.critic2 = DDPGCritic(state_dim, action_dim)
        self.critic1_target = DDPGCritic(state_dim, action_dim)
        self.critic2_target = DDPGCritic(state_dim, action_dim)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=critic_lr
        )
        
        self.memory = []
        self.total_it = 0
    
    def update(self, batch_size=100):
        """更新网络"""
        self.total_it += 1
        
        if len(self.memory) < batch_size:
            return None
        
        # ... 采样逻辑同 DDPG ...
        
        # 目标策略平滑
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-self.max_action, self.max_action)
            
            # 双 Q 学习，取最小值
            target_q1 = self.critic1_target(next_states, next_actions)
            target_q2 = self.critic2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + self.gamma * target_q * (1 - dones)
        
        # 更新 Critic
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 延迟更新 Actor
        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic1(states, self.actor(states)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # 软更新
            self._soft_update(self.actor_target, self.actor)
            self._soft_update(self.critic1_target, self.critic1)
            self._soft_update(self.critic2_target, self.critic2)
```

## SAC：软演员-评论家

SAC 结合了最大熵强化学习，具有良好的探索能力和鲁棒性。

### 最大熵目标

$$
J(\pi) = \sum_{t=0}^{T} \mathbb{E}_{(s_t, a_t) \sim \rho_\pi}\left[r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))\right]
$$

其中 $\mathcal{H}$ 是策略熵，$\alpha$ 是温度参数。

### SAC 实现

```python
class SACActor(nn.Module):
    """SAC Actor 网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(SACActor, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        self.log_std_min = -20
        self.log_std_max = 2
    
    def forward(self, x):
        features = self.network(x)
        mu = self.mu(features)
        log_std = torch.clamp(self.log_std(features), self.log_std_min, self.log_std_max)
        return mu, log_std
    
    def sample(self, x):
        """采样动作（重参数化）"""
        mu, log_std = self.forward(x)
        std = log_std.exp()
        
        normal = torch.distributions.Normal(mu, std)
        x_t = normal.rsample()  # 重参数化
        
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob


class SAC:
    """SAC 算法"""
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, tau=0.005,
                 alpha=0.2, auto_entropy=True):
        
        self.gamma = gamma
        self.tau = tau
        self.auto_entropy = auto_entropy
        
        # Actor
        self.actor = SACActor(state_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        # 双 Critic
        self.critic1 = DDPGCritic(state_dim, action_dim)
        self.critic2 = DDPGCritic(state_dim, action_dim)
        self.critic1_target = DDPGCritic(state_dim, action_dim)
        self.critic2_target = DDPGCritic(state_dim, action_dim)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr
        )
        
        # 自动熵调整
        if auto_entropy:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = alpha
        
        self.memory = []
    
    def update(self, batch_size=256):
        """更新网络"""
        if len(self.memory) < batch_size:
            return None
        
        # ... 采样逻辑 ...
        
        # 计算目标 Q 值（包含熵奖励）
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_next = self.critic1_target(next_states, next_actions)
            q2_next = self.critic2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            target_q = rewards + self.gamma * q_next * (1 - dones)
        
        # 更新 Critic
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 更新 Actor
        new_actions, log_probs = self.actor.sample(states)
        q1_new = self.critic1(states, new_actions)
        q2_new = self.critic2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha.detach() * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 自动调整温度
        if self.auto_entropy:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp()
        
        # 软更新
        self._soft_update(self.critic1_target, self.critic1)
        self._soft_update(self.critic2_target, self.critic2)
```

## 算法对比

| 算法 | 动作空间 | 策略类型 | 主要特点 | 适用场景 |
|------|----------|----------|----------|----------|
| PPO | 离散/连续 | 随机 | 稳定、易调参 | 通用、首选 |
| DDPG | 连续 | 确定性 | 经典连续控制 | 简单连续任务 |
| TD3 | 连续 | 确定性 | 解决过高估计 | 精确控制 |
| SAC | 连续 | 随机 | 最大熵、自动温度 | 鲁棒控制 |

## 核心考点

### 必须掌握

1. **PPO 裁剪机制**：理解为什么裁剪能保证稳定性
2. **TD3 三大改进**：双 Q 网络、目标策略平滑、延迟更新
3. **SAC 最大熵**：理解熵正则化的作用和自动温度调整
4. **软更新**：$\theta_{target} \leftarrow \tau \theta + (1-\tau) \theta_{target}$

### 超参数建议

| 参数 | PPO | TD3 | SAC |
|------|-----|-----|-----|
| 学习率 | 3e-4 | 1e-3 | 3e-4 |
| γ | 0.99 | 0.99 | 0.99 |
| τ | - | 0.005 | 0.005 |
| batch_size | 64 | 100 | 256 |

## 学习建议

### 实践要点

1. **算法选择**：优先尝试 PPO，连续控制用 SAC
2. **调参策略**：从论文默认值开始，逐步微调
3. **监控指标**：策略熵、Q 值均值、KL 散度

### 后续方向

- **分布式 RL**：A3C、IMPALA、PPO-MAPPO
- **离线 RL**：CQL、BCQ、IQL
- **大模型应用**：RLHF 中的 PPO

---

**上一章**：[策略梯度方法详解](./policy-gradient.md)  
**下一章**：[多智能体强化学习](./multi-agent-rl.md)
