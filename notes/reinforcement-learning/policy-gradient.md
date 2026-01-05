# 策略梯度方法详解

## 章节概述
本章将深入讲解策略梯度方法，这是强化学习中直接优化策略参数的重要技术。与价值函数方法不同，策略梯度方法直接在策略空间进行搜索，特别适用于连续动作空间和高维问题。我们将从REINFORCE算法开始，逐步介绍Actor-Critic架构及其现代变体。

## 核心知识点

### 1. 策略梯度基本概念

#### 1.1 策略参数化
将策略表示为参数化函数：

$$
π_θ(a|s) = P(A_t = a | S_t = s, θ_t = θ)
$$

其中θ是策略参数，可以是神经网络权重等。

#### 1.2 目标函数
策略梯度方法的目标是最大化期望回报：

$$
J(θ) = E_π[∑_{t=0}^∞ γ^t r_t]
$$

#### 1.3 策略梯度定理
策略梯度定理给出了目标函数对参数的梯度：

$$
∇_θ J(θ) = E_π[∇_θ log π_θ(a|s) Q_π(s,a)]
$$

这个定理是策略梯度方法的基础。

### 2. REINFORCE算法

#### 2.1 算法原理
REINFORCE（蒙特卡洛策略梯度）是最基本的策略梯度算法：

$$
∇_θ J(θ) ≈ \frac{1}{N} ∑_{i=1}^N ∑_{t=0}^T ∇_θ log π_θ(a_t^i|s_t^i) G_t^i
$$

其中G_t^i是从时刻t开始的累积回报。

#### 2.2 REINFORCE实现
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
            nn.Softmax(dim=-1)  # 输出动作概率
        )
    
    def forward(self, x):
        return self.network(x)

class REINFORCE:
    """REINFORCE算法"""
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        self.gamma = gamma
        
        # 策略网络
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # 存储经验
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
    
    def select_action(self, state):
        """根据策略选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.policy_net(state_tensor)
        
        # 根据概率分布采样动作
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        
        return action.item()
    
    def store_transition(self, state, action, reward):
        """存储转移"""
        self.episode_states.append(state)
        self.episode_actions.append(action)
        self.episode_rewards.append(reward)
    
    def update(self):
        """更新策略"""
        # 计算回报
        returns = []
        G = 0
        for r in reversed(self.episode_rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        # 标准化回报（减少方差）
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # 计算策略梯度
        policy_loss = []
        for state, action, G in zip(self.episode_states, self.episode_actions, returns):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_tensor = torch.LongTensor([action])
            
            # 计算log概率
            action_probs = self.policy_net(state_tensor)
            log_prob = torch.log(action_probs.gather(1, action_tensor.unsqueeze(1)))
            
            # 策略梯度损失
            policy_loss.append(-log_prob * G)
        
        # 优化
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        # 清空经验
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        
        return policy_loss.item()
```

#### 2.3 基线技巧
为减少方差，引入基线b(s)：

$$
∇_θ J(θ) = E_π[∇_θ log π_θ(a|s) (Q_π(s,a) - b(s))]
$$

常用基线：状态价值函数V(s)

### 3. Actor-Critic方法

#### 3.1 基本思想
结合策略梯度（Actor）和价值函数（Critic）：

- **Actor**：负责选择动作
- **Critic**：负责评估动作价值

#### 3.2 优势函数
使用优势函数A(s,a) = Q(s,a) - V(s)作为Critic的评估：

$$
∇_θ J(θ) = E_π[∇_θ log π_θ(a|s) A_π(s,a)]
$$

#### 3.3 Actor-Critic实现
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
            nn.Linear(hidden_dim, 1)  # 输出状态价值
        )
    
    def forward(self, x):
        return self.network(x)

class ActorCritic:
    """Actor-Critic算法"""
    def __init__(self, state_dim, action_dim, lr_actor=1e-3, lr_critic=1e-3, gamma=0.99):
        self.gamma = gamma
        
        # Actor网络
        self.actor = PolicyNetwork(state_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        # Critic网络
        self.critic = ValueNetwork(state_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # 存储经验
        self.episode_data = []
    
    def select_action(self, state):
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.actor(state_tensor)
        
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        
        # 存储log概率用于后续计算
        log_prob = action_dist.log_prob(action)
        
        return action.item(), log_prob
    
    def store_transition(self, state, log_prob, reward, next_state, done):
        """存储转移"""
        self.episode_data.append({
            'state': state,
            'log_prob': log_prob,
            'reward': reward,
            'next_state': next_state,
            'done': done
        })
    
    def update(self):
        """更新网络"""
        if len(self.episode_data) == 0:
            return
        
        # 计算优势函数
        advantages = []
        returns = []
        
        # 计算每个状态的回报
        G = 0
        for t in reversed(range(len(self.episode_data))):
            reward = self.episode_data[t]['reward']
            G = reward + self.gamma * G
            returns.insert(0, G)
        
        # 计算优势
        for t in range(len(self.episode_data)):
            state = self.episode_data[t]['state']
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # 状态价值
            state_value = self.critic(state_tensor).item()
            
            # 优势 = 回报 - 基线（状态价值）
            advantage = returns[t] - state_value
            advantages.append(advantage)
        
        # 标准化优势
        advantages = torch.FloatTensor(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 更新Actor
        actor_loss = 0
        for t in range(len(self.episode_data)):
            log_prob = self.episode_data[t]['log_prob']
            advantage = advantages[t]
            
            actor_loss += -log_prob * advantage
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 更新Critic
        critic_loss = 0
        for t in range(len(self.episode_data)):
            state = self.episode_data[t]['state']
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # 目标价值
            target_value = returns[t]
            
            # 预测价值
            predicted_value = self.critic(state_tensor).squeeze()
            
            critic_loss += (predicted_value - target_value) ** 2
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 清空经验
        self.episode_data = []
        
        return actor_loss.item(), critic_loss.item()
```

### 4. 连续动作空间

#### 4.1 高斯策略
对于连续动作空间，使用高斯分布：

$$
π_θ(a|s) = \mathcal{N}(μ_θ(s), σ_θ^2(s))
$$

#### 4.2 连续Actor-Critic
```python
class ContinuousPolicyNetwork(nn.Module):
    """连续动作策略网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ContinuousPolicyNetwork, self).__init__()
        self.action_dim = action_dim
        
        # 共享特征层
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 均值输出
        self.mu = nn.Linear(hidden_dim, action_dim)
        
        # 标准差输出（使用softplus确保正值）
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, x):
        features = self.shared(x)
        mu = torch.tanh(self.mu(features))  # 限制在[-1,1]
        std = torch.nn.functional.softplus(self.log_std)
        return mu, std

class ContinuousActorCritic(ActorCritic):
    """连续动作空间的Actor-Critic"""
    def __init__(self, state_dim, action_dim, **kwargs):
        self.gamma = kwargs.get('gamma', 0.99)
        
        # 连续动作策略网络
        self.actor = ContinuousPolicyNetwork(state_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), 
                                         lr=kwargs.get('lr_actor', 1e-3))
        
        # Critic网络
        self.critic = ValueNetwork(state_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), 
                                          lr=kwargs.get('lr_critic', 1e-3))
        
        self.episode_data = []
    
    def select_action(self, state):
        """选择连续动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        mu, std = self.actor(state_tensor)
        
        # 创建正态分布
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        
        # 计算log概率
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        # 限制动作范围
        action = torch.tanh(action)
        
        return action.detach().numpy()[0], log_prob
```

### 5. 实战应用：Pendulum环境

#### 5.1 Pendulum问题描述
- **状态**：角度、角速度
- **动作**：扭矩（连续值）
- **目标**：保持杆直立

#### 5.2 连续Actor-Critic实现
```python
import gym

def train_continuous_ac():
    """在Pendulum环境中训练连续Actor-Critic"""
    env = gym.make('Pendulum-v1')
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = ContinuousActorCritic(state_dim, action_dim)
    
    episodes = 1000
    scores = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        
        while True:
            # 选择动作
            action, log_prob = agent.select_action(state)
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            
            # 存储经验
            agent.store_transition(state, log_prob, reward, next_state, done)
            
            total_reward += reward
            state = next_state
            
            if done:
                # 回合结束，更新网络
                actor_loss, critic_loss = agent.update()
                break
        
        scores.append(total_reward)
        
        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f"Episode {episode}, Average Score: {avg_score:.2f}")
    
    env.close()
    return agent, scores

# 运行训练
# agent, scores = train_continuous_ac()
```

## 知识点间关联逻辑

### 方法演进关系
```
REINFORCE（蒙特卡洛策略梯度）
    ↓
带基线的REINFORCE（减少方差）
    ↓
Actor-Critic（结合价值函数）
    ↓
优势Actor-Critic（使用优势函数）
    ↓
现代策略梯度方法（PPO、TRPO等）
```

### 数学基础衔接
- **概率论**：策略分布、期望计算
- **梯度下降**：参数优化
- **时序差分**：Critic网络的学习

## 章节核心考点汇总

### 必须掌握的算法
1. **REINFORCE**：基本的蒙特卡洛策略梯度
2. **Actor-Critic**：结合策略和价值的框架
3. **优势函数**：减少方差的技巧

### 重要概念
- **策略参数化**：将策略表示为可优化函数
- **策略梯度定理**：策略优化的理论基础
- **基线技巧**：减少方差的关键技术

### 连续动作空间处理
- **高斯策略**：连续动作的概率分布
- **重参数化技巧**：可微分的采样方法
- **动作边界处理**：tanh等激活函数

## 学习建议/后续延伸方向

### 学习建议
1. **理解梯度推导**：掌握策略梯度定理的数学基础
2. **实现完整算法**：从REINFORCE到Actor-Critic
3. **比较方法差异**：理解不同策略梯度方法的优缺点

### 常见错误避免
1. **忽视基线**：REINFORCE方差过大
2. **Critic网络过拟合**：影响Actor学习
3. **学习率设置不当**：策略更新不稳定

### 后续学习方向
1. **信任域方法**：TRPO、PPO等稳定算法
2. **分布式策略梯度**：A3C、IMPALA等
3. **逆强化学习**：从示范中学习奖励函数
4. **大模型策略优化**：RLHF等技术

### 扩展阅读
- Sutton & Barto《强化学习导论》第13章
- OpenAI Spinning Up策略梯度教程
- DeepMind IMPALA论文

---

**下一章预告**：在掌握了策略梯度方法后，下一章将介绍现代强化学习算法，包括PPO、SAC、DDPG等前沿技术，这些算法在实际应用中表现出色。