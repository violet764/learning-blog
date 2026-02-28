# 多智能体强化学习

本章介绍多智能体强化学习（Multi-Agent Reinforcement Learning, MARL），这是强化学习的重要扩展，涉及多个智能体在共享环境中的协同与竞争。多智能体系统面临非平稳性、信用分配、通信协调等独特挑战。

## 问题定义

### 多智能体 MDP（MMDP）

扩展 MDP 到多智能体场景：

$$
\text{MMDP} = (\mathcal{S}, \mathcal{A}_1 \times \cdots \times \mathcal{A}_N, P, R_1, \ldots, R_N, \gamma)
$$

| 要素 | 说明 |
|------|------|
| 联合状态空间 | $\mathcal{S}$：全局状态 |
| 联合动作空间 | $\mathcal{A} = \mathcal{A}_1 \times \cdots \times \mathcal{A}_N$ |
| 状态转移 | $P(s'|s, a_1, \ldots, a_N)$：联合动作决定转移 |
| 奖励函数 | $R_i(s, a_1, \ldots, a_N)$：每个智能体可能获得不同奖励 |

### 关键挑战

| 挑战 | 说明 |
|------|------|
| **非平稳性** | 其他智能体的策略在变化，环境不再平稳 |
| **信用分配** | 如何将联合奖励合理分配给各智能体 |
| **可扩展性** | 状态-动作空间随智能体数量指数增长 |
| **通信协调** | 智能体间如何有效交换信息 |

## 学习框架

### 完全分布式（IQL）

每个智能体独立学习，忽略其他智能体：

```python
class IndependentQLearning:
    """独立 Q 学习"""
    def __init__(self, n_agents, state_dim, action_dim):
        self.agents = [
            DQNAgent(state_dim, action_dim) 
            for _ in range(n_agents)
        ]
    
    def select_actions(self, states):
        return [agent.select_action(s) for agent, s in zip(self.agents, states)]
    
    def update(self, experiences):
        for agent, exp in zip(self.agents, experiences):
            agent.update(exp)
```

**优点**：简单、可扩展  
**缺点**：环境非平稳，收敛性无保证

### 集中训练分散执行（CTDE）

训练时利用全局信息，执行时只依赖局部观察：

| 阶段 | 可用信息 |
|------|----------|
| 训练 | 全局状态、所有智能体动作 |
| 执行 | 局部观察 |

这是当前 MARL 的主流框架。

## MADDPG

多智能体深度确定性策略梯度（MADDPG）将 DDPG 扩展到多智能体场景。

### 核心思想

- **Actor**：只使用局部观察，分散执行
- **Critic**：使用全局状态和所有动作，集中训练

```python
import torch
import torch.nn as nn
import torch.optim as optim

class MADDPG:
    """MADDPG 算法"""
    def __init__(self, n_agents, obs_dims, action_dims, hidden_dim=64, 
                 lr=0.01, gamma=0.95, tau=0.01):
        
        self.n_agents = n_agents
        self.gamma = gamma
        self.tau = tau
        
        # 每个智能体的网络
        self.actors = []
        self.critics = []
        self.actor_targets = []
        self.critic_targets = []
        self.actor_optimizers = []
        self.critic_optimizers = []
        
        total_obs_dim = sum(obs_dims)
        total_action_dim = sum(action_dims)
        
        for i in range(n_agents):
            # Actor 网络（分散，只用局部观察）
            actor = nn.Sequential(
                nn.Linear(obs_dims[i], hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dims[i]),
                nn.Tanh()
            )
            self.actors.append(actor)
            self.actor_targets.append(self._copy_network(actor))
            self.actor_optimizers.append(optim.Adam(actor.parameters(), lr=lr))
            
            # Critic 网络（集中，用全局状态和所有动作）
            critic = nn.Sequential(
                nn.Linear(total_obs_dim + total_action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            self.critics.append(critic)
            self.critic_targets.append(self._copy_network(critic))
            self.critic_optimizers.append(optim.Adam(critic.parameters(), lr=lr))
    
    def _copy_network(self, network):
        """复制网络"""
        copy = type(network)()
        copy.load_state_dict(network.state_dict())
        return copy
    
    def select_actions(self, observations, noise_std=0.1):
        """分散执行：每个智能体独立决策"""
        actions = []
        for i, (actor, obs) in enumerate(zip(self.actors, observations)):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action = actor(obs_tensor).detach().numpy()[0]
            
            # 添加探索噪声
            noise = np.random.normal(0, noise_std, size=action.shape)
            action = np.clip(action + noise, -1, 1)
            actions.append(action)
        
        return actions
    
    def update(self, batch):
        """集中训练"""
        obs, actions, rewards, next_obs, dones = batch
        
        # 转换为张量
        obs = [torch.FloatTensor(o) for o in obs]
        actions = [torch.FloatTensor(a) for a in actions]
        rewards = [torch.FloatTensor(r) for r in rewards]
        next_obs = [torch.FloatTensor(o) for o in next_obs]
        dones = [torch.FloatTensor(d) for d in dones]
        
        # 拼接全局状态和动作
        global_obs = torch.cat(obs, dim=-1)
        global_actions = torch.cat(actions, dim=-1)
        global_next_obs = torch.cat(next_obs, dim=-1)
        
        for i in range(self.n_agents):
            # 计算目标 Q 值
            with torch.no_grad():
                next_actions = [
                    self.actor_targets[j](next_obs[j]) 
                    for j in range(self.n_agents)
                ]
                global_next_actions = torch.cat(next_actions, dim=-1)
                target_q = self.critic_targets[i](
                    torch.cat([global_next_obs, global_next_actions], dim=-1)
                )
                target_q = rewards[i] + self.gamma * target_q * (1 - dones[i])
            
            # 更新 Critic
            current_q = self.critics[i](
                torch.cat([global_obs, global_actions], dim=-1)
            )
            critic_loss = nn.MSELoss()(current_q, target_q)
            
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()
            
            # 更新 Actor
            pred_actions = [
                self.actors[j](obs[j]) if j == i else actions[j].detach()
                for j in range(self.n_agents)
            ]
            global_pred_actions = torch.cat(pred_actions, dim=-1)
            actor_loss = -self.critics[i](
                torch.cat([global_obs, global_pred_actions], dim=-1)
            ).mean()
            
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()
            
            # 软更新目标网络
            self._soft_update(self.actor_targets[i], self.actors[i])
            self._soft_update(self.critic_targets[i], self.critics[i])
    
    def _soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
```

## MAPPO

多智能体 PPO（MAPPO）是当前最流行的 MARL 算法之一。

### 核心思想

- 使用 PPO 作为基础算法
- 中心化 Critic，分散化 Actor
- 简单高效，效果优异

```python
class MAPPO:
    """多智能体 PPO"""
    def __init__(self, n_agents, obs_dims, action_dim, hidden_dim=64,
                 lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2):
        
        self.n_agents = n_agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        
        total_obs_dim = sum(obs_dims)
        
        # 每个智能体的 Actor（分散）
        self.actors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dims[i], hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim),
                nn.Softmax(dim=-1)
            ) for i in range(n_agents)
        ])
        
        # 共享的中心化 Critic
        self.critic = nn.Sequential(
            nn.Linear(total_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.optimizer = optim.Adam(
            list(self.actors.parameters()) + list(self.critic.parameters()),
            lr=lr
        )
        
        # 经验存储
        self.storage = [[] for _ in range(n_agents)]
    
    def select_actions(self, observations):
        """分散选择动作"""
        actions = []
        log_probs = []
        
        for i, (actor, obs) in enumerate(zip(self.actors, observations)):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            probs = actor(obs_tensor)
            
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            actions.append(action.item())
            log_probs.append(log_prob.item())
        
        return actions, log_probs
    
    def update(self, batch_data):
        """更新网络"""
        # 计算 GAE 和 PPO 更新
        # 实现类似单智能体 PPO，但 Critic 使用全局状态
        pass
```

## 通信机制

### 显式通信

智能体间传递消息：

```python
class CommNetwork(nn.Module):
    """通信网络"""
    def __init__(self, obs_dim, msg_dim, hidden_dim=64):
        super(CommNetwork, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.msg_layer = nn.Linear(hidden_dim, msg_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + msg_dim, hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, obs, neighbor_msgs):
        # 编码自身观察
        encoded = self.encoder(obs)
        
        # 生成消息
        msg = self.msg_layer(encoded)
        
        # 聚合邻居消息
        if neighbor_msgs:
            aggregated = torch.mean(torch.stack(neighbor_msgs), dim=0)
        else:
            aggregated = torch.zeros_like(msg)
        
        # 解码
        output = self.decoder(torch.cat([encoded, aggregated], dim=-1))
        
        return output, msg
```

### 注意力通信

```python
class AttentionComm(nn.Module):
    """基于注意力的通信"""
    def __init__(self, input_dim, key_dim=64, value_dim=64):
        super(AttentionComm, self).__init__()
        
        self.query = nn.Linear(input_dim, key_dim)
        self.key = nn.Linear(input_dim, key_dim)
        self.value = nn.Linear(input_dim, value_dim)
    
    def forward(self, self_feat, other_feats):
        # 生成 query, key, value
        q = self.query(self_feat)
        keys = torch.stack([self.key(f) for f in other_feats])
        values = torch.stack([self.value(f) for f in other_feats])
        
        # 计算注意力权重
        attn = torch.softmax(torch.matmul(q, keys.T) / (keys.size(-1) ** 0.5), dim=-1)
        
        # 加权聚合
        output = torch.matmul(attn, values)
        
        return output
```

## 实战：合作导航

```python
import numpy as np

class CooperativeNavigation:
    """合作导航环境
    
    多个智能体合作覆盖所有地标
    """
    def __init__(self, n_agents=3, n_landmarks=3):
        self.n_agents = n_agents
        self.n_landmarks = n_landmarks
    
    def reset(self):
        # 随机初始化智能体和地标位置
        self.agent_pos = np.random.uniform(-1, 1, (self.n_agents, 2))
        self.landmark_pos = np.random.uniform(-1, 1, (self.n_landmarks, 2))
        return self._get_obs()
    
    def _get_obs(self):
        """获取每个智能体的局部观察"""
        obs = []
        for i in range(self.n_agents):
            # 自身位置 + 相对地标位置
            o = list(self.agent_pos[i])
            for lm in self.landmark_pos:
                o.extend(lm - self.agent_pos[i])
            obs.append(np.array(o))
        return obs
    
    def step(self, actions):
        """执行动作"""
        # 更新位置
        for i, action in enumerate(actions):
            self.agent_pos[i] += action * 0.1
            self.agent_pos[i] = np.clip(self.agent_pos[i], -1, 1)
        
        # 计算奖励（鼓励覆盖地标）
        reward = 0
        for lm in self.landmark_pos:
            distances = [np.linalg.norm(agent - lm) for agent in self.agent_pos]
            reward -= min(distances)  # 最近智能体的距离
        
        rewards = [reward / self.n_agents] * self.n_agents
        
        return self._get_obs(), rewards, False, {}


def train_cooperative_navigation():
    """训练合作导航"""
    env = CooperativeNavigation(n_agents=3, n_landmarks=3)
    
    # 观察维度：自身(2) + 地标(3*2) = 8
    obs_dims = [8, 8, 8]
    action_dims = [2, 2, 2]  # 2D 移动
    
    maddpg = MADDPG(n_agents=3, obs_dims=obs_dims, action_dims=action_dims)
    
    episodes = 1000
    for episode in range(episodes):
        obs = env.reset()
        episode_reward = 0
        
        for step in range(50):
            actions = maddpg.select_actions(obs, noise_std=0.1)
            next_obs, rewards, done, _ = env.step(actions)
            
            # 存储经验
            # batch = (obs, actions, rewards, next_obs, dones)
            # maddpg.update(batch)
            
            episode_reward += sum(rewards)
            obs = next_obs
        
        if episode % 100 == 0:
            print(f"Episode {episode}: 平均奖励 {episode_reward / 3:.3f}")
```

## 知识点关联

```
独立 Q 学习（基线）
       ↓
   CTDE 框架
       ↓
   ┌───┼───┐
   ↓   ↓   ↓
MADDPG MAPPO QMIX
       ↓
   通信机制
       ↓
  复杂协作任务
```

## 核心考点

### 必须掌握

| 概念 | 说明 |
|------|------|
| CTDE 框架 | 集中训练分散执行 |
| 非平稳性 | 其他智能体策略变化导致环境不稳定 |
| 信用分配 | 如何分配联合奖励 |
| 通信机制 | 智能体间信息交换 |

### 算法对比

| 算法 | 框架 | 适用场景 |
|------|------|----------|
| IQL | 分布式 | 简单任务基线 |
| MADDPG | CTDE | 连续动作合作 |
| MAPPO | CTDE | 通用首选 |
| QMIX | CTDE | 离散动作合作 |

## 学习建议

### 实践要点

1. **环境设计**：从简单合作任务开始
2. **框架选择**：MAPPO 是当前首选
3. **调试技巧**：检查中心化 Critic 是否正确使用全局信息

### 后续方向

- **博弈论**：纳什均衡、斯坦克尔伯格博弈
- **零和博弈**：对抗性多智能体
- **大规模系统**：百/千智能体协调

---

**上一章**：[现代强化学习算法](./modern-algorithms.md)  
**下一章**：[大模型强化学习](./llm-rl.md)
