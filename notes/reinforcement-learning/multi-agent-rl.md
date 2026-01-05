# 多智能体强化学习

## 章节概述
本章将介绍多智能体强化学习（MARL），这是强化学习的重要扩展，涉及多个智能体在共享环境中的交互。多智能体系统面临非平稳性、信用分配、通信协调等独特挑战，需要专门的算法和技术来解决。

## 核心知识点

### 1. 多智能体问题基础

#### 1.1 多智能体MDP（MMDP）
扩展MDP到多智能体场景：
- **联合状态空间**：S = S₁ × S₂ × ... × S_N
- **联合动作空间**：A = A₁ × A₂ × ... × A_N
- **联合奖励函数**：R(s, a, s')

#### 1.2 关键挑战
- **非平稳性**：其他智能体的策略在变化
- **信用分配**：如何分配联合奖励给各个智能体
- **可扩展性**：状态动作空间随智能体数量指数增长
- **通信协调**：智能体间如何有效协作

### 2. 多智能体算法分类

#### 2.1 完全集中式
所有智能体共享一个中央控制器：
```python
class CentralizedController:
    def __init__(self, state_dim, action_dims):
        self.n_agents = len(action_dims)
        self.total_actions = sum(action_dims)
        
        # 集中式策略网络
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.total_actions)
        )
    
    def forward(self, joint_state):
        return self.policy_net(joint_state)
```

#### 2.2 完全分布式
每个智能体独立学习：
```python
class IndependentQLearning:
    """独立Q学习"""
    def __init__(self, n_agents, state_dims, action_dims):
        self.agents = []
        for i in range(n_agents):
            agent = DQNAgent(state_dims[i], action_dims[i])
            self.agents.append(agent)
    
    def select_actions(self, states):
        actions = []
        for i, agent in enumerate(self.agents):
            action = agent.select_action(states[i])
            actions.append(action)
        return actions
```

#### 2.3 集中训练分散执行（CTDE）
训练时集中信息，执行时分散决策：

### 3. 集中训练分散执行框架

#### 3.1 MADDPG算法
多智能体深度确定性策略梯度：
```python
class MADDPG:
    """MADDPG算法"""
    def __init__(self, n_agents, state_dims, action_dims, 
                 actor_hidden=64, critic_hidden=64, lr=0.01, gamma=0.95):
        
        self.n_agents = n_agents
        self.gamma = gamma
        
        # 每个智能体的Actor和Critic
        self.actors = []
        self.critics = []
        self.actor_optimizers = []
        self.critic_optimizers = []
        
        for i in range(n_agents):
            # Actor（分散执行）
            actor = ActorNetwork(state_dims[i], action_dims[i], actor_hidden)
            self.actors.append(actor)
            self.actor_optimizers.append(optim.Adam(actor.parameters(), lr=lr))
            
            # Critic（集中训练，输入所有智能体的状态和动作）
            critic_input_dim = sum(state_dims) + sum(action_dims)
            critic = CriticNetwork(critic_input_dim, 1, critic_hidden)
            self.critics.append(critic)
            self.critic_optimizers.append(optim.Adam(critic.parameters(), lr=lr))
    
    def select_actions(self, states):
        """分散执行：每个智能体独立决策"""
        actions = []
        for i, actor in enumerate(self.actors):
            state_tensor = torch.FloatTensor(states[i]).unsqueeze(0)
            action = actor(state_tensor).detach().numpy()[0]
            actions.append(action)
        return actions
    
    def update(self, batch):
        """集中训练：使用所有智能体的信息"""
        states, actions, rewards, next_states, dones = batch
        
        for i in range(self.n_agents):
            # 准备Critic输入（所有智能体的状态和动作）
            critic_input = []
            next_critic_input = []
            
            for j in range(self.n_agents):
                critic_input.extend(states[j])
                critic_input.extend(actions[j])
                
                # 目标动作（使用目标策略）
                next_state_tensor = torch.FloatTensor(next_states[j]).unsqueeze(0)
                next_action = self.actors[j](next_state_tensor).detach().numpy()[0]
                next_critic_input.extend(next_states[j])
                next_critic_input.extend(next_action)
            
            # Critic更新
            current_q = self.critics[i](torch.FloatTensor(critic_input).unsqueeze(0))
            target_q = rewards[i] + self.gamma * self.critics[i](torch.FloatTensor(next_critic_input).unsqueeze(0))
            critic_loss = nn.MSELoss()(current_q, target_q.detach())
            
            self.critic_optimizers[i].zero_grad()
            critic_loss.backward()
            self.critic_optimizers[i].step()
            
            # Actor更新
            actor_loss = -self.critics[i](torch.FloatTensor(critic_input).unsqueeze(0)).mean()
            self.actor_optimizers[i].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[i].step()
```

#### 3.2 MAPPO算法
多智能体近端策略优化：
```python
class MAPPO:
    """多智能体PPO"""
    def __init__(self, n_agents, state_dims, action_dims, 
                 hidden_dim=64, lr=3e-4, gamma=0.99, clip_epsilon=0.2):
        
        self.n_agents = n_agents
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        
        # 每个智能体的Actor-Critic网络
        self.actors = []
        self.critics = []
        self.optimizers = []
        
        for i in range(n_agents):
            # 中心化Critic，分散化Actor
            actor = PolicyNetwork(state_dims[i], action_dims[i], hidden_dim)
            critic = ValueNetwork(sum(state_dims), hidden_dim)  # 输入所有状态
            
            self.actors.append(actor)
            self.critics.append(critic)
            
            # 组合优化器
            params = list(actor.parameters()) + list(critic.parameters())
            self.optimizers.append(optim.Adam(params, lr=lr))
    
    def update(self, batch_data):
        """MAPPO更新"""
        for i in range(self.n_agents):
            states, actions, old_log_probs, rewards, next_states, dones = batch_data[i]
            
            # 计算优势函数（使用中心化Critic）
            joint_states = np.concatenate(states, axis=0)  # 所有智能体状态
            advantages = self.compute_advantages(joint_states, rewards, dones, self.critics[i])
            
            # PPO更新（与单智能体PPO类似）
            self.ppo_update(i, states, actions, old_log_probs, advantages)
```

### 4. 通信与协调

#### 4.1 通信网络
智能体间传递消息的机制：
```python
class CommNetwork(nn.Module):
    """通信网络"""
    def __init__(self, input_dim, comm_dim, hidden_dim=64):
        super(CommNetwork, self).__init__()
        
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.comm_layer = nn.Linear(hidden_dim, comm_dim)
        self.decoder = nn.Linear(hidden_dim + comm_dim * 2, hidden_dim)  # 自身+邻居信息
        
    def forward(self, self_state, neighbor_states):
        # 编码自身状态
        self_encoded = torch.relu(self.encoder(self_state))
        
        # 生成通信消息
        message = self.comm_layer(self_encoded)
        
        # 聚合邻居信息
        neighbor_messages = []
        for neighbor in neighbor_states:
            neighbor_encoded = torch.relu(self.encoder(neighbor))
            neighbor_msg = self.comm_layer(neighbor_encoded)
            neighbor_messages.append(neighbor_msg)
        
        if neighbor_messages:
            aggregated_msg = torch.mean(torch.stack(neighbor_messages), dim=0)
        else:
            aggregated_msg = torch.zeros_like(message)
        
        # 解码（自身信息+邻居信息）
        combined = torch.cat([self_encoded, message, aggregated_msg], dim=-1)
        output = torch.relu(self.decoder(combined))
        
        return output, message
```

#### 4.2 基于注意力的通信
```python
class AttentionComm:
    """基于注意力的通信"""
    def __init__(self, input_dim, key_dim=64, value_dim=64):
        self.key_net = nn.Linear(input_dim, key_dim)
        self.query_net = nn.Linear(input_dim, key_dim)
        self.value_net = nn.Linear(input_dim, value_dim)
    
    def forward(self, self_state, other_states):
        # 生成query, key, value
        query = self.query_net(self_state)
        keys = [self.key_net(state) for state in other_states]
        values = [self.value_net(state) for state in other_states]
        
        if keys:
            keys = torch.stack(keys)
            values = torch.stack(values)
            
            # 计算注意力权重
            attention_weights = torch.softmax(torch.matmul(query, keys.transpose(0,1)) / np.sqrt(keys.size(-1)), dim=-1)
            
            # 加权聚合
            aggregated = torch.matmul(attention_weights, values)
        else:
            aggregated = torch.zeros_like(query)
        
        return aggregated
```

### 5. 实战应用：多智能体粒子环境

#### 5.1 合作导航任务
多个智能体合作到达目标位置：
```python
class CooperativeNavigation:
    """合作导航环境"""
    def __init__(self, n_agents=3, n_landmarks=3):
        self.n_agents = n_agents
        self.n_landmarks = n_landmarks
        
        # 初始化位置
        self.agent_positions = np.random.uniform(-1, 1, (n_agents, 2))
        self.landmark_positions = np.random.uniform(-1, 1, (n_landmarks, 2))
        
    def reset(self):
        self.agent_positions = np.random.uniform(-1, 1, (self.n_agents, 2))
        return self.get_states()
    
    def get_states(self):
        """获取每个智能体的局部观察"""
        states = []
        for i in range(self.n_agents):
            # 自身位置 + 相对地标位置 + 相对其他智能体位置
            state = []
            state.extend(self.agent_positions[i])  # 自身位置
            
            # 相对地标位置
            for landmark in self.landmark_positions:
                relative_pos = landmark - self.agent_positions[i]
                state.extend(relative_pos)
            
            # 相对其他智能体位置
            for j in range(self.n_agents):
                if i != j:
                    relative_pos = self.agent_positions[j] - self.agent_positions[i]
                    state.extend(relative_pos)
                else:
                    state.extend([0, 0])  # 自身位置为0
            
            states.append(np.array(state))
        return states
    
    def step(self, actions):
        """执行动作"""
        rewards = []
        
        # 更新位置（简单的速度控制）
        for i, action in enumerate(actions):
            self.agent_positions[i] += action * 0.1
            self.agent_positions[i] = np.clip(self.agent_positions[i], -1, 1)
        
        # 计算奖励（鼓励覆盖所有地标）
        total_reward = 0
        for landmark in self.landmark_positions:
            # 找到最近智能体的距离
            distances = [np.linalg.norm(agent - landmark) for agent in self.agent_positions]
            min_distance = min(distances)
            total_reward -= min_distance  # 距离越小奖励越大
        
        # 平均分配奖励
        individual_reward = total_reward / self.n_agents
        rewards = [individual_reward] * self.n_agents
        
        return self.get_states(), rewards, False, {}  # 简单环境，不设终止条件

# 训练示例
def train_marl_navigation():
    """训练多智能体合作导航"""
    env = CooperativeNavigation(n_agents=3)
    
    # 状态维度：自身(2) + 地标(3*2) + 其他智能体(2*2) = 12
    state_dims = [12] * 3
    action_dims = [2] * 3  # 2D移动
    
    # 使用MADDPG
    maddpg = MADDPG(n_agents=3, state_dims=state_dims, action_dims=action_dims)
    
    episodes = 1000
    for episode in range(episodes):
        states = env.reset()
        total_rewards = [0] * 3
        
        for step in range(100):  # 最大步数
            # 选择动作
            actions = maddpg.select_actions(states)
            
            # 执行动作
            next_states, rewards, done, _ = env.step(actions)
            
            # 存储经验并更新
            # 这里简化了经验回放，实际需要更复杂的实现
            
            total_rewards = [r1 + r2 for r1, r2 in zip(total_rewards, rewards)]
            states = next_states
        
        if episode % 100 == 0:
            avg_reward = sum(total_rewards) / len(total_rewards)
            print(f"Episode {episode}, Average Reward: {avg_reward:.3f}")
```

## 知识点间关联逻辑

### 算法演进关系
```
独立Q学习（完全分布式）
    ↓
中心化Critic（CTDE框架）
    ↓
MADDPG（连续动作）
    ↓
MAPPO（策略优化）
    ↓
通信网络（智能协调）
```

## 章节核心考点汇总

### 必须掌握的概念
1. **CTDE框架**：集中训练分散执行的核心思想
2. **非平稳性挑战**：多智能体环境的独特问题
3. **信用分配**：联合奖励的分配机制

### 重要算法
- **MADDPG**：多智能体DDPG
- **MAPPO**：多智能体PPO
- **独立Q学习**：基线方法

## 学习建议/后续延伸方向

### 学习建议
1. **理解CTDE框架**：掌握多智能体算法的核心思想
2. **实现基础算法**：从独立Q学习开始
3. **研究通信机制**：理解智能体间协调的关键

### 后续学习方向
1. **层次强化学习**：多时间尺度决策
2. **博弈论基础**：纳什均衡等概念
3. **大规模多智能体**：可扩展性技术

---

**下一章预告**：接下来将进入重点章节——大模型强化学习，深入探讨RLHF等在大语言模型中的应用技术。