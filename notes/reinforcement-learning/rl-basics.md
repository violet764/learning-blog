# 强化学习基础入门

本章将系统介绍强化学习（Reinforcement Learning, RL）的基本概念、数学基础和核心思想，为零基础学习者建立完整的认知框架。强化学习作为机器学习的重要分支，通过智能体与环境交互学习最优策略，在游戏AI、机器人控制、推荐系统、大语言模型对齐等领域有广泛应用。

## 基本概念

### 智能体与环境

强化学习的核心是**智能体(Agent)** 与 **环境(Environment)** 的交互过程：

| 要素 | 英文 | 说明 |
|------|------|------|
| 智能体 | Agent | 学习并决策的主体，如游戏玩家、机器人 |
| 环境 | Environment | 智能体所处的外部世界，如游戏场景、物理世界 |
| 状态 | State | 环境的当前情况描述 |
| 动作 | Action | 智能体可以执行的操作 |
| 奖励 | Reward | 环境对智能体动作的即时反馈 |

**交互流程**：

```
智能体观察状态 s_t → 选择动作 a_t → 环境反馈奖励 r_{t+1} 和新状态 s_{t+1} → 循环
```

### 马尔可夫决策过程（MDP）

MDP 是强化学习的数学基础，包含五个要素：

$$
\text{MDP} = (\mathcal{S}, \mathcal{A}, P, R, \gamma)
$$

| 要素 | 符号 | 说明 |
|------|------|------|
| 状态空间 | $\mathcal{S}$ | 所有可能状态的集合 |
| 动作空间 | $\mathcal{A}$ | 所有可能动作的集合 |
| 状态转移概率 | $P(s'|s,a)$ | 在状态 $s$ 执行动作 $a$ 后转移到状态 $s'$ 的概率 |
| 奖励函数 | $R(s,a,s')$ | 状态转移获得的即时奖励 |
| 折扣因子 | $\gamma \in [0,1]$ | 未来奖励的衰减系数 |

**马尔可夫性质**：下一个状态仅依赖于当前状态和动作，与历史无关：

$$
P(S_{t+1}|S_t, A_t, S_{t-1}, A_{t-1}, ...) = P(S_{t+1}|S_t, A_t)
$$

### 策略与价值

#### 策略（Policy）

策略定义了智能体在给定状态下选择动作的方式：

- **确定性策略**：$\pi(s) = a$，给定状态直接输出动作
- **随机策略**：$\pi(a|s) = P(A_t=a|S_t=s)$，输出动作的概率分布

#### 价值函数

价值函数用于评估状态或状态-动作对的长期价值：

**状态价值函数** $V_\pi(s)$：从状态 $s$ 开始遵循策略 $\pi$ 的期望累积奖励

$$
V_\pi(s) = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \bigg| S_t = s\right]
$$

**动作价值函数** $Q_\pi(s,a)$：在状态 $s$ 执行动作 $a$ 后遵循策略 $\pi$ 的期望累积奖励

$$
Q_\pi(s,a) = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \bigg| S_t = s, A_t = a\right]
$$

### 贝尔曼方程

贝尔曼方程是强化学习的理论基础，描述了价值函数的递归关系：

**贝尔曼期望方程**：

$$
V_\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V_\pi(s')]
$$

**贝尔曼最优方程**：

$$
V^*(s) = \max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^*(s')]
$$

$$
Q^*(s,a) = \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma \max_{a'} Q^*(s',a')]
$$

## 探索与利用

### 问题定义

- **探索（Exploration）**：尝试新的动作以获得更多环境信息
- **利用（Exploitation）**：选择当前已知的最佳动作

这是强化学习中的核心权衡问题：探索可能发现更优策略，但短期收益可能较低；利用能获得稳定收益，但可能错过全局最优。

### 常用策略

#### ε-贪心策略

以 $\epsilon$ 概率随机探索，以 $1-\epsilon$ 概率选择最优动作：

$$
a = \begin{cases}
\text{random action} & \text{with probability } \epsilon \\
\arg\max_a Q(s,a) & \text{with probability } 1-\epsilon
\end{cases}
$$

```python
def epsilon_greedy(q_values, epsilon, n_actions):
    """ε-贪心策略"""
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)  # 探索
    else:
        return np.argmax(q_values)  # 利用
```

#### Softmax 策略

根据动作价值按概率分布选择，价值越高的动作被选中的概率越大：

$$
\pi(a|s) = \frac{\exp(Q(s,a)/\tau)}{\sum_{a'} \exp(Q(s,a')/\tau)}
$$

其中 $\tau$ 是温度参数，控制探索程度。

#### UCB（上置信界）

综合考虑期望价值和不确定性：

$$
a_t = \arg\max_a \left[ Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}} \right]
$$

## On-policy vs Off-policy

强化学习算法根据数据来源的不同，可以分为 On-policy 和 Off-policy 两类。理解这一区别对于选择合适的算法至关重要。

### On-policy（同策略）

**定义**：智能体使用**当前策略**收集的数据来更新该策略本身。

$$
\pi_{\text{behavior}} = \pi_{\text{target}}
$$

即行为策略（收集数据的策略）与目标策略（被优化的策略）相同。

**特点**：

| 优点 | 缺点 |
|------|------|
| 方差较低，训练更稳定 | 数据效率低，每次策略更新后旧数据作废 |
| 实现相对简单 | 需要大量交互采样 |
| 理论分析更清晰 | 不适合离线学习场景 |

**代表算法**：
- SARSA
- PPO（Proximal Policy Optimization）
- TRPO（Trust Region Policy Optimization）
- REINFORCE

**代码示例**：

```python
class OnPolicyAgent:
    """On-policy 智能体示例"""
    def __init__(self, policy):
        self.policy = policy
        self.buffer = []  # 存储当前策略收集的数据
    
    def collect_trajectory(self, env):
        """用当前策略收集数据"""
        state = env.reset()
        trajectory = []
        
        while True:
            action = self.policy.select_action(state)  # 当前策略
            next_state, reward, done, _ = env.step(action)
            trajectory.append((state, action, reward, next_state))
            state = next_state
            
            if done:
                break
        
        return trajectory
    
    def update(self):
        """使用收集的数据更新策略"""
        # 更新后，旧数据不再使用（on-policy 特点）
        for transition in self.buffer:
            self.policy.update(transition)
        self.buffer = []  # 清空缓冲区
```

### Off-policy（异策略）

**定义**：智能体可以使用**不同策略**收集的数据来更新目标策略。

$$
\pi_{\text{behavior}} \neq \pi_{\text{target}}
$$

行为策略与目标策略可以不同，这意味着可以使用历史数据、其他智能体的数据或专家演示数据。

**特点**：

| 优点 | 缺点 |
|------|------|
| 数据效率高，可重用历史数据 | 方差较高，训练可能不稳定 |
| 支持离线学习（Offline RL） | 需要重要性采样等技术处理分布偏差 |
| 可以利用专家演示数据 | 实现复杂度较高 |

**代表算法**：
- Q-Learning
- DQN（Deep Q-Network）
- DDPG（Deep Deterministic Policy Gradient）
- SAC（Soft Actor-Critic）

**代码示例**：

```python
class OffPolicyAgent:
    """Off-policy 智能体示例"""
    def __init__(self, policy, behavior_policy=None):
        self.policy = policy  # 目标策略
        self.behavior_policy = behavior_policy or policy  # 行为策略
        self.replay_buffer = []  # 经验回放缓冲区
    
    def collect_data(self, env, n_steps):
        """用行为策略收集数据"""
        for _ in range(n_steps):
            state = env.reset()
            action = self.behavior_policy.select_action(state)  # 行为策略
            next_state, reward, done, _ = env.step(action)
            
            self.replay_buffer.append((state, action, reward, next_state))
    
    def update(self, batch_size):
        """从缓冲区采样并更新目标策略"""
        batch = random.sample(self.replay_buffer, batch_size)
        
        for state, action, reward, next_state in batch:
            # 使用目标策略计算目标值
            target = reward + self.policy.get_value(next_state)
            self.policy.update(state, action, target)
```

### 关键区别总结

| 维度 | On-policy | Off-policy |
|------|-----------|------------|
| 数据来源 | 当前策略收集 | 任意策略收集 |
| 数据复用 | 不能复用 | 可以复用（经验回放） |
| 样本效率 | 低 | 高 |
| 算法稳定性 | 较高 | 需要额外技巧稳定 |
| 适用场景 | 在线学习、模拟环境 | 离线学习、真实环境 |

### 重要性采样

Off-policy 算法需要使用重要性采样来修正分布偏差：

$$
\mathbb{E}_{\pi_b}[f(x)] = \mathbb{E}_{\pi_b}\left[f(x) \cdot \frac{\pi_t(x)}{\pi_b(x)}\right]
$$

其中 $\pi_t$ 是目标策略，$\pi_b$ 是行为策略，重要性权重为 $\frac{\pi_t(a|s)}{\pi_b(a|s)}$。

---

## Online vs Offline Learning

强化学习还可以根据学习过程中是否需要与环境交互，分为 Online Learning 和 Offline Learning。

### Online Learning（在线学习）

**定义**：智能体在**实时与环境交互**过程中学习，边交互边优化策略。

```
交互 → 收集数据 → 更新策略 → 交互 → ...
```

**特点**：
- 📌 **实时更新**：每次交互后立即更新策略
- 📌 **探索必需**：需要主动探索未知状态
- 📌 **环境依赖**：需要能够与环境实时交互
- ⚠️ **安全风险**：在真实环境（如机器人、医疗）中可能造成损害

**适用场景**：
- 游戏AI（AlphaGo、Atari）
- 模拟环境训练
- 推荐系统（A/B测试）

**代码示例**：

```python
def online_training(env, agent, n_episodes):
    """在线学习流程"""
    for episode in range(n_episodes):
        state = env.reset()
        
        while True:
            # 实时与环境交互
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            # 立即更新策略
            agent.update(state, action, reward, next_state)
            
            state = next_state
            if done:
                break
```

### Offline Learning（离线学习）

**定义**：智能体**仅使用预先收集的静态数据集**学习，不再与环境交互。

```
已有数据集 → 训练策略 → 部署
```

**特点**：
- 📌 **数据固定**：只能使用已有的历史数据
- 📌 **无需交互**：不需要实时访问环境
- 📌 **安全可控**：不会在真实环境中造成损害
- ⚠️ **分布偏差**：策略可能访问数据中未见过的状态

**适用场景**：
- 医疗决策（使用历史病历数据）
- 自动驾驶（使用历史驾驶数据）
- 推荐系统（使用历史交互日志）
- 机器人控制（使用演示数据）

**代码示例**：

```python
def offline_training(dataset, agent, n_epochs):
    """离线学习流程"""
    # 数据集是预先收集的，不再与环境交互
    for epoch in range(n_epochs):
        for batch in dataset.get_batches():
            states, actions, rewards, next_states = batch
            
            # 仅使用数据集中的数据更新策略
            agent.update(states, actions, rewards, next_states)
    
    return agent  # 训练完成后部署
```

### 分布偏移问题（Distribution Shift）

Offline Learning 的核心挑战是**分布偏移**：训练数据中的状态分布与策略实际遇到的状态分布可能不同。

```
问题链条：
策略 π 学习后行为改变 → 访问数据中未见的状态 → 模型在这些状态上表现未知 → 可能做出错误决策
```

**解决方案**：

| 方法 | 思路 | 代表工作 |
|------|------|----------|
| 策略约束 | 限制策略不要偏离行为策略太远 | BCQ、BEAR |
| 悲观估计 | 对未见状态给较低的价值估计 | CQL |
| 数据增强 | 扩展数据覆盖范围 | MOPO |

### Online vs Offline 对比

| 维度 | Online Learning | Offline Learning |
|------|-----------------|------------------|
| 数据来源 | 实时交互收集 | 预先收集的静态数据 |
| 环境访问 | 需要 | 不需要 |
| 探索能力 | 可以主动探索 | 无法探索 |
| 安全性 | 有风险 | 安全 |
| 数据效率 | 可以针对性收集 | 受限于已有数据 |
| 典型算法 | PPO、SAC、DQN | CQL、BCQ、IQL |

### 与 LLM 对齐的关系

在大语言模型对齐中，这两种学习方式都有应用：

| 学习方式 | LLM 应用 | 说明 |
|----------|----------|------|
| **Online RLHF** | PPO 训练 | 实时采样、实时更新，效果更好但成本高 |
| **Offline RLHF** | DPO 训练 | 使用固定的偏好数据集，成本低但效果受限于数据质量 |

---

## 与其他机器学习方法的区别

| 特征 | 监督学习 | 无监督学习 | 强化学习 |
|------|----------|------------|----------|
| 数据形式 | 输入-输出对 | 无标签数据 | 状态-动作-奖励序列 |
| 反馈类型 | 明确的标签 | 无明确反馈 | 延迟的奖励信号 |
| 学习目标 | 最小化预测误差 | 发现数据结构 | 最大化累积奖励 |
| 数据依赖 | 独立同分布 | 独立同分布 | 序列相关 |

## 代码示例

### 简单网格世界环境

```python
import numpy as np

class GridWorld:
    """简单的网格世界环境
    
    智能体从起点出发，目标是最小步数到达终点
    """
    def __init__(self, size=4):
        self.size = size
        self.state = 0  # 起始状态（左上角）
        self.goal = size * size - 1  # 目标状态（右下角）
        
    def reset(self):
        """重置环境"""
        self.state = 0
        return self.state
    
    def step(self, action):
        """执行动作
        
        Args:
            action: 0-上，1-右，2-下，3-左
            
        Returns:
            next_state: 新状态
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        row, col = divmod(self.state, self.size)
        
        # 根据动作更新位置
        if action == 0:  # 上
            row = max(0, row - 1)
        elif action == 1:  # 右
            col = min(self.size - 1, col + 1)
        elif action == 2:  # 下
            row = min(self.size - 1, row + 1)
        elif action == 3:  # 左
            col = max(0, col - 1)
            
        self.state = row * self.size + col
        
        # 奖励设置
        if self.state == self.goal:
            return self.state, 10, True, {}
        else:
            return self.state, -1, False, {}
    
    def render(self):
        """可视化环境"""
        for i in range(self.size):
            row = ""
            for j in range(self.size):
                pos = i * self.size + j
                if pos == self.state:
                    row += "A "  # 智能体位置
                elif pos == self.goal:
                    row += "G "  # 目标位置
                else:
                    row += ". "
            print(row)
```

### ε-贪心智能体

```python
class EpsilonGreedyAgent:
    """ε-贪心智能体"""
    def __init__(self, n_actions, epsilon=0.1, alpha=0.1, gamma=0.9):
        self.n_actions = n_actions
        self.epsilon = epsilon  # 探索率
        self.alpha = alpha      # 学习率
        self.gamma = gamma      # 折扣因子
        self.q_values = {}      # 动作价值估计
        
    def get_action(self, state):
        """根据ε-贪心策略选择动作"""
        if state not in self.q_values:
            self.q_values[state] = np.zeros(self.n_actions)
            
        if np.random.random() < self.epsilon:
            return np.random.choice(self.n_actions)  # 探索
        else:
            return np.argmax(self.q_values[state])   # 利用
    
    def update(self, state, action, reward, next_state):
        """Q-learning 更新规则"""
        if state not in self.q_values:
            self.q_values[state] = np.zeros(self.n_actions)
        if next_state not in self.q_values:
            self.q_values[next_state] = np.zeros(self.n_actions)
            
        # TD 更新
        current_q = self.q_values[state][action]
        max_next_q = np.max(self.q_values[next_state])
        td_target = reward + self.gamma * max_next_q
        
        self.q_values[state][action] += self.alpha * (td_target - current_q)

# 训练示例
env = GridWorld(size=4)
agent = EpsilonGreedyAgent(n_actions=4, epsilon=0.2)

for episode in range(100):
    state = env.reset()
    total_reward = 0
    
    for step in range(50):
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.update(state, action, reward, next_state)
        
        total_reward += reward
        state = next_state
        
        if done:
            break
    
    if episode % 20 == 0:
        print(f"Episode {episode}: 总奖励 {total_reward}")
```

## 知识点关联

```
马尔可夫决策过程（数学基础）
         ↓
    智能体-环境交互框架
         ↓
    ┌────┴────┐
    ↓         ↓
  策略      价值函数
    ↓         ↓
    └────┬────┘
         ↓
    探索与利用权衡
         ↓
      算法实现
```

## 核心考点

### 必须掌握

1. **MDP 五元组**：状态空间、动作空间、转移概率、奖励函数、折扣因子
2. **价值函数定义**：$V_\pi(s)$ 和 $Q_\pi(s,a)$ 的含义与关系
3. **贝尔曼方程**：递归关系的理解与应用
4. **探索与利用**：ε-贪心策略的原理

### 常见误解

| 误解 | 正确理解 |
|------|----------|
| 奖励 = 价值 | 奖励是即时反馈，价值是长期期望累积奖励 |
| 策略 = 价值 | 策略是动作选择规则，价值是策略的评估指标 |
| 最优策略唯一 | 可能存在多个最优策略，但最优价值函数唯一 |

## 学习建议

### 学习路径

1. **理解概念本质**：重点理解智能体与环境交互的核心思想
2. **掌握数学基础**：MDP 和贝尔曼方程是后续所有算法的基础
3. **动手实践**：实现简单的网格世界环境，加深理解
4. **联系实际**：思考强化学习在游戏、机器人、推荐等场景的应用

### 后续方向

- **表格型方法**：Q-learning、SARSA 等经典算法
- **函数逼近**：处理大规模状态空间
- **策略梯度**：直接优化策略参数
- **深度强化学习**：结合神经网络

### 推荐资源

- 📚 Sutton & Barto《Reinforcement Learning: An Introduction》第1-3章
- 🌐 [OpenAI Spinning Up](https://spinningup.openai.com/) 基础教程
- 📺 David Silver RL 课程（UCL/DeepMind）

---

**下一章**：[表格型方法详解](./tabular-methods.md) - 掌握 Q-learning、SARSA 等经典算法
