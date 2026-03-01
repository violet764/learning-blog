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
