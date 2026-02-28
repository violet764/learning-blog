# 表格型方法详解

本章深入讲解表格型强化学习算法，这些方法通过维护价值表格来学习最优策略，适用于状态空间较小的问题。理解这些经典算法是掌握现代深度强化学习的基础。

## 动态规划方法

动态规划（Dynamic Programming, DP）是强化学习的理论基础，要求已知完整的环境模型（状态转移概率和奖励函数）。

### 策略评估

策略评估（Policy Evaluation）计算给定策略 $\pi$ 的状态价值函数 $V_\pi(s)$。

**迭代更新公式**：

$$
V_{k+1}(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V_k(s')]
$$

```python
def policy_evaluation(env, policy, gamma=0.9, theta=1e-6):
    """迭代策略评估
    
    Args:
        env: 环境（需要提供 P 状态转移概率）
        policy: 策略矩阵 [n_states, n_actions]
        gamma: 折扣因子
        theta: 收敛阈值
    """
    V = np.zeros(env.nS)  # 状态价值函数
    
    while True:
        delta = 0
        for s in range(env.nS):
            v = V[s]
            # 贝尔曼期望方程
            V[s] = sum([
                policy[s][a] * sum([
                    p * (r + gamma * V[s_]) 
                    for p, s_, r, _ in env.P[s][a]
                ]) 
                for a in range(env.nA)
            ])
            delta = max(delta, abs(v - V[s]))
        
        if delta < theta:
            break
    
    return V
```

### 策略改进

策略改进（Policy Improvement）基于当前价值函数改进策略：

$$
\pi'(s) = \arg\max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V_\pi(s')]
$$

**策略改进定理**：新策略 $\pi'$ 满足 $V_{\pi'}(s) \geq V_\pi(s)$ 对所有状态成立。

### 策略迭代

策略迭代（Policy Iteration）交替执行策略评估和策略改进：

```python
def policy_iteration(env, gamma=0.9):
    """策略迭代算法"""
    policy = np.ones([env.nS, env.nA]) / env.nA  # 随机初始策略
    V = np.zeros(env.nS)
    
    while True:
        # 1. 策略评估
        V = policy_evaluation(env, policy, gamma)
        
        # 2. 策略改进
        policy_stable = True
        for s in range(env.nS):
            old_action = np.argmax(policy[s])
            
            # 计算每个动作的价值
            action_values = np.array([
                sum([p * (r + gamma * V[s_]) for p, s_, r, _ in env.P[s][a]])
                for a in range(env.nA)
            ])
            
            best_action = np.argmax(action_values)
            
            if old_action != best_action:
                policy_stable = False
            
            # 更新为确定性策略
            policy[s] = np.eye(env.nA)[best_action]
        
        if policy_stable:
            break
    
    return policy, V
```

### 价值迭代

价值迭代（Value Iteration）直接优化价值函数，无需显式的策略评估：

$$
V_{k+1}(s) = \max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V_k(s')]
$$

```python
def value_iteration(env, gamma=0.9, theta=1e-6):
    """价值迭代算法"""
    V = np.zeros(env.nS)
    
    while True:
        delta = 0
        for s in range(env.nS):
            v = V[s]
            # 贝尔曼最优方程
            V[s] = max([
                sum([p * (r + gamma * V[s_]) for p, s_, r, _ in env.P[s][a]])
                for a in range(env.nA)
            ])
            delta = max(delta, abs(v - V[s]))
        
        if delta < theta:
            break
    
    # 从最优价值函数提取策略
    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        action_values = np.array([
            sum([p * (r + gamma * V[s_]) for p, s_, r, _ in env.P[s][a]])
            for a in range(env.nA)
        ])
        policy[s] = np.eye(env.nA)[np.argmax(action_values)]
    
    return policy, V
```

## 蒙特卡洛方法

蒙特卡洛（Monte Carlo, MC）方法基于完整回合的经验学习，无需环境模型。

### 首次访问 MC 预测

```python
from collections import defaultdict

def mc_prediction(env, policy, episodes=1000, gamma=0.9):
    """首次访问蒙特卡洛预测
    
    通过采样完整回合来估计状态价值
    """
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    V = defaultdict(float)
    
    for _ in range(episodes):
        # 生成一个完整回合
        episode = []
        state = env.reset()
        
        while True:
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            
            if done:
                break
            state = next_state
        
        # 反向计算回报并更新价值
        G = 0
        visited = set()
        
        for t in reversed(range(len(episode))):
            state, _, reward = episode[t]
            G = gamma * G + reward
            
            # 首次访问：只统计第一次出现的状态
            if state not in visited:
                visited.add(state)
                returns_sum[state] += G
                returns_count[state] += 1
                V[state] = returns_sum[state] / returns_count[state]
    
    return V
```

### MC 控制

结合策略改进的蒙特卡洛方法：

```python
def mc_control(env, episodes=10000, epsilon=0.1, gamma=0.9):
    """ε-软策略蒙特卡洛控制"""
    nA = env.action_space.n
    Q = defaultdict(lambda: np.zeros(nA))
    returns = defaultdict(list)
    
    def policy(state):
        """ε-贪心策略"""
        if np.random.random() < epsilon:
            return np.random.randint(nA)
        return np.argmax(Q[state])
    
    for _ in range(episodes):
        # 生成回合
        episode = []
        state = env.reset()
        
        while True:
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
        
        # 更新 Q 值
        G = 0
        visited = set()
        
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward
            
            if (state, action) not in visited:
                visited.add((state, action))
                returns[(state, action)].append(G)
                Q[state][action] = np.mean(returns[(state, action)])
    
    return Q, policy
```

## 时序差分学习

时序差分（Temporal Difference, TD）结合了动态规划和蒙特卡洛的优点，可以在线学习且无需环境模型。

### TD(0) 预测

$$
V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]
$$

**TD 误差**：$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$

### SARSA 算法

SARSA 是**在策略（On-Policy）**的 TD 控制算法：

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]
$$

```python
def sarsa(env, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
    """SARSA 算法
    
    特点：使用实际采取的下一个动作更新 Q 值
    """
    nA = env.action_space.n
    Q = defaultdict(lambda: np.zeros(nA))
    
    for episode in range(episodes):
        state = env.reset()
        
        # 选择初始动作
        if np.random.random() < epsilon:
            action = np.random.randint(nA)
        else:
            action = np.argmax(Q[state])
        
        done = False
        while not done:
            next_state, reward, done, _ = env.step(action)
            
            # 选择下一个动作（SARSA 的关键）
            if np.random.random() < epsilon:
                next_action = np.random.randint(nA)
            else:
                next_action = np.argmax(Q[next_state])
            
            # SARSA 更新
            td_target = reward + gamma * Q[next_state][next_action] * (1 - done)
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error
            
            state, action = next_state, next_action
    
    return Q
```

### Q-learning 算法

Q-learning 是**离策略（Off-Policy）**的 TD 控制算法，是强化学习中最著名的算法之一：

$$
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha [R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)]
$$

```python
def q_learning(env, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
    """Q-learning 算法
    
    特点：使用最大 Q 值更新，与行为策略无关
    """
    nA = env.action_space.n
    Q = defaultdict(lambda: np.zeros(nA))
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        
        while not done:
            # ε-贪心选择动作
            if np.random.random() < epsilon:
                action = np.random.randint(nA)
            else:
                action = np.argmax(Q[state])
            
            next_state, reward, done, _ = env.step(action)
            
            # Q-learning 更新（离策略）
            td_target = reward + gamma * np.max(Q[next_state]) * (1 - done)
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error
            
            state = next_state
    
    return Q
```

## 算法对比

| 方法 | 是否需要模型 | 引导性 | 方差 | 偏差 | 适用场景 |
|------|-------------|--------|------|------|----------|
| 动态规划 | 需要 | 是 | 低 | 低 | 已知模型、小状态空间 |
| 蒙特卡洛 | 不需要 | 否 | 高 | 无 | 无模型、可模拟完整回合 |
| TD 学习 | 不需要 | 是 | 中 | 低 | 在线学习、实时更新 |

### SARSA vs Q-learning

| 特性 | SARSA | Q-learning |
|------|-------|------------|
| 策略类型 | 在策略 | 离策略 |
| 更新方式 | 使用实际下一动作 | 使用最大 Q 值 |
| 风险偏好 | 较保守 | 较激进 |
| 适用场景 | 需要安全性 | 追求最优性能 |

## 代码实战：FrozenLake

```python
import gym
import numpy as np
from collections import defaultdict

def train_frozen_lake():
    """在 FrozenLake 环境中训练 Q-learning"""
    env = gym.make('FrozenLake-v1', is_slippery=True)
    
    # 超参数
    alpha = 0.1
    gamma = 0.99
    epsilon = 0.1
    episodes = 10000
    
    # Q 表
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # 训练
    rewards = []
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # ε-贪心策略
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])
            
            next_state, reward, done, _ = env.step(action)
            
            # Q-learning 更新
            td_target = reward + gamma * np.max(Q[next_state]) * (1 - done)
            Q[state][action] += alpha * (td_target - Q[state][action])
            
            total_reward += reward
            state = next_state
        
        rewards.append(total_reward)
        
        if episode % 1000 == 0:
            avg = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
            print(f"Episode {episode}: 平均奖励 {avg:.3f}")
    
    return Q

# 运行训练
Q_table = train_frozen_lake()

# 测试训练结果
def test_policy(env, Q, episodes=100):
    success = 0
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = np.argmax(Q[state])
            state, reward, done, _ = env.step(action)
            if reward > 0:
                success += 1
                break
    return success / episodes

env = gym.make('FrozenLake-v1', is_slippery=True)
success_rate = test_policy(env, Q_table)
print(f"成功率: {success_rate:.1%}")
```

## 知识点关联

```
动态规划（需要模型）
      ↓
      │ 无模型需求
      ↓
蒙特卡洛（完整回合）
      ↓
      │ 单步更新
      ↓
时序差分（在线学习）
      ↓
   ┌──┴──┐
   ↓     ↓
SARSA  Q-learning
(在策略) (离策略)
```

## 核心考点

### 必须掌握

1. **贝尔曼方程**：理解递归关系
2. **策略迭代 vs 价值迭代**：区别与联系
3. **SARSA vs Q-learning**：在策略与离策略的区别
4. **探索策略**：ε-贪心的作用

### 重要公式

| 公式名称 | 表达式 |
|----------|--------|
| 贝尔曼期望方程 | $V_\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'\|s,a)[R + \gamma V_\pi(s')]$ |
| 贝尔曼最优方程 | $V^*(s) = \max_a \sum_{s'} P(s'\|s,a)[R + \gamma V^*(s')]$ |
| Q-learning 更新 | $Q \leftarrow Q + \alpha[r + \gamma \max_{a'} Q(s',a') - Q]$ |
| SARSA 更新 | $Q \leftarrow Q + \alpha[r + \gamma Q(s',a') - Q]$ |

## 学习建议

### 学习路径

1. **理解贝尔曼方程**：所有表格方法的理论基础
2. **实现策略迭代**：理解"评估-改进"循环
3. **对比 TD 和 MC**：理解自举（bootstrapping）的作用
4. **调试 Q-learning**：观察收敛过程和策略变化

### 常见问题

| 问题 | 解决方案 |
|------|----------|
| Q 值不收敛 | 降低学习率，检查探索策略 |
| 过度乐观估计 | 使用 Double Q-learning |
| 样本效率低 | 使用经验回放 |

### 后续方向

- **函数逼近**：处理大规模状态空间
- **深度 Q 网络（DQN）**：神经网络 + Q-learning
- **多步 TD**：TD(λ) 方法

---

**上一章**：[强化学习基础入门](./rl-basics.md)  
**下一章**：[函数逼近与深度强化学习](./function-approximation.md)
