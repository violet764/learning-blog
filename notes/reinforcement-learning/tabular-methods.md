# 表格型方法详解

## 章节概述
本章将深入讲解表格型强化学习算法，这些方法适用于状态空间较小的问题，通过维护价值表格来学习最优策略。表格型方法是强化学习的基础，理解这些经典算法对于掌握后续更复杂的方法至关重要。

## 核心知识点

### 1. 动态规划方法

#### 1.1 策略评估（Policy Evaluation）
策略评估的目标是计算给定策略π的状态价值函数V_π(s)。

**迭代策略评估算法**：
$$
V_{k+1}(s) = \sum_a π(a|s) \sum_{s'} P(s'|s,a)[R(s,a,s') + γV_k(s')]
$$

**算法步骤**：
1. 初始化V(s)为任意值（通常为0）
2. 重复直到收敛：
   - 对每个状态s，更新V(s)
   - 使用当前策略π和状态转移概率

#### 1.2 策略改进（Policy Improvement）
基于当前价值函数改进策略：

$$
π'(s) = \arg\max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + γV_π(s')]
$$

**策略改进定理**：如果新策略π'满足上述条件，则V_{π'}(s) ≥ V_π(s)对于所有状态s成立。

#### 1.3 策略迭代（Policy Iteration）
交替进行策略评估和策略改进：

```python
def policy_iteration(env, gamma=0.9, theta=1e-6):
    """策略迭代算法"""
    # 初始化策略和价值函数
    policy = np.ones([env.nS, env.nA]) / env.nA
    V = np.zeros(env.nS)
    
    while True:
        # 策略评估
        while True:
            delta = 0
            for s in range(env.nS):
                v = V[s]
                # 贝尔曼期望方程
                V[s] = sum([policy[s][a] * 
                           sum([p * (r + gamma * V[s_]) 
                               for p, s_, r, _ in env.P[s][a]]) 
                           for a in range(env.nA)])
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break
        
        # 策略改进
        policy_stable = True
        for s in range(env.nS):
            old_action = np.argmax(policy[s])
            # 计算每个动作的价值
            action_values = np.zeros(env.nA)
            for a in range(env.nA):
                for p, s_, r, _ in env.P[s][a]:
                    action_values[a] += p * (r + gamma * V[s_])
            
            # 选择最优动作
            best_action = np.argmax(action_values)
            if old_action != best_action:
                policy_stable = False
            
            # 更新策略（贪心策略）
            policy[s] = np.eye(env.nA)[best_action]
        
        if policy_stable:
            break
    
    return policy, V
```

#### 1.4 价值迭代（Value Iteration）
直接优化价值函数，不需要显式的策略评估：

$$
V_{k+1}(s) = \max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + γV_k(s')]
$$

**算法特点**：
- 更高效，通常收敛更快
- 不需要等待策略评估完全收敛
- 直接使用贝尔曼最优方程

### 2. 蒙特卡洛方法

#### 2.1 首次访问MC预测
基于完整回合的经验估计价值函数：

```python
def mc_prediction_first_visit(env, policy, episodes=1000, gamma=0.9):
    """首次访问蒙特卡洛预测"""
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    V = defaultdict(float)
    
    for episode in range(episodes):
        # 生成一个回合
        episode_history = []
        state = env.reset()
        
        for t in range(100):  # 最大步数
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            episode_history.append((state, action, reward))
            
            if done:
                break
            state = next_state
        
        # 计算回报并更新价值估计
        G = 0
        visited_states = set()
        
        for t in range(len(episode_history)-1, -1, -1):
            state, action, reward = episode_history[t]
            G = gamma * G + reward
            
            # 首次访问：只统计第一次出现的状态
            if state not in visited_states:
                visited_states.add(state)
                returns_sum[state] += G
                returns_count[state] += 1.0
                V[state] = returns_sum[state] / returns_count[state]
    
    return V
```

#### 2.2 MC控制
结合策略改进的蒙特卡洛方法：

**ε-软策略MC控制**：
```python
def mc_control_epsilon_soft(env, episodes=10000, epsilon=0.1, gamma=0.9):
    """ε-软策略蒙特卡洛控制"""
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    returns = defaultdict(list)
    
    # ε-软策略
    def create_epsilon_policy(Q, epsilon, nA):
        def policy_fn(state):
            policy = np.ones(nA) * epsilon / nA
            best_action = np.argmax(Q[state])
            policy[best_action] += (1.0 - epsilon)
            return policy
        return policy_fn
    
    policy = create_epsilon_policy(Q, epsilon, env.action_space.n)
    
    for episode in range(episodes):
        # 生成回合
        episode_history = []
        state = env.reset()
        
        for t in range(100):
            probs = policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)
            episode_history.append((state, action, reward))
            
            if done:
                break
            state = next_state
        
        # 更新Q值
        G = 0
        visited = set()
        
        for t in range(len(episode_history)-1, -1, -1):
            state, action, reward = episode_history[t]
            G = gamma * G + reward
            
            if (state, action) not in visited:
                visited.add((state, action))
                returns[(state, action)].append(G)
                Q[state][action] = np.mean(returns[(state, action)])
    
    return Q, policy
```

### 3. 时序差分学习

#### 3.1 TD(0)预测
结合了蒙特卡洛和动态规划的思想：

$$
V(S_t) ← V(S_t) + α[R_{t+1} + γV(S_{t+1}) - V(S_t)]
$$

**TD误差**：δ_t = R_{t+1} + γV(S_{t+1}) - V(S_t)

#### 3.2 SARSA算法
在策略的TD控制算法：

$$
Q(S_t, A_t) ← Q(S_t, A_t) + α[R_{t+1} + γQ(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]
$$

```python
def sarsa(env, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
    """SARSA算法"""
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    for episode in range(episodes):
        state = env.reset()
        
        # ε-贪心选择动作
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
        
        done = False
        while not done:
            next_state, reward, done, _ = env.step(action)
            
            # 选择下一个动作
            if np.random.random() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(Q[next_state])
            
            # SARSA更新
            td_target = reward + gamma * Q[next_state][next_action]
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error
            
            state, action = next_state, next_action
    
    return Q
```

#### 3.3 Q-learning算法
离策略的TD控制算法，是强化学习中最著名的算法之一：

$$
Q(S_t, A_t) ← Q(S_t, A_t) + α[R_{t+1} + γ\max_a Q(S_{t+1}, a) - Q(S_t, A_t)]
$$

```python
def q_learning(env, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1):
    """Q-learning算法"""
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        
        while not done:
            # ε-贪心选择动作
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])
            
            next_state, reward, done, _ = env.step(action)
            
            # Q-learning更新（离策略）
            td_target = reward + gamma * np.max(Q[next_state])
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error
            
            state = next_state
    
    return Q
```

### 4. 算法比较与选择

#### 4.1 方法对比
| 方法 | 是否需要模型 | 引导性 | 方差 | 偏差 | 收敛性 |
|------|-------------|--------|------|------|--------|
| 动态规划 | 需要 | 是 | 低 | 低 | 保证 |
| 蒙特卡洛 | 不需要 | 否 | 高 | 无 | 保证 |
| TD学习 | 不需要 | 是 | 中 | 低 | 保证 |

#### 4.2 适用场景
- **动态规划**：已知完整环境模型，状态空间较小
- **蒙特卡洛**：需要完整回合，方差较大但无偏
- **时序差分**：在线学习，平衡偏差和方差

## 知识点间关联逻辑

### 算法演进关系
```
动态规划（理论基础，需要模型）
    ↓
蒙特卡洛（无模型，基于回合）
    ↓
时序差分（结合两者优点）
    ↓
Q-learning（最实用的表格方法）
```

### 数学基础衔接
- **贝尔曼方程**：所有方法的基础
- **期望计算**：蒙特卡洛方法的理论基础
- **随机逼近**：TD学习的收敛性保证

## 代码实战：FrozenLake环境应用

### FrozenLake环境介绍
FrozenLake是OpenAI Gym中的经典网格世界环境：
- **状态**：4x4或8x8网格
- **动作**：上下左右移动
- **奖励**：到达目标+1，掉入洞中0，其他-0.01
- **挑战**：冰面滑动，动作结果不确定

### Q-learning在FrozenLake中的应用
```python
import gym
import numpy as np
from collections import defaultdict

def train_q_learning_frozenlake():
    """在FrozenLake环境中训练Q-learning"""
    env = gym.make('FrozenLake-v1', is_slippery=True)
    
    # 超参数
    alpha = 0.1      # 学习率
    gamma = 0.99     # 折扣因子
    epsilon = 0.1    # 探索率
    episodes = 10000 # 训练回合数
    
    # 初始化Q表
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    
    # 训练统计
    success_rate = []
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            # ε-贪心策略
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])
            
            next_state, reward, done, info = env.step(action)
            
            # Q-learning更新
            td_target = reward + gamma * np.max(Q[next_state])
            td_error = td_target - Q[state][action]
            Q[state][action] += alpha * td_error
            
            total_reward += reward
            state = next_state
        
        # 每100回合评估一次
        if episode % 100 == 0:
            success_count = 0
            for _ in range(100):
                state = env.reset()
                done = False
                while not done:
                    action = np.argmax(Q[state])  # 贪心策略
                    state, reward, done, _ = env.step(action)
                    if reward > 0:
                        success_count += 1
                        break
            success_rate.append(success_count / 100)
            print(f"Episode {episode}: 成功率 {success_count}%")
    
    env.close()
    return Q, success_rate

# 运行训练
Q_table, success_history = train_q_learning_frozenlake()

# 测试训练结果
def test_policy(env, Q, test_episodes=100):
    """测试训练好的策略"""
    success_count = 0
    total_steps = 0
    
    for episode in range(test_episodes):
        state = env.reset()
        done = False
        steps = 0
        
        while not done and steps < 100:
            action = np.argmax(Q[state])
            state, reward, done, _ = env.step(action)
            steps += 1
            
            if reward > 0:
                success_count += 1
                break
        
        total_steps += steps
    
    success_rate = success_count / test_episodes * 100
    avg_steps = total_steps / test_episodes
    
    print(f"测试结果: 成功率 {success_rate:.1f}%, 平均步数 {avg_steps:.1f}")
    return success_rate, avg_steps

env = gym.make('FrozenLake-v1', is_slippery=True)
test_policy(env, Q_table)
env.close()
```

### 不同算法性能比较
```python
def compare_algorithms():
    """比较不同表格型算法的性能"""
    env = gym.make('FrozenLake-v1', is_slippery=True)
    
    algorithms = {
        'Q-learning': q_learning,
        'SARSA': sarsa
    }
    
    results = {}
    
    for name, algorithm in algorithms.items():
        print(f"训练{name}...")
        
        if name == 'Q-learning':
            Q = algorithm(env, episodes=5000)
        else:
            Q = algorithm(env, episodes=5000)
        
        success_rate, avg_steps = test_policy(env, Q, 100)
        results[name] = {'success_rate': success_rate, 'avg_steps': avg_steps}
    
    env.close()
    
    # 显示比较结果
    print("\n算法性能比较:")
    for name, result in results.items():
        print(f"{name}: 成功率 {result['success_rate']:.1f}%, 平均步数 {result['avg_steps']:.1f}")
    
    return results

# 运行比较
# results = compare_algorithms()
```

## 章节核心考点汇总

### 必须掌握的算法
1. **策略迭代**：评估-改进循环
2. **价值迭代**：直接优化价值函数
3. **蒙特卡洛方法**：基于回合的经验学习
4. **SARSA**：在策略TD控制
5. **Q-learning**：离策略TD控制

### 重要公式
1. **贝尔曼期望方程**：V_π(s) = E_π[R_{t+1} + γV_π(S_{t+1}) | S_t = s]
2. **贝尔曼最优方程**：V*(s) = max_a E[R_{t+1} + γV*(S_{t+1}) | S_t = s, A_t = a]
3. **Q-learning更新**：Q(s,a) ← Q(s,a) + α[r + γmax_a' Q(s',a') - Q(s,a)]

### 算法选择原则
- **小状态空间**：动态规划或表格方法
- **需要在线学习**：时序差分方法
- **安全性要求高**：SARSA（更保守）
- **最大化性能**：Q-learning（更激进）

## 学习建议/后续延伸方向

### 学习建议
1. **理解算法思想**：重点掌握每种方法的核心思想
2. **动手实现**：通过代码实现加深理解
3. **参数调优**：学习率、折扣因子等超参数的影响
4. **性能分析**：比较不同算法在相同环境下的表现

### 常见错误避免
1. **混淆在策略和离策略**：SARSA vs Q-learning
2. **忽视探索策略**：纯贪心策略可能不收敛
3. **学习率设置不当**：过大震荡，过小收敛慢

### 后续学习方向
1. **函数逼近**：处理大规模状态空间
2. **深度Q网络**：结合深度学习的Q-learning
3. **策略梯度**：直接优化策略参数
4. **多步TD学习**：TD(λ)等扩展方法

### 扩展阅读
- Sutton & Barto《强化学习导论》第4-6章
- OpenAI Gym官方文档
- 经典论文：Watkins的Q-learning原始论文

---

**下一章预告**：在掌握了表格型方法后，下一章将介绍函数逼近技术，解决状态空间过大时的维数灾难问题，为深度强化学习打下基础。