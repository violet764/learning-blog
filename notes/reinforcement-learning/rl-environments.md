# 强化学习实战环境

本章介绍强化学习环境的设计与实现，包括标准环境库的使用、自定义环境开发、环境包装器等。环境是强化学习算法测试和应用的基础，良好的环境设计对算法性能有重要影响。

## OpenAI Gym/Gymnasium

### 标准环境

```python
import gymnasium as gym

# 经典控制
env = gym.make('CartPole-v1')      # 平衡杆
env = gym.make('MountainCar-v0')   # 爬山车
env = gym.make('Pendulum-v1')      # 钟摆

# Box2D 物理
env = gym.make('LunarLander-v2')   # 月球着陆器
env = gym.make('BipedalWalker-v3') # 双足行走

# Atari 游戏
env = gym.make('Breakout-v4')      # 打砖块
env = gym.make('Pong-v4')          # 乒乓球

# MuJoCo 物理（需要安装 MuJoCo）
env = gym.make('HalfCheetah-v4')   # 半猎豹
env = gym.make('Humanoid-v4')      # 人形机器人
```

### 基本使用

```python
import gymnasium as gym

env = gym.make('CartPole-v1')

# 重置环境
observation, info = env.reset(seed=42)

for step in range(1000):
    # 随机动作
    action = env.action_space.sample()
    
    # 执行动作
    observation, reward, terminated, truncated, info = env.step(action)
    
    # 渲染
    env.render()
    
    # 检查是否结束
    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

### 空间类型

| 空间类型 | 说明 | 示例 |
|----------|------|------|
| `Discrete(n)` | 离散动作空间，n 个动作 | `Discrete(4)` = {0, 1, 2, 3} |
| `Box(low, high, shape)` | 连续空间 | `Box(-1, 1, (3,))` |
| `MultiDiscrete([n1, n2])` | 多维离散 | `MultiDiscrete([2, 3])` |
| `Dict({...})` | 字典空间 | `Dict({'obs': Box(...), 'mask': Box(...)})` |

```python
# 查看环境信息
env = gym.make('CartPole-v1')

print(f"观察空间: {env.observation_space}")  # Box(-4.8, 4.8, (4,), float32)
print(f"动作空间: {env.action_space}")        # Discrete(2)

# 采样
obs = env.observation_space.sample()
action = env.action_space.sample()
```

## 自定义环境

### 环境接口

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CustomEnv(gym.Env):
    """自定义环境模板"""
    
    def __init__(self):
        super().__init__()
        
        # 定义动作空间
        self.action_space = spaces.Discrete(4)
        
        # 定义观察空间
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(4,), 
            dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)
        
        # 初始化状态
        self.state = np.zeros(4)
        
        return self.state, {}
    
    def step(self, action):
        """执行动作"""
        # 更新状态
        self.state = self._update_state(action)
        
        # 计算奖励
        reward = self._compute_reward()
        
        # 检查终止条件
        terminated = self._check_terminated()
        truncated = False
        
        return self.state, reward, terminated, truncated, {}
    
    def render(self):
        """可视化"""
        pass
    
    def close(self):
        """清理资源"""
        pass
    
    def _update_state(self, action):
        """更新状态"""
        pass
    
    def _compute_reward(self):
        """计算奖励"""
        pass
    
    def _check_terminated(self):
        """检查是否终止"""
        pass
```

### 网格世界环境

```python
class GridWorldEnv(gym.Env):
    """网格世界环境
    
    智能体从起点出发，避开障碍物到达目标
    """
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, size=5, render_mode=None):
        super().__init__()
        
        self.size = size
        self.render_mode = render_mode
        
        # 动作：0=上，1=右，2=下，3=左
        self.action_space = spaces.Discrete(4)
        
        # 观察：智能体位置 (x, y)
        self.observation_space = spaces.Box(
            low=0, high=size-1, shape=(2,), dtype=np.int32
        )
        
        # 目标位置（右下角）
        self.target = (size - 1, size - 1)
        
        # 障碍物
        self.obstacles = {(1, 1), (2, 2), (3, 1)}
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 随机起点（避开障碍物和目标）
        while True:
            self.agent_pos = (
                self.np_random.integers(0, self.size),
                self.np_random.integers(0, self.size)
            )
            if self.agent_pos not in self.obstacles and self.agent_pos != self.target:
                break
        
        return np.array(self.agent_pos, dtype=np.int32), {}
    
    def step(self, action):
        x, y = self.agent_pos
        
        # 执行动作
        if action == 0:    # 上
            y = max(0, y - 1)
        elif action == 1:  # 右
            x = min(self.size - 1, x + 1)
        elif action == 2:  # 下
            y = min(self.size - 1, y + 1)
        elif action == 3:  # 左
            x = max(0, x - 1)
        
        new_pos = (x, y)
        
        # 检查是否撞到障碍物
        if new_pos in self.obstacles:
            reward = -5
            terminated = False
        else:
            self.agent_pos = new_pos
            
            if self.agent_pos == self.target:
                reward = 10
                terminated = True
            else:
                reward = -1  # 每步惩罚
                terminated = False
        
        return (
            np.array(self.agent_pos, dtype=np.int32),
            reward,
            terminated,
            False,
            {}
        )
    
    def render(self):
        if self.render_mode == 'human':
            self._render_text()
        elif self.render_mode == 'rgb_array':
            return self._render_rgb()
    
    def _render_text(self):
        """文本渲染"""
        for i in range(self.size):
            row = ""
            for j in range(self.size):
                pos = (j, i)
                if pos == self.agent_pos:
                    row += "A "
                elif pos == self.target:
                    row += "G "
                elif pos in self.obstacles:
                    row += "X "
                else:
                    row += ". "
            print(row)
        print()


# 注册环境
from gymnasium.envs.registration import register

register(
    id='GridWorld-v0',
    entry_point=GridWorldEnv,
    max_episode_steps=100,
)


# 使用自定义环境
env = gym.make('GridWorld-v0', size=5)
```

## 环境包装器

### 常用包装器

```python
from gymnasium.wrappers import (
    TimeLimit,           # 限制最大步数
    NormalizeObservation, # 标准化观察
    NormalizeReward,      # 标准化奖励
    RecordVideo,          # 录制视频
    FrameStack,           # 帧堆叠
    ResizeObservation,    # 调整观察大小
    GrayScaleObservation, # 灰度化
)

# 组合包装器
env = gym.make('CartPole-v1')
env = TimeLimit(env, max_episode_steps=500)
env = NormalizeObservation(env)
env = NormalizeReward(env)
```

### 自定义包装器

```python
class NormalizeActionWrapper(gym.ActionWrapper):
    """动作归一化包装器
    
    将 [-1, 1] 的动作映射到环境的实际动作范围
    """
    def __init__(self, env):
        super().__init__(env)
        
        self.low = env.action_space.low
        self.high = env.action_space.high
        
        # 新的动作空间
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=env.action_space.shape,
            dtype=np.float32
        )
    
    def action(self, action):
        # 从 [-1, 1] 映射到 [low, high]
        return self.low + (action + 1) * 0.5 * (self.high - self.low)


class FrameStackWrapper(gym.ObservationWrapper):
    """帧堆叠包装器"""
    def __init__(self, env, num_frames=4):
        super().__init__(env)
        
        self.num_frames = num_frames
        self.frames = deque(maxlen=num_frames)
        
        # 更新观察空间
        old_space = env.observation_space
        self.observation_space = spaces.Box(
            low=old_space.low.min(),
            high=old_space.high.max(),
            shape=(num_frames, *old_space.shape),
            dtype=old_space.dtype
        )
    
    def observation(self, observation):
        self.frames.append(observation)
        return np.array(self.frames)
    
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        
        # 填充初始帧
        for _ in range(self.num_frames - 1):
            self.frames.append(obs)
        
        return self.observation(obs), info


# 使用包装器
env = gym.make('Pendulum-v1')
env = NormalizeActionWrapper(env)
```

## Atari 环境处理

```python
from gymnasium.wrappers import (
    AtariPreprocessing,
    FrameStack,
    ResizeObservation,
    GrayScaleObservation,
)

def make_atari_env(env_id, render_mode=None):
    """创建 Atari 环境
    
    标准的 Atari 预处理流程
    """
    env = gym.make(env_id, render_mode=render_mode)
    
    # Atari 标准预处理
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        grayscale_obs=True,
        scale_obs=True
    )
    
    # 帧堆叠
    env = FrameStack(env, num_stack=4)
    
    return env


# 使用
env = make_atari_env('BreakoutNoFrameskip-v4')
```

## 环境向量化

```python
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv

def make_env(env_id, seed=0):
    """创建单个环境的函数"""
    def thunk():
        env = gym.make(env_id)
        env.reset(seed=seed)
        return env
    return thunk

# 同步向量化（单进程）
envs = SyncVectorEnv([
    make_env('CartPole-v1', seed=i) 
    for i in range(4)
])

# 异步向量化（多进程）
envs = AsyncVectorEnv([
    make_env('CartPole-v1', seed=i) 
    for i in range(4)
])

# 使用
observations, infos = envs.reset()
actions = np.array([envs.single_action_space.sample() for _ in range(4)])
observations, rewards, terminateds, truncateds, infos = envs.step(actions)

envs.close()
```

## 实战：测试环境

```python
def test_environment(env, episodes=10, max_steps=1000):
    """测试环境功能"""
    print(f"环境: {env.spec.id}")
    print(f"观察空间: {env.observation_space}")
    print(f"动作空间: {env.action_space}")
    
    rewards = []
    lengths = []
    
    for episode in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        rewards.append(total_reward)
        lengths.append(steps)
    
    print(f"\n测试结果（{episodes} 回合）:")
    print(f"  平均奖励: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  平均步数: {np.mean(lengths):.1f}")


# 测试
env = gym.make('CartPole-v1')
test_environment(env)
env.close()
```

## 核心要点

### 环境设计原则

| 原则 | 说明 |
|------|------|
| 稀疏 vs 稠密奖励 | 稠密奖励更易学习，稀疏奖励更接近真实场景 |
| 观察设计 | 包含足够信息但不过度 |
| 动作空间 | 离散更易探索，连续更灵活 |
| 终止条件 | 合理设置，避免无限循环 |

### 常见问题

| 问题 | 解决方案 |
|------|----------|
| 奖励过于稀疏 | 添加塑形奖励（reward shaping） |
| 观察空间过大 | 降维、特征提取 |
| 动作空间设计不当 | 重新定义动作语义 |
| 环境不稳定 | 固定随机种子 |

## 学习建议

### 实践要点

1. **先跑通标准环境**：熟悉 Gym 接口
2. **实现简单环境**：理解环境设计要点
3. **使用包装器**：模块化处理观察和动作

### 后续方向

- **环境基准测试**：MuJoCo、Atari、ProcGen
- **领域特定环境**：机器人、自动驾驶、金融
- **自定义环境**：针对具体应用开发

---

**上一章**：[大模型强化学习](./llm-rl.md)  
**下一章**：[强化学习工程实践](./rl-engineering.md)
