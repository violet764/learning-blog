# 强化学习实战环境

## 章节概述
本章将介绍强化学习环境的设计与实现，包括标准环境库的使用、自定义环境开发、多模态环境设计等。环境是强化学习算法测试和应用的基础，良好的环境设计对算法性能有重要影响。

## 核心知识点

### 1. 标准环境库

#### 1.1 OpenAI Gym
```python
import gym

# 经典控制环境
env = gym.make('CartPole-v1')
env = gym.make('Pendulum-v1')
env = gym.make('MountainCar-v0')

# Atari游戏环境
env = gym.make('Breakout-v0')
env = gym.make('Pong-v0')

# Box2D物理环境
env = gym.make('LunarLander-v2')

# 使用环境
observation = env.reset()
action = env.action_space.sample()
next_observation, reward, done, info = env.step(action)
```

#### 1.2 Gym环境封装类
```python
class GymWrapper:
    """Gym环境封装"""
    def __init__(self, env_name, max_episode_steps=1000):
        self.env = gym.make(env_name)
        self.max_episode_steps = max_episode_steps
        self.step_count = 0
    
    def reset(self):
        self.step_count = 0
        return self.env.reset()
    
    def step(self, action):
        self.step_count += 1
        obs, reward, done, info = self.env.step(action)
        
        # 添加步数限制
        if self.step_count >= self.max_episode_steps:
            done = True
        
        return obs, reward, done, info
    
    def close(self):
        self.env.close()
```

### 2. 自定义环境开发

#### 2.1 Gym环境接口
```python
import gym
from gym import spaces
import numpy as np

class CustomGridWorld(gym.Env):
    """自定义网格世界环境"""
    
    def __init__(self, grid_size=5):
        super(CustomGridWorld, self).__init__()
        
        self.grid_size = grid_size
        
        # 定义动作空间（上下左右）
        self.action_space = spaces.Discrete(4)
        
        # 定义观察空间（位置坐标）
        self.observation_space = spaces.Box(
            low=0, high=grid_size-1, shape=(2,), dtype=np.int32
        )
        
        # 目标位置
        self.goal = (grid_size-1, grid_size-1)
        
        # 障碍物位置
        self.obstacles = [(1, 1), (2, 2), (3, 1)]
        
        self.reset()
    
    def reset(self):
        # 随机起始位置（避开障碍物）
        while True:
            self.agent_pos = (np.random.randint(0, self.grid_size), 
                            np.random.randint(0, self.grid_size))
            if self.agent_pos not in self.obstacles and self.agent_pos != self.goal:
                break
        
        return np.array(self.agent_pos)
    
    def step(self, action):
        x, y = self.agent_pos
        
        # 动作映射
        if action == 0:  # 上
            y = max(0, y - 1)
        elif action == 1:  # 右
            x = min(self.grid_size-1, x + 1)
        elif action == 2:  # 下
            y = min(self.grid_size-1, y + 1)
        elif action == 3:  # 左
            x = max(0, x - 1)
        
        new_pos = (x, y)
        
        # 检查是否碰到障碍物
        if new_pos in self.obstacles:
            reward = -1
            done = False
        else:
            self.agent_pos = new_pos
            
            # 检查是否到达目标
            if self.agent_pos == self.goal:
                reward = 10
                done = True
            else:
                reward = -0.1  # 每步小惩罚
                done = False
        
        return np.array(self.agent_pos), reward, done, {}
    
    def render(self, mode='human'):
        # 简单文本渲染
        grid = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        
        # 标记目标
        grid[self.goal[1]][self.goal[0]] = 'G'
        
        # 标记障碍物
        for obs in self.obstacles:
            grid[obs[1]][obs[0]] = 'X'
        
        # 标记智能体
        grid[self.agent_pos[1]][self.agent_pos[0]] = 'A'
        
        # 打印网格
        for row in grid:
            print(' '.join(row))
        print()
```

### 3. 多模态环境

#### 3.1 视觉观察环境
```python
class VisualObservationWrapper(gym.ObservationWrapper):
    """添加视觉观察"""
    def __init__(self, env, render_size=(84, 84)):
        super(VisualObservationWrapper, self).__init__(env)
        self.render_size = render_size
        
        # 更新观察空间
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(render_size[0], render_size[1], 3), 
            dtype=np.uint8
        )
    
    def observation(self, observation):
        # 获取环境渲染
        frame = self.env.render(mode='rgb_array')
        
        # 调整尺寸
        import cv2
        frame = cv2.resize(frame, self.render_size)
        
        return frame
```

### 4. 环境测试与验证

#### 4.1 环境测试套件
```python
def test_environment(env, n_episodes=10):
    """测试环境功能"""
    print(f"测试环境: {env.__class__.__name__}")
    print(f"动作空间: {env.action_space}")
    print(f"观察空间: {env.observation_space}")
    
    for episode in range(n_episodes):
        obs = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            action = env.action_space.sample()
            next_obs, reward, done, info = env.step(action)
            
            total_reward += reward
            steps += 1
            obs = next_obs
            
            if done:
                break
        
        print(f"Episode {episode+1}: 奖励={total_reward:.2f}, 步数={steps}")
    
    env.close()

# 测试自定义环境
env = CustomGridWorld(grid_size=5)
test_environment(env)
```

## 实战应用

### 环境比较基准
```python
def benchmark_environments():
    """环境性能基准测试"""
    envs = [
        ('CartPole-v1', gym.make('CartPole-v1')),
        ('Pendulum-v1', gym.make('Pendulum-v1')),
        ('CustomGridWorld', CustomGridWorld())
    ]
    
    for name, env in envs:
        print(f"\n=== 测试 {name} ===")
        
        # 测试随机策略性能
        rewards = []
        for _ in range(100):
            obs = env.reset()
            total_reward = 0
            
            while True:
                action = env.action_space.sample()
                obs, reward, done, _ = env.step(action)
                total_reward += reward
                
                if done:
                    break
            
            rewards.append(total_reward)
        
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        
        print(f"平均奖励: {avg_reward:.2f} ± {std_reward:.2f}")
        env.close()
```

---

**下一章**：强化学习工程实践