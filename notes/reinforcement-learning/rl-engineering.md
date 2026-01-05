# 强化学习工程实践

## 章节概述
本章将介绍强化学习的工程实践，包括系统架构设计、实验管理、超参数调优、模型部署和性能监控等。良好的工程实践是强化学习项目成功的关键，能够显著提高开发效率和系统可靠性。

## 核心知识点

### 1. 强化学习系统架构

#### 1.1 分布式训练架构
```python
class DistributedRLSystem:
    """分布式强化学习系统"""
    def __init__(self, num_workers=4):
        self.num_workers = num_workers
        self.workers = []
        
        # 中央经验缓冲区
        self.replay_buffer = ReplayBuffer(capacity=1000000)
        
        # 中央模型
        self.central_model = DQN(state_dim=4, action_dim=2)
        
    def start_training(self):
        """启动分布式训练"""
        # 启动工作进程
        for i in range(self.num_workers):
            worker = Process(target=self.worker_loop, args=(i,))
            worker.start()
            self.workers.append(worker)
        
        # 中央训练循环
        self.central_training_loop()
    
    def worker_loop(self, worker_id):
        """工作进程循环"""
        env = gym.make('CartPole-v1')
        
        while True:
            # 获取最新模型参数
            model_params = self.get_latest_params()
            
            # 收集经验
            experiences = self.collect_experiences(env, model_params)
            
            # 发送经验到中央缓冲区
            self.send_experiences(experiences)
    
    def central_training_loop(self):
        """中央训练循环"""
        while True:
            # 从缓冲区采样
            batch = self.replay_buffer.sample(batch_size=32)
            
            # 更新模型
            loss = self.central_model.update(batch)
            
            # 定期保存模型
            if self.step_count % 1000 == 0:
                self.save_model()
```

### 2. 实验管理与跟踪

#### 2.1 实验配置管理
```python
import yaml
from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    """实验配置"""
    algorithm: str
    env_name: str
    learning_rate: float
    gamma: float
    batch_size: int
    max_episodes: int
    
    @classmethod
    def from_yaml(cls, config_path):
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, config_path):
        with open(config_path, 'w') as f:
            yaml.dump(self.__dict__, f)

# 示例配置
config = ExperimentConfig(
    algorithm="PPO",
    env_name="CartPole-v1",
    learning_rate=3e-4,
    gamma=0.99,
    batch_size=64,
    max_episodes=10000
)
```

#### 2.2 实验跟踪（W&B集成）
```python
import wandb

class ExperimentTracker:
    """实验跟踪器"""
    def __init__(self, project_name, config):
        wandb.init(project=project_name, config=config)
        self.step = 0
    
    def log_metrics(self, metrics):
        """记录指标"""
        wandb.log(metrics, step=self.step)
        self.step += 1
    
    def log_model(self, model, model_name):
        """记录模型"""
        torch.save(model.state_dict(), f"{model_name}.pth")
        wandb.save(f"{model_name}.pth")
    
    def finish(self):
        wandb.finish()

# 使用示例
tracker = ExperimentTracker("rl-baselines", config.__dict__)

for episode in range(config.max_episodes):
    # 训练逻辑...
    episode_reward = train_episode()
    
    # 记录指标
    tracker.log_metrics({
        "episode_reward": episode_reward,
        "epsilon": agent.epsilon
    })
```

### 3. 超参数优化

#### 3.1 网格搜索
```python
from itertools import product

def grid_search():
    """网格搜索超参数"""
    learning_rates = [1e-3, 3e-4, 1e-4]
    gammas = [0.9, 0.95, 0.99]
    batch_sizes = [32, 64, 128]
    
    best_reward = -float('inf')
    best_params = None
    
    for lr, gamma, batch_size in product(learning_rates, gammas, batch_sizes):
        print(f"测试参数: lr={lr}, gamma={gamma}, batch_size={batch_size}")
        
        # 使用当前参数训练
        agent = DQNAgent(learning_rate=lr, gamma=gamma, batch_size=batch_size)
        avg_reward = evaluate_agent(agent, episodes=10)
        
        if avg_reward > best_reward:
            best_reward = avg_reward
            best_params = (lr, gamma, batch_size)
    
    print(f"最佳参数: {best_params}, 最佳奖励: {best_reward}")
    return best_params
```

#### 3.2 贝叶斯优化
```python
from skopt import gp_minimize
from skopt.space import Real, Integer

def bayesian_optimization():
    """贝叶斯优化超参数"""
    # 定义搜索空间
    space = [
        Real(1e-5, 1e-2, name='learning_rate'),
        Real(0.8, 0.999, name='gamma'),
        Integer(16, 256, name='batch_size')
    ]
    
    def objective(params):
        lr, gamma, batch_size = params
        
        # 负奖励作为目标（最小化）
        agent = DQNAgent(learning_rate=lr, gamma=gamma, batch_size=batch_size)
        avg_reward = evaluate_agent(agent, episodes=5)
        
        return -avg_reward  # 最小化负奖励
    
    result = gp_minimize(objective, space, n_calls=50, random_state=42)
    
    print(f"最佳参数: {result.x}")
    print(f"最佳奖励: {-result.fun}")
    
    return result.x
```

### 4. 模型部署与监控

#### 4.1 模型服务化
```python
from flask import Flask, request, jsonify
import torch

app = Flask(__name__)

class RLModelServer:
    """强化学习模型服务"""
    def __init__(self, model_path):
        self.model = DQN(state_dim=4, action_dim=2)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
    
    def predict(self, state):
        """预测动作"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.model(state_tensor)
            action = q_values.max(1)[1].item()
        
        return action

# 初始化服务
model_server = RLModelServer("best_model.pth")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    state = data['state']
    
    action = model_server.predict(state)
    
    return jsonify({'action': action})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

#### 4.2 性能监控
```python
import psutil
import time

class PerformanceMonitor:
    """性能监控器"""
    def __init__(self):
        self.metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'inference_time': []
        }
    
    def start_monitoring(self):
        """开始监控"""
        self.running = True
        
        while self.running:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            self.metrics['cpu_usage'].append(cpu_percent)
            
            # 内存使用
            memory_info = psutil.virtual_memory()
            self.metrics['memory_usage'].append(memory_info.percent)
            
            time.sleep(5)  # 每5秒记录一次
    
    def record_inference(self, inference_time):
        """记录推理时间"""
        self.metrics['inference_time'].append(inference_time)
    
    def get_summary(self):
        """获取性能摘要"""
        summary = {}
        
        for metric, values in self.metrics.items():
            if values:
                summary[f'{metric}_mean'] = np.mean(values)
                summary[f'{metric}_max'] = np.max(values)
                summary[f'{metric}_min'] = np.min(values)
        
        return summary
```

### 5. 持续集成与测试

#### 5.1 单元测试
```python
import unittest

class TestRLComponents(unittest.TestCase):
    """强化学习组件测试"""
    
    def test_replay_buffer(self):
        """测试经验回放缓冲区"""
        buffer = ReplayBuffer(capacity=10)
        
        # 测试存储
        for i in range(5):
            buffer.push(i, i, i, i, False)
        
        self.assertEqual(len(buffer), 5)
        
        # 测试采样
        batch = buffer.sample(3)
        self.assertEqual(len(batch[0]), 3)
    
    def test_dqn_update(self):
        """测试DQN更新"""
        agent = DQNAgent(state_dim=4, action_dim=2)
        
        # 添加一些经验
        for i in range(10):
            agent.store_transition([0]*4, 0, 1, [0]*4, False)
        
        # 测试更新
        loss = agent.update()
        self.assertIsInstance(loss, float)

if __name__ == '__main__':
    unittest.main()
```

#### 5.2 集成测试
```python
def integration_test():
    """集成测试"""
    # 测试完整训练流程
    env = gym.make('CartPole-v1')
    agent = DQNAgent(state_dim=4, action_dim=2)
    
    # 训练几个回合
    for episode in range(10):
        state = env.reset()
        total_reward = 0
        
        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        print(f"Episode {episode}: Reward {total_reward}")
    
    # 验证模型能够预测
    test_state = env.reset()
    action = agent.select_action(test_state)
    assert action in [0, 1], "动作应该在有效范围内"
    
    print("集成测试通过!")
```

## 工程最佳实践

### 代码组织
```
rl_project/
├── src/
│   ├── agents/          # 智能体实现
│   ├── environments/    # 环境实现
│   ├── networks/        # 网络架构
│   └── utils/          # 工具函数
├── configs/            # 配置文件
├── experiments/        # 实验记录
├── tests/             # 测试代码
└── deployment/        # 部署配置
```

### 版本控制
- 使用Git进行版本控制
- 为每个实验创建独立分支
- 使用标签标记重要版本

### 文档编写
- 为每个模块编写文档字符串
- 维护README文件说明项目结构
- 记录实验设置和结果

---

**下一章**：前沿应用与未来发展