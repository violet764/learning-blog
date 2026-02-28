# 强化学习工程实践

本章介绍强化学习的工程实践，包括系统架构设计、实验管理、超参数调优、模型部署和性能监控等。良好的工程实践是强化学习项目成功的关键。

## 项目结构

### 推荐目录结构

```
rl_project/
├── configs/                 # 配置文件
│   ├── default.yaml
│   └── experiments/
├── src/
│   ├── agents/             # 智能体实现
│   │   ├── __init__.py
│   │   ├── dqn.py
│   │   └── ppo.py
│   ├── environments/       # 环境实现
│   │   ├── __init__.py
│   │   └── custom_env.py
│   ├── networks/           # 网络架构
│   │   ├── __init__.py
│   │   └── models.py
│   ├── utils/              # 工具函数
│   │   ├── replay_buffer.py
│   │   └── logger.py
│   └── train.py            # 训练入口
├── tests/                  # 测试代码
├── scripts/                # 脚本
│   └── run_experiment.sh
├── requirements.txt
└── README.md
```

## 配置管理

### 使用 YAML 配置

```yaml
# configs/default.yaml
algorithm: PPO
env:
  name: CartPole-v1
  max_episode_steps: 500

training:
  total_timesteps: 100000
  learning_rate: 3e-4
  batch_size: 64
  gamma: 0.99
  
ppo:
  clip_epsilon: 0.2
  gae_lambda: 0.95
  epochs: 10

logging:
  log_dir: ./logs
  save_freq: 10000
  eval_freq: 5000
```

### 配置加载

```python
import yaml
from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainingConfig:
    """训练配置"""
    algorithm: str
    env_name: str
    learning_rate: float
    batch_size: int
    gamma: float
    total_timesteps: int
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        return cls(
            algorithm=config['algorithm'],
            env_name=config['env']['name'],
            learning_rate=config['training']['learning_rate'],
            batch_size=config['training']['batch_size'],
            gamma=config['training']['gamma'],
            total_timesteps=config['training']['total_timesteps']
        )
    
    def to_dict(self):
        return self.__dict__


# 使用
config = TrainingConfig.from_yaml('configs/default.yaml')
print(config)
```

## 实验跟踪

### TensorBoard 集成

```python
from torch.utils.tensorboard import SummaryWriter
import numpy as np

class RLLogger:
    """强化学习日志记录器"""
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir)
        self.episode_rewards = []
        self.episode_lengths = []
    
    def log_step(self, step: int, metrics: dict):
        """记录单步指标"""
        for key, value in metrics.items():
            self.writer.add_scalar(f'train/{key}', value, step)
    
    def log_episode(self, episode: int, reward: float, length: int):
        """记录回合信息"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        
        self.writer.add_scalar('episode/reward', reward, episode)
        self.writer.add_scalar('episode/length', length, episode)
        
        # 滑动平均
        if len(self.episode_rewards) >= 100:
            avg_reward = np.mean(self.episode_rewards[-100:])
            avg_length = np.mean(self.episode_lengths[-100:])
            self.writer.add_scalar('episode/avg_reward_100', avg_reward, episode)
            self.writer.add_scalar('episode/avg_length_100', avg_length, episode)
    
    def log_hyperparams(self, config: dict):
        """记录超参数"""
        self.writer.add_hparams(config, {})
    
    def log_model(self, model, step: int):
        """记录模型"""
        for name, param in model.named_parameters():
            self.writer.add_histogram(f'parameters/{name}', param, step)
    
    def close(self):
        self.writer.close()
```

### Weights & Biases 集成

```python
import wandb

class WandBLogger:
    """W&B 日志记录器"""
    def __init__(self, project: str, config: dict, name: str = None):
        wandb.init(
            project=project,
            config=config,
            name=name
        )
    
    def log(self, metrics: dict, step: int = None):
        wandb.log(metrics, step=step)
    
    def log_model(self, model, name: str):
        torch.save(model.state_dict(), f'{name}.pt')
        wandb.save(f'{name}.pt')
    
    def finish(self):
        wandb.finish()


# 使用
logger = WandBLogger(
    project='rl-baselines',
    config=config.to_dict(),
    name='ppo-cartpole-v1'
)

# 训练循环中
logger.log({'reward': episode_reward, 'length': episode_length}, step=total_steps)
```

## 超参数调优

### 网格搜索

```python
from itertools import product

def grid_search(env_name, param_grid, train_fn, eval_episodes=10):
    """网格搜索"""
    best_reward = -float('inf')
    best_params = None
    results = []
    
    # 生成所有参数组合
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    for values in product(*param_values):
        params = dict(zip(param_names, values))
        print(f"Testing: {params}")
        
        # 训练模型
        agent = train_fn(env_name, params)
        
        # 评估
        avg_reward = evaluate(agent, env_name, episodes=eval_episodes)
        
        results.append({**params, 'reward': avg_reward})
        
        if avg_reward > best_reward:
            best_reward = avg_reward
            best_params = params
    
    print(f"\nBest params: {best_params}")
    print(f"Best reward: {best_reward:.2f}")
    
    return best_params, results


# 定义搜索空间
param_grid = {
    'learning_rate': [1e-4, 3e-4, 1e-3],
    'batch_size': [32, 64, 128],
    'gamma': [0.95, 0.99, 0.999]
}

best_params, results = grid_search('CartPole-v1', param_grid, train_ppo)
```

### Optuna 调优

```python
import optuna

def objective(trial):
    """Optuna 目标函数"""
    # 定义搜索空间
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    gamma = trial.suggest_float('gamma', 0.9, 0.999)
    clip_epsilon = trial.suggest_float('clip_epsilon', 0.1, 0.3)
    
    # 训练
    config = {
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'gamma': gamma,
        'clip_epsilon': clip_epsilon
    }
    
    agent = train_ppo('CartPole-v1', config, total_timesteps=50000)
    avg_reward = evaluate(agent, 'CartPole-v1', episodes=10)
    
    return avg_reward


# 运行优化
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print(f"Best params: {study.best_params}")
print(f"Best value: {study.best_value}")
```

## 模型保存与加载

```python
import torch
import os

class ModelCheckpoint:
    """模型检查点管理"""
    def __init__(self, save_dir: str, max_to_keep: int = 5):
        self.save_dir = save_dir
        self.max_to_keep = max_to_keep
        self.saved_models = []
        
        os.makedirs(save_dir, exist_ok=True)
    
    def save(self, model, optimizer, step: int, metrics: dict):
        """保存模型"""
        filename = f'model_step_{step}.pt'
        path = os.path.join(self.save_dir, filename)
        
        torch.save({
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }, path)
        
        self.saved_models.append(path)
        
        # 删除旧模型
        while len(self.saved_models) > self.max_to_keep:
            old_path = self.saved_models.pop(0)
            if os.path.exists(old_path):
                os.remove(old_path)
        
        # 同时保存最佳模型
        if metrics.get('is_best', False):
            best_path = os.path.join(self.save_dir, 'best_model.pt')
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics
            }, best_path)
    
    def load_latest(self, model, optimizer):
        """加载最新模型"""
        if not self.saved_models:
            return None
        
        latest_path = self.saved_models[-1]
        checkpoint = torch.load(latest_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint['step'], checkpoint['metrics']
    
    def load_best(self, model, optimizer):
        """加载最佳模型"""
        best_path = os.path.join(self.save_dir, 'best_model.pt')
        
        if not os.path.exists(best_path):
            return None
        
        checkpoint = torch.load(best_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint['step'], checkpoint['metrics']
```

## 分布式训练

### 数据并行

```python
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def train_worker(rank, world_size, env_name, shared_model, optimizer):
    """分布式训练工作进程"""
    # 初始化进程组
    dist.init_process_group('gloo', rank=rank, world_size=world_size)
    
    # 创建环境
    env = gym.make(env_name)
    
    # 包装模型
    model = DDP(shared_model)
    
    # 训练循环
    for episode in range(1000):
        # ... 收集经验 ...
        
        # 同步更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def distributed_train(env_name, num_workers=4):
    """启动分布式训练"""
    world_size = num_workers
    
    # 创建共享模型
    model = DQN(state_dim=4, action_dim=2)
    model.share_memory()  # 共享内存
    
    optimizer = torch.optim.Adam(model.parameters())
    
    # 启动工作进程
    mp.spawn(
        train_worker,
        args=(world_size, env_name, model, optimizer),
        nprocs=world_size
    )
```

### 经验收集并行

```python
from multiprocessing import Process, Queue

def collect_worker(env_name, policy_queue, experience_queue, worker_id):
    """经验收集工作进程"""
    env = gym.make(env_name)
    
    while True:
        # 获取最新策略参数
        policy_params = policy_queue.get()
        if policy_params is None:
            break
        
        # 收集经验
        experiences = []
        state = env.reset()
        
        for _ in range(1000):
            action = select_action(state, policy_params)
            next_state, reward, done, _ = env.step(action)
            experiences.append((state, action, reward, next_state, done))
            state = next_state
            
            if done:
                state = env.reset()
        
        experience_queue.put(experiences)


class ParallelCollector:
    """并行经验收集器"""
    def __init__(self, env_name, num_workers=4):
        self.policy_queue = Queue()
        self.experience_queue = Queue()
        
        self.workers = [
            Process(
                target=collect_worker,
                args=(env_name, self.policy_queue, self.experience_queue, i)
            )
            for i in range(num_workers)
        ]
        
        for w in self.workers:
            w.start()
    
    def collect(self, policy_params):
        """收集经验"""
        # 发送策略参数
        for _ in range(len(self.workers)):
            self.policy_queue.put(policy_params)
        
        # 收集经验
        all_experiences = []
        for _ in range(len(self.workers)):
            experiences = self.experience_queue.get()
            all_experiences.extend(experiences)
        
        return all_experiences
    
    def close(self):
        for _ in range(len(self.workers)):
            self.policy_queue.put(None)
        for w in self.workers:
            w.join()
```

## 模型部署

### Flask API 服务

```python
from flask import Flask, request, jsonify
import torch

app = Flask(__name__)

class RLService:
    """强化学习服务"""
    def __init__(self, model_path):
        self.model = DQN(state_dim=4, action_dim=2)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
    
    def predict(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.model(state_tensor)
            action = q_values.argmax(1).item()
        return action

service = None

@app.before_first_request
def init_service():
    global service
    service = RLService('best_model.pt')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    state = data['state']
    
    action = service.predict(state)
    
    return jsonify({'action': action})

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### ONNX 导出

```python
def export_to_onnx(model, state_dim, output_path):
    """导出为 ONNX 格式"""
    model.eval()
    
    dummy_input = torch.randn(1, state_dim)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['state'],
        output_names=['q_values'],
        dynamic_axes={
            'state': {0: 'batch_size'},
            'q_values': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported to {output_path}")


# 导出
export_to_onnx(trained_model, state_dim=4, output_path='dqn.onnx')
```

## 单元测试

```python
import unittest
import torch
import numpy as np

class TestReplayBuffer(unittest.TestCase):
    """测试经验回放缓冲区"""
    
    def setUp(self):
        self.buffer = ReplayBuffer(capacity=100)
    
    def test_push_and_sample(self):
        # 测试存储
        for i in range(50):
            self.buffer.push(
                state=np.zeros(4),
                action=i,
                reward=1.0,
                next_state=np.ones(4),
                done=False
            )
        
        self.assertEqual(len(self.buffer), 50)
        
        # 测试采样
        batch = self.buffer.sample(10)
        self.assertEqual(len(batch[0]), 10)
    
    def test_capacity(self):
        # 测试容量限制
        for i in range(150):
            self.buffer.push(
                state=np.zeros(4),
                action=i,
                reward=1.0,
                next_state=np.ones(4),
                done=False
            )
        
        self.assertEqual(len(self.buffer), 100)


class TestDQN(unittest.TestCase):
    """测试 DQN 网络"""
    
    def setUp(self):
        self.model = DQN(state_dim=4, action_dim=2)
    
    def test_forward(self):
        state = torch.randn(10, 4)
        q_values = self.model(state)
        
        self.assertEqual(q_values.shape, (10, 2))
    
    def test_select_action(self):
        agent = DQNAgent(state_dim=4, action_dim=2)
        state = np.zeros(4)
        
        action = agent.select_action(state)
        
        self.assertIn(action, [0, 1])


if __name__ == '__main__':
    unittest.main()
```

## 核心要点

### 工程最佳实践

| 实践 | 说明 |
|------|------|
| 配置管理 | 使用 YAML/JSON 分离配置和代码 |
| 实验跟踪 | 记录所有超参数和指标 |
| 版本控制 | Git 管理代码，DVC 管理数据 |
| 模型检查点 | 定期保存，保留最佳模型 |
| 单元测试 | 测试关键组件 |

### 调试技巧

| 问题 | 排查方法 |
|------|----------|
| 训练不稳定 | 检查梯度、学习率、奖励尺度 |
| 不收敛 | 检查网络架构、探索策略 |
| 内存溢出 | 减小 batch_size，使用经验回放 |

## 学习建议

### 实践要点

1. **从小项目开始**：先跑通简单环境和算法
2. **记录一切**：日志、配置、实验结果
3. **模块化设计**：便于复用和调试

### 后续方向

- **大规模训练**：分布式、多 GPU
- **生产部署**：模型压缩、推理优化
- **持续学习**：在线更新、A/B 测试

---

**上一章**：[强化学习实战环境](./rl-environments.md)  
**返回**：[强化学习总览](./index.md)
