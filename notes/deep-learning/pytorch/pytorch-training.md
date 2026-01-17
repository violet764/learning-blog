# PyTorch模型训练与优化

## 章节概述

模型训练是深度学习的核心环节，PyTorch提供了完整的训练框架。本章将深入讲解模型训练的全流程，包括优化器选择、损失函数配置、训练循环设计、性能监控和调试技巧。

## 基础训练流程

### 完整的训练循环
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

print("=== 基础训练循环 ===")

class BasicTrainer:
    """基础训练器"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.to(device)
        
        # 训练历史记录
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def train_epoch(self, train_loader, criterion, optimizer):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # 前向传播
            outputs = self.model(data)
            loss = criterion(outputs, target)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 统计信息
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'批次 {batch_idx}/{len(train_loader)} | '
                      f'损失: {loss.item():.3f}')
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader, criterion):
        """验证模型"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                loss = criterion(outputs, target)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train(self, train_loader, val_loader, criterion, optimizer, 
              scheduler=None, epochs=100):
        """完整训练过程"""
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            
            # 训练阶段
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            
            # 验证阶段
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            # 记录历史
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            # 学习率调度
            if scheduler:
                scheduler.step()
            
            print(f'训练损失: {train_loss:.3f} | 训练准确率: {train_acc:.2f}%')
            print(f'验证损失: {val_loss:.3f} | 验证准确率: {val_acc:.2f}%')
            
            # 早停检查（简化版）
            if epoch > 10 and val_loss > min(self.val_losses[:-5]):
                print("验证损失上升，考虑早停")
                break
        
        return self.train_losses, self.val_losses, self.train_accs, self.val_accs
    
    def plot_training_history(self):
        """绘制训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 损失曲线
        ax1.plot(self.train_losses, label='训练损失')
        ax1.plot(self.val_losses, label='验证损失')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.set_title('损失曲线')
        
        # 准确率曲线
        ax2.plot(self.train_accs, label='训练准确率')
        ax2.plot(self.val_accs, label='验证准确率')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.set_title('准确率曲线')
        
        plt.tight_layout()
        plt.show()

print("基础训练器定义完成")
```

## 优化器详解

### 常用优化器
```python
print("\n=== 优化器详解 ===")

class OptimizerComparison:
    """优化器比较"""
    
    def __init__(self, model, lr=0.01):
        self.model = model
        self.lr = lr
    
    def get_sgd(self, momentum=0.9):
        """随机梯度下降"""
        return optim.SGD(self.model.parameters(), lr=self.lr, momentum=momentum)
    
    def get_adam(self, betas=(0.9, 0.999)):
        """Adam优化器"""
        return optim.Adam(self.model.parameters(), lr=self.lr, betas=betas)
    
    def get_adamw(self, weight_decay=0.01):
        """AdamW优化器（解耦权重衰减）"""
        return optim.AdamW(self.model.parameters(), lr=self.lr, 
                          weight_decay=weight_decay)
    
    def get_rmsprop(self, alpha=0.99):
        """RMSProp优化器"""
        return optim.RMSprop(self.model.parameters(), lr=self.lr, alpha=alpha)
    
    def get_adagrad(self):
        """AdaGrad优化器"""
        return optim.Adagrad(self.model.parameters(), lr=self.lr)

# 优化器使用示例
model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 10)
)

optimizer_comp = OptimizerComparison(model)

sgd_optimizer = optimizer_comp.get_sgd()
adam_optimizer = optimizer_comp.get_adam()
adamw_optimizer = optimizer_comp.get_adamw()

print("SGD优化器:", sgd_optimizer)
print("Adam优化器:", adam_optimizer)
print("AdamW优化器:", adamw_optimizer)
```

### 优化器超参数调优
```python
print("\n=== 优化器超参数调优 ===")

def find_optimal_lr(model, train_loader, criterion, 
                   lr_min=1e-6, lr_max=1, num_iter=100):
    """学习率查找器"""
    
    # 保存原始状态
    original_state = model.state_dict().copy()
    
    # 设置优化器
    optimizer = optim.Adam(model.parameters(), lr=lr_min)
    
    # 学习率调度器（指数增长）
    lr_lambda = lambda iteration: (lr_max / lr_min) ** (iteration / num_iter)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    losses = []
    learning_rates = []
    
    model.train()
    iteration = 0
    
    for data, target in train_loader:
        if iteration >= num_iter:
            break
        
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        
        # 计算梯度但不更新参数
        loss.backward()
        
        losses.append(loss.item())
        learning_rates.append(optimizer.param_groups[0]['lr'])
        
        # 更新学习率
        scheduler.step()
        iteration += 1
    
    # 恢复原始状态
    model.load_state_dict(original_state)
    
    return learning_rates, losses

print("学习率查找器定义完成")
```

## 损失函数配置

### 常用损失函数
```python
print("\n=== 损失函数配置 ===")

class LossFunctionManager:
    """损失函数管理器"""
    
    def __init__(self):
        self.loss_functions = {}
    
    def register_loss(self, name, loss_fn):
        """注册损失函数"""
        self.loss_functions[name] = loss_fn
    
    def get_loss(self, name, **kwargs):
        """获取损失函数"""
        if name == 'mse':
            return nn.MSELoss(**kwargs)
        elif name == 'cross_entropy':
            return nn.CrossEntropyLoss(**kwargs)
        elif name == 'bce':
            return nn.BCEWithLogitsLoss(**kwargs)
        elif name == 'l1':
            return nn.L1Loss(**kwargs)
        elif name in self.loss_functions:
            return self.loss_functions[name]
        else:
            raise ValueError(f"未知的损失函数: {name}")

# 使用示例
loss_manager = LossFunctionManager()

# 注册自定义损失函数
class FocalLoss(nn.Module):
    """Focal Loss用于类别不平衡"""
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

loss_manager.register_loss('focal', FocalLoss())

# 获取不同损失函数
mse_loss = loss_manager.get_loss('mse')
ce_loss = loss_manager.get_loss('cross_entropy')
focal_loss = loss_manager.get_loss('focal')

print("MSE损失:", mse_loss)
print("交叉熵损失:", ce_loss)
print("Focal损失:", focal_loss)
```

### 多任务损失函数
```python
print("\n=== 多任务损失函数 ===")

class MultiTaskLoss(nn.Module):
    """多任务学习损失函数"""
    
    def __init__(self, task_weights=None):
        super(MultiTaskLoss, self).__init__()
        self.task_weights = task_weights or {}
        
        # 不同任务的损失函数
        self.classification_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()
        self.segmentation_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, outputs, targets):
        total_loss = 0.0
        
        # 分类任务损失
        if 'classification' in outputs:
            weight = self.task_weights.get('classification', 1.0)
            loss = self.classification_loss(
                outputs['classification'], 
                targets['classification']
            )
            total_loss += weight * loss
        
        # 回归任务损失
        if 'regression' in outputs:
            weight = self.task_weights.get('regression', 1.0)
            loss = self.regression_loss(
                outputs['regression'], 
                targets['regression']
            )
            total_loss += weight * loss
        
        # 分割任务损失
        if 'segmentation' in outputs:
            weight = self.task_weights.get('segmentation', 1.0)
            loss = self.segmentation_loss(
                outputs['segmentation'], 
                targets['segmentation']
            )
            total_loss += weight * loss
        
        return total_loss

# 使用示例
multi_loss = MultiTaskLoss({
    'classification': 1.0,
    'regression': 0.5,
    'segmentation': 2.0
})

print("多任务损失函数定义完成")
```

## 学习率调度

### 常用学习率调度器
```python
print("\n=== 学习率调度 ===")

class LearningRateSchedulerManager:
    """学习率调度器管理器"""
    
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.schedulers = {}
    
    def get_step_scheduler(self, step_size=30, gamma=0.1):
        """步长衰减调度器"""
        return optim.lr_scheduler.StepLR(
            self.optimizer, step_size=step_size, gamma=gamma
        )
    
    def get_cosine_scheduler(self, T_max=10, eta_min=0):
        """余弦退火调度器"""
        return optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=T_max, eta_min=eta_min
        )
    
    def get_exponential_scheduler(self, gamma=0.95):
        """指数衰减调度器"""
        return optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
    
    def get_plateau_scheduler(self, mode='min', patience=10, factor=0.1):
        """基于平台检测的调度器"""
        return optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode=mode, patience=patience, factor=factor
        )
    
    def get_one_cycle_scheduler(self, max_lr, epochs, steps_per_epoch):
        """单周期学习率调度器"""
        return optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=max_lr, epochs=epochs, 
            steps_per_epoch=steps_per_epoch
        )

# 使用示例
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler_manager = LearningRateSchedulerManager(optimizer)

# 不同调度器
step_scheduler = scheduler_manager.get_step_scheduler(step_size=30)
cosine_scheduler = scheduler_manager.get_cosine_scheduler(T_max=50)
plateau_scheduler = scheduler_manager.get_plateau_scheduler(patience=5)

print("步长衰减调度器:", step_scheduler)
print("余弦退火调度器:", cosine_scheduler)
print("平台检测调度器:", plateau_scheduler)
```

### 学习率预热
```python
print("\n=== 学习率预热 ===")

def get_warmup_scheduler(optimizer, warmup_epochs, base_lr):
    """学习率预热调度器"""
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # 线性预热
            return (epoch + 1) / warmup_epochs
        else:
            # 后续可以使用其他调度策略
            return 1.0
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# 组合预热和余弦退火
def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, base_lr):
    """预热+余弦退火调度器"""
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # 线性预热
            return (epoch + 1) / warmup_epochs
        else:
            # 余弦退火
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

print("学习率预热调度器定义完成")
```

## 高级训练技巧

### 梯度累积
```python
print("\n=== 梯度累积 ===")

def train_with_gradient_accumulation(model, train_loader, criterion, 
                                   optimizer, accumulation_steps=4):
    """梯度累积训练"""
    model.train()
    optimizer.zero_grad()  # 在累积开始前清零梯度
    
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # 前向传播
        outputs = model(data)
        loss = criterion(outputs, target)
        
        # 缩放损失（考虑累积步数）
        loss = loss / accumulation_steps
        
        # 反向传播（累积梯度）
        loss.backward()
        
        # 每accumulation_steps步更新一次参数
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            
            print(f'更新参数，批次: {i+1}')
    
    # 处理最后一个不完整的累积批次
    if len(train_loader) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

print("梯度累积训练函数定义完成")
```

### 混合精度训练
```python
print("\n=== 混合精度训练 ===")

from torch.cuda.amp import autocast, GradScaler

def train_with_mixed_precision(model, train_loader, criterion, optimizer):
    """混合精度训练"""
    model.train()
    scaler = GradScaler()  # 梯度缩放器
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        # 使用自动混合精度
        with autocast():
            outputs = model(data)
            loss = criterion(outputs, target)
        
        # 缩放损失并反向传播
        scaler.scale(loss).backward()
        
        # 梯度裁剪（可选）
        # scaler.unscale_(optimizer)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 更新参数
        scaler.step(optimizer)
        scaler.update()

print("混合精度训练函数定义完成")
```

### 知识蒸馏
```python
print("\n=== 知识蒸馏 ===")

class KnowledgeDistillationLoss(nn.Module):
    """知识蒸馏损失函数"""
    
    def __init__(self, alpha=0.7, temperature=4):
        super(KnowledgeDistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_logits, teacher_logits, targets):
        # 教师模型的软标签
        teacher_probs = torch.softmax(teacher_logits / self.temperature, dim=1)
        
        # 学生模型的软预测
        student_log_probs = torch.log_softmax(student_logits / self.temperature, dim=1)
        
        # 蒸馏损失（学生模仿教师）
        distillation_loss = self.kl_loss(student_log_probs, teacher_probs) * 
                           (self.temperature ** 2)
        
        # 学生自己的分类损失
        student_loss = self.ce_loss(student_logits, targets)
        
        # 组合损失
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * student_loss
        
        return total_loss

# 使用示例
kd_loss = KnowledgeDistillationLoss(alpha=0.7, temperature=4)

print("知识蒸馏损失函数定义完成")
```

## 模型评估与调试

### 评估指标计算
```python
print("\n=== 模型评估指标 ===")

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """重置统计信息"""
        self.correct = 0
        self.total = 0
        self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes)
    
    def update(self, outputs, targets):
        """更新统计信息"""
        _, predicted = outputs.max(1)
        self.total += targets.size(0)
        self.correct += predicted.eq(targets).sum().item()
        
        # 更新混淆矩阵
        for t, p in zip(targets.view(-1), predicted.view(-1)):
            self.confusion_matrix[t.long(), p.long()] += 1
    
    def accuracy(self):
        """计算准确率"""
        return 100. * self.correct / self.total if self.total > 0 else 0
    
    def precision(self, class_idx):
        """计算精确率"""
        true_positives = self.confusion_matrix[class_idx, class_idx]
        false_positives = self.confusion_matrix[:, class_idx].sum() - true_positives
        return true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    
    def recall(self, class_idx):
        """计算召回率"""
        true_positives = self.confusion_matrix[class_idx, class_idx]
        false_negatives = self.confusion_matrix[class_idx, :].sum() - true_positives
        return true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    def f1_score(self, class_idx):
        """计算F1分数"""
        prec = self.precision(class_idx)
        rec = self.recall(class_idx)
        return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    
    def print_report(self, class_names=None):
        """打印评估报告"""
        print(f"总体准确率: {self.accuracy():.2f}%")
        print("\n各类别指标:")
        for i in range(self.num_classes):
            class_name = class_names[i] if class_names else f"类别{i}"
            print(f"{class_name}: 精确率={self.precision(i):.3f}, "
                  f"召回率={self.recall(i):.3f}, F1={self.f1_score(i):.3f}")

# 使用示例
evaluator = ModelEvaluator(num_classes=10)
print("模型评估器定义完成")
```

### 训练过程可视化
```python
print("\n=== 训练过程可视化 ===")

import matplotlib.pyplot as plt
import seaborn as sns

def visualize_training_progress(trainer):
    """可视化训练进度"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 损失曲线
    ax1.plot(trainer.train_losses, label='训练损失')
    ax1.plot(trainer.val_losses, label='验证损失')
    ax1.set_title('损失曲线')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 准确率曲线
    ax2.plot(trainer.train_accs, label='训练准确率')
    ax2.plot(trainer.val_accs, label='验证准确率')
    ax2.set_title('准确率曲线')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    # 学习率曲线（如果有调度器）
    if hasattr(trainer, 'learning_rates') and trainer.learning_rates:
        ax3.plot(trainer.learning_rates)
        ax3.set_title('学习率变化')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True)
    
    # 梯度范数（如果有记录）
    if hasattr(trainer, 'grad_norms') and trainer.grad_norms:
        ax4.plot(trainer.grad_norms)
        ax4.set_title('梯度范数')
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Gradient Norm')
        ax4.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(evaluator, class_names):
    """绘制混淆矩阵"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(evaluator.confusion_matrix.numpy(), 
                annot=True, fmt='g', 
                xticklabels=class_names, 
                yticklabels=class_names,
                cmap='Blues')
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.show()

print("训练可视化函数定义完成")
```

## 实际训练示例

### 完整训练流程
```python
print("\n=== 完整训练流程示例 ===")

def complete_training_pipeline():
    """完整训练流程"""
    
    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 模型定义（简化）
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, 10)
    ).to(device)
    
    # 数据加载器（简化）
    # train_loader, val_loader = get_data_loaders()
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    # 创建训练器
    trainer = BasicTrainer(model, device)
    
    # 开始训练
    print("开始训练...")
    # train_losses, val_losses, train_accs, val_accs = trainer.train(
    #     train_loader, val_loader, criterion, optimizer, scheduler, epochs=100
    # )
    
    # 可视化训练结果
    # trainer.plot_training_history()
    
    print("训练流程定义完成")

complete_training_pipeline()
```

## 调试与性能优化

### 常见问题调试
```python
print("\n=== 训练调试技巧 ===")

def debug_training_issues(model, train_loader):
    """调试训练问题"""
    
    print("=== 训练问题调试 ===")
    
    # 1. 检查模型参数
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: 形状 {param.shape}, 均值 {param.mean().item():.6f}, "
                  f"标准差 {param.std().item():.6f}")
    
    # 2. 检查第一个批次的前向传播
    model.eval()
    with torch.no_grad():
        for data, target in train_loader:
            outputs = model(data)
            print(f"模型输出范围: [{outputs.min().item():.3f}, {outputs.max().item():.3f}]")
            print(f"输出标准差: {outputs.std().item():.3f}")
            break
    
    # 3. 检查梯度
    model.train()
    optimizer.zero_grad()
    for data, target in train_loader:
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        
        # 检查梯度
        total_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.norm()
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        print(f"梯度范数: {total_norm:.6f}")
        
        if total_norm > 1000:
            print("警告: 梯度爆炸!")
        elif total_norm < 1e-6:
            print("警告: 梯度消失!")
        
        break
    
    print("调试完成")

print("训练调试函数定义完成")
```

## 总结

本章详细介绍了PyTorch模型训练与优化的各个方面：

1. **基础训练流程**：完整的训练循环和验证过程
2. **优化器详解**：各种优化器的比较和选择
3. **损失函数配置**：单任务和多任务损失函数
4. **学习率调度**：预热、退火和平台检测等策略
5. **高级训练技巧**：梯度累积、混合精度、知识蒸馏
6. **模型评估**：准确率、精确率、召回率等指标
7. **可视化工具**：训练过程监控和结果分析
8. **调试技巧**：常见问题的诊断和解决

掌握这些训练技术对于构建高性能的深度学习模型至关重要。