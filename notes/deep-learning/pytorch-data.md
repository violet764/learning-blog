# PyTorch数据加载与处理

## 章节概述

数据是深度学习的核心，PyTorch提供了强大的数据加载和处理工具。本章将详细介绍`torch.utils.data`模块，包括Dataset、DataLoader的设计，数据预处理，数据增强技术，以及处理各种类型数据的最佳实践。

## Dataset与DataLoader基础

### 自定义Dataset类
```python
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import pandas as pd

print("=== 自定义Dataset类 ===")

class CustomDataset(Dataset):
    """自定义数据集类的基础实现"""
    
    def __init__(self, data, labels, transform=None):
        """
        参数:
            data: 特征数据
            labels: 标签数据
            transform: 数据变换
        """
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label

# 使用示例
# 模拟数据
X = np.random.randn(100, 3, 32, 32)  # 100个32x32 RGB图像
y = np.random.randint(0, 10, 100)    # 10个类别

# 创建数据集
dataset = CustomDataset(X, y)
print(f"数据集大小: {len(dataset)}")
print(f"第一个样本形状: {dataset[0][0].shape}")
print(f"第一个标签: {dataset[0][1]}")
```

### DataLoader配置
```python
print("\n=== DataLoader配置 ===")

# 创建DataLoader
dataloader = DataLoader(
    dataset=dataset,
    batch_size=32,
    shuffle=True,           # 训练时打乱数据
    num_workers=4,          # 并行加载的进程数
    pin_memory=True,        # 加速GPU数据传输
    drop_last=False         # 是否丢弃最后一个不完整的批次
)

print(f"DataLoader批次数量: {len(dataloader)}")
print(f"批次大小: {dataloader.batch_size}")

# 遍历DataLoader
for batch_idx, (data, target) in enumerate(dataloader):
    print(f"批次 {batch_idx}: 数据形状 {data.shape}, 标签形状 {target.shape}")
    if batch_idx == 2:  # 只显示前3个批次
        break
```

## 图像数据处理

### 图像变换与增强
```python
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

print("\n=== 图像变换与增强 ===")

# 基础图像变换
basic_transform = transforms.Compose([
    transforms.Resize((224, 224)),          # 调整大小
    transforms.ToTensor(),                   # 转换为张量
    transforms.Normalize(                    # 标准化
        mean=[0.485, 0.456, 0.406],         # ImageNet均值
        std=[0.229, 0.224, 0.225]           # ImageNet标准差
    )
])

# 训练时的数据增强
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),       # 随机裁剪并缩放
    transforms.RandomHorizontalFlip(0.5),    # 随机水平翻转
    transforms.RandomRotation(10),           # 随机旋转
    transforms.ColorJitter(                  # 颜色抖动
        brightness=0.2,                      # 亮度
        contrast=0.2,                        # 对比度
        saturation=0.2,                      # 饱和度
        hue=0.1                              # 色调
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# 验证/测试时的变换（通常更简单）
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

print("图像变换定义完成")
```

### 自定义图像变换
```python
print("\n=== 自定义图像变换 ===")

class RandomErasing(object):
    """随机擦除数据增强"""
    
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
    
    def __call__(self, img):
        if torch.rand(1) > self.probability:
            return img
        
        # 实现随机擦除逻辑
        # ...
        return img

class GaussianNoise(object):
    """高斯噪声数据增强"""
    
    def __init__(self, mean=0., std=0.1):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

# 使用自定义变换
custom_transform = transforms.Compose([
    transforms.ToTensor(),
    GaussianNoise(std=0.05),
    RandomErasing(probability=0.3)
])

print("自定义变换定义完成")
```

## 文本数据处理

### 文本Dataset实现
```python
print("\n=== 文本数据处理 ===")

class TextDataset(Dataset):
    """文本数据集类"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # 分词和编码
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 使用Hugging Face Transformers的tokenizer
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 模拟文本数据
texts = ["This is a positive review", "This is a negative review"]
labels = [1, 0]

text_dataset = TextDataset(texts, labels, tokenizer)
print(f"文本数据集大小: {len(text_dataset)}")
print(f"第一个样本的input_ids形状: {text_dataset[0]['input_ids'].shape}")
```

### 处理变长序列
```python
print("\n=== 变长序列处理 ===")

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class VariableLengthDataset(Dataset):
    """处理变长序列的数据集"""
    
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
        self.lengths = [len(seq) for seq in sequences]
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]), torch.tensor(self.labels[idx])

# 创建变长序列数据
sequences = [
    [1, 2, 3],           # 长度3
    [4, 5, 6, 7, 8],     # 长度5
    [9, 10, 11, 12],     # 长度4
    [13, 14]             # 长度2
]
labels = [0, 1, 0, 1]

var_dataset = VariableLengthDataset(sequences, labels)

# 自定义collate函数处理变长序列
def collate_fn(batch):
    """处理变长序列的collate函数"""
    sequences, labels = zip(*batch)
    
    # 获取序列长度
    lengths = torch.tensor([len(seq) for seq in sequences])
    
    # 填充序列
    sequences_padded = pad_sequence(sequences, batch_first=True)
    
    # 排序（按长度降序）
    lengths, sort_idx = lengths.sort(dim=0, descending=True)
    sequences_padded = sequences_padded[sort_idx]
    labels = torch.stack(labels)[sort_idx]
    
    return sequences_padded, labels, lengths

# 使用自定义collate函数的DataLoader
var_dataloader = DataLoader(
    var_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0
)

for batch in var_dataloader:
    sequences, labels, lengths = batch
    print(f"填充后序列形状: {sequences.shape}")
    print(f"序列长度: {lengths}")
    print(f"标签: {labels}")
    break  # 只显示第一个批次
```

## 高级数据加载技术

### 数据采样策略
```python
print("\n=== 数据采样策略 ===")

from torch.utils.data import WeightedRandomSampler, RandomSampler, SequentialSampler

# 模拟不平衡数据集
labels = [0] * 90 + [1] * 10  # 90个类别0，10个类别1

# 加权随机采样（处理类别不平衡）
class_counts = torch.bincount(torch.tensor(labels))
class_weights = 1. / class_counts
sample_weights = class_weights[labels]

weighted_sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(labels),
    replacement=True  # 允许重复采样
)

# 随机采样
random_sampler = RandomSampler(dataset, replacement=False)

# 顺序采样
sequential_sampler = SequentialSampler(dataset)

print("采样器定义完成")

# 使用采样器的DataLoader
weighted_dataloader = DataLoader(
    dataset,
    batch_size=32,
    sampler=weighted_sampler,
    num_workers=4
)

print(f"加权采样DataLoader批次数量: {len(weighted_dataloader)}")
```

### 分布式数据加载
```python
print("\n=== 分布式数据加载 ===")

from torch.utils.data.distributed import DistributedSampler

# 分布式采样器（用于多GPU训练）
distributed_sampler = DistributedSampler(
    dataset,
    num_replicas=2,  # GPU数量
    rank=0,          # 当前GPU的排名
    shuffle=True
)

distributed_dataloader = DataLoader(
    dataset,
    batch_size=32,
    sampler=distributed_sampler,
    num_workers=4
)

print("分布式数据加载器定义完成")
```

## 数据预处理与特征工程

### 数值数据预处理
```python
print("\n=== 数值数据预处理 ===")

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd

class TabularDataset(Dataset):
    """表格数据集"""
    
    def __init__(self, dataframe, target_column, transform=None):
        self.features = dataframe.drop(columns=[target_column]).values
        self.labels = dataframe[target_column].values
        self.transform = transform
        
        # 数据标准化
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        features = torch.FloatTensor(self.features[idx])
        label = torch.LongTensor([self.labels[idx]])
        
        if self.transform:
            features = self.transform(features)
        
        return features, label

# 创建模拟表格数据
data = {
    'feature1': np.random.randn(100),
    'feature2': np.random.randn(100),
    'feature3': np.random.randn(100),
    'target': np.random.randint(0, 2, 100)
}
df = pd.DataFrame(data)

tabular_dataset = TabularDataset(df, 'target')
print(f"表格数据集大小: {len(tabular_dataset)}")
print(f"特征形状: {tabular_dataset[0][0].shape}")
```

### 数据缓存与预加载
```python
print("\n=== 数据缓存与预加载 ===")

class CachedDataset(Dataset):
    """支持缓存的数据集"""
    
    def __init__(self, base_dataset, cache_size=1000):
        self.base_dataset = base_dataset
        self.cache = {}
        self.cache_size = cache_size
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        
        # 从基础数据集加载
        sample = self.base_dataset[idx]
        
        # 缓存样本
        if len(self.cache) < self.cache_size:
            self.cache[idx] = sample
        
        return sample

# 使用缓存数据集
cached_dataset = CachedDataset(dataset, cache_size=50)
print(f"缓存数据集大小: {len(cached_dataset)}")
```

## 数据可视化与调试

### 数据可视化工具
```python
print("\n=== 数据可视化 ===")

def visualize_batch(images, labels, class_names=None, num_samples=8):
    """可视化一个批次的数据"""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(min(num_samples, len(images))):
        # 反标准化（如果是ImageNet标准化）
        img = images[i].numpy().transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        if class_names:
            axes[i].set_title(f'Label: {class_names[labels[i]]}')
        else:
            axes[i].set_title(f'Label: {labels[i]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# 模拟类别名称
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

print("数据可视化函数定义完成")
```

### 数据质量检查
```python
print("\n=== 数据质量检查 ===")

def check_data_quality(dataloader):
    """检查数据质量"""
    print("数据质量检查:")
    
    for batch_idx, (data, target) in enumerate(dataloader):
        # 检查NaN值
        if torch.isnan(data).any():
            print(f"批次 {batch_idx}: 发现NaN值")
        
        # 检查无限值
        if torch.isinf(data).any():
            print(f"批次 {batch_idx}: 发现无限值")
        
        # 检查数据范围
        data_min, data_max = data.min(), data.max()
        if data_min < -10 or data_max > 10:
            print(f"批次 {batch_idx}: 数据范围异常 [{data_min:.3f}, {data_max:.3f}]")
        
        # 检查标签范围
        if target.min() < 0 or target.max() >= 10:  # 假设有10个类别
            print(f"批次 {batch_idx}: 标签范围异常")
        
        if batch_idx >= 2:  # 只检查前3个批次
            break
    
    print("数据质量检查完成")

# 执行数据质量检查
check_data_quality(dataloader)
```

## 实际应用示例

### CIFAR-10数据加载
```python
print("\n=== CIFAR-10数据加载示例 ===")

import torchvision.datasets as datasets

def get_cifar10_dataloaders(batch_size=128, num_workers=4):
    """获取CIFAR-10数据加载器"""
    
    # 数据变换
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # 加载数据集
    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    
    return train_loader, test_loader

# 使用示例
train_loader, test_loader = get_cifar10_dataloaders()
print(f"训练数据批次: {len(train_loader)}")
print(f"测试数据批次: {len(test_loader)}")

# 查看一个批次的数据
for images, labels in train_loader:
    print(f"图像形状: {images.shape}")
    print(f"标签形状: {labels.shape}")
    break
```

### 自定义图像数据集
```python
print("\n=== 自定义图像数据集示例 ===")

import os
from PIL import Image

class CustomImageDataset(Dataset):
    """自定义图像数据集"""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # 获取所有图像文件路径
        self.image_paths = []
        self.labels = []
        
        for class_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(os.path.join(class_dir, img_name))
                        self.labels.append(class_name)
        
        # 创建标签映射
        self.classes = sorted(set(self.labels))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.labels = [self.class_to_idx[label] for label in self.labels]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 加载图像
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

print("自定义图像数据集类定义完成")
```

## 性能优化技巧

### 数据加载性能优化
```python
print("\n=== 数据加载性能优化 ===")

# 1. 使用合适数量的workers
optimal_workers = min(4, os.cpu_count())  # 通常4个workers效果较好
print(f"推荐workers数量: {optimal_workers}")

# 2. 启用pin_memory（GPU训练时）
pin_memory = torch.cuda.is_available()
print(f"是否启用pin_memory: {pin_memory}")

# 3. 使用prefetch（PyTorch 1.7+）
# 在DataLoader中设置prefetch_factor=2

# 4. 数据预处理优化
class OptimizedDataset(Dataset):
    """优化数据加载性能的数据集"""
    
    def __init__(self, data, labels, transform=None):
        # 预加载数据到内存（如果内存允许）
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample, label

print("性能优化技巧介绍完成")
```

## 总结

本章详细介绍了PyTorch数据加载与处理的各个方面：

1. **Dataset与DataLoader**：基础数据加载框架
2. **图像数据处理**：变换、增强和自定义变换
3. **文本数据处理**：分词、编码和变长序列处理
4. **高级数据加载**：采样策略和分布式加载
5. **数据预处理**：数值数据处理和特征工程
6. **可视化与调试**：数据质量检查和可视化
7. **实际应用**：标准数据集和自定义数据集
8. **性能优化**：数据加载性能优化技巧

掌握数据加载与处理是深度学习项目成功的关键，建议在实际项目中练习这些技术。