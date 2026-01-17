# PyTorch实战项目示例

## 章节概述

本章将通过完整的实战项目展示PyTorch在实际应用中的使用。我们将构建图像分类、文本生成、目标检测等多个项目，涵盖从数据准备到模型部署的完整流程。

## 项目1：图像分类 - CIFAR-10分类器

### 项目概述
使用CIFAR-10数据集构建一个图像分类器，包含数据增强、模型训练、评估和可视化。

### 完整代码实现
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

print("=== CIFAR-10图像分类项目 ===")

class CIFAR10Classifier(nn.Module):
    """CIFAR-10图像分类模型"""
    
    def __init__(self, num_classes=10):
        super(CIFAR10Classifier, self).__init__()
        
        # 特征提取层
        self.features = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            
            # 第二个卷积块
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            
            # 第三个卷积块
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.classifier(x)
        return x

class CIFAR10Trainer:
    """CIFAR-10训练器"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 数据变换
        self.train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        
        # 加载数据集
        self.train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=self.train_transform
        )
        self.test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=self.test_transform
        )
        
        # 创建数据加载器
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=128, shuffle=True, num_workers=4
        )
        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset, batch_size=100, shuffle=False, num_workers=4
        )
        
        # 模型和优化器
        self.model = CIFAR10Classifier().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=200)
        
        # 训练历史
        self.train_losses = []
        self.test_losses = []
        self.train_accs = []
        self.test_accs = []
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch} | Batch: {batch_idx}/{len(self.train_loader)} | '
                      f'Loss: {loss.item():.3f}')
        
        train_loss = running_loss / len(self.train_loader)
        train_acc = 100. * correct / total
        
        return train_loss, train_acc
    
    def test(self):
        """测试模型"""
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        test_loss /= len(self.test_loader)
        test_acc = 100. * correct / total
        
        return test_loss, test_acc
    
    def train(self, epochs=100):
        """完整训练过程"""
        best_acc = 0
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 测试
            test_loss, test_acc = self.test()
            
            # 学习率调度
            self.scheduler.step()
            
            # 记录历史
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            self.train_accs.append(train_acc)
            self.test_accs.append(test_acc)
            
            print(f'训练损失: {train_loss:.3f} | 训练准确率: {train_acc:.2f}%')
            print(f'测试损失: {test_loss:.3f} | 测试准确率: {test_acc:.2f}%')
            
            # 保存最佳模型
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(self.model.state_dict(), 'best_cifar10_model.pth')
                print(f'新的最佳准确率: {best_acc:.2f}%, 模型已保存')
        
        print(f'\n训练完成! 最佳测试准确率: {best_acc:.2f}%')
        return self.train_losses, self.test_losses, self.train_accs, self.test_accs
    
    def visualize_results(self):
        """可视化训练结果"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 损失曲线
        ax1.plot(self.train_losses, label='训练损失')
        ax1.plot(self.test_losses, label='测试损失')
        ax1.set_title('损失曲线')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(self.train_accs, label='训练准确率')
        ax2.plot(self.test_accs, label='测试准确率')
        ax2.set_title('准确率曲线')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_predictions(self, num_samples=10):
        """可视化预测结果"""
        self.model.eval()
        
        # CIFAR-10类别名称
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                  'dog', 'frog', 'horse', 'ship', 'truck']
        
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        axes = axes.ravel()
        
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                if i >= 1:  # 只取第一个批次
                    break
                
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                
                for j in range(min(num_samples, len(images))):
                    # 反标准化
                    img = images[j].cpu().numpy().transpose(1, 2, 0)
                    mean = np.array([0.4914, 0.4822, 0.4465])
                    std = np.array([0.2023, 0.1994, 0.2010])
                    img = std * img + mean
                    img = np.clip(img, 0, 1)
                    
                    axes[j].imshow(img)
                    axes[j].set_title(f'真实: {classes[labels[j]]}\n预测: {classes[predicted[j]]}')
                    axes[j].axis('off')
                    
                    # 标记错误预测
                    if labels[j] != predicted[j]:
                        axes[j].title.set_color('red')
        
        plt.tight_layout()
        plt.show()

# 运行训练
# trainer = CIFAR10Trainer()
# trainer.train(epochs=50)
# trainer.visualize_results()
# trainer.visualize_predictions()

print("CIFAR-10分类器定义完成")
```

## 项目2：文本生成 - LSTM文本生成器

### 项目概述
使用LSTM网络构建一个文本生成器，能够根据给定的文本序列生成新的文本。

### 完整代码实现
```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import re

print("\n=== LSTM文本生成项目 ===")

class TextGenerator:
    """文本生成器"""
    
    def __init__(self, text, sequence_length=50):
        self.text = text
        self.sequence_length = sequence_length
        
        # 预处理文本
        self._preprocess_text()
        self._create_dataset()
        
        # 模型参数
        self.vocab_size = len(self.chars)
        self.embedding_dim = 128
        self.hidden_dim = 256
        self.num_layers = 2
        
        # 创建模型
        self.model = TextGenerationModel(
            self.vocab_size, self.embedding_dim, self.hidden_dim, 
            self.num_layers, self.vocab_size
        )
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.002)
    
    def _preprocess_text(self):
        """预处理文本"""
        # 清理文本
        text = re.sub(r'[^\w\s]', '', self.text.lower())
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 创建字符映射
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
        # 将文本转换为索引
        self.text_indices = [self.char_to_idx[ch] for ch in text]
        
        print(f"文本长度: {len(text)}")
        print(f"唯一字符数: {len(self.chars)}")
        print(f"字符集: {''.join(self.chars)}")
    
    def _create_dataset(self):
        """创建训练数据集"""
        sequences = []
        next_chars = []
        
        for i in range(0, len(self.text_indices) - self.sequence_length):
            sequences.append(self.text_indices[i:i + self.sequence_length])
            next_chars.append(self.text_indices[i + self.sequence_length])
        
        self.X = torch.tensor(sequences, dtype=torch.long)
        self.y = torch.tensor(next_chars, dtype=torch.long)
        
        print(f"训练样本数: {len(sequences)}")
        print(f"输入形状: {self.X.shape}")
        print(f"目标形状: {self.y.shape}")
    
    def train(self, epochs=100, batch_size=128):
        """训练模型"""
        self.model.train()
        
        dataset = torch.utils.data.TensorDataset(self.X, self.y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            total_loss = 0
            
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                
                # 初始化隐藏状态
                hidden = self.model.init_hidden(batch_X.size(0))
                
                # 前向传播
                output, hidden = self.model(batch_X, hidden)
                
                # 计算损失
                loss = self.criterion(output, batch_y)
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                
                self.optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')
                
                # 每10个epoch生成一个样本
                generated_text = self.generate_text(seed_text="the ", length=100)
                print(f'生成文本: {generated_text}')
    
    def generate_text(self, seed_text, length=100, temperature=0.8):
        """生成文本"""
        self.model.eval()
        
        # 将种子文本转换为索引
        seed_indices = [self.char_to_idx.get(ch, 0) for ch in seed_text]
        
        generated = seed_text
        hidden = self.model.init_hidden(1)
        
        with torch.no_grad():
            for _ in range(length):
                # 准备输入
                input_tensor = torch.tensor([seed_indices], dtype=torch.long)
                
                # 前向传播
                output, hidden = self.model(input_tensor, hidden)
                
                # 应用温度采样
                output = output / temperature
                probabilities = torch.softmax(output[-1], dim=0)
                
                # 采样下一个字符
                next_char_idx = torch.multinomial(probabilities, 1).item()
                next_char = self.idx_to_char[next_char_idx]
                
                # 添加到生成文本
                generated += next_char
                
                # 更新输入序列
                seed_indices = seed_indices[1:] + [next_char_idx]
        
        return generated

class TextGenerationModel(nn.Module):
    """文本生成模型"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, output_dim):
        super(TextGenerationModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x, hidden):
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        
        # 只取最后一个时间步的输出
        out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        """初始化隐藏状态"""
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, batch_size, self.hidden_dim),
                weight.new_zeros(self.num_layers, batch_size, self.hidden_dim))

# 示例文本（莎士比亚作品片段）
sample_text = """
To be or not to be that is the question
Whether tis nobler in the mind to suffer
The slings and arrows of outrageous fortune
Or to take arms against a sea of troubles
And by opposing end them
"""

# 创建和训练文本生成器
# text_generator = TextGenerator(sample_text)
# text_generator.train(epochs=100)

# 生成文本
# generated = text_generator.generate_text("to be ", length=200)
# print(f"生成的文本: {generated}")

print("文本生成器定义完成")
```

## 项目3：目标检测 - 简易YOLO实现

### 项目概述
实现一个简化的YOLO（You Only Look Once）目标检测器，能够检测图像中的物体。

### 完整代码实现
```python
import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet18
import matplotlib.pyplot as plt
import matplotlib.patches as patches

print("\n=== 简易YOLO目标检测项目 ===")

class SimpleYOLO(nn.Module):
    """简易YOLO目标检测器"""
    
    def __init__(self, num_classes=20, grid_size=7, num_boxes=2):
        super(SimpleYOLO, self).__init__()
        
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        
        # 使用预训练的ResNet作为特征提取器
        backbone = resnet18(pretrained=True)
        
        # 移除最后的全连接层
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        
        # YOLO检测头
        self.detection_head = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1),
            nn.Conv2d(1024, (5 * num_boxes + num_classes) * grid_size * grid_size, 
                     kernel_size=1)
        )
    
    def forward(self, x):
        # 特征提取
        features = self.features(x)
        
        # 检测头
        output = self.detection_head(features)
        
        # 重塑输出为 (batch, grid, grid, 5*B + C)
        batch_size = output.size(0)
        output = output.view(batch_size, self.grid_size, self.grid_size, -1)
        
        return output

class YOLOLoss(nn.Module):
    """YOLO损失函数"""
    
    def __init__(self, grid_size=7, num_boxes=2, num_classes=20, lambda_coord=5, lambda_noobj=0.5):
        super(YOLOLoss, self).__init__()
        self.grid_size = grid_size
        self.num_boxes = num_boxes
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.mse_loss = nn.MSELoss(reduction='sum')
    
    def forward(self, predictions, targets):
        """计算YOLO损失"""
        batch_size = predictions.size(0)
        
        # 解析预测
        pred_boxes = predictions[..., :5*self.num_boxes].contiguous().view(
            batch_size, self.grid_size, self.grid_size, self.num_boxes, 5
        )
        pred_class = predictions[..., 5*self.num_boxes:]
        
        # 解析目标
        target_boxes = targets[..., :5].unsqueeze(3)  # 假设每个网格只有一个目标
        target_class = targets[..., 5:]
        
        # 计算损失...
        # 这里简化实现，实际YOLO损失更复杂
        
        return self.mse_loss(predictions, targets)  # 简化版本

class ObjectDetector:
    """目标检测器"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleYOLO(num_classes=20).to(self.device)
        self.criterion = YOLOLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
    
    def train(self, dataloader, epochs=50):
        """训练目标检测器"""
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            
            for batch_idx, (images, targets) in enumerate(dataloader):
                images = images.to(self.device)
                targets = targets.to(self.device)
                
                self.optimizer.zero_grad()
                predictions = self.model(images)
                loss = self.criterion(predictions, targets)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 50 == 0:
                    print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.3f}')
            
            avg_loss = total_loss / len(dataloader)
            print(f'Epoch {epoch} 完成, 平均损失: {avg_loss:.3f}')
    
    def detect_objects(self, image):
        """检测图像中的物体"""
        self.model.eval()
        
        with torch.no_grad():
            # 预处理图像
            image_tensor = self._preprocess_image(image).to(self.device)
            
            # 预测
            predictions = self.model(image_tensor.unsqueeze(0))
            
            # 后处理
            boxes, scores, classes = self._postprocess_predictions(predictions[0])
            
            return boxes, scores, classes
    
    def _preprocess_image(self, image):
        """预处理图像"""
        # 调整大小、标准化等
        transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        return transform(image)
    
    def _postprocess_predictions(self, predictions):
        """后处理预测结果"""
        # 简化实现，实际需要非极大值抑制等
        boxes = []
        scores = []
        classes = []
        
        # 解析预测...
        
        return boxes, scores, classes
    
    def visualize_detection(self, image, boxes, scores, classes, class_names):
        """可视化检测结果"""
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(image)
        
        for box, score, cls in zip(boxes, scores, classes):
            if score > 0.5:  # 置信度阈值
                x, y, w, h = box
                
                # 创建边界框
                rect = patches.Rectangle((x, y), w, h, 
                                       linewidth=2, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
                
                # 添加标签
                label = f'{class_names[cls]}: {score:.2f}'
                ax.text(x, y-10, label, color='red', fontsize=12, 
                       bbox=dict(facecolor='white', alpha=0.7))
        
        plt.axis('off')
        plt.show()

# 使用示例
# detector = ObjectDetector()
# 假设有训练数据加载器
# detector.train(dataloader)

print("目标检测器定义完成")
```

## 项目4：生成对抗网络 - DCGAN图像生成

### 项目概述
实现深度卷积生成对抗网络（DCGAN），用于生成逼真的图像。

### 完整代码实现
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

print("\n=== DCGAN图像生成项目 ===")

class Generator(nn.Module):
    """生成器"""
    
    def __init__(self, latent_dim, img_channels=3, feature_map_size=64):
        super(Generator, self).__init__()
        
        self.latent_dim = latent_dim
        
        self.main = nn.Sequential(
            # 输入: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, feature_map_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_map_size * 8),
            nn.ReLU(True),
            # 输出: (feature_map_size*8) x 4 x 4
            
            nn.ConvTranspose2d(feature_map_size * 8, feature_map_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.ReLU(True),
            # 输出: (feature_map_size*4) x 8 x 8
            
            nn.ConvTranspose2d(feature_map_size * 4, feature_map_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.ReLU(True),
            # 输出: (feature_map_size*2) x 16 x 16
            
            nn.ConvTranspose2d(feature_map_size * 2, feature_map_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size),
            nn.ReLU(True),
            # 输出: feature_map_size x 32 x 32
            
            nn.ConvTranspose2d(feature_map_size, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # 输出: img_channels x 64 x 64
        )
    
    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    """判别器"""
    
    def __init__(self, img_channels=3, feature_map_size=64):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            # 输入: img_channels x 64 x 64
            nn.Conv2d(img_channels, feature_map_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出: feature_map_size x 32 x 32
            
            nn.Conv2d(feature_map_size, feature_map_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出: (feature_map_size*2) x 16 x 16
            
            nn.Conv2d(feature_map_size * 2, feature_map_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出: (feature_map_size*4) x 8 x 8
            
            nn.Conv2d(feature_map_size * 4, feature_map_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 输出: (feature_map_size*8) x 4 x 4
            
            nn.Conv2d(feature_map_size * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # 输出: 1 x 1 x 1
        )
    
    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

class DCGANTrainer:
    """DCGAN训练器"""
    
    def __init__(self, latent_dim=100, img_size=64):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.latent_dim = latent_dim
        self.img_size = img_size
        
        # 创建生成器和判别器
        self.generator = Generator(latent_dim).to(self.device)
        self.discriminator = Discriminator().to(self.device)
        
        # 损失函数和优化器
        self.criterion = nn.BCELoss()
        
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        # 固定噪声用于可视化
        self.fixed_noise = torch.randn(64, latent_dim, 1, 1, device=self.device)
        
        # 训练历史
        self.g_losses = []
        self.d_losses = []
    
    def train_epoch(self, dataloader, epoch):
        """训练一个epoch"""
        for i, (real_imgs, _) in enumerate(dataloader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(self.device)
            
            # 真实和假标签
            real_labels = torch.ones(batch_size, device=self.device)
            fake_labels = torch.zeros(batch_size, device=self.device)
            
            # ---------------------
            #  训练判别器
            # ---------------------
            self.discriminator.zero_grad()
            
            # 真实图像的损失
            output_real = self.discriminator(real_imgs)
            loss_real = self.criterion(output_real, real_labels)
            
            # 假图像的损失
            noise = torch.randn(batch_size, self.latent_dim, 1, 1, device=self.device)
            fake_imgs = self.generator(noise)
            output_fake = self.discriminator(fake_imgs.detach())
            loss_fake = self.criterion(output_fake, fake_labels)
            
            # 判别器总损失
            loss_D = loss_real + loss_fake
            loss_D.backward()
            self.optimizer_D.step()
            
            # ---------------------
            #  训练生成器
            # ---------------------
            self.generator.zero_grad()
            
            output_fake = self.discriminator(fake_imgs)
            loss_G = self.criterion(output_fake, real_labels)
            loss_G.backward()
            self.optimizer_G.step()
            
            # 记录损失
            self.g_losses.append(loss_G.item())
            self.d_losses.append(loss_D.item())
            
            if i % 100 == 0:
                print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] '
                      f'Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}')
    
    def train(self, dataloader, num_epochs=50):
        """完整训练过程"""
        for epoch in range(num_epochs):
            self.train_epoch(dataloader, epoch)
            
            # 每个epoch结束时生成样本
            if epoch % 5 == 0:
                self.generate_samples(epoch)
    
    def generate_samples(self, epoch, save_path=None):
        """生成样本图像"""
        self.generator.eval()
        
        with torch.no_grad():
            fake_images = self.generator(self.fixed_noise)
            fake_images = fake_images.detach().cpu()
        
        # 可视化生成的图像
        fig, axes = plt.subplots(8, 8, figsize=(12, 12))
        for i, ax in enumerate(axes.flat):
            img = fake_images[i].permute(1, 2, 0).numpy()
            img = (img + 1) / 2  # 从[-1,1]转换到[0,1]
            ax.imshow(img)
            ax.axis('off')
        
        plt.suptitle(f'Epoch {epoch}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f'{save_path}/epoch_{epoch}.png')
        
        plt.show()
        
        self.generator.train()

# 使用示例
# dcgan = DCGANTrainer()
# dcgan.train(dataloader)

print("DCGAN定义完成")
```

## 项目总结与扩展

### 项目共同特点
1. **完整的流程**：从数据准备到模型部署
2. **模块化设计**：易于理解和扩展
3. **最佳实践**：包含性能优化和调试技巧
4. **可视化支持**：训练过程和结果可视化

### 扩展建议
1. **添加更多的数据增强**
2. **实现更复杂的模型架构**
3. **添加模型解释性分析**
4. **实现自动化超参数调优**
5. **添加模型部署和API服务**

### 学习路径建议
1. 从简单的图像分类开始
2. 逐步尝试文本生成和目标检测
3. 最后挑战生成对抗网络
4. 每个项目都要理解背后的原理

这些实战项目涵盖了深度学习的多个重要领域，通过实践可以加深对PyTorch和深度学习的理解。