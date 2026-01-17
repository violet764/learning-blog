# PyTorch高级特性详解

## 章节概述

PyTorch提供了许多高级特性，包括分布式训练、模型部署、性能优化等。本章将深入讲解这些高级功能，帮助您构建生产级的深度学习应用。

## 分布式训练

### 数据并行训练
```python
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

print("=== 数据并行训练 ===")

# 基础数据并行
model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 10)
)

if torch.cuda.device_count() > 1:
    print(f"使用 {torch.cuda.device_count()} 个GPU进行数据并行训练")
    model = nn.DataParallel(model)

model = model.cuda()

# 数据并行训练示例
def train_data_parallel(model, dataloader, criterion, optimizer):
    model.train()
    
    for data, target in dataloader:
        data, target = data.cuda(), target.cuda()
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

print("数据并行训练定义完成")
```

### 分布式数据并行（DDP）
```python
print("\n=== 分布式数据并行（DDP） ===")

def setup_ddp(rank, world_size):
    """设置分布式环境"""
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """清理分布式环境"""
    dist.destroy_process_group()

class DDPTrainer:
    """分布式数据并行训练器"""
    
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        setup_ddp(rank, world_size)
        
        # 模型定义（每个进程有独立的模型副本）
        self.model = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        ).cuda(rank)
        
        # 包装为DDP模型
        self.model = nn.parallel.DistributedDataParallel(
            self.model, device_ids=[rank]
        )
        
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(self.rank), target.cuda(self.rank)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            if batch_idx % 100 == 0 and self.rank == 0:
                print(f'Rank {self.rank}, Batch {batch_idx}, Loss: {loss.item():.3f}')
    
    def cleanup(self):
        """清理资源"""
        cleanup_ddp()

def run_ddp_training(rank, world_size):
    """运行DDP训练"""
    trainer = DDPTrainer(rank, world_size)
    
    # 这里需要实际的训练数据加载器
    # train_loader = get_ddp_dataloader(rank, world_size)
    
    # 训练多个epoch
    for epoch in range(10):
        # trainer.train_epoch(train_loader)
        if rank == 0:
            print(f'Epoch {epoch} completed')
    
    trainer.cleanup()

# 启动DDP训练（在实际环境中使用）
# if __name__ == "__main__":
#     world_size = torch.cuda.device_count()
#     mp.spawn(run_ddp_training, args=(world_size,), nprocs=world_size)

print("DDP训练器定义完成")
```

### 模型并行训练
```python
print("\n=== 模型并行训练 ===")

class ModelParallelNN(nn.Module):
    """模型并行神经网络"""
    
    def __init__(self, device0, device1):
        super(ModelParallelNN, self).__init__()
        self.device0 = device0
        self.device1 = device1
        
        # 第一部分在device0上
        self.part1 = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 50)
        ).to(device0)
        
        # 第二部分在device1上
        self.part2 = nn.Sequential(
            nn.Linear(50, 30),
            nn.ReLU(),
            nn.Linear(30, 10)
        ).to(device1)
    
    def forward(self, x):
        # 在第一部分设备上处理
        x = x.to(self.device0)
        x = self.part1(x)
        
        # 转移到第二部分设备
        x = x.to(self.device1)
        x = self.part2(x)
        
        return x

# 使用示例
if torch.cuda.device_count() >= 2:
    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")
    
    model = ModelParallelNN(device0, device1)
    
    # 模拟输入数据
    input_data = torch.randn(32, 10)
    output = model(input_data)
    
    print(f"模型并行输出形状: {output.shape}")
    print(f"输出设备: {output.device}")

print("模型并行训练定义完成")
```

## 模型部署与优化

### ONNX模型导出
```python
print("\n=== ONNX模型导出 ===")

def export_to_onnx(model, dummy_input, onnx_path, input_names=None, output_names=None):
    """导出模型为ONNX格式"""
    
    model.eval()
    
    # 默认输入输出名称
    if input_names is None:
        input_names = ['input']
    if output_names is None:
        output_names = ['output']
    
    # 导出模型
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"模型已导出到: {onnx_path}")
    return onnx_path

# 使用示例
simple_model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 10)
)

dummy_input = torch.randn(1, 10)
onnx_path = export_to_onnx(simple_model, dummy_input, "model.onnx")

print("ONNX导出函数定义完成")
```

### ONNX Runtime推理
```python
print("\n=== ONNX Runtime推理 ===")

try:
    import onnxruntime as ort
    
    class ONNXInference:
        """ONNX模型推理"""
        
        def __init__(self, onnx_path, providers=None):
            if providers is None:
                providers = ['CPUExecutionProvider']
                if ort.get_device() == 'GPU':
                    providers.insert(0, 'CUDAExecutionProvider')
            
            # 创建推理会话
            self.session = ort.InferenceSession(onnx_path, providers=providers)
            
            # 获取输入输出信息
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
        
        def predict(self, input_data):
            """执行推理"""
            # 确保输入格式正确
            if isinstance(input_data, torch.Tensor):
                input_data = input_data.detach().cpu().numpy()
            
            # 执行推理
            outputs = self.session.run([self.output_name], 
                                     {self.input_name: input_data})
            return torch.from_numpy(outputs[0])
    
    # 使用示例
    onnx_inference = ONNXInference("model.onnx")
    test_input = torch.randn(5, 10)
    prediction = onnx_inference.predict(test_input)
    
    print(f"ONNX推理输出形状: {prediction.shape}")
    
except ImportError:
    print("ONNX Runtime未安装，跳过ONNX推理示例")

print("ONNX推理类定义完成")
```

### 模型量化
```python
print("\n=== 模型量化 ===")

class ModelQuantizer:
    """模型量化工具"""
    
    def __init__(self, model):
        self.model = model
    
    def dynamic_quantization(self):
        """动态量化（推理时量化）"""
        # 量化模型
        quantized_model = torch.quantization.quantize_dynamic(
            self.model, 
            {nn.Linear, nn.LSTM},  # 需要量化的层类型
            dtype=torch.qint8
        )
        return quantized_model
    
    def static_quantization(self, calibration_data):
        """静态量化（需要校准数据）"""
        # 设置模型为评估模式
        self.model.eval()
        
        # 融合操作（如Conv+ReLU）
        self.model.fuse_model()
        
        # 量化配置
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # 准备量化
        torch.quantization.prepare(self.model, inplace=True)
        
        # 校准（使用校准数据）
        with torch.no_grad():
            for data in calibration_data:
                self.model(data)
        
        # 转换为量化模型
        quantized_model = torch.quantization.convert(self.model, inplace=False)
        
        return quantized_model
    
    def print_quantization_info(self, quantized_model):
        """打印量化信息"""
        print("量化模型信息:")
        for name, module in quantized_model.named_modules():
            if isinstance(module, (nn.quantized.Linear, nn.quantized.Conv2d)):
                print(f"{name}: 量化权重形状 {module.weight().shape}")

# 使用示例
model_to_quantize = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 10)
)

quantizer = ModelQuantizer(model_to_quantize)
dynamic_quantized = quantizer.dynamic_quantization()

print("动态量化完成")
# quantizer.print_quantization_info(dynamic_quantized)
```

## 性能优化技巧

### 内存优化
```python
print("\n=== 内存优化 ===")

class MemoryOptimizer:
    """内存优化工具"""
    
    @staticmethod
    def enable_gradient_checkpointing(model):
        """启用梯度检查点（用时间换空间）"""
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            print("梯度检查点已启用")
        else:
            print("该模型不支持梯度检查点")
    
    @staticmethod
    def estimate_memory_usage(model, input_shape, dtype=torch.float32):
        """估算模型内存使用"""
        # 参数内存
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
        
        # 梯度内存（训练时）
        grad_memory = param_memory
        
        # 激活内存（估算）
        dummy_input = torch.randn(*input_shape, dtype=dtype)
        
        # 前向传播获取激活大小
        try:
            with torch.no_grad():
                output = model(dummy_input)
                # 简化估算：输入输出内存
                activation_memory = (dummy_input.numel() + output.numel()) * dtype.itemsize
        except:
            activation_memory = 0
        
        total_memory = param_memory + grad_memory + activation_memory
        
        print(f"参数内存: {param_memory / 1024**2:.2f} MB")
        print(f"梯度内存: {grad_memory / 1024**2:.2f} MB")
        print(f"激活内存: {activation_memory / 1024**2:.2f} MB")
        print(f"总内存估算: {total_memory / 1024**2:.2f} MB")
        
        return total_memory
    
    @staticmethod
    def optimize_model_memory(model):
        """优化模型内存使用"""
        # 1. 将模型设置为评估模式（减少缓存）
        model.eval()
        
        # 2. 清空梯度
        for param in model.parameters():
            if param.grad is not None:
                param.grad = None
        
        # 3. 释放GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("内存优化完成")

# 使用示例
memory_optimizer = MemoryOptimizer()
# memory_optimizer.estimate_memory_usage(model_to_quantize, (1, 10))

print("内存优化工具定义完成")
```

### 计算优化
```python
print("\n=== 计算优化 ===")

class ComputationOptimizer:
    """计算优化工具"""
    
    @staticmethod
    def enable_torchscript(model, example_input):
        """启用TorchScript优化"""
        try:
            scripted_model = torch.jit.trace(model, example_input)
            print("TorchScript优化已启用")
            return scripted_model
        except Exception as e:
            print(f"TorchScript优化失败: {e}")
            return model
    
    @staticmethod
    def benchmark_model(model, input_data, num_runs=100):
        """基准测试模型性能"""
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_data)
        
        # 基准测试
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time.record()
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(input_data)
        
        end_time.record()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed_time = start_time.elapsed_time(end_time) / 1000.0  # 转换为秒
        fps = num_runs / elapsed_time
        
        print(f"总时间: {elapsed_time:.3f} 秒")
        print(f"FPS: {fps:.1f}")
        print(f"单次推理时间: {elapsed_time/num_runs*1000:.1f} 毫秒")
        
        return fps
    
    @staticmethod
    def optimize_for_inference(model):
        """为推理优化模型"""
        # 1. 设置为评估模式
        model.eval()
        
        # 2. 启用推理模式（PyTorch 1.9+）
        if hasattr(torch, 'inference_mode'):
            model = torch.inference_mode()(model)
        
        # 3. 禁用梯度计算
        with torch.no_grad():
            # 这里可以进行更多的优化
            pass
        
        print("推理优化完成")
        return model

# 使用示例
comp_optimizer = ComputationOptimizer()
print("计算优化工具定义完成")
```

## 高级调试与监控

### 模型分析工具
```python
print("\n=== 模型分析工具 ===")

def analyze_model_complexity(model, input_size):
    """分析模型复杂度"""
    print("=== 模型复杂度分析 ===")
    
    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"总参数数: {total_params:,}")
    print(f"可训练参数数: {trainable_params:,}")
    print(f"不可训练参数数: {total_params - trainable_params:,}")
    
    # 逐层分析
    print("\n逐层分析:")
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 叶子模块
            num_params = sum(p.numel() for p in module.parameters())
            if num_params > 0:
                print(f"{name}: {num_params:,} 参数")

# 使用示例
analyze_model_complexity(model_to_quantize, (1, 10))

print("模型分析工具定义完成")
```

### 训练监控
```python
print("\n=== 训练监控 ===")

class TrainingMonitor:
    """训练监控器"""
    
    def __init__(self):
        self.history = {
            'loss': [],
            'accuracy': [],
            'learning_rate': [],
            'grad_norm': [],
            'memory_usage': []
        }
    
    def record(self, metrics):
        """记录指标"""
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
    
    def print_summary(self):
        """打印训练摘要"""
        print("=== 训练摘要 ===")
        for key, values in self.history.items():
            if values:
                print(f"{key}: 最后值={values[-1]:.4f}, "
                      f"平均值={sum(values)/len(values):.4f}")
    
    def detect_anomalies(self):
        """检测异常"""
        anomalies = []
        
        # 检查梯度爆炸
        if self.history['grad_norm']:
            last_grad_norm = self.history['grad_norm'][-1]
            if last_grad_norm > 1000:
                anomalies.append("梯度爆炸")
        
        # 检查损失NaN
        if any(torch.isnan(torch.tensor(self.history['loss']))):
            anomalies.append("损失出现NaN")
        
        # 检查准确率下降
        if len(self.history['accuracy']) > 10:
            recent_acc = self.history['accuracy'][-5:]
            if max(recent_acc) - min(recent_acc) > 20:
                anomalies.append("准确率大幅波动")
        
        if anomalies:
            print("检测到异常:", anomalies)
        else:
            print("未检测到异常")
        
        return anomalies

# 使用示例
monitor = TrainingMonitor()
monitor.record({'loss': 0.5, 'accuracy': 85.0})
monitor.print_summary()

print("训练监控器定义完成")
```

## 自定义操作与扩展

### 自定义CUDA内核
```python
print("\n=== 自定义操作 ===")

# 注意：实际的自定义CUDA内核需要C++/CUDA代码
# 这里展示Python层面的自定义操作

class CustomActivation(nn.Module):
    """自定义激活函数"""
    
    def __init__(self, alpha=0.1):
        super(CustomActivation, self).__init__()
        self.alpha = alpha
    
    def forward(self, x):
        # 自定义激活函数：Swish的变体
        return x * torch.sigmoid(self.alpha * x)
    
    def extra_repr(self):
        return f'alpha={self.alpha}'

# 使用示例
custom_activation = CustomActivation(alpha=1.0)
test_input = torch.randn(5, 10)
output = custom_activation(test_input)
print(f"自定义激活函数输出形状: {output.shape}")

print("自定义操作定义完成")
```

### 自动微分扩展
```python
print("\n=== 自动微分扩展 ===")

class CustomFunction(torch.autograd.Function):
    """自定义自动微分函数"""
    
    @staticmethod
    def forward(ctx, input, weight, bias):
        """前向传播"""
        ctx.save_for_backward(input, weight, bias)
        output = input @ weight.t() + bias
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """反向传播"""
        input, weight, bias = ctx.saved_tensors
        
        grad_input = grad_output @ weight
        grad_weight = grad_output.t() @ input
        grad_bias = grad_output.sum(0)
        
        return grad_input, grad_weight, grad_bias

# 使用示例
custom_op = CustomFunction.apply

input_data = torch.randn(3, 5, requires_grad=True)
weight = torch.randn(4, 5, requires_grad=True)
output = custom_op(input_data, weight, torch.zeros(4))

print(f"自定义函数输出形状: {output.shape}")
print("自定义自动微分函数定义完成")
```

## 实际应用示例

### 生产环境部署
```python
print("\n=== 生产环境部署示例 ===")

class ProductionModel:
    """生产环境模型包装器"""
    
    def __init__(self, model_path, use_onnx=True):
        self.use_onnx = use_onnx
        
        if use_onnx and model_path.endswith('.onnx'):
            # 使用ONNX Runtime
            try:
                import onnxruntime as ort
                self.session = ort.InferenceSession(model_path)
                self.input_name = self.session.get_inputs()[0].name
                print("使用ONNX Runtime进行推理")
            except ImportError:
                print("ONNX Runtime未安装，回退到PyTorch")
                self.use_onnx = False
        
        if not use_onnx:
            # 使用PyTorch
            self.model = torch.load(model_path)
            self.model.eval()
            print("使用PyTorch进行推理")
    
    def predict(self, input_data):
        """预测接口"""
        if self.use_onnx:
            # ONNX推理
            if isinstance(input_data, torch.Tensor):
                input_data = input_data.detach().cpu().numpy()
            
            outputs = self.session.run(None, {self.input_name: input_data})
            return torch.from_numpy(outputs[0])
        else:
            # PyTorch推理
            with torch.no_grad():
                if isinstance(input_data, np.ndarray):
                    input_data = torch.from_numpy(input_data)
                
                output = self.model(input_data)
                return output
    
    def batch_predict(self, input_batch, batch_size=32):
        """批量预测"""
        predictions = []
        
        for i in range(0, len(input_batch), batch_size):
            batch = input_batch[i:i+batch_size]
            pred = self.predict(batch)
            predictions.append(pred)
        
        return torch.cat(predictions)

print("生产环境模型包装器定义完成")
```

## 总结

本章详细介绍了PyTorch的高级特性：

1. **分布式训练**：数据并行、模型并行、DDP
2. **模型部署**：ONNX导出、ONNX Runtime推理
3. **性能优化**：模型量化、内存优化、计算优化
4. **高级调试**：模型分析、训练监控、异常检测
5. **自定义扩展**：自定义操作、自动微分函数
6. **生产部署**：生产环境模型包装器

这些高级特性对于构建大规模、高性能的深度学习应用至关重要。