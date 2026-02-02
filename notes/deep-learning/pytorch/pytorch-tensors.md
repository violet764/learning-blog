# PyTorch张量操作详解

张量（Tensor）是PyTorch中最基本的数据结构，类似于NumPy的多维数组，但支持GPU加速和自动微分。本章将深入讲解张量的创建、操作、数学运算和高级特性，帮助您熟练掌握PyTorch的核心数据操作。

## 张量创建与初始化

### 基础创建方法
```python
import torch
import numpy as np

print("=== 基础张量创建 ===")

# 从Python列表创建
tensor_from_list = torch.tensor([1, 2, 3, 4, 5])
print(f"从列表创建: {tensor_from_list}")

# 特殊张量创建
zeros_tensor = torch.zeros(3, 4)                    # 全零张量
ones_tensor = torch.ones(2, 3, 4)                  # 全一张量
empty_tensor = torch.empty(2, 2)                   # 未初始化张量
rand_tensor = torch.rand(3, 3)                     # [0,1)均匀分布
randn_tensor = torch.randn(2, 4)                   # 标准正态分布

print(f"全零张量: \n{zeros_tensor}")
print(f"随机张量: \n{rand_tensor}")

# 类似已有张量的创建
like_zeros = torch.zeros_like(rand_tensor)
like_ones = torch.ones_like(zeros_tensor)
print(f"类似随机张量的全零: \n{like_zeros}")
```

### 序列和范围张量
```python
print("\n=== 序列张量 ===")

# 等差数列
arange_tensor = torch.arange(0, 10, 2)            # [0, 10)步长2
linspace_tensor = torch.linspace(0, 1, 5)         # [0,1]等分5份
logspace_tensor = torch.logspace(0, 3, 4)         # 10^0到10^3对数等分

print(f"等差数列: {arange_tensor}")
print(f"线性等分: {linspace_tensor}")
print(f"对数等分: {logspace_tensor}")

# 网格坐标
y_coords, x_coords = torch.meshgrid(
    torch.arange(3), 
    torch.arange(4)
)
print(f"Y坐标网格: \n{y_coords}")
print(f"X坐标网格: \n{x_coords}")
```

### 从其他数据源创建
```python
print("\n=== 数据源转换 ===")

# NumPy互操作
np_array = np.array([[1, 2], [3, 4]])
torch_from_np = torch.from_numpy(np_array)
np_from_torch = torch_from_np.numpy()

print(f"NumPy数组: \n{np_array}")
print(f"PyTorch张量: \n{torch_from_np}")
print(f"转换回NumPy: \n{np_from_torch}")

# 注意：内存共享
np_array[0, 0] = 100
print(f"修改后张量: {torch_from_np}")  # 也会被修改

# 避免内存共享（创建副本）
torch_copy = torch.tensor(np_array)
np_array[0, 0] = 200
print(f"副本张量不受影响: {torch_copy}")
```

## 张量属性与信息

### 基本属性
```python
print("\n=== 张量属性 ===")

tensor = torch.randn(2, 3, 4)

print(f"张量形状: {tensor.shape}")                # 形状
trint(f"张量维度: {tensor.dim()}")                 # 维度数
print(f"元素总数: {tensor.numel()}")              # 元素数量
print(f"数据类型: {tensor.dtype}")                # 数据类型
print(f"存储设备: {tensor.device}")               # 设备信息
print(f"是否需要梯度: {tensor.requires_grad}")    # 梯度跟踪

# 改变数据类型
tensor_float = tensor.float()                    # 转为float32
tensor_double = tensor.double()                  # 转为float64
tensor_int = tensor.int()                        # 转为int32

print(f"Float32类型: {tensor_float.dtype}")
print(f"Float64类型: {tensor_double.dtype}")
```

### 设备管理
```python
print("\n=== 设备管理 ===")

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 设备迁移
cpu_tensor = torch.randn(3, 3)
if torch.cuda.is_available():
    gpu_tensor = cpu_tensor.to(device)           # 迁移到GPU
    back_to_cpu = gpu_tensor.cpu()               # 迁移回CPU
    
    print(f"CPU张量设备: {cpu_tensor.device}")
    print(f"GPU张量设备: {gpu_tensor.device}")
    print(f"返回CPU设备: {back_to_cpu.device}")
```

## 张量索引与切片

### 基础索引
```python
print("\n=== 基础索引 ===")

tensor = torch.randn(4, 5, 6)
print(f"原始张量形状: {tensor.shape}")

# 单元素索引
single_element = tensor[0, 1, 2]
print(f"单个元素: {single_element}")

# 切片操作
first_two = tensor[:2]                           # 前两个元素
last_three = tensor[-3:]                         # 后三个元素
middle = tensor[1:3]                             # 中间元素

print(f"前两个形状: {first_two.shape}")
print(f"后三个形状: {last_three.shape}")

# 步长切片
step_slice = tensor[::2]                         # 每隔一个取一个
reverse = tensor[::-1]                           # 反向
print(f"步长切片形状: {step_slice.shape}")
```

### 高级索引
```python
print("\n=== 高级索引 ===")

tensor = torch.arange(24).reshape(4, 6)
print(f"原始张量: \n{tensor}")

# 整数数组索引
indices = torch.tensor([0, 2, 3])
selected_rows = tensor[indices]
print(f"选择的行: \n{selected_rows}")

# 布尔索引
mask = tensor > 10
filtered = tensor[mask]
print(f"大于10的元素: {filtered}")

# 多维索引
rows = torch.tensor([0, 2])
cols = torch.tensor([1, 3])
selected_elements = tensor[rows, cols]
print(f"特定位置元素: {selected_elements}")
```

### 索引操作函数
```python
print("\n=== 索引函数 ===")

tensor = torch.randn(3, 4)

# 条件索引
mask = tensor > 0
positive_elements = torch.masked_select(tensor, mask)
print(f"正数元素: {positive_elements}")

# 收集操作（高级索引）
indices = torch.tensor([[0, 1], [2, 0]])
gathered = torch.gather(tensor, 1, indices)      # 沿维度1收集
print(f"收集结果: \n{gathered}")

# 分散操作（反向收集）
result = torch.zeros_like(tensor)
result.scatter_(1, indices, 1.0)                 # 沿维度1分散
print(f"分散结果: \n{result}")
```

## 张量形状操作

### 改变形状
```python
print("\n=== 形状操作 ===")

tensor = torch.arange(12)
print(f"原始张量: {tensor}")

# 改变形状（不复制数据）
reshaped = tensor.view(3, 4)
print(f"改变形状后: \n{reshaped}")

# 自动推断维度
auto_reshape = tensor.view(-1, 3)                # -1表示自动计算
print(f"自动推断形状: \n{auto_reshape}")

# 调整形状（可能复制数据）
resized = tensor.reshape(2, 6)
print(f"调整形状: \n{resized}")
```

### 维度操作
```python
print("\n=== 维度操作 ===")

tensor = torch.randn(2, 3, 4)

# 展平操作
flattened = tensor.flatten()                     # 完全展平
flatten_dim = tensor.flatten(start_dim=1)        # 从指定维度开始展平

print(f"完全展平形状: {flattened.shape}")
print(f"部分展平形状: {flatten_dim.shape}")

# 挤压和扩展维度
squeezed = torch.randn(1, 3, 1, 4).squeeze()     # 移除大小为1的维度
unsqueezed = tensor.unsqueeze(0)                 # 在指定位置添加维度

print(f"挤压后形状: {squeezed.shape}")
print(f"扩展后形状: {unsqueezed.shape}")

# 转置和维度交换
transposed = tensor.transpose(0, 1)              # 交换维度0和1
permuted = tensor.permute(2, 0, 1)               # 重新排列维度

print(f"转置形状: {transposed.shape}")
print(f"重排形状: {permuted.shape}")
```

### 连接与分割
```python
print("\n=== 连接与分割 ===")

# 张量连接
tensor1 = torch.randn(2, 3)
tensor2 = torch.randn(2, 3)

cat_result = torch.cat([tensor1, tensor2], dim=0)  # 沿维度0连接
stack_result = torch.stack([tensor1, tensor2])     # 创建新维度堆叠

print(f"连接形状: {cat_result.shape}")
print(f"堆叠形状: {stack_result.shape}")

# 张量分割
chunked = torch.chunk(tensor, chunks=3, dim=1)    # 均匀分割
split_result = torch.split(tensor, 2, dim=1)      # 指定大小分割

print(f"分割块数: {len(chunked)}")
print(f"每块形状: {chunked[0].shape}")
```

## 数学运算

### 基础数学运算
```python
print("\n=== 基础数学运算 ===")

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(f"加法: {a + b}")                          # 逐元素加法
print(f"减法: {a - b}")                          # 逐元素减法
print(f"乘法: {a * b}")                          # 逐元素乘法
print(f"除法: {a / b}")                          # 逐元素除法
print(f"幂运算: {a ** 2}")                       # 平方

# 原地操作（节省内存）
a.add_(b)                                        # 原地加法
print(f"原地加法后a: {a}")
```

### 矩阵运算
```python
print("\n=== 矩阵运算 ===")

A = torch.randn(3, 4)
B = torch.randn(4, 5)

# 矩阵乘法
matmul = torch.matmul(A, B)                      # 矩阵乘法
at_symbol = A @ B                                # @运算符（Python 3.5+）

print(f"矩阵A形状: {A.shape}")
print(f"矩阵B形状: {B.shape}")
print(f"乘积形状: {matmul.shape}")

# 转置和逆矩阵
A_T = A.T                                        # 转置
print(f"转置形状: {A_T.shape}")

# 批量矩阵乘法
batch_A = torch.randn(5, 3, 4)                   # 批量矩阵
batch_B = torch.randn(5, 4, 5)
batch_result = torch.bmm(batch_A, batch_B)       # 批量矩阵乘法
print(f"批量乘积形状: {batch_result.shape}")
```

### 归约运算
```python
print("\n=== 归约运算 ===")

tensor = torch.randn(3, 4)

print(f"所有元素和: {tensor.sum()}")
print(f"每列和: {tensor.sum(dim=0)}")
print(f"每行和: {tensor.sum(dim=1)}")

print(f"所有元素均值: {tensor.mean()}")
print(f"每列均值: {tensor.mean(dim=0)}")

print(f"最大值: {tensor.max()}")
print(f"每行最大值: {tensor.max(dim=1).values}")
print(f"最大值索引: {tensor.max(dim=1).indices}")

# 累积运算
cumsum = tensor.cumsum(dim=1)                    # 累积和
print(f"累积和: \n{cumsum}")
```

## 比较和逻辑运算

### 比较运算
```python
print("\n=== 比较运算 ===")

a = torch.tensor([1, 2, 3, 4, 5])
b = torch.tensor([3, 3, 3, 3, 3])

print(f"a > b: {a > b}")                         # 大于
print(f"a >= b: {a >= b}")                       # 大于等于
print(f"a < b: {a < b}")                         # 小于
print(f"a <= b: {a <= b}")                       # 小于等于
print(f"a == b: {a == b}")                       # 等于
print(f"a != b: {a != b}")                       # 不等于

# 所有和任意
print(f"所有元素大于0: {(a > 0).all()}")
print(f"任意元素大于4: {(a > 4).any()}")
```

### 逻辑运算
```python
print("\n=== 逻辑运算 ===")

x = torch.tensor([True, False, True])
y = torch.tensor([False, True, True])

print(f"逻辑与: {x & y}")                        # 逐元素与
print(f"逻辑或: {x | y}")                        # 逐元素或
print(f"逻辑非: {~x}")                           # 逐元素非
print(f"异或: {x ^ y}")                          # 逐元素异或
```

## 随机数生成

### 随机数分布
```python
print("\n=== 随机数生成 ===")

# 设置随机种子确保可重复性
torch.manual_seed(42)

# 均匀分布
uniform = torch.rand(3, 3)                       # [0,1)均匀分布
print(f"均匀分布: \n{uniform}")

# 正态分布
normal = torch.randn(2, 4)                       # 标准正态分布
print(f"正态分布: \n{normal}")

# 特定分布
bernoulli = torch.bernoulli(torch.tensor([0.3, 0.7, 0.5]))  # 伯努利分布
print(f"伯努利分布: {bernoulli}")

# 从特定范围采样
randint = torch.randint(0, 10, (3, 3))           # [0,10)整数
print(f"整数随机: \n{randint}")
```

### 随机种子管理
```python
print("\n=== 随机种子管理 ===")

# 设置全局种子
torch.manual_seed(123)
rand1 = torch.rand(2, 2)

# 创建独立生成器
generator = torch.Generator()
generator.manual_seed(456)
rand2 = torch.rand(2, 2, generator=generator)

print(f"全局种子结果: \n{rand1}")
print(f"独立生成器结果: \n{rand2}")

# 重置为随机状态
torch.seed()
rand3 = torch.rand(2, 2)
print(f"随机状态结果: \n{rand3}")
```

## 内存管理与性能优化

### 内存布局
```python
print("\n=== 内存管理 ===")

tensor = torch.randn(3, 4)

# 内存连续性
print(f"是否连续: {tensor.is_contiguous()}")

# 确保连续性
if not tensor.is_contiguous():
    contiguous_tensor = tensor.contiguous()
    print(f"连续化后: {contiguous_tensor.is_contiguous()}")

# 内存占用估算
memory_bytes = tensor.element_size() * tensor.numel()
print(f"内存占用: {memory_bytes} 字节")
```

### 原地操作与性能
```python
print("\n=== 性能优化 ===")

# 避免不必要复制
a = torch.randn(1000, 1000)

# 不好的做法（创建副本）
import time
start = time.time()
for _ in range(100):
    a = a + 1  # 创建新张量
end = time.time()
print(f"非原地操作时间: {end - start:.4f}秒")

# 好的做法（原地操作）
a = torch.randn(1000, 1000)
start = time.time()
for _ in range(100):
    a.add_(1)  # 原地操作
end = time.time()
print(f"原地操作时间: {end - start:.4f}秒")
```

## 实际应用示例

### 图像处理中的张量操作
```python
print("\n=== 图像处理示例 ===")

# 模拟图像数据（批次, 通道, 高, 宽）
images = torch.randn(4, 3, 224, 224)

# 图像归一化
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
normalized = (images - mean) / std

print(f"原始图像形状: {images.shape}")
print(f"归一化图像形状: {normalized.shape}")

# 图像裁剪
cropped = normalized[:, :, 100:150, 100:150]
print(f"裁剪后形状: {cropped.shape}")
```

### 自然语言处理中的张量操作
```python
print("\n=== NLP示例 ===")

# 词向量序列（批次, 序列长度, 词向量维度）
word_vectors = torch.randn(2, 10, 300)

# 序列掩码（处理变长序列）
sequence_lengths = torch.tensor([8, 10])
mask = torch.arange(10).unsqueeze(0) < sequence_lengths.unsqueeze(1)
print(f"序列掩码: \n{mask}")

# 应用掩码
masked_vectors = word_vectors * mask.unsqueeze(-1)
print(f"掩码后形状: {masked_vectors.shape}")
```

## 总结

本章详细介绍了PyTorch张量的各种操作，关键要点包括：

1. **多种创建方式**：从列表、NumPy、特殊函数等创建张量
2. **灵活的索引切片**：支持基础索引、高级索引和布尔索引
3. **丰富的形状操作**：改变形状、维度操作、连接分割等
4. **完整的数学运算**：基础运算、矩阵运算、归约运算等
5. **性能优化技巧**：原地操作、内存连续性、设备管理等

熟练掌握张量操作是使用PyTorch的基础，建议通过实际项目练习这些操作，加深理解。