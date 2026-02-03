# Python数据分析基础教程：NumPy与Pandas

## NumPy基础

### NumPy简介

NumPy（Numerical Python）是 Python 进行科学计算的一个扩展库，提供了大量的函数和操作，主要用于对多维数组执行计算，它比 Python 自身的嵌套列表结构要高效的多。

NumPy 数组和 Python 列表的主要区别：
- 数组会对元素的数据类型做统一，而列表不会。
- 数组创建后具有固定大小，而列表由于内存自动管理，可动态调整。

### 创建数组

#### `np.array()`函数

**作用**：创建一个数组对象并返回（`ndarray`实例对象）

**语法**：`np.array(object, dtype=None)`

**参数说明**：
- `object`：array_like，类似于数组的对象。如果object是标量，则返回包含object的0维数组
- `dtype`：data-type，数据类型。如果没有给出，会从输入数据推断数据类型

**常用数据类型**：
| dtype常用值 | 描述                    |
| ----------- | ----------------------- |
| `np.int8    ` | 有符号整数（1个字节）   |
| `np.int16   ` | 有符号整数（2个字节）   |
| `np.int32   ` | 有符号整数（4个字节）   |
| `np.int64   ` | 有符号整数（8个字节）   |
| `np.uint8   ` | 无符号整数（1个字节）   |
| `np.uint16  ` | 无符号整数（2个字节）   |
| `np.uint32  ` | 无符号整数（4个字节）   |
| `np.uint64  ` | 无符号整数（8个字节）   |
| `np.float16 ` | 半精度浮点数（2个字节） |
| `np.float32 ` | 单精度浮点数（4个字节） |
| `np.float64 ` | 双精度浮点数（8个字节） |

**`ndarray`常用属性**：
| ndarray常用属性 | 描述                           |
| --------------- | ------------------------------ |
| `ndarray.ndim    `| 秩，即轴的数量或维度的数量     |
| `ndarray.shape   `| 数组的形状                     |
| `ndarray.size    `| 数组中数据的总个数             |
| `ndarray.dtype   `| 数组中的数据类型               |
| `ndarray.itemsize`| 数组中的数据大小，以字节为单位 |

**示例代码**：
```python
import numpy as np

# 从标量创建数组
num = 789
arr = np.array(num)
print(num)          # 输出: 789
print(arr)           # 输出: [789]
print(type(num))     # 输出: <class 'int'>
print(type(arr))     # 输出: <class 'numpy.ndarray'>

# 从列表创建一维数组
lst = [6, 7, 1, 0, 9, 8]
arr = np.array(lst)
print(lst)           # 输出: [6, 7, 1, 0, 9, 8]
print(arr)           # 输出: [6 7 1 0 9 8]

# 从嵌套列表创建二维数组
lst = [[6, 7, 1], [0, 9, 8]]
arr = np.array(lst)
print(lst)           # 输出: [[6, 7, 1], [0, 9, 8]]
print(arr)           # 输出: [[6 7 1] [0 9 8]]
print(arr.ndim)      # 输出: 2 (二维数组)
print(arr.dtype)     # 输出: int32 (整数类型)
print(arr.itemsize)   # 输出: 4 (每个元素4字节)
print(arr.shape)     # 输出: (2, 3) (2行3列)
print(arr.size)       # 输出: 6 (总共6个元素)
```

#### `np.arange()`函数

**作用**：返回给定区间内的均匀间隔值构成的数组

**语法**：`np.arange([start,] stop[, step])`

**示例代码**：
```python
import numpy as np

# 创建等差数列
print(np.arange(3))        # 输出: [0 1 2]
print(np.arange(3.0))      # 输出: [0. 1. 2.]
print(np.arange(3, 7))     # 输出: [3 4 5 6]
print(np.arange(3, 7, 2))  # 输出: [3 5]
print(np.arange(7, 3, -2))# 输出: [7 5]
print(np.arange(3, 7, 0.5))# 输出: [3.  3.5 4.  4.5 5.  5.5 6.  6.5]
```

#### `np.linspace()`函数

**作用**：把给定区间分成指定数量的均匀间隔样本，构成数组并返回

**语法**：`np.linspace(start, stop, num=50, dtype=None)`

**参数说明**：
- `start`：序列的起始值
- `stop`：序列的结束值
- `num`：要生成的样本数量，默认为50
- `dtype`：数组的数据类型，默认自动推断，永远不会是整数

**示例代码**：
```python
import numpy as np

# 创建线性间隔的数组
print(np.linspace(1, 50))                  # 生成1到50之间50个均匀间隔的数
print(np.linspace(1, 10, num=10))           # 输出: [ 1.  2.  3.  4.  5.  6.  7.  8.  9. 10.]
print(np.linspace(1, 10, num=10, dtype=np.int32))  # 指定为整数类型
```

### 基本运算

**核心知识点**：数组的算术运算和比较运算为逐元素操作

**示例代码**：
```python
import numpy as np 

# 数组与标量的运算
a = np.array([[1, 2], [3, 4], [5, 6]])
print(a + 2)  # 输出: [[3 4] [5 6] [7 8]]
print(a - 2)  # 输出: [[-1  0] [ 1  2] [ 3  4]]
print(a * 2)  # 输出: [[ 2  4] [ 6  8] [10 12]]
print(a / 2)  # 输出: [[0.5 1. ] [1.5 2. ] [2.5 3. ]]
print(a < 4)  # 输出: [[ True  True] [ True False] [False False]]
print(a > 3)  # 输出: [[False False] [False  True] [ True  True]]

# 数组与数组的运算
b = np.array([[2, 2], [2, 1], [1, 1]])
print(a + b)  # 输出: [[3 4] [5 5] [6 7]]
print(a - b)  # 输出: [[-1  0] [ 1  3] [ 4  5]]
print(a * b)  # 输出: [[ 2  4] [ 6  4] [ 5  6]]
print(a / b)  # 输出: [[0.5 1. ] [1.5 4. ] [5.  6. ]]
print(a < b)  # 输出: [[ True False] [False False] [False False]]
print(a > b)  # 输出: [[False  True] [ True  True] [ True  True]]
```

### 广播机制

**核心知识点**：当两个数组的形状不同时，NumPy会通过广播机制使它们兼容运算。广播规则是：后缘维度相同或者不同的维度有1，可以广播。

**示例代码**：
```python
import numpy as np

# 不同形状数组的广播运算
a = np.arange(24).reshape((2, 3, 4))  # 形状(2,3,4)
b = np.arange(12).reshape((3, 4))     # 形状(3,4) -> 广播为(1,3,4)
c = np.arange(4).reshape((1, 4))      # 形状(1,4) -> 广播为(1,1,4)
d = np.arange(4).reshape(4)           # 形状(4,) -> 广播为(1,4)
e = np.arange(12).reshape((1, 3, 4))  # 形状(1,3,4) -> 广播为(2,3,4)
f = np.arange(6).reshape((2, 3, 1))   # 形状(2,3,1) -> 广播为(2,3,4)
g = np.arange(2).reshape((2, 1, 1))   # 形状(2,1,1) -> 广播为(2,3,4)
h = np.arange(2).reshape((1, 2, 1, 1))# 形状(1,2,1,1) -> 广播为(2,2,3,4)
i = np.arange(10).reshape((5, 2, 1, 1))# 形状(5,2,1,1) -> 不能与a广播

print((a + b).shape)  # 输出: (2, 3, 4)
print((a + c).shape)  # 输出: (2, 3, 4)
print((a + d).shape)  # 输出: (2, 3, 4)
print((a + e).shape)  # 输出: (2, 3, 4)
print((a + f).shape)  # 输出: (2, 3, 4)
print((a + g).shape)  # 输出: (2, 3, 4)
# print((a + h).shape)  # ValueError: operands could not be broadcast together
```

### 索引和切片

**核心知识点**：数组除了支持Python序列的索引和切片操作以外，还可以针对各个轴进行索引和切片操作。

#### 序列索引和切片

**示例代码**：
```python
import numpy as np

lst = [6, 8, 9, 1, 3]
arr = np.array(lst)
print(lst)  # 输出: [6, 8, 9, 1, 3]
print(arr)  # 输出: [6 8 9 1 3]

# 索引操作
item_lst = lst[2]    # 输出: 9
part_lst = lst[2:3]  # 输出: [9]
item_arr = arr[2]    # 输出: 9
part_arr = arr[2:3]  # 返回视图，输出: [9]
print(item_lst)
print(part_lst)
print(item_arr)
print(part_arr)

# 修改操作
lst[2] = 99
arr[2] = 99
print(item_lst)  # 输出: 9 (不受原列表修改影响)
print(part_lst)  # 输出: [9] (不受原列表修改影响)
print(item_arr)  # 输出: 99 (受原数组修改影响，显示动态性)
print(part_arr)  # 输出: [99] (受原数组修改影响，显示动态性)
```

#### 数组针对各个轴的索引和切片

**示例代码**：
```python
import numpy as np

# 创建三维数组
lst = [[[6, 7, 5, 1],
        [2, 9, 8, 0],
        [3, 4, 2, 8]],

       [[4, 5, 2, 3],
        [2, 9, 7, 1],
        [9, 5, 6, 7]]]

arr = np.array(lst)  # shape: (2, 3, 4)
print(lst[1][0][2])  # 输出: 2 (列表索引方式)
print(arr[1][0][2])  # 输出: 2 (数组索引方式)
print(arr[1, 0, 2])  # 输出: 2 (数组索引方式，更简洁)

# 切片操作
print(lst[1:2][:1])  # 列表切片
print(arr[1:2][:1])  # 数组切片
print(arr[1:2, :1])  # 数组针对轴的切片，更灵活

print(lst[1][::2][0])  # 输出: [4, 5, 2, 3]
print(arr[1][::2][0])  # 输出: [4, 5, 2, 3]
print(arr[1, ::2, 0])  # 输出: [4 2] (第1维索引为1，第2维步长为2，第3维索引为0)
```

#### 数组的高阶索引

**核心知识点**：使用整数列表或者布尔数组作为索引

**示例代码**：
```python
import numpy as np

x = np.arange(24).reshape((3, 2, 4))
print(x)

# 整数数组索引
# 可以理解为x[2], x[0], x[0]构成一个更高维度的数组
print(x[[2, 0, 0]])

# 可以理解为x[2, 0], x[0, 0], x[1, 1]构成一个更高维度的数组
print(x[[2, 0, 1], [0, 0, 1]])

# 可以理解为x[2, 0, 1], x[0, 0, 2], x[1, 1, 3]构成一个更高维度的数组
print(x[[2, 0, 1], [0, 0, 1], [1, 2, 3]])

# 基本索引和高阶索引组合时, 会发生广播
print(x[0, [0, 0, 1], [1, 2, 3]])  # 等价于下面两个
print(x[[0], [0, 0, 1], [1, 2, 3]])
print(x[[0, 0, 0], [0, 0, 1], [1, 2, 3]])

# 切片在高阶索引一侧, 按照轴的顺序定shape即可
print(x[::2, [0, 0, 1], [3, 0, 2]])  # shape: (2, 3)
print(x[[2, 0, 1], [1, 0, 1], ::2])  # shape: (3, 2)

# 切片两侧都有高阶索引时, 定shape时高阶索引在前, 切片在后
print(x[[2, 0, 1], 1:, [3, 0, 2]])  # shape: (3, 1)
```

#### 布尔索引

**核心知识点**：使用布尔数组作为索引，返回布尔值为True对应的元素

**示例代码**：
```python
import numpy as np

x = np.arange(24).reshape((3, 2, 4))
print(x)

# 对标量进行操作的bool索引 (针对最后一个维度)
bool_list = [[[True, False, True, False],
              [False, True, False, True]],
             [[True, False, True, False],
              [False, True, False, True]],
             [[True, False, True, False],
              [False, True, False, True]]]

print(x[np.array(bool_list)])

# 条件表达式创建bool索引
# x > 13 得到一个shape为(3, 2, 4)的bool数组
print(x[x > 13])  # 返回大于13的所有元素

# 对特定轴进行操作的bool索引
# 对1轴进行操作, 只需要创建一个shape为(3, 2)的bool索引
bool_list = [[True, False],
             [False, True],
             [True, False]]
print(x[np.array(bool_list)])  # 只选择第1维中True对应的行

# 对0轴进行操作, 只需要创建一个shape为(3,)的bool索引
bool_list = [True, False, True]
print(x[np.array(bool_list)])  # 只选择第0维中True对应的二维数组
```

### 常用操作

#### `np.reshape()`函数

**作用**：保证 `size` 不变，在不更改数据的情况下为数组赋予新的形状

**语法**：`np.reshape(arr_like, newshape)`

**参数说明**：
- `arr_like`：要重塑的数组
- `newshape`：新的形状，可以是整数或元组。如果是整数，则结果将是该长度的1-D数组。一个形状维度可以是-1，值将自行推断

**示例代码**：
```python
import numpy as np

# 重塑数组形状
arr1 = np.arange(6).reshape((2, 1, 3))
arr2 = np.reshape(arr1, 6)      # 转换为一维数组
arr3 = np.reshape(arr1, -1)     # 自动推断为一维数组
arr4 = np.reshape(arr1, (-1,))  # 明确指定为一维数组
print(arr2)  # 输出: [0 1 2 3 4 5]
print(arr3)  # 输出: [0 1 2 3 4 5]
print(arr4)  # 输出: [0 1 2 3 4 5]

# 使用-1自动推断维度
arr1 = np.arange(24)
arr2 = np.reshape(arr1, (2, 2, -1, 2))
print(arr2.shape)  # 输出: (2, 2, 3, 2) -1被推断为3
```

#### `ndarray.flatten()`方法

**作用**：返回扁平化到一维的数组

**语法**：`ndarray.flatten()`

**示例代码**：
```python
import numpy as np

a = np.array([[1,2], [3,4]])
print(a.flatten())  # 输出: [1 2 3 4]
```

#### `ndarray.T`属性

**作用**：转置数组

**语法**：`ndarray.T`

**示例代码**：
```python
import numpy as np

# 二维数组转置
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a)       # 输出: [[1 2 3] [4 5 6]]
print(a.T)     # 输出: [[1 4] [2 5] [3 6]]

# 多维数组转置
a = np.arange(24).reshape((2, 3, 4))
print(a.T.shape)  # 输出: (4, 3, 2) 维度反转
```

#### `np.swapaxes()`函数

**作用**：交换数组的两个轴

**语法**：`np.swapaxes(arr_like, axis1, axis2)`

**示例代码**：
```python
import numpy as np

# 交换二维数组的轴
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a)
print(np.swapaxes(a, 0, 1))

# 交换多维数组的轴
a = np.arange(24).reshape((2, 3, 4))
print(np.swapaxes(a, 0, 2).shape)  # 输出: (4, 3, 2)
```

#### `np.transpose()`函数

**作用**：通过`axes`参数排列数组的`shape`，`axes`没有指定，默认为转置

**语法**：`np.transpose(arr_like, axes=None)`

**示例代码**：
```python
import numpy as np

# 二维数组转置
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a)
print(np.transpose(a))

# 多维数组自定义转置
a = np.arange(24).reshape((2, 3, 4))
# 将第0轴和第1轴交换，保持第2轴不变
print(np.transpose(a, (1, 0, 2)).shape)  # 输出: (3, 2, 4)
```

#### `np.concatenate()`函数

**作用**：沿现有轴连接一系列数组

**语法**：`np.concatenate(arrays, axis=0)`

**参数说明**：
- `arrays`：Sequence[ArrayLike]，要连接的数组序列
- `axis`：沿哪个轴连接，如果`axis`为`None`，则数组在使用前会被扁平化

**示例代码**：
```python
import numpy as np

# 连接数组
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6]])
print(np.concatenate((a, b), axis=0))  # 按行连接
print(np.concatenate((a, b.T), axis=1))  # 按列连接
print(np.concatenate((a, b), axis=None))  # 扁平化后连接
```

#### `np.stack()`函数

**作用**：沿新轴连接一系列数组

**语法**：`np.stack(arrays, axis=0)`

**参数说明**：
- `arrays`：Sequence[ArrayLike]，要堆叠的数组序列
- `axis`：沿哪个轴堆叠

**示例代码**：
```python
import numpy as np

# 堆叠数组
a1 = np.arange(6).reshape((2, 3))
a2 = np.arange(10, 16).reshape((2, 3))
a3 = np.arange(20, 26).reshape((2, 3))
a4 = np.arange(30, 36).reshape((2, 3))
print(np.stack((a1, a2, a3, a4)).shape)      # 沿新轴0堆叠，形状(4,2,3)
print(np.stack((a1, a2, a3, a4), axis=1).shape)  # 沿新轴1堆叠，形状(2,4,3)
print(np.stack((a1, a2, a3, a4), axis=2).shape)  # 沿新轴2堆叠，形状(2,3,4)
```

### 数学函数

#### 三角函数

**函数列表**：

| 函数 | 作用 | 参数 | 返回值 |
|-----|------|------|--------|
| `np.sin(x)` | 正弦函数 | x：角度（弧度值） | 正弦值 |
| `np.cos(x)` | 余弦函数 | x：角度（弧度值） | 余弦值 |
| `np.tan(x)` | 正切函数 | x：角度（弧度值） | 正切值 |
| `np.arcsin(x)` | 反正弦函数 | x：正弦值 | 角度（弧度值） |
| `np.arccos(x)` | 反余弦函数 | x：余弦值 | 角度（弧度值） |
| `np.arctan(x)` | 反正切函数 | x：正切值 | 角度（弧度值） |

**示例代码**：
```python
import numpy as np

# 三角函数示例
print(np.sin(np.pi/2))  # 输出: 1.0
print(np.sin(np.array((0, 30, 90)) * np.pi / 180))  # 输出: [0.         0.5        1.        ]

print(np.cos(np.pi/2))  # 输出: 6.123233995736766e-17 (约等于0)
print(np.cos(np.array((0, 60, 90)) * np.pi / 180))  # 输出: [1.         0.5        6.123234e-17]

print(np.tan(-np.pi))  # 输出: -1.2246467991473532e-16 (约等于0)
print(np.tan(np.array((0, 180)) * np.pi / 180))  # 输出: [ 0.00000000e+00 -1.22464680e-16]

# 反三角函数示例
print(np.arcsin(1))  # 输出: 1.5707963267948966 (π/2)
print(np.arcsin(np.array([0.5, -0.5])))  # 输出: [ 0.52359878 -0.52359878]

print(np.arccos(-1))  # 输出: 3.141592653589793 (π)
print(np.arccos(np.array([0.5, 1])))  # 输出: [1.04719755 0.        ]

print(np.arctan(1))  # 输出: 0.7853981633974483 (π/4)
print(np.arctan(np.array([0, -1])))  # 输出: [ 0.         -0.78539816]
```

#### 舍入函数

**函数列表**：

| 函数 | 作用 | 参数 | 返回值 |
|-----|------|------|--------|
| `np.floor(x)` | 返回 x 的底限（向下取整） | x：数组或标量 | 向下取整后的值 |
| `np.ceil(x) `| 返回 x 的上限（向上取整） | x：数组或标量 | 向上取整后的值 |

**示例代码**：
```python
import numpy as np

# 舍入函数示例
a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
print(np.floor(a))  # 输出: [-2. -2. -1.  0.  1.  1.  2.] (向下取整)
print(np.ceil(a))   # 输出: [-1. -1. -0.  1.  2.  2.  2.] (向上取整)
```

#### 指数和对数函数

**函数列表**：

| 函数 | 作用 | 参数 | 返回值 |
|-----|------|------|--------|
| `np.exp(x) `| 计算 e 的 x 幂次方 | x：数组或标量 | `e^x` 的值 |
| `np.log(x) `| 计算 x 的自然对数 | x：数组或标量 | `ln(x)` 的值 |
| `np.log2(x)` | 计算 x 的以 2 为底的对数 | x：数组或标量 | `log₂(x)` 的值 |
| `np.log10(x) `| 计算 x 的以 10 为底的对数 | x：数组或标量 | `log₁₀(x)` 的值 |

**示例代码**：
```python
import numpy as np

# 指数函数示例
# e的0次方、e的1次方、e的2次方
print(np.exp([0, 1, 2]))  # 输出: [1.         2.71828183 7.3890561 ]

# 对数函数示例
print(np.log([1, np.e, np.e**2]))  # 输出: [0. 1. 2.]

print(np.log2(np.array([1, 2, 2**4])))  # 输出: [0. 1. 4.]

print(np.log10([1e-15, 1000]))  # 输出: [-15.   3.]
```

### 统计函数

**函数列表**：

| 函数 | 作用 | 参数 | 返回值 |
|-----|------|------|--------|
| `np.max(arr_like, axis=None) `| 返回沿给定轴的最大值 | arr_like: 数组, axis: 轴, keepdims: 是否保持维度 | 最大值 |
| `np.min(arr_like, axis=None) `| 返回沿给定轴的最小值 | arr_like: 数组, axis: 轴, keepdims: 是否保持维度 | 最小值 |
| `np.mean(arr_like, axis=None)` | 返回沿给定轴的平均值 | arr_like: 数组, axis: 轴, keepdims: 是否保持维度 | 平均值 |
| `np.var(arr_like, axis=None) `| 返回沿给定轴的方差 | arr_like: 数组, axis: 轴, keepdims: 是否保持维度 | 方差 |
| `np.std(arr_like, axis=None) `| 返回沿给定轴的标准差 | arr_like: 数组, axis: 轴, keepdims: 是否保持维度 | 标准差 |
| `np.sum(arr_like, axis=None) `| 返回给定轴上数组元素的和 | arr_like: 数组, axis: 轴, keepdims: 是否保持维度, initial: 初始值 | 总和 |
| `np.prod(arr_like, axis=None)` | 返回给定轴上数组元素的乘积 | arr_like: 数组, axis: 轴, keepdims: 是否保持维度, initial: 初始值 | 乘积 |
| `np.argmax(arr_like, axis=None) `| 返回沿轴的最大值的索引 | arr_like: 数组, axis: 轴 | 最大值的索引 |
| `np.argmin(arr_like, axis=None) `| 返回沿轴的最小值的索引 | arr_like: 数组, axis: 轴 | 最小值的索引 |

**示例代码**：
```python
import numpy as np

# 准备测试数据
lis = [[0, 1, 7, 3], [4, 9, 6, 2], [8, 5, 11, 10]]
arr1 = np.array(lis)
print(arr1)
# 输出: [[ 0  1  7  3]
#        [ 4  9  6  2]
#        [ 8  5 11 10]]

# 最大值和最小值
print(np.max(arr1))         # 输出: 11 (所有元素中的最大值)
print(np.max(arr1, axis=0)) # 输出: [ 8  9 11 10] (每列的最大值)
print(np.max(arr1, axis=1)) # 输出: [ 7  9 11] (每行的最大值)

print(np.min(arr1))         # 输出: 0 (所有元素中的最小值)
print(np.min(arr1, axis=0)) # 输出: [0 1 6 2] (每列的最小值)
print(np.min(arr1, axis=1)) # 输出: [0 2 5] (每行的最小值)

# 平均值
print(np.mean(arr1))         # 输出: 5.666666666666667 (所有元素的平均值)
print(np.mean(arr1, axis=0)) # 输出: [4.         5.         8.         5.        ] (每列的平均值)
print(np.mean(arr1, axis=1)) # 输出: [2.75 5.25 8.5 ] (每行的平均值)

# 方差和标准差
print(np.var(arr1))         # 输出: 14.222222222222223 (所有元素的方差)
print(np.var(arr1, axis=0)) # 输出: [10.66666667 10.66666667  4.66666667 14.        ] (每列的方差)
print(np.var(arr1, axis=1)) # 输出: [ 8.1875  7.1875  6.5   ] (每行的方差)

print(np.std(arr1))         # 输出: 3.773592453407634 (所有元素的标准差)
print(np.std(arr1, axis=0)) # 输出: [3.26598632 3.26598632 2.1602469  3.74165739] (每列的标准差)
print(np.std(arr1, axis=1)) # 输出: [2.86138283 2.68128158 2.54950976] (每行的标准差)

# 总和和乘积
print(np.sum(arr1))         # 输出: 66 (所有元素的总和)
print(np.sum(arr1, axis=0)) # 输出: [12 15 24 15] (每列的总和)
print(np.sum(arr1, axis=1)) # 输出: [11 21 34] (每行的总和)

print(np.prod([1, 2, 3, 4]))  # 输出: 24 (所有元素的乘积)
print(np.prod([[1, 2], [3, 4]]))  # 输出: 24 (所有元素的乘积)
print(np.prod([[1, 2], [3, 4]], axis=1))  # 输出: [ 2 12] (每行的乘积)

# 最大值和最小值的索引
a = np.arange(6).reshape(2, 3) + 10
print(a)
# 输出: [[10 11 12]
#        [13 14 15]]

print(np.argmax(a))         # 输出: 5 (扁平化后的索引)
print(np.argmax(a, axis=0)) # 输出: [1 1 1] (每列最大值的行索引)
print(np.argmax(a, axis=1)) # 输出: [2 2] (每行最大值的列索引)

print(np.argmin(a))         # 输出: 0 (扁平化后的索引)
print(np.argmin(a, axis=0)) # 输出: [0 0 0] (每列最小值的行索引)
print(np.argmin(a, axis=1)) # 输出: [0 0] (每行最小值的列索引)
```

#### 其他常用函数

**函数列表**：

| 函数 | 作用 | 参数 | 返回值 |
|-----|------|------|--------|
| `np.nonzero(arr_like)     ` | 返回非零元素的索引 | arr_like: 数组 | 非零元素的索引元组 |
| `np.where(condition, x, y)` | 条件选择 | condition: 条件, x, y: 选择值 | 根据条件选择的数组 |
| `np.argwhere(arr_like)    ` | 找出数组中按元素分组的非零元素的索引 | arr_like: 数组 | 非零元素的索引数组 |
| `np.maximum(x1, x2)       ` | 返回x1和x2逐个元素比较中的最大值 | x1, x2: 数组 | 逐元素最大值 |
| `np.minimum(x1, x2)       `| 返回x1和x2逐个元素比较中的最小值 | x1, x2: 数组 | 逐元素最小值 |

**示例代码**：
```python
import numpy as np

# np.nonzero()函数
x = np.array([[3, 0, 0], [0, 4, 0], [5, 6, 0]])
print(x)
print(np.nonzero(x))  # 输出: (array([0, 1, 2, 2]), array([0, 1, 0, 1]))
print(x[np.nonzero(x)])  # 输出: [3 4 5 6] (非零元素)

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(a > 3)  # 输出: [[False False False] [ True  True  True] [ True  True  True]]
print(np.nonzero(a > 3))  # 输出: (array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))
print(a[np.nonzero(a > 3)])  # 输出: [4 5 6 7 8 9] (大于3的元素)

# np.where()函数
a = np.arange(10)
print(np.where(a < 5, a, 10*a))  # 输出: [0 1 2 3 4 50 60 70 80 90]

# 当where内有三个参数时，当condition成立时返回x，当condition不成立时返回y
print(np.where([[True, False], [True, True]], 
                [[1, 2], [3, 4]], 
                [[9, 8], [7, 6]]))
# 输出: [[1 8] [3 4]]

# 如果只传第一个参数，返回符合条件的元素的索引
a = np.array([2, 4, 6, 8, 10])
print(np.where(a > 5))  # 输出: (array([2, 3, 4]),)

# np.argwhere()函数
x = np.arange(6).reshape(2, 3)
print(x)       # 输出: [[0 1 2] [3 4 5]]
print(x>1)     # 输出: [[False False  True] [ True  True  True]]
print(np.argwhere(x>1))  # 输出: [[0 2] [1 0] [1 1] [1 2]]

# np.maximum()和np.minimum()函数
print(np.maximum([2, 3, 4], [1, 5, 2]))  # 输出: [2 5 4]
print(np.minimum([2, 3, 4], [1, 5, 2]))  # 输出: [1 3 2]

print(np.maximum([[2, 3], [4, 5]], [[1, 5], [2, 6]]))  # 输出: [[2 5] [4 6]]
print(np.minimum([[2, 3], [4, 5]], [[1, 5], [2, 6]]))  # 输出: [[1 3] [2 5]]
```

### 随机数生成

**函数列表**：

| 函数 | 作用 | 参数 | 返回值 |
|-----|------|------|--------|
| `np.random.normal(loc, scale, size) `| 从正态分布中抽取随机样本 | loc: 均值, scale: 标准差, size: 输出形状 | 随机样本数组 |
| `np.random.randint(low, high, size) `| 从离散均匀分布中抽取随机整数 | low: 下限, high: 上限, size: 输出形状 | 随机整数数组 |
| `np.random.uniform(low, high, size) `| 从均匀分布中抽取随机样本 | low: 下限, high: 上限, size: 输出形状 | 随机样本数组 |
| `np.random.permutation(x) `| 随机排列序列 | x: int或array_like | 随机排列的数组 |
| `np.random.seed(x)` | 设置随机数种子 | x: 种子值 | None |

**示例代码**：
```python
import numpy as np

# np.random.normal()函数
print(np.random.normal(3, 2.5, size=(2, 4)))
# 输出: [[1.23456789 3.45678901 5.67890123 4.56789012]
#        [2.34567890 6.78901234 3.45678901 5.67890123]]

# np.random.randint()函数
print(np.random.randint(2, size=10))      # 等价于下一行
print(np.random.randint(0, 2, size=10))    # 等价于上一行
print(np.random.randint(1, 4, size=(2, 3)))
# 输出: [[1 2 3] [2 1 3]]

# np.random.uniform()函数
print(np.random.uniform(2, size=10))      # 等价于下一行
print(np.random.uniform(0, 2, size=10))    # 等价于上一行
print(np.random.uniform(1, 4, size=(2, 3)))
# 输出: [[2.12345678 3.45678901 1.78901234]
#        [2.56789012 3.89012345 1.12345678]]

# np.random.permutation()函数
print(np.random.permutation(6))  # 输出: [2 5 1 4 0 3] (0-5的随机排列)

arr1 = np.array([0, 1, 2, 3, 4, 5])
print(np.random.permutation(arr1))  # 输出: [3 0 5 2 4 1] (原数组的随机排列)

arr2 = np.arange(10).reshape(5, 2)
print(np.random.permutation(arr2))  # 只对第一个维度随机排列

# np.random.seed()函数
np.random.seed(3)
print(np.random.uniform(1, 2, size=4))  # 输出: [1.5507979  1.70814782 1.29090474 1.51082761]

np.random.seed(5)
print(np.random.uniform(1, 2, size=4))  # 输出: [1.22199317 1.87073263 1.20671916 1.96930076]

np.random.seed(3)
print(np.random.uniform(1, 2, size=4))  # 输出: [1.5507979  1.70814782 1.29090474 1.51082761]

np.random.seed()  # 重置种子
print(np.random.uniform(1, 2, size=4))  # 输出随机值
```

---

