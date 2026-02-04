## Pandas相关1 

### Pandas简介

Pandas 是 Python 进行数据分析的一个扩展库，是基于 NumPy 的一种工具。能够快速得从不同格式的文件中加载数据（比如 CSV 、Excel文件等），然后将其转换为可处理的对象。更多内容：[Pandas官方文档](https://pandas.pydata.org/pandas-docs/stable/reference/index.html)

Pandas 在 ndarray 的基础上构建出了两种更适用于数据分析的存储结构，分别是 Series（一维数据结构）和 DataFrame（二维数据结构）。在操作 Series 和 DataFrame 时，基本上可以看成是 NumPy 中的一维和二维数组来操作，数组的绝大多数操作它们都可以适用。

### Series数据结构

Series是一种一维数据结构，每一个元素都带有一个索引，与 NumPy 中的一维数组类似。Series 可以保存任何数据类型，比如整数、字符串、浮点数、Python 对象等，它的索引默认为整数，从 0 开始依次递增。

#### 创建 Series 对象

**函数**：`pd.Series(data=None, index=None, dtype=None, name=None)`

**参数说明**：
- `data`：array-like, dict, or scalar value（数据）
- `index`：索引必须是不可变数据类型，允许相同。不指定时，默认为从 0 开始依次递增的整数
- `dtype`：数据类型，如果没有指定，则会自动推断得出
- `name`：设置 Series 的名称

**示例代码**：
```python
import numpy as np
import pandas as pd

# 标量创建Series对象: 标量值按照 index 的数量进行重复，并与其一一对应
d = 99
ser = pd.Series(data=d)
print(ser)
# 输出:
# 0    99
# dtype: int64

ser = pd.Series(data=d, index=[1, 2, 3])
print(ser)
# 输出:
# 1    99
# 2    99
# 3    99
# dtype: int64

# str创建Series对象: 当作标量一样处理
d = 'abc'
ser = pd.Series(data=d, index=[1, 2, 3])
print(ser)
# 输出:
# 1    abc
# 2    abc
# 3    abc
# dtype: object

# list创建Series对象
d = ['a', 'b', 'c']
ser = pd.Series(data=d)
print(ser)
# 输出:
# 0    a
# 1    b
# 2    c
# dtype: object

# ndarray创建Series对象
d = np.array([1, 2, 3])
ser = pd.Series(data=d, dtype=np.float64, index=('one', 'two', 'three'), name='test-series')
print(ser)
# 输出:
# one      1.0
# two      2.0
# three    3.0
# Name: test-series, dtype: float64

# dict创建Series对象: 默认用字典的键作为index, 对应字典的值作为数据
d = {'a': 1, 'b': 2, 'c': 3}
ser = pd.Series(data=d)  # index=['a', 'b', 'c'] 可省略
print(ser)
# 输出:
# a    1
# b    2
# c    3
# dtype: int64

# dict创建Series对象: 如果指定索引不是字典的键, 那么会得到缺失值NaN
d = {'a': 1, 'b': 2, 'c': 3}
ser = pd.Series(data=d, index=['a', 'y', 'z'])
print(ser)
# 输出:
# a    1.0
# y    NaN
# z    NaN
# dtype: float64
```

#### 访问 Series 数据

**核心知识点**：两种方式访问 Series 数据 - 位置索引访问、索引标签访问

**示例代码**：
```python
import numpy as np
import pandas as pd

# 位置索引访问
d = np.array([1, 2, 3, 4, 5])
ser = pd.Series(data=d, index=('a', 'e', 'c', 'd', 'e'))
print(ser)
print(ser[1])        # 输出: 2 (第二个元素)
print(ser[1:3])      # 输出: [2 3] (第二个到第三个元素)
print(ser[:-2:2])    # 输出: [1 3] (从第一个到倒数第三个，步长为2)
print(ser[[2, 1, 3]]) # 输出: [3 2 4] (第三、第二、第四个元素)

# 索引标签访问
print(ser['c'])      # 输出: 3 (索引为'c'的元素)
print(ser['e'])      # 输出: a    2
                    #       e    5 (所有索引为'e'的元素)

# 索引标签切片时, 右边不是开区间哦
print(ser['a':'d'])  # 输出: a    1
                    #       e    2
                    #       c    3
                    #       d    4 (从索引'a'到索引'd'的所有元素)

print(ser[:'c':2])   # 输出: a    1
                    #       c    3 (从开始到索引'c'，步长为2)

print(ser[['c', 'e', 'd']])  # 输出: c    3
                            #       e    2
                            #       e    5
                            #       d    4 (指定索引的元素)
```

#### 修改 Series 索引

**核心知识点**：可以通过给 index 属性重新赋值达到修改索引的目的

**示例代码**：
```python
import pandas as pd

ser = pd.Series([4, 7, -5, 3], index=['a', 'b', 'c', 'd'])
print(ser)
# 输出:
# a    4
# b    7
# c   -5
# d    3
# dtype: int64

ser.index = ['aa', 'bb', 'cc', 'dd']  # 修改原数据
print(ser)
# 输出:
# aa    4
# bb    7
# cc   -5
# dd    3
# dtype: int64
```

#### 修改 Series 数据

**核心知识点**：可以通过索引和切片的方式修改数据

**示例代码**：
```python
import pandas as pd

ser = pd.Series([2, 3, 4, 5], index=['a', 'b', 'c', 'd'])
ser['a'] = 8  # 修改单个元素
print(ser)
# 输出:
# a    8
# b    3
# c    4
# d    5
# dtype: int64

ser['b':'d'] = [7, 8, 9]  # 修改多个元素
print(ser)
# 输出:
# a    8
# b    7
# c    8
# d    9
# dtype: int64
```

#### Series 常用属性

**属性列表**：

| 属性   | 描述                                      |
| ------ | ----------------------------------------- |
| `dtype  `| 返回 Series 对象数据类型                  |
| `name   `| 返回 Series 对象名称                      |
| `shape  `| 返回 Series 对象的形状                    |
| `size   `| 返回 Series 中的元素数量                  |
| `values `| 以 ndarray 数组的形式返回 Series 中的数据 |
| `index  `| 返回 index                                |

**示例代码**：
```python
import pandas as pd

d = [1, 2, 3, 4]
ser = pd.Series(data=d, index=['a', 'b', 'c', 'd'], name="Test-Series")
print(ser.dtype)    # 输出: int64 (数据类型)
print(ser.name)      # 输出: Test-Series (名称)
print(ser.size)      # 输出: 4 (元素数量)
print(ser.values)    # 输出: [1 2 3 4] (ndarray数组形式的数据)
print(ser.index)     # 输出: Index(['a', 'b', 'c', 'd'], dtype='object') (索引)
```

#### Series 运算

**核心知识点**：Series 保留了 NumPy 中的数组运算，且 Series 进行数组运算的时候，索引与值之间的映射关系不会发生改变。在进行 Series 和 Series 的运算时，把两个 Series 中索引一样的值进行运算，其他不一样的做并集，对应的值为 `NaN`

**示例代码**：
```python
import pandas as pd

# Series与标量的运算
ser1 = pd.Series([15, 20], index=["a", "b"])
print(ser1 + 1)  # 输出: a    16, b    21
print(ser1 - 1)  # 输出: a    14, b    19
print(ser1 * 2)  # 输出: a    30, b    40
print(ser1 / 2)  # 输出: a    7.5, b   10.0

# Series与Series的运算
ser2 = pd.Series([1, 2], index=["c", "a"])
print(ser1 + ser2)  # 输出: a    16.0, b    NaN, c    NaN
print(ser1 - ser2)  # 输出: a    14.0, b    NaN, c    NaN
print(ser1 * ser2)  # 输出: a    15.0, b    NaN, c    NaN
print(ser1 / ser2)  # 输出: a    15.0, b    NaN, c    NaN
```

### DataFrame数据结构

DataFrame 是一种表格型的二维数据结构，既有行索引（index），又有列索引（columns），且默认都是从0开始递增的整数。可以把每一列看作是共同用一个索引的 Series，且不同列的数据类型可以不同。

#### 创建 DataFrame 对象

**函数**：`pd.DataFrame(data=None, index=None, columns=None, dtype=None)`

**参数说明**：
- `data`：array-like, dict（数据）
- `index`：行索引。不指定时，默认为从 0 开始依次递增的整数
- `columns`：列索引。不指定时，默认为从 0 开始依次递增的整数
- `dtype`：数据类型，如果没有指定，则会自动推断得出

**示例代码**：
```python
import numpy as np
import pandas as pd

# ndarray创建DataFrame对象
d = np.array([[1, 2, 3], [4, 5, 6]])
df = pd.DataFrame(data=d, dtype=np.float64)
print(df)
# 输出:
#      0    1    2
# 0  1.0  2.0  3.0
# 1  4.0  5.0  6.0

# 单一列表创建DataFrame对象
d = ['Tom', 'Bob', 'Linda']
df = pd.DataFrame(data=d)
print(df)
# 输出:
#       0
# 0   Tom
# 1   Bob
# 2 Linda

# 嵌套列表创建DataFrame对象
d = [['Tom', 17], ['Bob', 18], ['Linda', 26]]
df = pd.DataFrame(data=d, index=['p1', 'p2', 'p3'], columns=['name', 'age'])
print(df)
# 输出:
#      name  age
# p1   Tom   17
# p2   Bob   18
# p3 Linda   26

# 字典嵌套列表创建DataFrame对象
# 字典data中, 所有键对应的值的元素个数必须相同
# 默认情况下，字典的键被用作列索引
d = {'name': ['Tom', 'Bob', 'Linda'], 'age': [17, 18, 26]}
df = pd.DataFrame(data=d, index=['p1', 'p2', 'p3'])
print(df)
# 输出:
#      name  age
# p1   Tom   17
# p2   Bob   18
# p3 Linda   26

# Series创建DataFrame对象
d = {'name': pd.Series(['Tom', 'Bob', 'Linda'], index=['p1', 'p2', 'p3']), 
     'age': pd.Series([17, 18, 26], index=['p1', 'p2', 'p8'])}
df = pd.DataFrame(data=d)
print(df)
# 输出:
#      name   age
# p1   Tom  17.0
# p2   Bob  18.0
# p3 Linda   NaN
# p8   NaN  26.0

d = [pd.Series(['Tom', 'Bob', 'Linda'], index=['p1', 'p2', 'p3'], name="name"), 
     pd.Series([17, 18, 26], index=['p1', 'p2', 'p3'], name='age')]
df = pd.DataFrame(data=d)
print(df)
# 输出:
#       p1   p2    p3
# name Tom  Bob Linda
# age   17   18    26
```

#### 访问 DataFrame 数据

**核心知识点**：索引获取列数据，切片获取行数据；`loc`指定标签获取数据，`iloc`指定下标获取数据

**示例代码**：
```python
import pandas as pd

# 创建示例DataFrame
d = {'name': ['Tom', 'Bob', 'Linda'], 'age': [17, 18, 26], 'height': [172, 176, 188]}
df = pd.DataFrame(data=d, index=['p1', 'p2', 'p3'])
print(df)
# 输出:
#       name  age  height
# p1    Tom   17     172
# p2    Bob   18     176
# p3  Linda   26     188

# 索引获取列数据
print(df['age'])          # 输出: p1    17, p2    18, p3    26, Name: age, dtype: int64
print(df[['age', 'name']]) # 输出: age, name列的数据

# 切片获取行数据
print(df[0: 1])          # 下标切片左闭右开，输出第一行
print(df['p1': 'p2'])     # 标签切片两边都是闭区间，输出p1到p2行

# 组合使用
print(df[['name', 'age']][0: : 2])  # 先选择列，再切片行
print(df[0: : 2][['name', 'age']])  # 先切片行，再选择列

# loc指定标签获取数据，iloc指定下标获取数据
# loc允许接两个参数分别是行和列, 且只能接收标签索引
print(df.loc['p1'])                    # 选取行索引为'p1'的数据
print(df.loc['p2', 'age'])             # 选取行索引为'p2'且列索引为'age'的数据
print(df.loc['p2', ['age', 'name']])  # 选取行索引为'p2'且列索引分别为'age'和'name'的数据
print(df.loc[['p3', 'p2'], ['age', 'name']])  # 选取行索引分别为'p3'和'p2'且列索引分别为'age'和'name'的数据

# iloc允许接两个参数分别是行和列, 且只能接收整数索引
print(df.iloc[0])               # 选取行索引为0的数据
print(df.iloc[1, 1])            # 选取行索引为1且列索引为1的数据
print(df.iloc[1, [1, 0]])      # 选取行索引为1且列索引分别为1和0的数据
print(df.iloc[[2, 1], [1, 0]]) # 选取行索引分别为2和1且列索引分别为1和0的数据
```

#### 修改 DataFrame 索引

**核心知识点**：修改对应的属性即可

**示例代码**：
```python
import pandas as pd

d = {'name': ['Tom', 'Bob', 'Linda'], 'age': [17, 18, 26], 'height': [172, 176, 188]}
df = pd.DataFrame(data=d, index=['p1', 'p2', 'p3'])
print(df)
# 输出:
#       name  age  height
# p1    Tom   17     172
# p2    Bob   18     176
# p3  Linda   26     188

# 修改行索引
df.index = ['n1', 'n2', 'n3']

# 修改列索引
df.columns = ['names', 'ages', 'heights']

print(df)
# 输出:
#     names  ages  heights
# n1   Tom    17      172
# n2   Bob    18      176
# n3 Linda   26      188
```

#### 修改 DataFrame 数据

**核心知识点**：对访问的数据重新赋值，即可修改数据；如果访问数据不存在，则会添加数据

**示例代码**：
```python
import pandas as pd

# 创建示例DataFrame
d = {'name': ['Tom', 'Bob', 'Linda'], 'age': [17, 18, 26], 'height': [172, 176, 188]}
df = pd.DataFrame(data=d, index=['p1', 'p2', 'p3'])
print(df)
# 输出:
#       name  age  height
# p1    Tom   17     172
# p2    Bob   18     176
# p3  Linda   26     188

# 修改单列数据
df['height'] = pd.Series([1.72, 1.88, 1.76], index=df.index)  # 使用Series
df['height'] = [1.72, 1.88, 1.76]                            # 使用列表
df.loc[:, 'height'] = [1.72, 1.88, 1.76]                      # 使用loc
df.iloc[:, 2:3] = [1.72, 1.88, 1.76]                          # 使用iloc
print(df)
# 输出:
#       name  age  height
# p1    Tom   17    1.72
# p2    Bob   18    1.88
# p3  Linda   26    1.76

# 修改多列数据
df[['name', 'age']] = pd.DataFrame({'name': ['Bob', 'Tom', 'Jack'], 'age': [19, 22, 27]}, index=df.index)
df[['name', 'age']] = [['Bob', 19], ['Tom', 22], ['Jack', 27]]
df.loc[:, ['name', 'age']] = [['Bob', 19], ['Tom', 22], ['Jack', 27]]
df.iloc[:, :2] = [['Bob', 19], ['Tom', 22], ['Jack', 27]]
print(df)
# 输出:
#      name  age  height
# p1    Bob   19    1.72
# p2    Tom   22    1.88
# p3  Jack   27    1.76

# 追加单列数据
df['weight'] = pd.Series([65, 75, 60], index=df.index)
df['weight'] = [65, 75, 60]
df.loc[:, 'weight'] = [65, 75, 60]
print(df)
# 输出:
#      name  age  height  weight
# p1    Bob   19    1.72      65
# p2    Tom   22    1.88      75
# p3  Jack   27    1.76      60

# 追加多列数据
df[['grade', 'address']] = pd.DataFrame({'grade': ['一', '二', '三'], 'address': ['威宁路', '长宁路', '大马路']}, index=df.index)
df[['grade', 'address']] = [['一', '威宁路'], ['二', '长宁路'], ['三', '大马路']]
df.loc[:, ['grade', 'address']] = [['一', '威宁路'], ['二', '长宁路'], ['三', '大马路']]
print(df)
# 输出:
#      name  age  height  weight grade address
# p1    Bob   19    1.72      65     一     威宁路
# p2    Tom   22    1.88      75     二     长宁路
# p3  Jack   27    1.76      60     三     大马路

# 修改单行数据
df[1:2] = ['Tony', 23, 178]
df.loc['p2'] = pd.Series(['Tony', 23, 178], index=df.columns)
df.iloc[1] = ['Tony', 23, 178]
df.iloc[1:2] = ['Tony', 23, 178]
print(df)
# 输出:
#       name  age  height  weight grade address
# p1     Bob   19    1.72      65     一     威宁路
# p2    Tony   23   178.00     NaN   NaN      NaN
# p3    Jack   27    1.76      60     三     大马路

# 修改多行数据
df[:2] = [['Jack', 27, 1.76], ['Tony', 19, 1.72]]
df.loc[:'p2'] = [['Jack', 27, 1.76], ['Tony', 19, 1.72]]
df.iloc[[0, 1]] = [['Jack', 27, 1.76], ['Tony', 19, 1.72]]
print(df)
# 输出:
#       name  age  height  weight grade address
# p1    Jack   27    1.76      65     一     威宁路
# p2    Tony   19    1.72      75     二     长宁路
# p3    Jack   27    1.76      60     三     大马路

# 追加单行数据
df.loc['p4'] = ['Toby', 23, 178]
print(df)
# 输出:
#       name  age  height  weight grade address
# p1    Jack   27    1.76      65     一     威宁路
# p2    Tony   19    1.72      75     二     长宁路
# p3    Jack   27    1.76      60     三     大马路
# p4    Toby   23  178.00     NaN   NaN      NaN
```

#### DataFrame 常用属性

**属性列表**：

| 属性   | 描述                                 |
| ------ | ------------------------------------ |
|` T      `| 转置                                 |
|` dtypes `| 返回每一列的数据类型                 |
|` shape  `| 返回 DataFrame 的形状                 |
|` size   `| 返回 DataFrame 中的元素数量           |
|` index  `| 返回行索引                           |
|` columns`| 返回列索引                           |
|` axes   `| 以列表形式返回行索引和列索引         |
|` values `| 以 ndarray 数组的形式返回 DataFrame 中的数据 |

**示例代码**：
```python
import pandas as pd
import numpy as np

d = [['Tom', 17], ['Bob', 18], ['Linda', 26]]
df = pd.DataFrame(data=d, index=['p1', 'p2', 'p3'], columns=['name', 'age'])
print(df)
# 输出:
#      name  age
# p1   Tom   17
# p2   Bob   18
# p3 Linda   26

print(df.T)
# 输出:
#        p1   p2    p3
# name  Tom  Bob Linda
# age    17   18    26

print(df.dtypes)
# 输出:
# name    object
# age      int64
# dtype: object

print(df.shape)  # 输出: (3, 2) (3行2列)
print(df.size)   # 输出: 6 (总共6个元素)
print(df.index)  # 输出: Index(['p1', 'p2', 'p3'], dtype='object')
print(df.columns)# 输出: Index(['name', 'age'], dtype='object')
print(df.axes)   # 输出: [Index(['p1', 'p2', 'p3'], dtype='object'), Index(['name', 'age'], dtype='object')]
print(df.values) # 输出: [['Tom' 17] ['Bob' 18] ['Linda' 26]]
```

#### DataFrame 常用方法

**函数列表**：

| 函数 | 作用 | 参数 | 返回值 |
|-----|------|------|--------|
| `DataFrame.isnull() `| 检测缺失值 | 无 | 布尔类型的DataFrame |
| `DataFrame.notnull()` | 检测非缺失值 | 无 | 布尔类型的DataFrame |
| `DataFrame.insert() `| 插入新列 | loc: 位置, column: 列名, value: 值 | None |
| `DataFrame.reindex()` | 重新索引 | labels: 标签, axis: 轴, fill_value: 填充值 | 新的DataFrame |
|` pd.concat()` | 连接DataFrame | objs: 对象列表, axis: 轴, join: 连接方式 | 新的DataFrame |
|` pd.merge() `| 合并DataFrame | left: 左表, right: 右表, how: 合并方式, on: 连接键 | 新的DataFrame |
| `DataFrame.drop()` | 删除行列 | labels: 标签, axis: 轴, inplace: 是否原地操作 | 新的DataFrame或None |
| `DataFrame.dropna()` | 删除缺失值 | axis: 轴, how: 方式, thresh: 阈值 | 新的DataFrame |
| `DataFrame.fillna()` | 填充缺失值 | value: 值, method: 方法, axis: 轴 | 新的DataFrame |
| `DataFrame.info() `| 打印摘要 | verbose: 是否详细, show_counts: 是否显示计数 | None |
| `DataFrame.describe()` | 描述性统计 | percentiles: 百分位数, include: 包含类型 | 统计信息DataFrame |
| `DataFrame.count()` | 非缺失值数量 | axis: 轴 | Series或DataFrame |
|` DataFrame.max()` | 最大值 | axis: 轴 | Series或DataFrame |
|` DataFrame.min()` | 最小值 | axis: 轴 | Series或DataFrame |
|` DataFrame.mean(`) | 平均值 | axis: 轴 | Series或DataFrame |
|` DataFrame.var()` | 方差 | axis: 轴 | Series或DataFrame |
|` DataFrame.std()` | 标准差 | axis: 轴 | Series或DataFrame |
| `DataFrame.sample()` | 随机采样 | n: 数量, frac: 比例, replace: 是否有放回 | 新的DataFrame |
| `DataFrame.drop_duplicates()` | 去重 | subset: 列, keep: 保留方式, inplace: 是否原地操作 | 新的DataFrame |
| `DataFrame.sort_values()` | 排序 | by: 排序列, axis: 轴, ascending: 是否升序 | 新的DataFrame |
| `DataFrame.apply()` | 应用函数 | func: 函数, axis: 轴 | Series或DataFrame |
|` DataFrame.groupby()` | 分组 | by: 分组键, as_index: 是否作为索引 | DataFrameGroupBy对象 |

**示例代码**：
```python
import pandas as pd
import numpy as np

# DataFrame.isnull() / DataFrame.notnull()
d = [[8, np.nan],
     [np.nan, 7],
     [0, 2],
     [np.nan, np.nan]]

df = pd.DataFrame(data=d)
print(df)
# 输出:
#      0    1
# 0  8.0  NaN
# 1  NaN  7.0
# 2  0.0  2.0
# 3  NaN  NaN

print(df.isnull())
# 输出:
#       0      1
# 0  False   True
# 1   True  False
# 2 False  False
# 3   True   True

print(df.notnull())
# 输出:
#       0      1
# 0  True  False
# 1 False   True
# 2  True   True
# 3 False  False

# DataFrame.insert(loc, column, value)
d = {'name': ['Tom', 'Bob', 'Linda'], 'age': [17, 18, 26]}
df = pd.DataFrame(data=d, index=['p1', 'p2', 'p3'])
print(df)
# 输出:
#      name  age
# p1   Tom   17
# p2   Bob   18
# p3 Linda   26

df.insert(2, 'weight', [65, 75, 60])  # 在位置2插入新列'weight'
print(df)
# 输出:
#      name  age  weight
# p1   Tom   17      65
# p2   Bob   18      75
# p3 Linda   26      60

# DataFrame.reindex()
data = np.arange(12).reshape(3, 4)
df = pd.DataFrame(data, index=['n1', 'n2', 'n3'], columns=['a', 'b', 'c', 'd'])
print(df)
# 输出:
#      a   b   c   d
# n1   0   1   2   3
# n2   4   5   6   7
# n3   8   9  10  11

df2 = df.reindex(index=['n2', 'n1', 'n4'], fill_value=np.pi)
print(df2)
# 输出:
#       a   b   c   d
# n2  4.0 5.0 6.0 7.0
# n1  0.0 1.0 2.0 3.0
# n4  NaN NaN NaN NaN -> 3.141593 3.141593 3.141593 3.141593

# pd.concat()
df = pd.DataFrame([[1, 2], [3, 4]], index=['p1', 'p2'], columns=list('AB'))
df2 = pd.DataFrame([[5, 6], [7, 8]], columns=list('AC'))
print(pd.concat([df, df2]))      # 默认按行拼接，外连接
print(pd.concat([df, df2], join='inner'))  # 内连接，只保留共有列
print(pd.concat([df, df2], axis=1))  # 按列拼接

# pd.merge()
d1 = {'name': ['Tom', 'Bob', 'Jack'], 'age': [18, 17, 19], 'weight': [65, 66, 67]}
df1 = pd.DataFrame(data=d1)
d2 = {'name': ['Tom', 'Jack'], 'height': [168, 187], 'weight': [65, 68]}
df2 = pd.DataFrame(data=d2)

print(pd.merge(df1, df2, how='inner', on='name'))  # 内连接
print(pd.merge(df1, df2, how='left', on='name'))   # 左连接
print(pd.merge(df1, df2, how='right', on='name'))  # 右连接
print(pd.merge(df1, df2, how='outer', on='name'))  # 外连接

# DataFrame.drop()
df = pd.DataFrame([[1, 2], [3, 4], [5, 6]], index=['n1', 'n2', 'n3'], columns=['a','b'])
print(df.drop(index='n2'))  # 删除行
print(df.drop(columns='b'))  # 删除列

# DataFrame.dropna()
d = {'name': ['Tom', np.nan, 'Bob'], 'age': [np.nan, np.nan, 19], 'height': [177, 182, 179]}
df = pd.DataFrame(data=d)
print(df.dropna())  # 删除包含缺失值的行
print(df.dropna(axis=1))  # 删除包含缺失值的列
print(df.dropna(axis=1, how='all'))  # 删除全部是缺失值的列
print(df.dropna(axis=1, thresh=2))  # 只保留至少2个非NaN值的列

# DataFrame.fillna()
df = pd.DataFrame([[np.nan, 2, np.nan, 0],
                    [3, 4, np.nan, 1],
                    [np.nan, np.nan, np.nan, np.nan],
                    [np.nan, 3, np.nan, 4]],
                    columns=list("ABCD"))
print(df.fillna(0))  # 用0填充所有NaN
print(df.fillna({'A': 6, 'B': 7}))  # 用字典填充不同列
print(df.fillna(method='ffill'))  # 用前一个非缺失值填充
print(df.fillna(method='bfill'))  # 用后一个非缺失值填充

# DataFrame.info()
df = pd.DataFrame(data={'name': ['Tom', 'Bob', np.nan], 'age': [18, 19, 17], 'height': [167, 177, 178]}, index=['n1', 'n2', 'n3'])
df.info()  # 打印完整摘要
df.info(verbose=False)  # 打印简短摘要

# DataFrame.describe()
df = pd.DataFrame(data={'name': ['Tom', 'Bob', 'Bob'], 'age': [18, 19, 17], 'height': [167, 177, 178]}, index=['n1', 'n2', 'n3'])
print(df.describe())  # 默认只对数值列进行统计
print(df.describe(include='all'))  # 对所有列进行统计
print(df.describe(include='object'))  # 只对字符串列进行统计

# DataFrame.count(), max(), min(), mean(), var(), std()
df = pd.DataFrame(data={'name': ['Tom', np.nan, 'Linda'], 'age': [18, 19, 17]}, index=['n1', 'n2', 'n3'])
print(df.count())  # 非缺失值数量
d = np.random.normal(size=(7, 2))
df = pd.DataFrame(data=d)
print(df.max(axis=0))  # 每列最大值
print(df.min(axis=1))  # 每行最小值
print(df.mean(axis=0))  # 每列平均值
print(df.var(axis=1))   # 每行方差
print(df.std(axis=0))   # 每列标准差

# DataFrame.sample()
df = pd.DataFrame(data={'name': ['Tom', 'Bob', 'Jack', 'Linda'], 'age': [18, 19, 17, 21], 'height': [167, 177, 178, 188]}, index=['n1', 'n2', 'n3', 'n4'])
print(df.sample())  # 随机采样一行
print(df.sample(n=2))  # 随机采样两行
print(df.sample(frac=0.75))  # 随机采样75%的数据
print(df.sample(n=2, replace=True))  # 随机有放回采样两行
print(df.sample(n=2, axis=1))  # 随机采样两列
print(df.sample(n=2, random_state=3))  # 随机采样两行，设置随机数种子

# DataFrame.drop_duplicates()
d = {'A': [1, 3, 3, 1], 'B':[0, 2, 5, 0], 'C': [4, 0, 4, 4], 'D':[1, 0, 0, 1]}
df = pd.DataFrame(data=d)
print(df.drop_duplicates())  # 删除重复行，保留第一次出现的
print(df.drop_duplicates(keep='last'))  # 删除重复行，保留最后一次出现的
print(df.drop_duplicates(keep=False))  # 删除重复行，全部删除
print(df.drop_duplicates(subset=['A', 'D'], keep='last'))  # 基于特定列去重

# DataFrame.sort_values()
df = pd.DataFrame({'col1': [4, 1, 2, np.nan, 5, 2],
                   'col2': [2, 1, 9, 8, 7, 6],
                   'col3': [0, 1, 9, 4, 2, 3],
                   'col4': ['a', 'B', 'c', 'D', 'e', 1]})
print(df.sort_values(by=['col1']))  # 按col1升序排序
print(df.sort_values(by=['col1', 'col2']))  # 先按col1排序，col1相同的按col2排序
print(df.sort_values(by=5, axis=1))  # 按行标签为5的行排序列
print(df.sort_values(['col1', 'col2'], ascending=[True, False]))  # col1升序，col2降序
print(df.sort_values(by='col1', na_position='first'))  # 缺失值排在前面

# DataFrame.apply()
d = [[1, 2, 0], [4, 1, 9], [2, 5, 7], [4, 3, 6]]
df = pd.DataFrame(d, columns=['A', 'B', 'C'])
print(df.apply(np.sum))  # 对每列求和
print(df.apply(np.sum, axis=1))  # 对每行求和

# DataFrame.groupby()
d = {
    'company': ['A', 'B', 'A', 'C', 'C', 'B', 'C', 'A'],
    'salary': [8, 15, 10, 15, np.nan, 28, 30, 15],
    'age': [26, 29, 26, 30, 50, 30, 30, 35]
}
df = pd.DataFrame(data=d)
df_gb = df.groupby(by='company', as_index=False)

# 遍历分组
for g, data in df_gb:
    print(f"Group: {g}")
    print(data)

# 分组属性
print(df_gb.ngroups)  # 分成了几组
print(df_gb.groups)   # 各个分组的index
print(df_gb.indices)  # 各个分组的index

# 获取指定组的数据
print(df_gb.get_group('A'))  # 获取company='A'的数据

# 聚合操作
print(df_gb.agg('mean'))    # 各组平均值
print(df_gb.agg(np.mean))   # 各组平均值
print(df_gb.agg('max'))     # 各组最大值
print(df_gb.agg('min'))     # 各组最小值
print(df_gb.agg('sum'))     # 各组总和
print(df_gb.agg('median'))  # 各组中位数
print(df_gb.agg('std'))     # 各组标准差
print(df_gb.agg('var'))     # 各组方差
print(df_gb.agg('count'))   # 各组计数

# 变换操作
print(df_gb.transform('mean'))  # 在聚合操作的结果之上, 还将值变换到分组前的对应位置上
df[['avg_salary', 'avg_age']] = df_gb.transform('mean')  # 新增两列数据
print(df)
```

#### DataFrame 的运算

**核心知识点**：DataFrame 保留了 NumPy 中的数组运算，且 DataFrame 进行数组运算的时候，索引与值之间的映射关系不会发生改变。在进行 DataFrame 和 DataFrame 的运算时，把两个 DataFrame 中行索引名和列索引名一样的值进行运算，其他不一样的做并集且对应的值为`NaN`

**示例代码**：
```python
import pandas as pd
import numpy as np

# DataFrame与标量的运算
d = np.arange(9).reshape((3, 3))
df1 = pd.DataFrame(data=d, columns=list('abc'), index=['n1', 'n2', 'n3'])
print(df1)
# 输出:
#      a  b  c
# n1  0  1  2
# n2  3  4  5
# n3  6  7  8

print(df1 + 1)  # 每个元素加1
print(df1 - 1)  # 每个元素减1
print(df1 * 2)  # 每个元素乘2
print(df1 / 2)  # 每个元素除2

# DataFrame与DataFrame的运算
d = np.arange(16).reshape((4, 4))
df2 = pd.DataFrame(data=d, columns=list('dacf'),index=['n1', 'n2', 'n3', 'n4'])
print(df2)
# 输出:
#      d   a   c   f
# n1   0   1   2   3
# n2   4   5   6   7
# n3   8   9  10  11
# n4  12  13  14  15

print(df1 + df2)  # 只有索引名和列名都相同的元素会相加，其他为NaN
print(df1 - df2)  # 只有索引名和列名都相同的元素会相减，其他为NaN
print(df1 * df2)  # 只有索引名和列名都相同的元素会相乘，其他为NaN
print(df1 / df2)  # 只有索引名和列名都相同的元素会相除，其他为NaN
```

### 数据清洗与处理

#### 缺失值处理

**核心知识点**：检测、删除和填充缺失值

**函数列表**：

| 函数 | 作用 | 参数 | 返回值 |
|-----|------|------|--------|
| `DataFrame.isnull() `| 检测缺失值 | 无 | 布尔DataFrame |
| `DataFrame.notnull()` | 检测非缺失值 | 无 | 布尔DataFrame |
| `DataFrame.dropna() `| 删除缺失值 | `axis`: 轴, `how`: 方式,`thresh`: 阈值, `subset`: 子集, `inplace`: 是否原地操作 | 新DataFrame或None |
| `DataFrame.fillna() `| 填充缺失值 | `valu`e: 值, `method`: 方法, `axis`: 轴, `inplace`: 是否原地操作, `limit`: 限制 | 新DataFrame或None |

**示例代码**：
```python
import pandas as pd
import numpy as np

# 创建包含缺失值的DataFrame
df = pd.DataFrame([[np.nan, 2, np.nan, 0],
                    [3, 4, np.nan, 1],
                    [np.nan, np.nan, np.nan, np.nan],
                    [np.nan, 3, np.nan, 4]],
                    columns=list("ABCD"))
print(df)
# 输出:
#      A    B   C    D
# 0  NaN  2.0 NaN  0.0
# 1  3.0  4.0 NaN  1.0
# 2  NaN  NaN NaN  NaN
# 3  NaN  3.0 NaN  4.0

# 检测缺失值
print(df.isnull())
# 输出:
#       A      B     C      D
# 0  True  False  True  False
# 1 False  False  True  False
# 2  True   True  True   True
# 3  True  False  True  False

# 删除缺失值
print(df.dropna())  # 删除包含NaN的行
# 输出:
#      A    B   C    D
# 1  3.0  4.0 NaN  1.0

print(df.dropna(axis=1))  # 删除包含NaN的列
# 输出:
#      B    D
# 0  2.0  0.0
# 1  4.0  1.0
# 2  NaN  NaN
# 3  3.0  4.0

print(df.dropna(axis=1, how='all'))  # 删除全部是NaN的列
# 输出:
#      A    B    D
# 0  NaN  2.0  0.0
# 1  3.0  4.0  1.0
# 2  NaN  NaN  NaN
# 3  NaN  3.0  4.0

print(df.dropna(axis=1, thresh=2))  # 只保留至少2个非NaN值的列
# 输出:
#      B    D
# 0  2.0  0.0
# 1  4.0  1.0
# 2  NaN  NaN
# 3  3.0  4.0

# 填充缺失值
print(df.fillna(0))  # 用0填充所有NaN
# 输出:
#      A    B    C    D
# 0  0.0  2.0  0.0  0.0
# 1  3.0  4.0  0.0  1.0
# 2  0.0  0.0  0.0  0.0
# 3  0.0  3.0  0.0  4.0

print(df.fillna({'A': 6, 'B': 7}))  # 用字典填充不同列
# 输出:
#      A    B   C    D
# 0  6.0  2.0 NaN  0.0
# 1  3.0  4.0 NaN  1.0
# 2  6.0  7.0 NaN  NaN
# 3  6.0  3.0 NaN  4.0

print(df.fillna(method='ffill'))  # 用前一个非缺失值填充
print(df.fillna(method='bfill'))  # 用后一个非缺失值填充

print(df.fillna(method='ffill', axis=1))  # 沿行方向用前一个非缺失值填充
print(df.fillna(method='bfill', axis=1))  # 沿行方向用后一个非缺失值填充
```

#### 重复值处理

**核心知识点**：检测和删除重复值

**函数列表**：

| 函数 | 作用 | 参数 | 返回值 |
|-----|------|------|--------|
| `DataFrame.duplicated() `| 检测重复行 | `subset`: 列标签, `keep`: 保留方式 | 布尔Series |
| `DataFrame.drop_duplicates() `| 删除重复行 | `subset`: 列标签, `keep`: 保留方式, `inplace`: 是否原地操作 | 新DataFrame或None |

**示例代码**：
```python
import pandas as pd

# 创建包含重复值的DataFrame
d = {'A': [1, 3, 3, 1], 'B':[0, 2, 5, 0], 'C': [4, 0, 4, 4], 'D':[1, 0, 0, 1]}
df = pd.DataFrame(data=d)
print(df)
# 输出:
#    A  B  C  D
# 0  1  0  4  1
# 1  3  2  0  0
# 2  3  5  4  0
# 3  1  0  4  1

# 检测重复行
print(df.duplicated())  # 检测重复行，保留第一次出现的
# 输出:
# 0    False
# 1    False
# 2    False
# 3     True
# dtype: bool

print(df.duplicated(keep='last'))  # 检测重复行，保留最后一次出现的
# 输出:
# 0     True
# 1    False
# 2    False
# 3    False
# dtype: bool

print(df.duplicated(subset=['A', 'D']))  # 基于特定列检测重复
# 输出:
# 0    False
# 1    False
# 2    False
# 3     True
# dtype: bool

# 删除重复行
print(df.drop_duplicates())  # 删除重复行，保留第一次出现的
# 输出:
#    A  B  C  D
# 0  1  0  4  1
# 1  3  2  0  0
# 2  3  5  4  0

print(df.drop_duplicates(keep='last'))  # 删除重复行，保留最后一次出现的
# 输出:
#    A  B  C  D
# 1  3  2  0  0
# 2  3  5  4  0
# 3  1  0  4  1

print(df.drop_duplicates(keep=False))  # 删除重复行，全部删除
# 输出:
#    A  B  C  D
# 1  3  2  0  0
# 2  3  5  4  0

print(df.drop_duplicates(subset=['A', 'D'], keep='last'))  # 基于特定列去重，保留最后一次出现的
# 输出:
#    A  B  C  D
# 1  3  2  0  0
# 2  3  5  4  0
# 3  1  0  4  1
```

#### 数据排序

**核心知识点**：按值排序和按索引排序

**函数列表**：

| 函数 | 作用 | 参数 | 返回值 |
|-----|------|------|--------|
| `DataFrame.sort_values()` | 按值排序 | `by`: 排序键, `axis`: 轴, `ascending`: 是否升序, `inplace`: 是否原地操作, `na_position`: NaN位置 | 新DataFrame或`None` |
| `DataFrame.sort_index()` | 按索引排序 | `axis`: 轴, `ascending`: 是否升序, `inplace`: 是否原地操作 | 新DataFrame或None |

**示例代码**：
```python
import pandas as pd
import numpy as np

# 创建DataFrame
df = pd.DataFrame({'col1': [4, 1, 2, np.nan, 5, 2],
                   'col2': [2, 1, 9, 8, 7, 6],
                   'col3': [0, 1, 9, 4, 2, 3],
                   'col4': ['a', 'B', 'c', 'D', 'e', 1]})
print(df)
# 输出:
#    col1  col2  col3 col4
# 0   4.0     2     0    a
# 1   1.0     1     1    B
# 2   2.0     9     9    c
# 3   NaN     8     4    D
# 4   5.0     7     2    e
# 5   2.0     6     3    1

# 按值排序
print(df.sort_values(by=['col1']))  # 按col1升序排序
# 输出:
#    col1  col2  col3 col4
# 1   1.0     1     1    B
# 2   2.0     9     9    c
# 5   2.0     6     3    1
# 0   4.0     2     0    a
# 4   5.0     7     2    e
# 3   NaN     8     4    D

print(df.sort_values(by=['col1', 'col2']))  # 先按col1排序，col1相同的按col2排序

print(df.sort_values(by=5, axis=1))  # 按行标签为5的行排序列
print(df.sort_values(['col1', 'col2'], ascending=[True, False]))  # col1升序，col2降序
print(df.sort_values(by='col1', na_position='first'))  # 缺失值排在前面

# 按索引排序
df.index = ['c', 'a', 'b', 'd', 'e', 'f']
print(df.sort_index())  # 按行索引排序
print(df.sort_index(axis=1))  # 按列索引排序
```

### 数据分组与聚合

**核心知识点**：使用`groupby`进行数据分组，然后进行聚合操作

**函数列表**：

| 函数 | 作用 | 参数 | 返回值 |
|-----|------|------|--------|
| `DataFrame.groupby()` | 分组 | `by:` 分组键, `as_index:` 是否作为索引, `sort:` 是否排序, `dropna:` 是否删除NaN | DataFrameGroupBy对象 |
| `GroupBy.agg()` | 聚合 | `func:` 聚合函数或函数列表 | 聚合结果DataFrame |
| `GroupBy.transform()` | 变换 | `func:` 变换函数 | 变换结果DataFrame |
| `GroupBy.filter()` | 过滤 | `func:` 过滤函数 | 过滤结果DataFrame |
| `GroupBy.apply()` | 应用 | `func:` 应用函数 | 应用结果DataFrame |

**示例代码**：
```python
import pandas as pd
import numpy as np

# 创建示例DataFrame
d = {
    'company': ['A', 'B', 'A', 'C', 'C', 'B', 'C', 'A'],
    'salary': [8, 15, 10, 15, np.nan, 28, 30, 15],
    'age': [26, 29, 26, 30, 50, 30, 30, 35]
}
df = pd.DataFrame(data=d)
print(df)
# 输出:
#   company  salary  age
# 0       A     8.0   26
# 1       B    15.0   29
# 2       A    10.0   26
# 3       C    15.0   30
# 4       C     NaN   50
# 5       B    28.0   30
# 6       C    30.0   30
# 7       A    15.0   35

# 分组
df_gb = df.groupby(by='company', as_index=False)

# 遍历分组
for g, data in df_gb:
    print(f"Group: {g}")
    print(data)
    print("-" * 20)

# GroupBy属性
print(df_gb.ngroups)  # 输出: 3 (分成3组)
print(df_gb.groups)   # 各个分组的index
print(df_gb.indices)  # 各个分组的index

# 获取指定组的数据
print(df_gb.get_group('A'))  # 获取company='A'的数据
# 输出:
#   company  salary  age
# 0       A     8.0   26
# 2       A    10.0   26
# 7       A    15.0   35

# 聚合操作
print(df_gb.agg('mean'))  # 各组平均值
# 输出:
#   company     salary        age
# 0       A  11.000000  29.000000
# 1       B  21.500000  29.500000
# 2       C  22.500000  36.666667

print(df_gb.agg('max'))  # 各组最大值
print(df_gb.agg('min'))  # 各组最小值
print(df_gb.agg('sum'))  # 各组总和
print(df_gb.agg('median'))  # 各组中位数
print(df_gb.agg('std'))  # 各组标准差
print(df_gb.agg('var'))  # 各组方差
print(df_gb.agg('count'))  # 各组计数

# 使用多个聚合函数
print(df_gb.agg(['mean', 'max', 'min']))  # 对每列使用多个聚合函数
print(df_gb.agg({'salary': 'mean', 'age': 'max'}))  # 对不同列使用不同聚合函数

# 变换操作
print(df_gb.transform('mean'))  # 在聚合操作的结果之上, 还将值变换到分组前的对应位置上
# 新增两列数据
df[['avg_salary', 'avg_age']] = df_gb.transform('mean')
print(df)
# 输出:
#   company  salary  age  avg_salary  avg_age
# 0       A     8.0   26        11.0    29.0
# 1       B    15.0   29        21.5    29.5
# 2       A    10.0   26        11.0    29.0
# 3       C    15.0   30        22.5    36.666667
# 4       C     NaN   50        22.5    36.666667
# 5       B    28.0   30        21.5    29.5
# 6       C    30.0   30        22.5    36.666667
# 7       A    15.0   35        11.0    29.0

# 过滤操作
# 保留组内平均age大于30的组
print(df_gb.filter(lambda x: x['age'].mean() > 30))
# 输出:
#   company  salary  age  avg_salary     avg_age
# 3       C    15.0   30        22.5  36.666667
# 4       C     NaN   50        22.5  36.666667
# 6       C    30.0   30        22.5  36.666667

# 应用操作
# 对每个组进行特定操作
def top_salary(df, n=3):
    return df.sort_values(by='salary', ascending=False).head(n)

print(df_gb.apply(top_salary))  # 对每个组获取salary最高的n条记录
```

### 文件读写操作

#### CSV文件读写

**函数列表**：

| 函数 | 作用 | 主要参数 | 返回值 |
|-----|------|----------|--------|
| `pd.read_csv()` | 读取CSV文件 | `filepath`: 文件路径, `sep`: 分隔符, `header`: 表头行, `names`: 列名, `index_col`: 索引列, `usecols`: 使用列,`nrows`: 读取行数, `skiprows`: 跳过行数 | DataFrame |
| `DataFrame.to_csv()` | 写入CSV文件 | `path_or_buf`: 文件路径, `sep`: 分隔符, `header`: 是否写表头, `index`: 是否写索引, `columns`: 写入列 | None |

**示例代码**：
```python
import pandas as pd
import numpy as np

# 读取CSV文件
df = pd.read_csv('./test01.csv')  # 读取默认分隔符(逗号)的CSV文件
print(df)

# 读取不同分隔符的CSV文件
df = pd.read_csv('./test02.csv', sep=';')  # 读取分号分隔的CSV文件
print(df)

# 处理表头
df = pd.read_csv('./test03.csv', sep=';', header=None)  # 不使用数据为表头，列索引为0,1,2...
print(df)

df = pd.read_csv('./test03.csv', sep=';', header=2)  # 指定第3行数据作为表头
print(df)

df = pd.read_csv('./test02.csv', sep=';', names=['name', 'age', 'height'])  # 指定列名
print(df)

# 读取部分数据
df = pd.read_csv('./test01.csv', nrows=2)  # 只读取前2行
print(df)

df = pd.read_csv('./test01.csv', skiprows=2)  # 跳过前2行
print(df)

df = pd.read_csv('./test01.csv', skiprows=[0, 2])  # 跳过第1行和第3行
print(df)

df = pd.read_csv('./test01.csv', usecols=[0, 2])  # 只读取第1列和第3列
print(df)

# 分块读取大文件
obj = pd.read_csv('./test01.csv', chunksize=2)  # 每次读取2行
for i in obj:
    print(i)

# 写入CSV文件
d = {
    '名字': ['张三', '李四', '王五', '赵六', '孙七'],
    '年龄': [18, 19, 20, 22, 17],
    '身高': [188, 178, 189, 175, 177]
}
df = pd.DataFrame(data=d)
print(df)

# 写入CSV文件
df.to_csv('./test04.csv')  # 使用默认参数

df.to_csv('./test05.csv', sep=';')  # 使用分号作为分隔符

df.to_csv('./test06.csv', index=False)  # 不写入行索引

df.to_csv('./test07.csv', header=False)  # 不写入列索引
```

#### EXCEL文件读写

**函数列表**：

| 函数 | 作用 | 主要参数 | 返回值 |
|-----|------|----------|--------|
| `pd.read_excel()` | 读取Excel文件 | `io`: 文件路径, `sheet_name`: 工作表名或索引, `header`: 表头行, `names`: 列名, `index_col`: 索引列, `usecols`: 使用列, `nrows`: 读取行数, `skiprows`: 跳过行数 | DataFrame |
| `DataFrame.to_excel()` | 写入Excel文件 | `excel_writer`: 文件路径或ExcelWriter对象, `sheet_name`: 工作表名, `header`: 是否写表头, `index`: 是否写索引, `columns`: 写入列 | None |

**示例代码**：
```python
import pandas as pd

# 写入Excel文件
d = {
    '名字': ['张三', '李四', '王五', '赵六', '孙七'],
    '年龄': [18, 19, 20, 22, 17],
    '身高': [188, 178, 189, 175, 177]
}
df = pd.DataFrame(data=d)
print(df)

# 写入单个工作表
df.to_excel('./test08.xlsx')  # 使用默认参数

df.to_excel('./test09.xlsx', index=False)  # 不写入行索引

df.to_excel('./test10.xlsx', header=False)  # 不写入列索引

# 写入多个工作表
writer = pd.ExcelWriter('./test11.xlsx')
df.to_excel(writer, sheet_name='工作表1', index=False)
df.iloc[:, :2].to_excel(writer, sheet_name='工作表2', index=False)
writer.close()

# 使用with语句更优雅
with pd.ExcelWriter('./test12.xlsx') as writer:
    df.to_excel(writer, sheet_name='工作表1', index=False)
    df.iloc[:, :2].to_excel(writer, sheet_name='工作表2', index=False)

# 读取Excel文件
df = pd.read_excel('./test11.xlsx')  # 默认读取第一个工作表
print(df)

df = pd.read_excel('./test11.xlsx', header=None)  # 不使用数据为表头
print(df)

df = pd.read_excel('./test11.xlsx', header=2)  # 指定第3行数据作为表头
print(df)

df = pd.read_excel('./test11.xlsx', names=['name', 'age', 'height'])  # 指定列名
print(df)

df = pd.read_excel('./test11.xlsx', header=None, names=['name', 'age', 'height'])  # 指定列名，header=None
print(df)

# 读取指定工作表
df = pd.read_excel('./test11.xlsx', sheet_name=1)  # 读取第2个工作表（索引为1）
print(df)

df = pd.read_excel('./test11.xlsx', sheet_name='工作表2')  # 按名称读取工作表
print(df)

df = pd.read_excel('./test11.xlsx', sheet_name=[0, '工作表2'])  # 读取多个工作表，返回字典
print(df)  # 字典的键为工作表名或索引，值为对应的DataFrame

# 读取部分数据
df = pd.read_excel('./test11.xlsx', nrows=2)  # 只读取前2行
print(df)

df = pd.read_excel('./test11.xlsx', skiprows=2)  # 跳过前2行
print(df)

df = pd.read_excel('./test11.xlsx', skiprows=[0, 2])  # 跳过第1行和第3行
print(df)

df = pd.read_excel('./test11.xlsx', usecols=[0, 2])  # 只读取第1列和第3列
print(df)
```

