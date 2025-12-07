# 数据可视化

## Matplotlib简介

Matplotlib 是 Python 常用的第三方 2D 绘图库，是 Python 中最受欢迎的数据可视化软件包之一。它提供了丰富的绘图功能，可以绘制各种高质量的图表，是数据分析和科学计算中不可或缺的工具。

**核心知识点**：
- Matplotlib 是一个用于创建出版质量图表的库
- 它可以与 NumPy、Pandas 等数据处理库无缝集成
- 提供了面向对象的 API 和简单的 pyplot 接口

---

## 基础绘图

### `plt.plot()`函数

**作用**：绘制折线图，是 Matplotlib 中最基础和常用的绘图函数

**语法**：`plt.plot(x, y, color, linestyle, linewidth, marker, markerfacecolor, markersize, label)`

**参数说明**：
- `x`：x轴数据
- `y`：y轴数据
- `color`：线的颜色
- `linestyle`：线条样式
- `linewidth`：线条宽度（用数字表示大小）
- `marker`：标记的样式
- `markerfacecolor`：标记填充颜色
- `markersize`：标记尺寸（用数字表示大小）
- `label`：线条的标签（后文结合 `legend()` 创建图例来讲）

**颜色参数(color)**：

查阅[颜色参数表](https://www.rapidtables.org/zh-CN/web/color/RGB_Color.html)


**线条样式(linestyle)**：

| 样式一 | 样式二    | 说明         |
| ------ | --------- | ------------ |
| `'-' `   | `'solid'  ` | 实线         |
| `'--'`   | `'dashed' ` | 虚线         |
| `'-.'`   | `'dashdot'` | 点画线       |
| `':' `   | `'dotted' ` | 点线         |
| `' '`    | `'None'   ` | 不显示线条了 |

**标记样式(ma`rker)**：`

| 样式 | 说明               | 样式     | 说明         |
| ---- | ------------------ | -------- | ------------ |
| `'.' ` | 点标记             |` '1' `     | 下花三角标记 |
| `',' ` | 像素标记（极小点） |` '2' `     | 上花三角标记 |
| `'o' ` | 实心圆标记         |` '3' `     | 左花三角标记 |
| `'v' ` | 倒三角标记         |` '4' `     | 右花三角标记 |
| `'^' ` | 上三角标记         |` 's' `     | 实心方形标记 |
| `'>' ` | 右三角标记         |` 'p' `     | 实心五角标记 |
| `'<' ` | 左三角标记         |` '*' `     | 星形标记     |
| `'h' ` | 竖六边形标记       |` 'd' `     | 瘦菱形标记   |
| `'H' ` | 横六边形标记       |` 'D' `     | 菱形标记     |
| `'+' ` | 十字标记           |`'\|' `     | 垂直线标记   |
| `'x' ` | x标记              |` '_' `     | 水平线标记   |

**示例代码**：
```python
import matplotlib.pyplot as plt
import numpy as np

# 创建数据
x = np.linspace(-3, 3, 50)  # 生成-3到3之间等间距的50个点
y = np.sin(x)  # 计算正弦值

# 绘制图形，设置多种样式参数
plt.plot(x, y, 
         color='skyblue',       # 线条颜色为天蓝色
         linestyle='-.',        # 线条样式为点画线
         linewidth=2,           # 线条宽度为2
         marker='h',            # 标记样式为竖六边形
         markerfacecolor='gold',# 标记填充颜色为金色
         markersize=8)          # 标记大小为8

plt.show()  # 显示图形
```

**代码说明**：
这个例子展示了如何使用`plt.plot()`函数绘制正弦函数图像，并自定义了线条和标记的各种样式属性。通过调整这些参数，可以创建出视觉效果丰富的图表。

### `plt.figure()`创建画布

**核心知识点**：想要绘图，必须先要创建一个 `figure`（画布），还要有 `axes`（坐标系）

#### 隐式创建 figure

**原理说明**：
- 当第一次执行 `plt.xxx()` 画图代码时，会自动判断是否已经存在画布
- 如果没有，则自动创建一个画布，并且在该画布上自动创建一个坐标系
- 如果不设置画布，一个画布上只有一个坐标系

**示例代码**：
```python
import matplotlib.pyplot as plt
import numpy as np

# 创建数据
x = np.linspace(-3, 3, 50)
y1 = 2*x + 2  # 直线
y2 = x**2     # 抛物线
y3 = np.sin(x) # 正弦曲线

# 隐式创建画布（自动创建）
plt.plot(x, y1, color='gold')  # 第一条线
plt.plot(x, y2, color='red')   # 第二条线
plt.plot(x, y3, color='green') # 第三条线

plt.show()  # 显示图形
```

**代码说明**：
在这个例子中，我们没有显式创建画布，Matplotlib会自动创建一个默认的画布，并在上面绘制三条曲线。这是最简单的绘图方式，适合快速绘制简单图形。

#### 显式创建 `figure`

**原理说明**：
- 利用 `plt.figure()` 手动创建画布，可以创建多个画布
- 在 `plt.figure()` 下面的 `plt.xxx()` 画图代码都会画在对应的画布上面

**示例代码**：
```python
import matplotlib.pyplot as plt
import numpy as np

# 创建数据
x = np.linspace(-3, 3, 50)
y1 = 2*x + 2  # 直线
y2 = x**2     # 抛物线
y3 = np.sin(x) # 正弦曲线

# 创建第一个画布
plt.figure()  
plt.plot(x, y1, color='gold')  # 在第一个画布上绘制直线
plt.plot(x, y2, color='red')   # 在第一个画布上绘制抛物线

# 创建第二个画布
plt.figure()  
plt.plot(x, y3, color='green') # 在第二个画布上绘制正弦曲线

plt.show()  # 显示所有画布
```

**代码说明**：
这个例子展示了如何显式创建多个画布，并在不同的画布上绘制不同的图形。这种方式允许我们更好地控制图形的组织结构。

#### plt.figure()参数详解

**语法**：`plt.figure(num=None, figsize=None, dpi=None, facecolor=None)`

**参数说明**：
- `num`：画布编号或名称（数字为编号，字符串为名称），不指定则按照数字顺序
- `figsize`：指定 figure 的宽和高（英寸），默认为（6.4, 4.8）
- `dpi`：指定画布显示的分辨率（用数字表示大小，默认为100）
- `facecolor`：背景的颜色

**示例代码**：
```python
import matplotlib.pyplot as plt
import numpy as np

# 创建数据
x = np.linspace(-3, 3, 50)
y = np.sin(x) # 正弦曲线

# 创建第一个画布，设置编号、大小、分辨率和背景颜色
plt.figure(num=3, figsize=(7, 3), dpi=72, facecolor="red")
plt.plot(x, y)

# 创建第二个画布，使用名称作为编号，设置不同的样式参数
plt.figure(num="画布二", figsize=(7, 3), dpi=72, facecolor="green")
plt.plot(x, y)

plt.show()  # 显示所有画布
```

**代码说明**：
这个例子展示了如何使用`plt.figure()`的各种参数来自定义画布的属性，包括大小、分辨率和背景颜色等。通过这些参数，可以精确控制图形的外观。

### 中文/负号显示问题

**问题说明**：matplotlib 在使用过程中，可能会遇到中文或者负号显示乱码的问题，这是因为默认字体不支持中文显示，并且负号可能显示为方块。

**解决方案**：把下面代码粘贴到matplotlib 使用的最前面即可

**示例代码**：
```python
import matplotlib.pyplot as plt

# 通过rc参数修改字体为黑体，就可以显示中文
plt.rcParams['font.sans-serif'] = ['SimHei']
# 通过rc参数修改字符显示，就可以正常显示负号
plt.rcParams['axes.unicode_minus'] = False

# 测试中文和负号显示
x = np.linspace(-5, 5, 50)
y = x**2

plt.plot(x, y)
plt.title("测试中文标题：y = x²")  # 中文标题
plt.xlabel("这是x轴")             # 中文x轴标签
plt.ylabel("这是y轴")             # 中文y轴标签
plt.grid(True)                   # 显示网格

plt.show()
```

**代码说明**：
这个例子展示了如何解决matplotlib中中文和负号显示问题。通过设置rcParams，可以指定使用的字体和字符显示方式。注意，SimHei是常见的黑体中文字体，在不同的操作系统中可能需要使用不同的中文字体。

---

## 坐标轴设置

### 设置坐标轴标签

**核心知识点**：可以用 `plt.xlabel()` 和 `plt.ylabel()` 来指定图像横纵坐标轴的标签

**函数列表**：

| 函数 | 作用 | 参数说明 |
|-----|------|----------|
| `plt.xlabel()` | 设置x轴标签 | `xlabel(xlabel, fontdict=None, labelpad=None, **kwargs)` |
| `plt.ylabel()` | 设置y轴标签 | `ylabel(ylabel, fontdict=None, labelpad=None, **kwargs)` |

**参数说明**：
- `xlabel/ylabel`：标签文本内容
- `fontdict`：字体字典，可设置字体大小、颜色等属性
- `labelpad`：标签与轴线的距离
- `**kwargs`：其他文本属性，如fontsize（字体大小）等

**示例代码**：
```python
import matplotlib.pyplot as plt
import numpy as np

# 规避中文或者负号可能显示乱码的问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建数据
x = np.linspace(-3, 3, 50)
y = np.sin(x)

plt.figure()
plt.plot(x, y)

# 指定横轴标签
plt.xlabel("这是x轴")
# 指定纵轴标签，并设置字体大小为14
plt.ylabel("这是y轴", fontsize=14)

plt.title("正弦函数图像")  # 添加标题
plt.grid(True)            # 添加网格线
plt.show()
```

**代码说明**：
这个例子展示了如何设置坐标轴标签，并演示了如何自定义标签的字体大小。坐标轴标签是图表的重要组成部分，可以帮助读者理解数据的含义。

### 设置坐标轴刻度

**函数列表**：

| 函数 | 作用 | 参数说明 |
|-----|------|----------|
| `plt.xticks()` | 设置x轴刻度 | `xticks(ticks=None, labels=None, **kwargs)` |
| `plt.yticks()` | 设置y轴刻度 | `yticks(ticks=None, labels=None, **kwargs)` |

**参数说明**：
- `ticks`：刻度点的位置组成的列表（可以指定为空列表，则去掉刻度，但轴还在）
- `labels`：刻度点的位置上的标签组成的列表（labels不指定，则标签显示ticks值）
- `**kwargs`：其他文本属性，如rotation（旋转角度）等

**示例代码**：
```python
import matplotlib.pyplot as plt
import numpy as np

# 创建数据
x = np.linspace(-3, 3, 50)
y = np.sin(x)

# 第一个图：自定义刻度和标签
plt.figure()  # 画布1
plt.plot(x, y)
# ticks的值作为刻度点的位置, labels的值作为刻度点的位置上的标签
plt.yticks(ticks=[-1, -0.8, -0.5, -0.1, 1], labels=["a", "b", "c", "d", "e"])

# 第二个图：去掉x轴刻度
plt.figure()  # 画布2
plt.plot(x, y)
# ticks指定为空列表, 去掉刻度, 但轴还在
plt.xticks(ticks=[])

# 第三个图：关闭整个坐标体系
plt.figure()  # 画布3
plt.plot(x, y)
plt.axis("off")  # 关闭整个坐标体系

# 第四个图：自定义刻度位置
plt.figure()  # 画布4
plt.plot(x, y)
new_xticks = np.linspace(-4, 4, 9)
# labels参数没有指定，默认和ticks参数取一样的值
plt.xticks(ticks=new_xticks)
plt.yticks(ticks=[-1, -0.8, -0.5, -0.1, 1])

plt.show()  # 显示所有图形
```

**代码说明**：
这个例子展示了多种设置坐标轴刻度的方法，包括自定义刻度位置和标签、去掉刻度、关闭坐标轴等。这些技巧可以帮助我们更好地控制图表的显示效果，使数据呈现更加清晰和有意义。

### 设置坐标边框

**核心知识点**：通过修改坐标轴边框的颜色和显示方式，可以改变图表的视觉效果

**步骤说明**：
1. 用 `plt.gca()` 获取到坐标体系（矩形坐标框）
2. 调用坐标体系的 `spines` 属性，通过 `'top'、'bottom'、'left'、'right'` 参数获得指定的边框
3. 结合 `set_color()` 方法来指定颜色（当颜色指定为 `'None'` 时，表示不显示该边框）

**示例代码**：
```python
import matplotlib.pyplot as plt
import numpy as np

# 创建数据
x = np.linspace(-3, 3, 50)
y = np.sin(x)

plt.figure()
plt.plot(x, y)
plt.yticks(ticks=[-1, -0.8, -0.5, -0.1, 1], labels=["a", "b", "c", "d", "e"])

# 获取当前坐标轴对象
ax = plt.gca()
# 设置边框颜色
ax.spines['right'].set_color('None')   # 隐藏右边框
ax.spines['top'].set_color('None')      # 隐藏上边框
ax.spines['left'].set_color('red')      # 设置左边框为红色
ax.spines['bottom'].set_color('green')  # 设置底边框为绿色

plt.title("自定义坐标边框颜色")
plt.grid(True)  # 添加网格线
plt.show()
```

**代码说明**：
这个例子展示了如何修改坐标轴边框的颜色，以及如何隐藏特定的边框。通过这种方式，可以创建出更加简洁或更加突出的图表样式。

### 移动坐标边框

**核心知识点**：通过移动坐标边框，可以将坐标轴放置在图表的任意位置，创建非标准的图表布局

**步骤说明**：
1. 用 `plt.gca() `获取到坐标体系（矩形坐标框）
2. 调用坐标体系的 `spines` 属性，通过 `'top'、'bottom'、'left'、'right'` 参数获得指定的边框
3. 结合 `set_position()` 方法来指定边框位置

**示例代码**：
```python
import matplotlib.pyplot as plt
import numpy as np

# 创建数据
x = np.linspace(-3, 3, 50)
y = np.sin(x)

plt.figure()
plt.plot(x, y)
plt.yticks(ticks=[-1, -0.8, -0.5, -0.1, 1], labels=['a', 'b', 'c', 'd', 'e'])

# 获取当前坐标轴对象
ax = plt.gca()
# 设置边框颜色和可见性
ax.spines['right'].set_color('None')   # 隐藏右边框
ax.spines['top'].set_color('None')      # 隐藏上边框
ax.spines['left'].set_color('red')      # 设置左边框为红色
ax.spines['bottom'].set_color('green')  # 设置底边框为绿色

# 移动边框位置
# 选择坐标体系的左边框, 设置位置到数据为0的地方(即x轴原点)
ax.spines['left'].set_position(('data', 0))
# 选择坐标体系的底边框, 设置位置到数据为-0.1的地方(即y轴的'd'点)
ax.spines['bottom'].set_position(('data', -0.1))

plt.title("移动坐标边框位置")
plt.grid(True)  # 添加网格线
plt.show()
```

**代码说明**：
这个例子展示了如何移动坐标轴边框的位置，将传统的框式坐标系转换为十字坐标系。这种技术在科学图表中很常见，可以使图表更加简洁和专业。

**注意**：`set_position()`方法接受两种格式的参数：
- `'data'`：表示将边框移动到数据坐标系的特定位置
- `'axes'`：表示将边框移动到轴坐标系的特定位置（0到1之间）

---

## 图表元素

### `plt.legend()`创建图例

**作用**：创建图例，用于标识图表中不同线条或元素的含义

**语法**：`plt.legend(handles, labels, loc, fontsize, edgecolor, facecolor)`

**参数说明**：
- `handles`：控制柄，默认是一个画布中所有线对象组成的列表
- `labels`：图例标签，默认是绘图函数中 label 参数值组成的列表
- `loc`：图例创建的位置（默认是loc="best"，代表自动找最好的位置）
- `fontsize`：图例字体大小（用数字大小表示）
- `edgecolor`：图例边框颜色
- `facecolor`：图例背景颜色

**图例位置(loc)参数选项**：
| 位置字符串 | 位置代码 | 说明 |
|-----------|----------|------|
| `'best'` | 0 | 自动选择最佳位置 |
| `'upper right'` | 1 | 右上角 |
| `'upper left' `| 2 | 左上角 |
| `'lower left' `| 3 | 左下角 |
| `'lower right'` | 4 | 右下角 |
| `'right'` | 5 | 右侧中间 |
| `'center left' `| 6 | 左侧中间 |
| `'center right'` | 7 | 右侧中间 |
| `'lower center'` | 8 | 底部中间 |
| `'upper center'` | 9 | 顶部中间 |
|` 'center'` | 10 | 正中间 |

**示例代码**：
```python
import matplotlib.pyplot as plt
import numpy as np

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建数据
x = np.linspace(-3, 3, 50)
y1 = 2*x + 1  # 直线
y2 = np.sin(x) # 正弦曲线

# 第一个图：使用label参数和自动图例
plt.figure()
plt.plot(x, y1, color='blue', label='直线')
plt.plot(x, y2, color='green', label='曲线')
# 设置图例位置在右下角，字体大小14，显示边框，边框红色，背景黄色
plt.legend(loc='lower right', fontsize=14, frameon=True, 
           edgecolor='red', facecolor='yellow')

# 第二个图：使用labels参数指定图例
plt.figure()
plt.plot(x, y1, color='blue')
plt.plot(x, y2, color='green')
# 指定labels列表
plt.legend(labels=['直线', '曲线'])

# 第三个图：使用handles参数指定图例
plt.figure()
# plot返回Line2D对象, 装在列表里, 所以解包
line1, = plt.plot(x, y1, color='blue', label='直线')
line2, = plt.plot(x, y2, color='green', label='曲线')
print(f"Line1对象: {line1}")
print(f"Line2对象: {line2}")
# 只指定第一条线创建图例, 且图例标签改为'线条1'
plt.legend(handles=[line1, ], labels=['线条1', ])

plt.show()
```

**代码说明**：
这个例子展示了创建图例的三种不同方式：
1. 使用绘图函数中的`label`参数，让`plt.legend()`自动创建图例
2. 使用`labels`参数直接指定图例标签
3. 使用`handles`参数选择特定的线条创建图例

图例是图表中非常重要的元素，它帮助读者理解不同线条或数据系列代表的含义。通过调整图例的位置、样式和内容，可以使图表更加清晰和专业。

### `plt.text()`文字说明

**作用**：在图表的指定位置添加文本说明

**语法**：`plt.text(x, y, s, size, color, ha, va)`

**参数说明**：
- `x`：文字开始的 x 位置（数据坐标）
- `y`：文字开始的 y 位置（数据坐标）
- `s`：需要写入的内容（字符串）
- `size`：文字大小
- `color`：文字颜色
- `ha`：设置水平对齐方式，可选参数：`'left', 'right', 'center'` 等
- `va`：设置垂直对齐方式，可选参数：`'center', 'top', 'bottom'` 等

**文本对齐方式(ha)**：
| 值 | 说明 |
|----|------|
|` 'center'` | 文本以指定的x坐标为中心 |
|` 'left'  ` | 文本的左边与指定的x坐标对齐 |
|` 'right' `| 文本的右边与指定的x坐标对齐 |

**文本对齐方式(va)**：
| 值 | 说明 |
|----|------|
|` 'center'  `| 文本以指定的y坐标为中心 |
|` 'top'     `| 文本的顶部与指定的y坐标对齐 |
|` 'bottom'  `| 文本的底部与指定的y坐标对齐 |
|` 'baseline'` | 文本的基线与指定的y坐标对齐 |

**示例代码**：
```python
import matplotlib.pyplot as plt
import numpy as np

# 创建数据
x = np.linspace(-3, 3, 50)
y = np.sin(x)

plt.figure()
plt.plot(x, y)

# 添加文本标注
plt.text(x=1.1, y=0.6, s="y=sinx", size=16, color="red")
plt.text(x=-2, y=-0.8, s="最小值", size=12, color="blue", 
         ha='center', va='center')
plt.text(x=0, y=1, s="最大值", size=14, color="green", 
         ha='center', va='bottom')

# 添加箭头和文本
plt.annotate('零点', xy=(0, 0), xytext=(1, -0.5),
             arrowprops=dict(facecolor='black', shrink=0.05))

plt.title("函数图像与文字说明")
plt.grid(True)
plt.show()
```

**代码说明**：
这个例子展示了如何在图表中添加文本说明和注释。文本说明可以帮助突出图表中的重要特征，如极值点、特殊点等。通过调整文本的位置、大小、颜色和对齐方式，可以使图表信息更加丰富和清晰。

**注意**：除了`plt.text()`函数，matplotlib还提供了`plt.annotate()`函数，它可以同时添加文本和箭头，更加适合指向图表中的特定点。

---

## 常用图表类型

### `plt.scatter()`散点图

**作用**：绘制散点图，用于展示两个变量之间的关系或数据点的分布

**语法**：`plt.scatter(x, y, s, c, marker, alpha, linewidths, edgecolors)`

**参数说明**：
- `x`：形状为(n,)的数组，绘图的 x 轴数据
- `y`：形状为(n,)的数组，绘图的 y 轴数据
- `s`：标记点的大小（用数字大小表示，可以是标量或数组）
- `c`：标记点的颜色（可以是颜色名称、颜色数组或颜色映射）
- `marker`：标记的样式，默认的是 'o'
- `alpha`：透明度（0-1之间）
- `linewidths`：标记点边框线的宽度
- `edgecolors`：标记点的边框线颜色

**示例代码**：
```python
import matplotlib.pyplot as plt
import numpy as np

# 从正态分布中抽取100个随机样本
x1 = np.random.normal(0, 1, 100)  # 第一组x数据
y1 = np.random.normal(0, 1, 100)  # 第一组y数据
x2 = np.random.normal(0, 1, 100)  # 第二组x数据
y2 = np.random.normal(0, 1, 100)  # 第二组y数据

# 绘制第一组散点
plt.scatter(x1, y1, 
           s=90,                    # 点的大小
           c='green',               # 点的颜色
           marker='D',               # 点的形状（菱形）
           alpha=0.2,                # 透明度
           linewidths=2,             # 边框线宽度
           edgecolors='red')         # 边框线颜色

# 绘制第二组散点
plt.scatter(x2, y2, 
           s=90,                    # 点的大小
           c='yellow',              # 点的颜色
           marker='D',               # 点的形状（菱形）
           alpha=0.8,                # 透明度
           linewidths=2,             # 边框线宽度
           edgecolors='black')       # 边框线颜色

plt.title("两组数据的散点图")
plt.xlabel("X值")
plt.ylabel("Y值")
plt.grid(True)
plt.show()
```

**代码说明**：
这个例子展示了如何使用`plt.scatter()`函数绘制两组数据的散点图。散点图非常适合用来观察两个变量之间的相关性、聚类情况或异常值。通过调整点的大小、颜色、透明度等参数，可以突出显示数据的特定特征。

### plt.bar()条形图

**作用**：绘制条形图，用于比较不同类别的数据大小

**语法**：`plt.bar(x, height, width, color, edgecolor, alpha, linewidth, bottom, align)`

**参数说明**：
- `x`：x 坐标（条形的位置）
- `height`：条形的高度
- `width`：条形的宽度，默认是0.8
- `color`：条形的颜色
- `edgecolor`：条形边框的颜色
- `alpha`：颜色的透明度（0~1）
- `linewidth`：条形边框的宽度
- `bottom`：条形底边的起始位置（即y轴的起始坐标）
- `align`：条形的对齐方式（默认为 "center"，表示刻度和条形中心对齐，"edge" 表示刻度和条形左边对齐）

**示例代码**：
```python
import matplotlib.pyplot as plt
import numpy as np

# 第一个图：上下条形图
plt.figure(figsize=(10, 6))
x = np.arange(1, 11)
h1 = np.random.randint(20, 35, 10)  # 随机生成10个20-35之间的整数
h2 = np.random.randint(15, 40, 10)  # 随机生成10个15-40之间的整数

# 绘制向上的条形
plt.bar(x, +h1, bottom=0.5, color='skyblue', edgecolor='navy', alpha=0.8)
# 绘制向下的条形
plt.bar(x, -h2, bottom=-0.5, color='lightcoral', edgecolor='darkred', alpha=0.8)

# 添加数值标签
for i in range(len(x)):
    plt.text(x[i], h1[i]+0.5, h1[i], ha='center')  # 上方条形标签
    plt.text(x[i], -h2[i]-0.5, h2[i], va='top', ha='center')  # 下方条形标签

# 设置y轴刻度标签
plt.yticks(ticks=range(-40, 31, 10), labels=[40, 30, 20, 10, 0, 10, 20, 30])
plt.title("上下条形图")
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# 第二个图：并排条形图
plt.figure(figsize=(10, 6))
x = np.arange(1, 25, 2.5)  # 条形位置
h1 = np.random.randint(20, 35, 10)  # 第一组数据
h2 = np.random.randint(15, 40, 10)  # 第二组数据

# 绘制第一组条形（靠右对齐）
plt.bar(x, h1, width=1.0, align='edge', color='lightgreen', 
        edgecolor='darkgreen', alpha=0.8, label='组1')
# 绘制第二组条形（靠左对齐，偏移一定位置）
plt.bar(x-0.8, h2, width=1.0, color='lightyellow', 
        edgecolor='orange', alpha=0.8, label='组2')

# 添加数值标签
for i in range(len(x)):
    plt.text(x[i]+0.2, h1[i], h1[i], size=8, ha='center')  # 第一组标签
    plt.text(x[i]-0.8, h2[i], h2[i], size=8, ha='center')  # 第二组标签

# 设置x轴刻度
plt.xticks(ticks=x-0.4, labels=[f'项目{i+1}' for i in range(10)])
plt.title("并排条形图")
plt.legend()
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()  # 自动调整子图参数
plt.show()
```

**代码说明**：
这个例子展示了两种常见的条形图类型：上下条形图（适合展示正负值对比）和并排条形图（适合比较不同组别的数据）。条形图是展示分类数据最常用的图表类型之一，通过调整条形的位置、宽度、颜色等属性，可以创建出清晰直观的比较图表。

### plt.imshow()数据转图像

**作用**：将数组数据转换为图像显示，适合显示二维数组、矩阵或图像数据

**语法**：`plt.imshow(X, cmap, alpha)`

**参数说明**：
- `X`：图像数据（可以是二维的，比如灰度图；也可以是三维的，比如RGB图）
- `cmap`：颜色映射（colormap），默认是彩色的，其他选项如："Greys"（灰度图）、"viridis"、"plasma"等
- `alpha`：透明度（0~1）

**常见颜色映射(cmap)选项**：
| cmap名称 | 描述 |
|---------|------|
| 'viridis' | 从深蓝到绿的渐变（色盲友好） |
| 'plasma' | 从深紫到黄的渐变 |
| 'inferno' | 从黑到黄的渐变 |
| 'magma' | 从黑到紫的渐变 |
| 'Greys' | 灰度图 |
| 'Blues' | 蓝色系渐变 |
| 'Reds' | 红色系渐变 |
| 'Greens' | 绿色系渐变 |

**示例代码**：
```python
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(12, 8))

# 第一个图：自定义数据图像
plt.subplot(2, 3, 1)
# 自己造一个shape为(3, 3)的图像数据
data = np.array([[0, 50, 200], 
                 [200, 100, 0], 
                 [0, 150, 200]])
plt.imshow(data)
plt.title("彩色图像")
plt.colorbar()  # 添加颜色条

# 第二个图：灰度图像
plt.subplot(2, 3, 2)
plt.imshow(data, cmap="Greys")
plt.title("灰度图像")
plt.colorbar()

# 第三个图：透明度设置
plt.subplot(2, 3, 3)
plt.imshow(data, cmap="Greys", alpha=0.5)
plt.title("半透明图像")
plt.colorbar()

# 第四个图：随机数据矩阵
plt.subplot(2, 3, 4)
random_data = np.random.rand(10, 10)  # 10x10的随机矩阵
plt.imshow(random_data, cmap="viridis")
plt.title("随机矩阵")
plt.colorbar()

# 第五个图：热力图示例
plt.subplot(2, 3, 5)
# 创建一个有规律的热力图数据
heatmap_data = np.outer(np.arange(1, 11), np.arange(1, 11))
plt.imshow(heatmap_data, cmap="plasma")
plt.title("热力图")
plt.colorbar()

# 第六个图：渐变图像
plt.subplot(2, 3, 6)
# 创建渐变图像
gradient = np.linspace(0, 1, 256).reshape(1, -1)
gradient = np.vstack((gradient, gradient, gradient))
plt.imshow(gradient, aspect='auto', cmap="rainbow")
plt.title("颜色渐变")
plt.colorbar()

plt.tight_layout()
plt.show()
```

**代码说明**：
这个例子展示了`plt.imshow()`函数的多种用法，包括显示自定义数据、灰度图、随机矩阵、热力图和颜色渐变。`imshow()`函数不仅可以用来显示图像文件，还可以将各种二维数据可视化为图像，特别适合展示矩阵、热力图和空间分布数据。

**注意**：当使用`imshow()`显示数值数据时，通常会结合`colorbar()`函数添加颜色条，以帮助理解数值与颜色之间的对应关系。

---

## 多图布局

### `plt.subplot()`创建子图

**作用**：在一个画布上创建多个子图，便于对比或同时展示多个相关图表

**语法**：`plt.subplot(nrows, ncols, index)`

**参数说明**：
- `nrows`：行数
- `ncols`：列数
- `index`：指定第几个子图（从1开始）

**使用方式**：
- 将画布划分为 `nrows × ncols` 的网格
- 选择第 `index `个子图进行绘图
- 子图编号从`1`开始，从左到右，从上到下

**示例代码**：
```python
import matplotlib.pyplot as plt
import numpy as np

# 创建数据
x = np.linspace(-3, 3, 50)
y1 = 2*x + 1  # 直线
y2 = x**2     # 抛物线
y3 = np.sin(x) # 正弦曲线
y4 = np.tan(x) # 正切曲线

# 第一个布局：2行2列网格
plt.figure(figsize=(10, 8))
plt.suptitle("2行2列子图布局", fontsize=16)  # 添加总标题

# 在画布上创建2行2列的子图，并在第1个子图中绘画
plt.subplot(2, 2, 1)
plt.plot(x, y1, color='blue')
plt.title("直线: y=2x+1")
plt.grid(True, alpha=0.3)

# 在画布上创建2行2列的子图，在第2个子图中绘画
plt.subplot(2, 2, 2)
plt.plot(x, y2, color='red')
plt.title("抛物线: y=x²")
plt.grid(True, alpha=0.3)

# 在画布上创建2行2列的子图，在第3个子图中绘画
plt.subplot(2, 2, 3)
plt.plot(x, y3, color='green')
plt.title("正弦曲线: y=sin(x)")
plt.grid(True, alpha=0.3)

# 在画布上创建2行2列的子图，在第4个子图中绘画
plt.subplot(2, 2, 4)
plt.plot(x, y4, color='purple')
plt.title("正切曲线: y=tan(x)")
plt.grid(True, alpha=0.3)

plt.tight_layout()  # 自动调整子图参数，防止重叠

# 第二个布局：不规则网格
plt.figure(figsize=(12, 8))
plt.suptitle("不规则子图布局", fontsize=16)

# 在画布上创建2行1列的子图，并在第1个子图中绘画
plt.subplot(2, 1, 1)
plt.plot(x, y1, color='blue', linewidth=2)
plt.title("大图: 直线")
plt.grid(True, alpha=0.3)

# 在画布上创建2行3列的子图，并在第4个子图中绘画
plt.subplot(2, 3, 4)
plt.plot(x, y2, color='red')
plt.title("抛物线")
plt.grid(True, alpha=0.3)

# 在画布上创建2行3列的子图，并在第5个子图中绘画
plt.subplot(2, 3, 5)
plt.plot(x, y3, color='green')
plt.title("正弦曲线")
plt.grid(True, alpha=0.3)

# 在画布上创建2行3列的子图，并在第6个子图中绘画
plt.subplot(2, 3, 6)
plt.plot(x, y4, color='purple')
plt.title("正切曲线")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**代码说明**：
这个例子展示了两种常见的子图布局方式：规则的2×2网格和不规则的网格布局。第一种布局将画布均匀划分为4个子图，每种函数占据一个子图；第二种布局则展示了如何创建不同大小的子图组合，上面是一个大图，下面是三个小图。

**注意**：
1. 子图编号从1开始，不是从0开始
2. 可以使用`plt.suptitle()`为整个画布添加总标题
3. `plt.tight_layout()`函数可以自动调整子图参数，防止标题和标签重叠
4. 子图可以有不同的背景色、网格线等样式

### `plt.axes()`画图中图

**作用**：通过指定相对画布的位置和宽高在一个画布中定制多个坐标轴，实现画中图效果

**语法**：`plt.axes([left, bottom, width, height])`

**参数说明**：
- `left`：坐标框左下角的x位置（相对于画布宽度的比例，0~1之间）
- `bottom`：坐标框左下角的y位置（相对于画布高度的比例，0~1之间）
- `width`：坐标框的宽度（相对于画布宽度的比例，0~1之间）
- `height`：坐标框的高度（相对于画布高度的比例，0~1之间）

**使用说明**：
- 所有参数都是相对于画布大小的比例值
- [0, 0, 1, 1]表示占据整个画布
- 可以通过重叠创建画中图效果

**示例代码**：
```python
import matplotlib.pyplot as plt
import numpy as np

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建数据
x = np.linspace(-3, 3, 50)
y1 = 2*x + 1  # 直线
y2 = x**2     # 抛物线
y3 = np.sin(x) # 正弦曲线

plt.figure(figsize=(10, 8))

# 创建主图（占据大部分画布）
"""
对这四个数字的说明:
0.1, 0.1表示坐标框左下角的位置, x和y坐标在画布的10%的位置
0.8, 0.8表示坐标框的宽和高为画布的80%大小
"""
plt.axes([0.1, 0.1, 0.8, 0.8])
plt.plot(x, y1, color='blue', linewidth=2)
plt.title('主图：直线', fontsize=14)  # 标题
plt.grid(True, alpha=0.3)
plt.xlabel('X轴')
plt.ylabel('Y轴')

# 创建第一个子图（右上角）
plt.axes([0.55, 0.55, 0.35, 0.35])
plt.plot(x, y2, color='red', linewidth=1.5)
plt.title('子图1：抛物线', fontsize=12)
plt.grid(True, alpha=0.3)

# 创建第二个子图（右下角）
plt.axes([0.55, 0.15, 0.35, 0.35])
plt.plot(x, y3, color='green', linewidth=1.5)
plt.title('子图2：正弦曲线', fontsize=12)
plt.grid(True, alpha=0.3)

# 添加总标题
plt.suptitle('画中图示例', fontsize=16, y=0.95)

plt.show()

# 更复杂的画中图示例
plt.figure(figsize=(12, 10))

# 背景图
plt.axes([0.1, 0.1, 0.8, 0.8])
background_data = np.random.rand(100, 100)
plt.imshow(background_data, cmap='gray', alpha=0.3)
plt.title('背景图像', fontsize=14)
plt.axis('off')  # 关闭坐标轴

# 左上角小图
plt.axes([0.15, 0.6, 0.25, 0.25])
x_small = np.linspace(0, 10, 100)
y_small = np.sin(x_small)
plt.plot(x_small, y_small, color='red', linewidth=2)
plt.title('左上角子图', fontsize=10)
plt.grid(True, alpha=0.3)

# 右上角小图
plt.axes([0.6, 0.6, 0.25, 0.25])
categories = ['A', 'B', 'C', 'D']
values = [23, 45, 56, 78]
plt.bar(categories, values, color=['blue', 'green', 'red', 'purple'], alpha=0.7)
plt.title('右上角子图', fontsize=10)
plt.grid(True, alpha=0.3, axis='y')

# 下方中央小图
plt.axes([0.4, 0.15, 0.2, 0.2])
scatter_x = np.random.randn(50)
scatter_y = np.random.randn(50)
plt.scatter(scatter_x, scatter_y, color='purple', alpha=0.6)
plt.title('下方子图', fontsize=10)
plt.grid(True, alpha=0.3)

plt.suptitle('复杂画中图布局', fontsize=16, y=0.95)
plt.show()
```

**代码说明**：
这个例子展示了两种画中图的实现方式：第一种是简单的画中图布局，一个主图和两个子图；第二种是更复杂的布局，包含背景图像和多个不同类型的小图。

**注意事项**：
1. `plt.axes()`中的坐标参数是相对于画布的比例，不是数据坐标
2. 可以通过调整坐标和大小参数实现各种复杂的布局
3. 画中图适合用来展示局部细节或不同视角的数据
4. 需要仔细调整子图位置，避免重叠和遮挡
5. 可以结合不同的图表类型（如折线图、柱状图、散点图等）创建信息丰富的复合图表

---

## 图像保存

### `plt.savefig()`保存图像

**作用**：将当前画布上的图像保存到指定的文件路径，支持多种图像格式

**语法**：`plt.savefig(fname, dpi=None, facecolor='w', edgecolor='w', 
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)`

**主要参数说明**：
- `fname`：保存的文件路径和名称
- `dpi`：分辨率，默认为None（使用figure的dpi）
- `facecolor`：图像背景色，默认为白色'w'
- `edgecolor`：图像边框色，默认为白色'w'
- `format`：图像格式，如'png', 'jpg', 'pdf', 'svg'等，默认从文件扩展名推断
- `transparent`：是否透明背景，默认为False
- `bbox_inches`：保存图像的边界框，'tight'表示紧密裁剪
- `pad_inches`：边界填充，默认为0.1
- `orientation`：图像方向，'portrait'（纵向）或'landscape'（横向）

**常见图像格式**：
| 格式 | 扩展名 | 说明 |
|------|--------|------|
|` PNG `| `.png` | 无损压缩，支持透明背景 |
|` JPEG` |` .jpg`, `.jpeg` | 有损压缩，适合照片 |
|` PDF `| `.pdf` | 矢量格式，适合打印 |
|` SVG `| `.svg` | 矢量格式，可缩放 |
|` EPS `| `.eps` | 矢量格式，适合出版 |

**示例代码**：
```python
import matplotlib.pyplot as plt
import numpy as np
import os

# 创建保存图像的目录
if not os.path.exists('saved_images'):
    os.makedirs('saved_images')

# 创建数据
x = np.linspace(-5, 5, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# 第一个图：基本保存
plt.figure(figsize=(8, 6))
plt.plot(x, y1, 'b-', label='sin(x)', linewidth=2)
plt.plot(x, y2, 'r--', label='cos(x)', linewidth=2)
plt.title('三角函数图像')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)

# 保存为PNG格式（默认设置）
plt.savefig('saved_images/trigonometric_default.png')
print("保存默认PNG图像成功")

# 第二个图：高质量保存
plt.figure(figsize=(10, 8))
plt.plot(x, y1, 'b-', label='sin(x)', linewidth=2)
plt.plot(x, y2, 'r--', label='cos(x)', linewidth=2)
plt.title('高质量三角函数图像')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)

# 高质量保存，设置高DPI和紧密边界
plt.savefig('saved_images/trigonometric_hq.png', 
            dpi=300,              # 高分辨率
            bbox_inches='tight', # 紧密裁剪
            pad_inches=0.1)       # 边界填充
print("保存高质量PNG图像成功")

# 第三个图：不同格式保存
plt.figure(figsize=(8, 6))
plt.scatter(x[::5], y1[::5], c=y1[::5], cmap='viridis', alpha=0.7)
plt.colorbar(label='sin(x)值')
plt.title('散点图示例')
plt.xlabel('x')
plt.ylabel('y')

# 保存为PDF格式（矢量图）
plt.savefig('saved_images/scatter_vector.pdf', format='pdf')
print("保存PDF矢量图像成功")

# 保存为SVG格式（可缩放矢量图）
plt.savefig('saved_images/scatter_vector.svg', format='svg')
print("保存SVG矢量图像成功")

# 第四个图：透明背景保存
plt.figure(figsize=(8, 6))
plt.plot(x, y1, 'g-', linewidth=3)
plt.title('透明背景图像')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True, alpha=0.5)

# 保存为带透明背景的PNG
plt.savefig('saved_images/transparent_bg.png', 
            transparent=True,      # 透明背景
            facecolor='none')      # 无背景色
print("保存透明背景图像成功")

# 第五个图：动画保存示例
plt.figure(figsize=(8, 6))
y2 = []
x_values = np.linspace(-3, 3, 50)

for i in range(len(x_values)):
    y2.append(x_values[i]**2)  # 逐步添加点
    plt.clf()  # 清除数据, 放在这个位置是为了循环的清除上一次的数据
    plt.plot(x_values[:i+1], y2, 'b-', linewidth=2)
    plt.title(f'动态图像 - 第{i+1}帧')
    plt.xlabel('x')
    plt.ylabel('y = x²')
    plt.grid(True, alpha=0.3)
    
    # 每隔5帧保存一次图像
    if i % 5 == 0:
        plt.savefig(f'saved_images/animation_frame_{i:02d}.png')
    
    plt.pause(0.05)  # 暂停0.05秒

print("保存动画帧图像成功")

# 第六个图：自定义尺寸和方向保存
plt.figure(figsize=(12, 4))
plt.plot(x, np.sin(x), 'r-', linewidth=2)
plt.plot(x, np.cos(x), 'b-', linewidth=2)
plt.title('横向布局图像')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['sin(x)', 'cos(x)'])
plt.grid(True, alpha=0.3)

# 保存为横向PDF
plt.savefig('saved_images/landscape_layout.pdf', 
            orientation='landscape',  # 横向
            format='pdf')
print("保存横向PDF图像成功")

# 显示所有保存的图像
print("\n所有图像已保存到 saved_images 目录中:")
for file in os.listdir('saved_images'):
    print(f"- {file}")

plt.show()
```

**代码说明**：
这个例子展示了多种图像保存方式，包括：
1. 基本保存（默认设置）
2. 高质量保存（高DPI和紧密边界）
3. 不同格式保存（PDF和SVG矢量图）
4. 透明背景保存
5. 动画帧保存
6. 自定义尺寸和方向保存

**注意事项**：
1. `plt.savefig()`应该在`plt.show()`之前调用，因为`show()`可能会消耗图形资源
2. 保存路径中的目录必须存在，否则会报错
3. 对于出版用途，建议使用矢量格式（PDF, SVG）和高DPI设置（300dpi以上）
4. `bbox_inches='tight'`参数可以自动裁剪空白区域，使图像更加紧凑
5. `transparent=True`参数可以创建透明背景的图像，适合叠加到其他背景上

