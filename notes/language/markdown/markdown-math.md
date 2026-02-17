# Markdown 数学公式语法

Markdown 支持使用 LaTeX 语法来编写数学公式。本文档介绍在 Markdown 中编辑数学公式的常用语法。

## 行内公式与块级公式

### 行内公式
使用单个美元符号 `$` 包围公式，将公式嵌入到文本中：

```
这是一个行内公式示例：$E = mc^2$，它表示质能方程。
```

显示效果：这是一个行内公式示例：$E = mc^2$，它表示质能方程。

### 块级公式
使用两个美元符号 `$$` 包围公式，将公式单独成行显示：

```
$$
\frac{d}{dx}(x^n) = nx^{n-1}
$$
```

显示效果：
$$
\frac{d}{dx}(x^n) = nx^{n-1}
$$

## 基本数学符号

### 上标和下标

```
上标：x^2, x^n
下标：x_1, x_{i+1}
组合：x_1^2, x_{i+1}^n
```

显示效果：
- 上标：$x^2$, $x^n$
- 下标：$x_1$, $x_{i+1}$
- 组合：$x_1^2$, $x_{i+1}^n$

### 分数

```
简单分数：\frac{a}{b}
复杂分数：\frac{x^2 + y^2}{x + y}
嵌套分数：\frac{\frac{a}{b}}{\frac{c}{d}} = \frac{ad}{bc}
```

显示效果：
- 简单分数：$\frac{a}{b}$
- 复杂分数：$\frac{x^2 + y^2}{x + y}$
- 嵌套分数：$\frac{\frac{a}{b}}{\frac{c}{d}} = \frac{ad}{bc}$

### 根号

```
平方根：\sqrt{x}
n次根号：\sqrt[n]{x}
复杂表达式：\sqrt{x^2 + y^2}
```

显示效果：
- 平方根：$\sqrt{x}$
- n次根号：$\sqrt[n]{x}$
- 复杂表达式：$\sqrt{x^2 + y^2}$

### 求和与积分

```
求和：\sum_{i=1}^{n} i = \frac{n(n+1)}{2}
无穷求和：\sum_{i=1}^{\infty} \frac{1}{i^2} = \frac{\pi^2}{6}
积分：\int_{a}^{b} f(x) dx
定积分：\int_{0}^{\infty} e^{-x} dx = 1
```

显示效果：
- 求和：$\sum_{i=1}^{n} i = \frac{n(n+1)}{2}$
- 无穷求和：$\sum_{i=1}^{\infty} \frac{1}{i^2} = \frac{\pi^2}{6}$
- 积分：$\int_{a}^{b} f(x) dx$
- 定积分：$\int_{0}^{\infty} e^{-x} dx = 1$

### 极限

```
极限：\lim_{x \to \infty} \frac{1}{x} = 0
单侧极限：\lim_{x \to 0^+} \frac{1}{x} = \infty
```

显示效果：
- 极限：$\lim_{x \to \infty} \frac{1}{x} = 0$
- 单侧极限：$\lim_{x \to 0^+} \frac{1}{x} = \infty$

## 常用希腊字母

| 字母 | 代码 | 字母 | 代码 | 字母 | 代码 |
|------|------|------|------|------|------|
| $\alpha$ | `\alpha` | $\lambda$ | `\lambda` | $\Gamma$ | `\Gamma` |
| $\beta$ | `\beta` | $\mu$ | `\mu` | $\Delta$ | `\Delta` |
| $\gamma$ | `\gamma` | $\nu$ | `\nu` | $\Theta$ | `\Theta` |
| $\delta$ | `\delta` | $\xi$ | `\xi` | $\Lambda$ | `\Lambda` |
| $\epsilon$ | `\epsilon` | $\pi$ | `\pi` | $\Xi$ | `\Xi` |
| $\zeta$ | `\zeta` | $\rho$ | `\rho` | $\Pi$ | `\Pi` |
| $\eta$ | `\eta` | $\sigma$ | `\sigma` | $\Sigma$ | `\Sigma` |
| $\theta$ | `\theta` | $\tau$ | `\tau` | $\Phi$ | `\Phi` |
| $\phi$ | `\phi` | $\chi$ | `\chi` | $\Psi$ | `\Psi` |
| $\psi$ | `\psi` | $\omega$ | `\omega` | $\Omega$ | `\Omega` |

## 常用运算符

| 符号 | 代码 | 符号 | 代码 | 符号 | 代码 |
|------|------|------|------|------|------|
| $+$ | `+` | $-$ | `-` | $\times$ | `\times` |
| $\div$ | `\div` | $=$ | `=` | $\neq$ | `\neq` |
| $<$ | `<` | $>$ | `>` | $\leq$ | `\leq` |
| $\geq$ | `\geq` | $\approx$ | `\approx` | $\equiv$ | `\equiv` |
| $\in$ | `\in` | $\subset$ | `\subset` | $\supset$ | `\supset` |
| $\cup$ | `\cup` | $\cap$ | `\cap` | $\emptyset$ | `\emptyset` |
| $\forall$ | `\forall` | $\exists$ | `\exists` | $\neg$ | `\neg` |
| $\land$ | `\land` | $\lor$ | `\lor` | $\implies$ | `\implies` |
| $\rightarrow$ | `\rightarrow` | $\leftarrow$ | `\leftarrow` | $\leftrightarrow$ | `\leftrightarrow` |
| $\Rightarrow$ | `\Rightarrow` | $\Leftarrow$ | `\Leftarrow` | $\Leftrightarrow$ | `\Leftrightarrow` |

### 高级运算符
| 符号 | 代码 | 符号 | 代码 | 符号 | 代码 |
|------|------|------|------|------|------|
| $\nabla$ | `\nabla` | $\partial$ | `\partial` | $\infty$ | `\infty` |
| $\propto$ | `\propto` | $\sim$ | `\sim` | $\cong$ | `\cong` |
| $\oplus$ | `\oplus` | $\otimes$ | `\otimes` | $\odot$ | `\odot` |
| $\bigcirc$ | `\bigcirc` | $\bigtriangleup$ | `\bigtriangleup` | $\bigtriangledown$ | `\bigtriangledown` |
| $\dagger$ | `\dagger` | $\ddagger$ | `\ddagger` | $\S$ | `\S` |
| $\ast$ | `\ast` | $\star$ | `\star` | $\circ$ | `\circ` |
| $\bullet$ | `\bullet` | $\cdot$ | `\cdot` | $\cdots$ | `\cdots` |
| $\vdots$ | `\vdots` | $\ddots$ | `\ddots` | $\ldots$ | `\ldots` |

## 矩阵与向量

### 矩阵

```
2x2矩阵：\begin{pmatrix} a & b \\ c & d \end{pmatrix}
3x3矩阵：\begin{bmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{bmatrix}
行列式：\begin{vmatrix} a & b \\ c & d \end{vmatrix}
```

显示效果：
- 2x2矩阵：$\begin{pmatrix} a & b \\ c & d \end{pmatrix}$
- 3x3矩阵：$\begin{bmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{bmatrix}$
- 行列式：$\begin{vmatrix} a & b \\ c & d \end{vmatrix}$

### 向量

```
列向量：\begin{pmatrix} x_1 \\ x_2 \\ x_3 \end{pmatrix}
行向量：\begin{pmatrix} x_1 & x_2 & x_3 \end{pmatrix}
单位向量：\mathbf{i}, \mathbf{j}, \mathbf{k}
向量点积：\vec{a} \cdot \vec{b} = a_x b_x + a_y b_y + a_z b_z
向量叉积：\vec{a} \times \vec{b} = \begin{vmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \\ a_x & a_y & a_z \\ b_x & b_y & b_z \end{vmatrix}
```

显示效果：
- 列向量：$\begin{pmatrix} x_1 \\ x_2 \\ x_3 \end{pmatrix}$
- 行向量：$\begin{pmatrix} x_1 & x_2 & x_3 \end{pmatrix}$
- 单位向量：$\mathbf{i}, \mathbf{j}, \mathbf{k}$
- 向量点积：$\vec{a} \cdot \vec{b} = a_x b_x + a_y b_y + a_z b_z$
- 向量叉积：$\vec{a} \times \vec{b} = \begin{vmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \\ a_x & a_y & a_z \\ b_x & b_y & b_z \end{vmatrix}$

## 高级公式示例

### 二次方程求根公式

```
$$
x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}
$$
```

显示效果：
$$
x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}
$$

### 欧拉公式

```
$$
e^{i\theta} = \cos\theta + i\sin\theta
$$

特殊情况（欧拉恒等式）：
$$
e^{i\pi} + 1 = 0
$$
```

显示效果：
$$
e^{i\theta} = \cos\theta + i\sin\theta
$$

特殊情况（欧拉恒等式）：
$$
e^{i\pi} + 1 = 0
$$

### 正态分布概率密度函数

```
$$
f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}
$$
```

显示效果：
$$
f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}
$$

### 泰勒级数

```
$$
f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!}(x-a)^n
$$
```

显示效果：
$$
f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!}(x-a)^n
$$

### 傅里叶级数

```
$$
f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty} \left[ a_n \cos\left(\frac{2\pi n x}{T}\right) + b_n \sin\left(\frac{2\pi n x}{T}\right) \right]
$$
```

显示效果：
$$
f(x) = \frac{a_0}{2} + \sum_{n=1}^{\infty} \left[ a_n \cos\left(\frac{2\pi n x}{T}\right) + b_n \sin\left(\frac{2\pi n x}{T}\right) \right]
$$

### 拉普拉斯变换

```
$$
F(s) = \mathcal{L}\{f(t)\} = \int_{0}^{\infty} e^{-st} f(t) dt
$$
```

显示效果：
$$
F(s) = \mathcal{L}\{f(t)\} = \int_{0}^{\infty} e^{-st} f(t) dt
$$

### 微分方程

```
$$
\frac{d^2y}{dx^2} + p(x)\frac{dy}{dx} + q(x)y = f(x)
$$

一阶线性微分方程：
$$
\frac{dy}{dx} + P(x)y = Q(x)
$$

热传导方程：
$$
\frac{\partial u}{\partial t} = \alpha \nabla^2 u
$$
```

显示效果：
$$
\frac{d^2y}{dx^2} + p(x)\frac{dy}{dx} + q(x)y = f(x)
$$

一阶线性微分方程：
$$
\frac{dy}{dx} + P(x)y = Q(x)
$$

热传导方程：
$$
\frac{\partial u}{\partial t} = \alpha \nabla^2 u
$$

### 概率统计公式

```
期望值：E[X] = \sum_{i=1}^{n} x_i p_i
方差：\text{Var}(X) = E[(X - \mu)^2]
协方差：\text{Cov}(X, Y) = E[(X - \mu_X)(Y - \mu_Y)]
相关系数：\rho_{X,Y} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}
贝叶斯定理：P(A|B) = \frac{P(B|A)P(A)}{P(B)}
```

显示效果：
- 期望值：$E[X] = \sum_{i=1}^{n} x_i p_i$
- 方差：$\text{Var}(X) = E[(X - \mu)^2]$
- 协方差：$\text{Cov}(X, Y) = E[(X - \mu_X)(Y - \mu_Y)]$
- 相关系数：$\rho_{X,Y} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$
- 贝叶斯定理：$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$

### 集合论公式

```
并集：A \cup B = \{x | x \in A \text{ 或 } x \in B\}
交集：A \cap B = \{x | x \in A \text{ 且 } x \in B\}
补集：A^c = \{x | x \notin A\}
德摩根定律：(A \cup B)^c = A^c \cap B^c
幂集：\mathcal{P}(A) = \{X | X \subseteq A\}
```

显示效果：
- 并集：$A \cup B = \{x | x \in A \text{ 或 } x \in B\}$
- 交集：$A \cap B = \{x | x \in A \text{ 且 } x \in B\}$
- 补集：$A^c = \{x | x \notin A\}$
- 德摩根定律：$(A \cup B)^c = A^c \cap B^c$
- 幂集：$\mathcal{P}(A) = \{X | X \subseteq A\}$

## 数学环境

### 方程组

```
$$
\begin{cases}
x + y = 5 \\
2x - y = 1
\end{cases}
$$
```

显示效果：
$$
\begin{cases}
x + y = 5 \\
2x - y = 1
\end{cases}
$$

### 对齐方程

```
$$
\begin{align}
f(x) &= (x+1)^2 \\
     &= x^2 + 2x + 1 \\
     &= (x+1)(x+1)
\end{align}
$$
```

显示效果：
$$
\begin{align}
f(x) &= (x+1)^2 \\
     &= x^2 + 2x + 1 \\
     &= (x+1)(x+1)
\end{align}
$$

## 特殊函数

| 函数 | 代码 | 函数 | 代码 | 函数 | 代码 |
|------|------|------|------|------|------|
| $\sin$ | `\sin` | $\cos$ | `\cos` | $\tan$ | `\tan` |
| $\cot$ | `\cot` | $\sec$ | `\sec` | $\csc$ | `\csc` |
| $\arcsin$ | `\arcsin` | $\arccos$ | `\arccos` | $\arctan$ | `\arctan` |
| $\log$ | `\log` | $\ln$ | `\ln` | $\lg$ | `\lg` |
| $\exp$ | `\exp` | $\lim$ | `\lim` | $\max$ | `\max` |
| $\min$ | `\min` | $\sup$ | `\sup` | $\inf$ | `\inf` |


## 常见问题与技巧

1. **空格处理**：在数学模式中，普通空格被忽略。如需添加空格，可使用：
   - `\,`：小空格
   - `\:`：中等空格
   - `\;`：大空格
   - `\quad`：大空格
   - `\qquad`：更大的空格

2. **文本模式**：在数学模式中插入普通文本，使用 `\text{}`：

```
$$
f(x) = \text{一个函数}
$$
```

显示效果：
$$
f(x) = \text{一个函数}
$$

3. **括号大小调整**：使用 `\left` 和 `\right` 自动调整括号大小：  

```
$$
\left( \frac{a}{b} \right)
$$
```

显示效果：

$$
\left( \frac{a}{b} \right)
$$


1. **字体样式**：
   - `\mathbf{}`：粗体
   - `\mathit{}`：斜体
   - `\mathrm{}`：正体
   - `\mathsf{}`：无衬线
   - `\mathtt{}`：打字机字体
   - `\mathcal{}`：书法体
   - `\mathbb{}`：黑板粗体
   - `\mathfrak{}`：哥特体

字体样式示例：
```
\mathbf{ABC} \quad \mathit{ABC} \quad \mathrm{ABC} \quad \mathsf{ABC}
\mathcal{ABC} \quad \mathbb{ABC} \quad \mathfrak{ABC}
```

显示效果：
$\mathbf{ABC} \quad \mathit{ABC} \quad \mathrm{ABC} \quad \mathsf{ABC}$
$\mathcal{ABC} \quad \mathbb{ABC} \quad \mathfrak{ABC}$

## VitePress 中的数学公式支持

在 VitePress 中，默认可能不直接支持数学公式渲染。可以通过以下方式启用：

1. 安装数学公式插件，如 `markdown-it-mathjax3` 或 `markdown-it-katex`

2. 配置 VitePress 的 `markdown` 选项：

```javascript
// .vitepress/config.js
module.exports = {
  markdown: {
    theme: {
      light: 'github-light',
      dark: 'github-dark'
    },
    lineNumbers: true,
    config: (md) => {
      // 使用 MathJax
      md.use(require('markdown-it-mathjax3'));
      
      // 或使用 KaTeX
      // md.use(require('markdown-it-katex'));
    }
  }
}
```

3. 相应地，需要在 HTML 中引入 MathJax 或 KaTeX 的 CSS/JS 文件。

这样配置后，就可以在 VitePress 中正常渲染 LaTeX 数学公式了。





