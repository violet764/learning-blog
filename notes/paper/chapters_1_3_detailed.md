# 第一至三章：理论部分

本部分给出“一类混杂生成扩散”的理论基础，依次讨论平稳分布、遍历性及其相关问题、以及参数估计。为便于叙述，先给出记号与预备知识。

## 记号与预备知识
- $\lVert\cdot\rVert$ 为欧氏范数，$\langle x,y\rangle$ 为内积；$\nabla$ 与 $\nabla^2$ 分别为梯度与 Hessian。
- $\mathcal{L}$ 表示标签（模式）集合，通常有限：$\mathcal{L}=\{1,\dots,M\}$。
- $W_t$ 为 $m$ 维布朗运动；$N(dt,dz)$ 为泊松随机测度，补偿测度 $\tilde N(dt,dz)=N(dt,dz)-dt\nu(dz)$。
- $\nu$ 为 Lévy 测度，满足 $\int_{\mathbb{R}^d\setminus\{0\}}(1\wedge\lVert z\rVert^2)\,\nu(dz)<\infty$。
- $(-\Delta)^{\alpha/2}$ 为分数阶拉普拉斯算子（$0<\alpha<2$），其 Fourier 符号为 $\lVert\xi\rVert^\alpha$。

下文按章节展开。

## 第一章 一类混杂生成扩散的平稳分布

### 1.1 模型定义：混杂生成扩散与马氏切换
本论文研究的对象是定义在组合空间 $\mathcal{X}=\mathbb{R}^d\times\mathcal{L}$ 上的强马氏过程
$$
X_t=(Y_t,L_t),\qquad Y_t\in\mathbb{R}^d,\ L_t\in\mathcal{L}.
$$
其中 $Y_t$ 表示连续状态（在语言应用中可解释为句向量或语义嵌入），$L_t$ 表示离散模式/语境（例如话题或生成“风格”）。

1) **连续部分（给定模式）**：在固定模式 $\ell$ 下，$Y_t$ 满足 Lévy 驱动的 SDE
$$
 dY_t = b(Y_t,\ell,t;\theta)dt + \Sigma(Y_t,\ell,t;\theta)dW_t + \int_{\mathbb{R}^d\setminus\{0\}} z\,\tilde N(dt,dz).
$$
这里 $b$、$\Sigma$ 可依赖参数 $\theta$；跳跃项由 Lévy 测度 $\nu_{\ell}(dz)$ 决定，允许不同模式具有不同“突变”强度与重尾行为。

2) **离散切换（模式链）**：$L_t$ 为（可能状态依赖的）连续时间马氏链，其生成矩阵为 $Q(y,t)=(q_{\ell\ell'}(y,t))$，满足
$$
\mathbb{P}(L_{t+dt}=\ell'\mid L_t=\ell, Y_t=y)=q_{\ell\ell'}(y,t)dt+o(dt),\quad \ell'\ne \ell,
$$
并取 $q_{\ell\ell}(y,t)=-\sum_{\ell'\ne\ell}q_{\ell\ell'}(y,t)$。当 $Q$ 与 $y$ 无关时，$L_t$ 的平稳分布 $\pi_L$ 满足 $\pi_L Q=0$。

3) **混杂性的来源**：
- 连续扩散 + Lévy 跳跃：同时包含高斯与非高斯噪声；
- 模式切换：允许系统在多个动力学之间跳转；
- 可能的状态依赖切换：$q_{\ell\ell'}$ 依赖 $y$ 时，更贴近“语义区域触发模式切换”。

#### 1.1.1 常见混杂生成扩散的数学原型（用于后文贯穿）
为避免“混杂”概念过于抽象，本节列出几类在文献中反复出现、且与本论文理论工具直接对应的原型模型。设 $L_t$ 为取值于 $\mathcal{L}$ 的连续时间马氏链，$W_t$ 为布朗运动，$Z_t$ 为 Lévy 过程（可取 α-稳定）。

1) **马氏切换扩散（regime-switching diffusion）**：
$$
dY_t=b(Y_t,L_t)\,dt+\Sigma(Y_t,L_t)\,dW_t.
$$
当 $b(\cdot,\ell)$ 在无穷远处具有耗散性（如 $\langle y,b(y,\ell)\rangle\le -c\lVert y\rVert^2+C$）时，常可得到平稳分布与几何遍历性。

2) **马氏切换跳扩散（regime-switching jump diffusion）**：
$$
dY_t=b(Y_t,L_t)\,dt+\Sigma(Y_t,L_t)\,dW_t+\int_{\mathbb{R}^d\setminus\{0\}}g(Y_{t^-},L_t,z)\,\tilde N(dt,dz).
$$
当 $g(y,\ell,z)=z$ 且 $\nu_\ell(dz)$ 为 α-稳定测度时，生成元出现 $(-\Delta)^{\alpha/2}$，对应“非局部”平稳方程与分数阶能量方法（见 1.3 与 1.6）。

3) **马氏切换分数阶 OU 原型（重尾稳态）**：在一维为例，
$$
dY_t=-\kappa_{L_t}Y_t\,dt+dZ_t^{(\alpha,L_t)},
$$
其中 $Z_t^{(\alpha,L_t)}$ 为在模式 $L_t$ 下的对称 α-稳定过程。该模型的平稳分布通常为重尾（一般并非高斯），体现 α-稳定噪声对稳态尾部的决定性影响。

4) **分布依赖/自洽漂移（与固定点问题相关）**：
$$
dY_t=b(Y_t,L_t,\rho_t)\,dt+dZ_t^{(\alpha,L_t)},\qquad \rho_t=\mathcal{L}(Y_t).
$$
当 $b$ 依赖于 $\rho_t$（例如 $b(y,\ell,\rho)= -\nabla U_\ell(y) - \nabla (W*\rho)(y)$）时，稳态密度满足分数阶 Fokker–Planck 的**自洽**（fixed-point）方程，这正是“分数阶拉普拉斯算子不动点问题”出现的典型来源（见 1.3.6）。

#### 1.1.2 连续时间马氏链与马氏切换过程：基本性质与研究脉络
本论文在多个章节反复使用马氏链（$L_t$ 或 $r_t$）的长期性质与可达性，因此在此集中给出常用结论（后文引用不再重复证明细节）。

**定义 1.1（连续时间马氏链与生成矩阵）** 设 $L_t$ 取值于有限状态集合 $\mathcal{L}=\{1,\dots,M\}$，其生成矩阵 $Q=(q_{ij})$ 满足：$q_{ij}\ge 0$（$i\neq j$），$q_{ii}=-\sum_{j\neq i}q_{ij}$。转移矩阵 $P(t)=\exp(tQ)$ 满足 Kolmogorov 前向/后向方程
$$
P'(t)=P(t)Q=QP(t),\qquad P(0)=I.
$$

**定义 1.2（不可约、平稳分布与遍历性）** 若由 $Q$ 诱导的有向图强连通，则称链不可约。不可约有限状态链存在唯一平稳分布 $\pi$，满足
$$
\pi^\top Q=0,\qquad \sum_{i=1}^M\pi_i=1,\qquad \pi_i>0.
$$
并有遍历定理：对任意函数 $h:\mathcal{L}\to\mathbb{R}$，
$$
\frac1t\int_0^t h(L_s)\,ds \xrightarrow[t\to\infty]{a.s.} \sum_{i=1}^M \pi_i h(i).
$$
该结论是后续“平均化观点”（1.5.3）与“遍历 + 鞅 CLT”（3.12）推断渐近的基础。

**研究脉络（简述）**：马氏切换模型在金融、控制、生物与通信中广泛出现。理论上主要围绕三条线展开：  
(i) **切换稳定性**（共用/多 Lyapunov、平均驻留时间、Markov 切换的矩阵不等式）；  
(ii) **可达性与不可约性**（Hörmander、支撑定理、跳跃支撑）；  
(iii) **统计推断**（隐马氏结构下的 EM/滤波、以及连续观测下的似然/Girsanov）。

#### 1.1.3 马氏切换跳扩散的解：存在唯一性与强马氏性（证明骨架）
为保证“平稳分布”“遍历性”“似然推断”的论证严谨，必须先确保过程的良定性。

**定理 1.1（马氏切换跳扩散的强解存在唯一性）** 设 $b(\cdot,\ell)$、$\Sigma(\cdot,\ell)$、$g(\cdot,\ell,\cdot)$ 满足对 $y$ 的局部 Lipschitz 与线性增长条件：存在常数 $C$ 使对所有 $\ell$，
$$
\lVert b(y,\ell)\rVert+\lVert \Sigma(y,\ell)\rVert+\int (1\wedge\lVert g(y,\ell,z)\rVert^2)\,\nu_\ell(dz)\le C(1+\lVert y\rVert),
$$
且 $Q(y)$ 在紧集上有界并满足基本可测性。则对任意初值 $(Y_0,L_0)$，方程
$$
dY_t=b(Y_t,L_t)\,dt+\Sigma(Y_t,L_t)\,dW_t+\int g(Y_{t^-},L_t,z)\,\tilde N(dt,dz)
$$
存在唯一强解；且 $(Y_t,L_t)$ 为强马氏过程并具有 càdlàg 样本路径。

**证明（骨架，写作可据此展开）**：  
（i）给定一条模式路径 $L_t$（分段常值），在每个驻留区间上方程退化为经典跳扩散 SDE，存在唯一强解；  
（ii）在切换时刻，将前一区间末值作为下一段初值，拼接得到全局解；  
（iii）利用停止时刻与强解唯一性证明拼接解不依赖于切换时刻的近似；  
（iv）利用“扩展状态空间”思想（把 $L_t$ 也视为状态变量）可验证强马氏性。

本章目标：给出平稳分布（或平稳密度）存在唯一性的条件，并解释 Lévy/α 稳定噪声、分数阶算子、切换稳定性如何影响平稳结构。

### 1.2 Lévy 过程、Lévy 测度与 α-稳定过程

#### 1.2.1 Lévy–Khintchine 公式与 Lévy–Itô 分解
设 $J_t$ 为 $d$ 维 Lévy 过程，其特征指数由 Lévy–Khintchine 公式给出：对任意 $u\in\mathbb{R}^d$，
$$
\mathbb{E}\,e^{iu\cdot J_t}=\exp\{-t\psi(u)\},
$$
其中
$$
\psi(u)=-i a\cdot u + \tfrac12 u^\top Q u + \int_{\mathbb{R}^d\setminus\{0\}}\Big(1-e^{iu\cdot z}+iu\cdot z\,\mathbf{1}_{\{\lVert z\rVert\le 1\}}\Big)\,\nu(dz).
$$
在 $\int (1\wedge\lVert z\rVert^2)\nu(dz)<\infty$ 下，有 Lévy–Itô 分解
$$
J_t = a t + Q^{1/2}B_t + \int_{\lVert z\rVert\le 1} z\,\tilde N((0,t],dz) + \int_{\lVert z\rVert>1} z\, N((0,t],dz).
$$
该分解揭示：小跳以补偿泊松积分形式出现，大跳以非补偿形式出现；这对于后续 Itô 公式与生成元的严格推导是关键。

#### 1.2.2 α-稳定过程与稳定分布
当 Lévy 测度取
$$
\nu(dz)=c_{d,\alpha}\,\lVert z\rVert^{-d-\alpha}dz,\quad 0<\alpha<2,
$$
且 $a=0,Q=0$ 时，$J_t$ 为对称 α-稳定过程。其特征函数满足
$$
\mathbb{E}\,e^{iu\cdot J_t}=\exp\{-t\gamma^\alpha\lVert u\rVert^\alpha\},
$$
并具有尺度性质 $J_t\overset{d}=t^{1/\alpha}J_1$。对应的 α-稳定分布 $S(\alpha,0,\gamma,0)$ 具有重尾：通常 $\mathbb{E}\lVert J_1\rVert^p<\infty$ 当且仅当 $p<\alpha$。

从建模角度看，在语义嵌入空间中，重尾跳跃可刻画“少量但幅度很大”的语义突变（例如话题突转、语义重写），因此 α-稳定驱动为“多峰/厚尾平稳分布”提供自然机制。

#### 1.2.3 一般 α-稳定分布：特征函数参数化与基本性质
在实际数据建模中（金融收益、文本嵌入差分等），常用四参数稳定分布族 $S_\alpha(\sigma,\beta,\mu)$（也记作 $S(\alpha,\beta,\sigma,\mu)$）。其核心定义并非密度（因为一般无显式密度），而是特征函数。

**定义 1.3（α-稳定随机变量的特征函数）** 随机变量 $X$ 称为 α-稳定（$0<\alpha\le 2$），若存在参数 $\sigma>0$（尺度）、$\beta\in[-1,1]$（偏斜）、$\mu\in\mathbb{R}$（位置），使其特征函数满足：
当 $\alpha\neq 1$ 时，
$$
\varphi_X(u)=\mathbb{E}e^{iuX}
=\exp\Big\{ -\sigma^\alpha |u|^\alpha\big(1-i\beta\,\mathrm{sgn}(u)\tan\tfrac{\pi\alpha}{2}\big)+i\mu u\Big\},
$$
当 $\alpha=1$ 时，
$$
\varphi_X(u)=\exp\Big\{ -\sigma |u|\big(1+i\beta\,\tfrac{2}{\pi}\mathrm{sgn}(u)\ln|u|\big)+i\mu u\Big\}.
$$
对称 α-稳定对应 $\beta=0$。

**性质 1（稳定性与尺度性）** 若 $X_1,X_2$ 独立同分布且为 α-稳定，则对任意 $c_1,c_2\in\mathbb{R}$，
$$
c_1X_1+c_2X_2\ \overset{d}{=}\ \Big(|c_1|^\alpha+|c_2|^\alpha\Big)^{1/\alpha}X + \text{（位置项）}.
$$
*证明*：直接利用特征函数相乘与指数相加即可。该性质解释了为何 α-稳定是“广义中心极限定理”的极限族。

**性质 2（矩存在性与重尾）** 若 $X$ 为非退化 α-稳定，则一般有
$$
\mathbb{E}|X|^p<\infty \Longleftrightarrow p<\alpha.
$$
*证明（要点）*：对称 α-稳定分布满足重尾渐近 $\mathbb{P}(|X|>x)\asymp x^{-\alpha}$（$x\to\infty$）。由分部积分公式
$$
\mathbb{E}|X|^p = p\int_0^\infty x^{p-1}\mathbb{P}(|X|>x)\,dx,
$$
即可得到：当 $p<\alpha$ 时上式在无穷远处可积，从而 $\mathbb{E}|X|^p<\infty$；当 $p\ge \alpha$ 时积分发散，从而 $\mathbb{E}|X|^p=\infty$。证毕。

#### 1.2.4 三个特例：高斯、柯西与 Lévy 分布（便于读者建立直觉）
α-稳定分布只有少数情形具有显式密度，这些特例常用于校验推导与数值实现：

1) **高斯分布**：$\alpha=2,\beta=0$ 时，$X\sim N(\mu,2\sigma^2)$。

2) **柯西分布**：$\alpha=1,\beta=0$ 时，
$$
f(x)=\frac{1}{\pi}\frac{\sigma}{(x-\mu)^2+\sigma^2}.
$$

3) **一侧 Lévy 分布**（稳定分布的一个偏斜极端）：$\alpha=\tfrac12,\beta=1$ 时在 $x>\mu$ 有
$$
f(x)=\sqrt{\frac{\sigma}{2\pi}}\,(x-\mu)^{-3/2}\exp\Big(-\frac{\sigma}{2(x-\mu)}\Big).
$$
这些例子强调：当 $\alpha<2$ 时密度尾部远重于高斯，从而稳态与遍历速度的分析必须转向分数阶工具（生成元与能量，见 1.3 与 2.4）。

#### 1.2.5 对称 α-稳定的模拟：Chambers–Mallows–Stuck（CMS）算法（用于后续数值模拟）
第四章若需要在 CPU 上模拟“α-稳定噪声驱动的混杂扩散”，最常用的是 CMS 算法。

**命题 1.4（CMS 采样公式）** 令 $U\sim \mathrm{Unif}(-\tfrac{\pi}{2},\tfrac{\pi}{2})$，$W\sim \mathrm{Exp}(1)$ 相互独立。对 $\alpha\in(0,2)$，定义
当 $\alpha\neq 1$ 时，
$$
X=\sigma\cdot \frac{\sin(\alpha U)}{(\cos U)^{1/\alpha}}
\cdot\left(\frac{\cos\big((1-\alpha)U\big)}{W}\right)^{\frac{1-\alpha}{\alpha}}+\mu,
$$
则 $X\sim S(\alpha,0,\sigma,\mu)$（对称 α-稳定）。

该算法基于稳定分布的特征函数与极坐标表示，可用于生成独立增量，从而得到 α-稳定 Lévy 过程的样本路径。

### 1.3 无穷小生成元、分数阶拉普拉斯与伴随算子

#### 1.3.1 跳扩散（含切换）的生成元
对足够光滑且增长受控的测试函数 $f(y,\ell)$，过程 $X_t$ 的无穷小生成元可写为
$$
\begin{aligned}
\mathcal{G}f(y,\ell)
&= b(y,\ell)\cdot \nabla_y f(y,\ell) + \tfrac12\mathrm{Tr}\big(\Sigma\Sigma^\top(y,\ell)\,\nabla_y^2 f(y,\ell)\big) \\
&\quad + \int_{\mathbb{R}^d\setminus\{0\}}\Big[f(y+z,\ell)-f(y,\ell)-\mathbf{1}_{\{\lVert z\rVert\le 1\}}\,z^\top\nabla_y f(y,\ell)\Big]\,\nu_{\ell}(dz) \\
&\quad + \sum_{\ell'\ne\ell} q_{\ell\ell'}(y)\,[f(y,\ell')-f(y,\ell)].
\end{aligned}
$$
其中最后一项是离散切换生成元（对 $\ell$ 方向的差分算子），前三项是给定模式下的 Lévy 跳扩散生成元。

#### 1.3.2 Itô 公式与 Dynkin 公式（证明生成元）
设 $f\in C^2$ 且满足适当增长条件，则跳扩散 Itô 公式给出
$$
\begin{aligned}
 f(Y_t,\ell)-f(Y_0,\ell)
 &= \int_0^t \nabla f(Y_{s^-},\ell)\cdot b(Y_{s^-},\ell)\,ds
 +\tfrac12\int_0^t \mathrm{Tr}(\Sigma\Sigma^\top\nabla^2 f)(Y_{s^-},\ell)\,ds \\
 &\quad +\int_0^t \nabla f(Y_{s^-},\ell)\cdot \Sigma(Y_{s^-},\ell)\,dW_s \\
 &\quad +\int_0^t\int \big(f(Y_{s^-}+z,\ell)-f(Y_{s^-},\ell)\big)\,\tilde N(ds,dz) \\
 &\quad +\int_0^t\int \big(f(Y_{s^-}+z,\ell)-f(Y_{s^-},\ell)-\mathbf{1}_{\{\lVert z\rVert\le 1\}}\nabla f(Y_{s^-},\ell)\cdot z\big)\,\nu_{\ell}(dz)\,ds.
\end{aligned}
$$
对上述等式取期望并利用鞅项期望为 0，即得 Dynkin 公式
$$
\mathbb{E}_x[f(X_t)] = f(x) + \mathbb{E}_x\int_0^t \mathcal{G}f(X_s)\,ds.
$$
这从严格意义上刻画了生成元的形式。

#### 1.3.3 分数阶拉普拉斯与 α-稳定：Fourier 推导
当 $\nu(dz)=c_{d,\alpha}\lVert z\rVert^{-d-\alpha}dz$（对称 α-稳定）时，跳跃算子等价于分数阶拉普拉斯：对 Schwartz 函数 $f$，
$$
\widehat{(-\Delta)^{\alpha/2}f}(\xi)=\lVert\xi\rVert^{\alpha}\hat f(\xi).
$$
另一方面，α-稳定半群满足 $\widehat{P_tf}(\xi)=e^{-t c\lVert\xi\rVert^{\alpha}}\hat f(\xi)$，对 $t$ 在 0 处求导得
$$
\widehat{\mathcal{A}f}(\xi) = -c\lVert\xi\rVert^{\alpha}\hat f(\xi),\quad \Rightarrow\quad \mathcal{A}=-c(-\Delta)^{\alpha/2}.
$$
因此在 α-稳定驱动下，生成元中非局部项可写为 $-(-\Delta)^{\alpha/2}$（差一个正比例系数）。

为与“Lévy 测度积分形式”的生成元一致，常用分数阶拉普拉斯的非局部表示：对足够光滑的 $f$，
$$
(-\Delta)^{\alpha/2}f(x)=C_{d,\alpha}\,\mathrm{P.V.}\int_{\mathbb{R}^d}\frac{f(x)-f(y)}{\lVert x-y\rVert^{d+\alpha}}\,dy
=C_{d,\alpha}\int_{\mathbb{R}^d\setminus\{0\}}\frac{f(x)-f(x+z)}{\lVert z\rVert^{d+\alpha}}\,dz,
$$
其中 P.V. 表示主值积分。

**引理 1.5（Fourier 符号与主值积分的一致性）** 对 Schwartz 函数 $f$，上式的 Fourier 变换满足
$$
\widehat{(-\Delta)^{\alpha/2}f}(\xi)=\lVert\xi\rVert^\alpha \hat f(\xi).
$$
**证明（可直接用于论文正文）**：对 $f$ 的平移差分作 Fourier 变换，
$$
\mathcal{F}[f(\cdot)-f(\cdot+z)](\xi)=(1-e^{i\xi\cdot z})\hat f(\xi).
$$
于是
$$
\widehat{(-\Delta)^{\alpha/2}f}(\xi)
=C_{d,\alpha}\left(\int_{\mathbb{R}^d\setminus\{0\}}\frac{1-e^{i\xi\cdot z}}{\lVert z\rVert^{d+\alpha}}\,dz\right)\hat f(\xi).
$$
括号内积分仅依赖于 $\lVert\xi\rVert$，通过变量代换 $z=\lVert\xi\rVert^{-1}w$ 得
$$
\int \frac{1-e^{i\xi\cdot z}}{\lVert z\rVert^{d+\alpha}}\,dz
=\lVert\xi\rVert^\alpha \int \frac{1-e^{i\frac{\xi}{\lVert\xi\rVert}\cdot w}}{\lVert w\rVert^{d+\alpha}}\,dw
=c_\alpha \lVert\xi\rVert^\alpha.
$$
取常数 $C_{d,\alpha}=c_\alpha^{-1}$ 即得结论。证毕。

#### 1.3.4 伴随算子与平稳方程（弱形式）
若平稳分布在每个模式上具有密度 $p_\infty(y,\ell)$（相对 Lebesgue 测度），则不变性条件
$$
\int \mathcal{G}f(y,\ell)\,p_\infty(y,\ell)\,dy=0
$$
可等价写为伴随方程 $\mathcal{G}^*p_\infty=0$。在 α-稳定情形下（形式上）得到耦合的分数阶 Fokker–Planck 系统：
$$
\begin{aligned}
0&= -\nabla\cdot (b_{\ell} p_\infty(\cdot,\ell)) + \tfrac12\nabla^2:(\Sigma_{\ell}\Sigma_{\ell}^\top p_\infty(\cdot,\ell)) - (-\Delta)^{\alpha/2}p_\infty(\cdot,\ell) \\
&\quad +\sum_{\ell'\ne \ell}\big(q_{\ell'\ell}p_\infty(\cdot,\ell')-q_{\ell\ell'}p_\infty(\cdot,\ell)\big).
\end{aligned}
$$
在论文中更严格的写法是：对任意测试函数 $\varphi$，有
$$
\sum_{\ell}\int \varphi(y,\ell)\,\mathcal{G}^*p_\infty(y,\ell)\,dy=0,
$$
即 $p_\infty$ 是该系统的弱解/分布解。

#### 1.3.5 马氏切换过程的平稳密度：耦合（分数阶）Fokker–Planck 系统
为突出“马氏切换 + α-稳定”的耦合结构，记 $p_i(y,t)$ 为 $(Y_t,L_t)$ 在模式 $i$ 上的时间边际密度（若存在）。形式上，它们满足耦合的 Kolmogorov 前向方程：
$$
\partial_t p_i
=\mathcal{L}_i^* p_i+\sum_{j\neq i}\Big(q_{ji}(y)p_j-q_{ij}(y)p_i\Big),\qquad i=1,\dots,M,
$$
其中 $\mathcal{L}_i^*$ 是固定模式下连续状态的伴随算子：在跳扩散情形
$$
\mathcal{L}_i^* p
 =-\nabla\cdot (b(\cdot,i)p)+\tfrac12\nabla^2:\big(\Sigma\Sigma^\top(\cdot,i)p\big)+\mathcal{I}_i^*p,
$$
而对称 α-稳定时 $\mathcal{I}_i^*p=-c_i(-\Delta)^{\alpha/2}p$。

**命题 1.6（总质量守恒）** 若 $p_i(\cdot,t)$ 在无穷远处衰减足够快，则
$$
\frac{d}{dt}\sum_{i=1}^M\int_{\mathbb{R}^d} p_i(y,t)\,dy=0.
$$
**证明**：对每个 $i$ 对前向方程积分，$\int \mathcal{L}_i^*p_i=0$；切换项在 $i$ 与 $j$ 方程间作为源/汇两两抵消。证毕。

在平稳态下，$p_i(y,t)\equiv p_{\infty,i}(y)$ 满足耦合稳态方程
$$
0=\mathcal{L}_i^* p_{\infty,i}+\sum_{j\neq i}\Big(q_{ji}(y)p_{\infty,j}-q_{ij}(y)p_{\infty,i}\Big),\qquad \sum_i\int p_{\infty,i}(y)\,dy=1.
$$
这给出“马氏切换过程的解和平稳密度”的统一 PDE 表达：平稳密度不再是单一方程，而是由 $M$ 个方程耦合而成。

#### 1.3.6 分数阶拉普拉斯算子的固定点问题：自洽稳态的一个典型推导
当漂移依赖于分布（或依赖于平稳均值等统计量）时，稳态方程将变为“未知量同时出现在算子与源项中”的自洽/固定点问题。以一类典型模型为例：
$$
dY_t = -\nabla U_{L_t}(Y_t)\,dt - \nabla (W*\rho_t)(Y_t)\,dt + dZ_t^{(\alpha,L_t)},\qquad \rho_t=\mathcal{L}(Y_t),
$$
其稳态密度 $\rho_\infty(y)=\sum_i p_{\infty,i}(y)$（或每个模式密度 $p_{\infty,i}$）满足形式方程
$$
0=\nabla\cdot\Big(\rho_\infty\nabla U_{\mathrm{eff}}[\rho_\infty]\Big)+c(-\Delta)^{\alpha/2}\rho_\infty,\qquad
U_{\mathrm{eff}}[\rho]=U+W*\rho.
$$
若引入分数阶 Green 核 $\mathcal{G}_\alpha$（在适当意义下 $(-\Delta)^{\alpha/2}\mathcal{G}_\alpha=\delta_0$），则可将上式形式改写为积分方程
$$
\rho = -\frac{1}{c}\,\mathcal{G}_\alpha*\nabla\cdot\Big(\rho\nabla U_{\mathrm{eff}}[\rho]\Big)=:T(\rho),
$$
即 $\rho$ 是映射 $T$ 的不动点。

**讨论（存在性的一条功能分析路线）**：  
（i）若 $U$ 强凸且 $W$ 足够小，可在 $L^p$ 或 $H^{\alpha/2}$ 上证明 $T$ 为压缩映射，从而由 Banach 不动点定理得到唯一稳态；  
（ii）一般情形可尝试证明 $T$ 在某个凸紧集上映射到自身且连续紧，从而用 Schauder 不动点定理给出存在性；  
（iii）HLS 不等式与分数阶 Sobolev 嵌入（1.6）用于控制卷积项与非局部能量，进而给出先验估计与紧性。
### 1.4 平稳分布的存在性：Foster–Lyapunov 条件与证明

本节给出“存在唯一平稳分布”的可验证条件，并给出相对完整的证明步骤。核心工具是 Meyn–Tweedie 理论中的 Foster–Lyapunov 漂移条件 + petite 集。

#### 1.4.1 Foster–Lyapunov 漂移条件
取函数 $V:\mathcal{X}\to[1,\infty)$。若存在常数 $c>0,d<\infty$ 与紧集 $K\subset\mathcal{X}$，使得
$$
\mathcal{G}V(x)\le -cV(x)+d\mathbf{1}_K(x),\qquad \forall x\in\mathcal{X},
$$
则称 $V$ 满足漂移条件。

在 Lévy/α-稳定噪声下，常用选择为 $V(y,\ell)=1+\lVert y\rVert^p$（$p>0$）。此时关键在于控制非局部项
$$
\mathcal{I}V(y):=\int\Big(V(y+z)-V(y)-\mathbf{1}_{\{\lVert z\rVert\le 1\}}\nabla V(y)\cdot z\Big)\nu(dz).
$$

**引理 1.1（跳跃项对多项式 Lyapunov 的估计）** 设 $V(y)=1+\lVert y\rVert^p$，且 Lévy 测度满足 $\int_{\lVert z\rVert>1}\lVert z\rVert^p\nu(dz)<\infty$。则存在常数 $C$ 使
$$
\mathcal{I}V(y)\le C(1+\lVert y\rVert^{p-1}).
$$
若进一步为对称 α-稳定情形，则可用缩放性质得到更精细的界：当 $p>\alpha$ 时
$$
\mathcal{I}V(y)\le C(1+\lVert y\rVert^{p-\alpha}).
$$
*证明要点*：将积分分解为小跳 $\lVert z\rVert\le 1$ 与大跳 $\lVert z\rVert>1$ 两部分；小跳部分用二阶 Taylor 展开控制，大跳部分用 $\lVert y+z\rVert^p\le C(\lVert y\rVert^p+\lVert z\rVert^p)$ 控制。

#### 1.4.2 平稳分布存在唯一性定理
**定理 1.2（正 Harris 递归与唯一平稳分布）** 假设：
1) 过程 $X_t$ 为强马氏 Feller 过程；
2) 存在 $V$ 满足漂移条件；
3) 子水平集 $C_R:=\{x:V(x)\le R\}$ 为 petite 集；
4) 过程不可约且非周期。
则 $X_t$ 正 Harris 递归，存在唯一平稳分布 $\pi$，并且
$$
\lVert \mu P_t-\pi\rVert_{TV}\to 0\quad (t\to\infty)
$$
对任意初始分布 $\mu$ 成立。

*证明（结构化）*：
- 第一步：漂移条件推出从任意初始点出发，$V(X_t)$ 的期望会被拉回紧集附近，进而推出返回时间的有限性；
- 第二步：petite 集保证存在“局部小量化”的下界（minorization），从而可应用 regeneration 技术；
- 第三步：不可约 + minorization 排除多个不变类，得到唯一平稳分布；
- 第四步：利用再生结构推得全变差收敛。

为便于后续章节直接引用，本节在漂移条件框架下给出较为详细的“返乡性与存在性”推导，并在此基础上给出唯一性与收敛性的标准结论。

**命题 1.7（漂移条件推出返乡性：首达时间的期望界）** 令 $C_R=\{x:V(x)\le R\}$，并设
$$
\tau_{C_R}:=\inf\{t\ge 0: X_t\in C_R\}.
$$
若漂移条件 $\mathcal{G}V\le -cV+d\mathbf{1}_{C_R}$ 成立，则对任意初值 $x$，
$$
\mathbb{E}_x[\tau_{C_R}] \le \frac{V(x)}{c}+\frac{d}{c}.
$$
**证明**：对停止时刻 $\tau_{C_R}\wedge t$ 应用 Dynkin 公式，
$$
\mathbb{E}_x[V(X_{\tau_{C_R}\wedge t})]
=V(x)+\mathbb{E}_x\int_0^{\tau_{C_R}\wedge t}\mathcal{G}V(X_s)\,ds
\le V(x)-c\,\mathbb{E}_x\int_0^{\tau_{C_R}\wedge t}V(X_s)\,ds+d\,\mathbb{E}_x[\tau_{C_R}\wedge t].
$$
注意到在 $[0,\tau_{C_R})$ 上有 $V(X_s)>R\ge 1$，故
$$
c\,\mathbb{E}_x\int_0^{\tau_{C_R}\wedge t}V(X_s)\,ds
\ge c\,\mathbb{E}_x[\tau_{C_R}\wedge t].
$$
并且 $V(X_{\tau_{C_R}\wedge t})\ge 1$，从而
$$
\mathbb{E}_x[\tau_{C_R}\wedge t]\le \frac{V(x)-1}{c}+\frac{d}{c}.
$$
令 $t\to\infty$ 并用单调收敛定理得到结论。证毕。

**命题 1.8（Krylov–Bogoliubov：不变分布的存在性）** 设过程是 Feller，且存在函数 $V$ 使
$$
\sup_{T>0}\frac1T\int_0^T \mathbb{E}_x[V(X_t)]\,dt<\infty.
$$
则经验测度 $\mu_T:=\frac1T\int_0^T \delta_{X_t}\,dt$ 的族在适当拓扑下紧，从而存在子列 $\mu_{T_n}\Rightarrow \pi$，极限 $\pi$ 是不变分布：$\pi P_t=\pi$。

**证明要点**：漂移条件给出对 $\mathbb{E}V(X_t)$ 的积分型控制，继而由 Markov 不等式可得
$$
\mu_T(\{V>R\})\le \frac{1}{R}\mu_T(V),
$$
从而子水平集 $\{V\le R\}$ 作为紧集（通常需再验证 $V$ 的“紧致性”）提供紧性；Feller 性保证极限测度不变。

**唯一性与收敛性**：
（i）若 $C_R$ 为 petite 集，则存在 $t_0>0$ 与概率测度 $\nu$ 使 $P_{t_0}(x,\cdot)\ge \varepsilon \nu(\cdot)$（$x\in C_R$），这给出再生结构；  
（ii）再结合不可约性，可排除多个不变类，从而不变分布唯一；  
（iii）再生结构可推出全变差收敛；若还能得到更强的漂移（如 $\mathcal{G}V\le -\lambda V +b\mathbf{1}_{C_R}$），可推出几何遍历与指数速率。

#### 1.4.3 petite 集的来源：扩散非退化与 Lévy 支撑
- 若扩散项非退化（$\Sigma\Sigma^\top$ 在紧集上一致正定），则强 Feller + topological irreducibility 往往可推得 petite 集性质。
- 若扩散退化但 Lévy 测度在所有方向上有支撑（例如 α-稳定），跳跃也可提供“遍历性驱动”，在很多情形下仍可构造 petite 集。

综上，只要漂移具有足够耗散性（抵消跳跃造成的外推），并配合适当的不可约性/小集条件，即可保证平稳分布存在且唯一。
### 1.5 切换稳定性：共用/切换 Lyapunov 函数与平稳性

由于 $L_t$ 的切换会改变动力学算子，平稳性分析需要考虑“在切换下仍然稳定”。常用两条路线：共用 Lyapunov 与多 Lyapunov + 驻留时间。

#### 1.5.1 共用 Lyapunov 函数（common Lyapunov）
若存在单个 $V(y)$ 与常数 $c,d$ 使得对所有模式 $\ell$ 都有
$$
\mathcal{G}_\ell V(y)\le -cV(y)+d,
$$
其中 $\mathcal{G}_\ell$ 是固定模式下（不含切换项）的生成元，则无论切换如何发生，整体过程仍满足漂移条件，从而直接得到平稳分布存在唯一性。

#### 1.5.2 多 Lyapunov 与平均驻留时间（average dwell-time）
更一般地，允许不同模式各自稳定但收敛率不同，甚至允许少数模式不稳定。

**定理 1.3（多 Lyapunov + 平均驻留时间的稳定性原理）** 设存在函数族 $\{V_\ell\}$ 与常数 $\lambda_\ell>0$、$c$ 使
$$
\mathcal{G}_\ell V_\ell(y)\le -\lambda_\ell V_\ell(y)+c,
$$
且切换时满足可比性：存在 $\kappa_{\ell\ell'}\ge 1$，使
$$
V_{\ell'}(y)\le \kappa_{\ell\ell'}V_\ell(y).
$$
若切换次数 $N(t)$ 满足平均驻留时间上界
$$
N(t)\le N_0 + \tfrac{t}{\tau_d},
$$
且 $\tau_d$ 足够大（切换不太频繁）使得整体放大效应被衰减抵消，则系统稳定并可推出存在平稳分布。

*证明要点*：在每个驻留区间上 $V_\ell$ 指数衰减；在切换时最多放大 $\kappa$ 倍。综合得到
$$
V_{L_t}(Y_t)\lesssim \exp\Big(-\int_0^t \lambda_{L_s}ds + \log\kappa\cdot N(t)\Big)V_{L_0}(Y_0)+\text{常数项},
$$
当 $\log\kappa\cdot N(t)$ 的增长被 $\int_0^t\lambda_{L_s}ds$ 压制时即可保证整体有界并回归紧集。

#### 1.5.3 模式链平稳分布与平均化观点
当 $L_t$ 独立于 $Y_t$ 且不可约时，$L_t$ 有平稳分布 $\pi_L$。在“快切换”极限下，可考虑平均化漂移
$$
\bar b(y)=\sum_{\ell\in\mathcal{L}}\pi_L(\ell)b(y,\ell),\qquad
\bar\Sigma\bar\Sigma^\top(y)=\sum_{\ell}\pi_L(\ell)\Sigma\Sigma^\top(y,\ell),
$$
并研究等效过程的平稳性，作为原系统平稳性的近似解释。
### 1.6 Sobolev 不等式、分数阶能量与平稳密度正则性

本节补充两条与“平稳密度存在且具有正则性”相关的技术线：分数阶 Sobolev 空间与 Sobolev/Poincaré 型不等式。

#### 1.6.1 分数阶 Sobolev 空间与 Gagliardo 半范数
对 $0<s<1$，分数阶 Sobolev 空间 $H^s(\mathbb{R}^d)$ 可用 Fourier 或 Gagliardo 半范数刻画：
$$
\lVert f\rVert_{H^s}^2 \asymp \lVert f\rVert_{L^2}^2 + \iint_{\mathbb{R}^d\times\mathbb{R}^d}\frac{|f(x)-f(y)|^2}{\lVert x-y\rVert^{d+2s}}dxdy.
$$
在 α-稳定情形 $s=\alpha/2$，非局部能量项与分数阶拉普拉斯自然对应：
$$
\langle (-\Delta)^{\alpha/2}f,f\rangle_{L^2} \asymp \iint \frac{|f(x)-f(y)|^2}{\lVert x-y\rVert^{d+\alpha}}dxdy.
$$

#### 1.6.2 Sobolev 嵌入与 Hardy–Littlewood–Sobolev (HLS) 不等式
若 $d>2s$，则 Sobolev 嵌入给出
$$
H^s(\mathbb{R}^d)\hookrightarrow L^p(\mathbb{R}^d),\qquad p=\frac{2d}{d-2s}.
$$
HLS 不等式进一步可控制分数积分算子，从而在处理非局部 PDE 的弱解存在性时提供紧性。

#### 1.6.3 Poincaré/谱间隙与指数收敛（思想框架）
若平稳分布 $\pi$ 满足 Poincaré 不等式
$$
\mathrm{Var}_\pi(f)\le C_P\,\mathcal{E}(f,f),
$$
其中 $\mathcal{E}$ 为 Dirichlet 形式（在 α-稳定驱动下对应分数阶能量），则可推出半群在 $L^2(\pi)$ 上指数收敛：
$$
\lVert P_tf-\pi(f)\rVert_{L^2(\pi)}\le e^{-t/C_P}\lVert f-\pi(f)\rVert_{L^2(\pi)}.
$$
进一步结合 minorization（Doeblin 型条件）可提升为全变差指数收敛，得到几何遍历性。该路线在第二章将与切换稳定性结合。

（实践关联）第四章需要比较“经验稳态分布 vs 目标分布”。Sobolev/能量视角提供可计算的距离（例如 MMD 或基于能量的度量），并解释为什么分数阶噪声会带来更强的尾部与多样性。

### 1.7 本章小结与与后续章节衔接
- Lévy/α-稳定噪声通过非局部生成元与分数阶拉普拉斯引入厚尾、多峰。
- Foster–Lyapunov + petite 集给出平稳分布存在唯一性的一般机制；切换稳定性需要共用或多 Lyapunov 技术。
- Sobolev 不等式与分数阶能量为“平稳密度正则性、谱间隙与收敛速率”提供功能分析工具。
第二章将把这些工具用于遍历性、混合速率、可达性与切换系统稳定性；第三章将基于遍历性发展参数估计与渐近理论。

## 第二章 混杂生成扩散的遍历性、切换稳定性与可达性

### 2.1 遍历性与混合性：定义层级
设 $P_t$ 为过程半群，$\pi$ 为平稳分布。

- **遍历定理（时间平均）**：若对任意 $f\in L^1(\pi)$，
  $$
  \frac{1}{T}\int_0^T f(X_t)dt \xrightarrow{a.s.} \int f\,d\pi,\qquad T\to\infty,
  $$
  则称过程遍历。
- **Harris 遍历**：若 $\lVert \mu P_t-\pi\rVert_{TV}\to 0$ 对任意初始分布 $\mu$ 成立。
- **几何遍历**：存在 $V\ge1$、$C>0$、$\rho\in(0,1)$ 使
  $$
  \lVert \delta_x P_t-\pi\rVert_{TV}\le C V(x)\rho^t.
  $$
- **多项式遍历**：在重尾跳跃下常见：$\lVert \delta_xP_t-\pi\rVert_{TV}\le C V(x)(1+t)^{-\beta}$。
- **混合系数**：$\beta(t)$、$\alpha(t)$ 等用来刻画相关性衰减，为第三章中心极限定理与渐近正态性服务。

#### 2.1.1 $V$-遍历、漂移条件与小集条件
为了把“存在平稳分布”推进到“收敛速率可控”，需要在第一章的 Foster–Lyapunov 漂移条件之外加入小集（minorization）结构。下面给出最常用的结论形式。

**定义 2.1（小集与 petite 集）** 设离散时间核 $P$（可取 $P=P_{t_0}$ 的 skeleton 核）。集合 $C$ 称为小集，若存在 $n\in\mathbb{N}$、$\varepsilon>0$ 与概率测度 $\nu$ 使
$$
P^n(x,\cdot)\ge \varepsilon\,\nu(\cdot),\qquad x\in C.
$$
petite 集是更一般的概念：允许用某个随机时间分布对 $P^n$ 的凸组合来得到类似下界。对强 Feller 且不可约的扩散（或非局部扩散）过程，紧集往往为 petite 集。

**定义 2.2（$V$-均匀遍历）** 给定 $V\ge 1$，若存在常数 $C>0$ 与 $\rho\in(0,1)$ 使对所有 $x$，
$$
\lVert P^n(x,\cdot)-\pi\rVert_V\le C V(x)\rho^n,
$$
其中
$$
\lVert \mu\rVert_V:=\sup_{|f|\le V}|\mu(f)|,
$$
则称链 $V$-均匀遍历。全变差几何遍历是取 $V\equiv 1$ 的特例。

**定理 2.1（漂移 + 小集 $\Rightarrow$ $V$-几何遍历）** 设存在函数 $V\ge 1$、常数 $\lambda\in(0,1)$、$b<\infty$ 与小集 $C$ 使
$$
PV(x)\le \lambda V(x)+b\mathbf{1}_C(x).
$$
若链 $\psi$-不可约且非周期，则存在唯一平稳分布 $\pi$，并且链 $V$-均匀几何遍历。

**证明（再生结构的主线）**：由小集条件可在每次进入 $C$ 后以概率 $\varepsilon$ “重新抽样”初值为 $\nu$，从而构造再生时刻 $\{T_k\}$。漂移条件给出返回时间矩的指数型控制（可由对 $V(X_{n\wedge\tau_C})$ 应用离散 Dynkin 公式得到），进而得到再生周期的几何尾界。最后把任意初值的链分解为“再生前段 + 独立同分布再生周期叠加”，即可推出 $V$-范数下的指数收敛。证毕。

#### 2.1.2 有限状态马氏链的指数混合与谱间隙
由于本文的离散模式 $L_t$（或 $r_t$）常取值于有限集合，其自身的混合性质可显式刻画，并将在切换系统的遍历性证明中反复使用。

**定理 2.2（有限状态连续时间链的指数收敛）** 设 $r_t$ 为不可约有限状态连续时间马氏链，平稳分布为 $\pi$。则存在常数 $C>0$ 与 $\lambda_Q>0$（由 $Q$ 的谱间隙决定）使
$$
\max_i \lVert \mathbb{P}(r_t\in\cdot\mid r_0=i)-\pi\rVert_{TV}\le C e^{-\lambda_Q t}.
$$
**证明要点**：$P(t)=e^{tQ}$ 的谱分解给出除 0 特征值之外其余特征值实部均为负；从而 $P(t)-\mathbf{1}\pi^\top$ 的算子范数指数衰减。利用范数等价（有限维）即可推出全变差界。证毕。

**推论 2.1（切换对连续状态的“随机环境”作用）** 在 $r_t$ 混合足够快时，连续状态的长程性质可在某些极限下被“平均化”漂移/扩散解释（参见 1.5.3）；而当 $r_t$ 混合较慢或存在不稳定模式时，需要多 Lyapunov 与驻留时间技术来保证整体遍历性。

本章在 Markov 切换 + Lévy/α-稳定驱动下讨论遍历速度：何时得到几何遍历，何时只能得到多项式遍历；并给出可达性/不可约性的常用验证手段。

### 2.2 不可约性与可达性（Reachability）

#### 2.2.1 可达性与 topological irreducibility
对任意开集 $O\subset\mathcal{X}$，若对所有初值 $x$ 存在 $t>0$ 使 $P_t(x,O)>0$，则称过程 topologically irreducible。该性质通常由两类机制保障：
- **扩散非退化**：若 $\Sigma\Sigma^\top$ 满秩并满足 Hörmander 括号条件，则转移密度正且光滑；
- **Lévy 支撑充分**：若 Lévy 测度在所有方向上有正质量（尤其 α-稳定），跳跃可“跨越”障碍，从而保证任何开集都有正概率到达。

#### 2.2.2 切换链的可达性
若 $L_t$ 的生成矩阵在任意 $y$ 下不可约，则标签空间可达。更强地，若存在 $\varepsilon>0$ 使 $q_{\ell\ell'}(y)\ge \varepsilon$（均匀不可约），则可在遍历性证明中获得更简单的 minorization。

### 2.3 切换系统稳定性：从稳定到遍历

#### 2.3.1 共用 Lyapunov 推出几何遍历（证明骨架）
若存在 $V$ 使对整体生成元成立漂移条件 $\mathcal{G}V\le -cV+d\mathbf{1}_K$，并且不可约/非周期，则由 Meyn–Tweedie 得到正 Harris 递归。若进一步满足“小集条件”（可由均匀不可约 + 强 Feller 导出），可推出几何遍历。

#### 2.3.2 多 Lyapunov + 平均驻留时间的稳定性到遍历性
当只有多 Lyapunov 条件可得时，可先证明 $V_{L_t}(Y_t)$ 的有界性与返乡性，再通过构造再生结构得到 Harris 递归。若切换过快会破坏几何遍历，得到的往往是多项式遍历。

### 2.4 α-稳定驱动下的遍历速度：几何 vs 多项式
在 α-稳定跳跃下，由于重尾，几何遍历需要更强的耗散漂移。一个典型充分条件是：存在 $c_0,c_1>0$ 使
$$
\langle y, b(y,\ell)\rangle \le c_1 - c_0\lVert y\rVert^{1+\alpha},
$$
这类“超线性耗散”可抵消重尾跳跃导致的远离趋势，从而恢复几何遍历。

若漂移仅线性耗散（OU 型），则对 α-稳定噪声往往得到稳态存在但混合速度为多项式（具体指数与 $\alpha$、漂移强度相关）。这一现象在语言应用中可解释为：重尾噪声提高多样性但可能减慢收敛到稳态的速度。

### 2.5 Sobolev/Dirichlet 形式方法：谱间隙与 Poincaré
在对称（或可对称化）情形下，生成元对应 Dirichlet 形式 $\mathcal{E}$。若能证明 Poincaré 不等式
$$
\mathrm{Var}_\pi(f)\le C\mathcal{E}(f,f),
$$
则得到谱间隙与 $L^2$ 指数收敛。对分数阶算子，$\mathcal{E}(f,f)$ 可取分数阶能量（见第一章）。

对切换系统，常用做法是把 Dirichlet 形式扩展为
$$
\mathcal{E}_{\text{switch}}(f,f)=\sum_{\ell}\mathcal{E}_\ell(f_\ell,f_\ell)+\tfrac12\sum_{\ell\ne\ell'}\int q_{\ell\ell'}(y)\,(f(y,\ell)-f(y,\ell'))^2\,\pi(dy,\ell),
$$
其中第二项体现模式差分的“离散能量”。若该能量控制方差，可获得整体谱间隙。

### 2.6 耦合方法与 Wasserstein 收缩（补充）
对扩散部分可用同步耦合，对跳跃部分可用 maximal coupling 或“共同跳 + 校正跳”。若能构造耦合使得
$$
\mathbb{E}\lVert X_t-Y_t\rVert \le e^{-\lambda t}\lVert X_0-Y_0\rVert,
$$
则得到 Wasserstein 指数收敛；结合 minorization 可进一步推出全变差收敛。

### 2.7 离散化遍历性：数值近似的长期性质
实践中需离散化：
$$
Y_{n+1}=Y_n+b(Y_n,L_n)\Delta+\Sigma(Y_n,L_n)\sqrt{\Delta}\xi_{n+1}+\Delta J_{n+1},
$$
其中 $\Delta J_{n+1}$ 用泊松采样或稳定噪声采样生成。需要证明离散链也满足类似漂移条件
$$
\mathbb{E}[V(X_{n+1})\mid X_n=x]\le (1-c\Delta)V(x)+d\Delta.
$$
该性质保证仿真得到的“经验平稳分布”能逼近连续模型的平稳分布。

### 2.8 本章小结
本章从不可约性/可达性、切换稳定性、α-稳定重尾影响、Sobolev–Dirichlet 能量与耦合方法等多角度刻画混杂生成扩散的遍历性质，为第三章的推断渐近理论提供必要的混合与极限定理前提。

## 第三章 一类混杂生成扩散的参数估计：MLE/EM、分数阶 DSM 与渐近理论

### 3.1 混杂几何布朗运动（马氏切换）

#### 3.1.1 引言与模型
本节以金融与混杂系统中经典的“马氏切换几何布朗运动”（Markov-modulated GBM）作为贯穿实例，用于连接本论文的长期性质分析（平稳性/遍历性）与统计推断（似然/滤波/EM）两条主线。与一般的混杂生成扩散相比，该模型具有“连续扩散 + 离散切换”的最简结构，许多关键结论都可写成可计算的边值问题或显式的估计公式。

我们研究资产价格过程 $\{S_t\}_{t\ge 0}$ 满足随机微分方程
$$
dS_t = \mu_{r_t} S_t\,dt + \sigma_{r_t} S_t\,dW_t.
$$
（3.1.1）
其中：
- $\{r_t\}_{t\ge 0}$ 为取值于有限状态空间 $\mathcal{L}=\{1,\dots,M\}$ 的连续时间马氏链，生成矩阵为 $Q=(q_{ij})$；
- $\mu_i,\sigma_i$ 为常数（或足够光滑的函数），并且 $\sigma_i>0$；
- $\{W_t\}_{t\ge0}$ 为一维标准布朗运动，且 $\{r_t\}$ 与 $\{W_t\}$ 相互独立。

为便于首出时/逃逸概率分析，令对数过程 $X_t=\ln S_t$。由伊藤公式得到
$$
dX_t = a_{r_t}\,dt + \sigma_{r_t}\,dW_t,\qquad a_i:=\mu_i-\tfrac12\sigma_i^2.
$$
（3.1.2）
于是 $X_t$ 是“马氏切换布朗运动”（Markov-modulated Brownian motion），其无穷小生成元对向量函数 $u=(u_1,\dots,u_M)^\top$ 作用为
$$
(\mathcal{A}u)_i(x)=\tfrac12\sigma_i^2 u_i''(x)+a_i u_i'(x)+\sum_{j\ne i} q_{ij}\big(u_j(x)-u_i(x)\big).
$$
（3.1.3）

**假设 3.1.1**（标准假设）
(A1) 马氏链不可约：对任意 $i\ne j$，存在路径使得 $i$ 可达 $j$；等价地，$Q$ 不可约。  
(A2) 独立性：$\{r_t\}$ 与 $\{W_t\}$ 独立。  
(A3) 参数有界：$|\mu_i|+\sigma_i \le C$，并且 $\min_i \sigma_i\ge \sigma_*>0$。
(A4) 首出问题初值：给定区间 $(a,b)$（$a<b$），取 $X_0=\ln S_0\in(a,b)$。

在假设 3.1.1 下，$(X_t,r_t)$ 是 $\mathbb{R}\times\mathcal{L}$ 上的强马氏过程。下面围绕 (i) 平均首出时，(ii) 尺度函数与逃逸概率，(iii) 参数估计，(iv) 隐状态估计四个问题展开。

**命题 3.1.1（生成元与鞅表征）** 设 $u=(u_1,\dots,u_M)^\top$，其中 $u_i\in C^2(\mathbb{R})$ 且满足适当增长条件，使得下式各项期望有限。则过程
$$
M_t:=u_{r_t}(X_t)-u_{r_0}(X_0)-\int_0^t (\mathcal{A}u)_{r_s}(X_s)\,ds
$$
是鞅，其中 $\mathcal{A}$ 由（3.1.3）给出。

**证明**：令 $I_t^{(i)}=\mathbf{1}_{\{r_t=i\}}$。记 $N_{ij}(t)$ 为从状态 $i$ 到 $j$ 的跳变计数过程（$i\neq j$），则
$$
N_{ij}(t)-\int_0^t q_{ij}I_s^{(i)}\,ds
$$
为鞅，并且
$$
dI_t^{(i)}=\sum_{j\neq i}\big(I_{t^-}^{(j)}\,dN_{ji}(t)-I_{t^-}^{(i)}\,dN_{ij}(t)\big).
$$
另一方面，对固定的 $i$，由 Itô 公式
$$
du_i(X_t)=u_i'(X_t)\,dX_t+\tfrac12 u_i''(X_t)\,d\langle X\rangle_t
=\big(a_{r_t}u_i'(X_t)+\tfrac12\sigma_{r_t}^2u_i''(X_t)\big)\,dt+\sigma_{r_t}u_i'(X_t)\,dW_t.
$$
对乘积 $I_t^{(i)}u_i(X_t)$ 应用乘积公式，并对 $i$ 求和（注意 $u_{r_t}(X_t)=\sum_i I_t^{(i)}u_i(X_t)$），漂移项可整理为
$$
\Big(\tfrac12\sigma_{r_t}^2 u_{r_t}''(X_t)+a_{r_t}u_{r_t}'(X_t)+\sum_{j\ne r_t} q_{r_tj}\big(u_j(X_t)-u_{r_t}(X_t)\big)\Big)\,dt,
$$
其余项为布朗鞅与跳鞅之和，从而得到所述鞅分解。证毕。

### 3.2 平均首出时（Mean First Exit Time）与 Poisson 边值问题
给定 $a<b$，定义首出时（对数域）
$$
\tau_{a,b}:=\inf\{t\ge 0: X_t\notin(a,b)\},
\qquad \tau_a:=\inf\{t\ge0:X_t=a\},\quad \tau_b:=\inf\{t\ge0:X_t=b\}.
$$
（3.2.1）
我们关心平均首出时函数
$$
u_i(x):=\mathbb{E}_{x,i}\big[\tau_{a,b}\big],\qquad (x,i)\in(a,b)\times\mathcal{L}.
$$
（3.2.2）

**命题 3.2.1**（Poisson 问题表征）在假设 3.1.1 下，$u=(u_1,\dots,u_M)^\top$ 是如下耦合 Poisson 边值问题的（足够光滑）解：
$$
\begin{cases}
\mathcal{A}u(x)=-\mathbf{1}, & x\in(a,b),\\
u_i(a)=u_i(b)=0, & i\in\mathcal{L},
\end{cases}
$$
（3.2.3）
其中 $\mathbf{1}=(1,\dots,1)^\top$。

**证明**：对任意足够光滑向量函数 $f$，由 Dynkin 公式（对 $X_t$ 的生成元 $\mathcal{A}$）有
$$
\mathbb{E}_{x,i}[f(X_{t\wedge\tau_{a,b}},r_{t\wedge\tau_{a,b}})]
= f(x,i)+\mathbb{E}_{x,i}\int_0^{t\wedge\tau_{a,b}}(\mathcal{A}f)_{r_s}(X_s)\,ds.
$$
（3.2.4）
令 $f=u$ 并假设 $\mathcal{A}u=-\mathbf{1}$，则右端积分变为 $-\mathbb{E}[t\wedge\tau_{a,b}]$。令 $t\to\infty$，并用边界条件 $u(X_{\tau_{a,b}},r_{\tau_{a,b}})=0$，得到 $u_i(x)=\mathbb{E}_{x,i}[\tau_{a,b}]$。反向方向（由定义推出 PDE）可通过小时间展开严格化。

**备注 3.2.1**：方程（3.2.3）是常系数二阶 ODE 的耦合系统，可写为矩阵形式
$$
\tfrac12\Sigma^2 u''(x) + A u'(x) + Q u(x) = -\mathbf{1},
$$
其中 $\Sigma^2=\mathrm{diag}(\sigma_1^2,\dots,\sigma_M^2)$，$A=\mathrm{diag}(a_1,\dots,a_M)$。可通过矩阵指数、特征分解或数值边值法（有限差分/谱方法）求解；在 $M=2$ 时可写出更显式的闭式表达。

#### 3.2.1 有界性、唯一性与先验估计
平均首出时不仅要满足边值问题，还应当验证其有界性与唯一性，以保证“由边值问题刻画首出时”在数学上闭合。

**引理 3.2.1（首出时有限性与粗界）** 在假设 3.1.1 下，对任意 $(x,i)\in(a,b)\times\mathcal{L}$，
$$
\mathbb{E}_{x,i}[\tau_{a,b}]<\infty,
$$
并且存在仅依赖于 $(a,b)$ 与 $\sigma_*:=\min_i\sigma_i$ 的常数 $C_{a,b,\sigma_*}$ 使得
$$
0\le u_i(x)\le C_{a,b,\sigma_*}\quad (x\in(a,b),\,i\in\mathcal{L}).
$$
**证明**：比较法即可。令 $\underline X_t$ 为常扩散系数 $\sigma_*$、漂移为 0 的布朗运动：$d\underline X_t=\sigma_*\,dW_t$。由扩散系数下界可构造耦合或利用比较不等式得到 $\tau_{a,b}\le_{\text{st}} \underline\tau_{a,b}$（直观上更大的扩散更快触边界；严格证明可用停时与二次变差比较）。而 $\underline\tau_{a,b}$ 的期望是经典结果：
$$
\mathbb{E}_x[\underline\tau_{a,b}]=\frac{(x-a)(b-x)}{\sigma_*^2},
$$
从而得到有限性与上界。证毕。

**定理 3.2.2（Poisson 边值问题的唯一性）** 若 $w=(w_1,\dots,w_M)^\top$ 在 $(a,b)$ 上二次连续可导且满足齐次边值问题
$$
\begin{cases}
\mathcal{A}w(x)=0,\quad x\in(a,b),\\
w_i(a)=w_i(b)=0,\quad i\in\mathcal{L},
\end{cases}
$$
则 $w\equiv 0$。从而（3.2.3）在适当函数类中至多有一个解。

**证明（最大值原理的一种写法）**：设 $w$ 非零。取
$$
m:=\max_{i}\sup_{x\in[a,b]} w_i(x).
$$
若 $m>0$，则存在 $(x_0,i_0)\in(a,b)\times\mathcal{L}$ 使 $w_{i_0}(x_0)=m$。在该点有 $w'_{i_0}(x_0)=0$ 且 $w''_{i_0}(x_0)\le 0$。又因 $w_j(x_0)-w_{i_0}(x_0)\le 0$ 对所有 $j$ 成立，代入生成元得
$$
0=(\mathcal{A}w)_{i_0}(x_0)=\tfrac12\sigma_{i_0}^2 w''_{i_0}(x_0)+a_{i_0}w'_{i_0}(x_0)+\sum_{j\ne i_0}q_{i_0j}(w_j(x_0)-w_{i_0}(x_0))\le 0.
$$
因此上式右端各项必须同时取等号，进而推出 $w''_{i_0}(x_0)=0$ 且 $w_j(x_0)=w_{i_0}(x_0)$ 对所有 $j$ 成立。结合马氏链不可约性可将“在某点所有分量取相同最大值”传播到整个区间（严格化可用连通性与强最大值原理对弱耦合系统的推广），最终与边界条件 $w_i(a)=w_i(b)=0$ 矛盾。故 $m\le 0$。同理对 $-w$ 得到 $\max_i\sup_x(-w_i(x))\le 0$，因此 $w\equiv 0$。证毕。

#### 3.2.2 一阶系统化与矩阵指数表示
为给出通解结构并便于数值求解，将（3.2.3）化为一阶线性系统。令 $v(x)=u'(x)$，并记
$$
D:=\tfrac12\Sigma^2=\mathrm{diag}\big(\tfrac12\sigma_1^2,\dots,\tfrac12\sigma_M^2\big).
$$
则（3.2.3）等价于
$$
\begin{cases}
u'(x)=v(x),\\
v'(x)=-D^{-1}\big(A v(x)+Q u(x)+\mathbf{1}\big),
\end{cases}
$$
写成增广向量 $y(x)=(u(x),v(x))^\top\in\mathbb{R}^{2M}$ 的形式：
$$
y'(x)=\mathsf{M}\,y(x)+\mathsf{b},
$$
其中
$$
\mathsf{M}=\begin{pmatrix}
0 & I\\
-D^{-1}Q & -D^{-1}A
\end{pmatrix},
\qquad
\mathsf{b}=\binom{0}{-D^{-1}\mathbf{1}}.
$$
因此通解满足（矩阵指数的变参数公式）
$$
y(x)=e^{\mathsf{M}(x-a)}y(a)+\int_a^x e^{\mathsf{M}(x-s)}\mathsf{b}\,ds.
$$
由于边界条件要求 $u(a)=0,u(b)=0$，而 $y(a)=(0,v(a))^\top$ 的未知量仅为 $v(a)\in\mathbb{R}^M$，从而可把边值问题归结为线性方程组：令 $\Pi_u$ 表示取 $y$ 的前 $M$ 个分量的投影，则
$$
0=u(b)=\Pi_u e^{\mathsf{M}(b-a)}\binom{0}{v(a)}+\Pi_u\int_a^b e^{\mathsf{M}(b-s)}\mathsf{b}\,ds,
$$
从而在唯一性成立时可以解出 $v(a)$，再由上式得到 $u(x)$。该表示为数值算法（矩阵指数/边值求解器）的理论基础。

#### 3.2.3 二次特征值问题与指数型基解
当系数为常数时，齐次系统的解可用指数函数展开。考虑齐次方程
$$
D u''(x)+A u'(x)+Q u(x)=0.
$$
取试探解 $u(x)=e^{\lambda x}\xi$（$\xi\in\mathbb{R}^M$ 非零），代入得二次特征值问题（quadratic eigenvalue problem, QEP）
$$
F(\lambda)\xi=0,\qquad F(\lambda):=D\lambda^2+A\lambda+Q.
$$
因此特征根由标量方程
$$
\det F(\lambda)=0
$$
给出。由于 $\det F(\lambda)$ 是次数 $2M$ 的多项式（一般情形下），其根（计重数）共有 $2M$ 个，记为 $\{\lambda_k\}_{k=1}^{2M}$，对应右特征向量 $\{\xi_k\}$。在“根不退化且特征向量构成完备系统”的典型情形下，齐次解可表示为
$$
u_{\mathrm{hom}}(x)=\sum_{k=1}^{2M} c_k e^{\lambda_k x}\xi_k.
$$
对非齐次项 $-\mathbf{1}$，可另取一个特解 $u_{\mathrm{part}}(x)$（例如利用一阶系统化的变参数公式构造），从而得到一般解
$$
u(x)=u_{\mathrm{part}}(x)+\sum_{k=1}^{2M} c_k e^{\lambda_k x}\xi_k,
$$
再由边界条件 $u(a)=u(b)=0$ 解出系数 $c_k$。

**备注 3.2.2（零根的出现）** 由于生成矩阵 $Q$ 的行和为 0，必有 $Q\mathbf{1}=0$，从而 $F(0)\mathbf{1}=0$。因此 $\lambda=0$ 总是 $\det F(\lambda)=0$ 的根之一，对应特征向量 $\mathbf{1}$。这与“在无边界约束时，常数函数属于齐次算子的核”相一致；在有边界条件时，该方向被边界条件排除。

#### 3.2.4 三状态链的典型设定与特征多项式（概述）
当 $M=3$ 时，$\det F(\lambda)=0$ 是 6 次方程。若进一步给定 $Q$ 的具体结构（例如某些“特殊 $n$ 状态马氏链生成矩阵”的情形），可对根的分布给出更细致的讨论，并据此写出显式的指数展开形式。这类分析的关键在于把 QEP 线性化为 $2M$ 维的广义特征值问题（companion linearization），并利用数值特征值算法稳定地求出 $\{\lambda_k,\xi_k\}$，再通过边界条件求解线性方程组获得系数。

### 3.3 尺度函数、逃逸概率与 Laplace 变换

#### 3.3.1 尺度函数（非切换与切换的对比）
先回顾非切换一维扩散 $dX_t=a\,dt+\sigma\,dW_t$ 的尺度密度 $s(x)$：
$$
 s(x)=\exp\left(-\int^x \frac{2a}{\sigma^2}\,dy\right)=\exp\left(-\frac{2a}{\sigma^2}x\right),\qquad S(x)=\int^x s(y)\,dy.
$$
（3.3.1）
尺度函数可将“先达上边界”的概率写成比例形式。对于马氏切换情形，尺度函数需推广为向量/矩阵对象。常用做法是引入**矩阵尺度函数（scale matrix）**：定义矩阵函数 $W(x)$ 使其 Laplace 变换满足
$$
\int_0^\infty e^{-\theta x} W(x)\,dx = F(\theta)^{-1},\quad \theta>\theta_0,
$$
（3.3.2）
其中 $F(\theta)=\tfrac12\Sigma^2\theta^2 + A\theta + Q$ 称为 Markov 加性过程的矩阵指数。该对象可用于系统化表达首达/首出概率与 Laplace 变换（本论文以 PDE 形式为主，矩阵尺度函数作为补充观点）。

#### 3.3.2 逃逸概率（上边界先达概率）
定义逃逸概率
$$
 p_i(x):=\mathbb{P}_{x,i}(\tau_b<\tau_a).
$$
（3.3.3）
则 $p=(p_1,\dots,p_M)^\top$ 满足耦合 Dirichlet 问题
$$
\begin{cases}
\mathcal{A}p(x)=0, & x\in(a,b),\\
 p_i(a)=0,\ p_i(b)=1, & i\in\mathcal{L}.
\end{cases}
$$
（3.3.4）
**证明**：令
$$
\tau:=\tau_{a,b},\qquad \tau_n:=\inf\{t\ge 0:|X_t|\ge n\}.
$$
对函数 $p$ 应用命题 3.1.1 的鞅表征到停止时刻 $t\wedge\tau\wedge\tau_n$，并用 $\mathcal{A}p=0$ 得到
$$
\mathbb{E}_{x,i}\big[p_{r_{t\wedge\tau\wedge\tau_n}}(X_{t\wedge\tau\wedge\tau_n})\big]=p_i(x).
$$
令 $t\to\infty$ 后再令 $n\to\infty$（利用 $p$ 有界、$\tau<\infty$ 几乎必然，配合支配收敛），得到
$$
p_i(x)=\mathbb{E}_{x,i}\big[p_{r_\tau}(X_\tau)\big].
$$
由边界条件可知，当 $\tau=\tau_a$ 时 $X_\tau=a$ 且 $p_{r_\tau}(a)=0$；当 $\tau=\tau_b$ 时 $X_\tau=b$ 且 $p_{r_\tau}(b)=1$。因此
$$
p_i(x)=\mathbb{P}_{x,i}(\tau_b<\tau_a),
$$
即（3.3.3）。反向方向（由概率定义推出 PDE）可由小时间展开严格化：对足够小的 $h>0$，将事件在 $[0,h]$ 上按是否发生切换分解并利用 Markov 性，再令 $h\downarrow 0$ 得到生成元方程。证毕。

**解的结构（指数展开）**：由于（3.3.4）为齐次常系数耦合二阶方程组，其解可用 3.2.3 的二次特征值问题表示。具体而言，齐次方程 $Du''+Au'+Qu=0$ 的基解为 $e^{\lambda_k x}\xi_k$，从而
$$
p(x)=\sum_{k=1}^{2M} c_k e^{\lambda_k x}\xi_k,
$$
并由边界条件 $p(a)=0,p(b)=\mathbf{1}$ 解出系数 $c_k$。唯一性可用与定理 3.2.2 相同的最大值原理证明。

#### 3.3.3 首出时的 Laplace 变换
对 $\lambda>0$，定义
$$
 v_i(x;\lambda):=\mathbb{E}_{x,i}\big[e^{-\lambda \tau_{a,b}}\big].
$$
（3.3.5）
则 $v$ 满足
$$
\begin{cases}
 (\mathcal{A}v)(x)=\lambda v(x), & x\in(a,b),\\
 v_i(a)=v_i(b)=1,& i\in\mathcal{L}.
\end{cases}
$$
（3.3.6）
与 3.2.3 类似，若取指数型试探解 $v(x)=e^{\theta x}\xi$ 代入齐次方程 $(\mathcal{A}v)(x)=\lambda v(x)$，则得到带谱参数的二次特征值问题
$$
\big(D\theta^2+A\theta+Q-\lambda I\big)\xi=0,
$$
从而特征根由
$$
\det\big(D\theta^2+A\theta+Q-\lambda I\big)=0
$$
确定。相应地，$v$ 仍可表示为指数展开并由边界条件 $v(a)=v(b)=\mathbf{1}$ 解出系数。该表示在 barrier 期权定价与风险过程的首出时间 Laplace 变换计算中具有直接应用。
该特征值型边值问题在金融风险度量（如 barrier option）与极限理论中非常常见。

### 3.4 参数估计：离散观测下的极大似然与 EM
假设在等间隔时刻 $t_k=k\Delta$ 观测到价格 $S_{t_k}$，令对数收益
$$
 R_k:=\ln\frac{S_{t_{k+1}}}{S_{t_k}}=X_{t_{k+1}}-X_{t_k}.
$$
（3.4.1）
在给定 $r_{t_k}=i$ 且忽略区间内切换（或用小步长近似）时，有近似条件分布
$$
 R_k\mid r_{t_k}=i \sim \mathcal{N}(a_i\Delta,\,\sigma_i^2\Delta).
$$
（3.4.2）

#### 3.4.1 完整观测下的极大似然（连续时间）
先讨论理想化情形：连续观测 $(X_t,r_t)_{t\in[0,T]}$。记在状态 $i$ 的占用时间与“加权增量”
$$
T_i:=\int_0^T \mathbf{1}_{\{r_t=i\}}\,dt,\qquad
\Delta X_i:=\int_0^T \mathbf{1}_{\{r_t=i\}}\,dX_t.
$$
由（3.1.2）可得
$$
\Delta X_i=\int_0^T \mathbf{1}_{\{r_t=i\}}a_i\,dt+\int_0^T \mathbf{1}_{\{r_t=i\}}\sigma_i\,dW_t=a_iT_i+\sigma_i B_i,
$$
其中 $B_i:=\int_0^T \mathbf{1}_{\{r_t=i\}}\,dW_t$ 为均值 0、方差 $T_i$ 的高斯随机变量。条件于路径 $r_{[0,T]}$，参数 $(a_i,\sigma_i)$ 的对数似然可写为
$$
\ell(a,\sigma\mid X,r)= -\frac12\sum_{i=1}^M\left(\frac{1}{\sigma_i^2}\int_0^T \mathbf{1}_{\{r_t=i\}}\big(dX_t-a_i\,dt\big)^2 + T_i\log\sigma_i^2\right)+\text{常数}.
$$
对 $a_i$ 与 $\sigma_i^2$ 求一阶条件得到
$$
\hat a_i=\frac{\Delta X_i}{T_i},\qquad
\hat\sigma_i^2=\frac{1}{T_i}\int_0^T \mathbf{1}_{\{r_t=i\}}\big(dX_t-\hat a_i\,dt\big)^2=\frac{1}{T_i}\int_0^T \mathbf{1}_{\{r_t=i\}}\,d\langle X\rangle_t,
$$
并据 $\mu_i=a_i+\tfrac12\sigma_i^2$ 还原 $\mu_i$ 的估计。

对切换强度矩阵 $Q$，完整观测下对数似然为
$$
\ell(Q\mid r)=\sum_{i\neq j} N_{ij}\log q_{ij}-\sum_{i} q_i T_i,\qquad q_i:=-q_{ii}=\sum_{j\neq i}q_{ij},
$$
从而得到极大似然估计
$$
\hat q_{ij}=\frac{N_{ij}}{T_i},\qquad i\neq j.
$$

#### 3.4.2 离散观测与隐状态下的近似似然
在实际数据中通常只能离散观测 $S_{t_k}$ 或 $X_{t_k}$，并且 $r_{t_k}$ 往往不可观测。采用小步长近似（3.4.2）时，发射密度为
$$
f_i(R_k)=\frac{1}{\sqrt{2\pi\sigma_i^2\Delta}}\exp\left(-\frac{(R_k-a_i\Delta)^2}{2\sigma_i^2\Delta}\right),
$$
隐状态转移概率为 $P(\Delta)=e^{Q\Delta}$。因此观测似然可写为对隐状态路径求和的形式：
$$
L(\theta;R_{0:n-1})=\sum_{r_{t_0},\dots,r_{t_n}}\pi_0(r_{t_0})\prod_{k=0}^{n-1} P_{r_{t_k}r_{t_{k+1}}}(\Delta)\,f_{r_{t_{k+1}}}(R_k),
$$
并可由前向算法在 $O(M^2n)$ 时间内计算。

于是观测似然为“马氏链调制的高斯混合”，可用 HMM 的前向后向算法计算。记 $\gamma_k(i)=\mathbb{P}(r_{t_k}=i\mid R_{0:k})$，$\xi_k(i,j)=\mathbb{P}(r_{t_k}=i,r_{t_{k+1}}=j\mid R_{0:n})$。

**M 步的闭式更新（常用参数化）**：先估计 $a_i$ 与 $\sigma_i^2$，再还原 $\mu_i=a_i+\tfrac12\sigma_i^2$：
$$
 \hat a_i = \frac{\sum_{k=0}^{n-1}\gamma_k(i)\,R_k}{\Delta\sum_{k=0}^{n-1}\gamma_k(i)},
 \qquad
 \hat\sigma_i^2 = \frac{\sum_{k=0}^{n-1}\gamma_k(i)\,(R_k-\hat a_i\Delta)^2}{\Delta\sum_{k=0}^{n-1}\gamma_k(i)},
 \qquad
 \hat\mu_i=\hat a_i+\tfrac12\hat\sigma_i^2.
$$
（3.4.3）
对切换强度 $Q$，若能获得连续时间停留时间统计，则可用
$$
 \hat q_{ij}=\frac{\mathbb{E}[N_{ij}]}{\mathbb{E}[T_i]},\quad i\ne j.
$$
（3.4.4）
其中 $N_{ij}$ 为从 $i$ 到 $j$ 的跳变次数，$T_i$ 为在状态 $i$ 的总停留时间。在离散观测下通常先估计转移矩阵 $P\approx e^{Q\Delta}$，再用矩阵对数或近似得到 $Q$。

### 3.5 隐藏状态估计：滤波（filtering）与平滑（smoothing）
当 $r_t$ 不可观测时，核心任务是给出 $\pi_k(i)=\mathbb{P}(r_{t_k}=i\mid R_{0:k})$ 的递推。

- **前向滤波递推（离散 HMM）**：设 $P_{ij}=P_{ij}(\Delta)=\mathbb{P}(r_{t_{k+1}}=j\mid r_{t_k}=i)$，并令发射密度
$$
f_i(R_k)=\frac{1}{\sqrt{2\pi\sigma_i^2\Delta}}\exp\left(-\frac{(R_k-a_i\Delta)^2}{2\sigma_i^2\Delta}\right).
$$
定义未归一化前向变量 $\alpha_k(i):=\mathbb{P}(R_{0:k-1},r_{t_k}=i)$，则递推为
$$
\alpha_{k+1}(j)=\Big(\sum_{i=1}^M \alpha_k(i)P_{ij}\Big)f_j(R_k),\qquad
\pi_{k+1}(j)=\frac{\alpha_{k+1}(j)}{\sum_{m=1}^M \alpha_{k+1}(m)}.
$$
（3.5.1）

- **后向递推与平滑**：定义后向变量 $\beta_k(i):=\mathbb{P}(R_{k:n-1}\mid r_{t_k}=i)$，则
$$
\beta_k(i)=\sum_{j=1}^M P_{ij} f_j(R_k)\beta_{k+1}(j),\qquad \beta_n(i)\equiv 1.
$$
平滑概率与两步后验概率分别为
$$
\gamma_k(i):=\mathbb{P}(r_{t_k}=i\mid R_{0:n-1})=\frac{\alpha_k(i)\beta_k(i)}{\sum_{m=1}^M \alpha_k(m)\beta_k(m)},
$$
$$
\xi_k(i,j):=\mathbb{P}(r_{t_k}=i,r_{t_{k+1}}=j\mid R_{0:n-1})
=\frac{\alpha_k(i)P_{ij}f_j(R_k)\beta_{k+1}(j)}{\sum_{i',j'}\alpha_k(i')P_{i'j'}f_{j'}(R_k)\beta_{k+1}(j')}.
$$
其中 $\{\gamma_k\}$ 与 $\{\xi_k\}$ 是 EM 更新（3.4.3）–（3.4.4）的充分统计量。

**备注 3.5.1（连续观测滤波）**：若连续观测 $X_t$ 且各状态具有相同扩散系数（$\sigma_i\equiv\sigma$），则滤波可写成 Wonham 滤波 SDE；若 $\sigma_i$ 不同，需要更一般的非线性滤波（Zakai/Kushner–Stratonovich 形式）。在资源受限的实验实现中，本文采用离散观测情形的 HMM 滤波与 EM 框架。

当 $\sigma_i\equiv\sigma$ 时，Wonham 滤波可写得更为具体。令 $\pi_t^i:=\mathbb{P}(r_t=i\mid \mathcal{F}_t^X)$，并记条件漂移的加权平均
$$
\bar a_t:=\sum_{i=1}^M \pi_t^i a_i.
$$
定义创新过程
$$
dI_t:=dX_t-\bar a_t\,dt,
$$
则 $I_t$ 是 $\mathcal{F}_t^X$-布朗运动，并且滤波概率满足
$$
d\pi_t^i=\sum_{j=1}^M \pi_t^j q_{ji}\,dt+\frac{1}{\sigma^2}\pi_t^i\big(a_i-\bar a_t\big)\,dI_t,\qquad i=1,\dots,M.
$$
该方程清晰展示了两类信息来源：马氏链自身的转移（第一项）与连续观测带来的似然校正（第二项）。

### 3.6 观测方案与可辨识性
我们考虑三类常见观测：
1. **连续观测（理想化）**：观测整个路径 $(Y_t,L_t)_{t\in[0,T]}$；
2. **离散观测（常见）**：观测 $\{(Y_{t_k}, \text{可能的 }L_{t_k})\}_{k=0}^n$，$t_k=k\Delta$；
3. **部分观测（嵌入场景）**：仅观测文本，经编码器映射为 $\hat Y_{t_k}$，$L_{t_k}$ 隐含。

可辨识性通常需要：不同参数 $\theta$ 导致不同的转移核/平稳分布；在切换系统中，还需要标签链的统计信息（例如停留时间分布）辅助区分。

### 3.7 扩散部分的似然：Girsanov 视角（连续观测）
在无跳跃或跳跃部分已知时，连续观测下扩散的对数似然可由 Girsanov 定理给出：设
$$
 dY_t = b_\theta(Y_t,L_t)dt + \Sigma(Y_t,L_t)dW_t,
$$
则相对参考漂移 $b_0$ 的似然比满足
$$
\log\frac{d\mathbb{P}_\theta}{d\mathbb{P}_0}
=\int_0^T \big(\Sigma^{-1}(b_\theta-b_0)\big)^\top dW_t
-\tfrac12\int_0^T \lVert \Sigma^{-1}(b_\theta-b_0)\rVert^2 dt.
$$
该形式为第三章的 LAN 展开与 Fisher 信息推导提供基础。

### 3.8 跳跃部分：Lévy 测度与准似然
含跳跃时，完整似然包含 Lévy 测度参数。若能观测跳跃时刻与跳幅，则完整数据对数似然可写为
$$
\log L(\theta)=\sum_{k}\log \nu_{\theta}(\Delta_k\mid L_{T_k^-}) - \int_0^T\int \nu_{\theta}(dz\mid L_s)\,ds + \text{扩散项} + \text{切换项}.
$$
在仅离散观测下，常用两类策略：
- **准似然（small-\Delta 近似）**：用稳定分布或正态-稳定混合近似一步转移密度；
- **特征函数法**：利用 α-稳定过程的特征函数估计 $\alpha,\gamma$ 等参数。

### 3.9 α-稳定参数的估计：经验特征函数（ECF）
对对称 α-稳定增量 $\Delta J_k$，其特征函数为
$$
\varphi(u)=\exp(-\gamma^\alpha |u|^\alpha).
$$
设经验特征函数 $\hat\varphi_n(u)=\frac1n\sum_{k=1}^n e^{iu\Delta Y_k}$，则可通过回归
$$
\log(-\log|\hat\varphi_n(u)|) \approx \alpha\log|u| + \alpha\log\gamma
$$
估计 $(\alpha,\gamma)$。该方法实现简单，适合资源受限的实验；其渐近性质依赖于混合条件与 CLT。

### 3.10 Markov 切换的 EM：前向后向与粒子平滑
当 $L_{t_k}$ 隐含时，可将模型视作连续状态 HMM：
- E 步：计算 $\mathbb{P}(L_{t_k}=\ell\mid Y_{0:n})$ 与 $\mathbb{P}(L_{t_k}=\ell,L_{t_{k+1}}=\ell'\mid Y_{0:n})$；
- M 步：在加权充分统计量下更新 $Q$、漂移/扩散/跳跃参数。

在高维嵌入空间或非高斯噪声下，前向后向可用粒子滤波近似：
$$
\hat p(L_{t_k}=\ell\mid Y_{0:k}) \propto \sum_{i} w_k^{(i)}\mathbf{1}\{L_k^{(i)}=\ell\}.
$$

### 3.11 分数阶 DSM：与生成元/Dirichlet 形式的一致性
在嵌入空间直接建模密度困难，采用 score matching/DSM 学习 score 函数 $s^*(x)=\nabla\log p_\infty(x)$。在 α-稳定噪声下，非局部算子提示应采用分数阶能量正则：
$$
\mathcal{J}(s)=\mathbb{E}\|s_\theta(X_t,t)-s^*(X_t,t)\|^2 + \lambda\,\|s_\theta\|_{H^{\alpha/2}}^2.
$$
Sobolev 正则确保估计函数具有足够平滑性，并与第二章的谱间隙/集中不等式相匹配。

### 3.12 渐近理论：遍历 + 鞅 CLT + LAN（证明框架）
在几何遍历或足够的混合条件下，对加性泛函
$$
S_T=\int_0^T g(X_t)dt
$$
成立中心极限定理。进一步，极大似然的 LAN 展开可写为
$$
\log L_T(\theta_0+h/\sqrt{T})-\log L_T(\theta_0)
= h^\top \Delta_T - \tfrac12 h^\top I(\theta_0)h + o_p(1),
$$
其中 $\Delta_T$ 是鞅项归一化极限，$I(\theta_0)$ 为信息矩阵。

证明要点：
1) 利用遍历性将经验平均替换为 $\pi$ 下期望；
2) 将得分函数表示为鞅 + 可忽略项；
3) 对鞅使用 Rebolledo 或 Martingale FCLT；
4) 用可辨识性确保 $I(\theta_0)$ 正定。

### 3.13 贝叶斯推断与一致性（要点）
选择先验 $p(\theta)$ 后，后验
$$
\Pi(d\theta\mid\mathcal{D})\propto L(\theta;\mathcal{D})\,p(\theta)\,d\theta.
$$
后验一致性一般需要：KL 支撑、测试函数存在、先验对真参数邻域赋正质量；在切换+重尾情形下可通过构造 sieve 与控制似然比实现。

### 3.14 分数阶拉普拉斯逆问题与可辨识性（补充）
若可估计平稳密度 $\hat p$，则可将参数恢复视为非局部反问题：
$$
(-\Delta)^{\alpha/2} p_\infty + \nabla\cdot (b p_\infty) = S(y),
$$
其中 $S$ 含切换项。可构造变分目标
$$
\min_{b,\alpha} \int \big|(-\Delta)^{\alpha/2} \hat p + \nabla\cdot (b\hat p) - \hat S\big|^2 dy + \lambda\|b\|_{H^s}^2,
$$
从而得到正则化估计。可辨识性需要额外约束（例如 $b$ 的参数化结构或多模式观测）。

### 3.15 本章小结
本章首先以带马氏切换的混杂几何布朗运动为核心例子，围绕平均首出时、尺度函数/逃逸概率、参数估计与隐藏状态估计给出可操作的数学刻画；
随后从更一般的混杂生成扩散出发，讨论连续观测与离散观测下的似然构造、EM/粒子平滑、分数阶（α-稳定）噪声下的 DSM 与 Sobolev 正则、以及 LAN/CLT 驱动的渐近理论与贝叶斯一致性。

第三章在第二章遍历性与集中性质的基础上，给出了切换+Lévy/α-稳定模型的主要估计路线：扩散部分 Girsanov 似然、跳跃部分特征函数法与准似然、切换隐变量的 EM/粒子平滑，以及与分数阶能量一致的 DSM 框架，并给出渐近正态性与贝叶斯一致性的证明路线。

