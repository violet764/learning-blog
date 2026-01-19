Makemore输入一个文本文件，每行假定为一个训练项目，并生成更多类似内容。底层是自回归字符级语言模型，拥有从双字母表到Transformer（正如GPT所见）的多种模型选择。例如，我们可以给它一个名字数据库，Makemore会生成一些听起来像名字但并非已有名字的酷炫婴儿名字创意。或者如果我们输入一个公司名称数据库，就能生成一个公司名称的新想法。或者我们直接输入有效的拼字词，生成类似英语的咿呀学语。
此项目根据<a href="https://github.com/karpathy/makemore">makemore</a>项目开发。
实现Bigram（一个字符通过计数查找表预测下一个字符）
MLP，遵循Bengio等人2003年。
CNN，继DeepMind WaveNet 2016（进行中）之后
RNN，遵循Mikolov等人2010
LSTM，遵循Graves等人2014
GRU，遵循Kyunghyun Cho等人2014年。
Transformer，遵循Vaswani等人，2017年等方法

点击 <a href="./file/name.txt" download>name.txt</a> 下载数据集

# Bigram

```python
import torch
import matplotlib.pyplot as plt

# -------------------------- 1. 加载数据 + 初始化核心变量 --------------------------
# 读取姓名文件 (name.txt 每行一个姓名)
words = open('name.txt', 'r', encoding='utf-8').read().splitlines()

# 初始化 27x27 频次矩阵 (26个字母 + 1个特殊符号 . ，共27个)
N = torch.zeros((27, 27), dtype=torch.int32)

# 提取所有出现的字符并排序，构建 字符<->索引 映射字典
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}  # 字符转索引: a/z → 1~26
stoi['.'] = 0                               # 特殊符号 . 对应索引0，代表【开头/结尾】
itos = {i:s for s,i in stoi.items()}        # 索引转字符: 0→.  1→a  反向映射

# -------------------------- 2. 核心：统计Bigram二元字符对的频次 --------------------------
# Bigram：就是「相邻的两个字符」，统计 字符A后面跟字符B 的出现次数
for w in words:
    chs = ['.'] + list(w) + ['.']  # 给每个姓名前后加上起止符 .  
    for ch1, ch2 in zip(chs, chs[1:]): # 遍历每一对相邻字符
        ix1 = stoi[ch1]  # 转成索引
        ix2 = stoi[ch2]
        N[ix1, ix2] += 1 # 频次+1  N[行,列] = ch1后面跟ch2的次数

# -------------------------- 3. 热力图可视化：直观展示字符频次分布 --------------------------
plt.figure(figsize=(16, 16))
plt.imshow(N, cmap='Blues')  # 热力图：颜色越深 = 出现频次越高

# 给每个单元格填充 【字符对】和【频次数字】
for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')  # 显示 字符对
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')  # 显示 频次值
        
plt.axis('off')  # 隐藏坐标轴
plt.show()
```  


<img src="./images/bigram_frequency.png" style="max-width: 100%; height: auto;">


```python
# -------------------------- 4. 构建Bigram概率矩阵 --------------------------
# 频次 → 概率，核心公式：P(下一个字符 | 当前字符)
# 加1平滑：防止出现概率为0的情况，避免后续采样时报错
P = (N + 1).float()  # 所有频次+1
P /= P.sum(1, keepdim=True)  # 按行归一化，每行的概率之和=1

# -------------------------- 5. 核心：基于概率采样 → 生成新的姓名/文本 --------------------------
g = torch.Generator().manual_seed(2147483647)  # 固定随机种子，结果可复现

for _ in range(20):  # 生成20个姓名
    out = []
    ix = 0  # 初始索引为0，对应字符 . ，代表【姓名开头】
    while True:
        p = P[ix]  # 获取当前字符对应的下一个字符的概率分布
        # 从概率分布中随机采样1个字符的索引
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        if ix == 0:  # 如果采样到0 → 对应字符 . ，代表【姓名结束】
            break
        out.append(itos[ix])  # 把索引转回字符，存入结果列表
    print(''.join(out))  # 拼接字符，输出最终生成的姓名
```

```text
cexze
momasurailezitynn
konimittain
llayn
ka
```

接下来评估学习质量,使用MLE来评估概率

```python
loglikelyhood = 0.0
n=0
for w in words[:3]:
    word = ['.'] + list(w) + ['.']
    for ch1,ch2 in zip(word,word[1:]):
        ix1 = str2idx[ch1]
        ix2 = str2idx[ch2]
        prob = P[ix1,ix2]
        logprob = torch.log(prob)
        loglikelyhood += logprob
        n+=1
        print(f"{ch1}{ch2} : {prob:.4f} {logprob:.4f}")
print(f"总对数似然值: {loglikelyhood:.4f}")
print(f"平均对数似然值: {loglikelyhood/n:.4f}")
```
以上方法的概率保存在一个表格中，接下来使用似然函数训练神经网络来预测模型

[MLP方法](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)

```python

```
