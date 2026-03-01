# 分词技术详解

## 章节概述

分词（Tokenization）是大语言模型处理文本的第一步，也是至关重要的一步。它将原始文本转换为模型可理解的离散token序列，直接影响模型的词汇表大小、表示能力、训练效率和推理质量。本章从分词的基本概念出发，深入解析BPE、WordPiece、SentencePiece、BBPE等主流分词算法的数学原理与实现细节，帮助读者全面理解分词技术的演进脉络与实践要点。

## 分词概述

### 为什么需要分词

自然语言文本是连续的字符序列，而神经网络模型需要处理离散的数值输入。分词的核心作用是将连续文本切分为有限的离散单元（token），并建立token与数值ID的映射关系。

**分词需要解决的核心问题：**

| 问题 | 描述 | 影响 |
|------|------|------|
| 词汇表大小 | token种类越多，embedding矩阵越大 | 模型参数量、内存占用 |
| OOV问题 | 未见过的词如何处理 | 模型泛化能力 |
| 子词分割 | 如何平衡词级与字符级 | 表示效率与语义完整性 |
| 多语言支持 | 不同语言的分词差异 | 跨语言迁移能力 |

**分词与模型的关系：**

```
原始文本 "Hello, world!"
    ↓ 分词器
Token序列 ["Hello", ",", "world", "!"]
    ↓ 词汇表映射
ID序列 [15496, 11, 995, 0]
    ↓ Embedding层
向量序列 [d_model维向量]
```

### 分词类型概览

按照切分粒度，分词方法可分为三类：

```
文本: "unhappiness"

┌─────────────────────────────────────────────────────┐
│  词级分词 (Word-level)                                │
│  ["unhappiness"]                                     │
│  ✅ 语义完整  ❌ 词汇表巨大  ❌ OOV问题严重            │
├─────────────────────────────────────────────────────┤
│  字符级分词 (Character-level)                         │
│  ["u", "n", "h", "a", "p", "p", "i", "n", "e", "s", "s"]│
│  ✅ 无OOV  ✅ 词汇表小  ❌ 序列过长  ❌ 语义稀疏        │
├─────────────────────────────────────────────────────┤
│  子词级分词 (Subword-level)                           │
│  ["un", "happiness"] 或 ["un", "happi", "ness"]      │
│  ✅ 平衡词汇表与序列长度  ✅ 缓解OOV  ✅ 语义可组合    │
└─────────────────────────────────────────────────────┘
```

## 基于词的分词

### 基本原理

词级分词将文本按空格和标点切分为独立的词单元，是最直观的分词方式。

```python
def word_tokenize(text):
    """简单的词级分词器"""
    import re
    # 按空格和标点切分
    tokens = re.findall(r'\w+|[^\w\s]', text.lower())
    return tokens

# 示例
text = "Hello, how are you?"
tokens = word_tokenize(text)
print(tokens)  # ['hello', ',', 'how', 'are', 'you', '?']
```

### 优点与缺点

**优点：**
- 📌 **语义完整**：每个token对应完整的语义单元
- 📌 **序列较短**：相同文本产生的token数最少
- 📌 **解释性强**：人可直观理解token含义

**缺点：**
- ⚠️ **词汇表巨大**：英语词汇量可达数十万
- ⚠️ **OOV问题严重**：新词、专有名词、拼写变体无法处理
- ⚠️ **形态变化冗余**："run"、"runs"、"running"被视为不同token

**适用场景：**
- 领域受限、词汇量可控的应用
- 对语义完整性要求极高的场景
- 与传统NLP方法（如TF-IDF）配合使用

## 基于字符的分词

### 基本原理

字符级分词将文本拆分为最小的字符单元，每个字符（或字节）作为独立token。

```python
def char_tokenize(text):
    """字符级分词器"""
    return list(text)

# 示例
text = "Hello"
tokens = char_tokenize(text)
print(tokens)  # ['H', 'e', 'l', 'l', 'o']

# 字节级字符分词
def byte_tokenize(text):
    """字节级分词器"""
    return list(text.encode('utf-8'))

byte_tokens = byte_tokenize("你好")
print(byte_tokens)  # [228, 189, 160, 229, 165, 189]
```

### 优点与缺点

**优点：**
- 📌 **无OOV问题**：任意文本都可编码
- 📌 **词汇表极小**：通常 < 300（覆盖所有Unicode字符或字节）
- 📌 **跨语言通用**：无需针对语言定制

**缺点：**
- ⚠️ **序列过长**：相同文本产生大量token
- ⚠️ **语义稀疏**：单个字符语义信息有限
- ⚠️ **计算开销大**：长序列增加Transformer的计算复杂度 $O(n^2)$

**适用场景：**
- 多语言模型
- 字符级别的生成任务（如拼写纠错）
- 作为子词分词的基础（字节级）

## BPE算法详解

### 算法原理

**Byte Pair Encoding (BPE)** 是一种基于统计的子词分词算法，最初用于数据压缩，后被引入NLP领域。其核心思想是**迭代合并最高频的字符对**，直到达到目标词汇表大小。

**算法步骤：**

```
输入: 语料库, 目标词汇表大小 V
输出: 子词词汇表

1. 初始化: 将所有词拆分为字符序列，统计词频
2. 统计所有相邻字符对的出现频率
3. 选择频率最高的字符对，合并为新token
4. 更新语料库中的所有该字符对
5. 重复步骤2-4，直到词汇表大小达到V
```

### 数学形式化

给定语料库 $\mathcal{C}$，设词 $w$ 的字符序列为 $(c_1, c_2, ..., c_n)$，词频为 $f(w)$。

字符对 $(c_i, c_{i+1})$ 的合并频率定义为：

$$
\text{freq}(c_i, c_{i+1}) = \sum_{w \in \mathcal{C}} f(w) \cdot \text{count}_w(c_i, c_{i+1})
$$

其中 $\text{count}_w(c_i, c_{i+1})$ 是字符对在词 $w$ 中出现的次数。

每次迭代选择：

$$
(c^*, c'^*) = \arg\max_{(c_i, c_{i+1})} \text{freq}(c_i, c_{i+1})
$$

### BPE训练实现

```python
import re
from collections import defaultdict

class BPETokenizer:
    """BPE分词器实现"""
    
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.merges = []  # 合并规则
        self.vocab = {}   # 词汇表
        
    def _get_word_freqs(self, corpus):
        """统计词频，将词拆分为字符序列"""
        word_freqs = defaultdict(int)
        for text in corpus:
            words = re.findall(r'\w+|[^\w\s]', text.lower())
            for word in words:
                # 添加结束符 </w>
                word_tuple = tuple(word) + ('</w>',)
                word_freqs[word_tuple] += 1
        return word_freqs
    
    def _get_pair_freqs(self, word_freqs):
        """统计所有字符对的频率"""
        pair_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            if len(word) < 2:
                continue
            for i in range(len(word) - 1):
                pair = (word[i], word[i+1])
                pair_freqs[pair] += freq
        return pair_freqs
    
    def _merge_pair(self, word_freqs, pair):
        """合并指定的字符对"""
        new_word_freqs = {}
        bigram = pair
        replacement = pair[0] + pair[1]
        
        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == bigram[0] and word[i+1] == bigram[1]:
                    new_word.append(replacement)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_freqs[tuple(new_word)] = freq
        
        return new_word_freqs
    
    def train(self, corpus):
        """训练BPE模型"""
        word_freqs = self._get_word_freqs(corpus)
        
        # 初始化词汇表（所有单个字符）
        vocab = set()
        for word in word_freqs:
            for char in word:
                vocab.add(char)
        
        print(f"初始词汇表大小: {len(vocab)}")
        
        # 迭代合并
        iteration = 0
        while len(vocab) < self.vocab_size:
            pair_freqs = self._get_pair_freqs(word_freqs)
            if not pair_freqs:
                break
                
            # 找到最高频字符对
            best_pair = max(pair_freqs, key=pair_freqs.get)
            
            # 合并
            word_freqs = self._merge_pair(word_freqs, best_pair)
            self.merges.append(best_pair)
            
            # 更新词汇表
            new_token = best_pair[0] + best_pair[1]
            vocab.add(new_token)
            
            iteration += 1
            if iteration % 100 == 0:
                print(f"迭代 {iteration}, 新token: '{new_token}', 频率: {pair_freqs[best_pair]}")
        
        self.vocab = sorted(vocab, key=lambda x: len(x), reverse=True)
        print(f"最终词汇表大小: {len(self.vocab)}")
        
        return self
    
    def tokenize(self, text):
        """使用训练好的BPE进行分词"""
        words = re.findall(r'\w+|[^\w\s]', text.lower())
        tokens = []
        
        for word in words:
            word_tokens = list(word) + ['</w>']
            
            # 应用合并规则
            for merge in self.merges:
                i = 0
                while i < len(word_tokens) - 1:
                    if word_tokens[i] == merge[0] and word_tokens[i+1] == merge[1]:
                        word_tokens = word_tokens[:i] + [merge[0] + merge[1]] + word_tokens[i+2:]
                    else:
                        i += 1
            
            tokens.extend(word_tokens)
        
        return tokens

# 训练示例
corpus = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "the cat and the dog are friends",
    "friends help friends",
    "cats and dogs are animals"
]

bpe = BPETokenizer(vocab_size=50)
bpe.train(corpus)

# 测试分词
test_text = "the cat sat on the mat"
tokens = bpe.tokenize(test_text)
print(f"\n测试文本: '{test_text}'")
print(f"分词结果: {tokens}")
```

### BPE编码与解码

```python
def encode(self, text):
    """将文本编码为token ID序列"""
    tokens = self.tokenize(text)
    return [self.vocab.index(t) if t in self.vocab else self.vocab.index('<unk>') for t in tokens]

def decode(self, ids):
    """将token ID序列解码为文本"""
    tokens = [self.vocab[i] for i in ids]
    text = ''.join(tokens).replace('</w>', ' ')
    return text.strip()
```

## WordPiece算法

### 与BPE的区别

WordPiece与BPE的核心区别在于**合并策略**：

| 特性 | BPE | WordPiece |
|------|-----|-----------|
| 合并标准 | 频率最高 | 似然概率增益最大 |
| 合并标记 | 直接拼接 | 使用 `##` 前缀标记子词 |
| 目标函数 | 频率统计 | 语言模型概率 |

**WordPiece合并准则：**

选择使训练数据似然概率增加最多的字符对：

$$
\text{score}(A, B) = \frac{P(AB)}{P(A) \cdot P(B)}
$$

其中 $P(\cdot)$ 表示token在语料中的概率。

### WordPiece实现

```python
import math
from collections import defaultdict

class WordPieceTokenizer:
    """WordPiece分词器实现"""
    
    def __init__(self, vocab_size, unk_token='[UNK]'):
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.vocab = {}
        self.ids_to_tokens = {}
        
    def _get_word_freqs(self, corpus):
        """统计词频"""
        word_freqs = defaultdict(int)
        for text in corpus:
            words = text.lower().split()
            for word in words:
                word_freqs[word] += 1
        return word_freqs
    
    def _initialize_vocab(self, word_freqs):
        """初始化词汇表：所有单个字符"""
        vocab = set([self.unk_token])
        for word in word_freqs:
            for char in word:
                vocab.add(char)
        return vocab
    
    def _compute_pair_scores(self, word_freqs, vocab):
        """计算所有字符对的得分"""
        # 先统计单个token的频率
        token_freqs = defaultdict(int)
        for word, freq in word_freqs.items():
            tokens = self._tokenize_word(word, vocab)
            for token in tokens:
                token_freqs[token] += freq
        
        # 计算字符对得分
        pair_scores = {}
        pair_freqs = defaultdict(int)
        
        for word, freq in word_freqs.items():
            tokens = self._tokenize_word(word, vocab)
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                pair_freqs[pair] += freq
        
        for pair, pair_freq in pair_freqs.items():
            # score = P(AB) / (P(A) * P(B))
            score = pair_freq / (token_freqs[pair[0]] * token_freqs[pair[1]] + 1e-10)
            pair_scores[pair] = score
        
        return pair_scores
    
    def _tokenize_word(self, word, vocab):
        """使用贪心算法将词分解为子词"""
        tokens = []
        i = 0
        while i < len(word):
            # 贪心匹配最长子词
            found = False
            for j in range(len(word), i, -1):
                subword = word[i:j]
                if i > 0:
                    subword = '##' + subword
                if subword in vocab:
                    tokens.append(subword)
                    i = j
                    found = True
                    break
            
            if not found:
                tokens.append(self.unk_token)
                i += 1
        
        return tokens
    
    def train(self, corpus):
        """训练WordPiece模型"""
        word_freqs = self._get_word_freqs(corpus)
        vocab = self._initialize_vocab(word_freqs)
        
        print(f"初始词汇表大小: {len(vocab)}")
        
        iteration = 0
        while len(vocab) < self.vocab_size:
            pair_scores = self._compute_pair_scores(word_freqs, vocab)
            
            if not pair_scores:
                break
            
            # 选择得分最高的字符对
            best_pair = max(pair_scores, key=pair_scores.get)
            
            # 合并：使用 ## 前缀
            if best_pair[1].startswith('##'):
                new_token = best_pair[0] + best_pair[1][2:]
            else:
                new_token = best_pair[0] + best_pair[1]
            
            vocab.add(new_token)
            iteration += 1
            
            if iteration % 100 == 0:
                print(f"迭代 {iteration}, 新token: '{new_token}'")
        
        self.vocab = {token: i for i, token in enumerate(sorted(vocab))}
        self.ids_to_tokens = {i: token for token, i in self.vocab.items()}
        
        print(f"最终词汇表大小: {len(self.vocab)}")
        return self
    
    def tokenize(self, text):
        """分词"""
        words = text.lower().split()
        tokens = []
        
        for word in words:
            word_tokens = self._tokenize_word(word, set(self.vocab.keys()))
            tokens.extend(word_tokens)
        
        return tokens

# 示例
corpus = [
    "the cat sat on the mat",
    "unhappiness is a feeling",
    "happiness comes from within",
    "the unfairness of the situation"
]

wp = WordPieceTokenizer(vocab_size=60)
wp.train(corpus)

print("\n分词测试:")
test_text = "unhappiness"
tokens = wp.tokenize(test_text)
print(f"'{test_text}' -> {tokens}")
```

### WordPiece的 `##` 标记说明

`##` 前缀用于标记**非词首子词**：

```
词汇表: {"un", "##happi", "##ness", "happi", "happy", ...}

分词结果:
  "unhappiness" -> ["un", "##happi", "##ness"]
  "happiness"   -> ["happi", "##ness"]  (注意: happi没有##前缀，因为是词首)
```

这种设计的好处：
- 📌 区分词首与非词首位置
- 📌 保留词边界信息
- 📌 便于后续还原原始文本

## SentencePiece

### 核心思想

SentencePiece是一个**语言无关的子词分词框架**，解决了传统分词方法的两个问题：

1. **预处理依赖**：传统方法需要先进行空格切分、标点处理等语言相关预处理
2. **分词器与模型耦合**：分词结果依赖具体的语言规则

**SentencePiece的创新：**
- 将输入视为**原始Unicode字符序列**，空格也作为一个特殊字符 `_` (U+2581)
- 使用**无监督学习**自动学习分词规则
- 支持**BPE、Unigram、Char、Word**等多种算法

### 数学模型（Unigram Language Model）

SentencePiece的默认算法是**Unigram Language Model**，其概率模型为：

$$
P(X) = \prod_{i=1}^{M} p(x_i)
$$

其中 $X = (x_1, x_2, ..., x_M)$ 是一个分词方案，$p(x_i)$ 是子词 $x_i$ 的概率。

**训练目标**（最大似然估计）：

$$
\mathcal{L} = \sum_{s \in \mathcal{D}} \log P(s) = \sum_{s \in \mathcal{D}} \log \left( \sum_{X \in \mathcal{S}(s)} P(X) \right)
$$

其中 $\mathcal{S}(s)$ 是句子 $s$ 所有可能的分词方案集合。

**EM算法迭代：**
1. **E步**：计算每个子词的期望计数
2. **M步**：更新子词概率
3. **剪枝**：移除贡献小的子词

### SentencePiece使用示例

```python
# 安装: pip install sentencepiece

import sentencepiece as spm

# 训练模型
def train_sentencepiece(corpus_file, model_prefix, vocab_size=8000):
    """
    训练SentencePiece模型
    
    参数:
        corpus_file: 语料库文件路径
        model_prefix: 模型前缀
        vocab_size: 词汇表大小
    """
    spm.SentencePieceTrainer.train(
        input=corpus_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        # 算法选择: 'unigram' (默认), 'bpe', 'char', 'word'
        model_type='bpe',
        # 特殊token
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece='[PAD]',
        unk_piece='[UNK]',
        bos_piece='[BOS]',
        eos_piece='[EOS]',
        # 字符覆盖
        character_coverage=0.9995,  # 覆盖99.95%的字符
        # 其他参数
        normalization_rule_name='nmt_nfkc_cf',  # Unicode规范化
    )
    print(f"模型训练完成: {model_prefix}.model")

# 使用模型
def use_sentencepiece(model_path):
    """加载并使用SentencePiece模型"""
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    
    print(f"词汇表大小: {sp.get_piece_size()}")
    
    # 编码
    text = "Hello, how are you?"
    tokens = sp.encode(text, out_type=str)
    ids = sp.encode(text, out_type=int)
    
    print(f"\n原文: {text}")
    print(f"Tokens: {tokens}")
    print(f"IDs: {ids}")
    
    # 解码
    decoded = sp.decode(ids)
    print(f"解码: {decoded}")
    
    return sp

# 中文示例
def chinese_example():
    """中文分词示例"""
    # 假设已有中文语料文件
    # train_sentencepiece('chinese_corpus.txt', 'chinese_model', vocab_size=32000)
    
    # 使用预训练模型
    sp = spm.SentencePieceProcessor()
    # sp.load('chinese_model.model')
    
    text = "自然语言处理是人工智能的重要分支"
    # tokens = sp.encode(text, out_type=str)
    # print(f"中文分词: {tokens}")

# 示例
if __name__ == "__main__":
    # 创建示例语料
    corpus = [
        "Hello, how are you?",
        "I am fine, thank you.",
        "Natural language processing is fascinating.",
        "Machine learning models need lots of data.",
    ]
    
    # 写入临时文件
    with open('temp_corpus.txt', 'w', encoding='utf-8') as f:
        for line in corpus:
            f.write(line + '\n')
    
    # 训练模型
    train_sentencepiece('temp_corpus.txt', 'demo_model', vocab_size=500)
    
    # 使用模型
    sp = use_sentencepiece('demo_model.model')
```

### SentencePiece的优势

| 特性 | 描述 |
|------|------|
| **语言无关** | 不依赖语言特定的预处理规则 |
| **端到端训练** | 与模型训练解耦，可独立训练 |
| **可逆性** | 编码后可完美解码还原原始文本 |
| **空格处理** | 空格作为特殊字符 `_`，保留格式信息 |
| **多算法支持** | 支持BPE、Unigram、字符级、词级 |
| **高效性** | C++实现，速度快，内存占用小 |

## 字节级BPE (BBPE)

### 原理介绍

**Byte-level BPE (BBPE)** 是BPE的改进版本，被GPT-2、GPT-3、RoBERTa等模型采用。其核心思想是：

- **基础单元为字节**而非字符
- 将所有文本统一转换为UTF-8字节序列
- 词汇表初始大小固定为256（覆盖所有可能字节值）

**优势：**
- 📌 **彻底解决OOV问题**：任意Unicode文本都可编码
- 📌 **跨语言一致性**：不同语言使用相同的字节基础
- 📌 **词汇表可控**：最终词汇表大小 = 256 + merge次数

### 字节与字符的对比

```python
def compare_byte_char(text):
    """对比字节级与字符级分词"""
    print(f"原文: {text}")
    print(f"字符级: {list(text)}")
    print(f"字节数: {list(text.encode('utf-8'))}")
    print(f"字节十六进制: {[hex(b) for b in text.encode('utf-8')]}")

# 示例
compare_byte_char("Hello")
# 字符级: ['H', 'e', 'l', 'l', 'o']
# 字节数: [72, 101, 108, 108, 111]

compare_byte_char("你好")
# 字符级: ['你', '好']
# 字节数: [228, 189, 160, 229, 165, 189]  # 每个中文字符占3字节
```

### GPT-2 BBPE实现思路

```python
import json
from collections import defaultdict

class ByteBPETokenizer:
    """字节级BPE分词器（简化实现）"""
    
    def __init__(self):
        # 基础字节词汇表：256个字节
        self.byte_encoder = self._bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        
    def _bytes_to_unicode(self):
        """
        将字节映射为Unicode字符
        
        GPT-2使用一种巧妙的映射方式：
        - 可打印ASCII字符（33-126）保持不变
        - 其他字节映射到更高的Unicode码点
        """
        bs = (
            list(range(ord("!"), ord("~") + 1)) +      # 33-126: 可打印ASCII
            list(range(ord("¡"), ord("¬") + 1)) +      # 161-172
            list(range(ord("®"), ord("ÿ") + 1))        # 174-255
        )
        cs = bs[:]
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        cs = [chr(c) for c in cs]
        return dict(zip(bs, cs))
    
    def _get_pairs(self, word):
        """获取单词中的所有相邻字符对"""
        pairs = set()
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
    
    def tokenize(self, text, bpe_ranks):
        """
        使用BBPE进行分词
        
        参数:
            text: 输入文本
            bpe_ranks: BPE合并规则的优先级字典
        """
        # 转换为字节，再映射为Unicode字符
        tokens = [self.byte_encoder[b] for b in text.encode('utf-8')]
        
        word = tuple(tokens)
        pairs = self._get_pairs(word)
        
        if not pairs:
            return [text]
        
        while True:
            # 找到优先级最高的字符对
            bigram = min(pairs, key=lambda pair: bpe_ranks.get(pair, float('inf')))
            
            if bigram not in bpe_ranks:
                break
            
            first, second = bigram
            new_word = []
            i = 0
            
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                
                new_word.extend(word[i:j])
                i = j
                
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            word = tuple(new_word)
            
            if len(word) == 1:
                break
            pairs = self._get_pairs(word)
        
        return list(word)

# 使用Hugging Face tokenizers库（推荐）
def use_hf_byte_bpe():
    """使用Hugging Face的tokenizers库"""
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.trainers import BpeTrainer
    
    # 创建tokenizer
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = ByteLevel()
    
    # 训练
    trainer = BpeTrainer(
        vocab_size=30000,
        special_tokens=["<|endoftext|>"]
    )
    
    # tokenizer.train(["corpus.txt"], trainer)
    
    print("Hugging Face ByteLevel BPE tokenizer 示例")

# 示例
bbpe = ByteBPETokenizer()
print("字节映射表（部分）:")
sample_bytes = list(bbpe.byte_encoder.items())[:20]
for byte_val, unicode_char in sample_bytes:
    print(f"  字节 {byte_val:3d} -> 字符 '{unicode_char}'")
```

### GPT系列模型的词汇表特点

```
GPT-2词汇表大小: 50,257
├── 基础字节: 256
├── BPE合并: 50,000
└── 特殊token: 1 (<|endoftext|>)

GPT-3/GPT-4词汇表大小: ~100,000 (具体未公开)
├── 基础字节: 256
├── BPE合并: ~99,000+
└── 特殊token: 若干
```

## 分词对模型性能的影响

### 词汇表大小的权衡

词汇表大小是分词器的关键超参数，需要在多个因素间平衡：

```
词汇表大小的影响:

词汇表小:
├── 优点: Embedding参数少, 内存占用小
├── 缺点: 序列长度增加, 计算开销大
└── 极端情况: 字符级 (vocab_size ~ 256)

词汇表大:
├── 优点: 序列长度短, 计算效率高
├── 缺点: 参数量大, 稀疏token学习不充分
└── 极端情况: 词级 (vocab_size ~ 100,000+)

最优解: 子词级 (vocab_size ~ 30,000-100,000)
```

### 序列长度与计算复杂度

Transformer的自注意力机制复杂度为 $O(n^2)$，其中 $n$ 是序列长度。分词粒度直接影响序列长度：

```python
def compare_sequence_length(texts, tokenizers):
    """比较不同分词器的序列长度"""
    results = {}
    
    for name, tokenizer in tokenizers.items():
        lengths = []
        for text in texts:
            tokens = tokenizer(text)
            lengths.append(len(tokens))
        results[name] = {
            'avg_length': sum(lengths) / len(lengths),
            'max_length': max(lengths),
            'min_length': min(lengths)
        }
    
    return results

# 示例对比
texts = [
    "The transformer architecture revolutionized NLP.",
    "Natural language processing enables machines to understand text.",
    "Subword tokenization balances vocabulary size and sequence length."
]
```

### 分词与模型下游任务性能

不同分词策略对不同任务的影响：

| 任务类型 | 推荐分词策略 | 原因 |
|----------|--------------|------|
| 机器翻译 | BBPE/SentencePiece | 跨语言一致性，处理低频词 |
| 文本分类 | BPE/WordPiece | 语义完整，序列较短 |
| 命名实体识别 | WordPiece | 保留词边界信息 |
| 代码生成 | BBPE | 处理特殊字符、标识符 |
| 多语言任务 | SentencePiece/BBPE | 语言无关，统一处理 |

### 信息损失与分词粒度

```python
def analyze_tokenization_info():
    """分析分词的信息保留情况"""
    
    # 信息熵视角
    import math
    
    def entropy(tokens):
        """计算token序列的信息熵"""
        from collections import Counter
        token_counts = Counter(tokens)
        total = len(tokens)
        probs = [count/total for count in token_counts.values()]
        return -sum(p * math.log2(p) for p in probs)
    
    # 示例
    text = "unhappiness"
    
    # 字符级
    char_tokens = list(text)
    # 子词级（假设）
    subword_tokens = ["un", "happi", "ness"]
    # 词级
    word_tokens = ["unhappiness"]
    
    print(f"文本: {text}")
    print(f"字符级: {char_tokens}, 熵: {entropy(char_tokens):.2f}")
    print(f"子词级: {subword_tokens}, 熵: {entropy(subword_tokens):.2f}")
    print(f"词级: {word_tokens}, 熵: {entropy(word_tokens):.2f}")
```

## 中文分词的特殊考虑

### 中文分词的挑战

中文文本具有独特的语言特性，给分词带来特殊挑战：

```
挑战1: 无空格分隔
├── 英文: "I love natural language processing" (空格天然分隔)
└── 中文: "我爱自然语言处理" (需要识别词边界)

挑战2: 歧义切分
├── "南京市长江大桥"
│   ├── 正确: 南京市/长江大桥
│   └── 错误: 南京/市长/江大桥

挑战3: 新词识别
├── 网络新词: 内卷、躺平、yyds
├── 专业术语: 知识图谱、大语言模型
└── 命名实体: 人名、地名、机构名

挑战4: 字符级语义弱
├── "葡萄" vs "葡" + "萄" (单个字语义不完整)
└── "蝴蝶" vs "蝴" + "蝶"
```

### 中文分词策略

#### 策略1: 直接使用SentencePiece/BBPE

```python
def chinese_sentencepiece_example():
    """中文SentencePiece分词示例"""
    import sentencepiece as spm
    
    # 中文语料示例
    corpus = [
        "自然语言处理是人工智能的重要分支",
        "深度学习模型需要大量训练数据",
        "分词是文本处理的第一步",
        "中文分词比英文分词更加复杂",
    ]
    
    # 写入文件
    with open('chinese_corpus.txt', 'w', encoding='utf-8') as f:
        for line in corpus:
            f.write(line + '\n')
    
    # 训练SentencePiece模型
    spm.SentencePieceTrainer.train(
        input='chinese_corpus.txt',
        model_prefix='chinese_sp',
        vocab_size=1000,
        model_type='bpe',  # 或 'unigram'
        character_coverage=0.9995,  # 覆盖中文字符
    )
    
    # 使用模型
    sp = spm.SentencePieceProcessor()
    sp.load('chinese_sp.model')
    
    text = "自然语言处理很重要"
    tokens = sp.encode(text, out_type=str)
    ids = sp.encode(text, out_type=int)
    
    print(f"原文: {text}")
    print(f"Tokens: {tokens}")
    print(f"IDs: {ids}")
    
    return sp
```

#### 策略2: 结合传统中文分词工具

```python
def chinese_word_segmentation():
    """结合jieba等中文分词工具"""
    import jieba
    
    # 精确模式
    text = "我爱自然语言处理"
    words = jieba.lcut(text)
    print(f"精确模式: {words}")
    
    # 全模式（所有可能的切分）
    words_full = jieba.lcut(text, cut_all=True)
    print(f"全模式: {words_full}")
    
    # 搜索引擎模式
    words_search = jieba.lcut_for_search(text)
    print(f"搜索引擎模式: {words_search}")

# 中文分词 + BPE的混合策略
def hybrid_chinese_tokenization():
    """
    混合分词策略:
    1. 先使用jieba进行粗粒度分词
    2. 对每个词内部使用BPE进行子词分割
    """
    import jieba
    
    def hybrid_tokenize(text, bpe_tokenizer):
        words = jieba.lcut(text)
        tokens = []
        for word in words:
            word_tokens = bpe_tokenizer.tokenize(word)
            tokens.extend(word_tokens)
        return tokens
    
    return hybrid_tokenize
```

#### 策略3: 字符级 + 位置编码

```python
class ChineseCharTokenizer:
    """
    字符级中文分词器
    适用于: 命名实体识别、细粒度文本分析
    """
    
    def __init__(self, vocab_file=None):
        self.char_to_id = {}
        self.id_to_char = {}
        
        if vocab_file:
            self.load_vocab(vocab_file)
    
    def build_vocab(self, corpus):
        """构建字符词汇表"""
        chars = set(['[PAD]', '[UNK]', '[CLS]', '[SEP]'])
        for text in corpus:
            chars.update(text)
        
        self.char_to_id = {char: i for i, char in enumerate(sorted(chars))}
        self.id_to_char = {i: char for char, i in self.char_to_id.items()}
        
        return len(self.char_to_id)
    
    def tokenize(self, text, max_length=None):
        """字符级分词"""
        tokens = ['[CLS]'] + list(text) + ['[SEP]']
        
        if max_length and len(tokens) > max_length:
            tokens = tokens[:max_length-1] + ['[SEP]']
        
        return tokens
    
    def encode(self, text, max_length=None):
        """编码为ID序列"""
        tokens = self.tokenize(text, max_length)
        return [self.char_to_id.get(t, self.char_to_id['[UNK]']) for t in tokens]

# 使用示例
corpus = ["我爱自然语言处理", "深度学习很有趣"]
char_tokenizer = ChineseCharTokenizer()
vocab_size = char_tokenizer.build_vocab(corpus)
print(f"词汇表大小: {vocab_size}")
print(f"分词结果: {char_tokenizer.tokenize('我爱学习')}")
```

### 中文大模型的分词实践

主流中文大模型的分词策略：

| 模型 | 分词方式 | 词汇表大小 | 特点 |
|------|----------|------------|------|
| BERT-base-Chinese | 字符级 | 21,128 | 每个汉字一个token |
| RoBERTa-wwm-ext | 全词掩码 | 21,128 | 结合分词的预训练 |
| Chinese-LLaMA | SentencePiece | ~50,000 | 中英文混合 |
| ChatGLM | SentencePiece | ~150,000 | 多语言支持 |

```python
# 使用Hugging Face中文模型
def use_chinese_bert_tokenizer():
    """使用中文BERT分词器"""
    from transformers import BertTokenizer
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    text = "我爱自然语言处理"
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.encode(text)
    
    print(f"原文: {text}")
    print(f"Tokens: {tokens}")
    print(f"IDs: {ids}")
    print(f"解码: {tokenizer.decode(ids)}")
```

## 知识点间关联逻辑

### 分词技术演进脉络

```
词级分词 (Word-level)
    │ 问题: OOV严重、词汇表巨大
    ↓
字符级分词 (Character-level)
    │ 问题: 序列过长、语义稀疏
    ↓
子词分词 (Subword)
    │ 解决OOV + 平衡词汇表与序列长度
    ├── BPE: 基于频率合并
    ├── WordPiece: 基于似然增益
    └── Unigram: 基于概率模型
    ↓
字节级分词 (Byte-level)
    │ 完全消除OOV、跨语言一致
    └── BBPE: GPT系列采用
    ↓
统一框架 (SentencePiece)
    │ 语言无关、端到端训练
    └── 支持多种算法、广泛应用于现代LLM
```

### 各分词方法对比

```
                    词汇表大小    序列长度    OOV问题    多语言支持    训练复杂度
                    ─────────────────────────────────────────────────────────
词级分词            大          短         严重       差          低
字符级分词          小(~256)    长         无         好          无
BPE                 中等        中等       轻微       中等        中
WordPiece           中等        中等       轻微       中等        中
Unigram(SentencePiece) 中等     中等       轻微       好          高
BBPE(GPT系列)       中等        中等       无         好          中
```

### 与下游技术的关联

```
分词技术
    │
    ├──→ Embedding层
    │       词汇表大小决定Embedding矩阵维度: vocab_size × d_model
    │
    ├──→ 位置编码
    │       分词粒度影响序列长度，进而影响位置编码设计
    │
    ├──→ Transformer计算复杂度
    │       O(n²)，序列长度 n 直接影响计算开销
    │
    ├──→ 预训练目标
    │       BERT: WordPiece + Word Masking
    │       GPT: BBPE + Token Prediction
    │
    └──→ 推理效率
            词汇表大小影响softmax计算量
```

## 章节核心考点汇总

### 概念理解
- 分词在NLP流程中的位置与作用
- 词级、字符级、子词级分词的优缺点对比
- OOV问题及其解决方案
- 分词粒度与模型性能的关系

### 算法原理
- **BPE**: 频率驱动的迭代合并算法，能写出训练流程
- **WordPiece**: 似然增益驱动的合并，`##`标记的作用
- **Unigram**: 概率模型，EM算法训练思想
- **BBPE**: 字节级编码的优势，GPT系列的应用

### 实践技能
- 使用SentencePiece训练自定义分词器
- 使用Hugging Face tokenizers库
- 中文分词的特殊处理方法
- 评估分词质量的方法

### 数学基础
- BPE合并频率的计算
- WordPiece得分函数: $\frac{P(AB)}{P(A) \cdot P(B)}$
- Unigram模型的最大似然估计
- 信息熵在分词质量评估中的应用

## 学习建议与延伸方向

### 深入学习建议

1. **阅读经典论文**
   - Neural Machine Translation of Rare Words with Subword Units (BPE原论文)
   - Japanese and Korean Voice Search (WordPiece原论文)
   - SentencePiece: A simple and language independent approach

2. **动手实践**
   - 从零实现BPE算法
   - 训练多语言SentencePiece模型
   - 对比不同分词器在相同任务上的性能

3. **代码阅读**
   - Hugging Face tokenizers库源码
   - SentencePiece C++实现
   - GPT-2 tokenizer实现

### 延伸方向

- **高效分词**: GPU加速、并行分词
- **自适应分词**: 根据上下文动态调整分词
- **多模态分词**: 图像patch、音频帧的tokenization
- **检索增强**: 分词对检索效果的影响

### 实践项目建议

```python
# 项目1: 构建领域专用分词器
def build_domain_tokenizer():
    """
    针对特定领域(如医疗、法律)训练分词器
    - 收集领域语料
    - 训练SentencePiece模型
    - 评估分词质量
    """
    pass

# 项目2: 分词器性能基准测试
def benchmark_tokenizers():
    """
    对比不同分词器在多个维度的性能:
    - 编码/解码速度
    - 序列长度统计
    - 下游任务性能
    - 多语言支持
    """
    pass

# 项目3: 中文分词优化
def optimize_chinese_tokenization():
    """
    研究中文分词的优化策略:
    - 对比字符级vs子词级
    - 结合jieba预分词
    - 评估NER任务效果
    """
    pass
```

---

*通过本章学习，您将全面理解分词技术的核心原理与实践要点，为深入理解大语言模型打下坚实基础。*
