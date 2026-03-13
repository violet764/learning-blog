# 离线推理

本章详细介绍 vLLM 的离线推理功能，包括 `LLM` 类的使用、`SamplingParams` 参数配置、批量推理和高级用法。

---

## 🎯 LLM 类概述

`LLM` 类是 vLLM 进行离线推理的核心入口，提供简洁的 API 来加载模型和生成文本。

### 基本架构

```
LLM 类架构
├── __init__()           # 初始化模型和引擎
├── generate()           # 批量生成文本
├── get_tokenizer()      # 获取分词器
└── llm_engine          # 底层推理引擎
    ├── model_executor   # 模型执行器
    ├── scheduler        # 请求调度器
    └── cache_engine     # KV Cache 引擎
```

### 完整初始化参数

```python
from vllm import LLM

llm = LLM(
    # ========== 模型配置 ==========
    model="Qwen/Qwen2.5-7B-Instruct",  # 模型名称或路径
    tokenizer=None,                     # 分词器路径
    tokenizer_mode="auto",              # "auto", "slow", "mistral"
    trust_remote_code=False,            # 是否信任远程代码
    download_dir=None,                  # 模型下载目录
    load_format="auto",                 # 加载格式: "auto", "pt", "safetensors"
    
    # ========== 精度配置 ==========
    dtype="auto",                       # "auto", "float16", "bfloat16", "float32"
    quantization=None,                  # None, "awq", "gptq", "fp8", "squeezellm"
    
    # ========== 内存配置 ==========
    gpu_memory_utilization=0.9,         # GPU 内存利用率 (0-1)
    max_model_len=None,                 # 最大序列长度
    block_size=16,                      # PagedAttention 块大小
    swap_space=4,                       # CPU swap 空间大小 (GB)
    enforce_eager=False,                # 禁用 CUDA graph
    
    # ========== 并行配置 ==========
    tensor_parallel_size=1,             # 张量并行 GPU 数量
    pipeline_parallel_size=1,           # 流水线并行 GPU 数量
    distributed_executor_backend=None,  # "ray", "mp" (multiprocessing)
    
    # ========== 其他配置 ==========
    device="auto",                      # "auto", "cuda", "cpu"
    seed=0,                             # 随机种子
    max_logprobs=20,                    # 返回的最大 logprobs 数量
    disable_custom_all_reduce=False,    # 禁用自定义 all-reduce
)
```

---

## ⚙️ SamplingParams 详解

`SamplingParams` 控制文本生成的行为，包括采样策略、长度限制等。

### 完整参数说明

```python
from vllm import SamplingParams

params = SamplingParams(
    # ========== 基本参数 ==========
    n=1,                      # 每个提示生成 n 个结果
    best_of=None,             # 从 best_of 个结果中选最好的 n 个
    presence_penalty=0.0,     # 存在惩罚 (-2.0 ~ 2.0)
    frequency_penalty=0.0,    # 频率惩罚 (-2.0 ~ 2.0)
    repetition_penalty=1.0,   # 重复惩罚 (>1 减少重复)
    
    # ========== 采样参数 ==========
    temperature=1.0,          # 温度参数 (0-2)，越低越确定
    top_p=1.0,                # 核采样 (0-1)
    top_k=-1,                 # Top-K 采样 (-1 表示禁用)
    min_p=0.0,                # 最小概率阈值
    seed=None,                # 随机种子
    
    # ========== 长度控制 ==========
    max_tokens=16,            # 最大生成 token 数
    min_tokens=0,             # 最小生成 token 数
    
    # ========== 停止条件 ==========
    stop=None,                # 停止词列表: ["stop_word1", "stop_word2"]
    stop_token_ids=None,      # 停止 token ID 列表
    include_stop_str_in_output=False,  # 输出是否包含停止词
    
    # ========== 输出控制 ==========
    logprobs=None,            # 返回前 N 个 token 的 logprobs
    prompt_logprobs=None,     # 返回提示词的 logprobs
    
    # ========== 特殊参数 ==========
    ignore_eos=False,         # 是否忽略 EOS token
    skip_special_tokens=True, # 是否跳过特殊 token
    spaces_between_special_tokens=True,
    truncate_prompt_tokens=None,  # 截断提示词长度
)
```

### 参数详解与示例

#### 温度 (temperature)

```python
# temperature = 0: 确定性输出（每次相同）
params_deterministic = SamplingParams(temperature=0, max_tokens=100)

# temperature = 0.7: 适度随机（推荐用于对话）
params_creative = SamplingParams(temperature=0.7, max_tokens=100)

# temperature = 1.5: 高度随机（创意写作）
params_random = SamplingParams(temperature=1.5, max_tokens=100)

# 演示效果
llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")
prompt = "讲一个短笑话"

for temp in [0, 0.7, 1.5]:
    params = SamplingParams(temperature=temp, max_tokens=50)
    output = llm.generate([prompt], params)
    print(f"temperature={temp}: {output[0].outputs[0].text}")
```

#### Top-P 核采样

```python
# top_p = 0.9: 从累计概率达到 90% 的 token 中采样
params = SamplingParams(top_p=0.9, temperature=0.7)

# top_p = 0.1: 更保守的选择
params_conservative = SamplingParams(top_p=0.1, temperature=0.7)

# 结合使用：推荐配置
params_recommended = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    top_k=50  # 同时限制 top-k
)
```

#### 惩罚参数

```python
# presence_penalty: 惩罚出现过的 token（鼓励新话题）
params_new_topics = SamplingParams(
    presence_penalty=1.0,  # 正值鼓励谈论新内容
    max_tokens=200
)

# frequency_penalty: 按出现次数惩罚（减少重复）
params_less_repeat = SamplingParams(
    frequency_penalty=0.5,  # 正值减少重复
    max_tokens=200
)

# repetition_penalty: 整体重复惩罚
params_no_repeat = SamplingParams(
    repetition_penalty=1.2,  # >1 减少重复，<1 增加重复
    max_tokens=200
)

# 演示重复惩罚效果
prompt = "请写一首关于春天的诗"
for penalty in [1.0, 1.2, 1.5]:
    params = SamplingParams(repetition_penalty=penalty, max_tokens=100)
    output = llm.generate([prompt], params)
    print(f"repetition_penalty={penalty}:")
    print(output[0].outputs[0].text)
    print("-" * 50)
```

#### 停止词设置

```python
# 使用字符串停止词
params = SamplingParams(
    max_tokens=500,
    stop=["</s>", "END", "---"]  # 遇到任意一个停止
)

# 使用 token ID 停止
params_token_ids = SamplingParams(
    max_tokens=500,
    stop_token_ids=[2, 0]  # EOS token ID
)

# 输出包含停止词
params_include_stop = SamplingParams(
    max_tokens=500,
    stop=["总结"],
    include_stop_str_in_output=True  # 保留停止词
)
```

#### 多候选生成

```python
# 生成多个候选并选择最佳
params = SamplingParams(
    n=3,           # 生成 3 个结果
    best_of=5,     # 从 5 个中选最好的 3 个
    temperature=0.8,
    max_tokens=100
)

outputs = llm.generate(["写一句励志名言"], params)

for i, output in enumerate(outputs[0].outputs):
    print(f"候选 {i+1}: {output.text}")
```

---

## 🔄 批量推理

### 基本批量处理

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")
sampling_params = SamplingParams(temperature=0.7, max_tokens=100)

# 批量提示词
prompts = [
    "什么是机器学习？",
    "解释一下量子计算",
    "如何学习编程？",
    "推荐几本好书",
    "健康生活建议"
]

# 一次推理处理所有提示
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"问题: {output.prompt}")
    print(f"回答: {output.outputs[0].text}")
    print("-" * 60)
```

### 动态批量大小

```python
def batch_generate(prompts, batch_size=32, max_tokens=100):
    """动态批量生成"""
    llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")
    sampling_params = SamplingParams(max_tokens=max_tokens)
    
    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        outputs = llm.generate(batch, sampling_params)
        results.extend([o.outputs[0].text for o in outputs])
        print(f"已处理 {min(i + batch_size, len(prompts))}/{len(prompts)}")
    
    return results

# 使用
prompts = [f"生成一个关于主题 {i} 的标题" for i in range(100)]
results = batch_generate(prompts, batch_size=16)
```

### 异构批量处理

```python
from vllm import LLM, SamplingParams, RequestOutput

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")

# 不同长度的提示词
prompts = [
    "Hi",                                           # 短提示
    "请详细解释一下深度学习的发展历史和未来趋势",    # 中等提示
    "请分析以下文本的主题、情感和关键词..." + "x" * 500  # 长提示
]

# 自适应采样参数
def adaptive_sampling(prompt):
    """根据提示长度调整参数"""
    if len(prompt) < 50:
        return SamplingParams(max_tokens=100, temperature=0.7)
    else:
        return SamplingParams(max_tokens=500, temperature=0.5)

# 分别处理（vLLM 会自动优化批处理）
for prompt in prompts:
    params = adaptive_sampling(prompt)
    output = llm.generate([prompt], params)
    print(output[0].outputs[0].text[:100] + "...")
```

---

## 📤 输出处理

### 输出结构

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")
params = SamplingParams(max_tokens=50, logprobs=5)

outputs = llm.generate(["你好"], params)

# 输出结构详解
output = outputs[0]

print(type(output))  # RequestOutput
print(f"请求 ID: {output.request_id}")
print(f"提示词: {output.prompt}")
print(f"提示词 token IDs: {output.prompt_token_ids}")
print(f"完成状态: {output.finished}")

# 生成的结果列表（n > 1 时有多个）
for completion in output.outputs:
    print(f"生成文本: {completion.text}")
    print(f"Token IDs: {completion.token_ids}")
    print(f"完成原因: {completion.finish_reason}")  # "stop", "length"
    
    # Logprobs（如果请求了）
    if completion.logprobs:
        for i, logprob_info in enumerate(completion.logprobs):
            print(f"Token {i}: {logprob_info}")
```

### 提取生成结果

```python
def extract_responses(outputs):
    """提取所有生成的文本"""
    return [output.outputs[0].text for output in outputs]

def extract_with_metadata(outputs):
    """提取带元数据的结果"""
    results = []
    for output in outputs:
        results.append({
            "prompt": output.prompt,
            "response": output.outputs[0].text,
            "prompt_tokens": len(output.prompt_token_ids),
            "completion_tokens": len(output.outputs[0].token_ids),
            "finish_reason": output.outputs[0].finish_reason
        })
    return results

# 使用
outputs = llm.generate(prompts, sampling_params)
responses = extract_responses(outputs)
detailed = extract_with_metadata(outputs)
```

### 处理多候选输出

```python
# 生成多个候选
params = SamplingParams(n=3, max_tokens=100)
outputs = llm.generate(["写一首诗"], params)

output = outputs[0]
print(f"提示词: {output.prompt}")
print(f"生成 {len(output.outputs)} 个候选:")

for i, completion in enumerate(output.outputs):
    print(f"\n候选 {i+1}:")
    print(completion.text)
    print(f"Token 数: {len(completion.token_ids)}")
```

---

## 🔧 高级用法

### 自定义分词器

```python
from vllm import LLM
from transformers import AutoTokenizer

# 使用自定义分词器
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    use_fast=True
)

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    tokenizer=tokenizer  # 传入自定义分词器
)

# 手动编码提示词
encoded = tokenizer.encode("你好，世界", return_tensors="pt")
print(f"编码结果: {encoded}")
```

### 前缀缓存

```python
# 利用相同前缀的缓存优化
llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    enable_prefix_caching=True  # 启用前缀缓存
)

# 相同系统提示的场景
system_prompt = "你是一个专业的编程助手，请用中文回答问题。"

prompts = [
    f"{system_prompt}\n用户：如何学习 Python？",
    f"{system_prompt}\n用户：什么是闭包？",
    f"{system_prompt}\n用户：解释一下装饰器。",
]

# 前缀会被自动缓存和复用
outputs = llm.generate(prompts, SamplingParams(max_tokens=100))
```

### LoRA 适配器

```python
from vllm import LLM, SamplingParams

# 加载基础模型并使用 LoRA
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_lora=True,
    max_loras=4,  # 最大同时加载的 LoRA 数量
    max_lora_rank=64,
)

# 使用 LoRA 生成
sampling_params = SamplingParams(max_tokens=100)

# 指定 LoRA 适配器
outputs = llm.generate(
    prompts=["解释一下机器学习"],
    sampling_params=sampling_params,
    lora_request=LoRARequest(
        "my-lora",  # LoRA 名称
        1,          # LoRA ID
        "/path/to/lora/adapter"  # LoRA 路径
    )
)
```

### 多模态推理

```python
from vllm import LLM, SamplingParams

# 加载视觉语言模型
llm = LLM(model="Qwen/Qwen2-VL-7B-Instruct")

# 图像+文本输入
prompts = [
    {
        "prompt": "<|image_pad|>描述这张图片的内容",
        "multi_modal_data": {
            "image": "path/to/image.jpg"
        }
    }
]

sampling_params = SamplingParams(max_tokens=200)
outputs = llm.generate(prompts, sampling_params)
```

---

## 📊 性能对比示例

### 与 HuggingFace 对比

```python
import time
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def benchmark_comparison():
    prompts = ["写一首关于春天的诗"] * 20
    
    # vLLM 推理
    print("vLLM 推理...")
    llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")
    params = SamplingParams(max_tokens=100)
    
    start = time.time()
    vllm_outputs = llm.generate(prompts, params)
    vllm_time = time.time() - start
    print(f"vLLM 耗时: {vllm_time:.2f}s")
    
    # HuggingFace 推理
    print("\nHuggingFace 推理...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    start = time.time()
    hf_outputs = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=100)
        hf_outputs.append(tokenizer.decode(outputs[0]))
    hf_time = time.time() - start
    print(f"HuggingFace 耗时: {hf_time:.2f}s")
    
    print(f"\n加速比: {hf_time / vllm_time:.2f}x")

benchmark_comparison()
```

---

## 📋 完整示例

### 问答系统

```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

class QAService:
    def __init__(self, model_name="Qwen/Qwen2.5-7B-Instruct"):
        self.llm = LLM(model=model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.params = SamplingParams(
            temperature=0.7,
            max_tokens=512,
            stop=["</answer>", "\n\n问题:"]
        )
    
    def build_prompt(self, question: str, context: str = None) -> str:
        """构建问答提示"""
        messages = [
            {"role": "system", "content": "你是一个专业的问答助手，请简洁准确地回答问题。"}
        ]
        
        if context:
            messages.append({
                "role": "user",
                "content": f"参考以下内容回答问题：\n{context}\n\n问题：{question}"
            })
        else:
            messages.append({"role": "user", "content": question})
        
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    
    def answer(self, questions: list[str], contexts: list[str] = None) -> list[str]:
        """批量回答问题"""
        if contexts:
            prompts = [self.build_prompt(q, c) for q, c in zip(questions, contexts)]
        else:
            prompts = [self.build_prompt(q) for q in questions]
        
        outputs = self.llm.generate(prompts, self.params)
        return [o.outputs[0].text for o in outputs]

# 使用
qa = QAService()
questions = [
    "什么是 Transformer 架构？",
    "解释一下 Attention 机制"
]
answers = qa.answer(questions)
for q, a in zip(questions, answers):
    print(f"Q: {q}\nA: {a}\n")
```

---

## 🔗 相关章节

- [基础概念与安装](./vllm-basics.md) - PagedAttention 原理
- [API 服务器部署](./vllm-server.md) - 部署 OpenAI 兼容服务
- [性能优化技巧](./vllm-optimization.md) - 深入优化方法
