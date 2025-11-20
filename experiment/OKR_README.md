# OKR (Opportunistic Keyed Routing) 实验指南

## 概述

OKR 是一个独立的水印算法实验框架，专门用于验证 Opportunistic Keyed Routing 方法在 MoE 模型中的表现。

**核心特点：**
- 完全独立于 MVES 代码，不耦合
- 使用 LSH（局部敏感哈希）实现语义锚点
- 机会主义路由：只在安全区域内修改路由
- 纯 Tensor 操作，零 CPU 交互，高性能

## 文件结构

```
experiment/
├── okr_config.py          # OKR 配置类（独立）
├── okr_experiment.py      # OKR 实验框架
├── okr_example.py         # 使用示例
├── okr_kernel.py          # OKR 核心路由逻辑
├── okr_patch.py           # 水印注入代码
├── okr_detector.py        # 水印检测器
└── OKR_README.md          # 本文档
```

## 快速开始

### 1. 基本使用

```python
from okr_patch import inject_okr
from okr_detector import OKRDetector
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# 加载模型
model = AutoModelForSeq2SeqLM.from_pretrained("google/switch-base-8")
tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")

# 注入水印
model = inject_okr(model, epsilon=1.5, secret_key="my_secret_key")

# 生成文本
text = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)

# 检测水印
detector = OKRDetector(model, epsilon=1.5)
score, verdict = detector.detect(inputs["input_ids"])
print(f"检测结果: {verdict}, 得分: {score:.4f}")
```

### 2. 使用配置

```python
from okr_config import get_default_okr_config
from okr_patch import inject_okr

# 创建配置
config = get_default_okr_config()
config.watermark.epsilon = 1.5
config.watermark.secret_key = "OKR_SECRET_KEY"

# 注入水印
model = inject_okr(
    model,
    epsilon=config.watermark.epsilon,
    secret_key=config.watermark.secret_key
)
```

### 3. 运行完整实验

```python
from okr_experiment import run_okr_experiment
from okr_config import get_quick_test_okr_config

# 使用快速测试配置
config = get_quick_test_okr_config()
config.watermark.epsilon = 1.5
config.watermark.secret_key = "EXPERIMENT_KEY"

# 运行基础实验
results = run_okr_experiment(config, experiment_type="basic")
print(f"平均命中率: {results['summary']['average_hit_rate']:.4f}")
```

## 配置说明

### OKRConfig

主要配置项：

- **model**: 模型配置
  - `model_name`: 模型名称（默认: "google/switch-base-8"）
  - `device`: 计算设备（默认: "auto"）
  - `torch_dtype`: 数据类型（默认: "float32"）

- **watermark**: 水印配置
  - `secret_key`: 私钥（用于初始化 secret_projection）
  - `epsilon`: 质量容忍阈值（Logit 差值，默认: 1.5）
  - `num_experts`: 专家数量（默认: 8）
  - `top_k`: Top-k 激活数（默认: 1）

- **detection**: 检测配置
  - `hit_rate_threshold`: 命中率阈值（默认: 0.8）
  - `min_opportunities`: 最小机会窗口数（默认: 10）

- **experiment**: 实验配置
  - `experiment_name`: 实验名称
  - `output_dir`: 输出目录（默认: "./okr_results"）
  - `num_samples`: 样本数（默认: 100）
  - `batch_size`: 批次大小（默认: 4）

## 实验类型

### 1. 基础实验 (OKRBasicExperiment)

验证水印注入和检测的基本功能：
- 生成带水印的文本
- 检测水印命中率
- 统计检测结果

### 2. 鲁棒性实验 (OKRRobustnessExperiment)

测试水印在释义攻击下的表现：
- 生成带水印的文本
- 进行释义攻击
- 检测攻击后的水印

## 算法原理

### 核心思想

1. **语义锚点 (Semantic Anchors)**
   - 使用 LSH（局部敏感哈希）将水印打在 Embedding 上
   - 语义相似的文本会产生相似的投影结果
   - 抗释义攻击

2. **机会主义路由 (Opportunistic Routing)**
   - 只在安全区域内修改路由
   - 安全区域定义：`max_logit - current_logit < epsilon`
   - 保证输出质量不受影响

### 实现细节

```python
# 1. 计算水印信号（LSH 投影）
watermark_bias = torch.matmul(hidden_states, secret_projection)

# 2. 计算安全掩码
max_logits, _ = raw_logits.max(dim=-1, keepdim=True)
safe_mask = raw_logits >= (max_logits - epsilon)

# 3. 机会主义注入
final_scores = torch.where(
    safe_mask,
    watermark_bias,  # 在安全区内，听水印的
    -1e9              # 在安全区外，直接淘汰
)

# 4. 路由选择
selected_experts = torch.argmax(final_scores, dim=-1)
```

## 日志和输出

实验会自动生成：
- 日志文件：`{output_dir}/experiment.log`
- 结果文件：`{output_dir}/results.json`（基础实验）
- 结果文件：`{output_dir}/robustness_results.json`（鲁棒性实验）

## 注意事项

1. **模型兼容性**
   - 当前主要支持 Switch Transformers（google/switch-base-8）
   - 理论上支持任何 MoE 模型，但需要调整路由层名称

2. **性能优化**
   - 使用 FP16/BF16 可以显著减少内存占用
   - 建议设置 `max_memory` 限制 GPU 内存使用

3. **配置验证**
   - 配置会自动验证，确保参数有效性
   - 如果验证失败，会抛出 `ValueError`

## 示例代码

更多示例请参考 `okr_example.py`：

- `example_basic_usage()` - 基本使用
- `example_with_config()` - 使用配置
- `example_run_experiment()` - 运行完整实验
- `example_custom_experiment()` - 自定义实验

## 参考文档

- `okr/Opportunistic Keyed Routing V2.1.md` - 算法详细说明
- `okr/水印算法深度分析-数学原理.md` - 数学原理分析

