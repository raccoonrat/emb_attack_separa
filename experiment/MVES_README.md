# MVES (最小验证实验) 使用指南

## 概述

MVES (Minimum Viable Experimental Setup) 是一个最小验证实验框架，用于验证MoE水印方法的有效性。

**核心特点**:
- ✅ 使用公开模型: `google/switch-base-8` (T5-based MoE)
- ✅ 基于LogitsProcessor API: 复用transformers的强大功能
- ✅ 预计算(pre-pass)方式: 在LogitsProcessor内部获取路由权重
- ✅ 配置驱动: 所有参数集中管理，保证可复现性

## 快速开始

### 1. 安装依赖

```bash
pip install torch transformers numpy scipy scikit-learn tqdm datasets
```

### 2. 运行快速测试

```bash
python mves_experiment.py --quick
```

这将使用快速测试配置（10个样本）运行实验。

### 3. 运行完整实验

```bash
python mves_experiment.py
```

使用默认配置运行完整实验。

### 4. 使用自定义配置

```bash
# 创建配置文件
python -c "
from mves_config import get_default_config
config = get_default_config()
config.experiment.num_samples = 50
config.watermark.epsilon = 0.02
config.save('my_config.json')
"

# 使用配置文件运行
python mves_experiment.py --config my_config.json
```

## 配置说明

### 模型配置 (`ModelConfig`)

```python
model = ModelConfig(
    model_name="google/switch-base-8",  # MVES关键：公开的MoE模型
    model_type="switch",
    device="auto",
    torch_dtype="float32"
)
```

### 水印配置 (`WatermarkConfig`)

```python
watermark = WatermarkConfig(
    secret_key="MY_SECRET_KEY",
    epsilon=0.01,  # 水印强度 ε = c²γ
    c_star=2.0,  # 安全系数
    gamma_design=0.03,  # 设计的攻击强度
    num_experts=8,  # 专家数量
    k_top=1  # Top-k激活数
)
```

### 检测配置 (`DetectionConfig`)

```python
detection = DetectionConfig(
    tau_alpha=20.0,  # LLR检测阈值
    alpha=0.01,  # 第一类错误率
    use_chernoff=True  # 是否计算Chernoff信息
)
```

### 攻击配置 (`AttackConfig`)

```python
attack = AttackConfig(
    attack_type="paraphrase",  # none, paraphrase, adversarial
    attack_strength="moderate",  # mild, moderate, strong
    gamma_estimation_method="upper_bound"  # upper_bound, kl_divergence
)
```

## 代码架构

### 1. 配置模块 (`mves_config.py`)

集中管理所有实验参数，支持：
- 配置验证
- JSON序列化/反序列化
- 预设配置（默认、快速测试、完整实验）

### 2. 水印模块 (`mves_watermark.py`)

基于LogitsProcessor API实现水印嵌入：

```python
class MoEWatermarkLogitsProcessor(LogitsProcessor):
    """
    核心功能:
    1. 预计算路由权重 (pre-pass方式)
    2. 计算水印偏置
    3. 确保KL(p1||p0) = ε
    4. 集成到model.generate()流程
    """
```

**关键方法**:
- `_precompute_router_weights()`: 预计算MoE路由权重
- `_compute_watermark_bias()`: 计算水印偏置向量
- `__call__()`: LogitsProcessor接口，修改生成logits

### 3. 实验模块 (`mves_experiment.py`)

完整的实验流程：

```python
class MVESExperiment:
    """
    实验阶段:
    1. 水印嵌入: embed_watermark()
    2. 水印检测: detect_watermark()
    3. 鲁棒性测试: detect_watermark(apply_attack=True)
    """
```

## 使用示例

### 示例1: 基本使用

```python
from mves_config import get_default_config
from mves_experiment import MVESExperiment

# 创建配置
config = get_default_config()
config.watermark.secret_key = "my_secret_key"
config.experiment.num_samples = 20

# 运行实验
experiment = MVESExperiment(config)
results = experiment.run_experiment()
```

### 示例2: 自定义配置

```python
from mves_config import MVESConfig, ModelConfig, WatermarkConfig

# 创建自定义配置
config = MVESConfig(
    model=ModelConfig(
        model_name="google/switch-base-8",
        device="cuda"
    ),
    watermark=WatermarkConfig(
        secret_key="custom_key",
        epsilon=0.015,
        c_star=2.5
    ),
    experiment=ExperimentConfig(
        num_samples=50,
        output_dir="./my_results"
    )
)

# 验证并保存
config.validate()
config.save("custom_config.json")

# 运行实验
experiment = MVESExperiment(config)
results = experiment.run_experiment()
```

### 示例3: 单独使用水印处理器

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from mves_config import get_default_config
from mves_watermark import create_watermark_processor
from transformers import LogitsProcessorList

# 加载模型
model = AutoModelForSeq2SeqLM.from_pretrained("google/switch-base-8")
tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")

# 创建配置
config = get_default_config()

# 创建水印处理器
processor = create_watermark_processor(model, config, tokenizer)

# 生成带水印的文本
inputs = tokenizer("Hello world", return_tensors="pt")
logits_processor = LogitsProcessorList([processor])

outputs = model.generate(
    **inputs,
    logits_processor=logits_processor,
    max_length=50
)

watermarked_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(watermarked_text)
```

## 实验结果

实验完成后，结果保存在 `output_dir` 目录下：

```
mves_results/
├── mves_results.json      # 完整实验结果
├── mves_config.json       # 实验配置
└── statistics.json        # 统计摘要 (如果生成)
```

### 结果格式

```json
{
  "config": {...},
  "embedding_results": [
    {
      "index": 0,
      "prompt": "...",
      "watermarked_text": "...",
      "metadata": {...}
    }
  ],
  "detection_results": [
    {
      "index": 0,
      "text": "...",
      "llr_score": 15.23,
      "is_detected": true,
      ...
    }
  ],
  "robustness_results": [...],
  "statistics": {
    "detection_accuracy": 0.95,
    "robustness_rate": 0.85,
    "avg_gamma": 0.025,
    ...
  }
}
```

## 理论保证

本实现严格遵循论文的理论框架：

1. **水印强度**: $\epsilon = c^2 \gamma$ (论文定义5.1)
2. **KL散度约束**: $\text{KL}(p_1 || p_0) = \epsilon$ (论文定义3.2)
3. **最优检测**: 基于Neyman-Pearson引理的LLR检测器 (论文定理3.1)
4. **次线性衰减**: MoE水印衰减为 $O(\sqrt{\gamma})$ (论文定理4.5)

## 注意事项

1. **模型架构**: 当前实现针对`google/switch-base-8`优化，其他MoE模型可能需要调整
2. **路由权重获取**: 预计算方式需要访问模型内部结构，不同模型架构可能需要适配
3. **计算资源**: switch-base-8相对较小，但完整实验仍需要一定计算资源
4. **检测方法**: 当前检测实现是简化版本，完整检测需要更复杂的实现

## 扩展开发

### 添加新的MoE模型支持

1. 在`MoEWatermarkLogitsProcessor._precompute_router_weights()`中添加模型特定的路由权重获取逻辑
2. 更新`ModelConfig`添加新模型类型
3. 测试验证

### 改进检测方法

1. 实现完整的LLR检测器（参考`detector.py`）
2. 添加Chernoff信息计算
3. 支持批量检测

### 添加新攻击方法

1. 在`attacks.py`中实现新攻击
2. 更新`AttackConfig`添加新攻击类型
3. 在实验流程中集成

## 故障排除

### 问题1: 模型加载失败

```
解决方案: 检查模型名称是否正确，确保网络连接正常
```

### 问题2: 路由权重获取失败

```
解决方案: 检查模型架构是否支持，可能需要手动适配
```

### 问题3: 内存不足

```
解决方案: 减少batch_size或num_samples，使用更小的模型
```

## 参考文献

- 论文: "Signal-Attack Decoupling in MoE Watermarks: A Rigorous Information-Theoretic Analysis of Provable Robustness"
- Switch Transformer: https://huggingface.co/google/switch-base-8
- Transformers LogitsProcessor: https://huggingface.co/docs/transformers/internal/generation_utils

