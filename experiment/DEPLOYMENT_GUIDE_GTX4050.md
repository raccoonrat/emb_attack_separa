# GTX 4050 部署指南 - switch-base-8

## 一、硬件环境

- **GPU**: GTX 4050 (6GB 显存)
- **模型**: google/switch-base-8 (~2.6B 参数)
- **显存限制**: 需要优化配置

## 二、环境准备

### 2.1 安装依赖

```bash
# 创建conda环境
conda create -n moe_watermark python=3.10
conda activate moe_watermark

# 安装PyTorch (CUDA 11.8或12.1)
# 检查CUDA版本: nvidia-smi
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 安装其他依赖
pip install transformers accelerate bitsandbytes datasets scipy scikit-learn tqdm numpy
```

### 2.2 验证GPU

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
```

---

## 三、显存优化配置

### 3.1 模型加载优化

针对6GB显存，需要使用以下优化策略：

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# 优化配置
model_config = {
    "torch_dtype": torch.float16,  # 使用FP16，节省50%显存
    "device_map": "auto",  # 自动分配
    "low_cpu_mem_usage": True,
    "max_memory": {0: "5GB"},  # 限制GPU显存使用
}

# 加载模型
model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/switch-base-8",
    **model_config
)
tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")
```

### 3.2 8-bit量化（可选，进一步节省显存）

```python
from transformers import BitsAndBytesConfig

# 8-bit量化配置
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/switch-base-8",
    quantization_config=quantization_config,
    device_map="auto"
)
```

**注意**: 量化可能影响精度，建议先测试FP16，如果显存不足再使用8-bit。

### 3.3 生成参数优化

```python
# 生成时的显存优化
generation_config = {
    "max_new_tokens": 128,  # 限制生成长度
    "do_sample": True,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 0.9,
    "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
    # 显存优化
    "use_cache": True,  # 使用KV cache
}
```

---

## 四、统一使用switch-base-8的配置

### 4.1 更新配置文件

创建针对GTX 4050的优化配置：

```python
from mves_config import MVESConfig, ModelConfig, WatermarkConfig

# GTX 4050优化配置
config = MVESConfig(
    model=ModelConfig(
        model_name="google/switch-base-8",
        model_type="switch",
        device="cuda",
        torch_dtype="float16",  # FP16优化
    ),
    watermark=WatermarkConfig(
        secret_key="GTX4050_DEPLOYMENT_KEY",
        epsilon=0.01,
        c_star=2.0,
        gamma_design=0.03,
        num_experts=8,  # switch-base-8默认
        k_top=1,  # switch-base-8默认
    ),
    experiment=ExperimentConfig(
        num_samples=50,  # 减少样本数以适应小显存
        batch_size=1,  # 小batch size
        max_length=256,  # 限制序列长度
    )
)
```

### 4.2 验证switch-base-8架构

```python
def verify_switch_architecture(model):
    """验证switch-base-8架构"""
    print("验证模型架构...")
    
    # 检查decoder结构
    assert hasattr(model, 'decoder'), "模型必须有decoder"
    assert hasattr(model.decoder, 'block'), "decoder必须有block"
    
    # 检查MoE层
    moe_layers = 0
    for layer in model.decoder.block:
        if hasattr(layer, 'layer') and len(layer.layer) > 1:
            ffn_layer = layer.layer[1]
            if hasattr(ffn_layer, 'mlp') and hasattr(ffn_layer.mlp, 'router'):
                moe_layers += 1
    
    print(f"✓ 找到 {moe_layers} 个MoE层")
    return moe_layers > 0
```

---

## 五、完整部署脚本

### 5.1 部署脚本 (deploy_switch_base8.py)

```python
"""
GTX 4050部署脚本 - switch-base-8
"""

import torch
import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from mves_config import MVESConfig, get_default_config
from mves_watermark_corrected import patch_switch_model_with_watermark
from moe_watermark_enhanced import create_watermark_wrapper
import gc

def setup_gpu():
    """GPU设置和优化"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，需要GPU")
    
    # 清空显存缓存
    torch.cuda.empty_cache()
    gc.collect()
    
    # 设置显存分配策略
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    return torch.device("cuda")

def load_model_optimized(model_name="google/switch-base-8", use_8bit=False):
    """优化加载模型"""
    print(f"\n加载模型: {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if use_8bit:
        # 8-bit量化
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
    else:
        # FP16加载
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            max_memory={0: "5GB"}  # 限制显存使用
        )
    
    model.eval()
    
    # 显存使用情况
    if torch.cuda.is_available():
        print(f"模型加载后显存使用: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    
    return model, tokenizer

def deploy_watermark_system(use_hook=True, use_8bit=False):
    """部署完整的水印系统"""
    
    # 1. GPU设置
    device = setup_gpu()
    
    # 2. 加载模型
    model, tokenizer = load_model_optimized(use_8bit=use_8bit)
    
    # 3. 创建配置
    config = get_default_config()
    config.model.model_name = "google/switch-base-8"
    config.model.torch_dtype = "float16"
    config.experiment.batch_size = 1  # 小batch
    config.experiment.max_length = 256  # 限制长度
    
    # 4. 验证架构
    from mves_watermark_corrected import verify_switch_architecture
    if not verify_switch_architecture(model):
        raise RuntimeError("模型架构验证失败")
    
    # 5. 部署水印
    if use_hook:
        print("\n使用Hook机制部署水印...")
        wrapper = create_watermark_wrapper(model, config, use_hook=True)
        wrapper.register_hooks()
        print("✓ Hook已注册")
    else:
        print("\n使用Patch方式部署水印...")
        model = patch_switch_model_with_watermark(model, config)
        print("✓ Patch完成")
    
    # 6. 显存检查
    if torch.cuda.is_available():
        print(f"\n最终显存使用: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        print(f"显存剩余: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1024**3:.2f} GB")
    
    return model, tokenizer, config

def test_generation(model, tokenizer, prompt="Hello, how are you?"):
    """测试生成"""
    print(f"\n测试生成: '{prompt}'...")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"生成文本: {generated_text}")
    
    return generated_text

if __name__ == "__main__":
    # 部署选项
    USE_HOOK = True  # 推荐使用Hook机制
    USE_8BIT = False  # 先尝试FP16，如果显存不足再启用8-bit
    
    try:
        # 部署系统
        model, tokenizer, config = deploy_watermark_system(
            use_hook=USE_HOOK,
            use_8bit=USE_8BIT
        )
        
        # 测试生成
        test_generation(model, tokenizer)
        
        print("\n✓ 部署成功！")
        
    except torch.cuda.OutOfMemoryError:
        print("\n✗ 显存不足！")
        print("建议:")
        print("1. 启用8-bit量化: USE_8BIT = True")
        print("2. 减小batch_size和max_length")
        print("3. 使用CPU卸载部分层")
        
    except Exception as e:
        print(f"\n✗ 部署失败: {e}")
        import traceback
        traceback.print_exc()
```

---

## 六、显存优化技巧

### 6.1 梯度检查点（如果训练）

```python
# 如果需要进行微调
from transformers import TrainingArguments

training_args = TrainingArguments(
    gradient_checkpointing=True,  # 节省显存
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,  # 模拟batch_size=4
)
```

### 6.2 清理显存

```python
def clear_memory():
    """清理显存"""
    torch.cuda.empty_cache()
    gc.collect()
    import psutil
    import os
    process = psutil.Process(os.getpid())
    print(f"内存使用: {process.memory_info().rss / 1024**3:.2f} GB")
```

### 6.3 分批处理

```python
def process_in_batches(texts, model, tokenizer, batch_size=1):
    """分批处理，避免显存溢出"""
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        
        # 处理batch
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=64)
        
        results.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
        
        # 清理显存
        del inputs, outputs
        torch.cuda.empty_cache()
    
    return results
```

---

## 七、快速测试脚本

### 7.1 最小测试 (test_minimal.py)

```python
"""最小测试 - 验证部署是否成功"""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from mves_config import get_default_config
from mves_watermark_corrected import patch_switch_model_with_watermark

# 1. 加载模型（FP16）
print("加载模型...")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/switch-base-8",
    torch_dtype=torch.float16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")

# 2. 部署水印
print("部署水印...")
config = get_default_config()
config.model.model_name = "google/switch-base-8"
patched_model = patch_switch_model_with_watermark(model, config)

# 3. 测试生成
print("测试生成...")
prompt = "The quick brown fox"
inputs = tokenizer(prompt, return_tensors="pt").to(patched_model.device)

with torch.no_grad():
    outputs = patched_model.generate(
        **inputs,
        max_new_tokens=32,
        do_sample=True,
        temperature=0.7
    )

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"结果: {result}")

# 4. 显存检查
print(f"\n显存使用: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
print("✓ 测试完成")
```

---

## 八、常见问题解决

### 8.1 显存不足 (OOM)

**症状**: `torch.cuda.OutOfMemoryError`

**解决方案**:
1. 启用8-bit量化
2. 减小batch_size到1
3. 减小max_length到128或更小
4. 使用CPU卸载: `device_map={"": "cpu"}` 然后手动移动部分层

### 8.2 模型加载失败

**症状**: 无法找到router或MoE层

**解决方案**:
```python
# 检查模型结构
print(model.config)
print(model.decoder.block[0].layer[1].mlp)  # 查看MoE层结构
```

### 8.3 Hook注册失败

**症状**: 找不到gating网络

**解决方案**:
- 使用patch方式作为备选
- 检查模型架构是否匹配switch-base-8

---

## 九、性能基准

### 9.1 预期显存使用

| 配置 | 显存使用 | 备注 |
|------|---------|------|
| FP32 | ~10GB | 超出GTX 4050能力 |
| FP16 | ~5GB | ✅ 推荐 |
| 8-bit | ~3GB | 备选方案 |

### 9.2 生成速度

- **FP16**: ~10-20 tokens/秒
- **8-bit**: ~8-15 tokens/秒

---

## 十、完整工作流

```bash
# 1. 激活环境
conda activate moe_watermark

# 2. 运行最小测试
python test_minimal.py

# 3. 如果成功，运行完整部署
python deploy_switch_base8.py

# 4. 运行MVES实验
python mves_experiment.py --quick
```

---

## 十一、监控脚本

```python
"""显存监控工具"""

import torch
import time

def monitor_memory(interval=1.0, duration=60):
    """监控显存使用"""
    start_time = time.time()
    
    while time.time() - start_time < duration:
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"显存: {allocated:.2f}GB / {reserved:.2f}GB / {total:.2f}GB")
        time.sleep(interval)
```

---

## 十二、检查清单

部署前检查：

- [ ] CUDA可用且版本匹配
- [ ] 显存 >= 6GB
- [ ] 模型下载完成
- [ ] 依赖包安装完成
- [ ] 配置文件正确（model_name = "google/switch-base-8"）
- [ ] 使用FP16或8-bit量化
- [ ] batch_size = 1
- [ ] max_length <= 256

部署后验证：

- [ ] 模型加载成功
- [ ] 水印patch/hook成功
- [ ] 生成测试通过
- [ ] 显存使用 < 6GB
- [ ] 检测功能正常

---

## 十三、故障排除

### 问题1: 显存不足

```python
# 解决方案A: 启用8-bit
model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/switch-base-8",
    load_in_8bit=True,
    device_map="auto"
)

# 解决方案B: CPU卸载
model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/switch-base-8",
    device_map={"": "cpu"},  # 先加载到CPU
    torch_dtype=torch.float16
)
# 然后手动移动关键层到GPU
```

### 问题2: 生成速度慢

- 减小max_new_tokens
- 使用greedy解码（do_sample=False）
- 启用use_cache

### 问题3: 水印检测失败

- 检查secret_key是否一致
- 验证模型是否已正确patch
- 检查检测数据是否收集成功

