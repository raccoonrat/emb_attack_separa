# GTX 4050 部署快速指南

## 一、快速开始（3步）

### 步骤1: 安装依赖

```bash
cd experiment
conda create -n moe_watermark python=3.10
conda activate moe_watermark
pip install -r requirements.txt
```

### 步骤2: 运行最小测试

```bash
python test_minimal_switch.py
```

如果测试通过，继续步骤3。

### 步骤3: 完整部署

```bash
# 方式1: 使用Hook机制（推荐）
python deploy_switch_base8.py

# 方式2: 如果显存不足，使用8-bit量化
python deploy_switch_base8.py --use_8bit

# 方式3: 使用Patch方式
python deploy_switch_base8.py --use_patch
```

---

## 二、一键启动（推荐）

```bash
python quick_start_switch.py
```

然后按提示选择模式。

---

## 三、配置说明

### 3.1 默认配置（已优化）

所有配置已针对GTX 4050优化：

- **模型**: `google/switch-base-8`
- **精度**: FP16（节省50%显存）
- **Batch size**: 1
- **Max length**: 256
- **水印强度**: ε = 0.01

### 3.2 自定义配置

编辑 `mves_config.py` 或创建自定义配置：

```python
from mves_config import MVESConfig, ModelConfig, WatermarkConfig

config = MVESConfig(
    model=ModelConfig(
        model_name="google/switch-base-8",
        torch_dtype="float16",  # 或 "int8" 用于8-bit
    ),
    watermark=WatermarkConfig(
        secret_key="your_key",
        epsilon=0.01,
        c_star=2.0,
    )
)
```

---

## 四、显存优化建议

### 如果显存不足（< 5GB可用）

1. **启用8-bit量化**:
   ```bash
   python deploy_switch_base8.py --use_8bit
   ```

2. **减小序列长度**:
   ```python
   config.experiment.max_length = 128  # 从256减小到128
   ```

3. **减小batch size**:
   ```python
   config.experiment.batch_size = 1  # 已经是1，不能再小
   ```

4. **关闭其他程序**: 释放显存

---

## 五、验证部署

### 5.1 检查清单

- [ ] 模型加载成功
- [ ] 水印部署成功（Hook或Patch）
- [ ] 生成测试通过
- [ ] 检测功能正常
- [ ] 显存使用 < 6GB

### 5.2 运行测试

```python
# 测试生成
python -c "
from deploy_switch_base8 import deploy_watermark_system, test_generation
model, tokenizer, config, _ = deploy_watermark_system()
test_generation(model, tokenizer)
"
```

---

## 六、常见问题

### Q1: 显存不足错误

**A**: 使用8-bit量化
```bash
python deploy_switch_base8.py --use_8bit
```

### Q2: 模型加载失败

**A**: 检查网络连接，确保能访问Hugging Face

### Q3: Hook注册失败

**A**: 使用Patch方式作为备选
```bash
python deploy_switch_base8.py --use_patch
```

### Q4: 生成速度慢

**A**: 
- 减小max_new_tokens
- 使用greedy解码（do_sample=False）
- 确保使用FP16而非FP32

---

## 七、性能基准

| 配置 | 显存使用 | 生成速度 | 推荐场景 |
|------|---------|---------|---------|
| FP16 | ~5GB | 10-20 tok/s | ✅ 默认推荐 |
| 8-bit | ~3GB | 8-15 tok/s | 显存不足时 |

---

## 八、下一步

部署成功后，可以：

1. **运行MVES实验**:
   ```bash
   python mves_experiment.py --quick
   ```

2. **标定参数**:
   ```bash
   python main.py --mode calibrate --model_name google/switch-base-8
   ```

3. **嵌入和检测水印**:
   ```bash
   python main.py --mode embed --model_name google/switch-base-8 --prompt "Your text"
   python main.py --mode detect --model_name google/switch-base-8 --text_to_check "Text to check"
   ```

---

## 九、技术支持

如遇问题，请检查：

1. CUDA版本是否匹配
2. 显存是否足够（>= 6GB）
3. 模型是否正确下载
4. 配置文件是否正确

详细文档: `DEPLOYMENT_GUIDE_GTX4050.md`

