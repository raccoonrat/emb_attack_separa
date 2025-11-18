# 开始部署实验 - 快速指南

## ✅ 已完成的配置

1. ✅ **缓存路径**: 已配置到 `D:\Dev\cache`
2. ✅ **Hugging Face镜像**: 已配置 `https://hf-mirror.com`
3. ✅ **所有脚本**: 已自动添加缓存和镜像配置

## 🚀 开始部署（3步）

### 步骤1: 测试镜像和缓存配置

```bash
cd experiment
python test_hf_mirror.py
```

**预期输出**:
- ✅ HF_ENDPOINT: https://hf-mirror.com
- ✅ 下载tokenizer成功
- ✅ 缓存保存到D盘

### 步骤2: 最小测试（验证部署）

```bash
python test_minimal_switch.py
```

**预期结果**:
- ✅ 模型加载成功
- ✅ 水印部署成功
- ✅ 生成测试通过
- ✅ 显存使用正常

### 步骤3: 完整部署

```bash
# 方式1: Hook机制（推荐）
python deploy_switch_base8.py

# 方式2: 如果显存不足，使用8-bit量化
python deploy_switch_base8.py --use_8bit

# 方式3: Patch方式
python deploy_switch_base8.py --use_patch
```

## 📋 部署检查清单

### 部署前检查

- [ ] 已运行 `test_hf_mirror.py` 验证镜像配置
- [ ] 确认D盘有足够空间（>10GB）
- [ ] 确认GPU可用（GTX 4050）
- [ ] 已安装所有依赖（`pip install -r requirements.txt`）

### 部署后验证

- [ ] 模型加载成功
- [ ] 水印部署成功（Hook或Patch）
- [ ] 生成测试通过
- [ ] 检测功能正常
- [ ] 显存使用 < 6GB

## 🔍 验证镜像配置

### 快速验证

```python
import os
print("HF_ENDPOINT:", os.environ.get("HF_ENDPOINT"))
```

应该显示：`HF_ENDPOINT: https://hf-mirror.com`

### 测试下载

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")
print("下载成功！")
```

## 📊 预期结果

### 模型下载

- **首次下载**: 约5-10分钟（取决于网络速度）
- **使用镜像**: 通常比原始地址快
- **缓存位置**: `D:\Dev\cache\huggingface\hub\`

### 显存使用

- **FP16**: ~5GB
- **8-bit**: ~3GB

### 生成速度

- **FP16**: 10-20 tokens/秒
- **8-bit**: 8-15 tokens/秒

## 🐛 常见问题

### Q1: 镜像下载失败

**解决方案**:
1. 检查网络连接
2. 尝试直接访问: https://hf-mirror.com
3. 如果不可用，移除 `HF_ENDPOINT` 环境变量

### Q2: 显存不足

**解决方案**:
```bash
python deploy_switch_base8.py --use_8bit
```

### Q3: 模型加载失败

**解决方案**:
1. 检查网络连接
2. 确认模型名称: `google/switch-base-8`
3. 检查缓存目录权限

## 📝 下一步

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
   python main.py --mode detect --model_name google/switch-base-8 --text_to_check "Text"
   ```

## 📚 相关文档

- **镜像配置**: `HF_MIRROR_SETUP.md`
- **缓存迁移**: `CACHE_MIGRATION.md`
- **部署指南**: `DEPLOYMENT_GUIDE_GTX4050.md`
- **快速参考**: `CACHE_QUICK_START.md`

---

**准备好了吗？开始部署！**

```bash
# 1. 测试镜像
python test_hf_mirror.py

# 2. 最小测试
python test_minimal_switch.py

# 3. 完整部署
python deploy_switch_base8.py
```

