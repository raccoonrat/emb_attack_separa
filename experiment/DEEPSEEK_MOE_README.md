# DeepSeek-MoE OKR 实验指南

## 概述

本文档说明如何在 DeepSeek-MoE 模型上运行 OKR（Opportunistic Keyed Routing）水印实验，并针对 A800 GPU 进行了优化。

## 快速开始

### 1. 使用本地模型（推荐）

如果模型已下载到本地（如 `/root/private_data/model/DeepSeek-MoE`），脚本会自动检测并使用本地路径：

```bash
cd experiment
python run_deepseek_moe_experiment.py
```

或者通过环境变量指定本地路径：

```bash
DEEPSEEK_MOE_LOCAL_PATH=/root/private_data/model/DeepSeek-MoE python run_deepseek_moe_experiment.py
```

### 2. 使用 HuggingFace 模型

如果本地路径不存在，脚本会自动从 HuggingFace 下载：

```bash
cd experiment
python run_deepseek_moe_experiment.py
```

### 3. 使用完整配置（非快速测试）

```bash
QUICK_TEST=0 python run_deepseek_moe_experiment.py
```

## 配置说明

### DeepSeek-MoE 特定配置

- **模型路径**: 
  - 本地路径（优先）: `/root/private_data/model/DeepSeek-MoE`
  - HuggingFace ID: `deepseek-ai/DeepSeek-MoE-16B`
- **模型类型**: `deepseek_moe` (decoder-only)
- **专家数量**: 64
- **Top-K**: 8（每次激活 8 个专家）
- **Epsilon**: 2.0（推荐值，可根据实验调整）

### 本地模型路径配置

脚本会自动检测本地模型路径，优先级如下：
1. 环境变量 `DEEPSEEK_MOE_LOCAL_PATH`
2. 默认路径 `/root/private_data/model/DeepSeek-MoE`
3. 如果本地路径不存在，使用 HuggingFace 模型 ID

### A800 GPU 优化

- **数据类型**: `bfloat16`（A800 原生支持，性能更好）
- **GPU 内存限制**: 70GB（留出 10GB 余量）
- **设备映射**: `auto`（自动分配到 GPU）

## 代码变更说明

### 1. 配置扩展 (`okr_config.py`)

- 添加了 `get_deepseek_moe_config()` 函数
- 添加了 `get_deepseek_moe_quick_test_config()` 函数
- 自动识别 DeepSeek-MoE 模型并设置相应参数

### 2. 模型结构适配 (`okr_patch.py`)

- 增强了对 decoder-only 模型的支持
- 添加了多种 router 查找方式（方式6-8），专门针对 DeepSeek-MoE
- 支持从模型结构自动推断专家数量

### 3. 实验框架更新 (`okr_experiment.py`)

- 支持自动检测模型类型（decoder-only vs encoder-decoder）
- 根据模型类型选择合适的加载方式和生成参数

## 模型结构差异

### Switch Transformers (原模型)
- **架构**: Encoder-Decoder
- **专家数量**: 8
- **Top-K**: 1
- **Router 位置**: `layer.layer[1].mlp.router`

### DeepSeek-MoE (新模型)
- **架构**: Decoder-Only
- **专家数量**: 64
- **Top-K**: 8
- **Router 位置**: 可能在 `layer.mlp.gate` 或 `layer.moe.gate`

## 注意事项

1. **模型大小**: DeepSeek-MoE-16B 模型较大，首次加载可能需要较长时间
2. **显存占用**: 即使使用 bfloat16，模型仍需要约 30-40GB 显存
3. **下载速度**: 如果使用 HF 镜像，确保网络连接稳定
4. **Router 查找**: 如果自动查找失败，检查日志中的 router 查找信息

## 故障排除

### 问题1: 无法找到 router

**症状**: 报错 "未找到任何gate层"

**解决方案**:
1. 检查模型是否正确加载
2. 查看日志中的 router 查找过程
3. 如果模型结构特殊，可能需要手动添加 router 查找逻辑

### 问题2: 显存不足

**症状**: CUDA out of memory

**解决方案**:
1. 减小 `batch_size`（默认已设为 1）
2. 减小 `max_length`
3. 使用量化模型（如果可用）
4. 调整 `max_memory` 限制

### 问题3: 模型加载失败

**症状**: 无法从 HuggingFace 加载模型

**解决方案**:
1. 检查网络连接
2. 确认模型名称正确
3. 设置 `trust_remote_code=True`（已在配置中设置）
4. 检查 HuggingFace 镜像配置

## 实验输出

实验完成后，结果保存在：
- **结果文件**: `{output_dir}/results.json`
- **日志文件**: `{output_dir}/deepseek_moe_experiment.log`

## 性能优化建议

1. **使用 bfloat16**: A800 原生支持，性能比 float16 更好
2. **合理设置 max_memory**: 留出余量避免 OOM
3. **小 batch_size**: DeepSeek-MoE 较大，batch_size=1 更稳定
4. **使用 device_map="auto"**: 自动管理 GPU 内存

## 参考

- [OKR 实验指南](./OKR_README.md)
- [DeepSeek-MoE 官方文档](https://huggingface.co/deepseek-ai/DeepSeek-MoE-16B)

