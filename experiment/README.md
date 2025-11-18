# Experiment 目录说明

> **主要文档**：请查看根目录的 [`README.md`](../README.md) 获取完整的使用手册和项目说明。

本目录包含 MoE 水印方案的 Python 实现代码。

## 快速开始

```bash
# 1. 嵌入水印
python main.py --mode embed \
    --model_name google/switch-base-8 \
    --prompt "Your text" \
    --secret_key "my_secret_key_123"

# 2. 检测水印
python main.py --mode detect \
    --model_name google/switch-base-8 \
    --text_to_check "生成的完整文本" \
    --secret_key "my_secret_key_123"
```

## 核心文件

- `main.py` - 主程序入口（支持 embed/detect/calibrate/experiment 模式）
- `detector.py` - LLR 水印检测器实现
- `mves_watermark_corrected.py` - 水印嵌入实现（Switch Transformers）
- `mves_config.py` - 配置管理
- `calibration.py` - 参数标定工具

## 详细文档

- **使用手册**: [`../README.md`](../README.md) - 完整的使用说明和快速开始
- **检测指南**: [`DETECTION_GUIDE.md`](DETECTION_GUIDE.md) - 检测模式的详细说明
- **阈值标定**: [`THRESHOLD_EXPLANATION.md`](THRESHOLD_EXPLANATION.md) - LLR 阈值理论依据
- **WSL 配置**: [`WSL_TEST_GUIDE.md`](WSL_TEST_GUIDE.md) - WSL 环境配置说明

## 环境要求

- Python 3.10+
- PyTorch（支持 CUDA 可选）
- Transformers
- 详见 `requirements.txt`

## 安装

```bash
# Conda（推荐）
conda env create -f environment.yml
conda activate emb_attack_separa

# 或 pip
pip install -r requirements.txt
```
