"""
WSL 环境缓存路径配置

在 WSL 环境下运行任何脚本前，先设置这些环境变量
支持 Hugging Face 镜像源加速下载
"""

import os
from pathlib import Path

# WSL 环境下的缓存路径（使用 Linux 路径）
# 放在用户主目录的 .cache 目录下
CACHE_BASE = Path.home() / ".cache" / "emb_attack_separa"

# 创建缓存目录
CACHE_BASE.mkdir(parents=True, exist_ok=True)

# 设置 Hugging Face 缓存
os.environ["HF_HOME"] = str(CACHE_BASE / "huggingface")
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_BASE / "huggingface" / "hub")
os.environ["HF_DATASETS_CACHE"] = str(CACHE_BASE / "huggingface" / "datasets")

# 设置 PyTorch 缓存
os.environ["TORCH_HOME"] = str(CACHE_BASE / "torch")

# 设置 pip 缓存
os.environ["PIP_CACHE_DIR"] = str(CACHE_BASE / "pip")

# 设置 Hugging Face 镜像源（加速下载）
# 使用官方镜像: https://hf-mirror.com
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

print("✓ WSL 缓存路径已设置")
print(f"  HF_HOME: {os.environ['HF_HOME']}")
print(f"  TORCH_HOME: {os.environ['TORCH_HOME']}")
print(f"  HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', '默认')}")

