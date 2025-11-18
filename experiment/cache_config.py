"""
缓存路径配置

在运行任何脚本前，先设置这些环境变量
支持Hugging Face镜像源加速下载
"""

import os
from pathlib import Path

# 缓存基础路径
CACHE_BASE = Path("D:/Dev/cache")

# 设置Hugging Face缓存
os.environ["HF_HOME"] = str(CACHE_BASE / "huggingface")
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_BASE / "huggingface" / "hub")
os.environ["HF_DATASETS_CACHE"] = str(CACHE_BASE / "huggingface" / "datasets")

# 设置PyTorch缓存
os.environ["TORCH_HOME"] = str(CACHE_BASE / "torch")

# 设置pip缓存
os.environ["PIP_CACHE_DIR"] = str(CACHE_BASE / "pip")

# 设置Hugging Face镜像源（加速下载）
# 使用官方镜像: https://hf-mirror.com
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

print("✓ 缓存路径已设置到D盘")
print(f"  HF_HOME: {os.environ['HF_HOME']}")
print(f"  TORCH_HOME: {os.environ['TORCH_HOME']}")
print(f"  HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', '默认')}")
