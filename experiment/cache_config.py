"""
缓存路径配置

在运行任何脚本前，先设置这些环境变量
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

print("✓ 缓存路径已设置到D盘")
print(f"  HF_HOME: {os.environ['HF_HOME']}")
print(f"  TORCH_HOME: {os.environ['TORCH_HOME']}")
