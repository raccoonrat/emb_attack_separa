"""
WSL 环境快速测试脚本

验证 WSL 环境配置是否正确
"""

import sys
import os

# 导入 WSL 缓存配置（必须在导入 transformers 之前）
try:
    import cache_config_wsl
    print("✓ 已加载 WSL 缓存配置")
except ImportError:
    print("⚠ 警告: 无法导入 cache_config_wsl，使用默认配置")
    # 设置基本环境变量
    cache_base = os.path.expanduser("~/.cache/emb_attack_separa")
    os.environ.setdefault("HF_HOME", f"{cache_base}/huggingface")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

print("\n" + "="*60)
print("WSL 环境测试")
print("="*60)

# 1. 检查 Python 版本
print("\n1. Python 环境检查:")
print(f"   Python 版本: {sys.version}")
if sys.version_info < (3, 10):
    print("   ⚠ 警告: 建议使用 Python 3.10 或更高版本")
else:
    print("   ✓ Python 版本符合要求")

# 2. 检查关键包
print("\n2. 关键包检查:")
required_packages = {
    "torch": "PyTorch",
    "transformers": "Transformers",
    "numpy": "NumPy",
    "scipy": "SciPy",
    "datasets": "Datasets",
}

missing_packages = []
for package, name in required_packages.items():
    try:
        mod = __import__(package)
        version = getattr(mod, "__version__", "未知")
        print(f"   ✓ {name}: {version}")
    except ImportError:
        print(f"   ✗ {name}: 未安装")
        missing_packages.append(package)

if missing_packages:
    print(f"\n   请安装缺失的包: pip install {' '.join(missing_packages)}")
    sys.exit(1)

# 3. 检查 CUDA
print("\n3. CUDA 检查:")
try:
    import torch
    if torch.cuda.is_available():
        print(f"   ✓ CUDA 可用")
        print(f"   GPU 设备: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA 版本: {torch.version.cuda}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("   ⚠ CUDA 不可用，将使用 CPU（速度较慢）")
        print("   提示: 如需使用 GPU，请安装 WSL 专用 NVIDIA 驱动和 CUDA Toolkit")
except Exception as e:
    print(f"   ✗ 检查 CUDA 时出错: {e}")

# 4. 检查环境变量
print("\n4. 环境变量检查:")
env_vars = ["HF_HOME", "TRANSFORMERS_CACHE", "HF_ENDPOINT", "TORCH_HOME"]
for var in env_vars:
    value = os.environ.get(var, "未设置")
    if value != "未设置":
        print(f"   ✓ {var}: {value}")
    else:
        print(f"   ⚠ {var}: 未设置")

# 5. 测试 Hugging Face 连接
print("\n5. Hugging Face 连接测试:")
try:
    from transformers import AutoTokenizer
    import time
    
    print("   正在测试下载 tokenizer...")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")
    elapsed = time.time() - start
    
    print(f"   ✓ 下载成功（耗时: {elapsed:.2f}秒）")
    print(f"   词汇表大小: {len(tokenizer)}")
except Exception as e:
    print(f"   ✗ 下载失败: {e}")
    print("   提示: 检查网络连接和镜像源配置")

# 6. 检查文件系统
print("\n6. 文件系统检查:")
import platform
print(f"   系统: {platform.system()}")
print(f"   平台: {platform.platform()}")

# 检查是否在 WSL
if "microsoft" in platform.release().lower() or "wsl" in platform.release().lower():
    print("   ✓ 检测到 WSL 环境")
else:
    print("   ⚠ 未检测到 WSL 环境（可能运行在原生 Linux）")

# 检查缓存目录
cache_base = os.path.expanduser("~/.cache/emb_attack_separa")
if os.path.exists(cache_base):
    print(f"   ✓ 缓存目录存在: {cache_base}")
else:
    print(f"   ⚠ 缓存目录不存在，将自动创建: {cache_base}")
    os.makedirs(cache_base, exist_ok=True)

print("\n" + "="*60)
if missing_packages:
    print("⚠ 部分检查未通过，请安装缺失的包后重试")
else:
    print("✓ 所有基础检查通过！")
    print("\n下一步:")
    print("  1. 运行最小测试: python test_minimal_switch.py")
    print("  2. 运行完整测试: python main.py --mode calibrate --num_calib_samples 10")
print("="*60)

