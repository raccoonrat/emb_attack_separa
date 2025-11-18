#!/usr/bin/env python3
"""
WSL 环境启动脚本

自动检测环境并使用相应的缓存配置
"""

import os
import sys
import platform

# 检测运行环境
def detect_environment():
    """检测当前运行环境"""
    system = platform.system()
    release = platform.release().lower()
    
    # 检查是否在 WSL
    is_wsl = (
        "microsoft" in release or 
        "wsl" in release or
        os.path.exists("/proc/version") and "microsoft" in open("/proc/version").read().lower()
    )
    
    # 检查是否在 Windows（通过路径分隔符）
    is_windows = os.name == 'nt' or system == 'Windows'
    
    if is_wsl or (not is_windows and system == 'Linux'):
        return "wsl"
    elif is_windows:
        return "windows"
    else:
        return "linux"

# 根据环境加载相应配置
env = detect_environment()
print(f"检测到环境: {env}")

if env == "wsl":
    try:
        import cache_config_wsl
        print("✓ 已加载 WSL 缓存配置")
    except ImportError:
        print("⚠ 警告: 无法导入 cache_config_wsl，使用默认配置")
        cache_base = os.path.expanduser("~/.cache/emb_attack_separa")
        os.makedirs(cache_base, exist_ok=True)
        os.environ.setdefault("HF_HOME", f"{cache_base}/huggingface")
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
elif env == "windows":
    try:
        import cache_config
        print("✓ 已加载 Windows 缓存配置")
    except ImportError:
        print("⚠ 警告: 无法导入 cache_config，使用默认配置")
else:
    # Linux 环境，使用 WSL 配置（路径格式相同）
    try:
        import cache_config_wsl
        print("✓ 已加载 Linux 缓存配置")
    except ImportError:
        print("⚠ 警告: 无法导入 cache_config_wsl，使用默认配置")

# 导入 main 并运行
if __name__ == "__main__":
    # 将命令行参数传递给 main.py
    from main import main
    main()

