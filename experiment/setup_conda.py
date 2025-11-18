"""
配置Miniconda3在Cursor中使用

检测miniconda3安装路径并配置环境
"""

import os
import sys
from pathlib import Path
import subprocess

def find_miniconda():
    """查找miniconda3安装路径"""
    possible_paths = [
        Path.home() / "miniconda3",
        Path.home() / "AppData" / "Local" / "Continuum" / "miniconda3",
        Path.home() / "AppData" / "Local" / "miniconda3",
        Path("C:/ProgramData/miniconda3"),
        Path("C:/miniconda3"),
    ]
    
    # 从环境变量查找
    if "CONDA_PREFIX" in os.environ:
        conda_path = Path(os.environ["CONDA_PREFIX"]).parent
        if conda_path.exists():
            return conda_path
    
    # 从PATH查找
    path_env = os.environ.get("PATH", "")
    for path_str in path_env.split(os.pathsep):
        path = Path(path_str)
        if "miniconda3" in str(path).lower() or "conda" in str(path).lower():
            # 查找conda.exe
            conda_exe = path / "conda.exe"
            if conda_exe.exists():
                return conda_exe.parent.parent
    
    # 尝试常见路径
    for path in possible_paths:
        if path.exists():
            conda_exe = path / "Scripts" / "conda.exe"
            if conda_exe.exists():
                return path
    
    return None


def get_conda_info():
    """获取conda信息"""
    try:
        result = subprocess.run(
            ["conda", "--version"],
            capture_output=True,
            text=True,
            shell=True
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    return None


def create_conda_env():
    """创建conda环境"""
    env_name = "moe_watermark"
    python_version = "3.10"
    
    print(f"\n创建conda环境: {env_name} (Python {python_version})...")
    
    try:
        # 检查环境是否已存在
        result = subprocess.run(
            ["conda", "env", "list"],
            capture_output=True,
            text=True,
            shell=True
        )
        
        if env_name in result.stdout:
            print(f"  [INFO] 环境 {env_name} 已存在")
            return env_name
        
        # 创建新环境
        print(f"  正在创建环境...")
        result = subprocess.run(
            ["conda", "create", "-n", env_name, f"python={python_version}", "-y"],
            capture_output=True,
            text=True,
            shell=True
        )
        
        if result.returncode == 0:
            print(f"  [OK] 环境创建成功")
            return env_name
        else:
            print(f"  [ERROR] 创建失败: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"  [ERROR] 创建环境时出错: {e}")
        return None


def create_activation_script(conda_path: Path):
    """创建激活脚本"""
    env_name = "moe_watermark"
    
    # Windows批处理脚本
    bat_content = f"""@echo off
REM 激活conda环境并设置缓存路径

REM 初始化conda
call "{conda_path}\\Scripts\\activate.bat"

REM 激活环境
call conda activate {env_name}

REM 设置缓存路径
set HF_HOME=D:\\Dev\\cache\\huggingface
set TRANSFORMERS_CACHE=D:\\Dev\\cache\\huggingface\\hub
set HF_DATASETS_CACHE=D:\\Dev\\cache\\huggingface\\datasets
set TORCH_HOME=D:\\Dev\\cache\\torch
set HF_ENDPOINT=https://hf-mirror.com

echo.
echo [OK] Conda环境已激活: {env_name}
echo [OK] 缓存路径已设置到D盘
echo [OK] Hugging Face镜像已配置
echo.
"""
    
    # 获取脚本所在目录
    script_dir = Path(__file__).parent
    bat_file = script_dir / "activate_env.bat"
    with open(bat_file, "w", encoding="utf-8") as f:
        f.write(bat_content)
    
    print(f"  [OK] 批处理脚本已创建: {bat_file}")
    
    # PowerShell脚本
    ps_content = f"""# 激活conda环境并设置缓存路径

# 初始化conda
& "{conda_path}\\Scripts\\Activate.ps1"

# 激活环境
conda activate {env_name}

# 设置缓存路径
$env:HF_HOME = "D:\\Dev\\cache\\huggingface"
$env:TRANSFORMERS_CACHE = "D:\\Dev\\cache\\huggingface\\hub"
$env:HF_DATASETS_CACHE = "D:\\Dev\\cache\\huggingface\\datasets"
$env:TORCH_HOME = "D:\\Dev\\cache\\torch"
$env:HF_ENDPOINT = "https://hf-mirror.com"

Write-Host ""
Write-Host "[OK] Conda环境已激活: {env_name}" -ForegroundColor Green
Write-Host "[OK] 缓存路径已设置到D盘" -ForegroundColor Green
Write-Host "[OK] Hugging Face镜像已配置" -ForegroundColor Green
Write-Host ""
"""
    
    script_dir = Path(__file__).parent
    ps_file = script_dir / "activate_env.ps1"
    with open(ps_file, "w", encoding="utf-8") as f:
        f.write(ps_content)
    
    print(f"  [OK] PowerShell脚本已创建: {ps_file}")


def create_cursor_settings():
    """创建Cursor设置说明"""
    conda_path = find_miniconda()
    if not conda_path:
        return
    
    env_name = "moe_watermark"
    
    settings_content = f"""# Cursor中使用Miniconda3配置

## 方法1: 使用终端配置文件（推荐）

### PowerShell配置

在Cursor中按 `Ctrl+Shift+P`，输入 "Preferences: Open User Settings (JSON)"

添加以下配置：

```json
{{
    "terminal.integrated.defaultProfile.windows": "PowerShell",
    "terminal.integrated.profiles.windows": {{
        "PowerShell (Conda)": {{
            "path": "C:\\\\Windows\\\\System32\\\\WindowsPowerShell\\\\v1.0\\\\powershell.exe",
            "args": [
                "-NoExit",
                "-Command",
                "& '{conda_path}\\\\Scripts\\\\Activate.ps1'; conda activate {env_name}"
            ],
            "icon": "terminal-powershell"
        }}
    }}
}}
```

### CMD配置

```json
{{
    "terminal.integrated.defaultProfile.windows": "Command Prompt (Conda)",
    "terminal.integrated.profiles.windows": {{
        "Command Prompt (Conda)": {{
            "path": "C:\\\\Windows\\\\System32\\\\cmd.exe",
            "args": [
                "/K",
                "{conda_path}\\\\Scripts\\\\activate.bat {env_name}"
            ]
        }}
    }}
}}
```

## 方法2: 手动激活

每次打开终端后运行：

```bash
# PowerShell
& "{conda_path}\\Scripts\\Activate.ps1"
conda activate {env_name}

# CMD
call "{conda_path}\\Scripts\\activate.bat"
conda activate {env_name}
```

## 方法3: 使用提供的脚本

```bash
# 在experiment目录下
.\\activate_env.bat    # Windows批处理
.\\activate_env.ps1   # PowerShell
```

## 验证配置

```bash
# 检查conda
conda --version

# 检查Python
python --version

# 检查环境
conda env list

# 检查当前环境
conda info --envs
```
"""
    
    script_dir = Path(__file__).parent
    settings_file = script_dir / "CURSOR_CONDA_SETUP.md"
    with open(settings_file, "w", encoding="utf-8") as f:
        f.write(settings_content)
    
    print(f"  [OK] Cursor配置说明已创建: {settings_file}")


def main():
    """主函数"""
    print("="*60)
    print("Miniconda3配置工具")
    print("="*60)
    
    # 1. 查找miniconda
    print("\n1. 查找Miniconda3安装路径...")
    conda_path = find_miniconda()
    
    if conda_path:
        print(f"  [OK] 找到Miniconda3: {conda_path}")
    else:
        print("  [ERROR] 未找到Miniconda3安装路径")
        print("\n请手动指定路径，或确保conda在PATH中")
        print("常见路径:")
        print("  - C:\\Users\\<username>\\miniconda3")
        print("  - C:\\Users\\<username>\\AppData\\Local\\miniconda3")
        return
    
    # 2. 检查conda
    print("\n2. 检查conda...")
    conda_version = get_conda_info()
    if conda_version:
        print(f"  [OK] {conda_version}")
    else:
        print("  [WARN] 无法运行conda命令，请确保conda在PATH中")
        print(f"  尝试添加: {conda_path}\\Scripts 到PATH")
    
    # 3. 创建环境
    print("\n3. 创建conda环境...")
    env_name = create_conda_env()
    
    if not env_name:
        print("  [WARN] 无法创建环境，请手动创建:")
        print("    conda create -n moe_watermark python=3.10 -y")
        env_name = "moe_watermark"  # 使用默认名称
    
    # 4. 创建激活脚本
    print("\n4. 创建激活脚本...")
    create_activation_script(conda_path)
    
    # 5. 创建Cursor设置说明
    print("\n5. 创建Cursor配置说明...")
    create_cursor_settings()
    
    print("\n" + "="*60)
    print("[OK] 配置完成！")
    print("="*60)
    print("\n下一步:")
    print("1. 查看 CURSOR_CONDA_SETUP.md 了解如何在Cursor中配置")
    print("2. 运行 activate_env.bat 或 activate_env.ps1 激活环境")
    print("3. 安装依赖: pip install -r requirements.txt")
    print(f"\nConda路径: {conda_path}")
    print(f"环境名称: {env_name}")


if __name__ == "__main__":
    main()

