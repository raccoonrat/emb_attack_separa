"""
设置缓存路径到D盘

将Hugging Face、PyTorch等缓存从C盘迁移到D盘
"""

import os
import shutil
import sys
from pathlib import Path

# 新的缓存路径
NEW_CACHE_BASE = Path("D:/Dev/cache")
OLD_CACHE_BASE = Path.home() / ".cache"

# 需要迁移的缓存目录
CACHE_DIRS = {
    "huggingface": {
        "old": OLD_CACHE_BASE / "huggingface",
        "new": NEW_CACHE_BASE / "huggingface",
        "env_vars": ["HF_HOME", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE"]
    },
    "torch": {
        "old": OLD_CACHE_BASE / "torch",
        "new": NEW_CACHE_BASE / "torch",
        "env_vars": ["TORCH_HOME"]
    },
    "pip": {
        "old": OLD_CACHE_BASE / "pip",
        "new": NEW_CACHE_BASE / "pip",
        "env_vars": ["PIP_CACHE_DIR"]
    }
}


def create_cache_dirs():
    """创建新的缓存目录"""
    print("创建新的缓存目录...")
    NEW_CACHE_BASE.mkdir(parents=True, exist_ok=True)
    for cache_info in CACHE_DIRS.values():
        cache_info["new"].mkdir(parents=True, exist_ok=True)
        print(f"  [OK] {cache_info['new']}")


def migrate_cache(cache_name: str, dry_run: bool = False):
    """迁移缓存目录"""
    cache_info = CACHE_DIRS[cache_name]
    old_path = cache_info["old"]
    new_path = cache_info["new"]
    
    if not old_path.exists():
        print(f"  [WARN] {cache_name} 缓存不存在: {old_path}")
        return False
    
    print(f"\n迁移 {cache_name} 缓存...")
    print(f"  源路径: {old_path}")
    print(f"  目标路径: {new_path}")
    
    if new_path.exists() and any(new_path.iterdir()):
        print(f"  [WARN] 目标目录已存在且非空，跳过迁移")
        return False
    
    if dry_run:
        try:
            total_size = sum(f.stat().st_size for f in old_path.rglob('*') if f.is_file()) / 1024**3
            print(f"  [DRY RUN] 将复制 {total_size:.2f} GB")
        except:
            print(f"  [DRY RUN] 将复制缓存文件")
        return True
    
    try:
        # 复制文件
        print(f"  正在复制文件...")
        shutil.copytree(old_path, new_path, dirs_exist_ok=True)
        
        # 计算大小
        total_size = sum(f.stat().st_size for f in new_path.rglob('*') if f.is_file())
        print(f"  [OK] 迁移完成 ({total_size / 1024**3:.2f} GB)")
        
        return True
    except Exception as e:
        print(f"  [ERROR] 迁移失败: {e}")
        return False


def set_environment_variables():
    """设置环境变量"""
    print("\n设置环境变量...")
    
    env_script = []
    env_script.append("@echo off")
    env_script.append("REM 设置缓存路径到D盘")
    env_script.append("")
    
    # 设置所有缓存路径
    for cache_info in CACHE_DIRS.values():
        for env_var in cache_info["env_vars"]:
            value = str(cache_info["new"]).replace("\\", "/")
            env_script.append(f'set {env_var}={value}')
            print(f"  {env_var} = {value}")
    
    # 保存到批处理文件
    bat_file = Path("D:/Dev/set_cache_env.bat")
    with open(bat_file, "w", encoding="utf-8") as f:
        f.write("\n".join(env_script))
    
    print(f"\n[OK] 环境变量脚本已保存到: {bat_file}")
    print("  每次使用前运行: D:\\Dev\\set_cache_env.bat")
    
    # 同时创建PowerShell版本
    ps_script = []
    ps_script.append("# 设置缓存路径到D盘")
    ps_script.append("")
    
    for cache_info in CACHE_DIRS.values():
        for env_var in cache_info["env_vars"]:
            value = str(cache_info["new"]).replace("\\", "/")
            ps_script.append(f'$env:{env_var} = "{value}"')
    
    ps_file = Path("D:/Dev/set_cache_env.ps1")
    with open(ps_file, "w", encoding="utf-8") as f:
        f.write("\n".join(ps_script))
    
    print(f"[OK] PowerShell脚本已保存到: {ps_file}")


def create_python_config():
    """创建Python配置文件"""
    print("\n创建Python缓存配置...")
    
    config_content = '''"""
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
'''
    
    # 获取脚本所在目录
    script_dir = Path(__file__).parent
    config_file = script_dir / "cache_config.py"
    with open(config_file, "w", encoding="utf-8") as f:
        f.write(config_content)
    
    print(f"[OK] Python配置文件已创建: {config_file}")


def update_requirements():
    """更新requirements.txt，添加说明"""
    script_dir = Path(__file__).parent
    req_file = script_dir / "requirements.txt"
    
    if req_file.exists():
        content = req_file.read_text(encoding="utf-8")
        
        if "cache_config" not in content:
            content += "\n\n# 缓存配置说明:\n"
            content += "# 运行前先设置环境变量或导入 cache_config.py\n"
            content += "# 详见: CACHE_MIGRATION.md\n"
            
            req_file.write_text(content, encoding="utf-8")
            print("[OK] requirements.txt 已更新")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="设置缓存路径到D盘")
    parser.add_argument("--migrate", action="store_true", help="迁移现有缓存")
    parser.add_argument("--dry-run", action="store_true", help="仅预览，不实际迁移")
    parser.add_argument("--all", action="store_true", help="迁移所有缓存")
    
    args = parser.parse_args()
    
    print("="*60)
    print("缓存路径迁移工具")
    print("="*60)
    print(f"\n新缓存路径: {NEW_CACHE_BASE}")
    print(f"旧缓存路径: {OLD_CACHE_BASE}")
    
    # 1. 创建新目录
    create_cache_dirs()
    
    # 2. 迁移缓存（如果指定）
    if args.migrate or args.all:
        print("\n" + "="*60)
        print("迁移缓存")
        print("="*60)
        
        if args.dry_run:
            print("\n[DRY RUN 模式 - 仅预览]")
        
        for cache_name in CACHE_DIRS.keys():
            migrate_cache(cache_name, dry_run=args.dry_run)
    
    # 3. 设置环境变量
    set_environment_variables()
    
    # 4. 创建Python配置
    create_python_config()
    
    # 5. 更新requirements
    update_requirements()
    
    print("\n" + "="*60)
    print("[OK] 配置完成！")
    print("="*60)
    print("\n下一步:")
    print("1. 如果迁移了缓存，可以删除旧缓存目录释放C盘空间")
    print("2. 在运行Python脚本前，先导入 cache_config.py 或运行 set_cache_env.bat")
    print("3. 或者将环境变量添加到系统环境变量中（永久生效）")
    print("\n示例:")
    print("  from cache_config import *  # 在脚本开头添加")
    print("  或")
    print("  D:\\Dev\\set_cache_env.bat  # 在命令行运行")


if __name__ == "__main__":
    main()

