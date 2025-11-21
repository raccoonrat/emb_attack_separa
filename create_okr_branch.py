#!/usr/bin/env python3
"""
创建OKR独立分支的脚本
"""

import subprocess
import sys
from pathlib import Path

def run_git_command(cmd, check=True):
    """运行git命令"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=check
        )
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except subprocess.CalledProcessError as e:
        return e.stdout.strip(), e.stderr.strip(), e.returncode

def main():
    print("=" * 60)
    print("创建 OKR 独立分支")
    print("=" * 60)
    
    # 检查当前分支
    stdout, stderr, code = run_git_command("git branch --show-current", check=False)
    if code != 0:
        print(f"错误: 无法获取当前分支")
        print(f"stderr: {stderr}")
        sys.exit(1)
    
    current_branch = stdout
    print(f"当前分支: {current_branch}")
    
    # 检查是否有未提交的更改
    stdout, stderr, code = run_git_command("git status --porcelain", check=False)
    has_changes = bool(stdout.strip())
    
    if has_changes:
        print("\n检测到未提交的更改:")
        print(stdout)
        print("\n选项:")
        print("1. 提交这些更改")
        print("2. 暂存这些更改 (git stash)")
        print("3. 放弃这些更改 (git restore)")
        
        choice = input("\n请选择 (1/2/3，直接回车默认暂存): ").strip()
        
        if choice == "1":
            print("\n暂存所有更改...")
            run_git_command("git add -A")
            commit_msg = input("请输入提交信息 (直接回车使用默认): ").strip()
            if not commit_msg:
                commit_msg = "WIP: OKR相关更改"
            run_git_command(f'git commit -m "{commit_msg}"')
        elif choice == "3":
            print("\n放弃所有更改...")
            run_git_command("git restore .")
            run_git_command("git clean -fd")
        else:
            print("\n暂存更改...")
            run_git_command('git stash push -m "临时暂存: 创建OKR分支前"')
    
    # 创建新分支
    branch_name = "okr"
    print(f"\n创建新分支: {branch_name}")
    
    # 检查分支是否已存在
    stdout, stderr, code = run_git_command(f"git branch --list {branch_name}", check=False)
    if stdout.strip():
        print(f"警告: 分支 {branch_name} 已存在")
        switch = input(f"是否切换到该分支? (y/n): ").strip().lower()
        if switch == 'y':
            run_git_command(f"git checkout {branch_name}")
        else:
            print("操作取消")
            sys.exit(0)
    else:
        # 创建并切换到新分支
        stdout, stderr, code = run_git_command(f"git checkout -b {branch_name}", check=False)
        if code != 0:
            print(f"错误: 无法创建分支")
            print(f"stderr: {stderr}")
            sys.exit(1)
    
    # 确认当前分支
    stdout, stderr, code = run_git_command("git branch --show-current", check=False)
    current_branch = stdout
    
    print("\n" + "=" * 60)
    print("分支创建完成！")
    print("=" * 60)
    print(f"当前分支: {current_branch}")
    print("\nOKR相关文件:")
    print("  - experiment/okr_*.py")
    print("  - experiment/OKR_*.md")
    print("  - experiment/run_okr_*.py")
    print("  - experiment/test_okr_*.py")
    print("  - okr/")
    print("  - OKR_Colab_Experiment.ipynb")
    print("\n可以使用以下命令查看分支:")
    print("  git branch -a")
    print("\n如果需要推送到远程:")
    print(f"  git push -u origin {branch_name}")

if __name__ == "__main__":
    main()

