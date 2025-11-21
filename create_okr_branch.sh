#!/bin/bash
# 创建OKR独立分支的脚本

set -e

echo "============================================================"
echo "创建 OKR 独立分支"
echo "============================================================"

# 检查当前分支
CURRENT_BRANCH=$(git branch --show-current)
echo "当前分支: $CURRENT_BRANCH"

# 检查是否有未提交的更改
if [ -n "$(git status --porcelain)" ]; then
    echo ""
    echo "检测到未提交的更改:"
    git status --short
    echo ""
    read -p "是否先提交这些更改? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "暂存所有更改..."
        git add -A
        read -p "请输入提交信息: " COMMIT_MSG
        if [ -z "$COMMIT_MSG" ]; then
            COMMIT_MSG="WIP: OKR相关更改"
        fi
        git commit -m "$COMMIT_MSG"
    else
        echo "将使用 git stash 暂存更改..."
        git stash push -m "临时暂存: 创建OKR分支前"
    fi
fi

# 创建新分支
BRANCH_NAME="okr"
echo ""
echo "创建新分支: $BRANCH_NAME"
git checkout -b $BRANCH_NAME

echo ""
echo "============================================================"
echo "分支创建完成！"
echo "============================================================"
echo "当前分支: $(git branch --show-current)"
echo ""
echo "OKR相关文件:"
echo "  - experiment/okr_*.py"
echo "  - experiment/OKR_*.md"
echo "  - experiment/run_okr_*.py"
echo "  - experiment/test_okr_*.py"
echo "  - okr/"
echo "  - OKR_Colab_Experiment.ipynb"
echo ""
echo "可以使用以下命令查看分支:"
echo "  git branch -a"
echo ""
echo "如果需要推送到远程:"
echo "  git push -u origin $BRANCH_NAME"

