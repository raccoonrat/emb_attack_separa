#!/bin/bash
# 使用 sudo 运行 OKR 测试脚本（访问 /root 目录下的模型）

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/test_okr_deepseek_moe_local.py"

echo "============================================================"
echo "使用 sudo 运行 OKR 测试（访问 /root 目录下的模型）"
echo "============================================================"
echo ""

# 检查脚本是否存在
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "错误: 找不到脚本 $PYTHON_SCRIPT"
    exit 1
fi

# 使用 sudo 运行 Python 脚本
# 注意：需要输入 sudo 密码
python "$PYTHON_SCRIPT"

