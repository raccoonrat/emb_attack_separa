# 缓存路径迁移 - 快速开始

## ✅ 已完成配置

缓存路径已配置为：`D:\Dev\cache`

## 🚀 快速使用（3种方式）

### 方式1: 在Python脚本中自动设置（推荐）

所有主要脚本已自动添加缓存配置，无需额外操作：
- `deploy_switch_base8.py` ✅
- `test_minimal_switch.py` ✅
- `main.py` ✅
- `mves_experiment.py` ✅

### 方式2: 导入配置文件

在新脚本开头添加：
```python
from cache_config import *
```

### 方式3: 运行环境变量脚本

**Windows批处理**:
```bash
D:\Dev\set_cache_env.bat
```

**PowerShell**:
```powershell
D:\Dev\set_cache_env.ps1
```

---

## 📁 缓存目录结构

```
D:\Dev\cache\
├── huggingface\      # Hugging Face模型和tokenizer
│   ├── hub\          # 模型权重
│   └── datasets\     # 数据集缓存
├── torch\            # PyTorch预训练权重
└── pip\              # pip包缓存
```

---

## 🔍 验证配置

```python
import os
print("HF_HOME:", os.environ.get("HF_HOME"))
print("TRANSFORMERS_CACHE:", os.environ.get("TRANSFORMERS_CACHE"))
print("TORCH_HOME:", os.environ.get("TORCH_HOME"))
```

应该显示：
```
HF_HOME: D:/Dev/cache/huggingface
TRANSFORMERS_CACHE: D:/Dev/cache/huggingface/hub
TORCH_HOME: D:/Dev/cache/torch
```

---

## 📦 迁移现有缓存（可选）

如果需要迁移C盘现有缓存：

```bash
# 预览迁移（不实际复制）
python setup_cache_path.py --dry-run

# 实际迁移
python setup_cache_path.py --migrate --all
```

---

## ⚙️ 系统环境变量（永久生效）

如果需要永久设置，添加到系统环境变量：

| 变量名 | 值 |
|--------|-----|
| `HF_HOME` | `D:\Dev\cache\huggingface` |
| `TRANSFORMERS_CACHE` | `D:\Dev\cache\huggingface\hub` |
| `HF_DATASETS_CACHE` | `D:\Dev\cache\huggingface\datasets` |
| `TORCH_HOME` | `D:\Dev\cache\torch` |
| `PIP_CACHE_DIR` | `D:\Dev\cache\pip` |

---

## 📝 相关文件

- `cache_config.py` - Python配置文件（自动生成）
- `setup_cache_path.py` - 迁移脚本
- `CACHE_MIGRATION.md` - 详细迁移指南
- `D:\Dev\set_cache_env.bat` - Windows批处理脚本
- `D:\Dev\set_cache_env.ps1` - PowerShell脚本

---

## ✅ 检查清单

- [x] 缓存目录已创建
- [x] 环境变量脚本已生成
- [x] Python配置文件已创建
- [x] 主要脚本已添加缓存配置

**下一步**: 直接运行脚本，缓存会自动保存到D盘！

