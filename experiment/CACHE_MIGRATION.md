# 缓存路径迁移指南

将Hugging Face、PyTorch等缓存从C盘迁移到D盘，释放C盘存储空间。

## 一、快速迁移（推荐）

### 方法1: 使用迁移脚本（自动）

```bash
cd experiment

# 1. 预览迁移（不实际复制）
python setup_cache_path.py --dry-run

# 2. 迁移所有缓存
python setup_cache_path.py --migrate --all

# 3. 迁移完成后，设置环境变量
python setup_cache_path.py
```

### 方法2: 手动迁移

```bash
# 1. 创建新缓存目录
mkdir D:\Dev\cache\huggingface
mkdir D:\Dev\cache\torch
mkdir D:\Dev\cache\pip

# 2. 复制缓存（如果存在）
xcopy /E /I /Y C:\Users\wangyh43\.cache\huggingface D:\Dev\cache\huggingface
xcopy /E /I /Y C:\Users\wangyh43\.cache\torch D:\Dev\cache\torch
xcopy /E /I /Y C:\Users\wangyh43\.cache\pip D:\Dev\cache\pip

# 3. 设置环境变量（见下文）
```

---

## 二、设置环境变量

### 方法1: 使用提供的脚本（临时，每次会话）

**Windows批处理**:
```bash
# 运行批处理文件
D:\Dev\set_cache_env.bat
```

**PowerShell**:
```powershell
# 运行PowerShell脚本
D:\Dev\set_cache_env.ps1
```

### 方法2: 在Python脚本中设置（推荐）

在每个Python脚本开头添加：

```python
# 在脚本开头添加
from cache_config import *
```

或直接设置：

```python
import os
from pathlib import Path

CACHE_BASE = Path("D:/Dev/cache")

os.environ["HF_HOME"] = str(CACHE_BASE / "huggingface")
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_BASE / "huggingface" / "hub")
os.environ["HF_DATASETS_CACHE"] = str(CACHE_BASE / "huggingface" / "datasets")
os.environ["TORCH_HOME"] = str(CACHE_BASE / "torch")
os.environ["PIP_CACHE_DIR"] = str(CACHE_BASE / "pip")
```

### 方法3: 系统环境变量（永久生效）

**Windows设置**:
1. 按 `Win + R`，输入 `sysdm.cpl`，回车
2. 点击"高级" → "环境变量"
3. 在"用户变量"中添加：

| 变量名 | 变量值 |
|--------|--------|
| `HF_HOME` | `D:\Dev\cache\huggingface` |
| `TRANSFORMERS_CACHE` | `D:\Dev\cache\huggingface\hub` |
| `HF_DATASETS_CACHE` | `D:\Dev\cache\huggingface\datasets` |
| `TORCH_HOME` | `D:\Dev\cache\torch` |
| `PIP_CACHE_DIR` | `D:\Dev\cache\pip` |

**PowerShell（管理员）**:
```powershell
[System.Environment]::SetEnvironmentVariable("HF_HOME", "D:\Dev\cache\huggingface", "User")
[System.Environment]::SetEnvironmentVariable("TRANSFORMERS_CACHE", "D:\Dev\cache\huggingface\hub", "User")
[System.Environment]::SetEnvironmentVariable("HF_DATASETS_CACHE", "D:\Dev\cache\huggingface\datasets", "User")
[System.Environment]::SetEnvironmentVariable("TORCH_HOME", "D:\Dev\cache\torch", "User")
[System.Environment]::SetEnvironmentVariable("PIP_CACHE_DIR", "D:\Dev\cache\pip", "User")
```

---

## 三、验证配置

### 检查环境变量

**Python**:
```python
import os
print("HF_HOME:", os.environ.get("HF_HOME", "未设置"))
print("TRANSFORMERS_CACHE:", os.environ.get("TRANSFORMERS_CACHE", "未设置"))
print("TORCH_HOME:", os.environ.get("TORCH_HOME", "未设置"))
```

**命令行**:
```bash
echo %HF_HOME%
echo %TRANSFORMERS_CACHE%
echo %TORCH_HOME%
```

### 测试模型下载

```python
from transformers import AutoTokenizer

# 这会下载到新路径
tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")
print(f"缓存路径: {tokenizer.cache_dir}")
```

---

## 四、更新现有脚本

### 在所有脚本开头添加

```python
# 在所有实验脚本开头添加
import sys
from pathlib import Path

# 添加缓存配置
sys.path.insert(0, str(Path(__file__).parent))
try:
    from cache_config import *
except ImportError:
    # 如果cache_config不存在，手动设置
    import os
    CACHE_BASE = Path("D:/Dev/cache")
    os.environ["HF_HOME"] = str(CACHE_BASE / "huggingface")
    os.environ["TRANSFORMERS_CACHE"] = str(CACHE_BASE / "huggingface" / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(CACHE_BASE / "huggingface" / "datasets")
    os.environ["TORCH_HOME"] = str(CACHE_BASE / "torch")
```

### 更新后的脚本示例

```python
# deploy_switch_base8.py 开头
from cache_config import *  # 添加这一行

import torch
# ... 其余代码
```

---

## 五、清理旧缓存（迁移完成后）

### 确认迁移成功

```bash
# 检查新路径
dir D:\Dev\cache\huggingface

# 检查模型是否在新路径
python -c "from transformers import AutoTokenizer; t = AutoTokenizer.from_pretrained('google/switch-base-8'); print(t.cache_dir)"
```

### 删除旧缓存（谨慎操作）

```bash
# 备份后删除（建议先备份）
rmdir /S /Q C:\Users\wangyh43\.cache\huggingface
rmdir /S /Q C:\Users\wangyh43\.cache\torch
rmdir /S /Q C:\Users\wangyh43\.cache\pip
```

**或使用Python脚本**:
```python
import shutil
from pathlib import Path

old_cache = Path.home() / ".cache"
new_cache = Path("D:/Dev/cache")

# 确认新缓存存在且可用
if (new_cache / "huggingface").exists():
    # 删除旧缓存（谨慎！）
    for dir_name in ["huggingface", "torch", "pip"]:
        old_dir = old_cache / dir_name
        if old_dir.exists():
            print(f"删除: {old_dir}")
            shutil.rmtree(old_dir)
```

---

## 六、常见问题

### Q1: 模型仍然下载到C盘

**A**: 检查环境变量是否正确设置
```python
import os
print(os.environ.get("HF_HOME"))
```

确保在**导入transformers之前**设置环境变量。

### Q2: 迁移后找不到模型

**A**: 检查新路径是否存在模型文件
```python
from pathlib import Path
cache_dir = Path("D:/Dev/cache/huggingface/hub")
print(f"模型文件: {list(cache_dir.glob('**/*.bin'))[:5]}")
```

### Q3: 权限错误

**A**: 确保D盘有写入权限
```bash
# 测试写入
echo test > D:\Dev\cache\test.txt
del D:\Dev\cache\test.txt
```

### Q4: 环境变量不生效

**A**: 
1. 重启终端/IDE
2. 检查系统环境变量（方法3）
3. 在Python脚本中直接设置（方法2）

---

## 七、缓存大小估算

| 缓存类型 | 典型大小 | 说明 |
|---------|---------|------|
| switch-base-8模型 | ~5GB | 模型权重 |
| tokenizer | ~10MB | 词汇表 |
| datasets缓存 | ~100MB-1GB | 数据集缓存 |
| PyTorch | ~100MB | 预训练权重 |
| **总计** | **~5-6GB** | |

迁移后可以释放C盘约5-6GB空间。

---

## 八、自动化脚本

### 创建启动脚本

**`start_experiment.bat`**:
```batch
@echo off
REM 设置缓存路径
call D:\Dev\set_cache_env.bat

REM 激活conda环境
call conda activate moe_watermark

REM 运行实验
python deploy_switch_base8.py %*
```

**`start_experiment.ps1`**:
```powershell
# 设置缓存路径
. D:\Dev\set_cache_env.ps1

# 激活conda环境
conda activate moe_watermark

# 运行实验
python deploy_switch_base8.py $args
```

---

## 九、检查清单

迁移前：
- [ ] 确认D盘有足够空间（>10GB）
- [ ] 备份重要数据
- [ ] 关闭所有使用缓存的程序

迁移中：
- [ ] 运行迁移脚本
- [ ] 验证文件复制成功
- [ ] 设置环境变量

迁移后：
- [ ] 测试模型加载
- [ ] 验证缓存路径
- [ ] 删除旧缓存（可选）

---

## 十、快速参考

```python
# 快速设置（复制到脚本开头）
import os
from pathlib import Path
CACHE_BASE = Path("D:/Dev/cache")
os.environ["HF_HOME"] = str(CACHE_BASE / "huggingface")
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_BASE / "huggingface" / "hub")
os.environ["HF_DATASETS_CACHE"] = str(CACHE_BASE / "huggingface" / "datasets")
os.environ["TORCH_HOME"] = str(CACHE_BASE / "torch")
```

---

## 十一、故障排除

### 问题1: 迁移脚本失败

**解决方案**:
```bash
# 手动创建目录
mkdir D:\Dev\cache\huggingface
mkdir D:\Dev\cache\torch

# 手动复制（使用资源管理器）
# 从 C:\Users\wangyh43\.cache\huggingface 复制到 D:\Dev\cache\huggingface
```

### 问题2: 环境变量不持久

**解决方案**: 使用系统环境变量（方法3）或每次运行前执行脚本

### 问题3: 磁盘空间不足

**解决方案**: 
- 清理不需要的模型
- 使用 `transformers-cli` 删除缓存
```bash
transformers-cli cache-clear
```

---

## 十二、相关文件

- `setup_cache_path.py` - 迁移脚本
- `cache_config.py` - Python配置（自动生成）
- `set_cache_env.bat` - Windows批处理脚本（自动生成）
- `set_cache_env.ps1` - PowerShell脚本（自动生成）

