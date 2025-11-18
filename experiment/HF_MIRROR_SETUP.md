# Hugging Face 镜像源配置指南

## 一、已配置的镜像源

所有脚本已自动配置使用Hugging Face官方镜像：**https://hf-mirror.com**

## 二、验证镜像配置

### 方法1: 检查环境变量

```python
import os
print("HF_ENDPOINT:", os.environ.get("HF_ENDPOINT", "未设置"))
```

应该显示：`HF_ENDPOINT: https://hf-mirror.com`

### 方法2: 测试下载

```python
from transformers import AutoTokenizer

# 这会使用镜像源下载
tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")
print("下载成功！")
```

## 三、手动设置镜像源

### 方法1: 环境变量（推荐）

**Windows批处理**:
```batch
set HF_ENDPOINT=https://hf-mirror.com
```

**PowerShell**:
```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

**系统环境变量（永久）**:
1. 按 `Win + R`，输入 `sysdm.cpl`
2. 点击"高级" → "环境变量"
3. 添加用户变量：
   - 变量名: `HF_ENDPOINT`
   - 变量值: `https://hf-mirror.com`

### 方法2: 在Python脚本中设置

```python
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 然后导入transformers
from transformers import AutoTokenizer
```

### 方法3: 使用配置文件

```python
from cache_config import *  # 已包含镜像配置
```

## 四、其他镜像源（可选）

如果官方镜像不可用，可以尝试：

### 1. 阿里云镜像（需要配置）

```python
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 官方镜像
```

### 2. 使用huggingface-cli配置

```bash
# 安装huggingface-cli
pip install huggingface_hub

# 配置镜像
huggingface-cli download --repo-id google/switch-base-8
```

## 五、验证下载速度

### 测试脚本

```python
import time
from transformers import AutoTokenizer

start = time.time()
print("开始下载tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")
elapsed = time.time() - start
print(f"下载完成，耗时: {elapsed:.2f}秒")
```

### 检查下载来源

下载时，transformers会显示实际使用的URL。如果看到 `hf-mirror.com`，说明镜像配置成功。

## 六、常见问题

### Q1: 仍然从原始地址下载

**A**: 确保在导入transformers之前设置环境变量：
```python
# ✅ 正确顺序
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoTokenizer

# ❌ 错误顺序
from transformers import AutoTokenizer
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 太晚了
```

### Q2: 镜像源不可用

**A**: 
1. 检查网络连接
2. 尝试直接访问: https://hf-mirror.com
3. 如果不可用，移除 `HF_ENDPOINT` 环境变量，使用原始地址

### Q3: 下载速度仍然很慢

**A**: 
1. 检查网络连接
2. 尝试使用VPN
3. 考虑使用本地已下载的模型

## 七、使用示例

### 示例1: 下载模型（自动使用镜像）

```python
# deploy_switch_base8.py 已自动配置
from transformers import AutoModelForSeq2SeqLM

# 自动使用镜像下载
model = AutoModelForSeq2SeqLM.from_pretrained("google/switch-base-8")
```

### 示例2: 下载数据集

```python
from datasets import load_dataset

# 自动使用镜像下载
dataset = load_dataset("wikitext", "wikitext-103-v1")
```

## 八、镜像源说明

### 官方镜像: https://hf-mirror.com

- **优点**: 官方维护，稳定可靠
- **速度**: 通常比原始地址快
- **覆盖**: 支持所有Hugging Face资源

### 使用方式

镜像会自动将以下URL转换：
- `https://huggingface.co/google/switch-base-8` 
- → `https://hf-mirror.com/google/switch-base-8`

## 九、检查清单

- [x] 所有脚本已添加 `HF_ENDPOINT` 配置
- [x] `cache_config.py` 已包含镜像配置
- [x] 环境变量脚本已更新
- [ ] 测试下载速度（可选）
- [ ] 验证模型加载（可选）

## 十、快速参考

```python
# 快速设置镜像（复制到脚本开头）
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 或使用配置文件
from cache_config import *
```

---

**注意**: 所有主要脚本已自动配置镜像源，无需额外操作即可使用！

