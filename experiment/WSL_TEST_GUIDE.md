# WSL 环境测试指南

> **注意**：本文档提供WSL环境的详细配置说明。快速开始请参考根目录的 [`README.md`](../README.md)。

本指南帮助你在 WSL (Windows Subsystem for Linux) 环境下测试本项目。

## 一、前置准备

### 1.1 确认 WSL 环境

```bash
# 检查 WSL 版本
wsl --version

# 检查当前发行版
wsl --list --verbose

# 进入 WSL
wsl
```

### 1.2 安装必要的系统依赖

```bash
# 更新包管理器
sudo apt update && sudo apt upgrade -y

# 安装 Python 3.10 和基础工具
sudo apt install -y python3.10 python3.10-venv python3-pip git curl

# 安装 CUDA 工具（如果使用 GPU）
# 注意：WSL 需要使用 Windows 的 CUDA 驱动
# 参考：https://docs.nvidia.com/cuda/wsl-user-guide/index.html
```

### 1.3 安装 Miniconda（推荐）

```bash
# 下载 Miniconda
cd /tmp
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 安装
bash Miniconda3-latest-Linux-x86_64.sh

# 按照提示完成安装，然后重新加载 shell
source ~/.bashrc

# 验证安装
conda --version
```

## 二、项目环境配置

### 2.1 克隆/进入项目目录

```bash
# 如果项目在 Windows 文件系统，路径通常是：
cd /mnt/c/path/to/project/emb_attack_separa

# 如果在 WSL 文件系统，直接：
cd ~/lab/github.com/raccoonrat/emb_attack_separa
```

### 2.2 创建 Conda 环境

```bash
cd experiment

# 使用 environment.yml 创建环境
conda env create -f environment.yml

# 激活环境
conda activate emb_attack_separa
```

### 2.3 安装依赖

```bash
# 如果使用 pip 虚拟环境
python3.10 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 2.4 配置缓存路径（WSL 适配）

WSL 环境下的缓存路径配置与 Windows 不同。创建或修改 `cache_config_wsl.py`：

```bash
# 在 experiment 目录下创建 WSL 专用配置
cat > cache_config_wsl.py << 'EOF'
"""
WSL 环境缓存路径配置
"""
import os
from pathlib import Path

# WSL 环境下的缓存路径（使用 Linux 路径）
# 可以放在用户主目录或 /tmp
CACHE_BASE = Path.home() / ".cache" / "emb_attack_separa"

# 创建缓存目录
CACHE_BASE.mkdir(parents=True, exist_ok=True)

# 设置 Hugging Face 缓存
os.environ["HF_HOME"] = str(CACHE_BASE / "huggingface")
os.environ["TRANSFORMERS_CACHE"] = str(CACHE_BASE / "huggingface" / "hub")
os.environ["HF_DATASETS_CACHE"] = str(CACHE_BASE / "huggingface" / "datasets")

# 设置 PyTorch 缓存
os.environ["TORCH_HOME"] = str(CACHE_BASE / "torch")

# 设置 pip 缓存
os.environ["PIP_CACHE_DIR"] = str(CACHE_BASE / "pip")

# 设置 Hugging Face 镜像源（加速下载）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

print("✓ WSL 缓存路径已设置")
print(f"  HF_HOME: {os.environ['HF_HOME']}")
print(f"  TORCH_HOME: {os.environ['TORCH_HOME']}")
print(f"  HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', '默认')}")
EOF
```

## 三、测试步骤

### 3.1 测试 Hugging Face 镜像源

```bash
# 修改 test_hf_mirror.py，使用 WSL 配置
python -c "
import sys
sys.path.insert(0, '.')
# 先导入 WSL 配置
import cache_config_wsl
# 然后运行测试
exec(open('test_hf_mirror.py').read())
"
```

或者直接运行（如果已修改导入）：

```bash
python test_hf_mirror.py
```

### 3.2 最小功能测试

```bash
# 运行最小测试（需要先修改 test_minimal_switch.py 使用 WSL 配置）
python test_minimal_switch.py
```

### 3.3 完整功能测试

#### 3.3.1 参数标定模式

```bash
python main.py --mode calibrate \
               --model_name "google/switch-base-8" \
               --dataset_name "wikitext" \
               --dataset_split "train" \
               --num_calib_samples 100 \
               --batch_size 2
```

#### 3.3.2 水印嵌入模式

```bash
python main.py --mode embed \
               --model_name "google/switch-base-8" \
               --prompt "Once upon a time, in a land far, far away," \
               --secret_key "my_secret_key_123"
```

#### 3.3.3 水印检测模式

```bash
python main.py --mode detect \
               --model_name "google/switch-base-8" \
               --text_to_check "Your watermarked text here" \
               --secret_key "my_secret_key_123" \
               --attack "none"
```

#### 3.3.4 实验模式

```bash
python main.py --mode experiment \
               --model_name "google/switch-base-8" \
               --dataset_name "wikitext" \
               --num_calib_samples 50 \
               --batch_size 2
```

## 四、WSL 特定配置

### 4.1 GPU 支持（如果使用 NVIDIA GPU）

WSL 2 支持 NVIDIA GPU，但需要：

1. **Windows 端安装 NVIDIA 驱动**（WSL 专用驱动）
   - 下载：https://www.nvidia.com/Download/index.aspx
   - 选择 "Windows Driver Type: Standard" → "WSL"

2. **WSL 端安装 CUDA Toolkit**
   ```bash
   # 添加 NVIDIA 仓库
   wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
   sudo dpkg -i cuda-keyring_1.1-1_all.deb
   sudo apt-get update
   sudo apt-get -y install cuda-toolkit-12-4
   
   # 验证安装
   nvidia-smi
   ```

3. **验证 PyTorch CUDA**
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
   ```

### 4.2 文件系统性能

**重要**：WSL 访问 Windows 文件系统（`/mnt/c/`）性能较差，建议：

1. **将项目放在 WSL 文件系统**（`~/` 或 `/home/`）
   ```bash
   # 从 Windows 文件系统复制到 WSL
   cp -r /mnt/c/path/to/project ~/projects/
   ```

2. **缓存目录放在 WSL 文件系统**
   - 使用 `cache_config_wsl.py` 中的配置（`~/.cache/`）

### 4.3 内存和显存限制

WSL 默认会限制内存使用，可以配置：

```bash
# 创建或编辑 WSL 配置文件（在 Windows 端）
# 路径：C:\Users\<YourUser>\.wslconfig

[wsl2]
memory=16GB          # 根据你的系统调整
processors=8         # CPU 核心数
swap=8GB             # 交换空间
```

重启 WSL：
```powershell
# 在 PowerShell 中
wsl --shutdown
wsl
```

## 五、常见问题排查

### Q1: `conda: command not found`

**A**: Conda 未正确初始化
```bash
# 初始化 conda
conda init bash
source ~/.bashrc
```

### Q2: CUDA 不可用

**A**: 检查以下几点：
1. Windows 端是否安装了 WSL 专用 NVIDIA 驱动
2. WSL 端是否安装了 CUDA Toolkit
3. PyTorch 版本是否支持 CUDA
   ```bash
   # 重新安装支持 CUDA 的 PyTorch
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

### Q3: 模型下载失败

**A**: 
1. 检查网络连接
2. 确认镜像源配置正确
   ```bash
   echo $HF_ENDPOINT  # 应该显示 https://hf-mirror.com
   ```
3. 手动测试镜像
   ```bash
   curl https://hf-mirror.com
   ```

### Q4: 显存不足

**A**: 
1. 使用 8-bit 量化
   ```bash
   python deploy_switch_base8.py --use_8bit
   ```
2. 减小 batch size 和序列长度
3. 使用 CPU 模式（很慢）
   ```bash
   # 在代码中设置
   device = "cpu"
   ```

### Q5: 文件路径错误

**A**: 确保使用 Linux 路径格式
- ❌ `D:/Dev/cache`（Windows 路径）
- ✅ `~/cache` 或 `/home/user/cache`（Linux 路径）

## 六、快速验证清单

运行以下命令验证环境：

```bash
# 1. Python 版本
python --version  # 应该是 3.10.x

# 2. Conda 环境
conda env list    # 应该看到 emb_attack_separa

# 3. 关键包
python -c "import torch; print(torch.__version__)"
python -c "import transformers; print(transformers.__version__)"

# 4. CUDA（如果使用 GPU）
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 5. 缓存配置
python -c "import cache_config_wsl; import os; print(os.environ['HF_HOME'])"
```

## 七、下一步

环境配置完成后，可以：

1. **运行完整测试套件**
   ```bash
   python test_minimal_switch.py
   ```

2. **开始实验**
   ```bash
   python main.py --mode experiment
   ```

3. **阅读详细文档**
   - `README.md` - 项目概述
   - `INSTALL.md` - 详细安装指南
   - `DEPLOYMENT_GUIDE_GTX4050.md` - 部署指南

---

**提示**：WSL 环境与原生 Linux 环境基本一致，但需要注意文件系统路径和 GPU 驱动的特殊配置。

