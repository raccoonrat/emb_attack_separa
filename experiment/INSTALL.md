# 环境安装指南

## 安装 Miniconda

### 方法1：使用 winget（推荐，Windows 10/11）

在 PowerShell 中运行：

```powershell
winget install Anaconda.Miniconda3
```

### 方法2：手动下载安装

1. 访问 Miniconda 官网：https://docs.conda.io/en/latest/miniconda.html
2. 下载 Windows 64-bit 安装程序
3. 运行安装程序，按照提示完成安装
4. **重要**：安装时勾选 "Add Miniconda3 to my PATH environment variable" 选项

### 方法3：使用 Chocolatey（如果已安装）

```powershell
choco install miniconda3
```

## 安装后配置

安装完成后，**重新打开终端**（或重启 PowerShell），然后验证安装：

```powershell
conda --version
```

如果显示版本号，说明安装成功。

## 创建项目环境

### 使用 environment.yml（推荐）

```powershell
cd experiment
conda env create -f environment.yml
```

### 手动创建环境

```powershell
conda create -n emb_attack_separa python=3.10
conda activate emb_attack_separa
pip install -r requirements.txt
```

## 激活环境

每次使用项目时，需要先激活环境：

```powershell
conda activate emb_attack_separa
```

## 验证安装

激活环境后，验证关键包是否安装成功：

```powershell
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
```

## 常见问题

### conda 命令未找到

如果安装后仍无法使用 conda 命令：

1. 检查是否在安装时勾选了 "Add to PATH" 选项
2. 手动添加 conda 到 PATH：
   - 找到 Miniconda 安装目录（通常在 `C:\Users\你的用户名\miniconda3`）
   - 将以下路径添加到系统环境变量 PATH：
     - `C:\Users\你的用户名\miniconda3`
     - `C:\Users\你的用户名\miniconda3\Scripts`
     - `C:\Users\你的用户名\miniconda3\Library\bin`
3. 重新打开终端

### 使用 Anaconda Prompt

如果 PATH 配置有问题，可以直接使用 Anaconda Prompt（安装 Miniconda 时会自动创建），它已经配置好了 conda 环境。

## 在 Cursor 中使用 Conda

详细指南请参考：[CURSOR_CONDA_SETUP.md](./CURSOR_CONDA_SETUP.md)

简要步骤：

1. **找到 Miniconda 路径**：
   - 常见位置：`C:\Users\你的用户名\miniconda3`
   - 或在 Anaconda Prompt 中运行：`conda info --base`

2. **在 Cursor 中选择 Python 解释器**：
   - 按 `Ctrl+Shift+P` → 输入 `Python: Select Interpreter`
   - 选择 conda 环境：`C:\Users\你的用户名\miniconda3\envs\emb_attack_separa\python.exe`

3. **配置终端使用 Conda**：
   - 打开终端（`` Ctrl+` ``）
   - 运行：`conda activate emb_attack_separa`

