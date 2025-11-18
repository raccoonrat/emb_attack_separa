# 在 Cursor 中使用 Miniconda 指南

## 第一步：找到 Miniconda 安装路径

如果 `conda` 命令不在 PATH 中，需要先找到 Miniconda 的安装位置。常见位置：

1. **用户目录下**（最常见）：
   - `C:\Users\你的用户名\miniconda3`
   - `C:\Users\你的用户名\anaconda3`

2. **使用 Anaconda Prompt 查找**：
   - 打开"Anaconda Prompt"（开始菜单中搜索）
   - 运行：`where conda` 或 `conda info --base`

3. **手动查找**：
   - 在文件资源管理器中搜索 `conda.exe`
   - 通常位于 `miniconda3\Scripts\conda.exe`

## 第二步：配置 Cursor 使用 Conda 环境

### 方法 1：使用 Cursor 的 Python 解释器选择器（推荐）

1. 按 `Ctrl+Shift+P` 打开命令面板
2. 输入并选择：`Python: Select Interpreter`
3. 如果看到 conda 环境，直接选择
4. 如果没有，点击"Enter interpreter path..."，然后输入：
   ```
   C:\Users\你的用户名\miniconda3\envs\emb_attack_separa\python.exe
   ```
   或者基础环境：
   ```
   C:\Users\你的用户名\miniconda3\python.exe
   ```

### 方法 2：修改 .vscode/settings.json

1. 打开项目根目录的 `.vscode/settings.json` 文件
2. 找到 `python.defaultInterpreterPath` 配置
3. 取消注释并修改为你的 conda 环境路径：
   ```json
   "python.defaultInterpreterPath": "C:\\Users\\你的用户名\\miniconda3\\envs\\emb_attack_separa\\python.exe"
   ```
   注意：Windows 路径中的反斜杠需要转义（使用 `\\`）

### 方法 3：使用环境变量（推荐用于多用户）

在 `.vscode/settings.json` 中使用环境变量：
```json
"python.defaultInterpreterPath": "${env:USERPROFILE}\\miniconda3\\envs\\emb_attack_separa\\python.exe"
```

## 第三步：在 Cursor 终端中使用 Conda

### 选项 A：使用 Anaconda Prompt

1. 在 Cursor 中按 `` Ctrl+` `` 打开终端
2. 点击终端右上角的下拉菜单（`+` 旁边的 `v`）
3. 选择"选择默认配置文件"
4. 如果配置了 PowerShell with Conda，选择它

### 选项 B：手动初始化 Conda

在 Cursor 终端中运行（替换为你的实际路径）：
```powershell
# 初始化 conda（只需运行一次）
& 'C:\Users\你的用户名\miniconda3\Scripts\conda.exe' shell.powershell hook | Out-String | Invoke-Expression

# 然后激活环境
conda activate emb_attack_separa
```

### 选项 C：配置终端自动初始化

`.vscode/settings.json` 中已配置了 "PowerShell with Conda" 终端配置文件。要使用它：

1. 打开终端（`` Ctrl+` ``）
2. 点击终端右上角的 `+` 旁边的下拉菜单
3. 选择 "PowerShell with Conda"

## 第四步：创建 Conda 环境（如果还没有）

如果还没有创建项目环境，在终端中运行：

```powershell
# 进入 experiment 目录
cd experiment

# 使用 environment.yml 创建环境
conda env create -f environment.yml

# 激活环境
conda activate emb_attack_separa
```

## 第五步：验证配置

1. **检查 Python 解释器**：
   - 在 Cursor 底部状态栏查看 Python 版本
   - 应该显示 conda 环境的路径

2. **在终端中验证**：
   ```powershell
   python --version
   conda info --envs  # 查看所有环境，当前激活的会标有 *
   ```

3. **测试导入包**：
   ```powershell
   python -c "import torch; print('PyTorch:', torch.__version__)"
   ```

## 在 Jupyter Notebook 中使用 Conda 环境

1. 打开 `experiment/sep.ipynb`
2. 点击右上角的 "Select Kernel"
3. 选择 "Python Environments"
4. 选择 `emb_attack_separa` conda 环境

或者直接在 notebook 中运行：
```python
import sys
sys.executable  # 查看当前使用的 Python 路径
```

## 常见问题

### 问题 1：找不到 conda 命令

**解决方案**：
- 使用 Anaconda Prompt（开始菜单中搜索）
- 或者手动添加 conda 到 PATH（见 `INSTALL.md`）

### 问题 2：Cursor 找不到 Python 解释器

**解决方案**：
1. 确认 conda 环境已创建：`conda env list`
2. 检查路径是否正确：`conda info --envs` 会显示所有环境的路径
3. 在 Cursor 中使用绝对路径而不是相对路径

### 问题 3：终端中 conda 命令不可用

**解决方案**：
- 使用配置好的 "PowerShell with Conda" 终端配置
- 或者每次手动初始化（见第三步选项 B）

### 问题 4：Jupyter Notebook 使用了错误的 kernel

**解决方案**：
1. 点击 notebook 右上角的 kernel 选择器
2. 选择正确的 conda 环境
3. 或者在 notebook 中运行：
   ```python
   !conda info --envs
   !which python
   ```

## 快速检查清单

- [ ] 找到 Miniconda 安装路径
- [ ] 创建了 `emb_attack_separa` conda 环境
- [ ] 在 Cursor 中选择了正确的 Python 解释器
- [ ] 终端中可以运行 `conda` 命令
- [ ] 可以成功导入项目依赖包（torch, transformers 等）
- [ ] Jupyter Notebook 使用了正确的 kernel

