# Cursor中使用Miniconda3 - 快速开始

## ✅ 已检测到的配置

- **Miniconda3路径**: `C:\Users\wangyh43\AppData\Local\miniconda3`
- **环境名称**: `moe_watermark`
- **激活脚本**: 已创建

## ⚠️ 重要：首次使用必须先初始化conda

**如果遇到 `CondaError: Run 'conda init' before 'conda activate'` 错误**，请先运行：

```powershell
# 方法1: 使用修复脚本（推荐）
.\experiment\fix_conda_init.ps1

# 方法2: 手动初始化
C:\Users\wangyh43\AppData\Local\miniconda3\Scripts\conda.exe init powershell
C:\Users\wangyh43\AppData\Local\miniconda3\Scripts\conda.exe init cmd.exe
```

**初始化后，必须关闭并重新打开终端才能生效！**

---

## 🚀 快速配置（3步）

### 步骤0: 初始化conda（首次使用必须）

**在Anaconda Prompt中运行**（或使用修复脚本）：

```bash
# 初始化conda
conda init powershell
conda init cmd.exe

# 关闭并重新打开终端
```

### 步骤1: 配置Cursor终端（自动激活环境）

1. 按 `Ctrl+Shift+P` 打开命令面板
2. 输入 `Preferences: Open User Settings (JSON)`
3. 添加以下配置：

```json
{
    "terminal.integrated.defaultProfile.windows": "PowerShell (Conda)",
    "terminal.integrated.profiles.windows": {
        "PowerShell (Conda)": {
            "path": "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe",
            "args": [
                "-NoExit",
                "-Command",
                "& 'C:\\Users\\wangyh43\\AppData\\Local\\miniconda3\\Scripts\\Activate.ps1'; conda activate moe_watermark"
            ],
            "icon": "terminal-powershell"
        }
    }
}
```

4. 保存文件（`Ctrl+S`）
5. 重启Cursor

### 步骤2: 创建conda环境（如果还没有）

**在Anaconda Prompt中运行**:

```bash
# 创建环境
conda create -n moe_watermark python=3.10 -y

# 激活环境
conda activate moe_watermark

# 安装依赖
cd D:\Dev\cursor\github.com\emb_attack_separa\experiment
pip install -r requirements.txt
```

### 步骤3: 验证配置

在Cursor中打开新终端（`Ctrl+Shift+` `），应该看到：

```
(moe_watermark) PS D:\Dev\cursor\github.com\emb_attack_separa>
```

运行验证：

```bash
python --version
# 应该显示: Python 3.10.x

conda env list
# 应该显示 moe_watermark 环境，前面有 *
```

## 📝 使用方法

### 方法1: 自动激活（推荐）

配置完成后，每次打开Cursor终端都会自动激活环境。

### 方法2: 使用激活脚本

如果自动激活不工作，可以手动运行：

```bash
# 在experiment目录下
.\activate_env.bat    # Windows批处理
.\activate_env.ps1    # PowerShell
```

### 方法3: 手动激活

```bash
conda activate moe_watermark
```

## 🔧 如果conda不在PATH中

### 添加到PATH

1. 按 `Win + R`，输入 `sysdm.cpl`
2. 点击"高级" → "环境变量"
3. 在"用户变量"的PATH中添加：
   - `C:\Users\wangyh43\AppData\Local\miniconda3`
   - `C:\Users\wangyh43\AppData\Local\miniconda3\Scripts`
   - `C:\Users\wangyh43\AppData\Local\miniconda3\Library\bin`
4. 重启Cursor

### 或使用完整路径

在Cursor设置中使用完整路径：

```json
{
    "terminal.integrated.profiles.windows": {
        "PowerShell (Conda)": {
            "path": "C:\\Users\\wangyh43\\AppData\\Local\\miniconda3\\Scripts\\activate.bat",
            "args": ["moe_watermark"],
            "icon": "terminal-powershell"
        }
    }
}
```

## ✅ 验证清单

- [ ] Cursor设置已配置
- [ ] conda环境已创建
- [ ] 终端自动激活环境
- [ ] Python版本正确（3.10）
- [ ] 依赖已安装

## 📚 详细文档

查看 `CURSOR_CONDA_SETUP.md` 获取完整配置说明。

## 🐛 常见问题

### Q1: CondaError: Run 'conda init' before 'conda activate'

**这是最常见的问题！**

**解决方案**:

```powershell
# 方法1: 使用修复脚本（推荐）
.\experiment\fix_conda_init.ps1

# 方法2: 手动初始化
C:\Users\wangyh43\AppData\Local\miniconda3\Scripts\conda.exe init powershell
C:\Users\wangyh43\AppData\Local\miniconda3\Scripts\conda.exe init cmd.exe

# 重要：初始化后必须关闭并重新打开终端！
```

### Q2: PowerShell执行策略错误

**错误**: `无法加载文件，因为在此系统上禁止运行脚本`

**解决方案**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Q3: 环境激活失败

**解决方案**:
```bash
# 确保已初始化conda
conda init powershell
conda init cmd.exe
# 重启终端
```

### Q4: 找不到conda命令

**解决方案**: 添加到PATH（见上方说明）

---

**准备好了吗？开始使用！**

```bash
# 1. 验证环境
conda activate moe_watermark
python --version

# 2. 运行实验
python experiment/deploy_switch_base8.py
```

