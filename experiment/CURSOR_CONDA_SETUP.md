# Cursor中使用Miniconda3配置

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

## 方法1: 使用终端配置文件（推荐）

### PowerShell配置

在Cursor中按 `Ctrl+Shift+P`，输入 "Preferences: Open User Settings (JSON)"

添加以下配置：

```json
{
    "terminal.integrated.defaultProfile.windows": "PowerShell",
    "terminal.integrated.profiles.windows": {
        "PowerShell (Conda)": {
            "path": "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe",
            "args": [
                "-NoExit",
                "-Command",
                "& 'C:\Users\wangyh43\AppData\Local\miniconda3\\Scripts\\Activate.ps1'; conda activate moe_watermark"
            ],
            "icon": "terminal-powershell"
        }
    }
}
```

### CMD配置

```json
{
    "terminal.integrated.defaultProfile.windows": "Command Prompt (Conda)",
    "terminal.integrated.profiles.windows": {
        "Command Prompt (Conda)": {
            "path": "C:\\Windows\\System32\\cmd.exe",
            "args": [
                "/K",
                "C:\Users\wangyh43\AppData\Local\miniconda3\\Scripts\\activate.bat moe_watermark"
            ]
        }
    }
}
```

## 方法2: 手动激活

每次打开终端后运行：

```bash
# PowerShell
& "C:\Users\wangyh43\AppData\Local\miniconda3\Scripts\Activate.ps1"
conda activate moe_watermark

# CMD
call "C:\Users\wangyh43\AppData\Local\miniconda3\Scripts\activate.bat"
conda activate moe_watermark
```

## 方法3: 使用提供的脚本

```bash
# 在experiment目录下
.\activate_env.bat    # Windows批处理
.\activate_env.ps1   # PowerShell
```

## 验证配置

```bash
# 检查conda
conda --version

# 检查Python
python --version

# 检查环境
conda env list

# 检查当前环境
conda info --envs
```

## 常见问题

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
