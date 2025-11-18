# Cursor中使用Miniconda3配置

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
