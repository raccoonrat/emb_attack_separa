# 修复 "CondaError: Run 'conda init' before 'conda activate'"

## 🔍 问题原因

这个错误表示 conda 还没有在 PowerShell 中初始化。conda 需要在每个 shell 类型中单独初始化。

## ✅ 解决方案

### 方法1: 使用修复脚本（推荐）

```powershell
# PowerShell
.\experiment\fix_conda_init.ps1

# 或批处理
.\experiment\fix_conda_init.bat
```

### 方法2: 手动初始化

```powershell
# 初始化PowerShell
C:\Users\wangyh43\AppData\Local\miniconda3\Scripts\conda.exe init powershell

# 初始化CMD
C:\Users\wangyh43\AppData\Local\miniconda3\Scripts\conda.exe init cmd.exe
```

### 方法3: 在Anaconda Prompt中初始化

1. 打开"Anaconda Prompt"（从开始菜单）
2. 运行：
   ```bash
   conda init powershell
   conda init cmd.exe
   ```

## ⚠️ 重要：初始化后必须重启终端

**conda init 会修改 PowerShell 配置文件，必须关闭并重新打开终端才能生效！**

### 验证初始化是否成功

重新打开终端后，运行：

```powershell
conda --version
# 应该显示版本号，而不是错误

conda activate moe_watermark
# 应该能成功激活环境
```

## 🔧 如果仍然失败

### 检查PowerShell配置文件

conda init 会修改 `$PROFILE`，检查文件是否存在：

```powershell
# 查看配置文件路径
$PROFILE

# 检查文件是否存在
Test-Path $PROFILE

# 如果不存在，创建它
New-Item -Path $PROFILE -Type File -Force
```

### 手动加载conda

如果自动加载失败，可以手动加载：

```powershell
# 加载conda初始化脚本
& "C:\Users\wangyh43\AppData\Local\miniconda3\Scripts\Activate.ps1"

# 然后激活环境
conda activate moe_watermark
```

### 使用提供的激活脚本

```bash
# 这些脚本会自动处理初始化
.\experiment\activate_env.bat
.\experiment\activate_env.ps1
```

## 📝 完整工作流

1. **初始化conda**（只需一次）:
   ```powershell
   C:\Users\wangyh43\AppData\Local\miniconda3\Scripts\conda.exe init powershell
   ```

2. **关闭当前终端**

3. **重新打开终端**（`Ctrl+Shift+` `）

4. **验证**:
   ```powershell
   conda --version
   conda activate moe_watermark
   ```

5. **如果成功，应该看到**:
   ```
   (moe_watermark) PS D:\Dev\cursor\github.com\emb_attack_separa>
   ```

## 🎯 快速修复命令

**复制粘贴到终端运行**:

```powershell
# 一键修复
C:\Users\wangyh43\AppData\Local\miniconda3\Scripts\conda.exe init powershell; C:\Users\wangyh43\AppData\Local\miniconda3\Scripts\conda.exe init cmd.exe; Write-Host "`n[OK] 初始化完成！请关闭并重新打开终端。`n" -ForegroundColor Green
```

运行后，**关闭并重新打开终端**，然后运行 `conda activate moe_watermark`。

