@echo off
REM 修复conda初始化问题
REM 运行此脚本以初始化conda

echo ========================================
echo Conda初始化修复工具
echo ========================================
echo.

set "CONDA_PATH=C:\Users\wangyh43\AppData\Local\miniconda3"

REM 检查conda是否存在
if not exist "%CONDA_PATH%\Scripts\conda.exe" (
    echo [ERROR] 未找到conda，请检查路径: %CONDA_PATH%
    pause
    exit /b 1
)

echo [1/3] 初始化conda for PowerShell...
call "%CONDA_PATH%\Scripts\conda.exe" init powershell

echo.
echo [2/3] 初始化conda for CMD...
call "%CONDA_PATH%\Scripts\conda.exe" init cmd.exe

echo.
echo ========================================
echo [OK] 初始化完成！
echo ========================================
echo.
echo 下一步操作:
echo 1. 关闭当前终端
echo 2. 重新打开终端
echo 3. 运行: conda activate moe_watermark
echo.
echo 或者使用提供的激活脚本:
echo   .\experiment\activate_env.bat
echo.
pause

