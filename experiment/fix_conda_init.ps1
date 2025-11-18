# 修复conda初始化问题
# 运行此脚本以初始化conda在PowerShell中

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Conda初始化修复工具" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$condaPath = "C:\Users\wangyh43\AppData\Local\miniconda3"

# 检查conda是否存在
if (-not (Test-Path "$condaPath\Scripts\conda.exe")) {
    Write-Host "[ERROR] 未找到conda，请检查路径: $condaPath" -ForegroundColor Red
    exit 1
}

Write-Host "[1/3] 初始化conda for PowerShell..." -ForegroundColor Yellow
& "$condaPath\Scripts\conda.exe" init powershell

Write-Host ""
Write-Host "[2/3] 初始化conda for CMD..." -ForegroundColor Yellow
& "$condaPath\Scripts\conda.exe" init cmd.exe

Write-Host ""
Write-Host "[3/3] 检查PowerShell执行策略..." -ForegroundColor Yellow
$executionPolicy = Get-ExecutionPolicy -Scope CurrentUser
if ($executionPolicy -eq "Restricted") {
    Write-Host "  当前执行策略: $executionPolicy" -ForegroundColor Yellow
    Write-Host "  需要修改执行策略..." -ForegroundColor Yellow
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
    Write-Host "  [OK] 执行策略已修改为: RemoteSigned" -ForegroundColor Green
} else {
    Write-Host "  [OK] 执行策略正常: $executionPolicy" -ForegroundColor Green
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "[OK] 初始化完成！" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "下一步操作:" -ForegroundColor Yellow
Write-Host "1. 关闭当前终端" -ForegroundColor White
Write-Host "2. 重新打开终端（Ctrl+Shift+`）" -ForegroundColor White
Write-Host "3. 运行: conda activate moe_watermark" -ForegroundColor White
Write-Host ""
Write-Host "或者使用提供的激活脚本:" -ForegroundColor Yellow
Write-Host "  .\experiment\activate_env.bat" -ForegroundColor White
Write-Host "  .\experiment\activate_env.ps1" -ForegroundColor White
Write-Host ""

