# PowerShell script to create OKR branch
# 创建OKR独立分支的PowerShell脚本

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "创建 OKR 独立分支" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# 检查当前分支
$currentBranch = git branch --show-current
Write-Host "当前分支: $currentBranch" -ForegroundColor Yellow

# 检查是否有未提交的更改
$status = git status --porcelain
if ($status) {
    Write-Host ""
    Write-Host "检测到未提交的更改:" -ForegroundColor Yellow
    git status --short
    Write-Host ""
    $choice = Read-Host "请选择操作: (1) 提交更改 (2) 暂存更改 (3) 放弃更改 [默认: 2]"
    
    if ($choice -eq "1") {
        Write-Host "暂存所有更改..." -ForegroundColor Green
        git add -A
        $commitMsg = Read-Host "请输入提交信息 [默认: WIP: OKR相关更改]"
        if ([string]::IsNullOrWhiteSpace($commitMsg)) {
            $commitMsg = "WIP: OKR相关更改"
        }
        git commit -m $commitMsg
    }
    elseif ($choice -eq "3") {
        Write-Host "放弃所有更改..." -ForegroundColor Yellow
        git restore .
        git clean -fd
    }
    else {
        Write-Host "暂存更改..." -ForegroundColor Green
        git stash push -m "临时暂存: 创建OKR分支前"
    }
}

# 创建新分支
$branchName = "okr"
Write-Host ""
Write-Host "创建新分支: $branchName" -ForegroundColor Green

# 检查分支是否已存在
$existingBranch = git branch --list $branchName
if ($existingBranch) {
    Write-Host "警告: 分支 $branchName 已存在" -ForegroundColor Yellow
    $switch = Read-Host "是否切换到该分支? (y/n) [默认: y]"
    if ($switch -ne "n") {
        git checkout $branchName
    }
    else {
        Write-Host "操作取消" -ForegroundColor Red
        exit
    }
}
else {
    # 创建并切换到新分支
    git checkout -b $branchName
    if ($LASTEXITCODE -ne 0) {
        Write-Host "错误: 无法创建分支" -ForegroundColor Red
        exit 1
    }
}

# 确认当前分支
$newBranch = git branch --show-current
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "分支创建完成！" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "当前分支: $newBranch" -ForegroundColor Green
Write-Host ""
Write-Host "OKR相关文件:" -ForegroundColor Cyan
Write-Host "  - experiment/okr_*.py"
Write-Host "  - experiment/OKR_*.md"
Write-Host "  - experiment/run_okr_*.py"
Write-Host "  - experiment/test_okr_*.py"
Write-Host "  - okr/"
Write-Host "  - OKR_Colab_Experiment.ipynb"
Write-Host ""
Write-Host "可以使用以下命令查看分支:" -ForegroundColor Yellow
Write-Host "  git branch -a"
Write-Host ""
Write-Host "如果需要推送到远程:" -ForegroundColor Yellow
Write-Host "  git push -u origin $branchName"

