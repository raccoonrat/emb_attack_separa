# 激活conda环境并设置缓存路径

# 初始化conda
& "C:\Users\wangyh43\AppData\Local\miniconda3\Scripts\Activate.ps1"

# 激活环境
conda activate moe_watermark

# 设置缓存路径
$env:HF_HOME = "D:\Dev\cache\huggingface"
$env:TRANSFORMERS_CACHE = "D:\Dev\cache\huggingface\hub"
$env:HF_DATASETS_CACHE = "D:\Dev\cache\huggingface\datasets"
$env:TORCH_HOME = "D:\Dev\cache\torch"
$env:HF_ENDPOINT = "https://hf-mirror.com"

Write-Host ""
Write-Host "[OK] Conda环境已激活: moe_watermark" -ForegroundColor Green
Write-Host "[OK] 缓存路径已设置到D盘" -ForegroundColor Green
Write-Host "[OK] Hugging Face镜像已配置" -ForegroundColor Green
Write-Host ""
