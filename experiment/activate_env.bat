@echo off
REM 激活conda环境并设置缓存路径

REM 初始化conda
call "C:\Users\wangyh43\AppData\Local\miniconda3\Scripts\activate.bat"

REM 激活环境
call conda activate moe_watermark

REM 设置缓存路径
set HF_HOME=D:\Dev\cache\huggingface
set TRANSFORMERS_CACHE=D:\Dev\cache\huggingface\hub
set HF_DATASETS_CACHE=D:\Dev\cache\huggingface\datasets
set TORCH_HOME=D:\Dev\cache\torch
set HF_ENDPOINT=https://hf-mirror.com

echo.
echo [OK] Conda环境已激活: moe_watermark
echo [OK] 缓存路径已设置到D盘
echo [OK] Hugging Face镜像已配置
echo.
