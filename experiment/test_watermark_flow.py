#!/usr/bin/env python3
"""
测试水印嵌入和检测的完整流程

演示正确的使用方法：
1. 嵌入水印并生成文本
2. 使用生成的文本进行检测
3. 确保使用相同的 secret_key
"""

import sys
import os
from pathlib import Path

# 设置缓存路径（在导入transformers之前）
import platform

def detect_and_setup_cache():
    """自动检测环境并设置缓存路径"""
    release = platform.release().lower()
    is_wsl = ("microsoft" in release or "wsl" in release)
    if not is_wsl and os.path.exists("/proc/version"):
        try:
            with open("/proc/version", "r") as f:
                proc_version = f.read().lower()
                is_wsl = "microsoft" in proc_version
        except (IOError, PermissionError):
            pass
    is_linux = platform.system() == "Linux"
    
    if os.environ.get("USE_WSL_CONFIG") == "1" or (is_wsl or is_linux):
        try:
            import cache_config_wsl
            return
        except ImportError:
            CACHE_BASE = Path.home() / ".cache" / "emb_attack_separa"
            CACHE_BASE.mkdir(parents=True, exist_ok=True)
    else:
        try:
            import cache_config
            return
        except ImportError:
            CACHE_BASE = Path("D:/Dev/cache")
    
    os.environ.setdefault("HF_HOME", str(CACHE_BASE / "huggingface"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_BASE / "huggingface" / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(CACHE_BASE / "huggingface" / "datasets"))
    os.environ.setdefault("TORCH_HOME", str(CACHE_BASE / "torch"))
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

detect_and_setup_cache()

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from mves_config import get_default_config
from mves_watermark_corrected import patch_switch_model_with_watermark
from detector import LLRDetector

def test_watermark_flow():
    """测试完整的水印流程"""
    print("="*60)
    print("水印嵌入和检测测试")
    print("="*60)
    
    # 配置
    model_name = "google/switch-base-8"
    prompt = "The quick brown fox"
    secret_key = "test_secret_key_123"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n1. 加载模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    
    # 配置水印
    config = get_default_config()
    config.watermark.secret_key = secret_key
    config.watermark.epsilon = 0.12  # c*² * γ = 2.0² * 0.03
    config.model.model_name = model_name
    
    print(f"\n2. Patch 模型（嵌入水印）")
    print(f"   secret_key: {secret_key}")
    print(f"   epsilon: {config.watermark.epsilon:.4f}")
    patched_model = patch_switch_model_with_watermark(model, config)
    
    print(f"\n3. 生成带水印的文本")
    print(f"   提示: '{prompt}'")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = patched_model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            top_k=50
        )
    
    watermarked_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"   生成的文本: '{watermarked_text}'")
    
    print(f"\n4. 检测水印")
    print(f"   检测文本: '{watermarked_text}'")
    print(f"   使用相同的 secret_key: {secret_key}")
    
    # 创建检测器（使用相同的配置）
    detector = LLRDetector(patched_model, tokenizer, tau_alpha=20.0)
    
    # 检测
    is_detected, llr_score = detector.detect(watermarked_text)
    
    print(f"\n5. 检测结果")
    print(f"   LLR 分数: {llr_score:.4f}")
    print(f"   阈值: {detector.tau_alpha:.4f}")
    if is_detected:
        print(f"   ✓ 检测到水印！")
    else:
        print(f"   ✗ 未检测到水印")
    
    print(f"\n6. 测试：检测不带水印的文本")
    normal_text = "This is a normal text without watermark."
    is_detected_normal, llr_score_normal = detector.detect(normal_text)
    print(f"   文本: '{normal_text}'")
    print(f"   LLR 分数: {llr_score_normal:.4f}")
    if is_detected_normal:
        print(f"   ✗ 误报（不应该检测到水印）")
    else:
        print(f"   ✓ 正确（未检测到水印）")
    
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)

if __name__ == "__main__":
    test_watermark_flow()

