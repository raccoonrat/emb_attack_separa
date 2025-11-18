"""
最小测试脚本 - switch-base-8

快速验证部署是否成功
"""

# 设置缓存路径到D盘（在导入transformers之前）
import os
from pathlib import Path
CACHE_BASE = Path("D:/Dev/cache")
os.environ.setdefault("HF_HOME", str(CACHE_BASE / "huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_BASE / "huggingface" / "hub"))
os.environ.setdefault("HF_DATASETS_CACHE", str(CACHE_BASE / "huggingface" / "datasets"))
os.environ.setdefault("TORCH_HOME", str(CACHE_BASE / "torch"))

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from mves_config import get_default_config
from mves_watermark_corrected import patch_switch_model_with_watermark

def main():
    print("="*60)
    print("最小测试 - switch-base-8")
    print("="*60)
    
    # 检查GPU
    if not torch.cuda.is_available():
        print("✗ CUDA不可用")
        return
    
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✓ 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 1. 加载模型（FP16）
    print("\n1. 加载模型...")
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/switch-base-8",
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            max_memory={0: "5GB"}
        )
        tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")
        print("✓ 模型加载成功")
        print(f"  显存使用: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return
    
    # 2. 部署水印
    print("\n2. 部署水印...")
    try:
        config = get_default_config()
        config.model.model_name = "google/switch-base-8"
        config.model.torch_dtype = "float16"
        
        patched_model = patch_switch_model_with_watermark(model, config)
        print("✓ 水印部署成功")
    except Exception as e:
        print(f"✗ 水印部署失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. 测试生成
    print("\n3. 测试生成...")
    try:
        prompt = "The quick brown fox"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64).to(patched_model.device)
        
        with torch.no_grad():
            outputs = patched_model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✓ 生成成功")
        print(f"  输入: {prompt}")
        print(f"  输出: {result}")
    except Exception as e:
        print(f"✗ 生成失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. 显存检查
    print("\n4. 显存检查...")
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"  已分配: {allocated:.2f} GB")
    print(f"  已保留: {reserved:.2f} GB")
    print(f"  总显存: {total:.2f} GB")
    print(f"  剩余: {total - reserved:.2f} GB")
    
    if reserved < 5.5:
        print("✓ 显存使用正常")
    else:
        print("⚠ 警告: 显存使用较高，建议启用8-bit量化")
    
    print("\n" + "="*60)
    print("✓ 所有测试通过！")
    print("="*60)

if __name__ == "__main__":
    main()

