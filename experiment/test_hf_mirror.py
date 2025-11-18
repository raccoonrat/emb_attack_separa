"""
测试Hugging Face镜像源配置

验证镜像是否正常工作
"""

import os
import time
from pathlib import Path

# 设置缓存和镜像（在导入transformers之前）
CACHE_BASE = Path("D:/Dev/cache")
os.environ.setdefault("HF_HOME", str(CACHE_BASE / "huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_BASE / "huggingface" / "hub"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

print("="*60)
print("Hugging Face镜像源测试")
print("="*60)

# 检查环境变量
print("\n1. 检查环境变量:")
print(f"   HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', '未设置')}")
print(f"   HF_HOME: {os.environ.get('HF_HOME', '未设置')}")
print(f"   TRANSFORMERS_CACHE: {os.environ.get('TRANSFORMERS_CACHE', '未设置')}")

# 测试下载tokenizer
print("\n2. 测试下载tokenizer...")
try:
    from transformers import AutoTokenizer
    
    start = time.time()
    print("   开始下载 google/switch-base-8 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")
    elapsed = time.time() - start
    
    print(f"   [OK] 下载成功！")
    print(f"   耗时: {elapsed:.2f}秒")
    print(f"   词汇表大小: {len(tokenizer)}")
    
    # 检查缓存位置
    if hasattr(tokenizer, 'cache_dir'):
        print(f"   缓存位置: {tokenizer.cache_dir}")
    
except Exception as e:
    print(f"   [ERROR] 下载失败: {e}")
    import traceback
    traceback.print_exc()

# 测试模型信息（不下载完整模型）
print("\n3. 测试获取模型信息...")
try:
    from transformers import AutoConfig
    
    start = time.time()
    print("   获取 google/switch-base-8 配置...")
    config = AutoConfig.from_pretrained("google/switch-base-8")
    elapsed = time.time() - start
    
    print(f"   [OK] 获取成功！")
    print(f"   耗时: {elapsed:.2f}秒")
    print(f"   模型类型: {config.model_type}")
    if hasattr(config, 'num_experts'):
        print(f"   专家数: {config.num_experts}")
    
except Exception as e:
    print(f"   [ERROR] 获取失败: {e}")

# 检查缓存目录
print("\n4. 检查缓存目录:")
cache_dir = Path(os.environ.get("HF_HOME", ""))
if cache_dir.exists():
    print(f"   缓存目录存在: {cache_dir}")
    
    # 检查hub目录
    hub_dir = cache_dir / "hub"
    if hub_dir.exists():
        models = list(hub_dir.glob("models--*"))
        print(f"   已缓存模型数: {len(models)}")
        if models:
            print(f"   示例: {models[0].name}")
else:
    print(f"   缓存目录不存在: {cache_dir}")

print("\n" + "="*60)
print("[OK] 测试完成！")
print("="*60)
print("\n如果下载成功，说明镜像配置正常。")
print("如果下载失败，请检查网络连接和镜像源配置。")

