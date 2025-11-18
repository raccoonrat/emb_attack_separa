"""
检查switch-base-8部署环境

验证所有配置是否正确统一到switch-base-8
"""

import sys
import os

def check_model_config():
    """检查模型配置是否统一"""
    print("检查模型配置...")
    
    from mves_config import get_default_config
    
    config = get_default_config()
    
    if config.model.model_name != "google/switch-base-8":
        print(f"✗ 模型名称不统一: {config.model.model_name}")
        print("  应该使用: google/switch-base-8")
        return False
    
    print(f"✓ 模型名称: {config.model.model_name}")
    print(f"✓ 模型类型: {config.model.model_type}")
    print(f"✓ 专家数: {config.watermark.num_experts}")
    print(f"✓ Top-k: {config.watermark.k_top}")
    
    return True

def check_imports():
    """检查导入是否正确"""
    print("\n检查导入...")
    
    try:
        from mves_watermark_corrected import patch_switch_model_with_watermark
        print("✓ mves_watermark_corrected 导入成功")
    except ImportError as e:
        print(f"✗ mves_watermark_corrected 导入失败: {e}")
        return False
    
    try:
        from moe_watermark_enhanced import create_watermark_wrapper
        print("✓ moe_watermark_enhanced 导入成功")
    except ImportError as e:
        print(f"✗ moe_watermark_enhanced 导入失败: {e}")
        return False
    
    return True

def check_gpu():
    """检查GPU"""
    print("\n检查GPU...")
    
    try:
        import torch
        if not torch.cuda.is_available():
            print("✗ CUDA不可用")
            return False
        
        device_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"✓ GPU: {device_name}")
        print(f"✓ 显存: {total_memory:.2f} GB")
        
        if total_memory < 6:
            print("⚠ 警告: 显存 < 6GB，建议使用8-bit量化")
        
        return True
    except ImportError:
        print("✗ PyTorch未安装")
        return False

def check_model_availability():
    """检查模型是否可用"""
    print("\n检查模型可用性...")
    
    try:
        from transformers import AutoTokenizer
        
        print("尝试加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")
        print("✓ Tokenizer加载成功")
        print(f"  词汇表大小: {len(tokenizer)}")
        
        return True
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        print("  请检查网络连接和Hugging Face访问")
        return False

def main():
    print("="*60)
    print("Switch-base-8 部署环境检查")
    print("="*60)
    
    all_ok = True
    
    # 检查1: 模型配置
    if not check_model_config():
        all_ok = False
    
    # 检查2: 导入
    if not check_imports():
        all_ok = False
    
    # 检查3: GPU
    if not check_gpu():
        all_ok = False
    
    # 检查4: 模型可用性
    if not check_model_availability():
        all_ok = False
    
    print("\n" + "="*60)
    if all_ok:
        print("✓ 所有检查通过！可以开始部署")
        print("\n下一步:")
        print("  python test_minimal_switch.py  # 最小测试")
        print("  python deploy_switch_base8.py  # 完整部署")
    else:
        print("✗ 部分检查失败，请修复后重试")
    print("="*60)

if __name__ == "__main__":
    main()

