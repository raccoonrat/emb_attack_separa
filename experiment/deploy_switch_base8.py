"""
GTX 4050部署脚本 - switch-base-8

针对6GB显存的优化部署
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
import gc
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from mves_config import MVESConfig, get_default_config, ModelConfig, WatermarkConfig, ExperimentConfig
from mves_watermark_corrected import patch_switch_model_with_watermark, get_watermark_data_from_switch_model
from moe_watermark_enhanced import create_watermark_wrapper
from detector import LLRDetector


def setup_gpu():
    """GPU设置和优化"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA不可用，需要GPU")
    
    # 清空显存缓存
    torch.cuda.empty_cache()
    gc.collect()
    
    # 设置显存分配策略（避免碎片）
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    
    device_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"GPU: {device_name}")
    print(f"总显存: {total_memory:.2f} GB")
    print(f"CUDA版本: {torch.version.cuda}")
    
    return torch.device("cuda")


def load_model_optimized(model_name="google/switch-base-8", use_8bit=False, max_memory_gb=5.0):
    """
    优化加载模型（针对GTX 4050）
    
    Args:
        model_name: 模型名称
        use_8bit: 是否使用8-bit量化
        max_memory_gb: 最大显存使用（GB）
    """
    print(f"\n{'='*60}")
    print(f"加载模型: {model_name}")
    print(f"{'='*60}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if use_8bit:
        print("使用8-bit量化加载...")
        try:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            print("✓ 8-bit量化加载成功")
        except ImportError:
            print("警告: bitsandbytes未安装，回退到FP16")
            use_8bit = False
    
    if not use_8bit:
        print("使用FP16加载...")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            max_memory={0: f"{max_memory_gb}GB"} if max_memory_gb > 0 else None
        )
        print("✓ FP16加载成功")
    
    model.eval()
    
    # 显存使用情况
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"显存使用: {allocated:.2f} GB (已分配) / {reserved:.2f} GB (已保留)")
    
    return model, tokenizer


def verify_switch_architecture(model):
    """验证switch-base-8架构"""
    print("\n验证模型架构...")
    
    # 检查基本结构
    if not hasattr(model, 'decoder'):
        print("✗ 模型没有decoder属性")
        return False
    
    if not hasattr(model.decoder, 'block'):
        print("✗ decoder没有block属性")
        return False
    
    # 检查MoE层
    moe_layers = []
    for layer_idx, layer in enumerate(model.decoder.block):
        if hasattr(layer, 'layer') and len(layer.layer) > 1:
            ffn_layer = layer.layer[1]
            if hasattr(ffn_layer, 'mlp') and hasattr(ffn_layer.mlp, 'router'):
                moe_layers.append((layer_idx, ffn_layer.mlp.router))
    
    if len(moe_layers) == 0:
        print("✗ 未找到MoE层")
        return False
    
    print(f"✓ 找到 {len(moe_layers)} 个MoE层")
    for layer_idx, router in moe_layers[:3]:  # 只显示前3个
        print(f"  - Layer {layer_idx}: router类型 = {type(router).__name__}")
    
    return True


def create_optimized_config():
    """创建针对GTX 4050的优化配置"""
    config = MVESConfig(
        model=ModelConfig(
            model_name="google/switch-base-8",
            model_type="switch",
            device="cuda",
            torch_dtype="float16",
        ),
        watermark=WatermarkConfig(
            secret_key="GTX4050_DEPLOYMENT_KEY",
            epsilon=0.01,
            c_star=2.0,
            gamma_design=0.03,
            num_experts=8,  # switch-base-8默认
            k_top=1,  # switch-base-8默认
        ),
        experiment=ExperimentConfig(
            num_samples=50,  # 减少样本数
            batch_size=1,  # 小batch size
            max_length=256,  # 限制序列长度
        )
    )
    return config


def deploy_watermark_system(use_hook=True, use_8bit=False):
    """
    部署完整的水印系统
    
    Args:
        use_hook: 是否使用Hook机制（推荐）
        use_8bit: 是否使用8-bit量化
    """
    # 1. GPU设置
    device = setup_gpu()
    
    # 2. 加载模型
    model, tokenizer = load_model_optimized(use_8bit=use_8bit, max_memory_gb=5.0)
    
    # 3. 验证架构
    if not verify_switch_architecture(model):
        raise RuntimeError("模型架构验证失败，请确认使用的是switch-base-8")
    
    # 4. 创建配置
    config = create_optimized_config()
    config.validate()
    
    # 5. 部署水印
    print(f"\n{'='*60}")
    print("部署水印系统")
    print(f"{'='*60}")
    
    if use_hook:
        print("使用Hook机制（推荐）...")
        try:
            wrapper = create_watermark_wrapper(model, config, use_hook=True)
            wrapper.register_hooks()
            print("✓ Hook已注册")
            watermark_method = "hook"
        except Exception as e:
            print(f"Hook注册失败: {e}")
            print("回退到Patch方式...")
            use_hook = False
    
    if not use_hook:
        print("使用Patch方式...")
        model = patch_switch_model_with_watermark(model, config)
        print("✓ Patch完成")
        watermark_method = "patch"
    
    # 6. 显存检查
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"\n显存使用情况:")
        print(f"  已分配: {allocated:.2f} GB")
        print(f"  已保留: {reserved:.2f} GB")
        print(f"  总显存: {total:.2f} GB")
        print(f"  剩余: {total - reserved:.2f} GB")
        
        if reserved > 5.5:
            print("⚠ 警告: 显存使用接近上限，建议启用8-bit量化")
    
    return model, tokenizer, config, watermark_method


def test_generation(model, tokenizer, prompt="The quick brown fox jumps over the lazy dog."):
    """测试生成功能"""
    print(f"\n{'='*60}")
    print("测试生成功能")
    print(f"{'='*60}")
    print(f"提示: '{prompt}'")
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            use_cache=True  # 使用KV cache节省显存
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"生成文本: {generated_text}")
    
    return generated_text


def test_detection(model, tokenizer, text, config):
    """测试检测功能"""
    print(f"\n{'='*60}")
    print("测试检测功能")
    print(f"{'='*60}")
    
    try:
        # 获取检测数据
        detection_data = get_watermark_data_from_switch_model(model)
        
        if not detection_data:
            print("⚠ 警告: 未获取到检测数据，可能需要重新运行模型")
            return None
        
        print(f"✓ 获取到 {len(detection_data)} 个检测数据点")
        
        # 简化检测（计算LLR）
        p_0, p_1, S_indices = detection_data[-1]  # 使用最后一个
        
        # 计算LLR
        p_0_clamped = torch.clamp(p_0, min=1e-9)
        p_1_clamped = torch.clamp(p_1, min=1e-9)
        
        # 只计算激活专家的LLR
        batch_size, seq_len, k_top = S_indices.shape
        llr_total = 0.0
        
        for b in range(batch_size):
            for s in range(seq_len):
                for k in range(k_top):
                    expert_idx = S_indices[b, s, k].item()
                    p0_val = p_0_clamped[b, s, expert_idx]
                    p1_val = p_1_clamped[b, s, expert_idx]
                    llr_total += torch.log(p1_val / p0_val).item()
        
        print(f"LLR统计量: {llr_total:.4f}")
        print(f"检测阈值: {config.detection.tau_alpha:.4f}")
        
        is_detected = llr_total > config.detection.tau_alpha
        print(f"检测结果: {'✓ 检测到水印' if is_detected else '✗ 未检测到水印'}")
        
        return {
            'llr_score': llr_total,
            'is_detected': is_detected,
            'threshold': config.detection.tau_alpha
        }
        
    except Exception as e:
        print(f"检测失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="GTX 4050部署脚本 - switch-base-8")
    parser.add_argument("--use_8bit", action="store_true", help="使用8-bit量化")
    parser.add_argument("--use_patch", action="store_true", help="使用Patch方式（而非Hook）")
    parser.add_argument("--test_only", action="store_true", help="仅测试，不部署水印")
    
    args = parser.parse_args()
    
    try:
        # 部署系统
        model, tokenizer, config, watermark_method = deploy_watermark_system(
            use_hook=not args.use_patch,
            use_8bit=args.use_8bit
        )
        
        if not args.test_only:
            # 测试生成
            generated_text = test_generation(model, tokenizer)
            
            # 测试检测
            detection_result = test_detection(model, tokenizer, generated_text, config)
        
        print(f"\n{'='*60}")
        print("✓ 部署成功！")
        print(f"{'='*60}")
        print(f"水印方法: {watermark_method}")
        print(f"模型: {config.model.model_name}")
        print(f"水印强度 ε: {config.watermark.epsilon:.6f}")
        print(f"安全系数 c*: {config.watermark.c_star:.2f}")
        
        # 保存配置
        config.save("deployment_config.json")
        print(f"\n配置已保存到: deployment_config.json")
        
    except torch.cuda.OutOfMemoryError:
        print(f"\n{'='*60}")
        print("✗ 显存不足！")
        print(f"{'='*60}")
        print("解决方案:")
        print("1. 启用8-bit量化: python deploy_switch_base8.py --use_8bit")
        print("2. 减小batch_size和max_length")
        print("3. 关闭其他占用显存的程序")
        
    except Exception as e:
        print(f"\n✗ 部署失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

