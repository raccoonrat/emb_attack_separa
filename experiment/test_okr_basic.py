"""
OKR 基础功能测试

快速验证 OKR 代码是否能正常工作
"""

# 设置缓存路径
import os
import platform
from pathlib import Path

def setup_cache():
    release = platform.release().lower()
    is_wsl = ("microsoft" in release or "wsl" in release)
    if not is_wsl and os.path.exists("/proc/version"):
        try:
            with open("/proc/version", "r") as f:
                is_wsl = "microsoft" in f.read().lower()
        except:
            pass
    is_linux = platform.system() == "Linux"
    
    if os.environ.get("USE_WSL_CONFIG") == "1" or (is_wsl or is_linux):
        try:
            import cache_config_wsl
            return
        except ImportError:
            CACHE_BASE = Path.home() / ".cache" / "emb_attack_separa"
    else:
        try:
            import cache_config
            return
        except ImportError:
            CACHE_BASE = Path("D:/Dev/cache")
    
    CACHE_BASE.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(CACHE_BASE / "huggingface"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_BASE / "huggingface" / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(CACHE_BASE / "huggingface" / "datasets"))
    os.environ.setdefault("TORCH_HOME", str(CACHE_BASE / "torch"))
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

setup_cache()

import sys
import torch

print("=" * 60)
print("OKR 基础功能测试")
print("=" * 60)

# 1. 测试导入
print("\n1. 测试模块导入...")
try:
    from okr_config import OKRConfig, get_default_okr_config
    from okr_kernel import OKRRouter
    from okr_patch import inject_okr, _initialize_secret_projection
    from okr_detector import OKRDetector
    print("✓ 所有模块导入成功")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)

# 2. 测试配置
print("\n2. 测试配置创建...")
try:
    config = get_default_okr_config()
    config.validate()
    print("✓ 配置创建和验证成功")
    print(f"  模型: {config.model.model_name}")
    print(f"  Epsilon: {config.watermark.epsilon}")
except Exception as e:
    print(f"✗ 配置失败: {e}")
    sys.exit(1)

# 3. 测试 OKRRouter
print("\n3. 测试 OKRRouter 核心逻辑...")
try:
    router = OKRRouter(input_dim=512, num_experts=8, top_k=1, epsilon=1.5)
    # 初始化 secret_projection（必须初始化才能使用）
    _initialize_secret_projection(router, "TEST_SECRET_KEY", input_dim=512, num_experts=8, device=None)
    # 创建测试输入
    test_input = torch.randn(2, 10, 512)  # [batch, seq, dim]
    routing_weights, selected_experts = router(test_input)
    print("✓ OKRRouter 前向传播成功")
    print(f"  输入形状: {test_input.shape}")
    print(f"  路由权重形状: {routing_weights.shape}")
    print(f"  选中专家形状: {selected_experts.shape}")
except Exception as e:
    print(f"✗ OKRRouter 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. 测试设备
print("\n4. 检查计算设备...")
if torch.cuda.is_available():
    print(f"✓ CUDA 可用")
    print(f"  设备数量: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("⚠ CUDA 不可用，将使用 CPU")

print("\n" + "=" * 60)
print("基础功能测试完成！")
print("=" * 60)
print("\n如果所有测试通过，可以运行完整实验：")
print("  python run_okr_experiment.py")

