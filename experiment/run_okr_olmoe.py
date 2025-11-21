"""
AllenAI OLMoE 实验启动脚本

Hardware Requirement: GTX 4050 (6GB) or better
Model: allenai/OLMoE-1B-7B-0924-Instruct
Quantization: 4-bit (bitsandbytes)
"""

# 设置缓存路径和 HuggingFace 镜像（必须在导入 transformers 之前）
import os
import platform
from pathlib import Path

def detect_and_setup_cache():
    """自动检测环境并设置缓存路径和镜像"""
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
    
    # 设置 HuggingFace 镜像（必须在导入 transformers 之前）
    if "HF_ENDPOINT" not in os.environ:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print(f"[INFO] 设置 HuggingFace 镜像: {os.environ['HF_ENDPOINT']}")
    else:
        print(f"[INFO] 使用已设置的 HuggingFace 端点: {os.environ['HF_ENDPOINT']}")

# 执行缓存配置（必须在导入任何 transformers 相关模块之前）
detect_and_setup_cache()

from okr_config import OKRConfig
from okr_experiment import run_okr_experiment

def main():
    # 针对 GTX 4050 的优化配置
    config = OKRConfig()
    
    # 1. 模型配置 (使用 OLMoE)
    config.model.model_name = "allenai/OLMoE-1B-7B-0924-Instruct"
    config.model.device = "auto"
    config.model.torch_dtype = "float16" 
    config.model.trust_remote_code = True
    
    # 2. 水印配置
    config.watermark.secret_key = "LinusTorvalds_Secret_Key_2025"
    config.watermark.watermark_alpha = 0.1  # OLMoE 比较健壮，可以给一点强度
    config.watermark.epsilon = 1.5          # 兼容参数
    config.watermark.threshold_ratio = 0.9  # 仅在 P_top2 / P_top1 > 0.9 时介入
    
    # 3. 实验参数
    config.experiment.num_samples = 5
    config.experiment.max_length = 128
    config.experiment.experiment_name = "OLMoE_4bit_Test"
    
    # 4. 自定义 Prompts (Open-ended generation)
    config.experiment.custom_prompts = [
        "Explain the concept of modularity in Linux kernel design.",
        "Write a short story about a robot who loves gardening.",
        "Why is Rust considered safer than C++ for system programming?",
        "The quick brown fox jumps over the lazy dog.",
        "Explain quantum entanglement to a 5-year old."
    ]

    print(">>> Starting OLMoE Experiment on GTX 4050...")
    results = run_okr_experiment(config, experiment_type="basic")
    
    print("\n>>> Results Summary:")
    print(f"Avg Hit Rate: {results['summary']['average_hit_rate']:.4f}")

if __name__ == "__main__":
    main()