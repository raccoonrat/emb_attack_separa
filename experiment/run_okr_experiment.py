"""
OKR 实验运行脚本

直接运行此脚本开始 OKR 实验
"""

# 设置缓存路径（在导入transformers之前）
import os
import platform
from pathlib import Path

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

# 执行缓存配置
detect_and_setup_cache()

import torch
import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from okr_experiment import run_okr_experiment
from okr_config import get_quick_test_okr_config, get_default_okr_config
from utils.logger import setup_logging, get_logger

logger = get_logger(__name__)


def main():
    """主函数"""
    print("=" * 60)
    print("OKR 实验开始")
    print("=" * 60)
    
    # 检查 CUDA 可用性
    if torch.cuda.is_available():
        logger.info(f"CUDA 可用，设备数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            logger.info(f"    内存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    else:
        logger.warning("CUDA 不可用，将使用 CPU（速度较慢）")
    
    # 创建配置
    # 使用快速测试配置（小样本，适合快速验证）
    config = get_quick_test_okr_config()
    config.watermark.epsilon = 1.5
    config.watermark.secret_key = "OKR_EXPERIMENT_KEY_2024"
    config.model.model_name = "google/switch-base-8"
    
    # 设置 GPU 内存限制（如果可用）
    if torch.cuda.is_available():
        config.model.max_memory = {0: "5GB"}
        config.model.torch_dtype = "float16"  # 使用 FP16 节省内存
    
    # 设置日志
    log_file = Path(config.experiment.output_dir) / "experiment.log"
    setup_logging(log_file=log_file, level=20)  # INFO level
    
    logger.info("=" * 60)
    logger.info("实验配置")
    logger.info("=" * 60)
    logger.info(f"模型: {config.model.model_name}")
    logger.info(f"Epsilon: {config.watermark.epsilon}")
    logger.info(f"样本数: {config.experiment.num_samples}")
    logger.info(f"批次大小: {config.experiment.batch_size}")
    logger.info(f"输出目录: {config.experiment.output_dir}")
    logger.info("=" * 60)
    
    try:
        # 运行基础实验
        logger.info("开始运行基础实验...")
        results = run_okr_experiment(config, experiment_type="basic")
        
        # 打印结果摘要
        print("\n" + "=" * 60)
        print("实验完成！")
        print("=" * 60)
        print(f"总样本数: {results['summary']['total_samples']}")
        print(f"检测为水印: {results['summary']['watermarked_samples']}")
        print(f"平均命中率: {results['summary']['average_hit_rate']:.4f}")
        print(f"检测阈值: {results['summary']['detection_threshold']}")
        print("=" * 60)
        print(f"\n详细结果已保存到: {config.experiment.output_dir}")
        print(f"日志文件: {log_file}")
        
    except KeyboardInterrupt:
        logger.warning("实验被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"实验失败: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

