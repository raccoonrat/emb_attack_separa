"""
OKR 使用示例

展示如何使用 OKR 方法为 MoE 模型注入水印和运行实验
"""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from okr_patch import inject_okr
from okr_detector import OKRDetector
from okr_config import OKRConfig, get_default_okr_config, get_quick_test_okr_config
from okr_experiment import OKRBasicExperiment, run_okr_experiment
from utils.logger import setup_logging, get_logger

# 设置日志
logger = get_logger(__name__)


def example_basic_usage():
    """
    基本使用示例：手动注入和检测水印
    """
    logger.info("=" * 60)
    logger.info("示例1: 基本使用")
    logger.info("=" * 60)
    
    # 1. 加载模型
    logger.info("加载模型...")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/switch-base-8")
    tokenizer = AutoTokenizer.from_pretrained("google/switch-base-8")
    
    # 2. 注入 OKR 水印
    logger.info("注入 OKR 水印...")
    model = inject_okr(
        model,
        epsilon=1.5,  # 质量容忍阈值
        secret_key="my_secret_key"
    )
    
    # 3. 使用模型进行推理
    logger.info("生成文本...")
    text = "The quick brown fox jumps over the lazy dog."
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=100, num_beams=1, do_sample=False)
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"生成文本: {generated_text[:100]}...")
    
    # 4. 检测水印
    logger.info("检测水印...")
    detector = OKRDetector(model, epsilon=1.5)
    score, verdict = detector.detect(inputs["input_ids"])
    logger.info(f"水印检测结果: {verdict}, 得分: {score:.4f}")


def example_with_config():
    """
    使用 OKRConfig 的示例
    """
    logger.info("=" * 60)
    logger.info("示例2: 使用配置")
    logger.info("=" * 60)
    
    # 1. 创建配置
    config = get_default_okr_config()
    config.watermark.epsilon = 1.5
    config.watermark.secret_key = "OKR_SECRET_KEY"
    config.model.model_name = "google/switch-base-8"
    
    # 2. 加载模型
    logger.info(f"加载模型: {config.model.model_name}")
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
    
    # 3. 注入 OKR 水印（使用配置）
    logger.info("注入 OKR 水印...")
    model = inject_okr(
        model,
        epsilon=config.watermark.epsilon,
        secret_key=config.watermark.secret_key
    )
    
    # 4. 使用模型
    logger.info("生成文本...")
    text = "In a world where technology advances rapidly, artificial intelligence plays a crucial role."
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=100, num_beams=1, do_sample=False)
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    logger.info(f"生成文本: {generated_text[:100]}...")
    
    # 5. 检测水印
    logger.info("检测水印...")
    detector = OKRDetector(model, epsilon=config.watermark.epsilon)
    score, verdict = detector.detect(inputs["input_ids"])
    logger.info(f"水印检测结果: {verdict}, 得分: {score:.4f}")


def example_run_experiment():
    """
    运行完整实验的示例
    """
    logger.info("=" * 60)
    logger.info("示例3: 运行完整实验")
    logger.info("=" * 60)
    
    # 使用快速测试配置（小样本，适合快速验证）
    config = get_quick_test_okr_config()
    config.watermark.epsilon = 1.5
    config.watermark.secret_key = "EXPERIMENT_SECRET_KEY"
    
    # 运行基础实验
    logger.info("运行基础实验...")
    results = run_okr_experiment(config, experiment_type="basic")
    
    logger.info("实验完成！")
    logger.info(f"总样本数: {results['summary']['total_samples']}")
    logger.info(f"检测为水印: {results['summary']['watermarked_samples']}")
    logger.info(f"平均命中率: {results['summary']['average_hit_rate']:.4f}")


def example_custom_experiment():
    """
    自定义实验的示例
    """
    logger.info("=" * 60)
    logger.info("示例4: 自定义实验")
    logger.info("=" * 60)
    
    # 创建自定义配置
    config = OKRConfig()
    config.experiment.experiment_name = "Custom_OKR_Test"
    config.experiment.num_samples = 5
    config.experiment.batch_size = 2
    config.watermark.epsilon = 2.0  # 更大的 epsilon，更强的水印
    config.watermark.secret_key = "CUSTOM_SECRET_KEY"
    
    # 自定义提示词
    config.experiment.custom_prompts = [
        "Climate change is one of the most pressing issues facing humanity today.",
        "Artificial intelligence will transform the way we work and live.",
        "The future of transportation lies in sustainable energy solutions."
    ]
    
    # 创建实验对象
    experiment = OKRBasicExperiment(config)
    
    # 运行实验
    results = experiment.run()
    
    logger.info("自定义实验完成！")
    logger.info(f"实验结果已保存到: {config.experiment.output_dir}")


if __name__ == "__main__":
    # 设置日志
    setup_logging()
    
    print("OKR 使用示例")
    print("=" * 60)
    print("可用的示例函数:")
    print("  1. example_basic_usage() - 基本使用")
    print("  2. example_with_config() - 使用配置")
    print("  3. example_run_experiment() - 运行完整实验")
    print("  4. example_custom_experiment() - 自定义实验")
    print("=" * 60)
    print()
    
    # 注意：实际运行时需要根据你的模型和硬件调整
    # 取消注释下面的行来运行示例
    
    # 默认运行快速实验
    print("\n开始运行快速实验...")
    try:
        example_run_experiment()
    except Exception as e:
        print(f"\n实验运行出错: {e}")
        print("请检查:")
        print("  1. 是否已安装所有依赖 (transformers, torch, tqdm)")
        print("  2. 是否有足够的 GPU 内存或使用 CPU")
        print("  3. 网络连接是否正常（下载模型）")
        import traceback
        traceback.print_exc()

