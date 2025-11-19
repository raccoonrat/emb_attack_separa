"""
OKR使用示例

展示如何使用OKR方法为MoE模型注入水印
"""

import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

from okr_patch import inject_okr, inject_okr_with_config
from okr_detector import OKRDetector
from mves_config import MVESConfig


def example_basic_usage():
    """
    基本使用示例
    """
    # 1. 加载模型
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
    
    # 2. 注入OKR水印
    model = inject_okr(
        model,
        epsilon=1.5,  # 质量容忍阈值
        secret_key="my_secret_key"
    )
    
    # 3. 使用模型进行推理
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    outputs = model.generate(input_ids, max_length=50)
    
    # 4. 检测水印
    detector = OKRDetector(model, epsilon=1.5)
    score, verdict = detector.detect(input_ids)
    print(f"水印检测结果: {verdict}, 得分: {score:.4f}")


def example_with_config():
    """
    使用MVESConfig的示例
    """
    # 1. 创建配置
    config = MVESConfig()
    config.watermark.epsilon = 1.5
    config.watermark.secret_key = "MVES_SECRET_KEY"
    
    # 2. 加载模型
    model = AutoModelForSeq2SeqLM.from_pretrained("google/switch-base-8")
    
    # 3. 注入OKR水印（使用配置）
    model = inject_okr_with_config(model, config)
    
    # 4. 使用模型
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    outputs = model.generate(input_ids, max_length=50)
    
    # 5. 检测水印
    detector = OKRDetector(model, epsilon=config.watermark.epsilon)
    score, verdict = detector.detect(input_ids)
    print(f"水印检测结果: {verdict}, 得分: {score:.4f}")


if __name__ == "__main__":
    print("OKR使用示例")
    print("=" * 50)
    
    # 注意：实际运行时需要根据你的模型和硬件调整
    # example_basic_usage()
    # example_with_config()
    
    print("请根据你的具体需求调用相应的示例函数")

