"""
OKR (Opportunistic Keyed Routing) 实验框架

独立于 MVES 实验，专门用于 OKR 算法验证
参考现有代码结构，但保持完全独立
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict

# OKR 核心模块（独立实现）
from okr_patch import inject_okr
from okr_detector import OKRDetector
from okr_config import OKRConfig, get_default_okr_config

# 使用项目的日志系统
from utils.logger import setup_logging, get_logger
from utils.exceptions import WatermarkError, DetectionError

# 设置日志
logger = get_logger(__name__)


class OKRExperimentFramework:
    """
    OKR 实验框架基类
    
    提供通用的模型加载、数据准备等功能
    保持与现有实验框架类似的接口，但不耦合
    """
    
    def __init__(self, config: OKRConfig):
        """
        初始化实验框架
        
        Args:
            config: OKR 配置对象
        """
        self.config = config
        self.device = self._setup_device()
        self.tokenizer = None
        self.model = None
        
        logger.info(f"初始化 OKR 实验框架: {config.experiment.experiment_name}")
        logger.info(f"设备: {self.device}")
        logger.info(f"模型: {config.model.model_name}")
        logger.info(f"Epsilon: {config.watermark.epsilon}")
    
    def _setup_device(self) -> torch.device:
        """设置计算设备"""
        if self.config.model.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.model.device)
        return device
    
    def load_tokenizer(self) -> AutoTokenizer:
        """加载分词器"""
        if self.tokenizer is None:
            logger.info(f"加载分词器: {self.config.model.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model.model_name,
                trust_remote_code=self.config.model.trust_remote_code
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer
    
    def load_model(self) -> AutoModelForSeq2SeqLM:
        """加载模型"""
        if self.model is None:
            logger.info(f"加载模型: {self.config.model.model_name}")
            
            # 确定数据类型
            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16
            }
            torch_dtype = dtype_map.get(self.config.model.torch_dtype, torch.float32)
            
            # 加载模型
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.config.model.model_name,
                torch_dtype=torch_dtype,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True,
                max_memory=self.config.model.max_memory,
                trust_remote_code=self.config.model.trust_remote_code
            )
            self.model.eval()
            logger.info("模型加载完成")
        
        return self.model
    
    def inject_watermark(self, model: Optional[AutoModelForSeq2SeqLM] = None) -> AutoModelForSeq2SeqLM:
        """
        注入 OKR 水印
        
        Args:
            model: 模型（如果为 None，使用 self.model）
            
        Returns:
            已注入水印的模型
        """
        if model is None:
            model = self.load_model()
        
        logger.info("开始注入 OKR 水印...")
        logger.info(f"Secret Key: {self.config.watermark.secret_key[:20]}...")
        logger.info(f"Epsilon: {self.config.watermark.epsilon}")
        
        watermarked_model = inject_okr(
            model,
            epsilon=self.config.watermark.epsilon,
            secret_key=self.config.watermark.secret_key
        )
        
        logger.info("OKR 水印注入完成")
        return watermarked_model
    
    def prepare_data(self) -> List[str]:
        """
        准备实验数据
        
        Returns:
            文本列表
        """
        texts = []
        
        if self.config.experiment.custom_prompts:
            texts = self.config.experiment.custom_prompts[:self.config.experiment.num_samples]
            logger.info(f"使用自定义提示词: {len(texts)} 条")
        elif self.config.experiment.dataset_name:
            # TODO: 实现数据集加载
            logger.warning("数据集加载功能待实现，使用默认提示词")
            texts = self.config.experiment.custom_prompts[:self.config.experiment.num_samples]
        else:
            texts = self.config.experiment.custom_prompts[:self.config.experiment.num_samples]
            logger.info(f"使用默认提示词: {len(texts)} 条")
        
        return texts
    
    def save_results(self, results: Dict[str, Any], filename: str = "results.json"):
        """保存实验结果"""
        output_path = Path(self.config.experiment.output_dir) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"实验结果已保存到: {output_path}")


class OKRBasicExperiment(OKRExperimentFramework):
    """
    基础 OKR 实验
    
    验证水印注入和检测的基本功能
    """
    
    def run(self) -> Dict[str, Any]:
        """
        运行基础实验
        
        Returns:
            实验结果字典
        """
        logger.info("=" * 60)
        logger.info("OKR 基础实验开始")
        logger.info("=" * 60)
        
        # 1. 加载模型和分词器
        tokenizer = self.load_tokenizer()
        model = self.load_model()
        
        # 2. 注入水印
        watermarked_model = self.inject_watermark(model)
        
        # 3. 准备数据
        texts = self.prepare_data()
        
        # 4. 生成带水印的文本
        logger.info("开始生成带水印的文本...")
        watermarked_texts = []
        for text in tqdm(texts, desc="生成文本"):
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, 
                            max_length=self.config.experiment.max_length).to(self.device)
            
            with torch.no_grad():
                outputs = watermarked_model.generate(
                    **inputs,
                    max_length=self.config.experiment.max_length,
                    num_beams=1,
                    do_sample=False
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            watermarked_texts.append(generated_text)
        
        # 5. 检测水印
        logger.info("开始检测水印...")
        detector = OKRDetector(watermarked_model, epsilon=self.config.watermark.epsilon)
        
        detection_results = []
        for i, (original_text, watermarked_text) in enumerate(zip(texts, watermarked_texts)):
            # 使用原始文本作为输入进行检测
            inputs = tokenizer(original_text, return_tensors="pt", padding=True, truncation=True,
                             max_length=self.config.experiment.max_length).to(self.device)
            
            score, verdict = detector.detect(inputs["input_ids"])
            detection_results.append({
                "sample_id": i,
                "original_text": original_text[:100],  # 截断显示
                "watermarked_text": watermarked_text[:100],
                "hit_rate": score,
                "verdict": verdict
            })
        
        # 6. 统计结果
        hit_rates = [r["hit_rate"] for r in detection_results]
        avg_hit_rate = np.mean(hit_rates) if hit_rates else 0.0
        watermarked_count = sum(1 for r in detection_results if r["verdict"] == "Watermarked")
        
        results = {
            "experiment_name": self.config.experiment.experiment_name,
            "config": self.config.to_dict(),
            "summary": {
                "total_samples": len(texts),
                "watermarked_samples": watermarked_count,
                "average_hit_rate": float(avg_hit_rate),
                "detection_threshold": self.config.detection.hit_rate_threshold
            },
            "detailed_results": detection_results
        }
        
        logger.info("=" * 60)
        logger.info("实验完成")
        logger.info(f"总样本数: {len(texts)}")
        logger.info(f"检测为水印: {watermarked_count}")
        logger.info(f"平均命中率: {avg_hit_rate:.4f}")
        logger.info("=" * 60)
        
        # 保存结果
        self.save_results(results)
        
        return results


class OKRRobustnessExperiment(OKRExperimentFramework):
    """
    OKR 鲁棒性实验
    
    测试水印在释义攻击下的表现
    """
    
    def run(self) -> Dict[str, Any]:
        """
        运行鲁棒性实验
        
        Returns:
            实验结果字典
        """
        logger.info("=" * 60)
        logger.info("OKR 鲁棒性实验开始")
        logger.info("=" * 60)
        
        # 1. 加载模型和分词器
        tokenizer = self.load_tokenizer()
        model = self.load_model()
        
        # 2. 注入水印
        watermarked_model = self.inject_watermark(model)
        
        # 3. 准备数据
        texts = self.prepare_data()
        
        # 4. 生成带水印的文本
        logger.info("生成带水印的文本...")
        watermarked_texts = []
        for text in tqdm(texts, desc="生成文本"):
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True,
                             max_length=self.config.experiment.max_length).to(self.device)
            
            with torch.no_grad():
                outputs = watermarked_model.generate(
                    **inputs,
                    max_length=self.config.experiment.max_length,
                    num_beams=1,
                    do_sample=False
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            watermarked_texts.append(generated_text)
        
        # 5. 进行释义攻击
        logger.info("进行释义攻击...")
        attacked_texts = self._paraphrase_attack(watermarked_texts)
        
        # 6. 检测攻击后的水印
        logger.info("检测攻击后的水印...")
        detector = OKRDetector(watermarked_model, epsilon=self.config.watermark.epsilon)
        
        detection_results = []
        for i, (original_text, attacked_text) in enumerate(zip(texts, attacked_texts)):
            # 使用攻击后的文本作为输入进行检测
            inputs = tokenizer(attacked_text, return_tensors="pt", padding=True, truncation=True,
                             max_length=self.config.experiment.max_length).to(self.device)
            
            score, verdict = detector.detect(inputs["input_ids"])
            detection_results.append({
                "sample_id": i,
                "original_text": original_text[:100],
                "attacked_text": attacked_text[:100],
                "hit_rate": score,
                "verdict": verdict
            })
        
        # 7. 统计结果
        hit_rates = [r["hit_rate"] for r in detection_results]
        avg_hit_rate = np.mean(hit_rates) if hit_rates else 0.0
        watermarked_count = sum(1 for r in detection_results if r["verdict"] == "Watermarked")
        
        results = {
            "experiment_name": "OKR_Robustness",
            "config": self.config.to_dict(),
            "summary": {
                "total_samples": len(texts),
                "watermarked_after_attack": watermarked_count,
                "average_hit_rate": float(avg_hit_rate),
                "detection_threshold": self.config.detection.hit_rate_threshold
            },
            "detailed_results": detection_results
        }
        
        logger.info("=" * 60)
        logger.info("鲁棒性实验完成")
        logger.info(f"总样本数: {len(texts)}")
        logger.info(f"攻击后仍检测为水印: {watermarked_count}")
        logger.info(f"平均命中率: {avg_hit_rate:.4f}")
        logger.info("=" * 60)
        
        # 保存结果
        self.save_results(results, "robustness_results.json")
        
        return results
    
    def _paraphrase_attack(self, texts: List[str]) -> List[str]:
        """
        对文本进行释义攻击
        
        Args:
            texts: 原始文本列表
            
        Returns:
            攻击后的文本列表
        """
        # TODO: 实现释义攻击
        # 这里先返回原始文本，实际应该调用释义模型
        logger.warning("释义攻击功能待实现，返回原始文本")
        return texts


def run_okr_experiment(config: Optional[OKRConfig] = None, experiment_type: str = "basic"):
    """
    运行 OKR 实验的便捷函数
    
    Args:
        config: OKR 配置（如果为 None，使用默认配置）
        experiment_type: 实验类型（"basic" 或 "robustness"）
        
    Returns:
        实验结果字典
    """
    if config is None:
        config = get_default_okr_config()
    
    # 设置日志
    log_file = Path(config.experiment.output_dir) / "experiment.log"
    setup_logging(log_file=log_file)
    
    # 运行实验
    if experiment_type == "basic":
        experiment = OKRBasicExperiment(config)
    elif experiment_type == "robustness":
        experiment = OKRRobustnessExperiment(config)
    else:
        raise ValueError(f"未知的实验类型: {experiment_type}")
    
    return experiment.run()


if __name__ == "__main__":
    # 示例：运行基础实验
    config = get_default_okr_config()
    results = run_okr_experiment(config, experiment_type="basic")
    print("\n实验结果:")
    print(json.dumps(results["summary"], indent=2))

