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
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
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
        """加载分词器（支持本地路径）"""
        if self.tokenizer is None:
            # 确定模型路径：优先使用本地路径
            model_path = self.config.model.local_model_path or self.config.model.model_name
            
            logger.info(f"加载分词器: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=self.config.model.trust_remote_code
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        return self.tokenizer
    
    def load_model(self):
        """加载模型（支持 encoder-decoder 和 decoder-only，支持本地路径）"""
        if self.model is None:
            # 确定模型路径：优先使用本地路径
            model_path = self.config.model.local_model_path or self.config.model.model_name
            
            if self.config.model.local_model_path:
                logger.info(f"从本地路径加载模型: {model_path}")
            else:
                logger.info(f"从 HuggingFace 加载模型: {model_path}")
            
            # 确定数据类型
            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16
            }
            torch_dtype = dtype_map.get(self.config.model.torch_dtype, torch.float32)
            
            # 根据模型类型选择加载方式
            model_name_lower = model_path.lower()
            is_decoder_only = (
                "deepseek" in model_name_lower or 
                "moe" in model_name_lower or
                self.config.model.model_type == "deepseek_moe"
            )
            
            # 尝试加载模型
            try:
                if is_decoder_only:
                    # DeepSeek-MoE 是 decoder-only 模型
                    logger.info("检测到 decoder-only 模型，使用 AutoModelForCausalLM")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch_dtype,
                        device_map="auto" if torch.cuda.is_available() else None,
                        low_cpu_mem_usage=True,
                        max_memory=self.config.model.max_memory,
                        trust_remote_code=self.config.model.trust_remote_code
                    )
                else:
                    # Switch Transformers 是 encoder-decoder 模型
                    logger.info("检测到 encoder-decoder 模型，使用 AutoModelForSeq2SeqLM")
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_path,
                        torch_dtype=torch_dtype,
                        device_map="auto" if torch.cuda.is_available() else None,
                        low_cpu_mem_usage=True,
                        max_memory=self.config.model.max_memory,
                        trust_remote_code=self.config.model.trust_remote_code
                    )
            except Exception as e:
                logger.warning(f"使用 AutoModelForSeq2SeqLM 加载失败: {e}")
                logger.info("尝试使用 AutoModelForCausalLM...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch_dtype,
                    device_map="auto" if torch.cuda.is_available() else None,
                    low_cpu_mem_usage=True,
                    max_memory=self.config.model.max_memory,
                    trust_remote_code=self.config.model.trust_remote_code
                )
            
            self.model.eval()
            logger.info("模型加载完成")
        
        return self.model
    
    def inject_watermark(self, model=None):
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
        
        # 使用配置对象注入 OKR，这样可以传递 num_experts 和 top_k
        try:
            from okr_patch import inject_okr_with_config
            watermarked_model = inject_okr_with_config(model, self.config)
        except ImportError:
            # 如果 inject_okr_with_config 不存在，使用原来的方法
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
        watermarked_token_ids = []  # 保存生成的token序列，用于检测
        sample_routing_data = []  # 保存每个样本的路由数据
        
        for text in tqdm(texts, desc="生成文本"):
            # 强制数据清空：使用注入的 clear_okr_stats() 方法
            if hasattr(watermarked_model, 'clear_okr_stats'):
                watermarked_model.clear_okr_stats()
                logger.debug(f"样本 {len(watermarked_texts) + 1}: 已调用 clear_okr_stats() 清空路由数据")
            else:
                # 兼容旧代码：手动清空
                if hasattr(watermarked_model, '_okr_routing_data'):
                    watermarked_model._okr_routing_data = {}
                # 清空所有router的累积数据
                for name, module in watermarked_model.named_modules():
                    if hasattr(module, '_okr_all_selected_experts'):
                        module._okr_all_selected_experts = []
            
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, 
                            max_length=self.config.experiment.max_length).to(self.device)
            
            # 判断模型类型：decoder-only 还是 encoder-decoder
            is_decoder_only = not hasattr(watermarked_model, 'encoder') or watermarked_model.encoder is None
            
            with torch.no_grad():
                if is_decoder_only:
                    # decoder-only 模型（如 DeepSeek-MoE）
                    # 添加 use_cache=False 以避免 DynamicCache.get_usable_length 兼容性问题
                    outputs = watermarked_model.generate(
                        **inputs,
                        max_length=self.config.experiment.max_length,
                        num_beams=1,
                        do_sample=False,
                        use_cache=False,  # 禁用 cache 以避免兼容性问题
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
                    )
                else:
                    # encoder-decoder 模型（如 Switch Transformers）
                    decoder_start_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                    outputs = watermarked_model.generate(
                        **inputs,
                        max_length=self.config.experiment.max_length,
                        num_beams=1,
                        do_sample=False,
                        use_cache=False,  # 禁用 cache 以避免兼容性问题
                        decoder_start_token_id=decoder_start_token_id,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
                    )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            watermarked_texts.append(generated_text)
            # 保存生成的token序列（用于检测）
            watermarked_token_ids.append(outputs[0])
            
            # 关键修复：立即提取并保存当前样本的路由数据，避免累积
            # 问题：每个decoder层都有自己的router，每个router都在保存路由数据
            # 解决方案：只使用第一个router的路由数据（避免重复）
            current_sample_routing_data = None
            router_found = False
            for name, module in watermarked_model.named_modules():
                if hasattr(module, '_okr_all_selected_experts') and module._okr_all_selected_experts:
                    # 只使用第一个router的路由数据（避免多层重复）
                    if not router_found:
                        all_experts = module._okr_all_selected_experts
                        if all_experts and len(all_experts) > 0:
                            current_sample_routing_data = torch.cat(all_experts, dim=1)  # [batch, seq_len, top_k]
                            router_found = True
                            logger.debug(f"样本 {len(watermarked_texts) + 1}: 从 {name} 提取路由数据")
                            break
            
            # 保存当前样本的路由数据
            if current_sample_routing_data is not None:
                sample_routing_data.append(current_sample_routing_data.clone())  # 深拷贝，避免后续被修改
                routing_data_len = current_sample_routing_data.shape[1]
            else:
                sample_routing_data.append(None)
                routing_data_len = 0
                logger.warning(f"样本 {len(watermarked_texts)}: 未找到路由数据")
            
            logger.info(f"样本 {len(watermarked_texts)}: 原始='{text[:50]}...', 生成='{generated_text[:50]}...', generated_token数={outputs[0].shape[0]}, 路由数据token数={routing_data_len}")
        
        # 5. 检测水印
        logger.info("开始检测水印...")
        detector = OKRDetector(watermarked_model, epsilon=self.config.watermark.epsilon)
        
        detection_results = []
        for i, (original_text, watermarked_text, generated_token_ids, sample_routing) in enumerate(zip(texts, watermarked_texts, watermarked_token_ids, sample_routing_data)):
            # 重要：检测时应该使用生成时保存的路由数据（每个样本独立的路由数据）
            # 但我们需要重新运行模型来获取 hidden_states
            # 使用原始文本作为 encoder 输入，使用生成的token序列作为 decoder 输入
            encoder_inputs = tokenizer(original_text, return_tensors="pt", padding=True, truncation=True,
                                      max_length=self.config.experiment.max_length).to(self.device)
            
            # 关键修复：使用generate()返回的原始token序列，而不是解码后的文本
            # 对于encoder-decoder模型，generate()返回的outputs[0]只包含decoder的输出序列
            # 路由数据是decoder生成时累积的，包含了所有生成的token（自回归生成时每次forward处理1个token）
            # 所以outputs[0]的长度应该等于路由数据的长度（或接近）
            decoder_start_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            
            # 使用generate()返回的原始token序列（不跳过special tokens，以保持完整）
            # generated_token_ids是1D tensor [seq_len]，包含了decoder生成的完整序列
            if len(generated_token_ids.shape) == 1:
                generated_seq = generated_token_ids
            else:
                generated_seq = generated_token_ids[0]
            
            # 对于encoder-decoder模型，generate()返回的outputs[0]只包含decoder输出
            # 但我们需要准备decoder输入：decoder_start_token_id + generated_seq[:-1]
            # 这样decoder输入的长度 = 1 + (len(generated_seq) - 1) = len(generated_seq)
            if generated_seq.shape[0] > 0:
                decoder_input_ids = torch.cat([
                    torch.tensor([[decoder_start_token_id]], device=self.device, dtype=torch.long),
                    generated_seq[:-1].unsqueeze(0) if generated_seq.shape[0] > 1 else torch.tensor([[decoder_start_token_id]], device=self.device, dtype=torch.long)
                ], dim=1)
            else:
                decoder_input_ids = torch.tensor([[decoder_start_token_id]], device=self.device, dtype=torch.long)
            
            # 关键修复：将当前样本的路由数据临时设置到router对象上，供检测器使用
            # 检测前先清空router的路由数据，然后设置当前样本的路由数据
            if hasattr(watermarked_model, 'clear_okr_stats'):
                watermarked_model.clear_okr_stats()
            
            # 将当前样本的路由数据设置到第一个router对象上（避免多层重复）
            if sample_routing is not None:
                router_for_detection = None
                router_found = False
                for name, module in watermarked_model.named_modules():
                    if hasattr(module, '_okr_all_selected_experts'):
                        # 只使用第一个router（避免多层重复）
                        if not router_found:
                            router_for_detection = module
                            router_found = True
                            logger.debug(f"样本 {i}: 使用router {name} 进行检测")
                            break
                
                if router_for_detection is not None:
                    # 将当前样本的路由数据转换为列表格式（每个token一个元素）
                    # sample_routing: [batch, seq_len, top_k]
                    # 需要转换为列表，每个元素是 [batch, 1, top_k]
                    routing_list = []
                    for j in range(sample_routing.shape[1]):
                        routing_list.append(sample_routing[:, j:j+1, :])
                    router_for_detection._okr_all_selected_experts = routing_list
                    routing_data_len = sample_routing.shape[1]
                else:
                    routing_data_len = 0
                    logger.warning(f"样本 {i}: 未找到router对象")
            else:
                routing_data_len = 0
                logger.warning(f"样本 {i}: 未找到路由数据")
            
            logger.info(f"样本 {i}: generated_seq长度={generated_seq.shape[0]}, decoder_input_ids长度={decoder_input_ids.shape[1]}, 路由数据长度={routing_data_len}, watermarked_text长度={len(watermarked_text)}")
            
            # 检测器会自动处理 encoder-decoder 模型和 decoder-only 模型
            # 对于 decoder-only 模型，使用 decoder_input_ids 作为 input_ids
            # 对于 encoder-decoder 模型，传入 decoder_input_ids 以便使用生成时的token序列
            is_decoder_only = not hasattr(watermarked_model, 'encoder') or watermarked_model.encoder is None
            if is_decoder_only:
                # decoder-only 模型：使用 decoder_input_ids 作为 input_ids
                score, verdict = detector.detect(
                    input_ids=decoder_input_ids,
                    attention_mask=None,  # decoder-only 模型可能不需要 attention_mask
                    decoder_input_ids=None
                )
            else:
                # encoder-decoder 模型：传入 decoder_input_ids
                score, verdict = detector.detect(
                    input_ids=encoder_inputs["input_ids"],
                    attention_mask=encoder_inputs.get("attention_mask"),
                    decoder_input_ids=decoder_input_ids  # 使用生成的完整文本重新tokenize后的序列
                )
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
                    do_sample=False,
                    use_cache=False  # 禁用 cache 以避免兼容性问题
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

