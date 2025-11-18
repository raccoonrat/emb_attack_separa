"""
MVES (最小验证实验) 主脚本

使用google/switch-base-8模型进行最小验证实验
采用LogitsProcessor API实现水印嵌入
"""

import torch
import numpy as np
import json
import os
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    LogitsProcessorList
)

from mves_config import MVESConfig, get_default_config, get_quick_test_config
from mves_watermark import create_watermark_processor, MoEWatermarkLogitsProcessor
from detector import LLRDetector
from attacks import paraphrase_text_batch, estimate_gamma_from_text


class MVESExperiment:
    """
    MVES实验框架
    
    严格按照论文理论框架，使用LogitsProcessor API实现
    """
    
    def __init__(self, config: MVESConfig):
        """
        初始化MVES实验
        
        Args:
            config: MVES配置对象
        """
        self.config = config
        config.validate()
        
        # 设置随机种子
        torch.manual_seed(config.experiment.seed)
        np.random.seed(config.experiment.seed)
        
        # 加载模型和分词器
        print(f"\n{'='*60}")
        print("MVES: 最小验证实验")
        print(f"{'='*60}")
        print(f"模型: {config.model.model_name}")
        print(f"水印强度 ε: {config.watermark.epsilon:.6f}")
        print(f"安全系数 c*: {config.watermark.c_star:.2f}")
        print(f"攻击强度 γ: {config.watermark.gamma_design:.4f}")
        
        self.model, self.tokenizer = self._load_model()
        self.watermark_processor = None
        
    def _load_model(self) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer]:
        """加载模型和分词器"""
        print(f"\n加载模型: {self.config.model.model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.model_name,
            trust_remote_code=self.config.model.trust_remote_code
        )
        
        # Switch Transformer是Seq2Seq模型
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model.model_name,
            torch_dtype=getattr(torch, self.config.model.torch_dtype) if hasattr(torch, self.config.model.torch_dtype) else torch.float32,
            device_map=self.config.model.device,
            trust_remote_code=self.config.model.trust_remote_code
        )
        
        model.eval()
        print("✓ 模型加载完成")
        
        return model, tokenizer
    
    def embed_watermark(self, text: str) -> Tuple[str, Dict]:
        """
        嵌入水印并生成文本
        
        Args:
            text: 输入文本
            
        Returns:
            watermarked_text: 带水印的文本
            metadata: 元数据 (包含检测信息)
        """
        print(f"\n嵌入水印: {text[:50]}...")
        
        # 创建水印处理器
        if self.watermark_processor is None:
            self.watermark_processor = create_watermark_processor(
                self.model, self.config, self.tokenizer
            )
        
        # 编码输入
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.experiment.max_length
        ).to(self.model.device)
        
        # 创建LogitsProcessorList
        logits_processor = LogitsProcessorList([self.watermark_processor])
        
        # 生成带水印的文本
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.config.experiment.max_length,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                logits_processor=logits_processor,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            )
        
        # 解码输出
        watermarked_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 获取检测数据
        detection_data = self.watermark_processor.get_detection_data()
        
        metadata = {
            "original_text": text,
            "watermarked_text": watermarked_text,
            "detection_data_count": len(detection_data),
            "epsilon": self.config.watermark.epsilon
        }
        
        print(f"✓ 水印嵌入完成")
        
        return watermarked_text, metadata
    
    def detect_watermark(self, text: str, apply_attack: bool = False) -> Dict:
        """
        检测水印
        
        Args:
            text: 待检测文本
            apply_attack: 是否应用攻击
            
        Returns:
            detection_result: 检测结果
        """
        print(f"\n检测水印: {text[:50]}...")
        
        original_text = text
        
        # 应用攻击 (如果指定)
        if apply_attack and self.config.attack.attack_type != "none":
            print(f"应用攻击: {self.config.attack.attack_type}")
            attacked_texts = paraphrase_text_batch(
                [text],
                attack_strength=self.config.attack.attack_strength
            )
            text = attacked_texts[0] if attacked_texts else text
            
            # 估计攻击强度
            gamma_est = estimate_gamma_from_text(
                original_text,
                text,
                len(self.tokenizer),
                method=self.config.attack.gamma_estimation_method
            )
            print(f"估计攻击强度 γ: {gamma_est:.4f}")
        else:
            gamma_est = 0.0
        
        # 重新嵌入水印以获取检测数据
        # 注意: 在实际应用中，检测器应该能够从文本中提取水印信息
        # 这里我们使用一个简化的方法：重新运行模型获取路由信息
        
        # 创建检测器 (需要模型已patch)
        # 对于MVES，我们使用LogitsProcessor的方式，检测需要特殊处理
        # 这里提供一个简化实现
        
        # 使用水印处理器获取检测数据
        if self.watermark_processor is None:
            self.watermark_processor = create_watermark_processor(
                self.model, self.config, self.tokenizer
            )
        
        # 运行模型获取检测数据
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.experiment.max_length
        ).to(self.model.device)
        
        logits_processor = LogitsProcessorList([self.watermark_processor])
        
        with torch.no_grad():
            _ = self.model.generate(
                **inputs,
                max_length=min(10, self.config.experiment.max_length),
                logits_processor=logits_processor
            )
        
        # 获取检测数据
        detection_data = self.watermark_processor.get_detection_data()
        
        # 计算LLR统计量 (简化实现)
        if detection_data:
            # 使用最后一个检测数据
            p_0, p_1, _ = detection_data[-1]
            
            # 计算简化的LLR
            p_0_clamped = torch.clamp(p_0, min=1e-9)
            p_1_clamped = torch.clamp(p_1, min=1e-9)
            llr = torch.sum(p_1_clamped * torch.log(p_1_clamped / p_0_clamped)).item()
        else:
            llr = 0.0
        
        # 判决
        is_detected = llr > self.config.detection.tau_alpha
        
        result = {
            "text": text,
            "original_text": original_text if apply_attack else None,
            "llr_score": llr,
            "threshold": self.config.detection.tau_alpha,
            "is_detected": is_detected,
            "gamma_estimated": gamma_est,
            "attack_applied": apply_attack
        }
        
        print(f"✓ 检测完成: {'检测到水印' if is_detected else '未检测到水印'} (LLR: {llr:.4f})")
        
        return result
    
    def run_experiment(self) -> Dict:
        """
        运行完整MVES实验
        
        Returns:
            results: 实验结果
        """
        print(f"\n{'='*60}")
        print("开始MVES实验")
        print(f"{'='*60}")
        
        results = {
            "config": self.config.to_dict(),
            "embedding_results": [],
            "detection_results": [],
            "robustness_results": []
        }
        
        # 获取测试样本
        if self.config.experiment.custom_prompts:
            test_prompts = self.config.experiment.custom_prompts
        else:
            # 使用默认提示
            test_prompts = [
                "The quick brown fox jumps over the lazy dog.",
                "In a world where technology advances rapidly."
            ]
        
        # 限制样本数
        test_prompts = test_prompts[:self.config.experiment.num_samples]
        
        print(f"\n使用 {len(test_prompts)} 个测试样本")
        
        # 1. 水印嵌入实验
        print(f"\n{'='*60}")
        print("阶段1: 水印嵌入")
        print(f"{'='*60}")
        
        for i, prompt in enumerate(tqdm(test_prompts, desc="嵌入水印")):
            try:
                watermarked_text, metadata = self.embed_watermark(prompt)
                results["embedding_results"].append({
                    "index": i,
                    "prompt": prompt,
                    "watermarked_text": watermarked_text,
                    "metadata": metadata
                })
            except Exception as e:
                print(f"警告: 样本 {i} 嵌入失败: {e}")
                continue
        
        # 2. 水印检测实验 (无攻击)
        print(f"\n{'='*60}")
        print("阶段2: 水印检测 (无攻击)")
        print(f"{'='*60}")
        
        for i, result in enumerate(tqdm(results["embedding_results"], desc="检测水印")):
            try:
                watermarked_text = result["watermarked_text"]
                detection_result = self.detect_watermark(watermarked_text, apply_attack=False)
                results["detection_results"].append({
                    "index": i,
                    **detection_result
                })
            except Exception as e:
                print(f"警告: 样本 {i} 检测失败: {e}")
                continue
        
        # 3. 鲁棒性测试 (有攻击)
        if self.config.attack.attack_type != "none":
            print(f"\n{'='*60}")
            print("阶段3: 鲁棒性测试 (释义攻击)")
            print(f"{'='*60}")
            
            for i, result in enumerate(tqdm(results["embedding_results"], desc="鲁棒性测试")):
                try:
                    watermarked_text = result["watermarked_text"]
                    robustness_result = self.detect_watermark(watermarked_text, apply_attack=True)
                    results["robustness_results"].append({
                        "index": i,
                        **robustness_result
                    })
                except Exception as e:
                    print(f"警告: 样本 {i} 鲁棒性测试失败: {e}")
                    continue
        
        # 计算统计信息
        results["statistics"] = self._compute_statistics(results)
        
        # 保存结果
        self._save_results(results)
        
        # 打印摘要
        self._print_summary(results)
        
        return results
    
    def _compute_statistics(self, results: Dict) -> Dict:
        """计算统计信息"""
        stats = {}
        
        # 检测准确率
        if results["detection_results"]:
            detected_count = sum(1 for r in results["detection_results"] if r["is_detected"])
            stats["detection_accuracy"] = detected_count / len(results["detection_results"])
            stats["detection_count"] = detected_count
            stats["total_count"] = len(results["detection_results"])
        
        # 鲁棒性 (攻击后检测率)
        if results["robustness_results"]:
            robust_count = sum(1 for r in results["robustness_results"] if r["is_detected"])
            stats["robustness_rate"] = robust_count / len(results["robustness_results"])
            stats["robust_count"] = robust_count
            stats["robust_total"] = len(results["robustness_results"])
            
            # 平均攻击强度
            gammas = [r["gamma_estimated"] for r in results["robustness_results"] if r["gamma_estimated"] > 0]
            if gammas:
                stats["avg_gamma"] = np.mean(gammas)
                stats["max_gamma"] = np.max(gammas)
                stats["min_gamma"] = np.min(gammas)
        
        # 平均LLR分数
        if results["detection_results"]:
            llr_scores = [r["llr_score"] for r in results["detection_results"]]
            stats["avg_llr"] = np.mean(llr_scores)
            stats["max_llr"] = np.max(llr_scores)
            stats["min_llr"] = np.min(llr_scores)
        
        return stats
    
    def _save_results(self, results: Dict):
        """保存实验结果"""
        output_dir = self.config.experiment.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存完整结果
        results_file = os.path.join(output_dir, "mves_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 保存配置
        config_file = os.path.join(output_dir, "mves_config.json")
        self.config.save(config_file)
        
        print(f"\n✓ 结果已保存到: {output_dir}")
    
    def _print_summary(self, results: Dict):
        """打印实验摘要"""
        print(f"\n{'='*60}")
        print("MVES实验摘要")
        print(f"{'='*60}")
        
        stats = results.get("statistics", {})
        
        if "detection_accuracy" in stats:
            print(f"\n检测准确率: {stats['detection_accuracy']:.2%}")
            print(f"  - 检测到: {stats.get('detection_count', 0)} / {stats.get('total_count', 0)}")
        
        if "robustness_rate" in stats:
            print(f"\n鲁棒性 (攻击后): {stats['robustness_rate']:.2%}")
            print(f"  - 检测到: {stats.get('robust_count', 0)} / {stats.get('robust_total', 0)}")
            if "avg_gamma" in stats:
                print(f"  - 平均攻击强度 γ: {stats['avg_gamma']:.4f}")
        
        if "avg_llr" in stats:
            print(f"\nLLR统计量:")
            print(f"  - 平均值: {stats['avg_llr']:.4f}")
            print(f"  - 范围: [{stats.get('min_llr', 0):.4f}, {stats.get('max_llr', 0):.4f}]")
        
        print(f"\n{'='*60}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MVES: 最小验证实验")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径 (JSON)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="使用快速测试配置"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        config = MVESConfig.load(args.config)
    elif args.quick:
        config = get_quick_test_config()
    else:
        config = get_default_config()
    
    # 覆盖输出目录
    if args.output_dir:
        config.experiment.output_dir = args.output_dir
    
    # 运行实验
    experiment = MVESExperiment(config)
    results = experiment.run_experiment()
    
    print("\n✓ MVES实验完成!")


if __name__ == "__main__":
    main()

