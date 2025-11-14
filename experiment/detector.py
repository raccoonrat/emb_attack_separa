import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple

from moe_watermark import get_watermark_data_from_model

class LLRDetector:
    """
    实现了基于 LLR 的最优检测器 (对标方案 3. 节)
    """
    def __init__(
        self, 
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
        tau_alpha: float = 20.0 # 判决阈值，应通过 H0 实验标定
    ):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.tau_alpha = tau_alpha
        self.device = model.device

    def compute_llr_from_data(
        self, 
        watermark_data_list: list
    ) -> float:
        """
        从模型输出中计算 LLR 统计量。
        """
        total_llr = 0.0
        
        # 遍历所有 MoE 层
        for layer_data in watermark_data_list:
            # (p_0, p_1, S_obs)
            p_0_dist_batch, p_1_dist_batch, S_indices_batch = layer_data
            
            # [batch, seq, K_experts]
            # [batch, seq, k_top]
            
            # 简化 LLR (如 thought 中所述):
            # Λ = Σ_i Σ_{e in S_i} log(p1(e)/p0(e))
            
            batch_size, seq_len, k_top = S_indices_batch.shape
            
            # 收集 S_i 对应的 p0 和 p1 概率
            # [batch, seq, k_top]
            p0_S = torch.gather(p_0_dist_batch, -1, S_indices_batch)
            p1_S = torch.gather(p_1_dist_batch, -1, S_indices_batch)
            
            # 防止 log(0)
            p0_S = torch.clamp(p0_S, min=1e-9)
            p1_S = torch.clamp(p1_S, min=1e-9)
            
            # 计算 LLR
            llr_per_expert = torch.log(p1_S) - torch.log(p0_S)
            
            # 对 k_top 个专家求和, 然后对 batch 和 seq 求和
            total_llr += torch.sum(llr_per_expert).item()
            
        return total_llr

    def detect(self, text: str) -> Tuple[bool, float]:
        """
        检测给定文本是否包含水印。
        返回 (是否检测到, LLR分数)
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # 运行模型 (已 patch)
        with torch.no_grad():
            self.model(**inputs)
        
        # 提取数据
        watermark_data = get_watermark_data_from_model(self.model)
        
        if not watermark_data:
            print("错误：未能从模型中提取到水印数据。模型是否已正确修补 (patch)？")
            return False, 0.0
            
        # 计算 LLR
        llr_score = self.compute_llr_from_data(watermark_data)
        
        # 判决
        is_detected = llr_score > self.tau_alpha
        
        print(f"LLR Score: {llr_score:.4f} (Threshold: {self.tau_alpha})")
        
        return is_detected, llr_score