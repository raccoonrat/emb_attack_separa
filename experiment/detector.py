import torch
import numpy as np
from scipy.optimize import minimize_scalar
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import Tuple, List, Optional

from mves_watermark_corrected import get_watermark_data_from_switch_model

class LLRDetector:
    """
    实现了基于 LLR 的最优检测器 (论文定理3.1: Neyman-Pearson引理)
    
    核心理论 (论文第3节):
    1. 最优检验器为似然比检验: Λ_n = Σ_i log(p1(S_i)/p0(S_i))
    2. 判决规则: Λ_n > τ_α => 判为H1 (有水印)
    3. 错误率指数衰减: log P_e(n) = -n·D*(p0, p1) + o(n) (论文定理3.2)
    """
    def __init__(
        self, 
        model: AutoModelForSeq2SeqLM, 
        tokenizer: AutoTokenizer, 
        tau_alpha: float = 20.0,  # 判决阈值，应通过H0实验标定 (论文第3节)
        alpha: float = 0.01  # 第一类错误率 (假阳性率)
    ):
        """
        初始化LLR检测器
        
        Args:
            model: 已patch的MoE模型
            tokenizer: 分词器
            tau_alpha: 检测阈值 (论文第3节)
            alpha: 第一类错误率 (用于自动标定tau_alpha)
        """
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.tau_alpha = tau_alpha
        self.alpha = alpha
        self.device = model.device

    def compute_chernoff_information(
        self, 
        p0: torch.Tensor, 
        p1: torch.Tensor
    ) -> float:
        """
        计算Chernoff信息 D*(p0, p1) (论文定理3.2)
        
        D* = -min_{λ in [0,1]} log(Σ_e p0(e)^(1-λ) * p1(e)^λ)
        
        等价形式:
        D* = max_{λ in [0,1]} [-log(Σ_e p0(e)^(1-λ) * p1(e)^λ)]
        
        Args:
            p0: 原始激活分布 [..., num_experts]
            p1: 修改后激活分布 [..., num_experts]
            
        Returns:
            chernoff_info: Chernoff信息值 (nats)
        """
        # 转换为numpy进行优化
        p0_np = p0.detach().cpu().numpy().flatten()
        p1_np = p1.detach().cpu().numpy().flatten()
        
        # 归一化 (防止数值问题)
        p0_np = p0_np / (p0_np.sum() + 1e-9)
        p1_np = p1_np / (p1_np.sum() + 1e-9)
        
        # 防止log(0)
        p0_np = np.clip(p0_np, 1e-9, 1.0)
        p1_np = np.clip(p1_np, 1e-9, 1.0)
        
        def objective(lambda_val):
            """目标函数: -log(Σ_e p0^(1-λ) * p1^λ)"""
            if lambda_val < 0 or lambda_val > 1:
                return np.inf
            try:
                # 计算 Σ_e p0^(1-λ) * p1^λ
                sum_term = np.sum(np.power(p0_np, 1 - lambda_val) * np.power(p1_np, lambda_val))
                if sum_term <= 0:
                    return np.inf
                log_sum = np.log(sum_term)
                return -log_sum
            except:
                return np.inf
        
        # 优化求解最优λ
        result = minimize_scalar(objective, bounds=(0, 1), method='bounded')
        
        if result.success:
            chernoff_info = -result.fun
        else:
            # 边界情况: 尝试λ=0和λ=1
            chernoff_info = max(objective(0), objective(1))
        
        return float(chernoff_info)

    def compute_llr_from_data(
        self, 
        watermark_data_list: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> Tuple[float, float]:
        """
        从模型输出中计算LLR统计量和Chernoff信息 (论文定理3.1-3.2)
        
        LLR统计量: Λ_n = Σ_i Σ_{e in S_i} log(p1(e)/p0(e))
        
        Args:
            watermark_data_list: 每层的数据 [(p_0, p_1, S_obs), ...]
            
        Returns:
            total_llr: 总LLR统计量
            avg_chernoff: 平均Chernoff信息
        """
        total_llr = 0.0
        total_chernoff = 0.0
        num_samples = 0
        
        # 遍历所有 MoE 层
        for layer_data in watermark_data_list:
            # (p_0, p_1, S_obs)
            p_0_dist_batch, p_1_dist_batch, S_indices_batch = layer_data
            
            # [batch, seq, K_experts]
            # [batch, seq, k_top]
            
            batch_size, seq_len, k_top = S_indices_batch.shape
            
            # 计算LLR (论文定理3.1)
            # Λ = Σ_i Σ_{e in S_i} log(p1(e)/p0(e))
            
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
            
            # 计算Chernoff信息 (论文定理3.2)
            # 对每个位置计算平均Chernoff信息
            for b in range(batch_size):
                for s in range(seq_len):
                    p0_pos = p_0_dist_batch[b, s, :]
                    p1_pos = p_1_dist_batch[b, s, :]
                    chernoff = self.compute_chernoff_information(p0_pos, p1_pos)
                    total_chernoff += chernoff
                    num_samples += 1
        
        avg_chernoff = total_chernoff / max(num_samples, 1)
        
        return total_llr, avg_chernoff

    def calibrate_threshold(
        self, 
        null_samples: List[str], 
        num_bootstrap: int = 1000
    ) -> float:
        """
        标定检测阈值 τ_α (论文第3节)
        
        在H0假设下 (无水印), 计算LLR统计量的分布,
        选择阈值使得 P(Λ > τ_α | H0) = α
        
        Args:
            null_samples: 无水印样本列表
            num_bootstrap: Bootstrap采样次数
            
        Returns:
            tau_alpha: 标定后的阈值
        """
        print(f"Calibrating detection threshold (α={self.alpha})...")
        
        llr_scores = []
        
        for text in null_samples[:min(100, len(null_samples))]:  # 限制样本数
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            
            with torch.no_grad():
                # 对于 Seq2Seq 模型，需要分别调用 encoder 和 decoder
                encoder_outputs = self.model.encoder(**inputs)
                
                # 创建 decoder 输入
                if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
                    decoder_start_token_id = self.tokenizer.pad_token_id
                elif hasattr(self.tokenizer, 'bos_token_id') and self.tokenizer.bos_token_id is not None:
                    decoder_start_token_id = self.tokenizer.bos_token_id
                else:
                    decoder_start_token_id = 0
                
                batch_size = inputs['input_ids'].shape[0]
                decoder_input_ids = torch.full(
                    (batch_size, 1), 
                    decoder_start_token_id, 
                    dtype=torch.long, 
                    device=self.device
                )
                
                self.model.decoder(
                    input_ids=decoder_input_ids,
                    encoder_hidden_states=encoder_outputs.last_hidden_state,
                    encoder_attention_mask=inputs.get('attention_mask', None)
                )
            
            watermark_data = get_watermark_data_from_switch_model(self.model)
            if watermark_data:
                llr, _ = self.compute_llr_from_data(watermark_data)
                llr_scores.append(llr)
        
        if not llr_scores:
            print("Warning: No valid LLR scores for calibration. Using default threshold.")
            return self.tau_alpha
        
        # 计算(1-α)分位数作为阈值
        llr_scores = np.array(llr_scores)
        tau_alpha = np.percentile(llr_scores, (1 - self.alpha) * 100)
        
        print(f"Calibrated threshold: τ_α = {tau_alpha:.4f} (α={self.alpha})")
        return float(tau_alpha)

    def detect(
        self, 
        text: str, 
        return_details: bool = False
    ):
        """
        检测给定文本是否包含水印 (论文定理3.1)
        
        Args:
            text: 待检测文本
            return_details: 是否返回详细信息
            
        Returns:
            如果return_details=False: (is_detected: bool, llr_score: float)
            如果return_details=True: (is_detected: bool, llr_score: float, details: dict)
        """
        # 对于 Seq2Seq 模型（如 Switch Transformers），需要提供 encoder 和 decoder 输入
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        # 运行模型 (已 patch)
        # 重要：使用 model.generate() 来触发所有层的 forward
        # 这会确保所有 MoE 层都被触发，从而收集水印数据
        with torch.no_grad():
            # 使用 generate 生成少量 token 以触发所有层
            # 注意：必须生成至少几个 token 才能触发所有 decoder 层
            # 对于检测，我们需要使用与嵌入时相同的文本作为输入
            # 但检测时我们不需要生成新文本，只需要 forward pass 来收集数据
            try:
                generated_outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=5,  # 生成5个token足以触发所有层
                    do_sample=False,  # 使用贪心解码
                    num_beams=1,
                    pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                    use_cache=False  # 禁用缓存以确保所有层都被触发
                )
            except Exception as e:
                print(f"警告: generate() 失败: {e}")
                print("尝试使用直接 forward pass...")
                # 如果 generate 失败，尝试直接 forward pass
                # 对于 Seq2Seq 模型，需要 encoder 和 decoder
                encoder_outputs = self.model.encoder(**inputs)
                
                # 创建 decoder 输入
                if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
                    decoder_start_token_id = self.tokenizer.pad_token_id
                elif hasattr(self.tokenizer, 'bos_token_id') and self.tokenizer.bos_token_id is not None:
                    decoder_start_token_id = self.tokenizer.bos_token_id
                else:
                    decoder_start_token_id = 0
                
                batch_size = inputs['input_ids'].shape[0]
                decoder_input_ids = torch.full(
                    (batch_size, 1), 
                    decoder_start_token_id, 
                    dtype=torch.long, 
                    device=self.device
                )
                
                # 多次 forward 以确保所有层都被触发
                for _ in range(5):
                    decoder_outputs = self.model.decoder(
                        input_ids=decoder_input_ids,
                        encoder_hidden_states=encoder_outputs.last_hidden_state,
                        encoder_attention_mask=inputs.get('attention_mask', None)
                    )
                    # 更新 decoder_input_ids 用于下一次 forward
                    decoder_input_ids = torch.cat([
                        decoder_input_ids,
                        torch.full((batch_size, 1), decoder_start_token_id, dtype=torch.long, device=self.device)
                    ], dim=1)
        
        # 提取数据（统一使用switch-base-8）
        watermark_data = get_watermark_data_from_switch_model(self.model)
        
        if not watermark_data:
            # 添加详细的调试信息
            print("警告: 未能从模型中提取到水印数据")
            print("  调试信息:")
            
            # 检查模型结构
            decoder = None
            if hasattr(self.model, 'decoder'):
                decoder = self.model.decoder
            elif hasattr(self.model, 'model') and hasattr(self.model.model, 'decoder'):
                decoder = self.model.model.decoder
            
            if decoder:
                print(f"  decoder类型: {type(decoder).__name__}")
                # 检查 decoder blocks
                decoder_blocks = None
                if hasattr(decoder, 'block'):
                    decoder_blocks = decoder.block
                elif hasattr(decoder, 'layers'):
                    decoder_blocks = decoder.layers
                elif hasattr(decoder, 'blocks'):
                    decoder_blocks = decoder.blocks
                
                if decoder_blocks:
                    print(f"  decoder blocks数量: {len(decoder_blocks)}")
                    # 检查第一层
                    if len(decoder_blocks) > 0:
                        first_layer = decoder_blocks[0]
                        print(f"  第一层类型: {type(first_layer).__name__}")
                        # 检查是否有 router
                        router_found = False
                        if hasattr(first_layer, 'layer') and len(first_layer.layer) > 1:
                            ffn_layer = first_layer.layer[1]
                            if hasattr(ffn_layer, 'mlp') and hasattr(ffn_layer.mlp, 'router'):
                                router = ffn_layer.mlp.router
                                router_found = True
                                print(f"  找到router: {type(router).__name__}")
                                if hasattr(router, '_watermark_detection_data'):
                                    print(f"  router有检测数据: True")
                                else:
                                    print(f"  router有检测数据: False")
                        
                        if not router_found:
                            print(f"  未找到router（可能patch失败）")
                else:
                    print(f"  未找到decoder blocks")
            else:
                print(f"  未找到decoder")
            
            print("  可能原因:")
            print("  1. 模型未正确patch（检查patch输出）")
            print("  2. 检测数据提取路径不正确")
            print("  3. secret_key 不一致（检测和嵌入必须使用相同的key）")
            print("  4. 检测的文本不是带水印的文本（应检测嵌入时生成的文本）")
            
            if return_details:
                return False, 0.0, {"error": "未能从模型中提取到水印数据"}
            return False, 0.0
        
        print(f"  提取到 {len(watermark_data)} 层的水印数据")
        
        # 计算 LLR 和 Chernoff信息
        llr_score, avg_chernoff = self.compute_llr_from_data(watermark_data)
        print(f"  计算得到的 LLR 分数: {llr_score:.4f}, 阈值: {self.tau_alpha:.4f}")
        
        # 判决 (论文定理3.1)
        is_detected = llr_score > self.tau_alpha
        
        if return_details:
            details = {
                "llr_score": llr_score,
                "threshold": self.tau_alpha,
                "chernoff_info": avg_chernoff,
                "is_detected": is_detected,
                "num_layers": len(watermark_data)
            }
            return is_detected, llr_score, details
        
        return is_detected, llr_score