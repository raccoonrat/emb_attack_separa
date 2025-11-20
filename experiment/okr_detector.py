"""
OKR Detector - 检测器

检测器不需要重跑整个大模型。如果你只想验证水印，理论上只需要跑这一层。
但在实际中，我们需要文本对应的 Embedding，所以通常还是得跑一遍 Inference。

这里的逻辑是：计算 Green-List Hit Rate（也就是水印命中率）。
"""

import torch
import numpy as np
from typing import Tuple, Optional


class OKRDetector:
    """
    OKR 水印检测器
    
    核心验证逻辑：
    1. 重算原本的 Logits (Ground Truth)
    2. 重算水印信号 (Expected Signal)
    3. 重算机会窗口 (Opportunities)
    4. 验证命中 (Check Hits)
    """
    
    def __init__(self, model, epsilon: float = 1.5):
        """
        初始化检测器
        
        Args:
            model: 已经注入了 OKRRouter 的模型
            epsilon: 质量容忍阈值（必须与路由器一致）
        """
        self.model = model
        self.epsilon = epsilon

    def detect(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, 
               decoder_input_ids: Optional[torch.Tensor] = None):
        """
        对一段文本进行检测。
        
        Args:
            input_ids: 输入token IDs [batch, seq_len] (encoder输入，对于encoder-decoder模型)
            attention_mask: 注意力掩码 [batch, seq_len]
            decoder_input_ids: decoder输入token IDs [batch, seq_len] (可选，如果不提供则使用input_ids)
            
        Returns:
            score: 水印命中率 (0-1)
            verdict: "Watermarked" 或 "Clean"
        """
        # 检查模型类型（encoder-decoder 还是 decoder-only）
        is_encoder_decoder = hasattr(self.model.config, 'is_encoder_decoder') and self.model.config.is_encoder_decoder
        
        # 对于 encoder-decoder 模型（如 Switch Transformers），需要生成文本来触发所有层
        # 这样路由信息才会被保存
        if is_encoder_decoder:
            # 使用 generate 来触发所有 decoder 层，这样会保存路由数据
            with torch.no_grad():
                # 生成少量 token 以触发所有层
                generated = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=input_ids.shape[1] + 10,  # 生成少量token
                    num_beams=1,
                    do_sample=False,
                    output_hidden_states=True
                )
            
            # 获取实际选择的专家（在generate过程中已保存）
            actual_selected_experts = self._extract_selected_experts()
            
            if actual_selected_experts is None:
                return 0.0, "No routing data available"
            
            # 重新运行模型获取 decoder 的 hidden states
            # decoder_input_ids 应该是生成的文本（去掉最后一个token，因为那是预测的）
            if decoder_input_ids is None:
                # 使用生成的文本作为 decoder 输入（去掉最后一个token）
                decoder_input_ids = generated[:, :-1] if generated.shape[1] > 1 else generated
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    output_hidden_states=True
                )
            
            # 获取 decoder 的最后一层 hidden states
            if hasattr(outputs, 'decoder_hidden_states') and outputs.decoder_hidden_states:
                hidden_states = outputs.decoder_hidden_states[-1]
            elif hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                hidden_states = outputs.hidden_states[-1]
            else:
                return 0.0, "No hidden states available"
        else:
            # 对于 decoder-only 模型（如 GPT）
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
            
            # 获取实际选择的专家
            actual_selected_experts = self._extract_selected_experts()
            
            if actual_selected_experts is None:
                return 0.0, "No routing data available"
        
        return self.verify_batch(hidden_states, actual_selected_experts)

    def verify_batch(self, hidden_states: torch.Tensor, actual_selected_experts: torch.Tensor) -> Tuple[float, str]:
        """
        核心验证逻辑 (纯数学计算)

        Args:
            hidden_states: [batch, seq, dim] - 重新推理得到的 Embedding
            actual_selected_experts: [batch, seq, top_k] - 也就是你想验证的 token 实际上走了哪条路
            
        Returns:
            score: 水印命中率 (0-1)
            verdict: "Watermarked" 或 "Clean"
        """
        # 获取 Router (假设模型只有一层或我们只看这一层)
        router = self._get_router()
        if router is None:
            return 0.0, "No router found"

        # 1. 重算原本的 Logits (Ground Truth)
        raw_logits = router.gate_network(hidden_states)
        max_logits, _ = raw_logits.max(dim=-1, keepdim=True)

        # 2. 重算水印信号 (Expected Signal)
        watermark_bias = torch.matmul(hidden_states, router.secret_projection)

        # 3. 重算机会窗口 (Opportunities)
        # 哪些 Token 拥有至少 2 个安全专家？
        safe_mask = raw_logits >= (max_logits - self.epsilon)
        num_safe_experts = safe_mask.sum(dim=-1)  # [batch, seq]

        # 只有 >= 2 个可选专家时，水印才有可能生效
        # 这就是我们的有效样本 (Valid Samples)
        valid_opportunity_mask = (num_safe_experts >= 2)

        if valid_opportunity_mask.sum() == 0:
            return 0.0, "No Opportunities (Text too clear or epsilon too small)"

        # 4. 验证命中 (Check Hits)
        # 在这些机会窗口里，水印想要谁？
        # 水印想要的是在 safe_mask 范围内，watermark_bias 最大的那个

        masked_watermark_scores = torch.where(
            safe_mask,
            watermark_bias,
            torch.tensor(-1e9, device=watermark_bias.device)
        )
        # 理论上水印指向的第一选择
        target_expert = torch.argmax(masked_watermark_scores, dim=-1)  # [batch, seq]

        # 5. 检查实际选择是否包含水印指向的专家
        # actual_selected_experts: [batch, seq, top_k]
        # 我们看 target_expert 是否在 top_k 里

        # 扩展 target_expert 维度以便比较: [batch, seq, 1]
        target_expert_expanded = target_expert.unsqueeze(-1)

        # hit: [batch, seq] (True/False)
        hits = (actual_selected_experts == target_expert_expanded).any(dim=-1)

        # 6. 统计分数 (仅在有效样本上)
        valid_hits = hits[valid_opportunity_mask]
        score = valid_hits.float().mean().item()

        return score, "Watermarked" if score > 0.8 else "Clean"
    
    def _get_router(self):
        """
        从模型中提取路由器
        
        由于我们只替换了 forward 方法，需要从 router 对象上获取 _okr_router
        """
        # 查找所有 router 对象，检查是否有 _okr_router 属性
        for name, module in self.model.named_modules():
            # 检查是否是 router（有 _okr_router 属性）
            if hasattr(module, '_okr_router'):
                return module._okr_router
            
            # 也检查是否是 OKRRouter（直接有 secret_projection 和 gate_network）
            if hasattr(module, 'secret_projection') and hasattr(module, 'gate_network'):
                return module
        
        return None
    
    def _extract_selected_experts(self) -> Optional[torch.Tensor]:
        """
        从模型中提取实际选择的专家
        
        这需要在forward过程中保存路由信息
        """
        # 尝试从模型的属性中获取（可能是列表，取最后一个）
        if hasattr(self.model, '_okr_routing_data'):
            routing_data = self.model._okr_routing_data
            if isinstance(routing_data, list) and len(routing_data) > 0:
                # 返回最后一个（最新的）
                return routing_data[-1]
            elif isinstance(routing_data, torch.Tensor):
                return routing_data
        
        # 尝试从各个 router 层中获取
        for name, module in self.model.named_modules():
            if hasattr(module, '_selected_experts'):
                selected = module._selected_experts
                if isinstance(selected, torch.Tensor):
                    return selected
        
        return None

