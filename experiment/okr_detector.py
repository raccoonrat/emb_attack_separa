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

    def detect(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        对一段文本进行检测。
        
        Args:
            input_ids: 输入token IDs [batch, seq_len]
            attention_mask: 注意力掩码 [batch, seq_len]
            
        Returns:
            score: 水印命中率 (0-1)
            verdict: "Watermarked" 或 "Clean"
        """
        # 1. 跑一遍模型，Hook 住 Router 的输入输出
        # 这里假设我们能拿到每一层的 hidden_states 和 router 的选择
        # 在实际工程中，这需要在此处注册 PyTorch Hook
        # 为了演示，我们简化为处理单层数据
        
        # 获取模型的hidden states
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]  # 最后一层的hidden states
        
        # 获取实际选择的专家（需要从模型中提取）
        # 这里假设模型已经保存了路由信息
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
        
        这是一个启发式方法，需要根据具体模型调整
        """
        # 如果没找到，尝试从第一层MoE获取
        for name, module in self.model.named_modules():
            # 检查是否是OKRRouter（通过检查是否有secret_projection和gate_network）
            if hasattr(module, 'secret_projection') and hasattr(module, 'gate_network'):
                return module
        
        return None
    
    def _extract_selected_experts(self) -> Optional[torch.Tensor]:
        """
        从模型中提取实际选择的专家
        
        这需要在forward过程中保存路由信息
        """
        # 尝试从模型的属性中获取
        if hasattr(self.model, '_okr_routing_data'):
            return self.model._okr_routing_data
        
        # 尝试从各个层中获取
        for name, module in self.model.named_modules():
            if hasattr(module, '_selected_experts'):
                return module._selected_experts
        
        return None

