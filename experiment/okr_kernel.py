"""
OKR Kernel - 核心路由逻辑

这是整个项目的灵魂。把它想象成 Linux 的 sched.c（调度器）。
它的任务是：在微秒级别内决定该把 Token 发给哪个专家。

修复版 V2.2:
1. 集成 Safe Watermarking (归一化 + 软掩码)
2. 严格遵守 Switch Transformers 接口契约 (返回 Mask/Probs/Logits)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class OKRRouter(nn.Module):
    """
    Opportunistic Keyed Routing 路由器 (Safe Version)
    """
    
    def __init__(self, 
                 input_dim: int, 
                 num_experts: int, 
                 top_k: int = 2, 
                 epsilon: float = 1.5, # 兼容性保留，实际使用自适应逻辑
                 threshold_ratio: float = 0.9,
                 dead_zone_threshold: float = 0.01,
                 watermark_alpha: float = 0.1): # 建议 0.1 - 0.3
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.watermark_alpha = watermark_alpha
        
        # 原始 Gate 层 (复制权重用)
        self.gate_network = nn.Linear(input_dim, num_experts, bias=False)

        # 水印私钥 (Buffer, 不参与梯度更新)
        self.register_buffer("secret_projection", torch.zeros(input_dim, num_experts))
        self.register_buffer("_secret_initialized", torch.tensor(False))

    def forward(self, hidden_states: torch.Tensor):
        """
        Args:
            hidden_states: [batch_size, seq_len, input_dim]
        Returns:
            tuple(router_mask, router_probs, router_logits) -> 符合 Switch Transformers 规范
        """
        # 0. 确保数据类型一致
        if hidden_states.dtype != self.gate_network.weight.dtype:
            hidden_states = hidden_states.to(self.gate_network.weight.dtype)

        # 1. 计算原始 Logits (Raw Logits)
        raw_logits = self.gate_network(hidden_states) # [batch, seq, num_experts]

        # 2. 提取统计量 (用于自适应缩放)
        with torch.no_grad():
            # 计算 Logits 的标准差，代表模型当前的"自信程度"或"响度"
            logits_std = raw_logits.std(dim=-1, keepdim=True) + 1e-6 

        # 3. 计算水印信号 (Watermark Signal)
        # LSH 投影
        if self.secret_projection.device != hidden_states.device:
            self.secret_projection = self.secret_projection.to(hidden_states.device)
            
        watermark_raw = torch.matmul(hidden_states, self.secret_projection)
        
        # 4. 关键修复：信号归一化 (Normalization)
        # 强制将水印信号拉回 N(0, 1)，防止量级失控
        w_mean = watermark_raw.mean(dim=-1, keepdim=True)
        w_std = watermark_raw.std(dim=-1, keepdim=True) + 1e-6
        watermark_norm = (watermark_raw - w_mean) / w_std

        # 5. 机会主义软掩码 (Opportunistic Soft Mask)
        # 计算 Top-2 概率差 (Gap)
        probs = F.softmax(raw_logits, dim=-1)
        top2_probs, _ = probs.topk(2, dim=-1)
        gap = top2_probs[..., 0] - top2_probs[..., 1] # [batch, seq]
        
        # 定义犹豫区间：Gap < 0.25 时开始介入
        uncertainty_threshold = 0.25
        # Sigmoid 平滑开关：Gap 越小，Mask 越接近 1.0
        mask = torch.sigmoid(10.0 * (uncertainty_threshold - gap)).unsqueeze(-1)
        
        # 6. 注入 (Injection)
        # New = Old + Mask * Alpha * Std * Watermark
        # 始终相对于原始 Logits 的标准差进行微扰
        injection = mask * self.watermark_alpha * logits_std * watermark_norm
        
        # 安全钳位 (Safety Clamp): 绝对不允许超过 1.5 倍标准差
        max_noise = logits_std * 1.5
        injection = torch.clamp(injection, -max_noise, max_noise)
        
        final_logits = raw_logits + injection

        # --- 以下是 Switch Transformers 必须的格式转换 ---
        
        # 7. 选专家 (Top-K Selection)
        # 基于修改后的 Logits 进行选择
        # router_probs_val: [batch, seq, top_k]
        # selected_experts: [batch, seq, top_k]
        router_probs_val, selected_experts = torch.topk(final_logits, self.top_k, dim=-1)
        
        if selected_experts.dtype != torch.long:
            selected_experts = selected_experts.long()

        # 8. 计算路由权重 (Routing Weights)
        # 这一步使用 Softmax 归一化 Top-K 的值
        # 注意：我们使用 final_logits 的值，这样水印才能影响权重
        routing_weights = F.softmax(router_probs_val, dim=-1)

        # 9. 构造 router_mask (One-Hot 风格)
        # [batch, seq, num_experts]
        batch_size, seq_len, _ = hidden_states.shape
        router_mask = torch.zeros(
            (batch_size, seq_len, self.num_experts),
            device=hidden_states.device,
            dtype=hidden_states.dtype
        )
        # scatter_ 期望 src 维度匹配，这里 routing_weights 是 [batch, seq, top_k]
        router_mask.scatter_(dim=-1, index=selected_experts, src=routing_weights)

        # 10. 构造 router_probs (Sum of weights per token)
        # [batch, seq, 1] - 通常用于负载均衡损失计算
        router_probs = routing_weights.sum(dim=-1, keepdim=True)

        # 11. 构造 router_logits (Sparse Logits)
        # [batch, seq, num_experts] - 只有选中的位置有值，其他为 -inf
        router_logits_out = torch.full(
            (batch_size, seq_len, self.num_experts),
            float('-inf'),
            device=hidden_states.device,
            dtype=hidden_states.dtype
        )
        # 存入 Log Softmax 值
        logits_values = torch.log(routing_weights + 1e-9)
        router_logits_out.scatter_(dim=-1, index=selected_experts, src=logits_values)

        # 12. 保存状态供检测器使用 (Hack for Detector)
        # 这是一个副作用，为了让 okr_detector.py 能读到
        # 在 forward 内部做这个有点脏，但为了兼容现有的 patch 逻辑
        self._last_selected_experts = selected_experts

        return router_mask, router_probs, router_logits_out