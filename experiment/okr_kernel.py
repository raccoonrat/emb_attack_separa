"""
OKR Kernel - 核心路由逻辑

这是整个项目的灵魂。把它想象成 Linux 的 sched.c（调度器）。
它的任务是：在微秒级别内决定该把 Token 发给哪个专家。

没有 for 循环，没有列表，没有 CPU 同步。全是纯粹的 CUDA 友好的张量运算。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OKRRouter(nn.Module):
    """
    Opportunistic Keyed Routing 路由器
    
    核心思想：
    1. 只在安全区域（indifference zone）内修改路由
    2. 安全区域定义：基于概率比率的自适应阈值（probability > max_prob * threshold_ratio）
    3. 使用LSH投影（矩阵乘法）计算水印偏好
    4. 如果不在安全区，保持原始logits
    5. 引入死区（Dead Zone）避免LSH零点噪声
    """
    
    def __init__(self, 
                 input_dim: int, 
                 num_experts: int, 
                 top_k: int = 2, 
                 epsilon: float = 1.5,
                 threshold_ratio: float = 0.9,
                 dead_zone_threshold: float = 0.01,
                 watermark_alpha: float = 0.1):
        """
        初始化 OKR 路由器。

        Args:
            input_dim: 输入 Embedding 的维度 (d_model)
            num_experts: 专家的总数
            top_k: 每次选几个专家 (通常是 2)
            epsilon: 质量容忍阈值 (Logit 差值，向后兼容，已弃用)
            threshold_ratio: 概率比率阈值（默认0.9，即 p_top2 / p_top1 > 0.9 时介入）
            dead_zone_threshold: LSH死区阈值（默认0.01，abs(dot_product) < 此值时不介入）
            watermark_alpha: 水印混合系数（默认0.1，raw_logits + alpha * watermark_bias）
        """
        super().__init__()
        self.top_k = top_k
        self.epsilon = epsilon  # 保留用于向后兼容，但不再使用
        self.threshold_ratio = threshold_ratio
        self.dead_zone_threshold = dead_zone_threshold
        self.watermark_alpha = watermark_alpha

        # 标准的 Gating 层：把 Embedding 映射到专家打分
        # 这是一个可训练的层，就是原来模型里的那个 Gate
        self.gate_network = nn.Linear(input_dim, num_experts, bias=False)

        # 水印私钥 (The Secret)
        # 使用 register_buffer 告诉 PyTorch：
        # 1. 这是一个状态，要随模型保存 (save_dict)。
        # 2. 这不是参数，不要给它算梯度 (Gradient)。
        # 注意：初始化应该在外部通过 _initialize_secret_projection 完成，确保一致性
        self.register_buffer(
            "secret_projection", 
            torch.zeros(input_dim, num_experts)  # 初始化为零，必须通过外部初始化
        )
        
        # 标记是否已初始化（用于一致性检查）
        self.register_buffer("_secret_initialized", torch.tensor(False))

    def forward(self, hidden_states: torch.Tensor):
        """
        Forward path. 这是热路径，每一毫秒都很珍贵。

        Args:
            hidden_states: [batch_size, seq_len, input_dim]
        Returns:
            routing_weights, selected_experts
        """
        # 检查 secret_projection 是否已初始化
        if not self._secret_initialized.item():
            raise RuntimeError(
                "secret_projection 未初始化！必须通过 _initialize_secret_projection "
                "或 inject_okr(secret_key=...) 进行初始化，以确保分布式一致性。"
            )
        
        # 1. 计算原始路由分数 (Original Logits)
        # [batch, seq, num_experts]
        raw_logits = self.gate_network(hidden_states)

        # --- OKR 介入开始 ---

        # 2. 计算水印偏好 (Semantic Anchor Signal)
        # 利用矩阵乘法做 LSH 投影。
        # 这是一个纯粹的几何操作，对语义漂移鲁棒。
        # 确保 secret_projection 在正确的设备上
        if self.secret_projection.device != hidden_states.device:
            self.secret_projection = self.secret_projection.to(hidden_states.device)
        
        watermark_bias = torch.matmul(hidden_states, self.secret_projection)
        
        # 2.1 死区检查 (Dead Zone Check)
        # 如果 LSH 投影接近零，信号不可靠，不介入
        watermark_bias_abs = torch.abs(watermark_bias)
        watermark_mask = watermark_bias_abs >= self.dead_zone_threshold
        
        # 3. 计算安全掩码 (Indifference Zone) - 基于概率比率
        # 使用 softmax 后的概率，而不是 logits 的绝对值
        raw_probs = F.softmax(raw_logits, dim=-1)  # [batch, seq, num_experts]
        
        # 获取 top-2 的概率
        top2_probs, top2_indices = torch.topk(raw_probs, k=min(2, self.num_experts), dim=-1)
        top1_probs = top2_probs[:, :, 0:1]  # [batch, seq, 1]
        top2_probs_values = top2_probs[:, :, 1:2] if top2_probs.size(-1) > 1 else top1_probs * 0  # [batch, seq, 1]
        
        # 安全区域：top2_prob / top1_prob >= threshold_ratio
        # 这意味着两个专家在概率上很接近，可以安全地切换
        prob_ratio = top2_probs_values / (top1_probs + 1e-9)  # 避免除零
        safe_mask = prob_ratio >= self.threshold_ratio  # [batch, seq, 1]
        safe_mask = safe_mask.expand_as(raw_logits)  # [batch, seq, num_experts]

        # 4. 归一化 watermark_bias 到与 raw_logits 相似的量级
        # 使用标准差进行归一化，避免量级不匹配
        raw_logits_std = torch.std(raw_logits, dim=-1, keepdim=True)  # [batch, seq, 1]
        watermark_bias_std = torch.std(watermark_bias, dim=-1, keepdim=True)  # [batch, seq, 1]
        
        # 归一化：将 watermark_bias 缩放到与 raw_logits 相似的量级
        scale_factor = raw_logits_std / (watermark_bias_std + 1e-9)
        normalized_watermark_bias = watermark_bias * scale_factor
        
        # 5. 机会主义注入 (Fail-Open Injection)
        # 在安全区内，使用 raw_logits + alpha * normalized_watermark_bias
        # 在安全区外，保持原始 raw_logits
        # 同时考虑死区：如果 watermark_bias 太小，也不介入
        combined_mask = safe_mask & watermark_mask  # [batch, seq, num_experts]
        
        # 使用 in-place 操作优化性能
        modified_scores = raw_logits.clone()  # 克隆以避免修改原始 logits
        
        # 在安全区内，添加归一化的水印偏差
        watermark_contribution = self.watermark_alpha * normalized_watermark_bias
        modified_scores = torch.where(
            combined_mask,
            modified_scores + watermark_contribution,
            modified_scores
        )

        # --- OKR 介入结束 ---

        # 6. 选取 Top-K
        # 基于修改后的分数选专家
        # indices: [batch, seq, top_k]
        _, selected_experts = torch.topk(modified_scores, self.top_k, dim=-1)

        # 7. 计算权重 (Softmax)
        # 注意：为了保持原本的数值稳定性，我们在计算权重时
        # 应该回退去使用 raw_logits 中对应专家的值，还是用 modified_scores？
        # 论文里通常只改 routing (indices)，不改 weight。
        # 我们只改变"谁工作"，不改变"给多少钱"。这是为了保持梯度流稳定。

        # 从 raw_logits 中gather出被选中专家的原始分数
        router_logits = torch.gather(raw_logits, -1, selected_experts)

        # 标准 MoE 操作：Softmax 归一化
        routing_weights = F.softmax(router_logits, dim=-1)

        return routing_weights, selected_experts
    
    @property
    def num_experts(self):
        """返回专家数量（用于兼容性）"""
        return self.gate_network.out_features

