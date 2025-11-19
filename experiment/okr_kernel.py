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
    2. 安全区域定义：max_logit - current_logit < epsilon
    3. 使用LSH投影（矩阵乘法）计算水印偏好
    4. 如果不在安全区，给-∞，让它滚蛋
    """
    
    def __init__(self, 
                 input_dim: int, 
                 num_experts: int, 
                 top_k: int = 2, 
                 epsilon: float = 1.5):
        """
        初始化 OKR 路由器。

        Args:
            input_dim: 输入 Embedding 的维度 (d_model)
            num_experts: 专家的总数
            top_k: 每次选几个专家 (通常是 2)
            epsilon: 质量容忍阈值 (Logit 差值)。越大水印越强，越小质量越稳。
        """
        super().__init__()
        self.top_k = top_k
        self.epsilon = epsilon

        # 标准的 Gating 层：把 Embedding 映射到专家打分
        # 这是一个可训练的层，就是原来模型里的那个 Gate
        self.gate_network = nn.Linear(input_dim, num_experts, bias=False)

        # 水印私钥 (The Secret)
        # 使用 register_buffer 告诉 PyTorch：
        # 1. 这是一个状态，要随模型保存 (save_dict)。
        # 2. 这不是参数，不要给它算梯度 (Gradient)。
        self.register_buffer(
            "secret_projection", 
            torch.randn(input_dim, num_experts)
        )

    def forward(self, hidden_states: torch.Tensor):
        """
        Forward path. 这是热路径，每一毫秒都很珍贵。

        Args:
            hidden_states: [batch_size, seq_len, input_dim]
        Returns:
            routing_weights, selected_experts
        """
        # 1. 计算原始路由分数 (Original Logits)
        # [batch, seq, num_experts]
        raw_logits = self.gate_network(hidden_states)

        # --- OKR 介入开始 ---

        # 2. 计算水印偏好 (Semantic Anchor Signal)
        # 利用矩阵乘法做 LSH 投影。
        # 这是一个纯粹的几何操作，对语义漂移鲁棒。
        watermark_bias = torch.matmul(hidden_states, self.secret_projection)

        # 3. 计算安全掩码 (Indifference Zone)
        # 只有当 (max_logit - current_logit) < epsilon 时，我们才敢乱动
        # max_logits: [batch, seq, 1]
        max_logits, _ = raw_logits.max(dim=-1, keepdim=True)
        safe_mask = raw_logits >= (max_logits - self.epsilon)

        # 4. 机会主义注入 (Fail-Open Injection)
        # 如果在安全区，听水印的 (watermark_bias)。
        # 如果不在安全区，给它一个负无穷，让它滚蛋。
        # 注意：这里不需要判断 safe_mask 的 True 数量。
        # 如果只有一个 True (只有原本的 best)，那其他的全是 -inf，结果还是原本的 best。
        # 系统自动退化为无水印模式。
        modified_scores = torch.where(
            safe_mask,
            watermark_bias, 
            torch.tensor(-1e9, device=raw_logits.device, dtype=raw_logits.dtype)
        )

        # --- OKR 介入结束 ---

        # 5. 选取 Top-K
        # 基于修改后的分数选专家
        # indices: [batch, seq, top_k]
        _, selected_experts = torch.topk(modified_scores, self.top_k, dim=-1)

        # 6. 计算权重 (Softmax)
        # 注意：为了保持原本的数值稳定性，我们在计算权重时
        # 应该回退去使用 raw_logits 中对应专家的值，还是用 modified_scores？
        # 论文里通常只改 routing (indices)，不改 weight。
        # 我们只改变"谁工作"，不改变"给多少钱"。这是为了保持梯度流稳定。

        # 从 raw_logits 中gather出被选中专家的原始分数
        router_logits = torch.gather(raw_logits, -1, selected_experts)

        # 标准 MoE 操作：Softmax 归一化
        routing_weights = F.softmax(router_logits, dim=-1)

        return routing_weights, selected_experts

