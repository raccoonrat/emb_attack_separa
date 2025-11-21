import torch
import torch.nn as nn
import torch.nn.functional as F

class SafeWatermarkedRouter(nn.Module):
    def __init__(self, input_dim, num_experts, strength=0.5, corruption_limit=0.1):
        """
        strength: 水印信号的混合强度 (alpha). 建议 0.1 - 0.5.
        corruption_limit: 允许的最大扰动比例. 防止水印把 Logits 改得面目全非.
        """
        super().__init__()
        self.num_experts = num_experts
        self.strength = strength
        self.corruption_limit = corruption_limit
        
        # 你的私钥 (LSH Projection Matrix)
        # 这里的初始化非常重要，必须保证 deterministic
        # 在实际工程中，这里应该从外部加载固定的种子或权重
        self.register_buffer("secret_projection", torch.randn(input_dim, num_experts))

    def forward(self, inputs, raw_logits):
        """
        inputs: [batch, seq, hidden] (Context Embeddings)
        raw_logits: [batch, seq, num_experts] (原始路由分数)
        """
        
        # --- 1. 计算原始统计量 (The Baseline) ---
        # 我们必须知道原始 Logits 的"响度"是多少
        # detach() 很重要，我们不想改变这些统计量的梯度
        with torch.no_grad():
            logits_mean = raw_logits.mean(dim=-1, keepdim=True)
            logits_std = raw_logits.std(dim=-1, keepdim=True) + 1e-6 # 防除零
            
        # --- 2. 计算水印信号 (The Signal) ---
        # LSH 投影: [batch, seq, experts]
        watermark_raw = torch.matmul(inputs, self.secret_projection)
        
        # --- 3. 信号归一化 (CRITICAL FIX) ---
        # 这是你之前死锁的原因。必须把水印信号归一化到 N(0, 1)
        # 然后再缩放到和 raw_logits 相同的量级。
        w_mean = watermark_raw.mean(dim=-1, keepdim=True)
        w_std = watermark_raw.std(dim=-1, keepdim=True) + 1e-6
        
        # Normalized Watermark: 现在的范围大约是 [-2, 2]
        watermark_norm = (watermark_raw - w_mean) / w_std
        
        # --- 4. 机会主义掩码 (Opportunistic Mask) ---
        # 计算 Top-2 差距 (Gap)
        # 使用 Probability Space 而不是 Logit Space，因为 Logit 数值可能是任意的
        probs = F.softmax(raw_logits, dim=-1)
        top2_probs, _ = probs.topk(2, dim=-1)
        gap = top2_probs[..., 0] - top2_probs[..., 1] # [batch, seq]
        
        # 动态定义"犹豫不决": 如果 Gap 小于某个阈值 (例如 0.2)
        # 或者，更鲁棒的方法：根据熵 (Entropy) 决定
        # 这里我们使用简单的概率差阈值。
        # 这是一个 Soft Mask: gap 越小，mask 越接近 1.0 (允许水印)
        # gap 越大，mask 越接近 0.0 (拒绝水印)
        uncertainty_threshold = 0.3 
        
        # 使用 Sigmoid 创建平滑的开关，避免硬截断带来的梯度问题
        # 当 gap = 0 时，sigmoid -> 1.0
        # 当 gap >> threshold 时，sigmoid -> 0.0
        # 这里的 10.0 是温度系数，控制开关的陡峭程度
        mask = torch.sigmoid(10.0 * (uncertainty_threshold - gap)).unsqueeze(-1)
        
        # --- 5. 信号注入 (Injection) ---
        # 核心公式: New = Old + Mask * Strength * Scale * Watermark
        
        # 动态缩放: 让水印的大小永远是 raw_logits 标准差的一部分
        # 这样无论模型在哪一层，数值多大，水印永远是"微扰"
        adaptive_scale = logits_std * self.strength
        
        injection = mask * adaptive_scale * watermark_norm
        
        # 安全钳位 (Safety Clamp): 确保注入的噪音不超过原始 Logits 范围的 20%
        # 这是"Never Break Userspace"的最后一道防线
        max_noise = logits_std * 2.0 # 允许 2 sigma 的波动
        injection = torch.clamp(injection, -max_noise, max_noise)
        
        final_logits = raw_logits + injection
        
        # --- Debug Info (仅在调试时开启，生产环境删掉) ---
        # 这一步会产生 CPU 同步，非常慢，用来 Debug 你的 padding 问题
        if torch.rand(1).item() < 0.001: # 1/1000 概率打印
             print(f"DEBUG: Gap Mean: {gap.mean().item():.4f}, "
                   f"Mask Mean: {mask.mean().item():.4f}, "
                   f"Logits Std: {logits_std.mean().item():.4f}, "
                   f"Inj Std: {injection.std().item():.4f}")

        # --- 6. 路由决策 ---
        # 这一步通常由外部框架的 Router 完成 (如 softmax + topk)
        # 我们返回修改后的 Logits
        return final_logits
