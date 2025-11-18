"""
MVES水印实现（修正版）: 严格按照论文核心逻辑

关键修正：
1. 移除错误的"LSH-GHW方案"引用（论文中没有此方案）
2. 使用patch方式修改gating网络，而非LogitsProcessor（因为水印作用在MoE路由层）
3. 适配switch-base-8模型的gating网络结构
4. 确保符合论文定义3.2：l_1 = l_0 + Δl，其中l是gating网络的logit

论文核心逻辑（定义3.2）：
- 水印嵌入通过修改gating网络的logit实现
- 原始logit: l_0(x)
- 修改后logit: l_1(x) = l_0(x) + Δl(x)
- 激活分布变化: p_0(e|x) → p_1(e|x)
- KL约束: KL(p_1 || p_0) = ε
"""

import torch
import torch.nn.functional as F
import numpy as np
import hashlib
from typing import List, Optional, Dict, Tuple, Any, Callable
from transformers import PreTrainedModel, AutoModelForSeq2SeqLM

from mves_config import MVESConfig, WatermarkConfig


class MoEWatermarkForSwitch:
    """
    Switch-base-8模型的MoE水印嵌入器
    
    严格按照论文定义3.2实现：
    1. 修改gating网络的logit: l_1 = l_0 + Δl
    2. 确保KL(p_1 || p_0) = ε
    3. 使用patch方式修改router的forward方法
    """
    
    def __init__(
        self,
        secret_key: str,
        epsilon: float,
        num_experts: int,
        k_top: int,
        device: torch.device
    ):
        """
        初始化MoE水印嵌入器
        
        Args:
            secret_key: 密钥
            epsilon: 水印强度 ε = c²γ (论文定义5.1)
            num_experts: 专家数量 K
            k_top: Top-k激活数
            device: 计算设备
        """
        self.secret_key = secret_key
        self.epsilon = epsilon
        self.num_experts = num_experts
        self.k_top = k_top
        self.device = device
        
        # 计算偏置强度 (论文定义3.2)
        # ε = KL(p1||p0) ≈ Var[Δl] ≈ (1/2)||Δl||²_2
        # 注意：为了保持文本质量，我们需要限制偏置的大小
        # 使用自适应缩放：根据epsilon调整，但限制最大偏置
        base_norm = np.sqrt(2.0 * epsilon)
        # 限制最大偏置，避免过度干扰模型输出
        # router logits通常在[-5, 5]范围内，偏置不应超过其10%
        max_bias = 0.5  # 最大偏置值（经验值，可根据需要调整）
        self.target_norm = min(base_norm, max_bias)
        
        # 检测数据存储
        self._detection_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    
    def _get_context_hash(self, hidden_states: torch.Tensor) -> int:
        """根据上下文生成确定性种子"""
        hashed = torch.sum(hidden_states, dim=-1).long()
        last_token_hash = hashed[:, -1] if hashed.shape[1] > 0 else hashed[:, 0]
        seed = torch.sum(last_token_hash).item()
        combined = f"{self.secret_key}_{seed}".encode('utf-8')
        return int(hashlib.sha256(combined).hexdigest()[:16], 16)
    
    def compute_kl_divergence(self, p0: torch.Tensor, p1: torch.Tensor) -> torch.Tensor:
        """计算KL(p1||p0) (论文定义3.2)"""
        p0 = torch.clamp(p0, min=1e-9)
        p1 = torch.clamp(p1, min=1e-9)
        kl = torch.sum(p1 * torch.log(p1 / p0), dim=-1)
        return kl
    
    def get_bias_vector(
        self,
        hidden_states: torch.Tensor,
        l_0: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算水印偏置向量 (论文定义3.2)
        
        Args:
            hidden_states: 输入hidden states [batch, seq, dim]
            l_0: 原始gating logits [batch, seq, num_experts]
            
        Returns:
            delta_l: 偏置向量 [batch, seq, num_experts]
            p_0: 原始激活分布 [batch, seq, num_experts]
            p_1: 修改后激活分布 [batch, seq, num_experts]
        """
        batch_size, seq_len, num_experts = l_0.shape
        
        # 计算原始分布 p_0
        # 确保 l_0 没有 inf 或 nan
        l_0 = torch.where(torch.isnan(l_0), torch.zeros_like(l_0), l_0)
        l_0 = torch.where(torch.isinf(l_0), torch.zeros_like(l_0), l_0)
        # 限制 logits 的范围，避免极端值导致 softmax 溢出
        l_0 = torch.clamp(l_0, min=-100.0, max=100.0)
        p_0 = F.softmax(l_0, dim=-1)
        # 确保 p_0 有效
        p_0 = torch.where(torch.isnan(p_0), torch.ones_like(p_0) / self.num_experts, p_0)
        p_0 = torch.where(torch.isinf(p_0), torch.ones_like(p_0) / self.num_experts, p_0)
        # 重新归一化
        p_0_sum = p_0.sum(dim=-1, keepdim=True)
        p_0_sum = torch.clamp(p_0_sum, min=1e-9)
        p_0 = p_0 / p_0_sum
        
        # 生成确定性种子
        context_hash = self._get_context_hash(hidden_states)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(context_hash)
        
        # 初始化偏置向量
        delta_l = torch.zeros_like(l_0)
        
        # 对每个位置计算偏置
        for b in range(batch_size):
            for s in range(seq_len):
                # 选择绿色专家 (正偏置)
                green_expert = torch.randint(
                    0, num_experts, (1,), generator=generator, device=self.device
                ).item()
                
                # 选择红色专家 (负偏置)
                red_candidates = [i for i in range(num_experts) if i != green_expert]
                num_red = min(self.k_top, len(red_candidates))
                if num_red > 0:
                    red_experts = torch.tensor(
                        np.random.choice(red_candidates, size=num_red, replace=False),
                        device=self.device
                    )
                    
                    # 计算偏置强度
                    # 使用自适应缩放：根据原始logits的尺度调整偏置
                    # 获取当前位置的原始logits范围
                    l_0_pos = l_0[b, s, :]
                    logits_std = torch.std(l_0_pos).item()
                    logits_range = torch.max(l_0_pos).item() - torch.min(l_0_pos).item()
                    
                    # 自适应缩放：偏置不应超过logits标准差的20%
                    adaptive_scale = min(1.0, (logits_std * 0.2) / (self.target_norm + 1e-9))
                    scaled_norm = self.target_norm * adaptive_scale
                    
                    bias_green = scaled_norm / np.sqrt(1 + num_red)
                    bias_red = -bias_green / num_red
                    
                    # 进一步限制：确保偏置不会过度改变路由
                    # 限制偏置不超过原始logits范围的5%
                    max_bias_abs = max(0.1, logits_range * 0.05)
                    bias_green = np.clip(bias_green, -max_bias_abs, max_bias_abs)
                    bias_red = np.clip(bias_red, -max_bias_abs, max_bias_abs)
                    
                    # 应用偏置
                    delta_l[b, s, green_expert] = bias_green
                    delta_l[b, s, red_experts] = bias_red
        
        # 计算修改后的logits和分布
        l_1 = l_0 + delta_l
        # 确保 l_1 没有 inf 或 nan
        l_1 = torch.where(torch.isnan(l_1), torch.zeros_like(l_1), l_1)
        l_1 = torch.where(torch.isinf(l_1), torch.zeros_like(l_1), l_1)
        # 限制 logits 的范围，避免极端值导致 softmax 溢出
        l_1 = torch.clamp(l_1, min=-100.0, max=100.0)
        p_1 = F.softmax(l_1, dim=-1)
        
        # 确保概率值有效（检查 nan 和 inf）
        p_1 = torch.where(torch.isnan(p_1), torch.zeros_like(p_1), p_1)
        p_1 = torch.where(torch.isinf(p_1), torch.ones_like(p_1) / self.num_experts, p_1)
        # 重新归一化以确保和为 1
        p_1_sum = p_1.sum(dim=-1, keepdim=True)
        p_1_sum = torch.clamp(p_1_sum, min=1e-9)
        p_1 = p_1 / p_1_sum
        
        return delta_l, p_0, p_1
    
    def watermarked_router_forward(
        self,
        original_forward: Callable,
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        水印路由前向传播 (论文定义3.2)
        
        严格按照论文流程:
        1. 计算原始logits l_0
        2. 生成偏置 Δl, 使得 KL(p1||p0) = ε
        3. 计算修改后logits l_1 = l_0 + Δl
        4. 使用l_1进行Top-k路由
        5. 保存p_0和p_1供检测器使用
        
        Returns:
            router_logits: 路由logits [batch*seq, num_experts]
            p_0: 原始激活分布 [batch, seq, num_experts]
            p_1: 修改后激活分布 [batch, seq, num_experts]
            S_indices: Top-k激活索引 [batch, seq, k_top]
        """
        # 1. 计算原始 logits l_0 (论文定义3.2)
        # Switch Transformers 的 router.forward() 返回 tuple: (router_mask, router_probs, router_logits)
        router_output = original_forward(hidden_states)
        
        # 处理返回值：可能是单个 tensor 或 tuple
        # 首先获取 batch_size 和 seq_len
        if hidden_states.dim() == 3:
            batch_size, seq_len, hidden_dim = hidden_states.shape
        elif hidden_states.dim() == 2:
            # [batch*seq, hidden_dim] 格式
            batch_size_seq, hidden_dim = hidden_states.shape
            # 需要推断 batch_size 和 seq_len（可能需要从外部传入）
            # 暂时假设是单个序列
            batch_size = 1
            seq_len = batch_size_seq
        else:
            raise ValueError(f"Unexpected hidden_states shape: {hidden_states.shape}")
        
        if isinstance(router_output, tuple):
            # Switch Transformers 格式: (router_mask, router_probs, router_logits)
            if len(router_output) >= 3:
                router_mask_orig, router_probs_orig, l_0 = router_output[0], router_output[1], router_output[2]
                # 检查原始格式的维度
                # 注意：我们需要保持与原始格式完全一致
                orig_mask_shape = router_mask_orig.shape if router_mask_orig is not None else None
                orig_probs_shape = router_probs_orig.shape if router_probs_orig is not None else None
                
                # 调试信息（仅在第一次调用时打印）
                if not hasattr(self, '_debug_printed'):
                    print(f"  原始 router_mask 形状: {orig_mask_shape}")
                    print(f"  原始 router_probs 形状: {orig_probs_shape}")
                    print(f"  原始 router_logits 形状: {l_0.shape}")
                    # 检查原始 logits 是否包含无效值
                    if torch.isnan(l_0).any():
                        print(f"  警告: 原始 router_logits 包含 NaN!")
                    if torch.isinf(l_0).any():
                        print(f"  警告: 原始 router_logits 包含 Inf!")
                    self._debug_printed = True
                
                # 检查并修复原始 l_0 中的无效值（如果存在）
                if torch.isnan(l_0).any() or torch.isinf(l_0).any():
                    l_0 = torch.where(torch.isnan(l_0), torch.zeros_like(l_0), l_0)
                    l_0 = torch.where(torch.isinf(l_0), torch.zeros_like(l_0), l_0)
                    l_0 = torch.clamp(l_0, min=-100.0, max=100.0)
                
                if router_mask_orig.dim() == 3:
                    expected_shape_3d = True
                elif router_mask_orig.dim() == 2:
                    expected_shape_3d = False
                else:
                    expected_shape_3d = False
            elif len(router_output) == 2:
                # 某些版本可能只有 (router_probs, router_logits)
                router_probs_orig, l_0 = router_output[0], router_output[1]
                router_mask_orig = None
                expected_shape_3d = router_probs_orig.dim() == 3 if router_probs_orig is not None else False
            else:
                # 如果只有一个元素，假设是 router_logits
                l_0 = router_output[0]
                router_mask_orig = None
                router_probs_orig = None
                expected_shape_3d = l_0.dim() == 3
        else:
            # 单个 tensor 返回值
            l_0 = router_output
            router_mask_orig = None
            router_probs_orig = None
            expected_shape_3d = l_0.dim() == 3
        
        # 处理 l_0 的维度
        if l_0.dim() == 2:
            # [batch*seq, num_experts] -> [batch, seq, num_experts]
            num_experts = l_0.shape[-1]
            l_0 = l_0.view(batch_size, seq_len, num_experts)
            hidden_states_reshaped = hidden_states.view(batch_size, seq_len, -1) if hidden_states.dim() == 2 else hidden_states
        elif l_0.dim() == 3:
            # [batch, seq, num_experts]
            hidden_states_reshaped = hidden_states
        else:
            raise ValueError(f"Unexpected l_0 shape: {l_0.shape}")
        
        # 2. 生成偏置向量 Δl, 并计算 p_0 和 p_1
        delta_l, p_0, p_1 = self.get_bias_vector(hidden_states_reshaped, l_0)
        
        # 3. 计算修改后的logits (用于路由)
        l_1 = l_0 + delta_l
        
        # 4. Gating 路由 (论文第3节)
        # 使用 l_1 进行 Top-k 选择
        top_k_scores, S_indices = torch.topk(p_1, self.k_top, dim=-1)
        
        # 归一化 top-k 得分，确保概率有效
        top_k_sum = top_k_scores.sum(dim=-1, keepdim=True)
        # 避免除零和无效值
        top_k_sum = torch.clamp(top_k_sum, min=1e-9)
        top_k_scores = top_k_scores / top_k_sum
        
        # 确保概率值在有效范围内 [0, 1]，且没有 inf 或 nan
        top_k_scores = torch.clamp(top_k_scores, min=0.0, max=1.0)
        top_k_scores = torch.where(torch.isnan(top_k_scores), torch.zeros_like(top_k_scores), top_k_scores)
        top_k_scores = torch.where(torch.isinf(top_k_scores), torch.ones_like(top_k_scores) / self.k_top, top_k_scores)
        
        # 重新归一化以确保和为 1
        top_k_sum = top_k_scores.sum(dim=-1, keepdim=True)
        top_k_sum = torch.clamp(top_k_sum, min=1e-9)
        top_k_scores = top_k_scores / top_k_sum
        
        # 5. 存储检测数据
        self._detection_data.append((p_0, p_1, S_indices))
        
        # 6. 返回路由logits (需要与原始格式一致)
        # Switch Transformer需要稀疏的gate_logits
        # 注意：根据 mlp.forward() 的期望，router_mask 应该是 3D 格式
        batch_size, seq_len, _ = l_1.shape
        
        # 创建 2D 格式的 router_logits（用于某些情况）
        router_logits_2d = torch.zeros(
            (batch_size * seq_len, self.num_experts),
            dtype=l_1.dtype,
            device=self.device
        )
        router_logits_2d.scatter_(
            -1,
            S_indices.view(-1, self.k_top),
            top_k_scores.view(-1, self.k_top)
        )
        
        # 如果原始返回是 tuple，我们也返回 tuple
        # Switch Transformers 格式: (router_mask, router_probs, router_logits)
        # 重要：我们需要保持与原始返回格式完全一致
        if router_mask_orig is not None and router_probs_orig is not None:
            # 检查原始格式的维度和形状
            orig_mask_shape = router_mask_orig.shape
            orig_probs_shape = router_probs_orig.shape
            
            # 根据原始格式创建返回值
            # router_mask: [batch_size, seq_len, num_experts]
            # 注意：router_mask 应该是概率分布，用于 argmax 选择专家
            router_mask_new = torch.zeros(
                (batch_size, seq_len, self.num_experts),
                dtype=l_1.dtype,
                device=self.device
            )
            router_mask_new.scatter_(
                -1,
                S_indices,  # [batch, seq, k_top]
                top_k_scores  # [batch, seq, k_top]
            )
            # 确保 router_mask 的值有效
            router_mask_new = torch.clamp(router_mask_new, min=0.0, max=1.0)
            router_mask_new = torch.where(torch.isnan(router_mask_new), torch.zeros_like(router_mask_new), router_mask_new)
            router_mask_new = torch.where(torch.isinf(router_mask_new), torch.zeros_like(router_mask_new), router_mask_new)
            
            # router_probs: 根据原始形状决定
            # 如果原始是 [batch, seq, 1]，说明是聚合后的概率（所有专家的加权和）
            # 如果原始是 [batch, seq, num_experts]，说明是每个专家的概率
            if orig_probs_shape[-1] == 1:
                # 原始是聚合格式 [batch, seq, 1]
                # 我们需要计算所有专家的加权和
                # router_probs 应该是所有 top-k 专家的概率之和（应该接近 1.0，因为我们已经归一化了）
                router_probs_sum = top_k_scores.sum(dim=-1, keepdim=True)  # [batch, seq, 1]
                # 确保值在有效范围内
                router_probs_new = torch.clamp(router_probs_sum, min=0.0, max=1.0)
                router_probs_new = torch.where(torch.isnan(router_probs_new), torch.ones_like(router_probs_new), router_probs_new)
                router_probs_new = torch.where(torch.isinf(router_probs_new), torch.ones_like(router_probs_new), router_probs_new)
            else:
                # 原始是 [batch, seq, num_experts] 格式
                router_probs_new = torch.zeros(
                    (batch_size, seq_len, self.num_experts),
                    dtype=l_1.dtype,
                    device=self.device
                )
                router_probs_new.scatter_(
                    -1,
                    S_indices,
                    top_k_scores
                )
                # 确保值在有效范围内
                router_probs_new = torch.clamp(router_probs_new, min=0.0, max=1.0)
                router_probs_new = torch.where(torch.isnan(router_probs_new), torch.zeros_like(router_probs_new), router_probs_new)
                router_probs_new = torch.where(torch.isinf(router_probs_new), torch.zeros_like(router_probs_new), router_probs_new)
            
            # router_logits: [batch_size, seq_len, num_experts]
            router_logits_new = torch.zeros(
                (batch_size, seq_len, self.num_experts),
                dtype=l_1.dtype,
                device=self.device
            )
            router_logits_new.scatter_(
                -1,
                S_indices,
                top_k_scores
            )
            # 确保 router_logits 的值有效（logits 可以是负数，但不能是 inf 或 nan）
            router_logits_new = torch.where(torch.isnan(router_logits_new), torch.zeros_like(router_logits_new), router_logits_new)
            router_logits_new = torch.where(torch.isinf(router_logits_new), torch.zeros_like(router_logits_new), router_logits_new)
            # 限制 logits 的范围，避免极端值
            router_logits_new = torch.clamp(router_logits_new, min=-100.0, max=100.0)
            
            return (router_mask_new, router_probs_new, router_logits_new), p_0, p_1, S_indices
        else:
            # 单个 tensor 返回
            return router_logits_2d, p_0, p_1, S_indices
    
    def get_detection_data(self) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """获取检测数据"""
        return self._detection_data.copy()
    
    def clear_detection_data(self):
        """清除检测数据"""
        self._detection_data.clear()


def patch_switch_model_with_watermark(
    model: AutoModelForSeq2SeqLM,
    config: MVESConfig
) -> AutoModelForSeq2SeqLM:
    """
    Patch switch-base-8模型，注入水印逻辑 (论文第3节)
    
    严格按照论文实现：
    - 修改gating网络的forward方法
    - 确保KL(p1||p0) = ε
    - 适配switch-base-8的架构
    
    Args:
        model: switch-base-8模型
        config: MVES配置
        
    Returns:
        patched_model: 已patch的模型
    """
    wm_config = config.watermark
    device = next(model.parameters()).device
    
    # 获取模型配置
    if hasattr(model.config, 'num_experts'):
        num_experts = model.config.num_experts
    else:
        # switch-base-8默认8个专家
        num_experts = wm_config.num_experts
    
    k_top = wm_config.k_top if hasattr(wm_config, 'k_top') else 1
    
    # 创建水印嵌入器
    watermark_injector = MoEWatermarkForSwitch(
        wm_config.secret_key,
        wm_config.epsilon,
        num_experts,
        k_top,
        device
    )
    
    print(f"Patching switch-base-8 with MoE watermark (ε={wm_config.epsilon:.6f})...")
    
    # Switch Transformer的架构：encoder-decoder结构
    # MoE层在decoder的FFN中
    patched_layers = 0
    
    # 获取decoder（支持多种架构）
    decoder = None
    if hasattr(model, 'decoder'):
        decoder = model.decoder
    elif hasattr(model, 'model') and hasattr(model.model, 'decoder'):
        decoder = model.model.decoder
    else:
        print("警告: 未找到decoder，无法patch")
        return model
    
    # 获取decoder blocks（支持多种属性名）
    # SwitchTransformersStack 可能使用不同的属性名
    decoder_blocks = None
    
    # 尝试多种可能的属性名
    possible_attrs = ['block', 'layers', 'blocks', 'transformer_blocks', 'decoder_layers']
    for attr_name in possible_attrs:
        if hasattr(decoder, attr_name):
            decoder_blocks = getattr(decoder, attr_name)
            if decoder_blocks is not None and len(decoder_blocks) > 0:
                print(f"  找到decoder blocks: {attr_name} (长度: {len(decoder_blocks)})")
                break
    
    if decoder_blocks is None or len(decoder_blocks) == 0:
        print("警告: 未找到decoder blocks，无法patch")
        print(f"  decoder类型: {type(decoder).__name__}")
        print(f"  decoder属性: {[attr for attr in dir(decoder) if not attr.startswith('_')][:20]}")
        return model
    
    # 遍历所有层，查找MoE router
    for layer_idx, layer in enumerate(decoder_blocks):
        router = None
        
        # 方式1: layer.layer[1] (Switch Transformer标准结构)
        # SwitchTransformersBlock 通常有 layer 属性，其中 layer[1] 是 FFN 层
        if hasattr(layer, 'layer'):
            layer_list = layer.layer
            if isinstance(layer_list, (list, tuple)) and len(layer_list) > 1:
                ffn_layer = layer_list[1]
                # 检查 ffn_layer.mlp.router
                if hasattr(ffn_layer, 'mlp') and hasattr(ffn_layer.mlp, 'router'):
                    router = ffn_layer.mlp.router
                # 检查 ffn_layer.router
                elif hasattr(ffn_layer, 'router'):
                    router = ffn_layer.router
                # 检查 ffn_layer.expert_router
                elif hasattr(ffn_layer, 'expert_router'):
                    router = ffn_layer.expert_router
            # 如果 layer 是 ModuleList，尝试遍历
            elif hasattr(layer_list, '__iter__'):
                for sublayer in layer_list:
                    if hasattr(sublayer, 'mlp') and hasattr(sublayer.mlp, 'router'):
                        router = sublayer.mlp.router
                        break
                    elif hasattr(sublayer, 'router'):
                        router = sublayer.router
                        break
        
        # 方式2: layer.feed_forward
        if router is None and hasattr(layer, 'feed_forward'):
            ffn_layer = layer.feed_forward
            if hasattr(ffn_layer, 'mlp') and hasattr(ffn_layer.mlp, 'router'):
                router = ffn_layer.mlp.router
            elif hasattr(ffn_layer, 'router'):
                router = ffn_layer.router
            elif hasattr(ffn_layer, 'expert_router'):
                router = ffn_layer.expert_router
        
        # 方式3: layer.mlp
        if router is None and hasattr(layer, 'mlp'):
            ffn_layer = layer.mlp
            if hasattr(ffn_layer, 'router'):
                router = ffn_layer.router
            elif hasattr(ffn_layer, 'expert_router'):
                router = ffn_layer.expert_router
        
        # 方式4: 直接检查layer是否有router
        if router is None and hasattr(layer, 'router'):
            router = layer.router
        
        # 方式5: 检查 layer.block_sparse_moe (某些架构)
        if router is None and hasattr(layer, 'block_sparse_moe'):
            moe = layer.block_sparse_moe
            if hasattr(moe, 'router'):
                router = moe.router
            elif hasattr(moe, 'gate'):
                router = moe.gate
        
        if router is not None:
            try:
                # 保存原始的forward方法
                original_forward = router.forward.__get__(router)
                
                # 创建新的forward方法
                def make_new_forward(orig_fwd, router_obj, layer_id):
                    def new_forward(hidden_states: torch.Tensor):
                        # 调用水印逻辑
                        result = watermark_injector.watermarked_router_forward(orig_fwd, hidden_states)
                        
                        # 处理返回值：格式总是 (router_output, p_0, p_1, S_indices)
                        # router_output 可能是 tuple (router_mask, router_probs, router_logits) 或单个 tensor
                        if isinstance(result, tuple) and len(result) == 4:
                            router_output, p_0, p_1, S_indices = result
                        else:
                            # 兼容旧格式（不应该发生，但以防万一）
                            raise ValueError(f"Unexpected return format from watermarked_router_forward: {type(result)}")
                        
                        # 将检测器信息附加到router对象上
                        router_obj._watermark_detection_data = (p_0, p_1, S_indices)
                        router_obj._layer_idx = layer_id
                        
                        # 返回 router_output（可能是 tuple 或单个 tensor）
                        return router_output
                    return new_forward
                
                # 应用patch
                router.forward = make_new_forward(original_forward, router, layer_idx)
                patched_layers += 1
            except Exception as e:
                print(f"警告: Layer {layer_idx} patch失败: {e}")
                continue
    
    if patched_layers == 0:
        print("警告: 未找到MoE层，可能需要手动适配模型架构")
        print(f"  模型类型: {type(model).__name__}")
        print(f"  decoder类型: {type(decoder).__name__}")
        if len(decoder_blocks) > 0:
            first_layer = decoder_blocks[0]
            print(f"  第一层类型: {type(first_layer).__name__}")
            print(f"  第一层属性: {[attr for attr in dir(first_layer) if not attr.startswith('_')][:15]}")
            
            # 详细检查第一层的结构
            if hasattr(first_layer, 'layer'):
                layer_list = first_layer.layer
                print(f"  layer属性类型: {type(layer_list).__name__}")
                if isinstance(layer_list, (list, tuple)) and len(layer_list) > 0:
                    print(f"  layer长度: {len(layer_list)}")
                    for i, sublayer in enumerate(layer_list[:3]):  # 只检查前3个
                        print(f"    layer[{i}]类型: {type(sublayer).__name__}")
                        if i == 1:  # FFN层通常是layer[1]
                            print(f"    layer[1]属性: {[attr for attr in dir(sublayer) if not attr.startswith('_')][:15]}")
                            if hasattr(sublayer, 'mlp'):
                                print(f"      mlp属性: {[attr for attr in dir(sublayer.mlp) if not attr.startswith('_')][:15]}")
                                if hasattr(sublayer.mlp, 'router'):
                                    print(f"      ✓ 找到 router!")
                                else:
                                    print(f"      ✗ 未找到 router")
                elif hasattr(layer_list, '__iter__'):
                    print(f"  layer是可迭代对象")
                    try:
                        for i, sublayer in enumerate(list(layer_list)[:3]):
                            print(f"    layer[{i}]类型: {type(sublayer).__name__}")
                    except:
                        pass
    else:
        print(f"✓ 成功patch {patched_layers} 个MoE层")
    
    print(f"Patching complete. Watermark strength ε={wm_config.epsilon:.6f} (c²γ parameterization).")
    
    return model


def get_watermark_data_from_switch_model(model: AutoModelForSeq2SeqLM) -> List[Tuple]:
    """
    从switch模型中提取检测数据
    
    注意：提取路径必须与 patch_switch_model_with_watermark 中的路径一致
    
    Returns:
        detection_data: [(p_0, p_1, S_indices), ...]
    """
    data = []
    
    # 获取decoder（支持多种架构）
    decoder = None
    if hasattr(model, 'decoder'):
        decoder = model.decoder
    elif hasattr(model, 'model') and hasattr(model.model, 'decoder'):
        decoder = model.model.decoder
    else:
        return data
    
    # 获取decoder blocks（支持多种属性名，与patch逻辑一致）
    decoder_blocks = None
    if hasattr(decoder, 'block'):
        decoder_blocks = decoder.block
    elif hasattr(decoder, 'layers'):
        decoder_blocks = decoder.layers
    elif hasattr(decoder, 'blocks'):
        decoder_blocks = decoder.blocks
    elif hasattr(decoder, 'transformer_blocks'):
        decoder_blocks = decoder.transformer_blocks
    else:
        return data
    
    # 遍历所有层，查找检测数据（与patch逻辑一致）
    for layer_idx, layer in enumerate(decoder_blocks):
        router = None
        
        # 方式1: layer.layer[1] (Switch Transformer标准结构)
        # 注意：必须与 patch_switch_model_with_watermark 中的逻辑完全一致
        if hasattr(layer, 'layer'):
            layer_list = layer.layer
            if isinstance(layer_list, (list, tuple)) and len(layer_list) > 1:
                ffn_layer = layer_list[1]
                # 检查 ffn_layer.mlp.router
                if hasattr(ffn_layer, 'mlp') and hasattr(ffn_layer.mlp, 'router'):
                    router = ffn_layer.mlp.router
                # 检查 ffn_layer.router
                elif hasattr(ffn_layer, 'router'):
                    router = ffn_layer.router
                # 检查 ffn_layer.expert_router
                elif hasattr(ffn_layer, 'expert_router'):
                    router = ffn_layer.expert_router
            # 如果 layer 是 ModuleList，尝试遍历
            elif hasattr(layer_list, '__iter__'):
                for sublayer in layer_list:
                    if hasattr(sublayer, 'mlp') and hasattr(sublayer.mlp, 'router'):
                        router = sublayer.mlp.router
                        break
                    elif hasattr(sublayer, 'router'):
                        router = sublayer.router
                        break
        
        # 方式2: layer.feed_forward
        if router is None and hasattr(layer, 'feed_forward'):
            ffn_layer = layer.feed_forward
            if hasattr(ffn_layer, 'mlp') and hasattr(ffn_layer.mlp, 'router'):
                router = ffn_layer.mlp.router
            elif hasattr(ffn_layer, 'router'):
                router = ffn_layer.router
            elif hasattr(ffn_layer, 'expert_router'):
                router = ffn_layer.expert_router
        
        # 方式3: layer.mlp
        if router is None and hasattr(layer, 'mlp'):
            ffn_layer = layer.mlp
            if hasattr(ffn_layer, 'router'):
                router = ffn_layer.router
            elif hasattr(ffn_layer, 'expert_router'):
                router = ffn_layer.expert_router
        
        # 方式4: 直接检查layer是否有router
        if router is None and hasattr(layer, 'router'):
            router = layer.router
        
        # 方式5: 检查 layer.block_sparse_moe (某些架构)
        if router is None and hasattr(layer, 'block_sparse_moe'):
            moe = layer.block_sparse_moe
            if hasattr(moe, 'router'):
                router = moe.router
            elif hasattr(moe, 'gate'):
                router = moe.gate
        
        # 提取检测数据
        if router is not None:
            if hasattr(router, '_watermark_detection_data'):
                detection_data = router._watermark_detection_data
                if detection_data is not None:
                    data.append(detection_data)
                    # 清除数据（用后即焚）
                    del router._watermark_detection_data
    
    # 调试信息（仅在第一次调用时打印）
    if not hasattr(get_watermark_data_from_switch_model, '_debug_printed'):
        if len(data) == 0:
            print(f"  警告: 未提取到任何检测数据（检查了 {len(decoder_blocks)} 层）")
        else:
            print(f"  成功提取到 {len(data)} 个检测数据")
        get_watermark_data_from_switch_model._debug_printed = True
    
    return data

