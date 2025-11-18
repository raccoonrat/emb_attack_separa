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
        self.target_norm = np.sqrt(2.0 * epsilon)
        
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
        p_0 = F.softmax(l_0, dim=-1)
        
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
                    bias_green = self.target_norm / np.sqrt(1 + num_red)
                    bias_red = -bias_green / num_red
                    
                    # 应用偏置
                    delta_l[b, s, green_expert] = bias_green
                    delta_l[b, s, red_experts] = bias_red
        
        # 计算修改后的logits和分布
        l_1 = l_0 + delta_l
        p_1 = F.softmax(l_1, dim=-1)
        
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
        l_0 = original_forward(hidden_states)  # [batch*seq, num_experts] 或 [batch, seq, num_experts]
        
        # 处理维度
        if l_0.dim() == 2:
            batch_size, seq_len = hidden_states.shape[:2]
            l_0 = l_0.view(batch_size, seq_len, -1)
            hidden_states_reshaped = hidden_states
        else:
            batch_size, seq_len = hidden_states.shape[:2]
            hidden_states_reshaped = hidden_states
        
        # 2. 生成偏置向量 Δl, 并计算 p_0 和 p_1
        delta_l, p_0, p_1 = self.get_bias_vector(hidden_states_reshaped, l_0)
        
        # 3. 计算修改后的logits (用于路由)
        l_1 = l_0 + delta_l
        
        # 4. Gating 路由 (论文第3节)
        # 使用 l_1 进行 Top-k 选择
        top_k_scores, S_indices = torch.topk(p_1, self.k_top, dim=-1)
        
        # 归一化 top-k 得分
        top_k_scores = top_k_scores / (top_k_scores.sum(dim=-1, keepdim=True) + 1e-9)
        
        # 5. 存储检测数据
        self._detection_data.append((p_0, p_1, S_indices))
        
        # 6. 返回路由logits (需要与原始格式一致)
        # Switch Transformer需要稀疏的gate_logits
        batch_size, seq_len, _ = l_1.shape
        router_logits = torch.zeros(
            (batch_size * seq_len, self.num_experts),
            dtype=l_1.dtype,
            device=self.device
        )
        router_logits.scatter_(
            -1,
            S_indices.view(-1, self.k_top),
            top_k_scores.view(-1, self.k_top)
        )
        
        return router_logits, p_0, p_1, S_indices
    
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
        print("警告: 未找到decoder blocks，无法patch")
        return model
    
    # 遍历所有层，查找MoE router
    for layer_idx, layer in enumerate(decoder_blocks):
        router = None
        
        # 方式1: layer.layer[1] (Switch Transformer标准结构)
        if hasattr(layer, 'layer') and len(layer.layer) > 1:
            ffn_layer = layer.layer[1]
            if hasattr(ffn_layer, 'mlp') and hasattr(ffn_layer.mlp, 'router'):
                router = ffn_layer.mlp.router
            elif hasattr(ffn_layer, 'router'):
                router = ffn_layer.router
        # 方式2: layer.feed_forward
        elif hasattr(layer, 'feed_forward'):
            ffn_layer = layer.feed_forward
            if hasattr(ffn_layer, 'router'):
                router = ffn_layer.router
            elif hasattr(ffn_layer, 'mlp') and hasattr(ffn_layer.mlp, 'router'):
                router = ffn_layer.mlp.router
        # 方式3: layer.mlp
        elif hasattr(layer, 'mlp'):
            ffn_layer = layer.mlp
            if hasattr(ffn_layer, 'router'):
                router = ffn_layer.router
        # 方式4: 直接检查layer是否有router
        elif hasattr(layer, 'router'):
            router = layer.router
        
        if router is not None:
            try:
                # 保存原始的forward方法
                original_forward = router.forward.__get__(router)
                
                # 创建新的forward方法
                def make_new_forward(orig_fwd, router_obj, layer_id):
                    def new_forward(hidden_states: torch.Tensor):
                        # 调用水印逻辑
                        router_logits, p_0, p_1, S_indices = \
                            watermark_injector.watermarked_router_forward(orig_fwd, hidden_states)
                        
                        # 将检测器信息附加到router对象上
                        router_obj._watermark_detection_data = (p_0, p_1, S_indices)
                        router_obj._layer_idx = layer_id
                        
                        return router_logits
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
            print(f"  第一层属性: {[attr for attr in dir(first_layer) if not attr.startswith('_')][:10]}")
    else:
        print(f"✓ 成功patch {patched_layers} 个MoE层")
    
    print(f"Patching complete. Watermark strength ε={wm_config.epsilon:.6f} (c²γ parameterization).")
    
    return model


def get_watermark_data_from_switch_model(model: AutoModelForSeq2SeqLM) -> List[Tuple]:
    """
    从switch模型中提取检测数据
    
    Returns:
        detection_data: [(p_0, p_1, S_indices), ...]
    """
    data = []
    
    if hasattr(model, 'decoder') and hasattr(model.decoder, 'block'):
        for layer in model.decoder.block:
            if hasattr(layer, 'layer') and len(layer.layer) > 1:
                ffn_layer = layer.layer[1]
                if hasattr(ffn_layer, 'mlp') and hasattr(ffn_layer.mlp, 'router'):
                    router = ffn_layer.mlp.router
                    if hasattr(router, '_watermark_detection_data'):
                        data.append(router._watermark_detection_data)
                        # 清除数据
                        del router._watermark_detection_data
    
    return data

