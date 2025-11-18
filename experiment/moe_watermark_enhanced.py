"""
MoE水印增强实现：融入1118文档的合理部分

核心改进：
1. 添加Hook机制（最小侵入式集成）
2. 梯度裁剪保护
3. 排名交叉处理（f_k(S)系数）
4. 支持专家模式配置
5. 完善的检测数据收集
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from typing import Callable, Optional, Tuple, List, Dict, Any
import hashlib
import numpy as np

from mves_config import MVESConfig, WatermarkConfig


class MoEWatermarkEnhanced:
    """
    增强版MoE水印嵌入器
    
    融入1118文档的改进：
    1. 支持专家模式配置（而非随机选择）
    2. 梯度裁剪保护
    3. 排名交叉处理
    4. 更完善的检测数据收集
    """
    
    def __init__(
        self,
        secret_key: str,
        epsilon: float,
        num_experts: int,
        k_top: int,
        device: torch.device,
        expert_pattern: Optional[List[int]] = None,
        gradient_clip: float = 3.0
    ):
        """
        初始化增强版MoE水印嵌入器
        
        Args:
            secret_key: 密钥
            epsilon: 水印强度 ε = c²γ
            num_experts: 专家数量 K
            k_top: Top-k激活数
            device: 计算设备
            expert_pattern: 专家激活模式 [1,0,1,0,...] (可选，1118文档推荐)
            gradient_clip: 梯度裁剪阈值 (1118文档推荐3.0)
        """
        self.secret_key = secret_key
        self.epsilon = epsilon
        self.num_experts = num_experts
        self.k_top = k_top
        self.device = device
        self.gradient_clip = gradient_clip
        
        # 专家模式配置（1118文档推荐方式）
        if expert_pattern is not None:
            self.expert_pattern = expert_pattern
            if len(expert_pattern) != num_experts:
                raise ValueError(f"expert_pattern长度({len(expert_pattern)})必须等于num_experts({num_experts})")
        else:
            # 默认：随机选择（保持向后兼容）
            self.expert_pattern = None
        
        # 计算偏置强度 (论文定义3.2)
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
    
    def _compute_ranking_gap_coefficient(
        self,
        l_0: torch.Tensor,
        S_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        计算排名间隔系数 f_k(S) (论文引理4.4')
        
        用于处理Top-k激活的排名交叉问题
        
        Args:
            l_0: 原始logits [batch, seq, num_experts]
            S_indices: Top-k激活索引 [batch, seq, k_top]
            
        Returns:
            f_k_coeff: 排名间隔系数 [batch, seq]
        """
        batch_size, seq_len, num_experts = l_0.shape
        
        # 对logits排序
        sorted_logits, sorted_indices = torch.sort(l_0, dim=-1, descending=True)
        
        f_k_coeff = torch.ones(batch_size, seq_len, device=self.device)
        
        for b in range(batch_size):
            for s in range(seq_len):
                # 获取激活专家的logits
                activated_logits = l_0[b, s, S_indices[b, s, :]]
                
                # 计算最小排名间隔
                # gap_i = l_(i) - l_(i+1) for i < k
                # gap_k = l_(k) - l_(k+1) (k-th和(k+1)-th的间隔)
                sorted_vals = sorted_logits[b, s, :]
                
                # k-th和(k+1)-th的间隔
                if self.k_top < num_experts:
                    gap_min = sorted_vals[self.k_top - 1] - sorted_vals[self.k_top]
                else:
                    gap_min = 1.0  # 如果k_top == num_experts，使用默认值
                
                # f_k(S) = min(1, σ/gap_min)
                # 其中σ是softmax平滑常数，通常≈1
                sigma = 1.0
                if gap_min > 0:
                    f_k_coeff[b, s] = min(1.0, sigma / gap_min)
                else:
                    # 排名非常接近，系数可能很大
                    f_k_coeff[b, s] = 10.0  # 警告值
        
        return f_k_coeff
    
    def get_bias_vector(
        self,
        hidden_states: torch.Tensor,
        l_0: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        生成水印偏置向量（增强版）
        
        改进：
        1. 支持专家模式配置（1118文档）
        2. 梯度裁剪保护
        3. 排名交叉处理
        
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
        
        # 初始化偏置向量
        delta_l = torch.zeros_like(l_0)
        
        # 生成确定性种子
        context_hash = self._get_context_hash(hidden_states)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(context_hash)
        
        # 对每个位置计算偏置
        for b in range(batch_size):
            for s in range(seq_len):
                if self.expert_pattern is not None:
                    # 方法1：使用配置的专家模式（1118文档推荐）
                    for expert_idx in range(num_experts):
                        if self.expert_pattern[expert_idx] == 1:
                            # 增大该专家的选择概率
                            delta_l[b, s, expert_idx] = self.target_norm / np.sqrt(sum(self.expert_pattern))
                        else:
                            # 减小该专家的选择概率（1118文档：0.3倍）
                            delta_l[b, s, expert_idx] = -self.target_norm * 0.3 / np.sqrt(num_experts - sum(self.expert_pattern))
                else:
                    # 方法2：随机选择（保持向后兼容）
                    green_expert = torch.randint(
                        0, num_experts, (1,), generator=generator, device=self.device
                    ).item()
                    
                    red_candidates = [i for i in range(num_experts) if i != green_expert]
                    num_red = min(self.k_top, len(red_candidates))
                    if num_red > 0:
                        red_experts = torch.tensor(
                            np.random.choice(red_candidates, size=num_red, replace=False),
                            device=self.device
                        )
                        
                        bias_green = self.target_norm / np.sqrt(1 + num_red)
                        bias_red = -bias_green / num_red
                        
                        delta_l[b, s, green_expert] = bias_green
                        delta_l[b, s, red_experts] = bias_red
        
        # 梯度裁剪保护（1118文档：防止梯度爆炸）
        delta_l = torch.clamp(delta_l, -self.gradient_clip, self.gradient_clip)
        
        # 计算修改后的logits和分布
        l_1 = l_0 + delta_l
        p_1 = F.softmax(l_1, dim=-1)
        
        # 排名交叉处理（论文引理4.4'）
        # 计算Top-k激活
        top_k_scores, S_indices = torch.topk(p_1, self.k_top, dim=-1)
        
        # 计算排名间隔系数
        f_k_coeff = self._compute_ranking_gap_coefficient(l_0, S_indices)
        
        # 如果排名间隔很小，调整偏置强度
        # 注意：这里我们只是记录系数，实际调整在watermarked_router_forward中进行
        # 存储f_k系数供后续使用
        self._last_f_k_coeff = f_k_coeff
        
        return delta_l, p_0, p_1
    
    def watermarked_router_forward(
        self,
        original_forward: Callable,
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        水印路由前向传播（增强版）
        
        改进：
        1. 排名交叉处理
        2. 更完善的检测数据收集
        
        Returns:
            router_logits: 路由logits [batch*seq, num_experts]
            p_0: 原始激活分布 [batch, seq, num_experts]
            p_1: 修改后激活分布 [batch, seq, num_experts]
            S_indices: Top-k激活索引 [batch, seq, k_top]
        """
        # 计算原始logits
        l_0 = original_forward(hidden_states)
        
        # 处理维度
        if l_0.dim() == 2:
            batch_size, seq_len = hidden_states.shape[:2]
            l_0 = l_0.view(batch_size, seq_len, -1)
            hidden_states_reshaped = hidden_states
        else:
            batch_size, seq_len = hidden_states.shape[:2]
            hidden_states_reshaped = hidden_states
        
        # 生成偏置向量
        delta_l, p_0, p_1 = self.get_bias_vector(hidden_states_reshaped, l_0)
        
        # 应用排名交叉调整（如果系数很大，减小偏置）
        if hasattr(self, '_last_f_k_coeff'):
            f_k_coeff = self._last_f_k_coeff
            # 如果f_k > 1，说明排名间隔很小，需要减小偏置
            adjustment = torch.clamp(1.0 / f_k_coeff.unsqueeze(-1), 0.1, 1.0)
            delta_l = delta_l * adjustment
        
        # 重新计算（调整后）
        l_1 = l_0 + delta_l
        p_1 = F.softmax(l_1, dim=-1)
        
        # Top-k路由
        top_k_scores, S_indices = torch.topk(p_1, self.k_top, dim=-1)
        top_k_scores = top_k_scores / (top_k_scores.sum(dim=-1, keepdim=True) + 1e-9)
        
        # 存储检测数据
        self._detection_data.append((p_0, p_1, S_indices))
        
        # 返回路由logits
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


class MoEWatermarkHookWrapper:
    """
    Hook机制包装器（1118文档推荐的最小侵入式集成）
    
    优点：
    1. 无需修改模型代码
    2. 可逆（可移除hook）
    3. 适用于所有PyTorch模型
    """
    
    def __init__(
        self,
        model: Any,
        watermark_config: WatermarkConfig,
        expert_pattern: Optional[List[int]] = None
    ):
        """
        初始化Hook包装器
        
        Args:
            model: 预训练模型
            watermark_config: 水印配置
            expert_pattern: 专家激活模式（可选）
        """
        self.model = model
        self.config = watermark_config
        self.device = next(model.parameters()).device
        
        # 获取模型配置
        if hasattr(model.config, 'num_experts'):
            num_experts = model.config.num_experts
        elif hasattr(model.config, 'num_local_experts'):
            num_experts = model.config.num_local_experts
        else:
            num_experts = watermark_config.num_experts
        
        k_top = watermark_config.k_top if hasattr(watermark_config, 'k_top') else 1
        
        # 创建水印嵌入器
        self.watermark_injector = MoEWatermarkEnhanced(
            watermark_config.secret_key,
            watermark_config.epsilon,
            num_experts,
            k_top,
            self.device,
            expert_pattern=expert_pattern
        )
        
        # 存储hook句柄
        self.hook_handles: List[Any] = []
        self.registered = False
        
        # 提取gating网络
        self.gating_networks = self._find_gating_networks()
    
    def _find_gating_networks(self) -> List[Any]:
        """查找所有gating网络"""
        gating_networks = []
        
        # 支持不同模型架构
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # LLaMA/Mixtral架构
            for layer in self.model.model.layers:
                if hasattr(layer, 'block_sparse_moe') and hasattr(layer.block_sparse_moe, 'gate'):
                    gating_networks.append(layer.block_sparse_moe.gate)
                elif hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate'):
                    gating_networks.append(layer.mlp.gate)
        elif hasattr(self.model, 'decoder') and hasattr(self.model.decoder, 'block'):
            # Switch Transformer架构
            for layer in self.model.decoder.block:
                if hasattr(layer, 'layer') and len(layer.layer) > 1:
                    ffn_layer = layer.layer[1]
                    if hasattr(ffn_layer, 'mlp') and hasattr(ffn_layer.mlp, 'router'):
                        gating_networks.append(ffn_layer.mlp.router)
        
        return gating_networks
    
    def register_hooks(self):
        """
        注册前向钩子（1118文档推荐方式）
        
        这是最小侵入式的集成方法
        """
        if self.registered:
            print("警告: Hook已经注册")
            return
        
        def make_hook(watermark_injector):
            def hook_fn(module, input, output):
                """
                Hook函数：在gating网络输出时修改logits
                
                Args:
                    module: gating网络模块
                    input: 输入hidden states
                    output: 原始gating logits
                """
                # output可能是[batch*seq, num_experts]或[batch, seq, num_experts]
                if output.dim() == 2:
                    # 需要获取对应的hidden states
                    # 注意：hook中无法直接获取hidden states，需要从input获取
                    if len(input) > 0 and isinstance(input[0], torch.Tensor):
                        hidden_states = input[0]
                        batch_size, seq_len = hidden_states.shape[:2]
                        output = output.view(batch_size, seq_len, -1)
                    else:
                        # 如果无法获取，使用简化处理
                        return output
                
                # 调用水印逻辑
                router_logits, p_0, p_1, S_indices = \
                    watermark_injector.watermarked_router_forward(
                        lambda x: output,  # 原始forward就是返回output
                        input[0] if len(input) > 0 else output
                    )
                
                # 存储检测数据到模块
                module._watermark_detection_data = (p_0, p_1, S_indices)
                
                # 返回修改后的logits（需要与原始格式一致）
                if router_logits.dim() == 3:
                    router_logits = router_logits.view(-1, router_logits.shape[-1])
                
                return router_logits
            
            return hook_fn
        
        # 为每个gating网络注册hook
        for gating_net in self.gating_networks:
            handle = gating_net.register_forward_hook(
                make_hook(self.watermark_injector)
            )
            self.hook_handles.append(handle)
        
        self.registered = True
        print(f"✓ 已注册 {len(self.hook_handles)} 个gating网络hook")
    
    def remove_hooks(self):
        """移除所有hook（可逆操作）"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
        self.registered = False
        print("✓ 已移除所有hook")
    
    def get_detection_data(self) -> List[Tuple]:
        """从模型中提取检测数据"""
        data = []
        
        for gating_net in self.gating_networks:
            if hasattr(gating_net, '_watermark_detection_data'):
                data.append(gating_net._watermark_detection_data)
                # 清除数据
                del gating_net._watermark_detection_data
        
        return data


def create_watermark_wrapper(
    model: Any,
    config: MVESConfig,
    use_hook: bool = True,
    expert_pattern: Optional[List[int]] = None
) -> MoEWatermarkHookWrapper:
    """
    创建水印包装器（工厂函数）
    
    Args:
        model: 预训练模型
        config: MVES配置
        use_hook: 是否使用hook机制（推荐True）
        expert_pattern: 专家激活模式（可选）
        
    Returns:
        wrapper: 水印包装器
    """
    if use_hook:
        return MoEWatermarkHookWrapper(model, config.watermark, expert_pattern)
    else:
        # 使用patch方式（向后兼容）
        from moe_watermark_corrected import patch_switch_model_with_watermark
        return patch_switch_model_with_watermark(model, config)

