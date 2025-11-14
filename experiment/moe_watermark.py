import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from typing import Callable, Optional, Tuple

class MoEWatermark:
    """
    实现了 MoE Gating 水印的核心逻辑。
    此类的方法将被注入 (patch) 到预训练模型的 MoE Gating 模块中。
    """
    
    def __init__(self, K_sec: str, epsilon: float, num_experts: int, k_top: int, device: torch.device):
        self.secret_key = K_sec
        self.epsilon = epsilon
        self.num_experts = num_experts
        self.k_top = k_top
        self.device = device
        
        # 理论: epsilon = Var[delta_l]
        # 实现: 我们在 "绿色专家" 上施加正偏置 b, 
        # 在 k_top 个 "红色专家" 上施加负偏置 -b', 以保持均值接近0
        # 这是一个简化的实现，更精确的实现需要匹配 Var
        self.bias_strength = torch.sqrt(torch.tensor(self.epsilon, device=self.device)) * 5.0 # 简化的启发式调整

    def get_context_hash(self, hidden_states: torch.Tensor) -> int:
        """
        根据上下文 (hidden_states) 生成一个用于 PRNG 的种子。
        这是一个简化的实现。
        """
        # [batch_size, seq_len, dim] -> [batch_size, seq_len]
        hashed = torch.sum(hidden_states, dim=-1).long() 
        # 使用最后一个 token 的哈希值
        # [batch_size]
        last_token_hash = hashed[:, -1] 
        # 合并 batch 中的哈希值
        seed = torch.sum(last_token_hash).item()
        return hash((self.secret_key, seed))

    def get_bias_vector(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        生成偏置向量 delta_l。
        严格遵循方案 2. 节。
        """
        batch_size, seq_len, _ = hidden_states.shape
        context_hash = self.get_context_hash(hidden_states)
        
        # 使用确定性的种子
        generator = torch.Generator(device=self.device)
        generator.manual_seed(context_hash)
        
        delta_l = torch.zeros((batch_size, seq_len, self.num_experts), device=self.device)
        
        # 1. 选择一个 "绿色专家"
        green_expert = torch.randint(0, self.num_experts, (batch_size, seq_len), generator=generator, device=self.device)
        
        # 2. 施加正偏置
        delta_l.scatter_(-1, green_expert.unsqueeze(-1), self.bias_strength)
        
        # 3. (可选) 施加负偏置以平衡
        # ...
        
        return delta_l

    def watermarked_router_forward(
        self, 
        original_forward: Callable, 
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        被修补 (patched) 的前向传播函数。
        """
        # 1. 原始 logits
        # [batch_size, seq_len, num_experts]
        l_0 = original_forward(hidden_states)
        
        # 2. 生成并注入偏置
        delta_l = self.get_bias_vector(hidden_states)
        l_1 = l_0 + delta_l
        
        # 3. 计算 p0 (用于检测器) 和 p1 (用于路由)
        # 注意: p0 和 p1 都是完整的 softmax 分布
        p_0_dist = torch.softmax(l_0, dim=-1)
        p_1_dist = torch.softmax(l_1, dim=-1)
        
        # 4. Gating 路由（实际执行）
        # 使用 l_1 (水马 logits) 进行 Top-k 选择
        # [batch_size, seq_len, k_top]
        top_k_scores, S_indices = torch.topk(p_1_dist, self.k_top, dim=-1)
        
        # 归一化 top-k 得分
        top_k_scores = top_k_scores / (top_k_scores.sum(dim=-1, keepdim=True) + 1e-9)
        
        # 5. 返回检测器所需信息
        # p0, p1 (完整分布), S_indices (k个激活索引)
        return top_k_scores, S_indices, p_0_dist, p_1_dist, S_indices

def patch_moe_model_with_watermark(
    model: AutoModelForCausalLM, 
    K_sec: str, 
    epsilon: float
) -> AutoModelForCausalLM:
    """
    "修补" 一个预训练的 MoE 模型，注入水印逻辑。
    注意：这高度依赖于模型的具体实现 (如此处的 Mixtral)。
    """
    
    # 假设模型是 Mixtral
    if "Mixtral" not in model.config.model_type:
        raise NotImplementedError("此修补脚本目前仅支持 Mixtral 架构。")

    num_experts = model.config.num_local_experts
    k_top = model.config.num_experts_per_tok
    device = model.device

    watermark_injector = MoEWatermark(K_sec, epsilon, num_experts, k_top, device)
    
    print(f"Patching {model.config.model_type} with MoE watermark...")
    
    for layer in model.model.layers:
        # 找到 MoE 层的 Gating network (router)
        router = layer.block_sparse_moe.gate
        
        # 保存原始的 forward 方法
        original_forward = router.forward.__get__(router)
        
        # 创建新的 forward 方法
        def new_forward(hidden_states: torch.Tensor):
            # 调用水印逻辑
            top_k_scores, S_indices, p_0, p_1, S_obs = \
                watermark_injector.watermarked_router_forward(original_forward, hidden_states)
            
            # 将检测器信息附加到 router 对象上 (以便后续访问)
            router._watermark_detection_data = (p_0, p_1, S_obs)
            
            # MoE 路由需要稀疏的 gate_logits
            # [batch_size * seq_len, num_experts]
            batch_size, seq_len, _ = hidden_states.shape
            router_logits = torch.zeros(
                (batch_size * seq_len, num_experts), 
                dtype=top_k_scores.dtype, 
                device=device
            )
            router_logits.scatter_(-1, S_indices.view(-1, k_top), top_k_scores.view(-1, k_top))
            
            return router_logits

        # 应用 patch
        router.forward = new_forward
        
    print("Patching complete.")
    return model

def get_watermark_data_from_model(model: AutoModelForCausalLM) -> list:
    """
    从模型中提取检测器所需的数据。
    """
    data = []
    for layer in model.model.layers:
        if hasattr(layer.block_sparse_moe.gate, '_watermark_detection_data'):
            data.append(layer.block_sparse_moe.gate._watermark_detection_data)
            # (可选) 用后即焚，清除数据
            del layer.block_sparse_moe.gate._watermark_detection_data
    return data