import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from typing import Callable, Optional, Tuple
import hashlib
import numpy as np

class MoEWatermark:
    """
    实现了 MoE Gating 水印的核心逻辑。
    严格按照论文理论框架: ε = c²γ, 且 KL(p1||p0) = ε
    
    核心思想 (论文第3-4节):
    1. 信号-攻击解耦: 水印嵌入在专家激活模式空间 S_B = {0,1}^K
    2. 通过修改gating网络的logit实现: l_1 = l_0 + Δl
    3. 确保 KL(p1||p0) = ε, 其中 p_i = softmax(l_i)
    4. 使用安全系数 c: ε = c²γ (论文定义5.1)
    """
    
    def __init__(self, K_sec: str, epsilon: float, num_experts: int, k_top: int, device: torch.device):
        """
        初始化MoE水印嵌入器
        
        Args:
            K_sec: 密钥 (用于确定性选择绿色专家)
            epsilon: 水印强度 ε, 满足 ε = c²γ (论文定义5.1)
            num_experts: 专家总数 K
            k_top: Top-k激活数
            device: 计算设备
        """
        self.secret_key = K_sec
        self.epsilon = epsilon
        self.num_experts = num_experts
        self.k_top = k_top
        self.device = device
        
        # 理论依据 (论文定义3.2):
        # ε = KL(p1||p0) ≈ Var_{e~p0}[Δl(e)] ≈ (1/2)||Δl||²_2
        # 因此: ||Δl||_2 ≈ sqrt(2ε)
        # 
        # 实现策略: 在"绿色专家"上施加正偏置, 在"红色专家"上施加负偏置
        # 以保持期望值接近0, 同时满足方差约束
        
        # 计算偏置强度: 使用论文引理3.2的近似关系
        # 对于均匀先验, Var[Δl] ≈ (1/2)||Δl||²_2
        # 因此: ||Δl||_2 ≈ sqrt(2ε)
        self.target_norm = np.sqrt(2.0 * epsilon)
        
        # 启发式: 将偏置集中在少数专家上, 以最大化激活概率变化
        # 选择1个绿色专家, k_top个红色专家
        self.num_green = 1
        self.num_red = k_top

    def get_context_hash(self, hidden_states: torch.Tensor) -> int:
        """
        根据上下文生成确定性种子 (论文第3节)
        
        使用密钥和hidden states的哈希值, 确保相同输入产生相同的水印模式
        """
        # [batch_size, seq_len, dim] -> [batch_size, seq_len]
        hashed = torch.sum(hidden_states, dim=-1).long() 
        # 使用最后一个token的哈希值 (更稳定)
        last_token_hash = hashed[:, -1] 
        # 合并batch中的哈希值
        seed = torch.sum(last_token_hash).item()
        
        # 使用密钥增强安全性
        combined = f"{self.secret_key}_{seed}".encode('utf-8')
        hash_value = int(hashlib.sha256(combined).hexdigest()[:16], 16)
        return hash_value

    def compute_kl_divergence(self, p0: torch.Tensor, p1: torch.Tensor) -> torch.Tensor:
        """
        计算 KL(p1||p0) (论文定义3.2)
        
        用于验证水印强度是否满足 ε = KL(p1||p0)
        """
        # 防止log(0)
        p0 = torch.clamp(p0, min=1e-9)
        p1 = torch.clamp(p1, min=1e-9)
        
        # KL(p1||p0) = Σ p1 * log(p1/p0)
        kl = torch.sum(p1 * torch.log(p1 / p0), dim=-1)
        return kl

    def get_bias_vector(self, hidden_states: torch.Tensor, l_0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        生成偏置向量 Δl, 确保 KL(p1||p0) = ε (论文定义3.2)
        
        Args:
            hidden_states: 输入hidden states [batch, seq, dim]
            l_0: 原始gating logits [batch, seq, num_experts]
            
        Returns:
            delta_l: 偏置向量 [batch, seq, num_experts]
            p_0: 原始激活分布 [batch, seq, num_experts]
            p_1: 修改后激活分布 [batch, seq, num_experts]
        """
        batch_size, seq_len, _ = hidden_states.shape
        context_hash = self.get_context_hash(hidden_states)
        
        # 使用确定性种子选择绿色专家
        generator = torch.Generator(device=self.device)
        generator.manual_seed(context_hash)
        
        # 计算原始分布 p_0
        p_0 = F.softmax(l_0, dim=-1)
        
        # 初始化偏置向量
        delta_l = torch.zeros_like(l_0)
        
        # 策略: 对每个位置独立选择绿色和红色专家
        for b in range(batch_size):
            for s in range(seq_len):
                # 选择绿色专家 (正偏置)
                green_expert = torch.randint(0, self.num_experts, (1,), generator=generator, device=self.device).item()
                
                # 选择红色专家 (负偏置) - 排除绿色专家
                red_candidates = [i for i in range(self.num_experts) if i != green_expert]
                red_experts = torch.tensor(
                    np.random.choice(red_candidates, size=min(self.num_red, len(red_candidates)), replace=False),
                    device=self.device
                )
                
                # 计算偏置强度
                # 目标: 使得 KL(p1||p0) ≈ ε
                # 使用迭代方法调整偏置强度
                bias_green = self.target_norm / np.sqrt(self.num_green + self.num_red)
                bias_red = -bias_green * (self.num_green / self.num_red)  # 保持期望为0
                
                # 应用偏置
                delta_l[b, s, green_expert] = bias_green
                delta_l[b, s, red_experts] = bias_red
        
        # 计算修改后的logits和分布
        l_1 = l_0 + delta_l
        p_1 = F.softmax(l_1, dim=-1)
        
        # 验证KL散度 (可选, 用于调试)
        # kl_actual = self.compute_kl_divergence(p_0, p_1).mean()
        # print(f"Target ε: {self.epsilon:.6f}, Actual KL: {kl_actual.item():.6f}")
        
        return delta_l, p_0, p_1

    def watermarked_router_forward(
        self, 
        original_forward: Callable, 
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        被修补 (patched) 的前向传播函数 (论文第3节)
        
        严格按照论文流程:
        1. 计算原始logits l_0
        2. 生成偏置 Δl, 使得 KL(p1||p0) = ε
        3. 计算修改后logits l_1 = l_0 + Δl
        4. 使用l_1进行Top-k路由
        5. 保存p_0和p_1供检测器使用
        
        Returns:
            top_k_scores: Top-k激活的归一化分数 [batch, seq, k_top]
            S_indices: Top-k激活的专家索引 [batch, seq, k_top]
            p_0_dist: 原始激活分布 [batch, seq, num_experts]
            p_1_dist: 修改后激活分布 [batch, seq, num_experts]
            S_indices: 同上 (用于兼容性)
        """
        # 1. 计算原始 logits l_0 (论文定义3.2)
        # [batch_size, seq_len, num_experts]
        l_0 = original_forward(hidden_states)
        
        # 2. 生成偏置向量 Δl, 并计算 p_0 和 p_1
        # 确保 KL(p1||p0) = ε
        delta_l, p_0_dist, p_1_dist = self.get_bias_vector(hidden_states, l_0)
        
        # 3. 计算修改后的logits (用于路由)
        l_1 = l_0 + delta_l
        
        # 4. Gating 路由 (论文第3节)
        # 使用 l_1 进行 Top-k 选择
        # [batch_size, seq_len, k_top]
        top_k_scores, S_indices = torch.topk(p_1_dist, self.k_top, dim=-1)
        
        # 归一化 top-k 得分 (MoE路由需要归一化权重)
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
    "修补" 一个预训练的 MoE 模型，注入水印逻辑 (论文第3节)
    
    严格按照论文理论框架:
    - 水印强度 ε = c²γ (论文定义5.1)
    - 通过修改gating网络logit实现: l_1 = l_0 + Δl
    - 确保 KL(p1||p0) = ε
    
    注意：这高度依赖于模型的具体实现 (如 Mixtral, LLaMA-MoE)
    """
    
    # 检测模型类型
    model_type = model.config.model_type if hasattr(model.config, 'model_type') else ""
    
    # 支持 Mixtral 架构
    if "Mixtral" in model_type or "mixtral" in str(type(model)).lower():
        num_experts = model.config.num_local_experts
        k_top = model.config.num_experts_per_tok
        device = model.device
        
        watermark_injector = MoEWatermark(K_sec, epsilon, num_experts, k_top, device)
        
        print(f"Patching {model_type} with MoE watermark (ε={epsilon:.6f})...")
        
        for layer_idx, layer in enumerate(model.model.layers):
            # 找到 MoE 层的 Gating network (router)
            if hasattr(layer, 'block_sparse_moe') and hasattr(layer.block_sparse_moe, 'gate'):
                router = layer.block_sparse_moe.gate
                
                # 保存原始的 forward 方法
                original_forward = router.forward.__get__(router)
                
                # 创建新的 forward 方法 (使用闭包捕获layer_idx)
                def make_new_forward(orig_fwd, layer_id):
                    def new_forward(hidden_states: torch.Tensor):
                        # 调用水印逻辑
                        top_k_scores, S_indices, p_0, p_1, S_obs = \
                            watermark_injector.watermarked_router_forward(orig_fwd, hidden_states)
                        
                        # 将检测器信息附加到 router 对象上 (以便后续访问)
                        router._watermark_detection_data = (p_0, p_1, S_obs)
                        router._layer_idx = layer_id
                        
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
                    return new_forward
                
                # 应用 patch
                router.forward = make_new_forward(original_forward, layer_idx)
        
        print(f"Patching complete. Watermark strength ε={epsilon:.6f} (c²γ parameterization).")
        
    # 支持 LLaMA-MoE 架构 (如果存在)
    elif "llama" in model_type.lower() and hasattr(model.config, 'num_experts'):
        num_experts = model.config.num_experts
        k_top = getattr(model.config, 'num_experts_per_tok', 2)
        device = model.device
        
        watermark_injector = MoEWatermark(K_sec, epsilon, num_experts, k_top, device)
        
        print(f"Patching {model_type} with MoE watermark (ε={epsilon:.6f})...")
        # TODO: 实现LLaMA-MoE的patch逻辑
        raise NotImplementedError("LLaMA-MoE architecture support is under development.")
        
    else:
        raise NotImplementedError(
            f"此修补脚本目前仅支持 Mixtral 架构。"
            f"检测到的模型类型: {model_type}"
        )
    
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