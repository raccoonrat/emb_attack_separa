"""
OKR Patch - 注入代码

这就是我们如何"黑"进现有的模型。我不喜欢破坏现有的代码库。
最好的方式是在运行时动态替换。
"""

import torch
from typing import Union, Optional
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

from okr_kernel import OKRRouter


def inject_okr(model: Union[AutoModelForCausalLM, AutoModelForSeq2SeqLM], 
               epsilon: float = 1.5,
               secret_key: Optional[str] = None) -> Union[AutoModelForCausalLM, AutoModelForSeq2SeqLM]:
    """
    把 OKR 注入到任何标准的 MoE 模型中 (如 Mixtral, Switch)。
    前提：你知道模型里 Router 层的名字。
    
    Args:
        model: 预训练的MoE模型
        epsilon: 质量容忍阈值
        secret_key: 可选的密钥（用于初始化secret_projection）
        
    Returns:
        patched_model: 已注入OKR的模型
    """
    count = 0
    
    # 检测模型类型
    model_type = model.config.model_type if hasattr(model.config, 'model_type') else ""
    
    # 获取模型配置
    if hasattr(model.config, 'num_local_experts'):
        num_experts = model.config.num_local_experts
        top_k = getattr(model.config, 'num_experts_per_tok', 2)
    elif hasattr(model.config, 'num_experts'):
        num_experts = model.config.num_experts
        top_k = getattr(model.config, 'num_experts_per_tok', 1)
    else:
        raise ValueError("无法检测模型配置：找不到num_experts或num_local_experts")
    
    # 获取输入维度（从gate层推断）
    input_dim = None
    
    # 遍历模型所有模块，找到gate层并提取配置
    for name, module in model.named_modules():
        # 这是一个启发式规则，你需要根据具体模型调整
        # 比如 Mixtral 的 Gate 叫 "gate"
        if ("gate" in name.lower() or "router" in name.lower()) and isinstance(module, torch.nn.Linear):
            print(f"Found gate layer: {name}")
            
            # 提取参数
            input_dim = module.in_features
            num_experts_found = module.out_features
            
            # 验证一致性
            if num_experts_found != num_experts:
                print(f"警告: 配置中的num_experts ({num_experts}) 与gate层 ({num_experts_found}) 不一致")
                num_experts = num_experts_found
            
            # 保存原始forward方法，以便了解返回格式
            original_forward = module.forward
            
            # 创建我们的 Kernel
            new_router = OKRRouter(input_dim, num_experts, top_k=top_k, epsilon=epsilon)
            
            # 关键：把原来训练好的权重拷过来！
            # 我们只换引擎，不换司机
            with torch.no_grad():
                new_router.gate_network.weight.copy_(module.weight)
                if module.bias is not None:
                    # OKR 默认没开 bias，如果原模型有，这得改一下 Kernel
                    # 这里假设无 bias 或简单忽略
                    print(f"警告: 原gate层有bias，但OKRRouter不支持bias，已忽略")
            
            # 如果提供了secret_key，用它初始化secret_projection
            if secret_key is not None:
                _initialize_secret_projection(new_router, secret_key, input_dim, num_experts)
            
            # 创建适配的forward方法，确保返回格式与原始gate层兼容
            new_router.forward = _create_compatible_forward(
                new_router, original_forward, model, name
            )
            
            # 暴力替换
            # 这种 getattr/setattr 递归有点丑，但这是 Python
            parent_name = name.rsplit('.', 1)[0]
            child_name = name.rsplit('.', 1)[1]
            parent = model.get_submodule(parent_name)
            setattr(parent, child_name, new_router)
            
            count += 1
    
    if count == 0:
        raise ValueError(f"未找到任何gate层。模型类型: {model_type}")
    
    print(f"Injected OKR into {count} layers.")
    return model


def _okr_forward_core(router: OKRRouter, hidden_states: torch.Tensor):
    """
    OKR核心前向传播逻辑（避免递归调用）
    
    这是OKRRouter.forward的核心实现，但不通过forward方法调用
    """
    import torch.nn.functional as F
    
    # 1. 计算原始路由分数
    raw_logits = router.gate_network(hidden_states)
    
    # 2. 计算水印偏好
    watermark_bias = torch.matmul(hidden_states, router.secret_projection)
    
    # 3. 计算安全掩码
    max_logits, _ = raw_logits.max(dim=-1, keepdim=True)
    safe_mask = raw_logits >= (max_logits - router.epsilon)
    
    # 4. 机会主义注入
    modified_scores = torch.where(
        safe_mask,
        watermark_bias,
        torch.tensor(-1e9, device=raw_logits.device, dtype=raw_logits.dtype)
    )
    
    # 5. 选取Top-K
    _, selected_experts = torch.topk(modified_scores, router.top_k, dim=-1)
    
    # 6. 计算权重（使用原始logits）
    router_logits = torch.gather(raw_logits, -1, selected_experts)
    routing_weights = F.softmax(router_logits, dim=-1)
    
    return routing_weights, selected_experts


def _initialize_secret_projection(router: OKRRouter, secret_key: str, input_dim: int, num_experts: int):
    """
    使用密钥初始化secret_projection
    
    这确保了相同密钥产生相同的投影矩阵
    """
    import hashlib
    
    # 使用密钥生成确定性随机数
    seed = int(hashlib.sha256(secret_key.encode()).hexdigest()[:16], 16)
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    # 生成投影矩阵
    with torch.no_grad():
        router.secret_projection.data = torch.randn(
            input_dim, num_experts, 
            generator=generator,
            device=router.secret_projection.device,
            dtype=router.secret_projection.dtype
        )


def _create_compatible_forward(
    router: OKRRouter, 
    original_forward, 
    model: Union[AutoModelForCausalLM, AutoModelForSeq2SeqLM],
    layer_name: str
):
    """
    创建兼容的forward方法，确保返回格式与原始gate层一致
    
    不同的MoE模型可能有不同的返回格式：
    - Mixtral: 返回 router_logits [batch*seq, num_experts]
    - Switch: 可能返回 tuple (router_mask, router_probs, router_logits)
    """
    def compatible_forward(hidden_states: torch.Tensor):
        # OKR核心逻辑（直接调用OKRRouter的核心方法，避免递归）
        routing_weights, selected_experts = _okr_forward_core(router, hidden_states)
        
        # 保存路由信息供检测器使用
        model._okr_routing_data = selected_experts
        
        # 根据原始forward的返回格式适配
        # 尝试调用一次原始forward（使用dummy输入）来检测返回格式
        # 但为了避免副作用，我们直接构造兼容的返回格式
        
        batch_size, seq_len, _ = hidden_states.shape
        num_experts = router.gate_network.out_features
        
        # 构造router_logits: [batch*seq, num_experts]
        # 只保留被选中专家的logits，其他为-inf
        router_logits = torch.full(
            (batch_size * seq_len, num_experts),
            float('-inf'),
            device=hidden_states.device,
            dtype=hidden_states.dtype
        )
        
        # 将选中的专家和权重填充到router_logits
        # 注意：routing_weights已经是softmax归一化的
        # 我们需要将它们转换回logits形式
        # 使用log(routing_weights) + offset来避免-inf
        selected_experts_flat = selected_experts.view(-1, router.top_k)  # [batch*seq, top_k]
        routing_weights_flat = routing_weights.view(-1, router.top_k)  # [batch*seq, top_k]
        
        # 将权重转换为logits（加上一个小的偏移避免数值问题）
        logits_values = torch.log(routing_weights_flat + 1e-9)
        
        # 填充到router_logits
        router_logits.scatter_(
            dim=1,
            index=selected_experts_flat,
            src=logits_values
        )
        
        # 大多数MoE模型的gate层只返回router_logits
        # 如果原始模型需要tuple格式，这里需要进一步适配
        return router_logits
    
    return compatible_forward


# 便捷函数：支持MVESConfig
def inject_okr_with_config(model: Union[AutoModelForCausalLM, AutoModelForSeq2SeqLM], 
                           config) -> Union[AutoModelForCausalLM, AutoModelForSeq2SeqLM]:
    """
    使用MVESConfig注入OKR
    
    Args:
        model: 预训练的MoE模型
        config: MVESConfig对象
        
    Returns:
        patched_model: 已注入OKR的模型
    """
    wm_config = config.watermark
    return inject_okr(
        model, 
        epsilon=wm_config.epsilon,
        secret_key=wm_config.secret_key
    )

