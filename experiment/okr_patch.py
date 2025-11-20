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
    
    # 参考 mves_watermark_corrected.py 的查找逻辑
    # Switch Transformers 使用 encoder-decoder 架构
    print(f"模型类型: {model_type}")
    print(f"模型类: {type(model).__name__}")
    
    decoder = None
    if hasattr(model, 'decoder'):
        decoder = model.decoder
        print(f"找到 decoder: {type(decoder).__name__}")
    elif hasattr(model, 'model') and hasattr(model.model, 'decoder'):
        decoder = model.model.decoder
        print(f"找到 decoder (model.decoder): {type(decoder).__name__}")
    
    if decoder is None:
        print("警告: 无法找到 decoder，尝试查找 encoder")
        # Switch Transformers 可能只有 encoder 或 decoder
        if hasattr(model, 'encoder'):
            decoder = model.encoder
            print(f"使用 encoder 作为 decoder: {type(decoder).__name__}")
        elif hasattr(model, 'model') and hasattr(model.model, 'encoder'):
            decoder = model.model.encoder
            print(f"使用 encoder 作为 decoder (model.encoder): {type(decoder).__name__}")
    
    if decoder is None:
        raise ValueError("无法找到 decoder 或 encoder，模型可能不是 encoder-decoder 架构")
    
    # 获取 decoder blocks
    decoder_blocks = None
    if hasattr(decoder, 'block'):
        decoder_blocks = decoder.block
        print(f"找到 decoder.block: {len(decoder_blocks)} 层")
    elif hasattr(decoder, 'layers'):
        decoder_blocks = decoder.layers
        print(f"找到 decoder.layers: {len(decoder_blocks)} 层")
    elif hasattr(decoder, 'transformer_blocks'):
        decoder_blocks = decoder.transformer_blocks
        print(f"找到 decoder.transformer_blocks: {len(decoder_blocks)} 层")
    elif hasattr(decoder, 'layer'):
        decoder_blocks = decoder.layer
        print(f"找到 decoder.layer: {len(decoder_blocks)} 层")
    
    if decoder_blocks is None:
        print(f"decoder 的属性: {[attr for attr in dir(decoder) if not attr.startswith('_')]}")
        raise ValueError("无法找到 decoder blocks")
    
    # 遍历所有层，查找MoE router
    print(f"开始查找 router，共有 {len(decoder_blocks)} 层")
    for layer_idx, layer in enumerate(decoder_blocks):
        router = None
        
        # 方式1: layer.layer[1] (Switch Transformer标准结构)
        if hasattr(layer, 'layer'):
            layer_list = layer.layer
            layer_len = len(layer_list) if hasattr(layer_list, '__len__') else 'N/A'
            print(f"Layer {layer_idx}: layer 类型: {type(layer_list).__name__}, 长度: {layer_len}")
            
            if isinstance(layer_list, (list, tuple)) and len(layer_list) > 1:
                ffn_layer = layer_list[1]
                print(f"Layer {layer_idx}: layer[1] 类型: {type(ffn_layer).__name__}")
                
                if hasattr(ffn_layer, 'mlp'):
                    print(f"Layer {layer_idx}: layer[1].mlp 类型: {type(ffn_layer.mlp).__name__}")
                    mlp_attrs = [attr for attr in dir(ffn_layer.mlp) if not attr.startswith('_')]
                    print(f"Layer {layer_idx}: layer[1].mlp 属性 (前10个): {mlp_attrs[:10]}")
                    if hasattr(ffn_layer.mlp, 'router'):
                        router = ffn_layer.mlp.router
                        print(f"Layer {layer_idx}: ✓ 找到 router (方式1: layer.layer[1].mlp.router)")
                elif hasattr(ffn_layer, 'router'):
                    router = ffn_layer.router
                    print(f"Layer {layer_idx}: ✓ 找到 router (方式1: layer.layer[1].router)")
                elif hasattr(ffn_layer, 'expert_router'):
                    router = ffn_layer.expert_router
                    print(f"Layer {layer_idx}: ✓ 找到 router (方式1: layer.layer[1].expert_router)")
                else:
                    ffn_attrs = [attr for attr in dir(ffn_layer) if not attr.startswith('_')]
                    print(f"Layer {layer_idx}: ✗ layer[1] 没有找到 router，属性: {ffn_attrs[:10]}")
            # 如果 layer 是 ModuleList，尝试遍历
            elif hasattr(layer_list, '__iter__') and not isinstance(layer_list, (str, bytes)):
                print(f"Layer {layer_idx}: layer 是可迭代对象，尝试遍历")
                for sub_idx, sublayer in enumerate(layer_list):
                    if hasattr(sublayer, 'mlp') and hasattr(sublayer.mlp, 'router'):
                        router = sublayer.mlp.router
                        print(f"Layer {layer_idx}: ✓ 找到 router (方式1: layer[{sub_idx}].mlp.router)")
                        break
                    elif hasattr(sublayer, 'router'):
                        router = sublayer.router
                        print(f"Layer {layer_idx}: ✓ 找到 router (方式1: layer[{sub_idx}].router)")
                        break
                if router is None:
                    print(f"Layer {layer_idx}: ✗ 遍历 layer 后仍未找到 router")
            else:
                print(f"Layer {layer_idx}: layer 不是 list/tuple/iterable 或长度不足")
        
        # 方式2: layer.feed_forward
        if router is None and hasattr(layer, 'feed_forward'):
            ffn_layer = layer.feed_forward
            if hasattr(ffn_layer, 'mlp') and hasattr(ffn_layer.mlp, 'router'):
                router = ffn_layer.mlp.router
                print(f"Layer {layer_idx}: 找到 router (方式2: feed_forward.mlp.router)")
            elif hasattr(ffn_layer, 'router'):
                router = ffn_layer.router
                print(f"Layer {layer_idx}: 找到 router (方式2: feed_forward.router)")
        
        # 方式3: layer.mlp
        if router is None and hasattr(layer, 'mlp'):
            ffn_layer = layer.mlp
            if hasattr(ffn_layer, 'router'):
                router = ffn_layer.router
                print(f"Layer {layer_idx}: 找到 router (方式3: mlp.router)")
        
        # 方式4: 直接检查layer是否有router
        if router is None and hasattr(layer, 'router'):
            router = layer.router
            print(f"Layer {layer_idx}: 找到 router (方式4: layer.router)")
        
        # 方式5: 检查 layer.block_sparse_moe
        if router is None and hasattr(layer, 'block_sparse_moe'):
            moe = layer.block_sparse_moe
            if hasattr(moe, 'router'):
                router = moe.router
                print(f"Layer {layer_idx}: 找到 router (方式5: block_sparse_moe.router)")
            elif hasattr(moe, 'gate'):
                router = moe.gate
                print(f"Layer {layer_idx}: 找到 router (方式5: block_sparse_moe.gate)")
        
        # 如果找到了 router，处理它
        if router is not None:
            print(f"Found router at layer {layer_idx}, type: {type(router).__name__}")
            
            # 获取 router 的内部 gate network
            # SwitchTransformersTop1Router 的结构
            gate_layer = None
            input_dim = None
            num_experts_found = None
            router_dtype = None  # 获取 router 的 dtype
            
            # 先获取 router 的 dtype（从参数推断）
            try:
                for param in router.parameters():
                    router_dtype = param.dtype
                    break
            except:
                router_dtype = torch.float32  # 默认
            
            # 方式1: router.gate (Linear层) - SwitchTransformersTop1Router 的标准结构
            if hasattr(router, 'gate') and isinstance(router.gate, torch.nn.Linear):
                gate_layer = router.gate
                input_dim = gate_layer.in_features
                num_experts_found = gate_layer.out_features
                router_dtype = gate_layer.weight.dtype
                print(f"  使用 router.gate: {input_dim} -> {num_experts_found}, dtype: {router_dtype}")
            
            # 方式2: router 直接有 weight 参数
            elif hasattr(router, 'weight') and isinstance(router.weight, torch.nn.Parameter):
                if len(router.weight.shape) >= 2:
                    input_dim = router.weight.shape[0]
                    num_experts_found = router.weight.shape[-1]
                    router_dtype = router.weight.dtype
                    print(f"  使用 router.weight: {input_dim} -> {num_experts_found}, dtype: {router_dtype}")
                else:
                    print(f"警告: Layer {layer_idx} - router.weight 形状异常: {router.weight.shape}")
            
            # 方式3: 检查 router 的其他属性（SwitchTransformersTop1Router 可能有 q_proj 等）
            elif hasattr(router, 'q_proj') and isinstance(router.q_proj, torch.nn.Linear):
                gate_layer = router.q_proj
                input_dim = gate_layer.in_features
                num_experts_found = gate_layer.out_features
                router_dtype = gate_layer.weight.dtype
                print(f"  使用 router.q_proj: {input_dim} -> {num_experts_found}, dtype: {router_dtype}")
            
            # 如果还是无法获取维度，尝试从模型配置推断
            if input_dim is None or num_experts_found is None:
                # 尝试从模型配置获取
                if hasattr(model.config, 'd_model'):
                    input_dim = model.config.d_model
                    print(f"  从模型配置获取 input_dim: {input_dim}")
                if hasattr(model.config, 'num_local_experts'):
                    num_experts_found = model.config.num_local_experts
                    print(f"  从模型配置获取 num_experts: {num_experts_found}")
                elif hasattr(model.config, 'num_experts'):
                    num_experts_found = model.config.num_experts
                    print(f"  从模型配置获取 num_experts: {num_experts_found}")
            
            # 如果仍然无法获取，跳过这一层
            if input_dim is None or num_experts_found is None:
                print(f"警告: Layer {layer_idx} - 无法获取 router 的维度信息，跳过")
                print(f"  router 属性: {[attr for attr in dir(router) if not attr.startswith('_')]}")
                continue
            
            # 确保 router_dtype 不为 None
            if router_dtype is None:
                router_dtype = torch.float32
            
            # 验证一致性
            if num_experts_found != num_experts:
                print(f"警告: 配置中的num_experts ({num_experts}) 与router层 ({num_experts_found}) 不一致")
                num_experts = num_experts_found
            
            # 保存原始forward方法
            original_forward = router.forward
            
            # 获取设备
            device = next(router.parameters()).device
            
            # 创建 OKR router（仅用于计算，不替换原对象）
            # 使用与 router 相同的 dtype
            okr_router = OKRRouter(input_dim, num_experts, top_k=top_k, epsilon=epsilon)
            okr_router = okr_router.to(device=device, dtype=router_dtype)
            
            # 复制原始权重
            try:
                if gate_layer is not None:
                    with torch.no_grad():
                        # 确保 dtype 匹配
                        if gate_layer.weight.dtype != okr_router.gate_network.weight.dtype:
                            okr_router.gate_network.weight.data = gate_layer.weight.data.to(dtype=okr_router.gate_network.weight.dtype)
                        else:
                            okr_router.gate_network.weight.copy_(gate_layer.weight)
                        print(f"  已复制 gate_layer 权重: {gate_layer.weight.shape}, dtype: {gate_layer.weight.dtype}")
                elif hasattr(router, 'weight') and router.weight is not None:
                    with torch.no_grad():
                        # 检查形状是否匹配
                        if router.weight.shape == okr_router.gate_network.weight.shape:
                            if router.weight.dtype != okr_router.gate_network.weight.dtype:
                                okr_router.gate_network.weight.data = router.weight.data.to(dtype=okr_router.gate_network.weight.dtype)
                            else:
                                okr_router.gate_network.weight.copy_(router.weight)
                            print(f"  已复制 router.weight: {router.weight.shape}, dtype: {router.weight.dtype}")
                        else:
                            print(f"警告: 权重形状不匹配 - router: {router.weight.shape}, okr: {okr_router.gate_network.weight.shape}")
                            # 尝试转置或调整
                            if router.weight.shape == okr_router.gate_network.weight.shape[::-1]:
                                weight_t = router.weight.T
                                if weight_t.dtype != okr_router.gate_network.weight.dtype:
                                    okr_router.gate_network.weight.data = weight_t.data.to(dtype=okr_router.gate_network.weight.dtype)
                                else:
                                    okr_router.gate_network.weight.copy_(weight_t)
                                print(f"  已复制并转置权重")
                            else:
                                print(f"警告: 无法复制权重，使用随机初始化")
                else:
                    print(f"警告: 无法找到 router 的权重，使用随机初始化")
            except Exception as e:
                print(f"警告: 复制权重时出错: {e}，使用随机初始化")
                import traceback
                traceback.print_exc()
            
            # 初始化 secret_projection
            if secret_key is not None:
                _initialize_secret_projection(okr_router, secret_key, input_dim, num_experts, device)
            
            # 将 OKR router 保存到 router 对象上，供检测器使用
            router._okr_router = okr_router
            
            # 创建新的 forward 方法，只替换方法，不替换对象
            def make_okr_forward(orig_fwd, okr_rt, mdl, nm):
                def okr_forward(hidden_states: torch.Tensor):
                    # 调用 OKR 核心逻辑
                    routing_weights, selected_experts = _okr_forward_core(okr_rt, hidden_states)
                    
                    # 保存路由信息到模型和 router
                    if not hasattr(mdl, '_okr_routing_data'):
                        mdl._okr_routing_data = []
                    mdl._okr_routing_data.append(selected_experts)
                    
                    # 也保存到 router 对象上
                    router._selected_experts = selected_experts
                    
                    batch_size, seq_len, _ = hidden_states.shape
                    num_experts = okr_rt.gate_network.out_features
                    
                    # 构造 Switch Transformers 格式的返回值
                    # router_mask: [batch_size, seq_len, num_experts]
                    router_mask = torch.zeros(
                        (batch_size, seq_len, num_experts),
                        device=hidden_states.device,
                        dtype=hidden_states.dtype
                    )
                    router_mask.scatter_(
                        dim=-1,
                        index=selected_experts,
                        src=routing_weights
                    )
                    
                    # router_probs: [batch_size, seq_len, 1]
                    router_probs = routing_weights.sum(dim=-1, keepdim=True)
                    
                    # router_logits: [batch_size, seq_len, num_experts]
                    router_logits = torch.full(
                        (batch_size, seq_len, num_experts),
                        float('-inf'),
                        device=hidden_states.device,
                        dtype=hidden_states.dtype
                    )
                    logits_values = torch.log(routing_weights + 1e-9)
                    router_logits.scatter_(
                        dim=-1,
                        index=selected_experts,
                        src=logits_values
                    )
                    
                    return (router_mask, router_probs, router_logits)
                return okr_forward
            
            # 只替换 forward 方法，保持原对象不变
            router.forward = make_okr_forward(original_forward, okr_router, model, f"layer_{layer_idx}")
            
            count += 1
    
    if count == 0:
        # 添加更详细的错误信息
        error_msg = f"未找到任何gate层。模型类型: {model_type}\n"
        error_msg += f"已检查 {len(decoder_blocks) if decoder_blocks else 0} 层\n"
        if decoder_blocks and len(decoder_blocks) > 0:
            first_layer = decoder_blocks[0]
            error_msg += f"第一层的属性: {[attr for attr in dir(first_layer) if not attr.startswith('_')]}\n"
            if hasattr(first_layer, 'layer'):
                layer_list = first_layer.layer
                if isinstance(layer_list, (list, tuple)) and len(layer_list) > 0:
                    error_msg += f"第一层 layer[0] 的类型: {type(layer_list[0]).__name__}\n"
                    if len(layer_list) > 1:
                        error_msg += f"第一层 layer[1] 的类型: {type(layer_list[1]).__name__}\n"
                        if hasattr(layer_list[1], 'mlp'):
                            error_msg += f"第一层 layer[1].mlp 的属性: {[attr for attr in dir(layer_list[1].mlp) if not attr.startswith('_')]}\n"
        raise ValueError(error_msg)
    
    print(f"Injected OKR into {count} layers.")
    return model


def _okr_forward_core(router: OKRRouter, hidden_states: torch.Tensor):
    """
    OKR核心前向传播逻辑（避免递归调用）
    
    这是OKRRouter.forward的核心实现，但不通过forward方法调用
    """
    import torch.nn.functional as F
    
    # 确保 hidden_states 的 dtype 与 gate_network 匹配
    if hidden_states.dtype != router.gate_network.weight.dtype:
        hidden_states = hidden_states.to(dtype=router.gate_network.weight.dtype)
    
    # 1. 计算原始路由分数
    raw_logits = router.gate_network(hidden_states)
    
    # 2. 计算水印偏好
    # 确保 secret_projection 在正确的设备上
    if router.secret_projection.device != hidden_states.device:
        router.secret_projection = router.secret_projection.to(hidden_states.device)
    
    watermark_bias = torch.matmul(hidden_states, router.secret_projection)
    
    # 3. 计算安全掩码
    max_logits, _ = raw_logits.max(dim=-1, keepdim=True)
    safe_mask = raw_logits >= (max_logits - router.epsilon)
    
    # 4. 机会主义注入
    # 确保 watermark_bias 在正确的设备上
    if watermark_bias.device != raw_logits.device:
        watermark_bias = watermark_bias.to(raw_logits.device)
    
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


def _initialize_secret_projection(router: OKRRouter, secret_key: str, input_dim: int, num_experts: int, device: torch.device = None):
    """
    使用密钥初始化secret_projection
    
    这确保了相同密钥产生相同的投影矩阵
    
    Args:
        router: OKRRouter 实例
        secret_key: 密钥字符串
        input_dim: 输入维度
        num_experts: 专家数量
        device: 目标设备（如果为None，使用router当前设备）
    """
    import hashlib
    
    # 确定设备
    if device is None:
        device = router.secret_projection.device
    
    # 使用密钥生成确定性随机数
    seed = int(hashlib.sha256(secret_key.encode()).hexdigest()[:16], 16)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    
    # 生成投影矩阵，确保在正确的设备上
    with torch.no_grad():
        router.secret_projection.data = torch.randn(
            input_dim, num_experts, 
            generator=generator,
            device=device,
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
        
        batch_size, seq_len, _ = hidden_states.shape
        num_experts = router.gate_network.out_features
        
        # Switch Transformers 需要返回 tuple: (router_mask, router_probs, router_logits)
        # 1. router_mask: [batch_size, seq_len, num_experts] - 专家掩码（one-hot或概率分布）
        router_mask = torch.zeros(
            (batch_size, seq_len, num_experts),
            device=hidden_states.device,
            dtype=hidden_states.dtype
        )
        
        # 将选中的专家标记为1（或使用权重）
        # selected_experts: [batch_size, seq_len, top_k]
        # routing_weights: [batch_size, seq_len, top_k]
        router_mask.scatter_(
            dim=-1,
            index=selected_experts,  # [batch, seq, top_k]
            src=routing_weights  # [batch, seq, top_k]
        )
        
        # 2. router_probs: [batch_size, seq_len, 1] - 聚合后的路由概率
        # 这是所有选中专家的权重之和
        router_probs = routing_weights.sum(dim=-1, keepdim=True)  # [batch, seq, 1]
        
        # 3. router_logits: [batch_size, seq_len, num_experts] - 路由 logits
        # Switch Transformers 期望的是 3 维的 logits
        router_logits = torch.full(
            (batch_size, seq_len, num_experts),
            float('-inf'),
            device=hidden_states.device,
            dtype=hidden_states.dtype
        )
        
        # 将选中的专家和权重填充到router_logits
        # selected_experts: [batch, seq, top_k]
        # routing_weights: [batch, seq, top_k]
        # 将权重转换为logits（加上一个小的偏移避免数值问题）
        logits_values = torch.log(routing_weights + 1e-9)  # [batch, seq, top_k]
        
        # 填充到router_logits
        router_logits.scatter_(
            dim=-1,
            index=selected_experts,  # [batch, seq, top_k]
            src=logits_values  # [batch, seq, top_k]
        )
        
        # Switch Transformers 格式: 返回 tuple (router_mask, router_probs, router_logits)
        return (router_mask, router_probs, router_logits)
    
    return compatible_forward


# 便捷函数：支持配置对象
def inject_okr_with_config(model: Union[AutoModelForCausalLM, AutoModelForSeq2SeqLM], 
                           config) -> Union[AutoModelForCausalLM, AutoModelForSeq2SeqLM]:
    """
    使用配置对象注入OKR
    
    支持 OKRConfig 和 MVESConfig（向后兼容）
    
    Args:
        model: 预训练的MoE模型
        config: OKRConfig 或 MVESConfig 对象
        
    Returns:
        patched_model: 已注入OKR的模型
    """
    # 检查配置类型
    if hasattr(config, 'watermark'):
        # OKRConfig 或 MVESConfig
        wm_config = config.watermark
        epsilon = wm_config.epsilon
        secret_key = wm_config.secret_key
    else:
        # 直接传入 watermark 配置对象
        wm_config = config
        epsilon = wm_config.epsilon
        secret_key = wm_config.secret_key
    
    return inject_okr(
        model, 
        epsilon=epsilon,
        secret_key=secret_key
    )

