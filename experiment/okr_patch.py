"""
OKR Patch - 注入代码 (Production Ready)

功能：
1. 自动探测各种架构 (Switch, DeepSeek, Mixtral) 的 Router 层。
2. 将 OKRRouter (Kernel) 挂载到现有模型上。
3. 创建 Proxy Forward，处理数据记录和接口适配。

Reviewer: Linus Torvalds
Status: Stable
"""

import torch
import torch.nn as nn
from typing import Union, Optional, List, Any
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

# 引入新的 Kernel (V2.2)
from okr_kernel import OKRRouter

def inject_okr(model: Union[AutoModelForCausalLM, AutoModelForSeq2SeqLM], 
               epsilon: float = 1.5,
               secret_key: Optional[str] = None,
               threshold_ratio: float = 0.9,
               dead_zone_threshold: float = 0.01,
               watermark_alpha: float = 0.1) -> Union[AutoModelForCausalLM, AutoModelForSeq2SeqLM]:
    """
    把 OKR 注入到任何标准的 MoE 模型中。
    """
    
    print(f"[OKR Injector] Starting injection... Alpha={watermark_alpha}, Threshold={threshold_ratio}")
    if secret_key:
        print(f"[OKR Injector] Secret Key: {secret_key[:8]}...")
    else:
        print(f"[OKR Injector] WARNING: No secret key provided. Using random initialization (inconsistent across devices).")

    # --- 1. 深度探测 Router 层 (Deep Discovery) ---
    # 我们需要找到模型中所有的 Decoder Block
    
    decoder_blocks = []
    model_type = "unknown"

    # 1.1 尝试定位 Decoder Blocks
    if hasattr(model, 'decoder'): # Switch Transformers / T5 (Encoder-Decoder)
        model_type = "encoder-decoder"
        if hasattr(model.decoder, 'block'): 
            decoder_blocks = model.decoder.block
        elif hasattr(model.decoder, 'layers'):
            decoder_blocks = model.decoder.layers
    elif hasattr(model, 'model'): # Llama / DeepSeek / Mixtral (Decoder-Only)
        model_type = "decoder-only"
        if hasattr(model.model, 'layers'):
            decoder_blocks = model.model.layers
        elif hasattr(model.model, 'block'):
            decoder_blocks = model.model.block
    elif hasattr(model, 'layers'): # Some other architectures
        decoder_blocks = model.layers
    
    if len(decoder_blocks) == 0:
        raise ValueError(f"Could not find decoder blocks in model type: {type(model).__name__}")

    print(f"[OKR Injector] Detected model type: {model_type}. Found {len(decoder_blocks)} decoder blocks.")

    # --- 2. 遍历并注入 (Iterate and Inject) ---
    
    injected_count = 0
    
    for layer_idx, layer in enumerate(decoder_blocks):
        target_router = None
        router_location = ""

        # 2.1 穷举查找 Router (Heuristic Search)
        # 不同的模型架构把 router 藏在不同的地方
        
        # 策略 A: Switch Transformers (通常在 layer[1].mlp.router 或 layer[1].router)
        if hasattr(layer, 'layer') and isinstance(layer.layer, (list, nn.ModuleList)):
            for sub_layer in layer.layer:
                if hasattr(sub_layer, 'mlp') and hasattr(sub_layer.mlp, 'router'):
                    target_router = sub_layer.mlp.router
                    router_location = "layer.X.mlp.router"
                    break
                if hasattr(sub_layer, 'router'):
                    target_router = sub_layer.router
                    router_location = "layer.X.router"
                    break
        
        # 策略 B: Mixtral / Generic MoE (block_sparse_moe.gate)
        if target_router is None and hasattr(layer, 'block_sparse_moe'):
            if hasattr(layer.block_sparse_moe, 'gate'):
                target_router = layer.block_sparse_moe.gate
                router_location = "block_sparse_moe.gate"
            elif hasattr(layer.block_sparse_moe, 'router'):
                target_router = layer.block_sparse_moe.router
                router_location = "block_sparse_moe.router"

        # 策略 C: DeepSeek MoE (mlp.gate 或 mlp.router)
        if target_router is None and hasattr(layer, 'mlp'):
            if hasattr(layer.mlp, 'gate'):
                # DeepSeek 的 gate 有时是 MLP 投影层，有时是 Router
                # 我们通过权重形状区分: Router 输出通常较小 (num_experts < hidden_dim)
                candidate = layer.mlp.gate
                if hasattr(candidate, 'weight'):
                    w = candidate.weight
                    # 假设: 如果输出维度 <= 256 (专家数)，它很可能是 Router
                    # 如果输出维度 > 1000 (如 14336)，它是 FFN Projection
                    out_dim = w.shape[0] if w.shape[0] < w.shape[1] else w.shape[1] # Handle Transpose
                    if out_dim <= 512: # 保守阈值
                        target_router = candidate
                        router_location = "mlp.gate"
            elif hasattr(layer.mlp, 'router'):
                target_router = layer.mlp.router
                router_location = "mlp.router"

        # 策略 D: 直接属性 (layer.router)
        if target_router is None and hasattr(layer, 'router'):
            target_router = layer.router
            router_location = "layer.router"

        # 2.2 执行注入
        if target_router is not None:
            # 避免重复注入
            if hasattr(target_router, '_okr_router'):
                print(f"  Skipping Layer {layer_idx}: Already injected.")
                continue

            _inject_single_layer(
                target_router, 
                layer_idx, 
                epsilon, 
                secret_key, 
                threshold_ratio, 
                watermark_alpha,
                model
            )
            injected_count += 1
            # print(f"  Layer {layer_idx}: Injected OKR at {router_location}")
    
    if injected_count == 0:
        raise RuntimeError("Failed to inject OKR: No routers found! Please check model architecture.")
    
    print(f"[OKR Injector] Success! Injected into {injected_count} layers.")
    
    # 注入清理方法 (Utility)
    model.clear_okr_stats = lambda: _clear_stats(model)
    
    return model


def _inject_single_layer(original_router: nn.Module, 
                         layer_idx: int,
                         epsilon: float,
                         secret_key: Optional[str],
                         threshold_ratio: float,
                         watermark_alpha: float,
                         model_ref: nn.Module):
    """
    对单个 Router 层进行手术。
    """
    
    # 1. 推断维度 (Dimension Inference)
    input_dim = None
    num_experts = None
    
    # 获取权重矩阵
    weight = None
    if hasattr(original_router, 'classifier'): weight = original_router.classifier.weight # Switch
    elif hasattr(original_router, 'gate'): weight = original_router.gate.weight # Mixtral
    elif hasattr(original_router, 'weight'): weight = original_router.weight # Generic
    elif hasattr(original_router, 'q_proj'): weight = original_router.q_proj.weight # Some variants
    
    if weight is None:
        print(f"  WARNING: Layer {layer_idx}: Could not find weights in router. Skipping.")
        return

    # 判断形状 (Handle Transposed Weights)
    # 通常 Linear(in, out) 的 weight 是 [out, in]
    # Router 通常是 [num_experts, hidden_dim]
    # 启发式：hidden_dim 通常远大于 num_experts
    dim0, dim1 = weight.shape
    if dim0 > dim1: 
        # e.g. [2048, 8] -> hidden=2048, experts=8 (Transposed logic? Rare but possible)
        # 或者 [Intermediate, Hidden] -> Not a router
        # Switch Base: [8, 768] -> experts=8, hidden=768
        input_dim = dim0
        num_experts = dim1
    else:
        # Standard: [8, 768] or [num_experts, hidden]
        # Wait, Linear weight is (out_features, in_features)
        # So [8, 768] means out=8 (experts), in=768 (hidden)
        num_experts = dim0
        input_dim = dim1

    # 二次确认：如果 num_experts 太大，可能判断反了
    if num_experts > 512 and input_dim < 512:
        num_experts, input_dim = input_dim, num_experts
    
    # print(f"    -> Config: Input={input_dim}, Experts={num_experts}, Alpha={watermark_alpha}")

    # 2. 创建 Kernel (OKRRouter)
    # 务必保持 dtype 和 device 一致
    okr_router = OKRRouter(
        input_dim=input_dim,
        num_experts=num_experts,
        top_k=1, # Switch Base 默认 Top-1，这里硬编码为 1 (或者从 config 读取)
        threshold_ratio=threshold_ratio,
        watermark_alpha=watermark_alpha
    ).to(weight.device, dtype=weight.dtype)

    # 3. 复制原始权重 (Clone Weights)
    with torch.no_grad():
        # 确保形状匹配，可能需要转置
        if okr_router.gate_network.weight.shape == weight.shape:
            okr_router.gate_network.weight.copy_(weight)
        elif okr_router.gate_network.weight.shape == weight.T.shape:
            okr_router.gate_network.weight.copy_(weight.T)
        else:
            print(f"    ERROR: Shape mismatch. OKR: {okr_router.gate_network.weight.shape}, Orig: {weight.shape}")
            return

    # 4. 初始化密钥 (Secret Key Init)
    if secret_key:
        _initialize_secret_projection(okr_router, secret_key)

    # 5. 挂载并替换 Forward (The Hook)
    original_router._okr_router = okr_router
    
    # 关键：创建闭包时，要捕获当前的 original_router 和 okr_router
    original_router.forward = _create_proxy_forward(original_router, okr_router)


def _create_proxy_forward(original_router: nn.Module, okr_router: OKRRouter):
    """
    创建代理 Forward 函数。
    
    职责：
    1. 调用 Kernel 计算路由。
    2. 处理副作用（记录路由选择）。
    3. 严格返回 Tuple 格式。
    """
    def forward(hidden_states: torch.Tensor):
        # 1. Kernel 计算
        # 返回: (mask, probs, logits)
        router_mask, router_probs, router_logits = okr_router(hidden_states)
        
        # 2. 数据记录 (Side Effects)
        # 获取 Kernel 内部刚刚做出的选择
        selected_experts = okr_router._last_selected_experts # [batch, seq, top_k]
        
        # 初始化记录容器
        if not hasattr(original_router, '_okr_all_selected_experts'):
            original_router._okr_all_selected_experts = []
            
        # 区分生成模式 (Accumulate) 和 检测模式 (Overwrite/Batch)
        batch_size, seq_len, _ = hidden_states.shape
        
        if seq_len == 1:
            # 生成模式：逐个 Token 累积
            original_router._okr_all_selected_experts.append(selected_experts.detach().cpu())
        else:
            # 检测模式 / Prefill 模式：一次性处理长序列
            # 为了兼容 Detector 的读取逻辑 (cat dim=1)，我们将长序列切片存入
            # 或者清空旧数据存入新数据（取决于这是不是第一次）
            
            # 简单策略：如果是长序列，我们就认为这是检测阶段，直接存列表
            # 这样 Detector 做 torch.cat 时能还原
            
            # 如果列表已经有数据且长度不匹配，可能是混合调用，我们选择清空以防万一
            # (但在生成后的检测中，我们通常会手动调 clear_stats)
            if len(original_router._okr_all_selected_experts) > 0:
                 # 如果是检测阶段，通常 hidden_states 包含了所有 token
                 # 我们把之前累积的清空，只保留这次完整的
                 original_router._okr_all_selected_experts = []
            
            # 将 [batch, seq, topk] 拆成 seq 个 [batch, 1, topk]
            # 这样与生成模式的数据结构保持一致
            for t in range(seq_len):
                original_router._okr_all_selected_experts.append(
                    selected_experts[:, t:t+1, :].detach().cpu()
                )

        # 3. 返回结果 (Strict Contract)
        return (router_mask, router_probs, router_logits)
        
    return forward


def _initialize_secret_projection(router: OKRRouter, key: str):
    """
    确定性初始化私钥。
    """
    import hashlib
    # 使用 SHA256 生成 64位 Seed
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
    
    # 必须使用本地 Generator，避免影响全局随机状态
    gen = torch.Generator(device=router.secret_projection.device)
    gen.manual_seed(seed)
    
    with torch.no_grad():
        # 使用正态分布初始化
        router.secret_projection.normal_(generator=gen)
        
    router._secret_initialized.fill_(True)
    # print(f"    -> Secret Key initialized with seed {seed}")


def _clear_stats(model: nn.Module):
    """
    工具函数：清空所有 Router 的统计数据。
    """
    count = 0
    for m in model.modules():
        if hasattr(m, '_okr_all_selected_experts'):
            m._okr_all_selected_experts = []
            count += 1
    # print(f"[OKR] Cleared stats for {count} routers.")


# 保持向后兼容的便捷接口
def inject_okr_with_config(model, config):
    """
    支持通过 Config 对象注入
    """
    wm = config.watermark
    return inject_okr(
        model, 
        epsilon=wm.epsilon, 
        secret_key=wm.secret_key, 
        threshold_ratio=getattr(wm, 'threshold_ratio', 0.9),
        watermark_alpha=getattr(wm, 'watermark_alpha', 0.1)
    )