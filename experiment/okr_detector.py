"""
OKR Detector - 检测器

检测器不需要重跑整个大模型。如果你只想验证水印，理论上只需要跑这一层。
但在实际中，我们需要文本对应的 Embedding，所以通常还是得跑一遍 Inference。

这里的逻辑是：计算 Green-List Hit Rate（也就是水印命中率）。
"""

import torch
import numpy as np
from typing import Tuple, Optional


class OKRDetector:
    """
    OKR 水印检测器
    
    核心验证逻辑：
    1. 重算原本的 Logits (Ground Truth)
    2. 重算水印信号 (Expected Signal)
    3. 重算机会窗口 (Opportunities)
    4. 验证命中 (Check Hits)
    """
    
    def __init__(self, model, epsilon: float = 1.5):
        """
        初始化检测器
        
        Args:
            model: 已经注入了 OKRRouter 的模型
            epsilon: 质量容忍阈值（必须与路由器一致）
        """
        self.model = model
        self.epsilon = epsilon

    def detect(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, 
               decoder_input_ids: Optional[torch.Tensor] = None):
        """
        对一段文本进行检测。
        
        Args:
            input_ids: 输入token IDs [batch, seq_len] (encoder输入，对于encoder-decoder模型)
            attention_mask: 注意力掩码 [batch, seq_len]
            decoder_input_ids: decoder输入token IDs [batch, seq_len] (可选，如果不提供则使用input_ids)
            
        Returns:
            score: 水印命中率 (0-1)
            verdict: "Watermarked" 或 "Clean"
        """
        # 检查模型类型（encoder-decoder 还是 decoder-only）
        is_encoder_decoder = hasattr(self.model.config, 'is_encoder_decoder') and self.model.config.is_encoder_decoder
        
        # 对于 encoder-decoder 模型（如 Switch Transformers）
        # 检测时应该使用生成时保存的路由数据，而不是重新生成
        if is_encoder_decoder:
            # 先获取生成时保存的路由数据（如果存在）
            actual_selected_experts = self._extract_selected_experts()
            
            if actual_selected_experts is None:
                import logging
                logging.getLogger("okr_detector").warning("检测时未找到路由数据，需要重新生成")
                # 如果没有保存的路由数据，需要重新生成（但会覆盖之前的数据）
                with torch.no_grad():
                    generated = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=input_ids.shape[1] + 50,  # 生成更多token以触发路由
                        num_beams=1,
                        do_sample=False,
                        output_hidden_states=True,
                        return_dict_in_generate=True
                    )
                
                # 再次尝试获取路由数据
                actual_selected_experts = self._extract_selected_experts()
                if actual_selected_experts is None:
                    return 0.0, "No routing data available"
                
                # 使用生成的序列作为 decoder 输入
                if isinstance(generated, torch.Tensor):
                    decoder_input_ids = generated[:, :-1] if generated.shape[1] > 1 else generated
                elif hasattr(generated, 'sequences'):
                    decoder_input_ids = generated.sequences[:, :-1] if generated.sequences.shape[1] > 1 else generated.sequences
                else:
                    decoder_input_ids = input_ids
            else:
                # 有保存的路由数据，使用 decoder_input_ids 重新运行模型获取 hidden_states
                # 如果没有提供 decoder_input_ids，使用 input_ids（可能不准确）
                if decoder_input_ids is None:
                    decoder_input_ids = input_ids
                    import logging
                    logging.getLogger("okr_detector").warning("检测时未提供 decoder_input_ids，使用 input_ids（可能不准确）")
            
            import logging
            logging.getLogger("okr_detector").info(f"检测时获取到路由数据: {actual_selected_experts.shape}")
            
            # 重要：确保 decoder_input_ids 的长度与路由数据的长度匹配
            # 路由数据是生成时累积的，包含了所有生成的token（自回归生成时每次forward处理1个token）
            routing_seq_len = actual_selected_experts.shape[1]
            
            # 如果 decoder_input_ids 的长度小于路由数据的长度，需要根据路由数据长度来扩展
            # 注意：我们不能简单地截断路由数据，因为这会丢失信息
            # 我们应该根据路由数据的长度，使用生成的文本重新tokenize（不truncate）
            if decoder_input_ids.shape[1] < routing_seq_len:
                import logging
                logging.getLogger("okr_detector").warning(f"decoder_input_ids长度({decoder_input_ids.shape[1]})小于路由数据长度({routing_seq_len})")
                logging.getLogger("okr_detector").info(f"将截断路由数据以匹配decoder_input_ids长度（可能丢失信息）")
                # 截断路由数据以匹配 decoder_input_ids 的长度
                # 注意：路由数据是累积的，包含了所有生成的token（自回归生成时每次forward处理1个token）
                # 对于encoder-decoder模型，router只在decoder层中，所以路由数据应该只包含decoder的token
                # 但我们仍然需要确保长度匹配
                decoder_seq_len = decoder_input_ids.shape[1]
                routing_seq_len = actual_selected_experts.shape[1]
                
                # 如果路由数据长度远大于decoder输入长度，说明可能包含了encoder的token或其他层的累积
                # 我们使用路由数据的最后decoder_seq_len个token（对应decoder输出）
                if routing_seq_len > decoder_seq_len * 2:
                    import logging
                    logging.getLogger("okr_detector").warning(f"路由数据长度({routing_seq_len})远大于decoder输入长度({decoder_seq_len})，可能包含了encoder的token")
                    # 使用最后decoder_seq_len个token
                    actual_selected_experts = actual_selected_experts[:, -decoder_seq_len:, :]
                else:
                    # 如果长度接近，直接截断到decoder_seq_len
                    actual_selected_experts = actual_selected_experts[:, :decoder_seq_len, :]
            elif decoder_input_ids.shape[1] > routing_seq_len:
                import logging
                logging.getLogger("okr_detector").warning(f"decoder_input_ids长度({decoder_input_ids.shape[1]})大于路由数据长度({routing_seq_len})，将截断decoder_input_ids")
                # 截断 decoder_input_ids 以匹配路由数据的长度
                decoder_input_ids = decoder_input_ids[:, :routing_seq_len]
            
            # 重新运行模型获取 decoder 的 hidden states（用于计算水印信号和机会窗口）
            # 重要：使用生成的文本序列作为 decoder_input_ids，确保 hidden_states 与路由数据对应
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    output_hidden_states=True
                )
            
            # 获取 decoder 的最后一层 hidden states
            if hasattr(outputs, 'decoder_hidden_states') and outputs.decoder_hidden_states:
                hidden_states = outputs.decoder_hidden_states[-1]
            elif hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                hidden_states = outputs.hidden_states[-1]
            else:
                return 0.0, "No hidden states available"
            
            # 确保 hidden_states 的形状与 selected_experts 匹配
            # selected_experts: [batch, seq, top_k] - 来自生成时的累积数据（已经截断为最后N个token）
            # hidden_states: [batch, seq, dim] - 来自重新运行的模型
            if hidden_states.shape[1] != actual_selected_experts.shape[1]:
                # 调整形状以匹配（取较小的长度）
                min_seq_len = min(hidden_states.shape[1], actual_selected_experts.shape[1])
                hidden_states = hidden_states[:, :min_seq_len, :]
                # 注意：actual_selected_experts已经是从路由数据最后截取的，所以这里也取前min_seq_len个
                actual_selected_experts = actual_selected_experts[:, :min_seq_len, :]
                import logging
                logging.getLogger("okr_detector").info(f"调整形状以匹配: hidden_states={hidden_states.shape}, selected_experts={actual_selected_experts.shape}")
        else:
            # 对于 decoder-only 模型（如 GPT）
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
            
            # 获取实际选择的专家
            actual_selected_experts = self._extract_selected_experts()
            
            if actual_selected_experts is None:
                return 0.0, "No routing data available"
        
        return self.verify_batch(hidden_states, actual_selected_experts)

    def verify_batch(self, hidden_states: torch.Tensor, actual_selected_experts: torch.Tensor) -> Tuple[float, str]:
        """
        核心验证逻辑 (纯数学计算)

        Args:
            hidden_states: [batch, seq, dim] - 重新推理得到的 Embedding
            actual_selected_experts: [batch, seq, top_k] - 也就是你想验证的 token 实际上走了哪条路
            
        Returns:
            score: 水印命中率 (0-1)
            verdict: "Watermarked" 或 "Clean"
        """
        # 获取 Router (假设模型只有一层或我们只看这一层)
        router = self._get_router()
        if router is None:
            return 0.0, "No router found"

        # 获取 OKR router（包含 gate_network 和 secret_projection）
        okr_router = None
        # 方式1: 从原始 router 对象获取 _okr_router
        if hasattr(router, '_okr_router'):
            okr_router = router._okr_router
        elif hasattr(router, 'gate_network') and hasattr(router, 'secret_projection'):
            # 方式2: router 本身可能就是 OKR router
            okr_router = router
        
        if okr_router is None:
            import logging
            logging.getLogger("okr_detector").error(f"检测时未找到 OKR router。router类型: {type(router).__name__}, router属性: {[attr for attr in dir(router) if not attr.startswith('__')][:10]}")
            return 0.0, "No OKR router found (watermark not injected?)"

        # 1. 重算原本的 Logits (Ground Truth)
        # 使用 OKR router 的 gate_network（它应该已经复制了原始权重）
        raw_logits = okr_router.gate_network(hidden_states)
        max_logits, _ = raw_logits.max(dim=-1, keepdim=True)

        # 2. 重算水印信号 (Expected Signal)
        # 使用 OKR router 的 secret_projection
        if okr_router.secret_projection.device != hidden_states.device:
            secret_proj = okr_router.secret_projection.to(hidden_states.device)
        else:
            secret_proj = okr_router.secret_projection
        
        # watermark_bias: [batch, seq, num_experts]
        watermark_bias = torch.matmul(hidden_states, secret_proj)
        
        # 添加调试信息：检查watermark_bias的形状和值
        import logging
        logging.getLogger("okr_detector").debug(f"watermark_bias形状: {watermark_bias.shape}, 范围: [{watermark_bias.min().item():.4f}, {watermark_bias.max().item():.4f}]")

        # 3. 重算机会窗口 (Opportunities)
        # 哪些 Token 拥有至少 2 个安全专家？
        safe_mask = raw_logits >= (max_logits - self.epsilon)
        num_safe_experts = safe_mask.sum(dim=-1)  # [batch, seq]

        # 只有 >= 2 个可选专家时，水印才有可能生效
        # 这就是我们的有效样本 (Valid Samples)
        valid_opportunity_mask = (num_safe_experts >= 2)

        if valid_opportunity_mask.sum() == 0:
            import logging
            logging.getLogger("okr_detector").warning(f"没有机会窗口 (epsilon={self.epsilon} 可能太小)")
            return 0.0, "No Opportunities (Text too clear or epsilon too small)"
        
        import logging
        logging.getLogger("okr_detector").info(f"有效机会窗口: {valid_opportunity_mask.sum().item()}/{valid_opportunity_mask.numel()}")

        # 4. 验证命中 (Check Hits)
        # 在这些机会窗口里，水印想要谁？
        # 水印想要的是在 safe_mask 范围内，watermark_bias 最大的那个

        masked_watermark_scores = torch.where(
            safe_mask,
            watermark_bias,
            torch.tensor(-1e9, device=watermark_bias.device, dtype=watermark_bias.dtype)
        )
        # 理论上水印指向的第一选择
        target_expert = torch.argmax(masked_watermark_scores, dim=-1)  # [batch, seq]

        # 5. 检查实际选择是否包含水印指向的专家
        # actual_selected_experts: [batch, seq, top_k]
        # 我们看 target_expert 是否在 top_k 里

        # 扩展 target_expert 维度以便比较: [batch, seq, 1]
        target_expert_expanded = target_expert.unsqueeze(-1)

        # hit: [batch, seq] (True/False)
        hits = (actual_selected_experts == target_expert_expanded).any(dim=-1)
        
        # 添加调试信息：检查前几个token的匹配情况
        import logging
        if valid_opportunity_mask.sum() > 0:
            # 获取前5个有效机会窗口的匹配情况
            valid_indices = torch.where(valid_opportunity_mask)[0][:5]
            for idx in valid_indices:
                idx_item = idx.item()
                target = target_expert[0, idx_item].item()
                actual = actual_selected_experts[0, idx_item, 0].item()
                hit = hits[0, idx_item].item()
                logging.getLogger("okr_detector").debug(f"Token {idx_item}: target_expert={target}, actual_expert={actual}, hit={hit}")

        # 6. 统计分数 (仅在有效样本上)
        valid_hits = hits[valid_opportunity_mask]
        score = valid_hits.float().mean().item()
        
        # 判断阈值：根据OKR论文，无水印文本的命中率应该接近 1/K（随机概率）
        # 对于8个专家，随机基线是 1/8 = 12.5%
        # 有水印文本的命中率应该显著高于随机基线
        # 使用动态阈值：随机基线的2-3倍
        num_experts = raw_logits.shape[-1]
        random_baseline = 1.0 / num_experts
        threshold = random_baseline * 2.0  # 2倍随机基线作为阈值
        
        # 添加调试信息
        import logging
        total_opportunities = valid_opportunity_mask.sum().item()
        total_hits = valid_hits.sum().item()
        logging.getLogger("okr_detector").info(f"命中率: {score:.4f} ({total_hits}/{total_opportunities}), 随机基线: {random_baseline:.4f}, 阈值: {threshold:.4f}")
        
        # 如果命中率低于随机基线，说明水印没有生效
        if score < random_baseline:
            logging.getLogger("okr_detector").warning(f"命中率({score:.4f})低于随机基线({random_baseline:.4f})，可能水印未生效")
        
        verdict = "Watermarked" if score > threshold else "Clean"
        return score, verdict
    
    def _get_router(self):
        """
        从模型中提取路由器
        
        由于我们只替换了 forward 方法，需要从 router 对象上获取 _okr_router
        """
        # 查找所有 router 对象，检查是否有 _okr_router 属性
        for name, module in self.model.named_modules():
            # 检查是否是 router（有 _okr_router 属性）
            # 返回原始的 router 对象（不是 _okr_router），因为我们需要访问它
            if hasattr(module, '_okr_router'):
                return module
            
            # 也检查是否是 OKRRouter（直接有 secret_projection 和 gate_network）
            if hasattr(module, 'secret_projection') and hasattr(module, 'gate_network'):
                return module
        
        return None
    
    def _extract_selected_experts(self) -> Optional[torch.Tensor]:
        """
        从模型中提取实际选择的专家
        
        这需要在forward过程中保存路由信息
        """
        # 优先从 router 的累积数据中获取（包含所有token）
        # 注意：可能有多个router（每个层一个），我们需要找到与检测时使用的router对应的那个
        # 或者合并所有router的数据（如果它们都包含相同的token）
        router_for_detection = self._get_router()
        if router_for_detection and hasattr(router_for_detection, '_okr_all_selected_experts'):
            all_experts = router_for_detection._okr_all_selected_experts
            if all_experts and len(all_experts) > 0:
                # 拼接所有token的数据
                # 每个元素: [batch, seq_len, top_k]，通常是 [batch, 1, top_k]
                # 拼接后: [batch, total_seq_len, top_k]
                concatenated = torch.cat(all_experts, dim=1)  # 在seq_len维度拼接
                import logging
                logging.getLogger("okr_detector").info(f"从 router._okr_all_selected_experts 获取: {len(all_experts)} tokens, 拼接后形状: {concatenated.shape}")
                return concatenated
        
        # 如果没有找到，尝试从所有router中找第一个
        for name, module in self.model.named_modules():
            if hasattr(module, '_okr_all_selected_experts'):
                all_experts = module._okr_all_selected_experts
                if all_experts and len(all_experts) > 0:
                    # 拼接所有token的数据
                    concatenated = torch.cat(all_experts, dim=1)
                    import logging
                    logging.getLogger("okr_detector").info(f"从 router._okr_all_selected_experts 获取 (fallback): {len(all_experts)} tokens, 拼接后形状: {concatenated.shape}")
                    return concatenated
        
        # 尝试从模型的属性中获取（按层存储的字典）
        if hasattr(self.model, '_okr_routing_data'):
            routing_data = self.model._okr_routing_data
            if isinstance(routing_data, dict):
                # 取第一层的数据（或者合并所有层）
                for layer_id, layer_data in routing_data.items():
                    if layer_data and len(layer_data) > 0:
                        # 拼接该层所有token的数据
                        concatenated = torch.cat(layer_data, dim=1)
                        import logging
                        logging.getLogger("okr_detector").info(f"从 model._okr_routing_data[{layer_id}] 获取: {len(layer_data)} tokens, 拼接后形状: {concatenated.shape}")
                        return concatenated
            elif isinstance(routing_data, list) and len(routing_data) > 0:
                # 兼容旧格式：返回最后一个
                return routing_data[-1]
            elif isinstance(routing_data, torch.Tensor):
                return routing_data
        
        # 尝试从各个 router 层中获取（单个token，用于兼容）
        for name, module in self.model.named_modules():
            if hasattr(module, '_selected_experts'):
                selected = module._selected_experts
                if isinstance(selected, torch.Tensor):
                    import logging
                    logging.getLogger("okr_detector").warning(f"从 router._selected_experts 获取（可能不完整）: {selected.shape}")
                    return selected
        
        import logging
        logging.getLogger("okr_detector").warning("未找到路由数据")
        return None

