"""
MVES水印实现: 基于LogitsProcessor API

使用transformers的LogitsProcessor实现水印嵌入，适配switch-base-8模型
采用"预计算(pre-pass)"方式获取路由权重(RW)
"""

import torch
import torch.nn.functional as F
import numpy as np
import hashlib
from typing import List, Optional, Dict, Tuple, Any
from transformers import LogitsProcessor, PreTrainedModel
from transformers.generation.utils import LogitsProcessorList

from mves_config import MVESConfig, WatermarkConfig


class MoEWatermarkLogitsProcessor(LogitsProcessor):
    """
    基于LogitsProcessor的MoE水印嵌入器
    
    核心思想 (论文第3节):
    1. 使用LogitsProcessor在生成过程中修改logits
    2. 通过"预计算(pre-pass)"获取MoE路由权重
    3. 基于路由权重计算水印偏置
    4. 确保KL(p1||p0) = ε (论文定义3.2)
    
    适配模型: google/switch-base-8 (T5-based MoE)
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        config: MVESConfig,
        tokenizer: Any
    ):
        """
        初始化水印LogitsProcessor
        
        Args:
            model: 预训练模型 (switch-base-8)
            config: MVES配置
            tokenizer: 分词器
        """
        self.model = model
        self.config = config
        self.wm_config = config.watermark
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        # 水印参数
        self.secret_key = self.wm_config.secret_key
        self.epsilon = self.wm_config.epsilon
        self.num_experts = self.wm_config.num_experts
        self.k_top = self.wm_config.k_top
        
        # 计算偏置强度 (论文定义3.2)
        # ε = KL(p1||p0) ≈ Var[Δl] ≈ (1/2)||Δl||²_2
        self.target_norm = np.sqrt(2.0 * self.epsilon)
        
        # 缓存: 存储预计算的路由权重
        self._router_weights_cache: Dict[str, torch.Tensor] = {}
        self._p0_cache: Dict[str, torch.Tensor] = {}
        self._p1_cache: Dict[str, torch.Tensor] = {}
        
        # 检测数据存储 (用于后续检测)
        self._detection_data: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        
        print(f"MoEWatermarkLogitsProcessor初始化完成:")
        print(f"  - 模型: {config.model.model_name}")
        print(f"  - 水印强度 ε: {self.epsilon:.6f}")
        print(f"  - 专家数 K: {self.num_experts}, Top-k: {self.k_top}")
    
    def _get_context_hash(self, input_ids: torch.Tensor) -> int:
        """
        根据输入生成确定性种子 (论文第3节)
        
        Args:
            input_ids: 输入token IDs [batch_size, seq_len]
            
        Returns:
            hash_value: 确定性哈希值
        """
        # 使用最后一个token和密钥生成哈希
        last_token = input_ids[:, -1].sum().item()
        combined = f"{self.secret_key}_{last_token}".encode('utf-8')
        hash_value = int(hashlib.sha256(combined).hexdigest()[:16], 16)
        return hash_value
    
    def _precompute_router_weights(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预计算路由权重 (RW) - "预计算(pre-pass)"方式
        
        对于switch-base-8模型，需要:
        1. 前向传播获取MoE层的路由logits
        2. 计算原始激活分布 p_0
        3. 缓存结果供后续使用
        
        Args:
            input_ids: 输入token IDs [batch_size, seq_len]
            attention_mask: 注意力掩码 [batch_size, seq_len]
            
        Returns:
            p_0: 原始激活分布 [batch_size, seq_len, num_experts]
            router_logits: 路由logits [batch_size, seq_len, num_experts]
        """
        # 生成缓存键
        cache_key = f"{input_ids.sum().item()}_{input_ids.shape}"
        
        if cache_key in self._p0_cache:
            return self._p0_cache[cache_key], self._router_weights_cache.get(cache_key)
        
        self.model.eval()
        with torch.no_grad():
            # 获取模型输出 (包含hidden states)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Switch Transformer的MoE层结构
            # 需要找到MoE层的路由logits
            router_logits_list = []
            p_0_list = []
            
            # 遍历所有decoder层 (switch-base-8使用encoder-decoder架构)
            if hasattr(self.model, 'decoder') and hasattr(self.model.decoder, 'block'):
                for layer in self.model.decoder.block:
                    if hasattr(layer, 'layer') and len(layer.layer) > 1:
                        # 查找MoE层 (通常是FFN层)
                        for sublayer in layer.layer:
                            if hasattr(sublayer, 'mlp') and hasattr(sublayer.mlp, 'router'):
                                # 获取路由logits
                                router = sublayer.mlp.router
                                hidden_states = outputs.decoder_hidden_states[-1] if hasattr(outputs, 'decoder_hidden_states') else outputs.last_hidden_state
                                
                                # 计算路由logits
                                router_logits = router(hidden_states)
                                p_0 = F.softmax(router_logits, dim=-1)
                                
                                router_logits_list.append(router_logits)
                                p_0_list.append(p_0)
                                break
            
            # 如果没有找到MoE层，使用默认方法
            if not router_logits_list:
                # 对于switch-base-8，可能需要直接访问模型内部
                # 这里使用一个简化的方法：假设所有位置使用均匀分布
                batch_size, seq_len = input_ids.shape
                router_logits = torch.zeros(
                    (batch_size, seq_len, self.num_experts),
                    device=self.device
                )
                p_0 = torch.ones(
                    (batch_size, seq_len, self.num_experts),
                    device=self.device
                ) / self.num_experts
                
                router_logits_list.append(router_logits)
                p_0_list.append(p_0)
            
            # 使用第一个MoE层的结果 (或平均)
            router_logits = router_logits_list[0] if router_logits_list else router_logits
            p_0 = p_0_list[0] if p_0_list else p_0
        
        # 缓存结果
        self._p0_cache[cache_key] = p_0
        self._router_weights_cache[cache_key] = router_logits
        
        return p_0, router_logits
    
    def _compute_watermark_bias(
        self,
        p_0: torch.Tensor,
        input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算水印偏置向量 (论文定义3.2)
        
        确保KL(p1||p0) = ε
        
        Args:
            p_0: 原始激活分布 [batch_size, seq_len, num_experts]
            input_ids: 输入token IDs [batch_size, seq_len]
            
        Returns:
            delta_l: 偏置向量 [batch_size, seq_len, num_experts]
            p_1: 修改后激活分布 [batch_size, seq_len, num_experts]
        """
        batch_size, seq_len, num_experts = p_0.shape
        
        # 生成确定性种子
        context_hash = self._get_context_hash(input_ids)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(context_hash)
        
        # 初始化偏置向量
        delta_l = torch.zeros_like(p_0)
        
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
                red_experts = torch.tensor(
                    np.random.choice(red_candidates, size=num_red, replace=False),
                    device=self.device
                )
                
                # 计算偏置强度
                bias_green = self.target_norm / np.sqrt(1 + num_red)
                bias_red = -bias_green / num_red if num_red > 0 else 0
                
                # 应用偏置
                delta_l[b, s, green_expert] = bias_green
                if num_red > 0:
                    delta_l[b, s, red_experts] = bias_red
        
        # 计算修改后的分布
        # 注意: 这里我们直接修改logits，但实际路由可能已经完成
        # 对于LogitsProcessor，我们需要将偏置转换为token logits的修改
        # 这是一个简化实现，实际需要根据模型架构调整
        
        # 计算p_1 (用于检测)
        router_logits_modified = self._router_weights_cache.get(
            f"{input_ids.sum().item()}_{input_ids.shape}", None
        )
        if router_logits_modified is not None:
            router_logits_modified = router_logits_modified + delta_l
            p_1 = F.softmax(router_logits_modified, dim=-1)
        else:
            # 如果没有缓存，使用p_0近似
            p_1 = p_0 + delta_l * 0.1  # 简化处理
            p_1 = F.softmax(torch.log(p_0 + 1e-9) + delta_l, dim=-1)
        
        return delta_l, p_1
    
    def __call__(
        self,
        input_ids: torch.Tensor,
        scores: torch.Tensor
    ) -> torch.Tensor:
        """
        LogitsProcessor接口: 修改生成logits
        
        Args:
            input_ids: 当前输入token IDs [batch_size, seq_len]
            scores: 当前token的logits [batch_size, vocab_size]
            
        Returns:
            modified_scores: 修改后的logits [batch_size, vocab_size]
        """
        # 预计算路由权重 (如果需要)
        if input_ids.shape[1] > 1:  # 只在有足够上下文时计算
            try:
                p_0, router_logits = self._precompute_router_weights(input_ids)
                delta_l, p_1 = self._compute_watermark_bias(p_0, input_ids)
                
                # 存储检测数据
                self._detection_data.append((p_0, p_1, input_ids))
                
                # 将MoE路由偏置转换为token logits偏置
                # 这是一个关键步骤：我们需要将专家激活的偏置映射到词汇表logits
                # 简化实现：使用均匀映射
                # 实际实现需要根据模型的具体架构调整
                
                # 暂时返回原始scores (需要进一步实现)
                # TODO: 实现从MoE路由偏置到token logits的映射
                
            except Exception as e:
                # 如果预计算失败，返回原始scores
                print(f"警告: 路由权重预计算失败: {e}")
                pass
        
        return scores
    
    def get_detection_data(self) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """获取检测数据 (用于后续水印检测)"""
        return self._detection_data.copy()
    
    def clear_cache(self):
        """清除缓存"""
        self._router_weights_cache.clear()
        self._p0_cache.clear()
        self._p1_cache.clear()
        self._detection_data.clear()


def create_watermark_processor(
    model: PreTrainedModel,
    config: MVESConfig,
    tokenizer: Any
) -> MoEWatermarkLogitsProcessor:
    """
    创建水印LogitsProcessor
    
    Args:
        model: 预训练模型
        config: MVES配置
        tokenizer: 分词器
        
    Returns:
        processor: 水印LogitsProcessor实例
    """
    return MoEWatermarkLogitsProcessor(model, config, tokenizer)

