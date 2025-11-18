import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict

# 全局缓存释义模型
_paraphrase_model = None
_paraphrase_tokenizer = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

def _load_paraphrase_model():
    """辅助函数：加载 T5 释义模型"""
    global _paraphrase_model, _paraphrase_tokenizer
    if _paraphrase_model is None:
        print("Loading paraphrase model (t5-base)...")
        model_name = "Vamsi/T5_Paraphrase" # 使用一个标准的 T5 释义模型
        _paraphrase_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _paraphrase_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(_device)
        _paraphrase_model.eval()
        print("Paraphrase model loaded.")

def paraphrase_text_batch(
    text_list: List[str], 
    attack_strength: str = "moderate"
) -> List[str]:
    """
    对一批文本进行释义攻击 (论文定义1.3: 释义攻击)
    
    攻击满足:
    1. 语义保持: cos(BERT(x), BERT(x')) > τ_semantic (通常0.85-0.95)
    2. 编辑距离约束: ED(x, x') ≤ L (通常L ≤ 5或L ≤ 0.1×|x|)
    
    Args:
        text_list: 原始文本列表
        attack_strength: 攻击强度 ("mild", "moderate", "strong")
            - mild: 编辑距离 ~2-3
            - moderate: 编辑距离 ~3-5
            - strong: 编辑距离 ~5-8
            
    Returns:
        paraphrased_texts: 释义后的文本列表
    """
    _load_paraphrase_model()
    
    # 根据攻击强度调整生成参数
    if attack_strength == "mild":
        num_beams = 3
        temperature = 0.7
    elif attack_strength == "moderate":
        num_beams = 5
        temperature = 0.8
    elif attack_strength == "strong":
        num_beams = 7
        temperature = 0.9
    else:
        num_beams = 5
        temperature = 0.8
    
    # T5 需要一个前缀
    inputs = _paraphrase_tokenizer(
        [f"paraphrase: {text}" for text in text_list],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(_device)
    
    with torch.no_grad():
        outputs = _paraphrase_model.generate(
            **inputs,
            max_length=512,
            num_beams=num_beams,
            num_return_sequences=1,
            early_stopping=True,
            temperature=temperature,
            do_sample=(temperature > 0.7)
        )
        
    paraphrased_texts = _paraphrase_tokenizer.batch_decode(
        outputs, 
        skip_special_tokens=True
    )
    
    # 处理batch输出
    final_texts = []
    for i in range(len(text_list)):
        if i < len(paraphrased_texts):
            final_texts.append(paraphrased_texts[i])
        else:
            # 如果输出不足, 返回原文本
            final_texts.append(text_list[i])
        
    return final_texts

def estimate_gamma_from_text(
    text_original: str, 
    text_attacked: str, 
    vocab_size: int,
    method: str = "upper_bound",
    use_semantic_correction: bool = False
) -> float:
    """
    估算攻击强度 γ (论文第7.4节: 攻击强度γ的上界估计)
    
    方法1 (论文引理7.1): 基于编辑距离的上界
    γ_upper = (L/N) * log(|V|/|V_freq|) ≈ (L/N) * log(|V|/10)
    
    方法2: 基于实际KL散度的精确估计
    γ_KL = D_KL(D(X') || D(X))
    
    Args:
        text_original: 原始文本
        text_attacked: 攻击后文本
        vocab_size: 词汇表大小 |V|
        method: 估计方法 ("upper_bound" 或 "kl_divergence")
        
    Returns:
        gamma: 攻击强度 (nats)
    """
    if method == "upper_bound":
        # 方法1: 编辑距离上界 (论文引理7.1)
        tokens_orig = text_original.split()
        tokens_atk = text_attacked.split()
        
        # 计算编辑距离 L
        L = abs(len(tokens_orig) - len(tokens_atk))  # 插入/删除
        
        # 替换操作
        min_len = min(len(tokens_orig), len(tokens_atk))
        for i in range(min_len):
            if tokens_orig[i] != tokens_atk[i]:
                L += 1
        
        N = max(len(tokens_orig), 1)
        
        # H(V) = log|V|, 但考虑常用词集合 (论文引理7.1)
        # 对于paraphrase攻击, 替换主要发生在低频词
        # |V_freq| ≈ |V|/10
        H_V_effective = np.log(vocab_size / 10.0)
        
        gamma = (L / N) * H_V_effective
        
        return float(gamma)
        
    elif method == "kl_divergence":
        # 方法2: 精确KL散度估计 (论文定义1.3)
        # 需要token级别的分布
        tokens_orig = text_original.split()
        tokens_atk = text_attacked.split()
        
        # 计算token频率分布
        from collections import Counter
        
        count_orig = Counter(tokens_orig)
        count_atk = Counter(tokens_atk)
        
        # 归一化为概率分布
        total_orig = sum(count_orig.values())
        total_atk = sum(count_atk.values())
        
        if total_orig == 0 or total_atk == 0:
            return 0.0
        
        # 计算KL散度: D_KL(P_atk || P_orig)
        # 使用词汇表的并集
        all_tokens = set(count_orig.keys()) | set(count_atk.keys())
        
        kl_sum = 0.0
        for token in all_tokens:
            p_orig = count_orig.get(token, 0) / total_orig
            p_atk = count_atk.get(token, 0) / total_atk
            
            # 防止log(0)
            if p_atk > 0 and p_orig > 0:
                kl_sum += p_atk * np.log(p_atk / p_orig)
            elif p_atk > 0 and p_orig == 0:
                # 如果p_orig=0但p_atk>0, KL散度无穷大
                # 使用平滑: p_orig = 1/(total_orig + vocab_size)
                p_orig_smooth = 1.0 / (total_orig + vocab_size)
                kl_sum += p_atk * np.log(p_atk / p_orig_smooth)
        
        return float(kl_sum)
        
    elif method == "hybrid":
        # 方法3: 混合策略（1118文档推荐）
        # 结合编辑距离上界和实测KL散度
        gamma_upper = estimate_gamma_from_text(
            text_original, text_attacked, vocab_size, method="upper_bound"
        )
        gamma_kl = estimate_gamma_from_text(
            text_original, text_attacked, vocab_size, method="kl_divergence"
        )
        # 取平均值，兼顾保守性和准确性
        gamma = (gamma_upper + gamma_kl) / 2.0
        return float(gamma)
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'upper_bound', 'kl_divergence', or 'hybrid'")


def estimate_gamma_with_semantic_correction(
    text_original: str,
    text_attacked: str,
    vocab_size: int,
    model_name: str = "bert-base-uncased"
) -> Dict[str, float]:
    """
    使用BERT语义相似度补正的攻击强度估计（1118文档方法2）
    
    三层估计策略：
    1. 编辑距离上界（最保守）
    2. BERT语义相似度补正（中等）
    3. 实测KL散度（最精确）
    
    Args:
        text_original: 原始文本
        text_attacked: 攻击后文本
        vocab_size: 词汇表大小
        model_name: BERT模型名称
        
    Returns:
        estimates: 包含各种估计方法的字典
    """
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch.nn.functional as F
        
        # 方法1: 编辑距离上界
        gamma_upper = estimate_gamma_from_text(
            text_original, text_attacked, vocab_size, method="upper_bound"
        )
        
        # 方法2: BERT语义相似度补正
        bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
        bert_model = AutoModel.from_pretrained(model_name)
        bert_model.eval()
        
        def get_bert_embedding(text: str) -> torch.Tensor:
            """获取BERT embedding"""
            inputs = bert_tokenizer(
                text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            with torch.no_grad():
                outputs = bert_model(**inputs)
            # 使用[CLS] token的隐藏状态
            return outputs.last_hidden_state[:, 0, :].squeeze()
        
        emb_original = get_bert_embedding(text_original)
        emb_attacked = get_bert_embedding(text_attacked)
        
        # 计算余弦相似度
        semantic_similarity = F.cosine_similarity(
            emb_original.unsqueeze(0),
            emb_attacked.unsqueeze(0)
        ).item()
        
        # KL散度与语义相似度的关系
        # 当sim接近1时，KL应较小；当sim降低时，KL增加
        semantic_penalty = 1.0 - semantic_similarity
        gamma_semantic_adjusted = gamma_upper * (1 + semantic_penalty)
        
        # 方法3: 实测KL散度
        gamma_measured = estimate_gamma_from_text(
            text_original, text_attacked, vocab_size, method="kl_divergence"
        )
        
        # 混合策略（推荐）
        gamma_hybrid = (gamma_upper + gamma_measured) / 2.0
        
        return {
            'method_1_edit_distance': {
                'gamma': gamma_upper,
                'type': 'upper_bound',
                'conservativeness': 'high'
            },
            'method_2_semantic': {
                'gamma': gamma_semantic_adjusted,
                'semantic_similarity': semantic_similarity,
                'type': 'upper_bound_refined'
            },
            'method_3_measured_kl': {
                'gamma': gamma_measured,
                'type': 'measured',
                'conservativeness': 'none'
            },
            'hybrid': {
                'gamma': gamma_hybrid,
                'reasoning': 'average of upper_bound and measured'
            },
            'recommended_gamma': gamma_hybrid,
            'upper_bound': gamma_upper,
            'lower_bound': gamma_measured
        }
        
    except ImportError:
        print("警告: transformers未安装，无法使用BERT语义补正，回退到基础方法")
        return {
            'method_1_edit_distance': {
                'gamma': estimate_gamma_from_text(
                    text_original, text_attacked, vocab_size, method="upper_bound"
                ),
                'type': 'upper_bound'
            },
            'method_3_measured_kl': {
                'gamma': estimate_gamma_from_text(
                    text_original, text_attacked, vocab_size, method="kl_divergence"
                ),
                'type': 'measured'
            },
            'recommended_gamma': estimate_gamma_from_text(
                text_original, text_attacked, vocab_size, method="hybrid"
            )
        }