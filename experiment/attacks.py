import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List

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

def paraphrase_text_batch(text_list: List[str]) -> List[str]:
    """
    对一批文本进行释义攻击。
    """
    _load_paraphrase_model()
    
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
            num_beams=5,
            num_return_sequences=1,
            early_stopping=True
        )
        
    paraphrased_texts = _paraphrase_tokenizer.batch_decode(
        outputs, 
        skip_special_tokens=True
    )
    
    # T5 可能返回多个，我们只取第一个
    # (TODO: 修复num_return_sequences=1 时的 batch 处理)
    # 临时的 batch 修复
    final_texts = []
    for i in range(len(text_list)):
        # 假设 num_return_sequences=1
        final_texts.append(paraphrased_texts[i])
        
    return final_texts

def estimate_gamma_from_text(
    text_original: str, 
    text_attacked: str, 
    vocab_size: int
) -> float:
    """
    估算攻击强度 γ (对标文稿 7.4.1 节)
    使用 KL 散度的上界：γ ≈ (L/N) * H(V)
    """
    # 这是一个粗略的 token-level 编辑距离
    tokens_orig = text_original.split()
    tokens_atk = text_attacked.split()
    
    L = abs(len(tokens_orig) - len(tokens_atk)) # 插入/删除
    
    # 替换
    for t1, t2 in zip(tokens_orig, tokens_atk):
        if t1 != t2:
            L += 1
            
    N = max(len(tokens_orig), 1)
    
    # H(V) = log|V| (以 nats 为单位)
    H_V = np.log(vocab_size)
    
    gamma = (L / N) * H_V
    
    return float(gamma)