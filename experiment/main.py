# 设置缓存路径（在导入transformers之前）
# 自动检测环境：WSL/Linux 或 Windows
import os
import platform
from pathlib import Path

def detect_and_setup_cache():
    """自动检测环境并设置缓存路径"""
    # 检查是否在 WSL 或 Linux 环境
    release = platform.release().lower()
    is_wsl = (
        "microsoft" in release or 
        "wsl" in release
    )
    # 检查 /proc/version（如果存在）
    if not is_wsl and os.path.exists("/proc/version"):
        try:
            with open("/proc/version", "r") as f:
                proc_version = f.read().lower()
                is_wsl = "microsoft" in proc_version
        except (IOError, PermissionError):
            pass
    is_linux = platform.system() == "Linux"
    
    # 优先使用环境变量指定的配置
    if os.environ.get("USE_WSL_CONFIG") == "1" or (is_wsl or is_linux):
        # WSL/Linux 环境：使用 Linux 路径
        try:
            import cache_config_wsl
            return  # cache_config_wsl 已经设置了所有环境变量
        except ImportError:
            # 如果导入失败，使用默认 Linux 路径
            CACHE_BASE = Path.home() / ".cache" / "emb_attack_separa"
            CACHE_BASE.mkdir(parents=True, exist_ok=True)
    else:
        # Windows 环境：使用 Windows 路径
        try:
            import cache_config
            return  # cache_config 已经设置了所有环境变量
        except ImportError:
            # 如果导入失败，使用默认 Windows 路径
            CACHE_BASE = Path("D:/Dev/cache")
    
    # 设置环境变量（如果导入配置模块失败）
    os.environ.setdefault("HF_HOME", str(CACHE_BASE / "huggingface"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_BASE / "huggingface" / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(CACHE_BASE / "huggingface" / "datasets"))
    os.environ.setdefault("TORCH_HOME", str(CACHE_BASE / "torch"))
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 执行缓存配置
detect_and_setup_cache()

import argparse
import torch
import gc
import sys
import threading
import atexit
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset
import json

def cleanup_resources(verbose=False):
    """清理所有资源，确保程序能快速退出"""
    if verbose:
        print("开始清理资源...")
    
    # 1. 清理 CUDA 缓存和上下文
    # 注意：使用全局的 torch 模块，不要在这里导入
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            # 尝试释放CUDA上下文（激进方法）
            try:
                torch.cuda.ipc_collect()
            except:
                pass
    except Exception as e:
        if verbose:
            print(f"CUDA清理警告: {e}")
    
    # 2. 关闭transformers的文件句柄和缓存
    try:
        # 尝试关闭transformers的文件系统缓存
        from transformers.file_utils import cached_path
        # 清除transformers的缓存管理器
        import transformers
        if hasattr(transformers, '_file_cache'):
            transformers._file_cache = {}
    except:
        pass
    
    # 3. 强制垃圾回收（多次以确保彻底）
    for _ in range(5):  # 增加到5次
        collected = gc.collect()
        if verbose and collected > 0:
            print(f"  垃圾回收: 释放了 {collected} 个对象")
    
    # 4. 检查并处理后台线程
    if verbose:
        threads_before = [t for t in threading.enumerate() if t != threading.main_thread() and t.is_alive()]
        if threads_before:
            print(f"检测到 {len(threads_before)} 个后台线程:")
            for t in threads_before:
                print(f"  - {t.name} (daemon={t.daemon}, alive={t.is_alive()})")
    
    # 5. 处理非守护线程：尝试将它们标记为守护线程
    non_daemon_threads = []
    for thread in threading.enumerate():
        if thread != threading.main_thread() and thread.is_alive():
            if not thread.daemon:
                non_daemon_threads.append(thread)
                # 尝试将线程标记为守护线程（允许在后台运行）
                try:
                    thread.daemon = True
                    if verbose:
                        print(f"  将线程 '{thread.name}' 标记为守护线程")
                except (RuntimeError, AttributeError):
                    # 某些线程在启动后无法修改daemon属性（如Thread-auto_conversion）
                    if verbose:
                        print(f"  警告: 无法将线程 '{thread.name}' 标记为守护线程（将在退出时强制终止）")
    
    # 6. 等待非守护线程完成（最多等待0.2秒）
    for thread in non_daemon_threads:
        thread.join(timeout=0.2)
    
    # 7. 最后一次垃圾回收
    gc.collect()
    
    if verbose:
        threads_after = [t for t in threading.enumerate() if t != threading.main_thread() and t.is_alive()]
        if threads_after:
            daemon_count = sum(1 for t in threads_after if t.daemon)
            non_daemon_count = sum(1 for t in threads_after if not t.daemon)
            if non_daemon_count > 0:
                print(f"仍有 {non_daemon_count} 个非守护线程未退出（将强制终止）:")
                for t in threads_after:
                    if not t.daemon:
                        print(f"  - {t.name} (daemon={t.daemon})")
            if daemon_count > 0:
                print(f"  ({daemon_count} 个守护线程仍在运行，不会阻止退出)")

# 注册退出时的清理函数
atexit.register(cleanup_resources)

import numpy as np
from mves_watermark_corrected import patch_switch_model_with_watermark
from calibration import calibrate_Lg, calibrate_C, calibrate_C_star
from detector import LLRDetector
from attacks import paraphrase_text_batch, estimate_gamma_from_text
from experiments import run_all_experiments, ExperimentA, ExperimentC, ExperimentD, ExperimentE

def calculate_text_ppl(model, tokenizer, text: str, device: str) -> float:
    """
    计算单个文本的困惑度 (Perplexity, PPL)
    
    对于T5/Switch Transformers（encoder-decoder模型），正确的方法是：
    1. 将文本作为decoder输入（自回归）
    2. encoder输入可以是文本本身或空
    3. 计算模型对文本的负对数似然
    4. PPL = exp(平均负对数似然)
    
    Args:
        model: 模型（google/switch-base-8）
        tokenizer: 分词器
        text: 待评估的文本
        device: 设备
        
    Returns:
        ppl: 困惑度值
    """
    model.eval()
    
    # 将文本编码为token IDs
    tokens = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    
    if tokens.size(1) < 2:
        return 1000.0  # 文本太短，返回一个高值
    
    with torch.no_grad():
        try:
            # 对于T5/Switch Transformers，正确的方法是：
            # 1. encoder输入：使用文本本身（或空，但T5需要encoder输入）
            # 2. decoder输入：从start token开始，逐步添加token
            # 3. 计算每个位置的loss
            
            # T5的decoder需要一个start token
            # Switch Transformers使用pad_token_id作为decoder的start token
            decoder_start_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            
            # 方法：使用模型的forward方法，传入labels来计算loss
            # 这是最准确的方法，因为模型内部会正确处理encoder-decoder结构
            
            # encoder输入：使用文本本身
            encoder_input_ids = tokens.clone()
            encoder_attention_mask = torch.ones_like(encoder_input_ids)
            
            # decoder输入：从start token开始
            decoder_input_ids = torch.cat([
                torch.full((1, 1), decoder_start_token_id, dtype=torch.long, device=device),
                tokens  # 完整的文本作为decoder输入
            ], dim=1)
            
            # labels：用于计算loss（需要shift，因为decoder预测下一个token）
            # labels应该与decoder_input_ids对齐，但需要shift
            labels = decoder_input_ids[:, 1:].clone()  # 去掉第一个start token
            # 在末尾添加pad token以匹配长度
            labels = torch.cat([
                labels,
                torch.full((1, 1), tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100, dtype=torch.long, device=device)
            ], dim=1)
            
            # 调用模型计算loss
            outputs = model(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask,
                decoder_input_ids=decoder_input_ids,
                labels=labels
            )
            
            # 获取loss
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                loss = outputs.loss.item()
                ppl = np.exp(loss)
            else:
                # 如果没有loss，手动计算
                logits = outputs.logits  # [batch_size, seq_len, vocab_size]
                
                # 对齐logits和labels
                # logits[i]预测的是decoder_input_ids[i+1]，即labels[i]
                if logits.size(1) == labels.size(1):
                    shift_logits = logits.view(-1, logits.size(-1))
                    shift_labels = labels.view(-1)
                    loss_fct = torch.nn.CrossEntropyLoss(
                        ignore_index=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100
                    )
                    loss = loss_fct(shift_logits, shift_labels).item()
                    ppl = np.exp(loss)
                else:
                    # 长度不匹配，使用手动计算每个token的概率
                    total_nll = 0.0
                    num_tokens = 0
                    
                    # logits的每个位置对应预测下一个token
                    min_len = min(logits.size(1), labels.size(1))
                    for i in range(min_len):
                        token_id = labels[0, i].item()
                        if token_id != tokenizer.pad_token_id and token_id != -100:
                            log_probs = torch.nn.functional.log_softmax(logits[0, i, :], dim=-1)
                            token_log_prob = log_probs[token_id].item()
                            total_nll -= token_log_prob
                            num_tokens += 1
                    
                    if num_tokens > 0:
                        avg_nll = total_nll / num_tokens
                        ppl = np.exp(avg_nll)
                    else:
                        ppl = 1000.0
            
        except Exception as e:
            # 如果上述方法失败，尝试更简单的方法
            try:
                # 简化方法：只使用decoder部分
                decoder_start_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
                
                # 创建decoder输入和labels
                decoder_input_ids = torch.cat([
                    torch.full((1, 1), decoder_start_token_id, dtype=torch.long, device=device),
                    tokens[:, :-1]  # 去掉最后一个token
                ], dim=1)
                
                labels = tokens.clone()  # 完整的文本作为labels
                
                # encoder输入：使用文本本身
                encoder_input_ids = tokens.clone()
                encoder_attention_mask = torch.ones_like(encoder_input_ids)
                
                outputs = model(
                    input_ids=encoder_input_ids,
                    attention_mask=encoder_attention_mask,
                    decoder_input_ids=decoder_input_ids,
                    labels=labels
                )
                
                if hasattr(outputs, 'loss') and outputs.loss is not None:
                    loss = outputs.loss.item()
                    ppl = np.exp(loss)
                else:
                    # 手动计算
                    logits = outputs.logits
                    shift_logits = logits.view(-1, logits.size(-1))
                    shift_labels = labels[:, 1:].view(-1)  # shift labels
                    loss_fct = torch.nn.CrossEntropyLoss(
                        ignore_index=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100
                    )
                    loss = loss_fct(shift_logits, shift_labels).item()
                    ppl = np.exp(loss)
                    
            except Exception as e2:
                print(f"    PPL计算错误: {e2}")
                # 返回一个合理的默认值
                ppl = 1000.0
    
    # 限制PPL在合理范围内
    # 注意：不要设置最小值，因为可能确实很低
    ppl = max(0.1, min(ppl, 10000.0))
    return float(ppl)

def load_model_and_tokenizer(model_name: str, device: str):
    """
    加载模型和分词器（统一使用switch-base-8）
    
    针对GTX 4050优化：
    - 使用FP16而非bfloat16（更好的兼容性）
    - 限制显存使用
    """
    print(f"Loading model and tokenizer: {model_name}...")
    
    # 统一使用switch-base-8（Seq2Seq模型）
    from transformers import AutoModelForSeq2SeqLM
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # GTX 4050优化配置
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # FP16节省显存
        device_map="auto",
        low_cpu_mem_usage=True,
        max_memory={0: "5GB"} if torch.cuda.is_available() else None
    )
    
    print("Model and tokenizer loaded.")
    print(f"显存使用: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB" if torch.cuda.is_available() else "")
    
    return model, tokenizer

def get_dataloader(dataset_name: str, split: str, tokenizer: AutoTokenizer, batch_size: int, num_samples: int):
    print(f"Loading dataset: {dataset_name} (split: {split})...")
    dataset = load_dataset(dataset_name, name="wikitext-103-v1", split=split) # 示例
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    # 截取子集
    subset_dataset = Subset(tokenized_dataset, range(min(num_samples, len(tokenized_dataset))))
    
    dataloader = DataLoader(subset_dataset, batch_size=batch_size)
    print(f"Dataset loaded with {len(subset_dataset)} samples.")
    return dataloader

def main():
    parser = argparse.ArgumentParser(description="MoE Provably Robust Watermark Project")
    parser.add_argument("--mode", type=str, required=True, 
                       choices=["calibrate", "embed", "detect", "experiment"], help="操作模式")
    parser.add_argument("--model_name", type=str, default="google/switch-base-8", help="要使用的 MoE 模型（统一使用switch-base-8）")
    parser.add_argument("--dataset_name", type=str, default="wikitext", help="用于标定的数据集")
    parser.add_argument("--dataset_split", type=str, default="train", help="数据集分片")
    parser.add_argument("--num_calib_samples", type=int, default=100, help="用于标定的样本数量")
    parser.add_argument("--batch_size", type=int, default=4, help="标定时的 batch size")
    
    # Embed & Detect
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="用于生成的提示")
    parser.add_argument("--text_to_check", type=str, help="用于检测的文本")
    parser.add_argument("--secret_key", type=str, default="DEFAULT_SECRET_KEY", help="水印密钥")
    parser.add_argument("--attack", type=str, choices=["none", "paraphrase"], default="none", help="在检测前施加的攻击")
    
    # Watermark Params (可被 calibrate 覆盖)
    parser.add_argument("--gamma_design", type=float, default=0.01, help="设计的攻击强度 γ (默认0.01，较小值可提高文本质量)")
    parser.add_argument("--C_system", type=float, default=1.5, help="系统常数 C (来自标定)")
    parser.add_argument("--c_star", type=float, default=2.0, help="安全系数 c* (来自标定)")
    parser.add_argument("--tau_alpha", type=float, default=5.0, help="LLR 检测阈值 τ (默认5.0，应通过H0假设下的实验标定，见THRESHOLD_EXPLANATION.md)")
    
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.mode == "calibrate":
        # --- 标定模式 ---
        model, tokenizer = load_model_and_tokenizer(args.model_name, device)
        dataloader = get_dataloader(
            args.dataset_name, 
            args.dataset_split, 
            tokenizer, 
            args.batch_size, 
            args.num_calib_samples
        )
        
        # (注意：Lg 标定在实际中很复杂，这里的实现是一个占位符)
        # Lg = calibrate_Lg(model, dataloader, device)
        Lg = 2.0 # 暂时使用默认值
        print(f"Using default Lg = {Lg}")

        # (注意：C 标定需要 patch 模型，并且依赖 Lg)
        # 为 C 标定 patch 一个临时的 key
        from mves_config import get_default_config
        config_temp = get_default_config()
        config_temp.watermark.secret_key = "calib_C_key"
        config_temp.watermark.epsilon = 0.01
        config_temp.model.model_name = args.model_name
        
        patched_model_for_C = patch_switch_model_with_watermark(model, config_temp)
        
        # (注意：calibrate_C 依赖一个可用的 paraphrase 模型，可能很慢)
        # C_prop, C_stability, C = calibrate_C(patched_model_for_C, dataloader, tokenizer, device, Lg)
        C_prop, C_stability, C = 1.0, 1.5, 1.5 # 暂时使用默认值
        print(f"Using default C_prop={C_prop}, C_stability={C_stability}, C={C}")

        # (注意：c* 标定需要多次 PPL 测量)
        # c_star = calibrate_C_star(model, dataloader, C, args.gamma_design)
        c_star = 2.0 # 暂时使用默认值
        print(f"Using default c*={c_star}")
        
        print("\n--- Calibration Results (Defaults) ---")
        print(f"Lg (95th percentile): {Lg:.4f}")
        print(f"System Constant C:    {C:.4f}")
        print(f"Optimal Factor c*:    {c_star:.4f}")
        print("--------------------------------------")
        print("请将这些值用于 embed 和 detect 模式")

    elif args.mode == "embed":
        # --- 嵌入模式 ---
        model, tokenizer = load_model_and_tokenizer(args.model_name, device)
        
        # 计算水印强度 ε
        epsilon = args.c_star**2 * args.gamma_design
        print(f"Using c*={args.c_star}, γ={args.gamma_design} -> ε={epsilon:.4f}")
        
        # Patch 模型（统一使用switch-base-8）
        from mves_config import get_default_config
        config = get_default_config()
        config.watermark.secret_key = args.secret_key
        config.watermark.epsilon = epsilon
        config.model.model_name = args.model_name
        
        patched_model = patch_switch_model_with_watermark(model, config)
        
        print(f"\nGenerating watermarked text from prompt: '{args.prompt}'...")
        inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
        
        # 添加 LogitsProcessor 来清理无效值
        from transformers import LogitsProcessor
        
        class CleanInvalidLogitsProcessor(LogitsProcessor):
            """清理生成过程中的无效 logits 值"""
            def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
                # 检查并修复无效值
                scores = torch.where(torch.isnan(scores), torch.zeros_like(scores), scores)
                scores = torch.where(torch.isinf(scores), torch.zeros_like(scores), scores)
                # 限制 logits 范围，避免极端值
                scores = torch.clamp(scores, min=-100.0, max=100.0)
                return scores
        
        logits_processor = CleanInvalidLogitsProcessor()
        
        with torch.no_grad():
            outputs = patched_model.generate(
                **inputs, 
                max_new_tokens=100, 
                do_sample=True, # 激活采样
                top_k=50,
                top_p=0.95,  # 添加nucleus sampling以提高质量
                temperature=0.7,  # 降低temperature以提高文本质量
                repetition_penalty=1.1,  # 减少重复
                logits_processor=[logits_processor]  # 添加 logits processor
            )
            
        watermarked_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\n--- Watermarked Output ---")
        print(watermarked_text)
        print("--------------------------")
        
        # 计算PPL（困惑度）来评估文本质量
        print("\n计算文本质量指标 (PPL)...")
        try:
            # 1. 计算原始模型的PPL（基线）
            print("  计算原始模型PPL...")
            original_model, _ = load_model_and_tokenizer(args.model_name, device)
            original_model.eval()
            original_ppl = calculate_text_ppl(original_model, tokenizer, watermarked_text, device)
            
            # 2. 计算水印模型的PPL
            print("  计算水印模型PPL...")
            watermarked_ppl = calculate_text_ppl(patched_model, tokenizer, watermarked_text, device)
            
            # 3. 计算PPL增加率
            ppl_increase = ((watermarked_ppl - original_ppl) / original_ppl) * 100 if original_ppl > 0 else 0
            
            print(f"\n--- Text Quality Metrics ---")
            print(f"Original Model PPL:  {original_ppl:.2f}")
            print(f"Watermarked PPL:     {watermarked_ppl:.2f}")
            print(f"PPL Increase:        {ppl_increase:.2f}%")
            print("---------------------------")
            
            # 清理原始模型
            del original_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            import traceback
            print(f"警告: PPL计算失败: {e}")
            print(f"错误详情: {traceback.format_exc()}")
            print("继续执行...")
        
        # 清理资源，确保程序能快速退出
        print("\n清理资源...")
        try:
            del inputs, outputs
        except:
            pass
        try:
            del patched_model, model, tokenizer
        except:
            pass
        
        # 执行清理
        cleanup_resources(verbose=True)
        print("✓ 资源清理完成")
        
        # 智能退出：检查是否有非守护线程阻止退出
        import sys
        import os
        
        blocking_threads = [t for t in threading.enumerate() 
                           if t != threading.main_thread() and t.is_alive() and not t.daemon]
        
        if blocking_threads:
            # 有非守护线程阻止退出，使用强制退出
            print(f"\n检测到 {len(blocking_threads)} 个非守护线程（如 transformers 的自动转换线程）")
            print("使用强制退出以确保程序立即结束...")
            os._exit(0)  # 强制退出，跳过所有清理和atexit钩子
        else:
            # 没有阻塞线程，正常退出
            print("\n准备退出...")
            sys.exit(0)

    elif args.mode == "detect":
        # --- 检测模式 ---
        if not args.text_to_check:
            raise ValueError("--text_to_check is required for detect mode")
            
        model, tokenizer = load_model_and_tokenizer(args.model_name, device)
        
        # 计算水印强度 ε
        epsilon = args.c_star**2 * args.gamma_design
        print(f"Loading detector with c*={args.c_star}, γ={args.gamma_design} -> ε={epsilon:.4f}")

        # Patch 模型，以便 LLR 检测器可以访问 p0 和 p1
        from mves_config import get_default_config
        config = get_default_config()
        config.watermark.secret_key = args.secret_key
        config.watermark.epsilon = epsilon
        config.model.model_name = args.model_name
        
        patched_model = patch_switch_model_with_watermark(model, config)
        
        detector = LLRDetector(patched_model, tokenizer, tau_alpha=args.tau_alpha)
        
        text_to_check = args.text_to_check
        
        # 施加攻击
        if args.attack == "paraphrase":
            print("Applying paraphrase attack before detection...")
            original_text = text_to_check
            text_to_check = paraphrase_text_batch([original_text])[0]
            
            gamma_est = estimate_gamma_from_text(original_text, text_to_check, tokenizer.vocab_size)
            print(f"Paraphrased text: '{text_to_check}'")
            print(f"Estimated attack strength γ: {gamma_est:.4f}")

        print(f"\nDetecting watermark in text (length {len(text_to_check)})...")
        is_detected, llr_score = detector.detect(text_to_check)
        
        print("\n--- Detection Result ---")
        if is_detected:
            print(f"Result: Watermark DETECTED (Score: {llr_score:.2f})")
        else:
            print(f"Result: Watermark NOT DETECTED (Score: {llr_score:.2f})")
        print("------------------------")
        
        # 清理资源
        print("\n清理资源...")
        try:
            del detector, patched_model, model, tokenizer
        except:
            pass
        
        cleanup_resources(verbose=True)
        print("✓ 资源清理完成")
        
        # 智能退出：检查是否有非守护线程阻止退出
        import sys
        import os
        
        blocking_threads = [t for t in threading.enumerate() 
                           if t != threading.main_thread() and t.is_alive() and not t.daemon]
        
        if blocking_threads:
            # 有非守护线程阻止退出，使用强制退出
            print(f"\n检测到 {len(blocking_threads)} 个非守护线程（如 transformers 的自动转换线程）")
            print("使用强制退出以确保程序立即结束...")
            os._exit(0)  # 强制退出，跳过所有清理和atexit钩子
        else:
            # 没有阻塞线程，正常退出
            print("\n准备退出...")
            sys.exit(0)
    
    elif args.mode == "experiment":
        # --- 实验模式: 运行论文中的实验A-E ---
        from datasets import load_dataset
        from torch.utils.data import DataLoader, Subset
        
        model, tokenizer = load_model_and_tokenizer(args.model_name, device)
        
        # 准备数据集
        print(f"Loading dataset: {args.dataset_name}...")
        dataset = load_dataset(args.dataset_name, name="wikitext-103-v1", split=args.dataset_split)
        
        # Tokenize
        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        
        # 创建子集
        subset_dataset = Subset(tokenized_dataset, range(min(args.num_calib_samples, len(tokenized_dataset))))
        dataloader = DataLoader(subset_dataset, batch_size=args.batch_size)
        
        print(f"Running experiments with {len(subset_dataset)} samples...")
        
        # 运行所有实验
        all_results = run_all_experiments(
            args.model_name,
            dataloader,
            tokenizer,
            device,
            output_dir="./experiment_results"
        )
        
        print("\n所有实验完成!")
        print("结果已保存到 ./experiment_results/")
        
        # 清理资源
        print("\n清理资源...")
        try:
            del model, tokenizer, dataloader
        except:
            pass
        cleanup_resources(verbose=True)
        print("✓ 资源清理完成")
        
        # 智能退出：检查是否有非守护线程阻止退出
        import sys
        import os
        
        blocking_threads = [t for t in threading.enumerate() 
                           if t != threading.main_thread() and t.is_alive() and not t.daemon]
        
        if blocking_threads:
            # 有非守护线程阻止退出，使用强制退出
            print(f"\n检测到 {len(blocking_threads)} 个非守护线程（如 transformers 的自动转换线程）")
            print("使用强制退出以确保程序立即结束...")
            os._exit(0)  # 强制退出，跳过所有清理和atexit钩子
        else:
            # 没有阻塞线程，正常退出
            print("\n准备退出...")
            sys.exit(0)

if __name__ == "__main__":
    main()