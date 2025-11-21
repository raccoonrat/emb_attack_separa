"""
测试不同 repetition_penalty 值对生成质量的影响

快速验证不同参数值的效果，找到最佳平衡点
"""

import os
import sys
import platform
from pathlib import Path

# 重要：必须在导入 transformers 之前设置环境变量
def setup_cache():
    """设置缓存路径和 HuggingFace 镜像"""
    release = platform.release().lower()
    is_wsl = ("microsoft" in release or "wsl" in release)
    if not is_wsl and os.path.exists("/proc/version"):
        try:
            with open("/proc/version", "r") as f:
                is_wsl = "microsoft" in f.read().lower()
        except:
            pass
    is_linux = platform.system() == "Linux"
    
    if os.environ.get("USE_WSL_CONFIG") == "1" or (is_wsl or is_linux):
        try:
            import cache_config_wsl
            return
        except ImportError:
            CACHE_BASE = Path.home() / ".cache" / "emb_attack_separa"
    else:
        try:
            import cache_config
            return
        except ImportError:
            CACHE_BASE = Path("D:/Dev/cache")
    
    CACHE_BASE.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(CACHE_BASE / "huggingface"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_BASE / "huggingface" / "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(CACHE_BASE / "huggingface" / "datasets"))
    os.environ.setdefault("TORCH_HOME", str(CACHE_BASE / "torch"))
    
    # 设置 HuggingFace 镜像（必须在导入 transformers 之前）
    # 支持多种镜像配置方式
    if "HF_ENDPOINT" not in os.environ:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print(f"[INFO] 设置 HuggingFace 镜像: {os.environ['HF_ENDPOINT']}")
    else:
        print(f"[INFO] 使用已设置的 HuggingFace 端点: {os.environ['HF_ENDPOINT']}")
    
    # 对于 hf-mirror.com，还需要设置镜像 URL 格式
    # HuggingFace Transformers 会检查这些环境变量
    if "hf-mirror.com" in os.environ.get("HF_ENDPOINT", ""):
        # 确保使用镜像下载
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # 启用快速下载
        print(f"[INFO] 已启用 HuggingFace 镜像下载")

# 必须在导入 transformers 之前调用
setup_cache()

# 现在可以安全导入 transformers
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from okr_patch import inject_okr

# 显示当前使用的 HuggingFace 端点
hf_endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
print(f"[INFO] HuggingFace 端点: {hf_endpoint}")

print("=" * 60)
print("Repetition Penalty 参数测试")
print("=" * 60)

# 测试参数
MODEL_NAME = "google/switch-base-8"
TEST_PROMPT = "The quick brown fox jumps over the lazy dog."
REPETITION_PENALTIES = [1.0, 1.1, 1.2, 1.3, 1.5, 2.0]  # 测试不同的值
MAX_NEW_TOKENS = 50

print(f"\n模型: {MODEL_NAME}")
print(f"测试提示词: {TEST_PROMPT}")
print(f"最大新token数: {MAX_NEW_TOKENS}")
print(f"\n测试 repetition_penalty 值: {REPETITION_PENALTIES}")
print("=" * 60)

# 1. 加载模型和分词器
print("\n1. 加载模型和分词器...")
print(f"   使用端点: {hf_endpoint}")

# HuggingFace Transformers 会自动使用 HF_ENDPOINT 环境变量
# 如果设置了镜像，会自动从镜像下载
# 注意：某些版本的 transformers 可能需要额外的配置
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
except Exception as e:
    print(f"   加载失败: {e}")
    print("   提示：如果网络问题，请检查 HF_ENDPOINT 环境变量是否正确设置")
    print(f"   当前 HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', '未设置')}")
    raise

# 2. 注入 OKR 水印
print("\n2. 注入 OKR 水印...")
model = inject_okr(
    model,
    epsilon=1.5,
    secret_key="TEST_KEY",
    watermark_alpha=0.1  # 使用默认值
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

print(f"设备: {device}")

# 3. 准备输入
inputs = tokenizer(TEST_PROMPT, return_tensors="pt", padding=True, truncation=True).to(device)

# 4. 测试不同的 repetition_penalty 值
print("\n3. 测试不同的 repetition_penalty 值...")
print("=" * 60)

results = []

for rep_penalty in REPETITION_PENALTIES:
    print(f"\n[Repetition Penalty = {rep_penalty}]")
    print("-" * 60)
    
    # 清空路由数据
    if hasattr(model, 'clear_okr_stats'):
        model.clear_okr_stats()
    
    # 添加任务前缀（T5 模型需要）
    input_text = f"generate: {TEST_PROMPT}"
    inputs_with_prefix = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs_with_prefix,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=1,
            do_sample=False,
            use_cache=False,
            decoder_start_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            repetition_penalty=rep_penalty
        )
    
    # 解码生成的文本
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 移除任务前缀（如果存在）
    if generated_text.startswith("generate: "):
        generated_text = generated_text[len("generate: "):]
    
    # 计算重复度（简单的启发式：连续重复的token数）
    tokens = generated_text.split()
    if len(tokens) > 0:
        # 计算重复的token比例
        unique_tokens = len(set(tokens))
        total_tokens = len(tokens)
        diversity = unique_tokens / total_tokens if total_tokens > 0 else 0
        
        # 检查是否有明显的重复模式
        has_repetition = False
        for i in range(len(tokens) - 2):
            if tokens[i] == tokens[i+1] == tokens[i+2]:
                has_repetition = True
                break
    else:
        diversity = 0
        has_repetition = False
    
    print(f"生成文本: {generated_text[:200]}...")
    print(f"文本长度: {len(generated_text)} 字符")
    print(f"Token 多样性: {diversity:.2%} ({unique_tokens}/{total_tokens})")
    print(f"明显重复: {'是' if has_repetition else '否'}")
    
    results.append({
        'repetition_penalty': rep_penalty,
        'text': generated_text,
        'length': len(generated_text),
        'diversity': diversity,
        'has_repetition': has_repetition
    })

# 5. 总结
print("\n" + "=" * 60)
print("测试总结")
print("=" * 60)

print("\n推荐值分析:")
print("-" * 60)

# 找到最佳平衡点（高多样性 + 无重复）
best_candidates = []
for r in results:
    score = r['diversity'] * (0.5 if not r['has_repetition'] else 0.1)  # 有重复的惩罚
    best_candidates.append((r['repetition_penalty'], score, r))

best_candidates.sort(key=lambda x: x[1], reverse=True)

print("\n按综合得分排序:")
for i, (penalty, score, result) in enumerate(best_candidates[:3]):
    print(f"{i+1}. Repetition Penalty = {penalty:.1f}")
    print(f"   多样性: {result['diversity']:.2%}")
    print(f"   重复: {'有' if result['has_repetition'] else '无'}")
    print(f"   得分: {score:.3f}")
    print(f"   预览: {result['text'][:100]}...")
    print()

print("=" * 60)
print("测试完成！")
print("=" * 60)

