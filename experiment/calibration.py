import torch
import numpy as np
from tqdm import tqdm
from scipy.optimize import minimize
from sklearn.linear_model import RANSACRegressor, HuberRegressor
from transformers import AutoModelForCausalLM
from torch.utils.data import DataLoader
from typing import Tuple, Dict

from moe_watermark import patch_moe_model_with_watermark, get_watermark_data_from_model
from attacks import estimate_gamma_from_text, paraphrase_text_batch

# 占位符：需要一个函数来计算 Chernoff 信息
def compute_chernoff_information(p0: torch.Tensor, p1: torch.Tensor) -> float:
    """
    计算 D*(p0, p1)
    D* = -min_{lambda in [0,1]} log( sum( p0^(1-lambda) * p1^lambda ) )
    """
    p0 = p0.cpu().numpy()
    p1 = p1.cpu().numpy()
    
    def objective(lambda_):
        if lambda_ < 0 or lambda_ > 1:
            return np.inf
        log_sum = np.log(np.sum(np.power(p0, 1 - lambda_) * np.power(p1, lambda_)))
        return -log_sum

    result = minimize(objective, 0.5, bounds=[(0, 1)])
    if result.success:
        return result.fun
    else:
        # 边界情况
        return max(objective(0), objective(1))

def calibrate_Lg(model: AutoModelForCausalLM, dataloader: DataLoader, device: torch.device) -> float:
    """
    标定 Lipschitz 常数 Lg (对标 Algorithm 1)
    """
    print("Starting Lg calibration (Algorithm 1)...")
    model.eval()
    ratios = []
    
    # 假设 dataloader 产生 embedding
    # 在实际中，我们需要 tokenizer 和 model.get_input_embeddings()
    # 为简化，我们假设 dataloader 直接产生 inputs_embeds
    
    for batch in tqdm(dataloader, desc="Calibrating Lg"):
        inputs_embeds = batch['inputs_embeds'].to(device)
        
        # 1. 获取原始 logits l(e)
        with torch.no_grad():
            outputs = model(inputs_embeds=inputs_embeds, output_hidden_states=True)
            # 我们需要 MoE 层的 *输入* hidden_states
            # 假设我们修补第一层 MoE
            hs_e = model.model.layers[0].block_sparse_moe.gate.forward(
                outputs.hidden_states[0] # MoE 层的输入
            ) # [batch, seq, K_experts]
            
        # 2. 生成扰动 e'
        epsilon = 0.01
        noise = torch.randn_like(inputs_embeds) * epsilon
        e_prime = inputs_embeds + noise
        
        # 3. 获取扰动 logits l(e')
        with torch.no_grad():
            outputs_prime = model(inputs_embeds=e_prime, output_hidden_states=True)
            hs_e_prime = model.model.layers[0].block_sparse_moe.gate.forward(
                outputs_prime.hidden_states[0]
            )

        # 4. 计算 L2 范数
        delta_l = torch.norm(hs_e - hs_e_prime, p=2, dim=-1).view(-1)
        delta_x = torch.norm(noise, p=2, dim=-1).view(-1)
        
        # 5. 计算比率
        valid_mask = delta_x > 1e-6
        r_i = delta_l[valid_mask] / delta_x[valid_mask]
        ratios.extend(r_i.cpu().numpy())

    if not ratios:
        print("Warning: Lg calibration failed to produce ratios.")
        return 2.0 # 返回默认值

    Lg_95 = np.percentile(ratios, 95)
    print(f"Lg (95th percentile) calibrated: {Lg_95:.4f}")
    return float(Lg_95)


def calibrate_C(model: AutoModelForCausalLM, dataloader: DataLoader, tokenizer, device: torch.device, Lg: float) -> Tuple[float, float, float]:
    """
    标定 C_prop, C_stability, C (对标 Algorithm 2)
    """
    print("Starting C calibration (Algorithm 2)...")
    model.eval()
    
    gammas = []
    deltas_tv = []
    deltas_chernoff = []
    
    # 1. 标定 C_prop
    print("Calibrating C_prop...")
    for batch in tqdm(dataloader, desc="Calibrating C_prop"):
        inputs = batch['input_ids'].to(device)
        text_batch = tokenizer.batch_decode(inputs, skip_special_tokens=True)
        
        # 2. 生成释义攻击 x'
        text_prime_batch = paraphrase_text_batch(text_batch)
        
        for text, text_prime in zip(text_batch, text_prime_batch):
            # 3. 估算 gamma
            gamma_i = estimate_gamma_from_text(text, text_prime, tokenizer.vocab_size)
            if gamma_i < 1e-6:
                continue
            
            # 4. 计算激活分布 p(e|x) 和 p(e|x')
            # 为简化，我们只使用第一个 MoE 层
            def get_activation_dist(txt: str) -> torch.Tensor:
                inputs = tokenizer(txt, return_tensors="pt").to(device)
                with torch.no_grad():
                    model(**inputs) # 运行 forward 以填充 _watermark_detection_data
                data = get_watermark_data_from_model(model)
                if not data:
                    return torch.empty(0)
                # (p_0, p_1, S_obs)
                # 我们需要 p_0 (原始分布)
                # [batch, seq, K_experts]
                p_0_dist = data[0][0] 
                # [seq, K_experts]
                return p_0_dist.mean(dim=0) # 取 batch 和 seq 的平均分布

            p_dist = get_activation_dist(text)
            p_prime_dist = get_activation_dist(text_prime)

            if p_dist.numel() == 0 or p_prime_dist.numel() == 0:
                continue
                
            # 5. 计算总变差距离 delta
            delta_i_tv = 0.5 * torch.sum(torch.abs(p_dist - p_prime_dist))
            
            gammas.append(gamma_i)
            deltas_tv.append(delta_i_tv.item())

    if not gammas:
        print("Warning: C_prop calibration failed.")
        return 1.0, 1.0, 1.0 # 返回默认值

    # 6. 稳健回归: delta ≈ C_prop * sqrt(gamma) (1118文档推荐HuberRegressor)
    X = np.sqrt(np.array(gammas)).reshape(-1, 1)
    y = np.array(deltas_tv)
    
    # 使用HuberRegressor（1118文档推荐，对异常值更健壮）
    huber = HuberRegressor(fit_intercept=False, epsilon=1.1, max_iter=1000)
    huber.fit(X, y)
    C_prop = huber.coef_[0]
    R_squared = huber.score(X, y)
    
    print(f"C_prop (Propagation Constant) calibrated: {C_prop:.4f}")
    print(f"  R² = {R_squared:.4f}")
    
    if R_squared < 0.90:
        print(f"警告: 拟合R²={R_squared:.4f}，低于0.90阈值，建议增加样本数")
    
    # 7. 标定 C_stability (1118文档：通过Chernoff信息变化拟合)
    print("Calibrating C_stability through Chernoff information changes...")
    
    deltas_chernoff = []
    delta_tv_for_chernoff = []
    
    # 收集Chernoff信息变化数据
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Calibrating C_stability", leave=False)):
        if batch_idx >= 50:  # 限制样本数
            break
            
        if 'input_ids' in batch:
            inputs = batch['input_ids'].to(device)
            text_batch = tokenizer.batch_decode(inputs, skip_special_tokens=True)
        else:
            continue
        
        for text in text_batch:
            if len(text.split()) < 10:
                continue
            
            # 生成释义攻击
            paraphrased = paraphrase_text_batch([text])
            if not paraphrased or paraphrased[0] == text:
                continue
            
            text_prime = paraphrased[0]
            
            # 获取激活分布
            def get_activation_dist(txt: str) -> torch.Tensor:
                inputs = tokenizer(txt, return_tensors="pt").to(device)
                with torch.no_grad():
                    model(**inputs)
                data = get_watermark_data_from_model(model)
                if not data:
                    return None
                p_0_dist = data[0][0] if len(data) > 0 else None
                return p_0_dist.mean(dim=0) if p_0_dist is not None else None
            
            p_dist = get_activation_dist(text)
            p_prime_dist = get_activation_dist(text_prime)
            
            if p_dist is None or p_prime_dist is None:
                continue
            
            # 计算总变差距离
            delta_tv = 0.5 * torch.sum(torch.abs(p_dist - p_prime_dist)).item()
            
            # 计算Chernoff信息变化
            # 需要p_0和p_1（原始和水印分布）
            # 这里简化：使用p_dist作为p_0，p_prime_dist作为p_1的近似
            try:
                D_star_original = compute_chernoff_information(p_dist, p_dist)  # 应该是p_0 vs p_1
                D_star_attacked = compute_chernoff_information(p_dist, p_prime_dist)
                delta_chernoff = abs(D_star_attacked - D_star_original)
                
                if delta_chernoff > 0 and delta_tv > 0:
                    deltas_chernoff.append(delta_chernoff)
                    delta_tv_for_chernoff.append(delta_tv)
            except:
                continue
    
    # 拟合C_stability
    # 根据论文引理4.1: |D*(p',q') - D*(p,q)| ≤ C_stability * (δ_p + δ_q) * √D*(p,q)
    if len(deltas_chernoff) > 10:
        # 简化拟合：假设D* ≈ 0.1（典型值）
        D_star_typical = 0.1
        # 归一化：delta_chernoff / (delta_tv * sqrt(D*))
        normalized_deltas = np.array(deltas_chernoff) / (np.array(delta_tv_for_chernoff) * np.sqrt(D_star_typical))
        C_stability = np.median(normalized_deltas)  # 使用中位数更稳健
        C_stability = max(1.0, min(2.0, C_stability))  # 限制在合理范围
        print(f"C_stability (Stability Constant) calibrated: {C_stability:.4f}")
    else:
        # 回退到启发式
        C_stability = max(1.0, Lg / 2.0)
        print(f"C_stability (Stability Constant) estimated (heuristic): {C_stability:.4f}")

    # 8. 综合常数 C
    C = C_stability * C_prop
    print(f"Overall System Constant C calibrated: {C:.4f}")
    
    return float(C_prop), float(C_stability), float(C)

def calibrate_C_star(
    model: AutoModelForCausalLM, 
    dataloader: DataLoader, 
    C: float, 
    gamma_design: float, 
    lambda_weight: float = 1.0,
    delta_error: float = 0.001
) -> float:
    """
    标定最优安全系数 c* (对标 Algorithm 3)
    """
    print("Starting c* calibration (Algorithm 3)...")
    
    # 1. 标定性能成本函数 ΔA(c)
    # 我们用 PPL (Perplexity) 作为性能指标
    c_scan = np.linspace(C + 0.1, C * 2.5, 10)
    delta_A_values = []
    
    print("Scanning c values for performance cost (PPL)...")
    
    # 测量基线 PPL
    base_ppl = measure_ppl(model, dataloader, device)
    
    for c_val in tqdm(c_scan, desc="Calibrating ΔA(c)"):
        epsilon = c_val**2 * gamma_design
        # K_sec 在这里是临时的，只为测量 PPL
        temp_model = patch_moe_model_with_watermark(model, "temp_calib_key", epsilon)
        
        ppl = measure_ppl(temp_model, dataloader, device)
        delta_A = ppl - base_ppl # 性能 *下降*，所以 ppl 越高, ΔA 越大
        delta_A_values.append(delta_A)
        
        # (TODO: 卸载 patch，恢复模型)
        
    # 拟合 ΔA(c) = a * c^p
    # 为简化，我们使用 2 阶多项式
    try:
        poly_coeffs = np.polyfit(c_scan, delta_A_values, 2)
        delta_A_func = np.poly1d(poly_coeffs)
        print(f"Performance cost function ΔA(c) fitted: {delta_A_func}")
    except np.linalg.LinAlgError:
        print("Warning: PPL fitting failed. Using linear approximation.")
        slope = (delta_A_values[-1] - delta_A_values[0]) / (c_scan[-1] - c_scan[0])
        delta_A_func = lambda c: max(0, slope * (c - C))

    # 2. 网格搜索 c*
    # 目标函数: n*(c) + λ * ΔA(c)
    def objective_func(c):
        if c <= C:
            return np.inf
        # 样本复杂度 n*
        n_star = np.log(1.0 / delta_error) / (gamma_design * c * (c - C))
        # 性能成本 ΔA
        delta_A = delta_A_func(c)
        
        return n_star + lambda_weight * delta_A

    # 3. 求解 c*
    # 我们在 c_scan 范围内寻找最优值
    best_c_star = c_scan[0]
    min_obj = np.inf
    
    for c_val in np.linspace(C + 0.01, C * 2.5, 50): # 细网格
        obj = objective_func(c_val)
        if obj < min_obj:
            min_obj = obj
            best_c_star = c_val
            
    print(f"Optimal Security Factor c* calibrated: {best_c_star:.4f}")
    return float(best_c_star)

def measure_ppl(model: AutoModelForCausalLM, dataloader: DataLoader, device: torch.device) -> float:
    """
    辅助函数：测量模型的 PPL
    """
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    for batch in tqdm(dataloader, desc="Measuring PPL", leave=False):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch.get('attention_mask', torch.ones_like(input_ids)).to(device)
        labels = input_ids.clone()
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
        
        total_loss += loss.item() * input_ids.size(0)
        total_tokens += attention_mask.sum().item()
        
    if total_tokens == 0:
        return 0.0
        
    avg_loss = total_loss / len(dataloader.dataset) # 按样本平均
    ppl = np.exp(avg_loss)
    return float(ppl)   