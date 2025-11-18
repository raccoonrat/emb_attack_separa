# MoE水印实现核心方法体系

## 一、架构分层设计

### 1.1 信号-攻击解耦的本质
```
Token-Logit范式（脆弱）：
信号空间 S_A = R^|V|  ⊂ 攻击空间 A
└─ 水印在logit空间，对手通过改变输入词元直接破坏 → 线性衰减 O(γ)

MoE范式（强健）：
信号空间 S_B = {0,1}^K  ∩  攻击空间 A_direct = ∅
└─ 水印在专家激活模式，对手改变输入无法直接修改gating权重 → 次线性衰减 O(√γ)
```

**关键约束**：
- 对手能力：输入级释义攻击（ED(x,x') ≤ L）
- 对手无法访问模型参数或gating网络权重
- 限制条件：开源MoE模型场景

---

## 二、核心三大定理的实现路线

### 2.1 定理2.6：Token-Logit线性衰减（对比基准）

**数学模型**：
```
在KGW水印下：
- 原始z-score: Z_KGW = √N · δ · √(γ_G(1-γ_G))
- 衰减规律：∆Z(γ) = C_linear · γ
- 其中 C_linear ≈ 125（实验标定值）
```

**实现步骤**：
```
1. 嵌入阶段：
   ├─ 构建绿名单 G（词汇表的5%，|G|=6400）
   ├─ 设置logit偏置强度 δ（通常0.5-2.0）
   └─ 在logit空间修改：logit_green += δ

2. 检测阶段：
   ├─ 统计生成文本中落在G中的词元数k
   ├─ 计算z-score：Z = (k - N·γ_G) / √(N·γ_G(1-γ_G))
   └─ 判决：Z > τ_detect(≈4) → 有水印

3. 攻击下的衰减追踪：
   ├─ 释义攻击强度 γ = D_KL(D(X')∥D(X))
   ├─ 测量被替换的绿名单词元数 ∆k ∝ γ
   └─ 验证：∆Z_measured ≈ C_linear · γ_measured
```

**实验验证（表B1预期结果）**：
```
δ=1.0, γ从0.01到0.05的变化：
Z值序列：[6.0, 3.8, 1.5, -0.3]
拟合直线：Z(γ) = 6.0 - 125·γ  (R²=0.98)
```

---

### 2.2 定理4.7：MoE次线性衰减（核心创新）

**数学基础链条**：
```
第一步 - Pinsker不等式（信息论基础）：
∥D(X') - D(X)∥_TV ≤ √(γ/2)

第二步 - Lipschitz传播（输入→激活）：
∥p'_i - p_i∥_TV ≤ L_g · ∥D(X') - D(X)∥_TV

第三步 - 综合结果：
∥p'_i - p_i∥_TV ≤ L_g · √(γ/2) =: C_prop · √γ

第四步 - Chernoff信息衰减（检测能力）：
D*_adv ≥ D*(p_0, p_1) - C_stability · C_prop · √γ · √D*(p_0,p_1)
```

**关键参数的理论来源**：

| 参数 | 定义 | 理论界 | 实测值 | 用途 |
|------|------|-------|-------|------|
| L_g | gating网络Lipschitz常数 | L_local · √d_model ≤ 128 | 1.5-2.8 | 分布级扰动放大因子 |
| C_prop | 激活分布扰动系数 | L_g · √(1/2) | 1.0-2.0 | 次线性基系数 |
| C_stability | Chernoff信息稳定性常数 | ≈1.0 | 1.2-1.5 | 检测能力衰减放大 |
| C | 综合常数 | C_stability × C_prop | 1.5-2.0 | 总体鲁棒性指标 |

---

## 三、工程实现的三大支柱

### 3.1 支柱1：水印嵌入机制

**位置**：gating网络输出的logit修改

```python
# 伪代码结构
def embed_moe_watermark(x, c, model, watermark_config):
    """
    输入：
      x: 输入文本token序列
      c: 安全系数（决定水印强度）
      model: MoE模型
      watermark_config: 水印参数集合
    """
    
    # 阶段1：前向传播到gating层
    hidden_states = model.encoder(x)
    
    # 阶段2：修改gating logit
    # 原始gating logit
    gate_logits_original = model.gating_network.linear(hidden_states)
    
    # 构建水印信号
    # S_watermark ∈ {0,1}^K，其中K为专家数
    S_watermark = select_expert_pattern(watermark_config)  # e.g., [1,0,1,0,...]
    
    # 计算logit修改量 ∆ℓ
    # ∆ℓ满足约束：∥∆ℓ∥²_2 ≈ 2·ε（KL散度ε与∆ℓ的二次关系）
    epsilon = c * sqrt(gamma_attack)  # gamma_attack需外部估计
    delta_logit = compute_logit_perturbation(
        S_watermark, 
        epsilon,
        strength_control=watermark_config['strength']
    )
    
    # 阶段3：修改后的logit
    gate_logits_watermarked = gate_logits_original + delta_logit
    
    # 阶段4：执行top-k激活
    activated_experts = topk_activation(
        gate_logits_watermarked,
        k=model.num_active_experts
    )
    
    return activated_experts, gate_logits_watermarked
```

**关键约束**：
- ∆ℓ的大小由 c 和 γ 共同决定：∥∆ℓ∥²_2 ≈ 2·c²·γ
- 为避免梯度爆炸：应用梯度裁剪，确保L_g ≤ 3
- 离散性处理：Top-k操作导致的排名交叉问题由f_k(S)系数处理

---

### 3.2 支柱2：检测统计量的计算

**从似然比检验到Chernoff信息**：

```python
def compute_detection_statistic(x_prime, model, num_samples=1000):
    """
    计算对抗后的检测能力指标 D*_adv
    基于Neyman-Pearson最优性和Chernoff-Stein定理
    """
    
    # 步骤1：收集激活分布样本
    activation_patterns = []
    for i in range(num_samples):
        # 前向传播
        hidden = model.encoder(x_prime[i])
        gate_logits = model.gating_network(hidden)
        
        # 提取激活模式 S = {0,1}^K
        activated = topk_activation(gate_logits, k=2)  # Top-2激活
        activation_patterns.append(activated)
    
    # 步骤2：估计两个分布
    # p_0: 无水印模型的激活分布
    # p_1: 有水印模型的激活分布
    p_0_empirical = estimate_distribution(activation_patterns, watermarked=False)
    p_1_empirical = estimate_distribution(activation_patterns, watermarked=True)
    
    # 步骤3：计算Chernoff信息 D*(p_0, p_1)
    # 定义：D* = max_{0≤λ≤1} [-log Σ_e p_0(e)^(1-λ) · p_1(e)^λ]
    D_chernoff = compute_chernoff_information(p_0_empirical, p_1_empirical)
    
    # 步骤4：估计对抗衰减
    # 理论下界：D*_adv ≥ D* - C·√γ·√D*
    gamma_attack = estimate_attack_strength(x_prime)  # KL散度估计
    C_total = 1.5  # 实验标定值
    
    D_adv_lower_bound = D_chernoff - C_total * sqrt(gamma_attack) * sqrt(D_chernoff)
    
    return {
        'D_chernoff': D_chernoff,
        'gamma_attack': gamma_attack,
        'D_adv_bound': max(0, D_adv_lower_bound),
        'sample_complexity': log(1/delta) / max(D_adv_lower_bound, 1e-6)
    }
```

**统计量对应关系**：
```
无水印 H_0: 激活分布 p_0(e)
有水印 H_1: 激活分布 p_1(e)，通过gating修改δℓ实现

LLR检验：Λ_n = Σ log[p_1(S_i)/p_0(S_i)]

判决规则：
  Λ_n > τ_α  ⇒  判定为有水印
  
误差率指数衰减（Chernoff-Stein定理）：
  log P_e(n) = -n·D*(p_0,p_1) + o(n)
  
这意味着样本复杂度 n* = log(1/δ)/D*
```

---

### 3.3 支柱3：参数标定三部曲

#### 标定第一步：Lipschitz常数L_g

```python
def calibrate_lipschitz_constant(validation_data, model):
    """
    目标：通过实验测量gating网络对输入扰动的敏感度
    公式：L_g = max_i [∥ℓ(x_i) - ℓ(x'_i)∥_2 / ∥x_i - x'_i∥_2]
    """
    
    ratios = []
    
    for x in validation_data:
        # 方法A：Embedding空间扰动（推荐）
        e = model.embed(x)  # [L, d_model]
        epsilon_values = [0.01, 0.02, 0.05, 0.1]
        
        for eps in epsilon_values:
            e_prime = e + eps * np.random.normal(0, 1, e.shape)
            
            # 计算logit差异
            gate_logit = model.gating_network(e)
            gate_logit_prime = model.gating_network(e_prime)
            delta_logit = norm(gate_logit - gate_logit_prime)
            
            # 计算embedding差异
            delta_e = norm(e - e_prime)
            
            # 计算Lipschitz比率
            ratio = delta_logit / delta_e
            ratios.append(ratio)
    
    ratios = np.array(ratios)
    
    # 统计指标
    L_max = np.max(ratios)
    L_95 = np.percentile(ratios, 95)  # 推荐使用
    L_mean = np.mean(ratios)
    
    # 验证阈值
    if L_max > 10:
        print("WARNING: 梯度爆炸迹象，需要梯度裁剪")
        apply_gradient_clipping(model, threshold=3.0)
    
    return {
        'L_max': L_max,
        'L_95': L_95,
        'L_mean': L_mean,
        'recommended': L_95  # 用于理论计算
    }
```

**输出示例（表D1）**：
```
LLaMA-7B-MoE:   L_95=2.3  (理论假设2.0 ✓)
Mixtral-8x7B:   L_95=2.8  (略高，需注意排名交叉)
DeepSeek-16B:   L_95=1.9  (接近理论值)
```

---

#### 标定第二步：综合常数C

```python
def calibrate_combined_constant(validation_data, model, paraphrase_fn):
    """
    目标：验证 ∥p'_i - p_i∥_TV ≈ C_prop · √γ_i 的系数
    通过释义攻击样本对进行非线性回归
    """
    
    gamma_samples = []
    delta_tv_samples = []
    
    for x in validation_data:
        # 生成释义版本
        x_prime = paraphrase_fn(x)
        
        # 计算攻击强度 γ = D_KL(D(X')||D(X))
        gamma_kl = compute_kl_divergence(x, x_prime, model)
        
        # 计算激活分布的总变差距离
        p_original = get_activation_distribution(model, x)
        p_paraphrased = get_activation_distribution(model, x_prime)
        delta_tv = total_variation_distance(p_original, p_paraphrased)
        
        gamma_samples.append(gamma_kl)
        delta_tv_samples.append(delta_tv)
    
    # 非线性回归：δ_i ≈ C_prop · √γ_i
    gamma_samples = np.array(gamma_samples)
    delta_tv_samples = np.array(delta_tv_samples)
    sqrt_gamma = np.sqrt(gamma_samples)
    
    # 健壮回归（使用Huber loss避免异常值影响）
    from sklearn.linear_model import HuberRegressor
    regressor = HuberRegressor(fit_intercept=False, epsilon=1.1, max_iter=1000)
    regressor.fit(sqrt_gamma.reshape(-1, 1), delta_tv_samples)
    
    C_prop = regressor.coef_[0]
    R_squared = regressor.score(sqrt_gamma.reshape(-1, 1), delta_tv_samples)
    
    # 验证拟合质量
    if R_squared < 0.90:
        print(f"WARNING: 拟合R²={R_squared}，低于0.90阈值")
    
    return {
        'C_prop': C_prop,
        'R_squared': R_squared,
        'confidence_interval': compute_ci(regressor, delta_tv_samples)
    }
```

**预期关系**：
```
在释义攻击下：
γ_KL = 0.01 nats  →  δ_TV ≈ 0.5·√0.01 = 0.05  →  C_prop ≈ 5
γ_KL = 0.03 nats  →  δ_TV ≈ 0.5·√0.03 = 0.09  →  确认C_prop ≈ 5
```

---

#### 标定第三步：最优安全系数c*

```python
def calibrate_optimal_safety_coefficient(model, validation_data, lambda_weight=1.0):
    """
    目标：最小化 f(c) = n*(γ,c) + λ·∆A(c)
    
    其中：
      n*(γ,c) = log(1/δ) / [γ·c·(c-C)]  -- 样本复杂度
      ∆A(c) = a·c^p + b·c^q            -- 性能成本
    """
    
    # 阶段1：标定性能成本函数 ∆A(c)
    c_values = np.linspace(C - 0.2, 2.5*C + 0.2, 40)
    performance_costs = []
    
    for c in c_values:
        # 嵌入水印
        embed_watermark_with_strength(model, c)
        
        # 测量下游任务性能下降（通常用困惑度PPL）
        ppl_drop = measure_perplexity_drop(model, validation_data)
        performance_costs.append(ppl_drop)
    
    # 多项式拟合：∆A(c) = a·c^p + b·c^q
    # 通常 p ∈ [1,2], q ∈ [2,3]
    from scipy.optimize import curve_fit
    
    def cost_model(c, a, b, p, q):
        return a * (c ** p) + b * (c ** q)
    
    params, _ = curve_fit(
        cost_model, 
        c_values, 
        performance_costs,
        p0=[0.1, 0.05, 1.5, 2.8],
        maxfev=10000
    )
    a, b, p, q = params
    
    # 阶段2：网格搜索最优c*
    def objective(c):
        # 样本复杂度
        gamma = 0.03  # 假设攻击强度
        delta = 0.01  # 目标检测精度
        n_star = np.log(1/delta) / (gamma * c * (c - C))
        
        # 性能成本
        delta_a = a * (c ** p) + b * (c ** q)
        
        return n_star + lambda_weight * delta_a
    
    # 粗网格搜索
    c_coarse = np.linspace(C, 2.5*C, 8)
    objectives_coarse = [objective(c) for c in c_coarse]
    c_best_coarse = c_coarse[np.argmin(objectives_coarse)]
    
    # 细网格搜索
    c_fine = np.linspace(c_best_coarse - 0.4, c_best_coarse + 0.4, 20)
    objectives_fine = [objective(c) for c in c_fine]
    c_optimal = c_fine[np.argmin(objectives_fine)]
    
    return {
        'c_optimal': c_optimal,
        'objective_value': min(objectives_fine),
        'cost_model_params': (a, b, p, q),
        'sensitivity': {
            'lambda_minus_50%': optimize_for_lambda(lambda_weight * 0.5),
            'lambda_plus_50%': optimize_for_lambda(lambda_weight * 1.5)
        }
    }
```

**应用场景的λ选择**：
```
严格保密（银行、军事）    → λ = 500  → c* ≈ 2.5C  (强鲁棒性)
内容验证（新闻、社交）    → λ = 5    → c* ≈ 1.5C  (平衡)
学术署名（灵活要求）      → λ = 0.5  → c* ≈ 1.2C  (低成本)
```

---

## 四、攻击强度γ的估计机制

### 4.1 编辑距离→KL散度映射

```python
def estimate_attack_strength_upper_bound(x_original, x_paraphrased, vocab_size=128000):
    """
    基于引理7.1：D_KL(D(X')||D(X)) ≤ (L/N)·log(|V|/|V_freq|)
    """
    
    # 计算编辑距离
    edit_distance = compute_edit_distance(x_original, x_paraphrased)
    N = len(x_original.split())  # 文本长度（词数）
    
    # 情况1：通用估计
    gamma_upper_coarse = (edit_distance / N) * np.log(vocab_size)
    
    # 情况2：细粒度估计（考虑词频）
    # 实际paraphrase模型倾向于替换低频词
    V_freq = 1000  # 常用词数
    gamma_upper_fine = (edit_distance / N) * np.log(vocab_size / V_freq)
    
    # 情况3：实测验证（通过KL散度直接计算）
    # 对token级别计算分布变化
    tokens_original = tokenize(x_original)
    tokens_paraphrased = tokenize(x_paraphrased)
    
    # 构造token级分布
    dist_original = empirical_distribution(tokens_original)
    dist_paraphrased = empirical_distribution(tokens_paraphrased)
    
    gamma_measured = kl_divergence(dist_paraphrased, dist_original)
    
    return {
        'gamma_upper_coarse': gamma_upper_coarse,
        'gamma_upper_fine': gamma_upper_fine,
        'gamma_measured': gamma_measured,
        'recommended': gamma_measured  # 最准确
    }
```

**预期结果对标（表A1）**：
```
攻击类型        编辑距离  γ_upper   γ_measured  误差
GPT-3.5         2.3      0.022    0.018      18%
T5              4.1      0.041    0.035      15%
对抗样本生成    6.5      0.065    0.052      20%

结论：上界与实测的紧密度 ≈ 80-85%，可用于保守估计
```

---

## 五、完整实现的执行流程

### 5.1 部署阶段

```python
class MOEWatermarkSystem:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        # 标定参数
        self.L_g = config['L_g']  # 通过calibrate_lipschitz_constant得到
        self.C_prop = config['C_prop']
        self.C_stability = config['C_stability']
        self.C = self.C_stability * self.C_prop
    
    def embed(self, x, c):
        """
        嵌入水印：修改gating网络logit
        参数c由应用场景决定（见表5.5）
        """
        # 预测攻击强度（保守估计）
        gamma_attack = self.estimate_gamma(x)
        
        # 计算水印强度
        epsilon = c * np.sqrt(gamma_attack)
        
        # 修改gating logit
        return self._modify_gating_logit(x, epsilon)
    
    def detect(self, x_prime, num_samples=1000):
        """
        检测水印：计算Chernoff信息
        """
        # 收集激活模式样本
        activations = [self._get_activation(x_prime) for _ in range(num_samples)]
        
        # 估计分布
        p_0 = estimate_distribution(activations, watermarked=False)
        p_1 = estimate_distribution(activations, watermarked=True)
        
        # 计算Chernoff信息
        D_chernoff = self._compute_chernoff(p_0, p_1)
        
        # 估计衰减
        gamma = self.estimate_gamma(x_prime)
        D_adv = D_chernoff - self.C * np.sqrt(gamma) * np.sqrt(D_chernoff)
        
        return {
            'D_chernoff': D_chernoff,
            'D_adv_lower_bound': max(0, D_adv),
            'sample_complexity': np.log(1/0.01) / max(D_adv, 1e-6),
            'is_watermarked': D_adv > 0
        }
```

---

### 5.2 实验验证清单

| 实验 | 目标 | 核心指标 | 预期结果 | 对应定理 |
|------|------|---------|---------|---------|
| 实验A | 验证γ上界 | γ_measured/γ_upper | 0.80-0.85 | 引理7.1 |
| 实验B | Token-Logit线性衰减 | R²(线性拟合) | >0.95 | 定理2.6 |
| 实验C | **MoE次线性衰减对比** | **衰减率比值** | **1/√γ增长** | **定理4.7** |
| 实验D | L_g标定准确性 | L_95与理论的吻合 | ±20% | 引理4.6 |
| 实验E | c*最优性 | 目标函数最小值 | λ=1时c*≈1.33C | 定理5.5 |

---

## 六、关键实现避坑清单

### 6.1 数学陷阱

```
1. Pinsker不等式的使用
   ✗ 错误：直接假设∥D(X')-D(X)∥_TV = √(γ/2)
   ✓ 正确：这是上界，实际值需通过Bhattacharyya系数更精细估计
   
2. Chernoff信息的连续性
   ✗ 错误：Top-k激活导致的离散性可忽略不计
   ✓ 正确：排名交叉时f_k(S)系数可能>1，需用引理4.4'处理
   
3. Lipschitz常数的理论值
   ✗ 错误：L_g ≈ L_local · √d_model ≤ 128总是成立
   ✓ 正确：只是上界，实测通常L_95 ≈ 2，但需实验标定
   
4. 释义模型的假设
   ✗ 错误：假设词元替换均匀分布
   ✓ 正确：实际paraphrase非均匀，需引入修正项g(θ)·γ^(3/2)
```

### 6.2 工程陷阱

```
1. 梯度爆炸
   症状：L_max > 10（对应排名交叉频率5-10%）
   解决：应用梯度裁剪 clip_norm=3.0 或spectral norm正则化
   
2. 检测样本不足
   症状：Chernoff信息估计方差大（>20%）
   解决：增加样本数到 n* = log(1/δ)/(γ·c·(c-C))
   
3. 攻击强度估计偏低
   症状：实测D_adv << 理论下界
   解决：使用保守上界γ_upper而非γ_measured
   
4. Top-k激活的排名边界
   症状：激活模式随微小输入扰动剧变
   解决：在logit差异<σ的情况下采用f_k(S)系数调整
```

---

## 七、定量验证的关键数字

### 理论预测vs实验对标

```
基准预测1 - KGW临界失效点：
  理论：γ_crit = 4.0/(125×6) ≈ 0.0053 nats (假设均匀替换)
  实测：γ_crit ≈ 0.027 nats (实际非均匀替换)
  修正：引入g(θ)·γ^(3/2)后理论→实验

基准预测2 - MoE衰减下界：
  输入：D*(p_0,p_1) = 0.1 nats, C = 1.5, γ = 0.03 nats
  理论下界：D*_adv ≥ 0.1 - 1.5·√0.03·√0.1 = 0.018 nats
  实测值：D*_adv ≈ 0.075 nats
  结论：理论保守性 24% (理论下界/实测 = 0.018/0.075)
        实际性能优于保证

决定性对比：
  γ=0.03时 范式A完全失效(Z→0)，范式B保持77%初始强度
  γ=0.05时 范式A无法检测，范式B保持65%初始强度
  衰减速率优势：范式B/范式A = O(√γ)/O(γ) = 1/√γ
```

---

## 八、实际代码框架与集成点

### 8.1 模型改造的最小化集成

```python
# 以LLaMA-MoE为例的集成改造
class MOE_Watermark_Wrapper:
    """
    轻量级包装器，无需修改原始模型代码
    """
    
    def __init__(self, base_model, watermark_config):
        self.base_model = base_model
        self.config = watermark_config
        
        # 提取gating网络（关键挂载点）
        self.gating_networks = [
            layer.mlp.gate for layer in base_model.layers 
            if hasattr(layer.mlp, 'gate')
        ]
        
        self.watermark_pattern = watermark_config['expert_pattern']  # e.g., [1,0,1,0,...]
        self.strength = watermark_config['c']
        self.registered = False
    
    def register_hooks(self):
        """
        在gating网络输出处注册前向钩子（hook）
        这是最少侵入式的集成方法
        """
        def modify_gating_logit(module, input, output):
            # output shape: [batch, num_experts]
            batch_size = output.shape[0]
            
            # 计算logit修改量
            delta_logit = self._compute_delta_logit(output)
            
            # 返回修改后的logit
            return output + delta_logit
        
        for i, gating_net in enumerate(self.gating_networks):
            gating_net.register_forward_hook(modify_gating_logit)
        
        self.registered = True
    
    def remove_hooks(self):
        """移除水印钩子"""
        for gating_net in self.gating_networks:
            gating_net._forward_hooks.clear()
        self.registered = False
    
    def _compute_delta_logit(self, gate_logits):
        """
        计算logit修改量∆ℓ
        约束条件：∥∆ℓ∥²_2 ≈ 2·ε²，其中ε = c·√γ
        """
        epsilon = self.strength * np.sqrt(self.gamma_attack)
        
        # 方法1：针对特定专家激活（推荐）
        delta = torch.zeros_like(gate_logits)
        for expert_idx in range(len(self.watermark_pattern)):
            if self.watermark_pattern[expert_idx] == 1:
                # 增大该专家的选择概率
                delta[:, expert_idx] += epsilon
            else:
                # 减小该专家的选择概率
                delta[:, expert_idx] -= epsilon * 0.3
        
        # 方法2：梯度裁剪保护
        delta = torch.clamp(delta, -3.0, 3.0)
        
        return delta
    
    def forward(self, x):
        """透传前向，水印通过hook自动应用"""
        return self.base_model(x)

# 使用示例
model = load_model("llama-7b-moe")
watermark_system = MOE_Watermark_Wrapper(
    model, 
    {
        'expert_pattern': [1, 0, 1, 0, 1, 0, 1, 0],  # 交替激活
        'c': 1.5,  # 安全系数
        'gamma_attack': 0.03  # 预估的攻击强度
    }
)
watermark_system.register_hooks()

# 生成带水印的文本
with torch.no_grad():
    output = model.generate(prompt, max_length=100)
```

---

### 8.2 检测模块的实现

```python
class MOE_Watermark_Detector:
    """
    独立的检测模块，可用于离线分析
    """
    
    def __init__(self, model, reference_model=None):
        self.model = model
        self.reference_model = reference_model or model
        self.cache_activations = {}
    
    def collect_activation_patterns(self, text, num_passes=100):
        """
        收集多次前向传播的激活模式
        用于估计激活分布 p_1(e)
        """
        patterns = []
        
        for _ in range(num_passes):
            hidden_states = self.model.encode(text)
            
            for layer_idx, layer in enumerate(self.model.layers):
                if hasattr(layer, 'mlp') and hasattr(layer.mlp, 'gate'):
                    gate_logits = layer.mlp.gate(hidden_states)
                    
                    # 提取Top-k激活模式 S ∈ {0,1}^K
                    activated_experts = torch.topk(
                        gate_logits, 
                        k=self.model.num_active_experts
                    )[1]  # 返回index
                    
                    pattern = self._logits_to_pattern(
                        gate_logits, 
                        activated_experts
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def estimate_activation_distribution(self, patterns):
        """
        从激活模式样本估计分布 p(S)
        S ∈ {0,1}^K，共 C(K,k) 种可能（k为激活数）
        """
        from collections import Counter
        
        # 转换为可哈希形式
        pattern_tuples = [tuple(p) for p in patterns]
        pattern_counts = Counter(pattern_tuples)
        
        # 经验分布
        distribution = {}
        for pattern, count in pattern_counts.items():
            distribution[pattern] = count / len(patterns)
        
        return distribution
    
    def compute_chernoff_information(self, p0_dict, p1_dict):
        """
        计算Chernoff信息 D*(p_0, p_1)
        定义：D* = max_{0≤λ≤1} [-log Σ_s p_0(s)^(1-λ)·p_1(s)^λ]
        """
        def renyi_divergence(lambda_param):
            # 对所有可能的激活模式 s
            all_patterns = set(p0_dict.keys()) | set(p1_dict.keys())
            
            sum_term = 0
            for pattern in all_patterns:
                p0_val = p0_dict.get(pattern, 1e-10)
                p1_val = p1_dict.get(pattern, 1e-10)
                
                sum_term += (p0_val ** (1 - lambda_param)) * (p1_val ** lambda_param)
            
            return -np.log(sum_term + 1e-10)
        
        # 网格搜索最优λ
        lambdas = np.linspace(0, 1, 101)
        renyi_values = [renyi_divergence(lam) for lam in lambdas]
        
        D_chernoff = np.max(renyi_values)
        optimal_lambda = lambdas[np.argmax(renyi_values)]
        
        return D_chernoff, optimal_lambda
    
    def detect_watermark(self, text, gamma_attack=0.03, threshold=0.01):
        """
        主检测函数
        
        返回：
          - D_chernoff: 无攻击下的Chernoff信息
          - D_adv_bound: 被攻击后的下界
          - verdict: 是否检测到水印
        """
        # 收集激活模式
        patterns_test = self.collect_activation_patterns(text)
        p1_empirical = self.estimate_activation_distribution(patterns_test)
        
        # 参考分布（无水印）
        patterns_ref = self.collect_activation_patterns(
            self._generate_random_text(len(text))
        )
        p0_empirical = self.estimate_activation_distribution(patterns_ref)
        
        # 计算Chernoff信息
        D_chernoff, opt_lambda = self.compute_chernoff_information(
            p0_empirical, 
            p1_empirical
        )
        
        # 估计衰减（使用标定的常数C=1.5）
        C = 1.5
        D_adv_lower_bound = max(
            0, 
            D_chernoff - C * np.sqrt(gamma_attack) * np.sqrt(D_chernoff)
        )
        
        # 判决
        verdict = D_adv_lower_bound > threshold
        
        return {
            'D_chernoff': D_chernoff,
            'D_adv_lower_bound': D_adv_lower_bound,
            'gamma_estimated': gamma_attack,
            'optimal_lambda': opt_lambda,
            'watermarked': verdict,
            'confidence': D_adv_lower_bound / max(D_chernoff, 1e-6) * 100,
            'sample_complexity': np.log(1/0.01) / max(D_adv_lower_bound, 1e-6)
        }
```

---

### 8.3 攻击强度估计的精细化

```python
def estimate_attack_strength_comprehensive(
    x_original, 
    x_attacked, 
    model,
    estimation_method='hybrid'
):
    """
    三层估计策略，从保守到精确
    """
    
    results = {}
    
    # 方法1：编辑距离上界（最保守）
    edit_dist = compute_edit_distance(x_original, x_attacked)
    text_len = len(x_original.split())
    vocab_size = model.vocab_size
    
    gamma_upper_bound = (edit_dist / text_len) * np.log(vocab_size / 1000)
    results['method_1_edit_distance'] = {
        'gamma': gamma_upper_bound,
        'type': 'upper_bound',
        'conservativeness': 'high'
    }
    
    # 方法2：BERT语义相似度补正
    from transformers import AutoTokenizer, AutoModel
    
    bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    bert_model = AutoModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
    
    def get_bert_embedding(text):
        inputs = bert_tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        # 使用[CLS] token的隐藏状态
        return outputs.hidden_states[-1][:, 0, :].squeeze()
    
    emb_original = get_bert_embedding(x_original)
    emb_attacked = get_bert_embedding(x_attacked)
    
    semantic_similarity = torch.nn.functional.cosine_similarity(
        emb_original.unsqueeze(0), 
        emb_attacked.unsqueeze(0)
    ).item()
    
    # KL散度与语义相似度的关系
    # 当sim接近1时，KL应较小；当sim降低时，KL增加
    semantic_penalty = 1.0 - semantic_similarity
    gamma_semantic_adjusted = gamma_upper_bound * (1 + semantic_penalty)
    
    results['method_2_semantic'] = {
        'gamma': gamma_semantic_adjusted,
        'semantic_similarity': semantic_similarity,
        'type': 'upper_bound_refined'
    }
    
    # 方法3：实测KL散度（最精确，需要token级统计）
    tokens_original = model.tokenize(x_original)
    tokens_attacked = model.tokenize(x_attacked)
    
    # 构造token级分布（使用smoothing避免零概率）
    from collections import Counter
    
    alpha = 1.0  # Laplace smoothing
    counter_original = Counter(tokens_original)
    counter_attacked = Counter(tokens_attacked)
    
    vocab = set(tokens_original) | set(tokens_attacked)
    
    prob_original = {
        t: (counter_original.get(t, 0) + alpha) / (len(tokens_original) + alpha * len(vocab))
        for t in vocab
    }
    
    prob_attacked = {
        t: (counter_attacked.get(t, 0) + alpha) / (len(tokens_attacked) + alpha * len(vocab))
        for t in vocab
    }
    
    gamma_measured = sum(
        prob_attacked[t] * np.log(prob_attacked[t] / prob_original[t])
        for t in vocab
    )
    
    results['method_3_measured_kl'] = {
        'gamma': gamma_measured,
        'type': 'measured',
        'conservativeness': 'none'
    }
    
    # 混合策略
    if estimation_method == 'hybrid':
        # 取中间值，兼顾保守性和准确性
        gamma_hybrid = (gamma_upper_bound + gamma_measured) / 2
        results['hybrid'] = {
            'gamma': gamma_hybrid,
            'reasoning': 'average of upper_bound and measured'
        }
    elif estimation_method == 'conservative':
        gamma_final = gamma_upper_bound
    else:  # 'aggressive'
        gamma_final = gamma_measured
    
    return {
        'estimates': results,
        'recommended_gamma': results.get('hybrid', {}).get('gamma', gamma_measured),
        'upper_bound': gamma_upper_bound,
        'lower_bound': gamma_measured
    }
```

---

## 九、生产环境的完整检查清单

### 9.1 部署前验证（Pre-Deployment Verification）

```python
class DeploymentValidator:
    """
    确保系统的理论约束在实际环境中满足
    """
    
    def validate_all(self, model, config):
        checks = []
        
        # 检查1：Lipschitz常数在预期范围
        L_g_measured = self.measure_lipschitz_constant(model)
        check1 = {
            'name': 'Lipschitz Constant Check',
            'measured': L_g_measured,
            'expected_range': (1.5, 3.0),
            'status': 1.5 <= L_g_measured <= 3.0,
            'action': 'Apply gradient clipping if failed'
        }
        checks.append(check1)
        
        # 检查2：综合常数C的拟合质量
        C_prop, R2 = self.calibrate_constant(model, config['validation_data'])
        check2 = {
            'name': 'Combined Constant Calibration',
            'C_prop': C_prop,
            'R_squared': R2,
            'status': R2 > 0.90,
            'action': 'Increase sample size if R² < 0.90'
        }
        checks.append(check2)
        
        # 检查3：安全系数c的最优性验证
        c_opt = config['c']
        lambda_weight = config['lambda']
        c_min_required = config['C']  # 通过calibrate_combined_constant得到
        
        check3 = {
            'name': 'Safety Coefficient Validity',
            'c_configured': c_opt,
            'c_minimum_required': c_min_required,
            'status': c_opt > c_min_required,
            'margin': (c_opt - c_min_required) / c_min_required * 100,
            'action': 'Increase c if margin < 10%'
        }
        checks.append(check3)
        
        # 检查4：性能成本可接受性
        ppl_drop = self.measure_perplexity_drop(model)
        max_acceptable_ppl_drop = config.get('max_ppl_drop', 2.0)
        
        check4 = {
            'name': 'Performance Cost Acceptability',
            'measured_ppl_drop': ppl_drop,
            'max_acceptable': max_acceptable_ppl_drop,
            'status': ppl_drop <= max_acceptable_ppl_drop,
            'action': 'Reduce c if PPL drop exceeds threshold'
        }
        checks.append(check4)
        
        # 检查5：Top-k激活的排名稳定性
        ranking_exchange_freq = self.measure_ranking_exchange_frequency(model)
        
        check5 = {
            'name': 'Ranking Exchange Frequency',
            'frequency': ranking_exchange_freq,
            'expected': '<10%',
            'status': ranking_exchange_freq < 0.10,
            'action': 'Reduce delta_logit magnitude if too high'
        }
        checks.append(check5)
        
        # 总体判决
        all_passed = all(check['status'] for check in checks)
        
        return {
            'passed': all_passed,
            'checks': checks,
            'deployment_ready': all_passed,
            'issues': [ch for ch in checks if not ch['status']]
        }
```

---

### 9.2 运行时监控（Runtime Monitoring）

```python
class RuntimeMonitor:
    """
    持续监控系统是否偏离理论预设
    """
    
    def __init__(self, model, baseline_metrics):
        self.model = model
        self.baseline = baseline_metrics
        self.deviations = []
    
    def monitor_detection_quality(self, batch_texts, batch_results):
        """
        监控：检测结果是否符合理论预测
        """
        
        for text, result in zip(batch_texts, batch_results):
            gamma_actual = estimate_attack_strength_comprehensive(
                text, 
                text,  # 这里应该是受攻击版本
                self.model
            )['recommended_gamma']
            
            D_adv_actual = result['D_adv_lower_bound']
            
            # 与理论下界对比
            D_adv_theoretical = (
                result['D_chernoff'] - 
                self.baseline['C'] * np.sqrt(gamma_actual) * np.sqrt(result['D_chernoff'])
            )
            
            deviation = abs(D_adv_actual - D_adv_theoretical) / max(D_adv_theoretical, 1e-6)
            
            if deviation > 0.3:  # 偏离超过30%
                self.deviations.append({
                    'timestamp': datetime.now(),
                    'text': text[:100],  # 前100个字符
                    'deviation_ratio': deviation,
                    'alert': 'HIGH_DEVIATION'
                })
    
    def check_model_drift(self):
        """
        检测：模型权重是否有异常漂移
        """
        current_L_g = self.measure_lipschitz_constant()
        baseline_L_g = self.baseline['L_g']
        
        if abs(current_L_g - baseline_L_g) / baseline_L_g > 0.2:
            return {
                'alert': 'MODEL_DRIFT_DETECTED',
                'previous_L_g': baseline_L_g,
                'current_L_g': current_L_g,
                'action': 'Re-calibrate L_g and re-test robustness'
            }
        
        return {'status': 'normal'}
    
    def get_health_report(self):
        """生成健康报告"""
        return {
            'total_monitored': len(self.deviations),
            'high_deviations': sum(1 for d in self.deviations if d['deviation_ratio'] > 0.3),
            'model_drift_status': self.check_model_drift(),
            'anomalies': self.deviations[-10:]  # 最近10条
        }
```

---

## 十、关键差异点总结：Token-Logit vs MoE

### 10.1 核心机理对比表

| 维度 | Token-Logit (范式A) | MoE Expert Activation (范式B) |
|------|-------------------|------------------------------|
| **信号空间** | R^{\|V\|} (词汇表维) | {0,1}^K (专家激活) |
| **水印位置** | logit修改 | gating网络输出 |
| **攻击入口** | 输入词元 ✓直接 | 输入词元 ✗间接 |
| **解耦性** | S_A ⊂ A (无) | S_B ∩ A=∅ (有) |
| **衰减规律** | ∆Z ∝ γ | ∆D* ∝ √γ |
| **临界失效点** | γ_crit ≈ 0.01-0.03 | γ_crit ≫ 0.1 |
| **样本复杂度** | O(1/γ) | O(1/(√γ)) |
| **性能成本** | 低(<1% PPL) | 中等(1-3% PPL) |
| **部署难度** | 简单 | 中等 |

### 10.2 定量优势对标

```
场景：对手执行强度γ=0.03的释义攻击

Token-Logit:
  初始检测能力：Z=6.0
  衰减：∆Z = 125 × 0.03 = 3.75
  剩余能力：Z' = 6.0 - 3.75 = 2.25 < 4.0 (检测失效)
  结论：被击破

MoE Expert:
  初始检测能力：D*=0.1 nats
  衰减：∆D* ≈ 1.5×√0.03×√0.1 ≈ 0.025 nats
  剩余能力：D*' = 0.1 - 0.025 = 0.075 nats > 0 (保存)
  结论：生存，并保持75%初始强度
```

---

## 十一、最终实现架构图

```
┌─────────────────────────────────────────────────────────────┐
│  应用层：内容验证、版权保护、模型签名                          │
└───────────────┬─────────────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────────────┐
│  业务逻辑层                                                    │
│  ├─ Embed Module: 根据λ选择最优c，嵌入水印                     │
│  ├─ Detect Module: 计算Chernoff信息，判定是否有水印            │
│  └─ Calibrate Module: 标定L_g, C, c*三个参数                  │
└───────────────┬─────────────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────────────┐
│  信息论核心层                                                  │
│  ├─ Pinsker不等式: ∥D(X')-D(X)∥_TV ≤ √(γ/2)                 │
│  ├─ Lipschitz传播: 输入扰动→激活分布变化                       │
│  ├─ Chernoff稳定性: 分布扰动→检测能力衰减                     │
│  └─ 综合结果: ∆D* ∝ √γ (次线性衰减)                          │
└───────────────┬─────────────────────────────────────────────┘
                │
┌───────────────▼─────────────────────────────────────────────┐
│  模型层                                                        │
│  ├─ gating_network: 修改logit → 改变激活模式                  │
│  ├─ top_k_activation: 选择K个激活专家                         │
│  └─ forward: 使用激活的专家执行计算                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 十二、发表与可重复性

### 论文关键数字总结

**表B1** - Token-Logit线性衰减（对比基准）
```
验证了：∆Z(γ) = C_linear·γ，其中C_linear ≈ 125
```

**表C1** - MoE次线性衰减（核心贡献）
```
验证了：∆D*(γ) ≈ 0.051·√γ  (范式B保持显著优势)
```

**表D1** - Lipschitz常数标定
```
LLaMA-7B-MoE:   L_95 = 2.3 ± 0.2 (与理论2.0吻合)
```

**表E1** - 最优安全系数c*
```
λ=1时，c* ≈ 1.33C (目标函数最小值=452)
```

### 可重复性保证

1. **开源数据集**：WikiText-103验证集（已公开）
2. **标准模型**：LLaMA-7B-MoE、Mixtral-8x7B（Hugging Face）
3. **算法伪代码**：Algorithm 1-3完整给出
4. **参数设置**：所有超参数已具体数值化
5. **误差分析**：包含理论-实验偏差的原因分析

---

## 总结

该论文的核心贡献不在于新的水印思路，而在于：

1. **严格的信息论证明**：从Pinsker不等式→Chernoff稳定性→次线性衰减的完整链条

2. **信号-攻击解耦的形式化**：明确了为什么MoE激活模式能逃脱输入级攻击

3. **可落地的工程参数化**：L_g、C、c三个参数的标定方法完整可执行

4. **定量的性能保证**：在γ=0.03时，范式B仍保持77%检测能力，而范式A已完全失效

实现这个系统的关键在于精确理解Pinsker不等式如何通过Lipschitz传播放大扰动，以及Top-k离散性对Chernoff信息的影响。