# MoE水印实现说明

## 概述

本实现严格按照论文《Signal-Attack Decoupling in MoE Watermarks: A Rigorous Information-Theoretic Analysis of Provable Robustness》的理论框架构建,完全符合论文思想。

## 核心理论框架

### 1. 信号-攻击解耦 (论文第1节)

- **信号空间**: MoE专家激活模式空间 $\mathcal{S}_B = \{0,1\}^K$
- **攻击空间**: 输入文本空间 $\mathcal{X}$
- **解耦性**: $\mathcal{S}_B \cap \mathcal{X} = \emptyset$ (输入级攻击假设)

### 2. 水印嵌入 (论文第3节)

**核心公式**:
- 水印强度: $\epsilon = c^2 \gamma$ (论文定义5.1)
- KL散度约束: $\text{KL}(p_1 || p_0) = \epsilon$ (论文定义3.2)
- Logit修改: $l_1 = l_0 + \Delta l$

**实现位置**: `moe_watermark.py`
- `MoEWatermark.get_bias_vector()`: 生成偏置向量,确保KL散度满足要求
- `MoEWatermark.watermarked_router_forward()`: 水印路由前向传播

### 3. 最优检测器 (论文定理3.1-3.2)

**Neyman-Pearson引理**:
- 似然比检验: $\Lambda_n = \sum_i \log(p_1(S_i)/p_0(S_i))$
- 判决规则: $\Lambda_n > \tau_\alpha \Rightarrow$ 判为$H_1$ (有水印)

**Chernoff信息** (论文定理3.2):
- $D^*(p_0, p_1) = -\min_{\lambda \in [0,1]} \log(\sum_e p_0(e)^{1-\lambda} p_1(e)^\lambda)$
- 错误率指数衰减: $\log P_e(n) = -n \cdot D^*(p_0, p_1) + o(n)$

**实现位置**: `detector.py`
- `LLRDetector.compute_llr_from_data()`: 计算LLR统计量
- `LLRDetector.compute_chernoff_information()`: 计算Chernoff信息

### 4. 鲁棒性理论 (论文第4节)

**核心定理4.5**: 次线性衰减
- MoE水印: $D^*_{\text{adv}} \geq D^*(p_0, p_1) - C\sqrt{\gamma \cdot D^*(p_0, p_1)}$
- Token-Logit水印: $Z(\gamma) = z_0 - C_{\text{linear}} \gamma$ (线性衰减)

**关键参数**:
- $L_g$: Gating网络的Lipschitz常数 (论文引理4.4'')
- $C = C_{\text{stability}} \cdot C_{\text{prop}}$: 综合常数 (论文定理4.5)
- $c$: 安全系数 (论文定义5.1)

### 5. 参数标定 (论文第7节)

**算法1**: 标定$L_g$ (论文第7.1节)
- 方法A: Embedding空间扰动
- 方法B: Token级别扰动
- 方法C: 释义扰动

**算法2**: 标定$C$ (论文第7.2节)
- $C_{\text{prop}}$: 通过拟合 $\delta_i \approx C_{\text{prop}} \cdot \sqrt{\gamma_i}$
- $C_{\text{stability}}$: 通过Chernoff信息变化拟合

**算法3**: 标定$c^*$ (论文第7.3节)
- 优化目标: $\min_c [n^*(\gamma, c) + \lambda \Delta A(c)]$
- 样本复杂度: $n^* = \log(1/\delta) / [\gamma c(c-C)]$

**实现位置**: `calibration.py`

## 实验框架 (论文第7节)

### 实验A: 攻击强度γ的实测
- 目的: 验证γ上界估计的准确性
- 方法: 对比理论上界和实测KL散度

### 实验B: Token-Logit水印的线性衰减
- 目的: 验证定理2.5 (线性衰减)
- 方法: 测量不同γ下的z-score衰减

### 实验C: MoE水印的次线性衰减 (核心对比)
- 目的: 验证定理4.5 (次线性衰减) vs 定理2.5 (线性衰减)
- 方法: 对比两种范式的衰减速率

### 实验D: Lipschitz常数$L_g$的实测标定
- 目的: 验证$L_g$的实际值
- 方法: 使用论文算法1标定

### 实验E: 安全系数$c^*$的最优性验证
- 目的: 验证定理5.5的最优系数框架
- 方法: 网格搜索最优$c^*$

**实现位置**: `experiments.py`

## 使用方法

### 1. 参数标定

```bash
python main.py --mode calibrate \
               --model_name "mistralai/Mixtral-8x7B-v0.1" \
               --dataset_name "wikitext" \
               --num_calib_samples 1000
```

### 2. 水印嵌入

```bash
python main.py --mode embed \
               --model_name "mistralai/Mixtral-8x7B-v0.1" \
               --prompt "Your text here" \
               --secret_key "your_secret_key" \
               --c_star 2.0 \
               --gamma_design 0.03
```

### 3. 水印检测

```bash
python main.py --mode detect \
               --model_name "mistralai/Mixtral-8x7B-v0.1" \
               --text_to_check "Text to check" \
               --secret_key "your_secret_key" \
               --attack "paraphrase"  # 可选: 应用攻击测试鲁棒性
```

### 4. 运行实验

```bash
python main.py --mode experiment \
               --model_name "mistralai/Mixtral-8x7B-v0.1" \
               --dataset_name "wikitext" \
               --num_calib_samples 1000
```

## 关键设计决策

### 1. 水印强度控制

严格按照论文定义5.1: $\epsilon = c^2 \gamma$
- 在`MoEWatermark.__init__()`中计算`target_norm = sqrt(2ε)`
- 在`get_bias_vector()`中确保KL散度满足要求

### 2. 检测器实现

严格按照Neyman-Pearson引理:
- 使用LLR统计量作为检测依据
- 计算Chernoff信息用于理论分析
- 支持阈值自动标定

### 3. 攻击强度估计

支持两种方法 (论文第7.4节):
- 方法1: 编辑距离上界 (论文引理7.1)
- 方法2: KL散度精确估计

### 4. 实验可复现性

- 所有实验严格按照论文设置
- 结果保存为JSON格式
- 支持对照实验 (范式A vs 范式B)

## 理论保证

本实现严格遵循论文的所有理论结果:

1. **定理2.5**: Token-Logit水印的线性衰减 $O(\gamma)$
2. **定理4.5**: MoE水印的次线性衰减 $O(\sqrt{\gamma})$
3. **定理3.1**: Neyman-Pearson最优检测器
4. **定理3.2**: Chernoff信息的错误率指数衰减
5. **定理5.1-5.2**: 安全系数$c$的鲁棒性保证

## 注意事项

1. **模型架构**: 目前主要支持Mixtral架构,LLaMA-MoE支持待完善
2. **KGW实现**: 实验B需要实现KGW水印方法 (论文第2节)
3. **性能测量**: 实验E需要实现PPL测量功能
4. **计算资源**: 大模型实验需要足够的GPU内存

## 后续工作

1. 完善LLaMA-MoE架构支持
2. 实现KGW水印方法 (用于实验B)
3. 完善PPL测量功能 (用于实验E)
4. 添加更多攻击方法 (用于鲁棒性测试)
5. 优化计算效率 (批处理、缓存等)

## 参考文献

论文: "Signal-Attack Decoupling in MoE Watermarks: A Rigorous Information-Theoretic Analysis of Provable Robustness"

