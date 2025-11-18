# LLR 阈值 τ_α 的理论依据和标定方法

## 理论依据（论文第3节）

### 1. Neyman-Pearson 引理

根据论文定理3.1（Neyman-Pearson引理），最优检测器为似然比检验：

```
判决规则: Λ_n > τ_α => 判为H1 (有水印)
```

其中：
- **Λ_n**: LLR统计量 = Σ_i log(p1(S_i)/p0(S_i))
- **τ_α**: 检测阈值
- **α**: 第一类错误率（假阳性率），即 P(判为H1 | H0为真)

### 2. 阈值的标定方法

**正确的阈值标定方法**（论文第3节）：

1. **在H0假设下（无水印）**，收集大量样本
2. 对每个样本计算LLR统计量
3. 选择阈值使得：**P(Λ > τ_α | H0) = α**

即：阈值应该是H0分布下的 **(1-α) 分位数**

### 3. 代码实现

代码中已经实现了 `calibrate_threshold` 方法：

```python
def calibrate_threshold(self, null_samples: List[str], num_bootstrap: int = 1000) -> float:
    """
    标定检测阈值 τ_α (论文第3节)
    
    在H0假设下 (无水印), 计算LLR统计量的分布,
    选择阈值使得 P(Λ > τ_α | H0) = α
    """
    llr_scores = []
    
    # 对每个无水印样本计算LLR
    for text in null_samples:
        # ... 计算LLR ...
        llr_scores.append(llr)
    
    # 计算(1-α)分位数作为阈值
    tau_alpha = np.percentile(llr_scores, (1 - self.alpha) * 100)
    return tau_alpha
```

## 为什么直接改为5.0不合理？

### 问题分析

1. **20.0 的问题**：
   - 阈值过高，可能导致漏检（假阴性）
   - 即使有水印，LLR分数也可能低于20.0

2. **5.0 的问题**：
   - 仍然是**随意设置**的，没有理论依据
   - 可能导致假阳性（误判无水印文本为有水印）
   - 没有考虑实际的H0分布

3. **正确的做法**：
   - 使用 `calibrate_threshold` 方法在H0假设下标定
   - 根据实际的无水印样本分布来确定阈值

## 如何正确标定阈值？

### 方法1：使用 calibrate 模式（推荐）

```bash
# 1. 准备无水印样本数据集
# 2. 运行标定
python experiment/main.py --mode calibrate \
    --model_name google/switch-base-8 \
    --dataset_name your_dataset \
    --num_calib_samples 1000 \
    --secret_key "my_key_123"
```

**注意**：当前 `calibrate` 模式只标定了 Lg、C、c*，**没有标定 τ_α**。需要改进。

### 方法2：手动标定阈值

```python
from detector import LLRDetector
from mves_watermark_corrected import patch_switch_model_with_watermark

# 1. 准备无水印样本
null_samples = [
    "This is a normal text without watermark.",
    "Another normal text.",
    # ... 更多样本
]

# 2. Patch 模型（使用相同的配置）
model = ...  # 加载模型
config = ...  # 配置
patched_model = patch_switch_model_with_watermark(model, config)

# 3. 创建检测器
detector = LLRDetector(patched_model, tokenizer, tau_alpha=20.0, alpha=0.01)

# 4. 标定阈值
tau_alpha_calibrated = detector.calibrate_threshold(null_samples)

print(f"标定后的阈值: {tau_alpha_calibrated:.4f}")

# 5. 使用标定的阈值进行检测
detector.tau_alpha = tau_alpha_calibrated
```

### 方法3：根据实际LLR分布调整

如果无法进行完整标定，可以：

1. **收集一些无水印样本的LLR分数**
2. **观察分布**：
   - 如果大部分无水印样本的LLR < 5.0，则阈值可以设为 5.0-10.0
   - 如果大部分无水印样本的LLR < 2.0，则阈值可以设为 2.0-5.0
3. **根据假阳性率要求调整**：
   - α = 0.01（1%假阳性率）：使用99分位数
   - α = 0.05（5%假阳性率）：使用95分位数

## 当前代码的问题

### 问题1：calibrate 模式没有标定 τ_α

当前 `calibrate` 模式只标定了：
- Lg（Lipschitz常数）
- C（系统常数）
- c*（安全系数）

**但没有标定 τ_α**！

### 问题2：默认阈值是随意设置的

- 原始默认值：20.0（可能太高）
- 修改后：5.0（仍然没有理论依据）

## 建议的改进方案

### 方案1：改进 calibrate 模式

在 `calibrate` 模式中添加 τ_α 标定：

```python
# 在 calibrate 模式中
# 1. 使用无水印样本标定阈值
detector = LLRDetector(patched_model, tokenizer, tau_alpha=20.0, alpha=0.01)
tau_alpha_calibrated = detector.calibrate_threshold(null_samples)
print(f"标定后的阈值 τ_α: {tau_alpha_calibrated:.4f}")
```

### 方案2：提供阈值标定脚本

创建一个独立的阈值标定脚本：

```bash
python experiment/calibrate_threshold.py \
    --model_name google/switch-base-8 \
    --null_samples_file null_samples.txt \
    --alpha 0.01 \
    --secret_key "my_key_123"
```

### 方案3：根据经验值设置

如果无法进行完整标定，可以根据经验：

- **保守设置**（低假阳性率）：τ_α = 10.0-15.0
- **平衡设置**：τ_α = 5.0-10.0
- **宽松设置**（低漏检率）：τ_α = 2.0-5.0

## 总结

1. **阈值应该通过H0假设下的实验标定**，而不是随意设置
2. **5.0 比 20.0 更合理**（因为20.0太高），但仍然**不是最优的**
3. **正确的做法**：使用 `calibrate_threshold` 方法在无水印样本上标定
4. **当前代码的改进方向**：在 `calibrate` 模式中添加 τ_α 标定功能

## 实际使用建议

如果暂时无法进行完整标定，可以：

1. **先使用较低的阈值**（如5.0）进行测试
2. **观察假阳性率**：如果误判太多，提高阈值
3. **观察漏检率**：如果检测不到水印，降低阈值
4. **最终目标**：实现完整的阈值标定流程

