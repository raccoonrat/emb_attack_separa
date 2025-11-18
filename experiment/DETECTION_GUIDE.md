# 水印检测指南

## 重要提示：`secret_key` 的作用

**`secret_key` 是水印检测的关键参数！**

- **嵌入水印时**：使用 `secret_key` 生成确定性种子，选择"绿色专家"（正偏置）和"红色专家"（负偏置）
- **检测水印时**：**必须使用相同的 `secret_key`**，才能正确提取和计算 p0、p1 分布

**如果 `secret_key` 不匹配，检测将失败！**

## 检测步骤

### 1. 基本检测（使用相同的 secret_key）

```bash
# 嵌入水印
python main.py --mode embed \
    --model_name google/switch-base-8 \
    --prompt "Your text" \
    --secret_key "my_secret_key_123"

# 检测水印（使用相同的 secret_key）
python main.py --mode detect \
    --model_name google/switch-base-8 \
    --text_to_check "生成的完整文本" \
    --secret_key "my_secret_key_123"
```

### 2. 检测参数说明

```bash
python main.py --mode detect \
    --model_name google/switch-base-8 \
    --text_to_check "要检测的文本" \
    --secret_key "my_secret_key_123" \  # 必须与嵌入时相同！
    --c_star 2.0 \                       # 安全系数（默认 2.0）
    --gamma_design 0.03 \                # 设计攻击强度（默认 0.03）
    --tau_alpha 20.0                     # 检测阈值（默认 20.0）
```

### 3. 参数说明

- `--secret_key`: **必须与嵌入时使用的相同**
- `--c_star`: 安全系数 c*（默认 2.0）
- `--gamma_design`: 设计攻击强度 γ（默认 0.03）
- `--tau_alpha`: LLR 检测阈值（默认 20.0）

### 4. 检测输出

```
--- Detection Result ---
Result: Watermark DETECTED (Score: 25.34)
------------------------
```

或

```
--- Detection Result ---
Result: Watermark NOT DETECTED (Score: 5.21)
------------------------
```

## 完整示例

### 示例 1：基本使用

```bash
# 1. 嵌入水印
python main.py --mode embed \
    --model_name google/switch-base-8 \
    --prompt "The quick brown fox jumps over the lazy dog" \
    --secret_key "my_key_123"

# 输出：生成的带水印文本（例如："The quick brown fox jumps over the lazy dog. It is a beautiful day.")

# 2. 检测水印（使用生成的文本）
python main.py --mode detect \
    --model_name google/switch-base-8 \
    --text_to_check "The quick brown fox jumps over the lazy dog. It is a beautiful day." \
    --secret_key "my_key_123"
```

### 示例 2：使用不同的 secret_key（应该检测不到）

```bash
# 嵌入时使用 secret_key_1
python main.py --mode embed \
    --model_name google/switch-base-8 \
    --prompt "Hello world" \
    --secret_key "key_1"

# 检测时使用不同的 secret_key（应该检测不到）
python main.py --mode detect \
    --model_name google/switch-base-8 \
    --text_to_check "生成的文本" \
    --secret_key "key_2"  # 不同的 key，检测应该失败
```

## 检测原理

1. **Patch 模型**：使用相同的 `secret_key` 和 `epsilon` 值 patch 模型
2. **提取数据**：从 patch 后的模型中提取 p0（原始分布）和 p1（修改后分布）
3. **计算 LLR**：计算对数似然比 Λ_n = Σ_i log(p1(S_i)/p0(S_i))
4. **判决**：如果 LLR > τ_α，判为有水印

## 常见问题

### Q: 为什么检测不到水印？

**可能原因**：
1. `secret_key` 不匹配（最常见）
2. 检测的文本不是嵌入时生成的完整文本
3. 文本经过了攻击（如改写），导致水印被破坏
4. 阈值 `tau_alpha` 设置过高

### Q: 如何选择合适的阈值？

使用 `calibrate` 模式来标定阈值：

```bash
python main.py --mode calibrate \
    --model_name google/switch-base-8 \
    --num_calib_samples 100 \
    --secret_key "my_key_123"
```

### Q: 检测时可以使用不同的 epsilon 吗？

**不建议**。检测时应该使用与嵌入时相同的 `epsilon` 值（由 `c_star` 和 `gamma_design` 计算得出）。

## 注意事项

1. **`secret_key` 必须匹配**：这是最重要的！
2. **检测完整文本**：使用嵌入时生成的完整文本，而不是原始提示
3. **参数一致性**：检测时的 `c_star` 和 `gamma_design` 应该与嵌入时相同
4. **模型版本**：确保使用相同的模型版本

