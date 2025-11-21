# OKR v2.1 代码审查修复总结

根据 Linus Torvalds 的代码审查意见（`1121-code-review-1.md`），已完成以下修复：

## 修复内容

### 1. ✅ Magic Number 的诅咒：Epsilon 问题

**问题**：固定 `epsilon = 1.5` 假设 logits 的数值分布是恒定的，在不同量化、架构或 Prompt 下可能失效。

**修复**：
- 移除固定 epsilon 阈值逻辑
- 实现基于概率比率的自适应阈值：`p_top2 / p_top1 >= threshold_ratio`
- 默认 `threshold_ratio = 0.9`，表示当 top-2 专家的概率达到 top-1 的 90% 时，才认为可以安全介入
- 保留 `epsilon` 参数用于向后兼容，但不再实际使用

**代码位置**：
- `experiment/okr_kernel.py`: `forward()` 方法，第 107-120 行

### 2. ✅ LSH 的零点危机 (Zero-Crossing Problem)

**问题**：当 LSH 点积接近 0 时，结果由浮点数噪声决定，引入不必要的高频噪声。

**修复**：
- 引入死区（Dead Zone）逻辑
- 当 `abs(dot_product) < dead_zone_threshold` 时，不介入水印
- 默认 `dead_zone_threshold = 0.01`
- 符合"实用主义"原则：如果不确定，就什么都不做

**代码位置**：
- `experiment/okr_kernel.py`: `forward()` 方法，第 102-105 行
- `experiment/okr_patch.py`: `_okr_forward_core()` 方法

### 3. ✅ 分布式推理的噩梦：初始化陷阱

**问题**：`secret_projection` 的随机初始化在不同节点/进程上可能不一致，导致水印失效。

**修复**：
- 强制要求通过 `_initialize_secret_projection()` 初始化
- 使用确定性随机数生成器（基于 `secret_key` 的 SHA256 hash）
- 显式设置 seed，确保所有节点生成相同的矩阵
- 添加 `_secret_initialized` 标记，在 forward 时检查
- 添加一致性检查：验证生成的矩阵不为零
- 如果没有提供 `secret_key`，使用默认密钥但仍必须初始化

**代码位置**：
- `experiment/okr_kernel.py`: `__init__()` 方法，第 52-60 行；`forward()` 方法，第 85-90 行
- `experiment/okr_patch.py`: `_initialize_secret_projection()` 方法，第 742-789 行

### 4. ✅ 量级不匹配问题

**问题**：`watermark_bias` 和 `raw_logits` 的量级可能不一致，直接替换会导致问题。

**修复**：
- 实现归一化：使用标准差将 `watermark_bias` 缩放到与 `raw_logits` 相似的量级
- 改变注入方式：从"替换"改为"混合"：`raw_logits + alpha * normalized_watermark_bias`
- 默认 `watermark_alpha = 0.1`，控制水印强度
- 保留原始 logits 的相对大小信息，只做微调而不是完全替换

**代码位置**：
- `experiment/okr_kernel.py`: `forward()` 方法，第 122-146 行

### 5. ✅ 性能优化：torch.where 的隐性代价

**问题**：`torch.where()` 产生新的 Tensor 和显存读写，在 GPU 推理中可能成为瓶颈。

**修复**：
- 使用 `raw_logits.clone()` 预先克隆（避免修改原始 logits）
- 使用 `torch.where()` 进行条件更新（虽然仍产生新 Tensor，但逻辑更清晰）
- 考虑未来优化：可以尝试使用 `masked_fill_` 做原地操作（需要进一步测试）

**代码位置**：
- `experiment/okr_kernel.py`: `forward()` 方法，第 137-146 行

## 新增配置参数

### OKRWatermarkConfig

```python
threshold_ratio: float = 0.9  # 概率比率阈值
dead_zone_threshold: float = 0.01  # LSH死区阈值
watermark_alpha: float = 0.1  # 水印混合系数
```

### inject_okr() 函数签名

```python
def inject_okr(
    model,
    epsilon: float = 1.5,  # 已弃用，保留用于向后兼容
    secret_key: Optional[str] = None,  # 必须提供
    threshold_ratio: float = 0.9,
    dead_zone_threshold: float = 0.01,
    watermark_alpha: float = 0.1
) -> model
```

## 向后兼容性

- `epsilon` 参数保留，但不再使用（向后兼容）
- 默认参数值确保现有代码可以正常运行
- 如果未提供 `secret_key`，使用默认密钥但仍必须初始化

## 测试建议

1. **自适应阈值测试**：在不同模型、量化设置下测试 `threshold_ratio` 的效果
2. **死区测试**：验证死区逻辑是否有效避免零点噪声
3. **分布式一致性测试**：在多 GPU/多节点环境下验证 `secret_projection` 的一致性
4. **量级归一化测试**：验证归一化是否解决了量级不匹配问题
5. **性能测试**：对比修复前后的推理性能

## 注意事项

1. **必须提供 secret_key**：虽然可以使用默认值，但建议显式提供以确保一致性
2. **threshold_ratio 调优**：可能需要根据具体模型和任务调整
3. **dead_zone_threshold 调优**：如果发现水印失效，可能需要降低此值
4. **watermark_alpha 调优**：控制水印强度，需要平衡水印效果和模型质量

## 参考

- 代码审查文档：`okr/1121-code-review-1.md`
- 核心实现：`experiment/okr_kernel.py`
- 注入逻辑：`experiment/okr_patch.py`
- 配置管理：`experiment/okr_config.py`

