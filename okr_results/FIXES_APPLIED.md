# OKR 实验问题修复记录

## 修复日期
2024-12-19

## 问题总结

根据实验结果分析，发现以下关键问题：

1. **权重未正确复制** - 日志显示 "警告: 无法找到 router 的权重，使用随机初始化"
2. **命中率过低** - 平均命中率 2.78%，远低于 12.5% 的随机基线
3. **检测器无法找到 OKR router** - 检测时无法访问 gate_network 和 secret_projection

## 修复内容

### 1. 修复权重复制逻辑 (`experiment/okr_patch.py`)

**问题**: SwitchTransformersTop1Router 使用 `classifier` 而不是 `gate` 作为权重层。

**修复**:
- 优先检查 `router.classifier`（SwitchTransformersTop1Router 的标准结构）
- 其次检查 `router.gate`（某些变体）
- 添加形状匹配检查和转置处理
- 添加详细的调试输出，显示权重复制状态

**关键代码**:
```python
# 方式1: router.classifier (SwitchTransformersTop1Router 使用 classifier)
if hasattr(router, 'classifier') and isinstance(router.classifier, torch.nn.Linear):
    gate_layer = router.classifier
    # ... 复制权重逻辑
```

### 2. 修复检测器 OKR Router 获取 (`experiment/okr_detector.py`)

**问题**: 检测器无法正确获取 OKR router 的 `gate_network` 和 `secret_projection`。

**修复**:
- 修改 `_get_router()` 返回原始 router 对象（而不是 `_okr_router`）
- 在 `verify_batch()` 中从 router 对象获取 `_okr_router`
- 确保使用 OKR router 的 `gate_network` 和 `secret_projection` 进行检测
- 添加设备检查，确保 `secret_projection` 在正确的设备上

**关键代码**:
```python
# 获取 OKR router
if hasattr(router, '_okr_router'):
    okr_router = router._okr_router
elif hasattr(router, 'gate_network') and hasattr(router, 'secret_projection'):
    okr_router = router

# 使用 OKR router 进行计算
raw_logits = okr_router.gate_network(hidden_states)
watermark_bias = torch.matmul(hidden_states, okr_router.secret_projection)
```

### 3. 改进路由数据保存 (`experiment/okr_patch.py`)

**问题**: 路由数据保存逻辑可能不够清晰。

**修复**:
- 简化路由数据保存：只保存最新的 `selected_experts`
- 同时保存到 `model._okr_routing_data` 和 `router._selected_experts`
- 添加 `router._okr_routing_weights` 供调试使用

### 4. 增强调试输出

**改进**:
- 在权重复制时添加详细的成功/失败信息（✓/✗ 标记）
- 在检测器中添加路由数据获取的调试信息
- 显示权重形状和 dtype 信息

## 预期效果

修复后应该能够：

1. **正确复制权重**: 从 `router.classifier` 成功复制权重到 OKR router
2. **提高命中率**: 使用正确的权重和 secret_projection，命中率应该显著提高
3. **正确检测**: 检测器能够找到并使用 OKR router 进行验证

## 下一步测试

运行实验验证修复效果：

```bash
cd experiment
python test_okr_basic.py
```

检查：
- 日志中是否显示 "✓ 已复制权重"
- 命中率是否提高到合理水平（> 12.5%）
- 是否有样本被正确分类为 "Watermarked"

## 技术细节

### SwitchTransformersTop1Router 结构

SwitchTransformersTop1Router 使用 `classifier` 作为其线性层：
- `router.classifier`: `torch.nn.Linear(input_dim, num_experts)`
- 权重形状: `[num_experts, input_dim]` 或 `[input_dim, num_experts]`

### OKR Router 结构

OKR router 使用 `gate_network` 作为其线性层：
- `okr_router.gate_network`: `torch.nn.Linear(input_dim, num_experts)`
- 权重形状: `[num_experts, input_dim]`

需要确保形状匹配，必要时进行转置。

## 参考

- `experiment/okr_patch.py`: 权重复制逻辑
- `experiment/okr_detector.py`: 检测器逻辑
- `experiment/mves_watermark_corrected.py`: 参考实现

