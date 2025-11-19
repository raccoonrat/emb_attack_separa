# 项目架构文档

## 论文逻辑验证

### ✅ 核心公式实现
- **安全系数c**: `ε = c²γ` (论文定义5.1) - ✅ 已正确实现
- **KL约束**: `KL(p1||p0) = ε` (论文定义3.2) - ✅ 已正确实现
- **次线性衰减**: `D*_adv ≥ D*(p0,p1) - C√(γ·D*(p0,p1))` (论文定理4.5) - ✅ 已正确实现

### ✅ 模块对应关系

| 论文章节 | 代码模块 | 状态 |
|---------|---------|------|
| 定义3.2 (水印嵌入) | `mves_watermark_corrected.py` | ✅ 完成 |
| 定理3.1 (LLR检测器) | `detector.py` | ✅ 完成 |
| 定理3.2 (Chernoff信息) | `detector.py::compute_chernoff_information` | ✅ 完成 |
| 第7.1节 (L_g标定) | `calibration.py::calibrate_Lg` | ⚠️ 部分完成 |
| 第7.2节 (C标定) | `calibration.py::calibrate_C` | ⚠️ 部分完成 |
| 第7.3节 (c*标定) | `calibration.py::calibrate_C_star` | ⚠️ 部分完成 |
| 实验A-E | `experiments.py` | ⚠️ 实验B需要KGW实现 |

## 代码结构

```
experiment/
├── core/                    # 核心模块（待重构）
│   ├── watermark.py        # 水印嵌入（统一实现）
│   ├── detector.py          # 检测器
│   └── config.py            # 配置管理
├── experiments/             # 实验框架
│   ├── base.py              # 实验基类
│   ├── experiment_a.py      # 实验A: γ实测
│   ├── experiment_c.py      # 实验C: 次线性衰减
│   └── experiment_e.py      # 实验E: c*最优性
├── utils/                    # 工具函数
│   ├── attacks.py           # 攻击实现
│   ├── calibration.py       # 标定方法
│   └── metrics.py            # 评估指标
└── main.py                   # 主入口
```

## 待清理文件

以下文件可以删除或归档：
- `moe_watermark.py` - 旧实现（CausalLM），已被`mves_watermark_corrected.py`替代
- `moe_watermark_enhanced.py` - 未使用
- `mves_watermark.py` - 旧实现（LogitsProcessor方式），已被`mves_watermark_corrected.py`替代

## 工程化改进计划

1. **模块化重构** - 建立清晰的目录结构
2. **错误处理** - 统一异常处理和错误恢复
3. **日志系统** - 添加结构化日志
4. **性能优化** - 减少重复计算，批量处理
5. **单元测试** - 为核心模块添加测试
6. **文档完善** - API文档和使用指南

