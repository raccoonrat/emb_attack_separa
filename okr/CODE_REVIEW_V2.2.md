# Linux Torvalds Code Review: Post-Mortem of OKR V2.2 Fix

**Status:** MERGED (Conceptually)  
**Verdict:** IT RUNS. (Finally.)

---

## 1. 崩溃修复 (The Crash Fix)

在 `okr_detector.py` 里，我看到了这一行：

```python
if actual_selected_experts.device != hidden_states.device:
    actual_selected_experts = actual_selected_experts.to(hidden_states.device)
```

**Good.** 这就是我在上一轮 Review 里要求的"防御性编程"。你不再假设世界是完美的，不再假设所有张量都神奇地住在同一块显存里。这是编写内核级代码（或者任何底层基础设施）应有的态度。

日志证明了它的有效性：

```
2025-11-21 16:07:28,544 [INFO] okr_detector: 命中率: 0.3333 (4/12), 随机基线: 0.1250, 阈值: 0.2500
```

检测器工作了。它拿到了数据，对了齐，算了分，给了结论。

---

## 2. 结果分析：理论与现实的碰撞

让我们看看 `results.json` 里的数字。这才是真正的试金石。

### A. 命中率 (Hit Rate)

- **随机基线:** 0.1250 (1/8 专家)
- **你的结果:** 0.3333, 0.3333, 0.4019
- **平均:** 0.3562

**结论:**

**It works.** 你的水印信号显著高于噪音。

0.35 的命中率意味着大概有 1/3 的 Token 是完全听从水印指挥的，而另外 2/3 是由模型自己决定的（或者水印尝试了但失败了）。

这比之前那个虚假的 1.0 好太多了。这表明"机会主义路由"机制正在生效——模型只在部分时刻让步。

### B. 生成质量 (Text Quality)

这里有个坏消息。看看模型吐出来的东西：

- **Sample 0:** `. fox. The lazy dog.`
- **Sample 2:** `... Climate change is a global crisis.... Climat`

**垃圾。** (Rubbish.)

模型陷入了重复循环（Repetition Loop），或者生成的句子支离破碎。

**原因诊断：**

1. **模型太弱 (Switch Base is dumb):** `google/switch-base-8` 是个很老的模型，而且是 T5 架构。它本身就不擅长这种 Open-ended Generation（续写）。它更擅长"翻译"或"填空"。这不是你代码的错，是模型选得烂。

2. **扰动还是太强:** 虽然你用了 `watermark_alpha=0.1`，但对于 Switch Transformer 这种本身就很脆弱的 MoE 架构来说，可能还是太重了。Switch Transformer 的 Router 极其敏感，稍微推一把，它就可能选到一个完全不相关的专家，导致上下文崩塌。

---

## 3. 下一步建议 (The Next Step)

代码逻辑现在是健壮的。你不需要再改 kernel 或 patch 的核心逻辑了。

现在的任务是 **调参 (Tuning)**。

如果你想看到漂亮的句子和高命中率并存：

1. **换个模型:** 别用 `switch-base-8` 了。去用 `deepseek-ai/deepseek-moe-16b-base` 或者 `mistralai/Mixtral-8x7B-v0.1`。那才是现代的 MoE。它们对 Logits 扰动的容忍度要高得多。

2. **降低 Alpha:** 如果非要用 Switch，把 `watermark_alpha` 降到 `0.05`。

3. **增加 Repetition Penalty:** 在 `generate()` 函数里加一个参数 `repetition_penalty=1.2`。这能防止那个愚蠢的 `... global crisis.... Climat` 循环。

---

## 最终判决

**代码通过。可以合并。**

但作为架构师，我建议你在生产环境部署前，把那个该死的 Switch 模型换掉。我们是做水印，不是做复读机。

---

**Linus**

---

## 实施状态

- [x] 设备不匹配修复（`okr_detector.py`）
- [x] 添加 `repetition_penalty` 参数（`okr_experiment.py`）
- [x] 配置类中添加 `repetition_penalty` 默认值 1.2（`okr_config.py`）
- [ ] 测试新的生成质量
- [ ] 考虑迁移到更好的模型（DeepSeek-MoE 或 Mixtral）

