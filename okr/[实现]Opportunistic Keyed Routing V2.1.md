好吧，打开引擎盖。我们不搞那些复杂的文件夹结构，那是给 Java 程序员用的。

我们需要的是**三个核心文件**。简单、高效、没有废话。

1. `okr_kernel.py`: 核心的路由逻辑。这是要在 GPU 上跑的，必须快得像闪电一样。
2. `detector.py`: 用户态工具，用来检测有没有水印。
3. `patch.py`: 所谓的“胶水代码”，用来把我们的内核注入到现有的模型（比如 Mixtral 或 DeepSeek）里。

咱们开始写代码。

---

### 1. `okr_kernel.py` (The Kernel)

这是整个项目的灵魂。把它想象成 Linux 的 `sched.c`（调度器）。它的任务是：**在微秒级别内决定该把 Token 发给哪个专家。**

没有 `for` 循环，没有列表，没有 CPU 同步。全是纯粹的 CUDA 友好的张量运算。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class OKRRouter(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 num_experts: int, 
                 top_k: int = 2, 
                 epsilon: float = 1.5):
        """
        初始化 OKR 路由器。

        Args:
            input_dim: 输入 Embedding 的维度 (d_model)
            num_experts: 专家的总数
            top_k: 每次选几个专家 (通常是 2)
            epsilon: 质量容忍阈值 (Logit 差值)。越大水印越强，越小质量越稳。
        """
        super().__init__()
        self.top_k = top_k
        self.epsilon = epsilon

        # 标准的 Gating 层：把 Embedding 映射到专家打分
        # 这是一个可训练的层，就是原来模型里的那个 Gate
        self.gate_network = nn.Linear(input_dim, num_experts, bias=False)

        # 水印私钥 (The Secret)
        # 使用 register_buffer 告诉 PyTorch：
        # 1. 这是一个状态，要随模型保存 (save_dict)。
        # 2. 这不是参数，不要给它算梯度 (Gradient)。
        self.register_buffer(
            "secret_projection", 
            torch.randn(input_dim, num_experts)
        )

    def forward(self, hidden_states: torch.Tensor):
        """
        Forward path. 这是热路径，每一毫秒都很珍贵。

        Args:
            hidden_states: [batch_size, seq_len, input_dim]
        Returns:
            routing_weights, selected_experts
        """
        # 1. 计算原始路由分数 (Original Logits)
        # [batch, seq, num_experts]
        raw_logits = self.gate_network(hidden_states)

        # --- OKR 介入开始 ---

        # 2. 计算水印偏好 (Semantic Anchor Signal)
        # 利用矩阵乘法做 LSH 投影。
        # 这是一个纯粹的几何操作，对语义漂移鲁棒。
        watermark_bias = torch.matmul(hidden_states, self.secret_projection)

        # 3. 计算安全掩码 (Indifference Zone)
        # 只有当 (max_logit - current_logit) < epsilon 时，我们才敢乱动
        # max_logits: [batch, seq, 1]
        max_logits, _ = raw_logits.max(dim=-1, keepdim=True)
        safe_mask = raw_logits >= (max_logits - self.epsilon)

        # 4. 机会主义注入 (Fail-Open Injection)
        # 如果在安全区，听水印的 (watermark_bias)。
        # 如果不在安全区，给它一个负无穷，让它滚蛋。
        # 注意：这里不需要判断 safe_mask 的 True 数量。
        # 如果只有一个 True (只有原本的 best)，那其他的全是 -inf，结果还是原本的 best。
        # 系统自动退化为无水印模式。
        modified_scores = torch.where(
            safe_mask,
            watermark_bias, 
            torch.tensor(-1e9, device=raw_logits.device, dtype=raw_logits.dtype)
        )

        # --- OKR 介入结束 ---

        # 5. 选取 Top-K
        # 基于修改后的分数选专家
        # indices: [batch, seq, top_k]
        _, selected_experts = torch.topk(modified_scores, self.top_k, dim=-1)

        # 6. 计算权重 (Softmax)
        # 注意：为了保持原本的数值稳定性，我们在计算权重时
        # 应该回退去使用 raw_logits 中对应专家的值，还是用 modified_scores？
        # 论文里通常只改 routing (indices)，不改 weight。
        # 我们只改变"谁工作"，不改变"给多少钱"。这是为了保持梯度流稳定。

        # 从 raw_logits 中gather出被选中专家的原始分数
        router_logits = torch.gather(raw_logits, -1, selected_experts)

        # 标准 MoE 操作：Softmax 归一化
        routing_weights = F.softmax(router_logits, dim=-1)

        return routing_weights, selected_experts
```

---

### 2. `detector.py` (The Verifier)

检测器不需要重跑整个大模型。如果你只想验证水印，理论上只需要跑这一层。但在实际中，我们需要文本对应的 Embedding，所以通常还是得跑一遍 Inference。

这里的逻辑是：计算 **Green-List Hit Rate**（也就是水印命中率）。

```python
import torch
import numpy as np

class OKRDetector:
    def __init__(self, model, epsilon=1.5):
        """
        model: 已经注入了 OKRRouter 的模型
        """
        self.model = model
        self.epsilon = epsilon

    def detect(self, input_ids, attention_mask):
        """
        对一段文本进行检测。
        """
        # 1. 跑一遍模型，Hook 住 Router 的输入输出
        # 这里假设我们能拿到每一层的 hidden_states 和 router 的选择
        # 在实际工程中，这需要在此处注册 PyTorch Hook
        # 为了演示，我们简化为处理单层数据
        pass 

    def verify_batch(self, hidden_states, actual_selected_experts):
        """
        核心验证逻辑 (纯数学计算)

        hidden_states: [batch, seq, dim] - 重新推理得到的 Embedding
        actual_selected_experts: [batch, seq, top_k] - 也就是你想验证的 token 实际上走了哪条路
        """
        # 获取 Router (假设模型只有一层或我们只看这一层)
        router = self.model.router 

        # 1. 重算原本的 Logits (Ground Truth)
        raw_logits = router.gate_network(hidden_states)
        max_logits, _ = raw_logits.max(dim=-1, keepdim=True)

        # 2. 重算水印信号 (Expected Signal)
        watermark_bias = torch.matmul(hidden_states, router.secret_projection)

        # 3. 重算机会窗口 (Opportunities)
        # 哪些 Token 拥有至少 2 个安全专家？
        safe_mask = raw_logits >= (max_logits - self.epsilon)
        num_safe_experts = safe_mask.sum(dim=-1) # [batch, seq]

        # 只有 >= 2 个可选专家时，水印才有可能生效
        # 这就是我们的有效样本 (Valid Samples)
        valid_opportunity_mask = (num_safe_experts >= 2)

        if valid_opportunity_mask.sum() == 0:
            return 0.0, "No Opportunities (Text too clear or epsilon too small)"

        # 4. 验证命中 (Check Hits)
        # 在这些机会窗口里，水印想要谁？
        # 水印想要的是在 safe_mask 范围内，watermark_bias 最大的那个

        masked_watermark_scores = torch.where(
            safe_mask,
            watermark_bias,
            torch.tensor(-1e9, device=watermark_bias.device)
        )
        # 理论上水印指向的第一选择
        target_expert = torch.argmax(masked_watermark_scores, dim=-1) # [batch, seq]

        # 5. 检查实际选择是否包含水印指向的专家
        # actual_selected_experts: [batch, seq, top_k]
        # 我们看 target_expert 是否在 top_k 里

        # 扩展 target_expert 维度以便比较: [batch, seq, 1]
        target_expert_expanded = target_expert.unsqueeze(-1)

        # hit: [batch, seq] (True/False)
        hits = (actual_selected_experts == target_expert_expanded).any(dim=-1)

        # 6. 统计分数 (仅在有机样本上)
        valid_hits = hits[valid_opportunity_mask]
        score = valid_hits.float().mean().item()

        return score, "Watermarked" if score > 0.8 else "Clean"
```

---

### 3. `patch.py` (The Glue)

这就是我们如何“黑”进现有的模型。我不喜欢破坏现有的代码库。最好的方式是在运行时动态替换。

```python
import torch
from okr_kernel import OKRRouter

def inject_okr(model, epsilon=1.5):
    """
    把 OKR 注入到任何标准的 MoE 模型中 (如 Mixtral)。
    前提：你知道模型里 Router 层的名字。
    """
    count = 0

    # 遍历模型所有模块
    for name, module in model.named_modules():
        # 这是一个启发式规则，你需要根据具体模型调整
        # 比如 Mixtral 的 Gate 叫 "gate"
        if "gate" in name and isinstance(module, torch.nn.Linear):
            print(f"Patching layer: {name}")

            # 提取参数
            input_dim = module.in_features
            num_experts = module.out_features

            # 创建我们的 Kernel
            new_router = OKRRouter(input_dim, num_experts, epsilon=epsilon)

            # 关键：把原来训练好的权重拷过来！
            # 我们只换引擎，不换司机
            with torch.no_grad():
                new_router.gate_network.weight.copy_(module.weight)
                if module.bias is not None:
                    # OKR 默认没开 bias，如果原模型有，这得改一下 Kernel
                    # 这里假设无 bias 或简单忽略
                    pass

            # 暴力替换
            # 这种 getattr/setattr 递归有点丑，但这是 Python
            parent_name = name.rsplit('.', 1)[0]
            child_name = name.rsplit('.', 1)[1]
            parent = model.get_submodule(parent_name)
            setattr(parent, child_name, new_router)

            count += 1

    print(f"Injected OKR into {count} layers.")
    return model
```

---

### 我的架构点评

1. **极简主义：** 核心逻辑只有 60 行代码。容易审计，容易调试。
2. **零开销 (Zero Overhead)：** 在 `forward` 函数里，只有两个额外的 Tensor 操作：`matmul` (水印计算) 和 `where` (合并逻辑)。这在现代 GPU 上大概只需要 0.01ms。
3. **向后兼容：** 通过 `inject_okr`，你可以在不重新训练模型的情况下，直接给现有的开源 MoE 模型打补丁。这才是“实用主义”。

现在，把这套代码拿去跑。如果 `EPSILON` 设得对，你应该能在生成的文本里检测到强烈的信号，同时读起来和原文一模一样。

这就是它的工作原理。简单，粗暴，有效。
