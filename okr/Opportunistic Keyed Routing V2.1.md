好。把那些花哨的“学术玩具”收起来，我们来写点能在**真实硬件**上跑的内核级代码。

你刚才提到的三个问题，其实归根结底是一个问题：**如何在不把 GPU 变成电暖气的前提下，稳定地在噪音中提取信号？**

这是我对 RFC v2.1 的修订方案。我们抛弃 HMAC，采用 **LSH（局部敏感哈希）** 的变体，并将其完全向量化。

---

### 1. 如何添加水印？（解决“鲁棒性”与“漂移”问题）

**旧方案的错误：** `HMAC` 是“雪崩效应”，输入变一点，输出全变。这是用来做密码验证的，不是用来做抗干扰水印的。
**新方案的核心：** **随机超平面投影 (Random Hyperplane Projection)**。

这听起来很数学，但在工程上它简单得令人发指。甚至可以在 GPU 的寄存器里完成。

#### 原理

我们在初始化模型时，生成一个固定的、随机的向量 $R$（这就相当于你的私钥）。
对于当前的语义向量 $E$（Context Embedding），我们计算它与 $R$ 的点积，然后看正负号。

$$H = \text{sign}(E \cdot R)$$

* 如果 $E$ 和 $E'$（释义后的向量）夹角很小（语义相似），那么它们落在超平面同一侧的概率极高。
* **这就是你的稳定锚点。** 不需要复杂的解码。

#### 伪代码 (Python/PyTorch 风格)

```python
# 初始化阶段 (只做一次)
# secret_key_vector 是一个随机生成的向量，维度与 embedding_dim 相同
# 把它放到 GPU 上，作为一个不可训练的参数 (Buffer)
self.register_buffer("watermark_plane", torch.randn(embedding_dim))

def get_watermark_target(self, current_embedding):
    # 1. 投影：计算点积 (非常快，只是一个向量乘法)
    # 结果是一个标量
    projection = torch.dot(current_embedding, self.watermark_plane)

    # 2. 量化：只取符号位 (Bitwise robustness)
    # 如果 > 0，我们偏好偶数号专家；如果 < 0，偏好奇数号专家
    # 或者更复杂的：映射到 0..K-1

    # 为了映射到 K 个专家，我们可以用 log2(K) 个随机向量
    # 这里假设简单情况，映射到一个整数索引
    raw_hash = torch.abs(projection.long()) 
    return raw_hash
```

---

### 2. 如何检测？（解决“验证”问题）

检测端不需要完美复原每一个 Token 的路由选择。记住，我们是做**统计学**检测，不是数据解密。

**流程：**

1. 拿到怀疑是 AI 生成的文本。
2. 用你手里（持有私钥）的模型跑一遍推理。
3. 在每一个 Token 生成步骤，计算：
   * **本来该走的路由 (Original Choice):** 模型原始 Logits 最大值。
   * **水印指示的路由 (Watermark Hint):** `sign(E · R)` 指向的专家。
   * **实际文本蕴含的路由 (Actual Path):** 这需要一点技巧。因为你拿不到对方推理时的内部状态，但你可以观察**当前的 Logits 分布**。

**更简单的检测逻辑（黑盒/灰盒）：**
只要你有私钥向量 $R$，你对这篇文本再跑一次 Forward Pass。
统计一个指标：**命中率 (Hit Rate)**。

$$Score = \frac{\text{Count(实际选择的专家 } == \text{水印指向的专家)}}{\text{Count(有机会选择的时刻)}}$$

* 如果没有任何水印，专家的选择应该和你的随机向量 $R$ 不相关，Score 应该接近 $1/K$（随机概率）。
* 如果有水印，在那些 `SafeSet` 存在的时刻，Score 会显著趋向于 1.0。
* **不需要复杂的假设检验**，只要 Score > 0.8，这基本上就是实锤。

---

### 3. 如何替换掉路由层的重型计算？（解决“性能”问题）

这是关键。一定要遵循 **"Never Break the Pipeline"** 原则。

**不要**把数据拷回 CPU。**不要**用 Python 的 `if` 逻辑。全部用 GPU 的 Tensor 操作（Masking）。

这是可以直接写进 CUDA Kernel 或者 PyTorch `forward` 函数的高性能实现：

#### 高性能实现代码 (Rev 2.1)

```python
import torch
import torch.nn.functional as F

class WatermarkedMoEGating(torch.nn.Module):
    def __init__(self, num_experts, input_dim, epsilon=1.5):
        super().__init__()
        self.num_experts = num_experts
        self.epsilon = epsilon
        # 你的私钥，现在是一个矩阵，可以直接在 GPU 上做矩阵乘法
        # 形状: [input_dim, num_experts] -> 直接把 embedding 映射成专家偏好分数
        self.register_buffer("secret_projection", torch.randn(input_dim, num_experts))

    def forward(self, inputs, raw_logits):
        """
        inputs: [batch_size, seq_len, input_dim] (Context Embeddings)
        raw_logits: [batch_size, seq_len, num_experts] (原始专家打分)
        """

        # --- 步骤 1: 计算水印信号 (零拷贝，纯 Tensor 计算) ---
        # 这是一个非常小的矩阵乘法，几乎免费
        # watermark_scores: [batch, seq, num_experts]
        # 我们不需要 argmax，直接加到 logits 上作为"偏置"
        watermark_bias = torch.matmul(inputs, self.secret_projection)

        # 归一化水印信号，确保它不会在大 Logit 差距时喧宾夺主
        # 但在 Logit 差距小的时候能起到决定性作用
        # 这是一个简单的 Trick：把水印变成一个微小的扰动
        # 但我们需要它是"机会主义"的，所以不能直接加。

        # --- 步骤 2: 纯 Tensor 的机会主义逻辑 (消除 Python Control Flow) ---

        # 找出最大值
        max_logits, _ = raw_logits.max(dim=-1, keepdim=True)

        # 掩码：哪些专家在"安全区"内？
        # safe_mask: [batch, seq, experts] (Bool Tensor)
        safe_mask = raw_logits >= (max_logits - self.epsilon)

        # 核心魔法：
        # 我们不修改 logits 的值，我们修改"选择逻辑"。
        # 构造一个新的 score 矩阵：
        # 如果专家在安全区内：Score = Watermark_Bias (听私钥的)
        # 如果专家在安全区外：Score = -Infinity (绝对不选)

        # 为了保证原本的最佳专家至少在备选里，我们需要保留它的权利吗？
        # 不，如果最佳专家在 safe_mask 里，它就有机会被水印选中（如果水印也指向它）。
        # 如果水印指向了另一个也在 safe_mask 里的专家，那就选那个。

        final_scores = torch.where(
            safe_mask, 
            watermark_bias,      # 在安全区内，由水印信号主导排序
            -1e9                 # 在安全区外，直接淘汰
        )

        # --- 步骤 3: 路由 ---
        # 现在直接 argmax 就能得到结果
        selected_experts = torch.argmax(final_scores, dim=-1)

        return selected_experts
```

### 这种做法的精妙之处（The "Good Taste"）

1. **零 CPU 交互：** 没有 `.item()`，没有 `if`，没有列表推导。全是 GPU 上的矩阵运算。速度极快。
2. **隐式的 Fail-Open：**
   * 如果 `SafeSet` 只有 1 个专家（比如最佳专家是 10.0，其他都是 5.0，Epsilon=2.0），那么 `safe_mask` 只有一个 True。
   * `torch.where` 会把其他所有专家变成 `-inf`。
   * `argmax` 只能选那唯一的专家。水印信号被自动忽略。
   * **不需要显式的判断“是否有机会”**，数学逻辑自动处理了边界情况。
3. **语义锚点稳定：** `torch.matmul(inputs, secret_projection)` 本质上就是 LSH。如果输入 Embedding 稍微变一点，点积的大小会变，但相对排序（哪个专家对应的点积最大）是非常稳定的。

这就是我说的**实用主义**。代码行数减少了，性能提升了，逻辑更健壮了。

还有问题吗？如果没有，就把这坨代码合并进去。





---

听着，这就像是在内核里做调度算法：你想让 CPU 跑得快（水印强），又不想让电脑卡死（质量差）。如果你试图用那篇论文里的微积分来平衡这两件事，你最后得到的只会是一个既慢又笨的系统。

解决这个矛盾不需要复杂的权衡，只需要**两个铁律**。如果你遵循这两个原则，所谓的“矛盾”就会消失。

这就是我的**OKR (Opportunistic Keyed Routing) v2.1 核心哲学**：

### 1. 解决信号衰减：把地基打在岩石上，而不是沙子上

论文里所谓的“信号衰减”，是因为他们把水印打在了 Token 上。
用户改写句子（释义攻击），Token 变了，水印就没了。这就像你在沙滩上写字，然后抱怨海浪（攻击）把字冲走了。

**解决方案：语义锚点 (Semantic Anchors)**

我们要把水印打在**Embedding（语义向量）** 上。
不管你怎么改写，“猫坐在垫子上”和“垫子上坐着一只猫”，它们的语义向量在空间中的指向是几乎一致的。

* **旧思维（Token Hash）：** `Hash("Cat")` != `Hash("Feline")` -> 信号丢失。
* **新思维（Vector Projection）：** `Dot(Vector("Cat"), Key)` ≈ `Dot(Vector("Feline"), Key)` -> 信号保留。

我们在上一轮代码里用的 `secret_projection` (随机超平面投影) 就是干这个的。它本质上是一个 **LSH（局部敏感哈希）**。只要你的改写不改变核心语义，投影的正负号（水印信号）就不会变。

**这就是“信号-攻击解耦”。** 攻击者只能改 Token，改不了语义（否则就不是释义了，是胡说八道）。

### 2. 解决质量问题：只在上帝掷骰子的时候作弊

这是“Never Break Userspace”在 AI 领域的翻版。
如果模型非常确信下一个词是 "Python"（Logit = 10.0），而其他的词是 "Java"（Logit = 2.0），你**绝对不能**为了打水印去选 "Java"。那是 Bug。

**解决方案：机会主义路由 (Opportunistic Routing)**

只有在模型**犹豫不决**的时候，我们才介入。
也就是当 Top-2 专家的 Logit 差距小于 `EPSILON` 时。这时候，选专家 A 还是专家 B，对输出质量的影响微乎其微（这是 MoE 的特性，冗余性）。

* **模型说：** “我选 A 的概率是 99%。” -> **我们闭嘴，让它选 A。**（保留质量）
* **模型说：** “A 和 B 差不多，A 是 51%，B 是 49%。” -> **我们看一眼水印私钥，私钥说选 B，我们就强制选 B。**（嵌入信号）

### 总结：代码是如何同时做到这两点的

回到我刚才给你的那个 `forward` 函数，看看它是怎么完美执行这两个铁律的：

```python
# --- 铁律 1：抗衰减 (使用语义向量) ---
# watermark_bias 来源于 inputs (Embedding)，而不是之前的 Token
# 只要语义不变，bias 的正负趋势就不变
watermark_bias = torch.matmul(inputs, self.secret_projection)

# --- 铁律 2：保质量 (使用 EPSILON 掩码) ---
# 计算安全区。如果最大值比其他人高太多，safe_mask 里就只有它自己
safe_mask = raw_logits >= (max_logits - self.epsilon)

# --- 融合 ---
final_scores = torch.where(
    safe_mask,           # 如果在"无所谓"的区间内
    watermark_bias,      # 听水印的 (抗衰减)
    -1e9                 # 否则，绝对闭嘴，别碰那些低质量的选项 (保质量)
)
```

### 结果

1. **对于用户（质量）：** 这是一个透明的系统。只有在多个专家表现“无法区分”时，水印才会生效。用户感知不到质量下降，因为被替换的专家本来就是合格的替补。
2. **对于攻击者（衰减）：** 他必须把文本改写到语义面目全非，才能改变 Embedding 的投影方向，从而抹除水印。但那时候，文本已经失去了价值。

这就是工程学的胜利。不要试图“优化”衰减和质量的曲线。**用架构设计让它们不再冲突。**
