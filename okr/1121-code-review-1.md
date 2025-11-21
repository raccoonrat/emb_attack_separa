# **Linux Torvalds Code Review: Opportunistic Keyed Routing (OKR) v2.1**

Date: 2025-11-21  
Reviewer: Linus Torvalds  
Topic: Analysis of MoE Watermarking Implementation  
听着，我不在乎你们的论文里用了多少个 $\\epsilon$ 或 $\\gamma$ 来证明它是“虽然次线性衰减但依然鲁棒”的。我只在乎这堆代码跑在数千个 GPU 上时，会不会因为浮点数舍入误差导致模型输出垃圾，或者因为锁竞争把吞吐量拖慢 50%。

你们的 V2.1 方案用 LSH（局部敏感哈希）替代 HMAC 是正确的方向。HMAC 在这种场景下就是错误的工具，把它删掉是对的。

但是，目前的实现里藏着几个足以炸毁“用户空间”（User Space，即模型推理质量）的地雷。

## **1\. "Magic Number" 的诅咒：Epsilon 是个烂主意**

我在代码里看到了这个：

self.epsilon \= epsilon  \# e.g., 1.5  
\# ...  
safe\_mask \= raw\_logits \>= (max\_logits \- self.epsilon)

**这是糟糕的品味。**

你在假设 logits 的数值分布是恒定的。大错特错。

* 如果用户对模型进行了 **Quantization (量化)**，比如从 FP16 转到 INT8，logits 的数值范围会剧烈变化。  
* 如果模型架构改变，LayerNorm 的实现改变，logits 的方差也会变。  
* 不同的 Prompt 可能会导致 logits 的“尖锐度”（Kurtosis）完全不同。

后果：  
固定 epsilon \= 1.5。在某些情况下，这涵盖了所有专家（水印太强，破坏质量）；在另一些情况下，它谁也涵盖不了（水印失效）。  
我的建议：  
抛弃绝对值阈值。使用自适应阈值。  
比如，计算 Top-K logit 的标准差，或者使用 softmax 后的概率比率。  
让 safe\_mask 基于 probability \> max\_prob \* 0.9，而不是 logit \> max\_logit \- 1.5。让数学去适应数据，而不是让数据适应你拍脑袋想出来的数字。

## **2\. LSH 的零点危机 (The Zero-Crossing Problem)**

你们的 LSH 逻辑很简洁：

watermark\_bias \= torch.matmul(inputs, self.secret\_projection)  
\# 隐含逻辑：利用正负号或大小来决定偏好

这在数学上很美，但在工程上很危险。  
当 inputs 和 secret\_projection 正交时（点积接近 0），结果由什么决定？  
由浮点数噪声决定。  
如果 dot\_product 是 0.000001 和 \-0.000001，在语义上它们没有区别，但在你的逻辑里，它们可能指向完全相反的专家。  
对于一个抗干扰系统来说，这引入了不必要的“高频噪声”。  
我的建议：  
引入一个死区 (Dead Zone)。  
如果 abs(dot\_product) \< threshold，不要让水印介入。这时候的信号是不可靠的，强行使用不可靠的信号去干扰路由，是在破坏用户体验。  
这也符合“实用主义”——如果不确定，就什么都别做。

## **3\. 分布式推理的噩梦：初始化陷阱**

我看了一眼初始化代码：

\# 初始化阶段  
self.register\_buffer("secret\_projection", torch.randn(input\_dim, num\_experts))

如果这段代码跑在分布式的 8 张卡上，或者跨节点的推理集群上，你如何保证每张卡上的 torch.randn 生成了一样的随机矩阵？

如果你不显式地设置 Seed，或者在主节点生成后广播给所有 worker， 每一张卡都会有不同的“私钥”。  
结果就是：Token 1 在 GPU 0 上被打上了水印 A，Token 2 在 GPU 1 上被打上了水印 B。  
检测器会看到一团乱麻。这是典型的“并发 Bug”，只不过这里是“一致性 Bug”。  
修复它：  
要么硬编码 Seed（不推荐），要么在加载模型权重时强制校验 secret\_projection 的一致性。不要假设 \_\_init\_\_ 会在所有进程中神奇地同步。

## **4\. 性能与内存访问：torch.where 的隐性代价**

final\_scores \= torch.where(safe\_mask, watermark\_bias, \-1e9)

这行代码看起来很 Pythonic，很短。但在 GPU 层面，它产生了一个全新的 Tensor，并且产生了一次显存读写。  
在 LLM 推理中，Memory Bandwidth (显存带宽) 是瓶颈，不是计算。  
虽然这比 if/else 好，但还不够好。  
如果你能把这个融合到 Softmax kernel 里（我知道这很难，但我们是搞内核的，不是写脚本的），或者使用 masked\_fill\_ 做原地操作（In-place operation），可能会挤出那宝贵的 1-2% TPS。  
**更好的写法（In-place）：**

\# 预先 clone 是为了不破坏原始 logits 用于其他用途，如果 raw\_logits 不再使用，甚至可以原地改  
\# 但为了安全起见：  
final\_scores \= raw\_logits.clone()   
\# 这里的逻辑需要微调，原本是替换，现在是 Masking。  
\# V2.1 的逻辑是：安全区内用水印分，安全区外由原始分决定（实际上是 \-inf 淘汰）  
\# 你的实现直接把安全区外设为 \-1e9，这实际上丢弃了原始 logits 的信息。  
\# 如果水印导致所有安全区的专家都被选完了（假设有 capacity limit），你怎么办？

等一下，我看出了逻辑上的一个 Bug。  
你的代码：  
final\_scores \= torch.where(safe\_mask, watermark\_bias, \-1e9)

你丢弃了原始 Logits 的相对大小信息！  
在 safe\_mask 内部，你完全用 watermark\_bias 替代了 raw\_logits。  
这意味着，如果在安全区内，专家 A 的原始分是 10.0，专家 B 是 9.9（非常接近），但水印偏差让 B 变成了 1.0，A 变成了 \-1.0。你选了 B。这没问题，这是设计的初衷。  
但如果水印偏差非常微小，或者 watermark\_bias 的量级（Scale）和 raw\_logits 不一致怎么办？  
watermark\_bias 是 dot(embedding, random\_vec)。它的数值范围取决于 Embedding 的 Norm。  
而 raw\_logits 的数值范围取决于模型的输出。  
你正在比较苹果和橘子。 如果 Embedding Norm 很大，水印分数会彻底压倒一切；如果很小，它可能毫无作用。  
**必须做 Normalization。** 把 watermark\_bias 归一化到和 logits 相似的量级，或者只用它做 argsort 的依据，而不是直接作为分数。

## **5\. 总结：给团队的指令**

这份 V2.1 代码有“好品味”的影子，但在工程落地层面太幼稚了。它假设了一个理想的、单机的、数值稳定的环境。现实不是这样的。

**Action Items:**

1. **重写阈值逻辑：** 删掉 self.epsilon。实现基于概率比率的动态掩码 (p\_top2 / p\_top1 \> 0.9 意味着犹豫，可以介入)。  
2. **解决量级不匹配：** 不要直接用 watermark\_bias 替换分数。在安全区内，应该由 raw\_logits \+ alpha \* watermark\_bias 决定，或者在安全区内仅使用水印进行**重新排序 (Re-ranking)**，而不是数值替换。目前的替换逻辑太粗暴了，容易破坏 torch.argmax 的梯度（如果在训练时用的话），虽然我们现在只谈推理。  
3. **确定性初始化：** 必须确保 secret\_projection 在所有节点一致。把它存进 state\_dict 是不够的，要在代码里加 Assert 检查。  
4. **死区逻辑：** 在 LSH 投影接近 0 时，返回一个 "Neutral" 信号，不要强行施加水印。

把这些修好，然后我们再谈合并代码的事。

Linus