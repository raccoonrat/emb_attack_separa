MoE 可证鲁棒水印方案 (Provably Robust MoE Watermark)
============================================

本项目基于《Signal-Attack Decoupling in MoE Watermarks》及其工程方案 (`align_to_proofs.md`)，提供了一个基于 MoE Gating 机制的可证鲁棒水印方案的 Python 实现。
核心特性
----

* **信号-攻击解耦**: 水印嵌入在专家激活模式（$\mathcal{S}_B = \{0,1\}^K$），而攻击发生在输入文本空间（$\mathcal{X}$），实现 $\mathcal{S}_B \cap \mathcal{X} = \emptyset$。

* **可证鲁棒性**: 方案的鲁棒性衰减满足 $O(\sqrt{\gamma})$，优于传统方案的 $O(\gamma)$。

* **最优检测**: 采用基于 Neyman-Pearson 引理的 LLR（对数似然比）检测器。

* **参数化安全**: 引入“安全系数 $c$”，将水印强度 $\epsilon$ 与设计的攻击强度 $\gamma$ 关联（$\epsilon = c^2\gamma$），实现鲁棒性与性能的可控权衡。

项目结构
----

    .
    ├── main.py             # 主程序入口 (运行标定、嵌入、检测)
    ├── moe_watermark.py    # MoE Gating 水印核心实现 (模型 Patch)
    ├── calibration.py      # 参数标定 (Lg, C, c*)
    ├── detector.py         # LLR 水印检测器
    ├── attacks.py          # 攻击模拟 (Paraphrase) 与 γ 估算
    ├── requirements.txt    # Python 依赖库
    └── README.md           # 本文档

安装
--

1. 克隆本项目

2. 安装依赖:
      pip install -r requirements.txt

3. （可选）根据您的 `torch` 版本和 CUDA 配置，确保 `torch` 已正确安装。

使用说明
----

本项目通过 `main.py` 提供了三种操作模式。

### 1. 模式一: `calibrate` (参数标定)

此模式用于运行 `calibration.py` 中的算法，标定系统的核心常数 $C$ 和最优安全系数 $c^*$。
    python main.py --mode calibrate \
                   --model_name "mistralai/Mixtral-8x7B-v0.1" \
                   --dataset_name "wikitext" \
                   --dataset_split "train" \
                   --num_calib_samples 1000

* 此过程计算密集，将输出标定结果（$L_g$, $C_{prop}$, $C_{stability}$, $C$, $c^*$），并建议将其保存到配置文件中（未来扩展）。

### 2. 模式二: `embed` (水印嵌入与生成)

使用标定好的参数（或默认值）来嵌入水印并生成文本。
    python main.py --mode embed \
                   --model_name "mistralai/Mixtral-8x7B-v0.1" \
                   --prompt "Once upon a time, in a land far, far away," \
                   --secret_key "my_secret_key_123"

* 程序将加载模型，应用水印 patch，然后生成水马文。

### 3. 模式三: `detect` (水印检测)

检测给定文本是否包含水印。
    # 待检测的文本
    TEXT_TO_CHECK="... (此处为待检测的文本) ..."

    python main.py --mode detect \
                   --model_name "mistralai/Mixtral-8x7B-v0.1" \
                   --text_to_check "$TEXT_TO_CHECK" \
                   --secret_key "my_secret_key_123" \
                   --attack "paraphrase" # 可选：在检测前应用攻击

* `--attack "paraphrase"`: 模拟对手在检测前进行释义攻击，以测试鲁棒性。

* 程序将输出 LLR 统计量以及检测结果（Detected / Not Detected）。
