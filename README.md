# MoE 水印论文与实现项目

本项目包含关于 MoE（混合专家模型）专家激活水印对抗释义攻击的理论证明论文的 LaTeX 源文件，以及相应的 Python 实现代码。

## 📋 目录

- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [使用手册](#使用手册)
- [环境配置](#环境配置)
- [理论文档](#理论文档)
- [常见问题](#常见问题)

## 📁 项目结构

```
.
├── README.md                    # 项目说明文件（本文件）
├── *.tex                        # LaTeX 源文件（论文主文件）
├── styles/                      # LaTeX 样式文件目录
│   └── usenix2020_SOUPS.sty    # USENIX SOUPS 2020 会议模板样式
├── build/                       # 编译输出目录
└── experiment/                  # Python 实现代码
    ├── main.py                 # 主程序入口
    ├── detector.py             # 水印检测器
    ├── mves_watermark_corrected.py  # 水印嵌入实现
    ├── requirements.txt         # Python 依赖
    └── *.md                    # 技术文档
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 进入实验目录
cd experiment

# 创建 Conda 环境（推荐）
conda env create -f environment.yml
conda activate emb_attack_separa

# 或使用 pip
pip install -r requirements.txt
```

### 2. 基本使用

#### 嵌入水印

```bash
python main.py --mode embed \
    --model_name google/switch-base-8 \
    --prompt "Your text here" \
    --secret_key "my_secret_key_123"
```

#### 检测水印

```bash
python main.py --mode detect \
    --model_name google/switch-base-8 \
    --text_to_check "生成的完整文本" \
    --secret_key "my_secret_key_123"
```

**⚠️ 重要**：检测时必须使用与嵌入时**相同的 `secret_key`**！

## 📖 使用手册

### 模式说明

项目支持以下四种模式：

#### 1. `embed` - 嵌入水印

在文本生成过程中嵌入水印。

**必需参数**：
- `--model_name`: 模型名称（如 `google/switch-base-8`）
- `--prompt`: 输入提示文本
- `--secret_key`: 水印密钥（用于生成确定性种子）

**可选参数**：
- `--c_star`: 安全系数 c*（默认 2.0）
- `--gamma_design`: 设计攻击强度 γ（默认 0.03）

**示例**：
```bash
python main.py --mode embed \
    --model_name google/switch-base-8 \
    --prompt "The quick brown fox" \
    --secret_key "my_key_123"
```

**输出**：生成的带水印文本

#### 2. `detect` - 检测水印

检测文本是否包含水印。

**必需参数**：
- `--model_name`: 模型名称
- `--text_to_check`: 待检测的文本
- `--secret_key`: 水印密钥（**必须与嵌入时相同**）

**可选参数**：
- `--c_star`: 安全系数（默认 2.0，应与嵌入时相同）
- `--gamma_design`: 设计攻击强度（默认 0.03，应与嵌入时相同）
- `--tau_alpha`: LLR 检测阈值（默认 5.0，建议通过标定获得）

**示例**：
```bash
python main.py --mode detect \
    --model_name google/switch-base-8 \
    --text_to_check "生成的完整文本" \
    --secret_key "my_key_123" \
    --tau_alpha 8.0
```

**输出**：
```
--- Detection Result ---
Result: Watermark DETECTED (Score: 25.34)
------------------------
```

#### 3. `calibrate` - 参数标定

标定水印系统的参数（Lg、C、c*）。

**参数**：
- `--model_name`: 模型名称
- `--dataset_name`: 数据集名称（如 `wikitext`）
- `--num_calib_samples`: 标定样本数量（默认 100）

**示例**：
```bash
python main.py --mode calibrate \
    --model_name google/switch-base-8 \
    --dataset_name wikitext \
    --num_calib_samples 100
```

#### 4. `experiment` - 完整实验

运行完整的实验流程。

**参数**：与 `calibrate` 模式类似

### 完整工作流程示例

```bash
# 步骤 1: 嵌入水印
python main.py --mode embed \
    --model_name google/switch-base-8 \
    --prompt "The quick brown fox jumps over the lazy dog" \
    --secret_key "my_key_123"

# 输出示例：
# --- Watermarked Output ---
# The quick brown fox jumps over the lazy dog. It is a beautiful day.
# --------------------------

# 步骤 2: 检测水印（使用生成的文本和相同的 secret_key）
python main.py --mode detect \
    --model_name google/switch-base-8 \
    --text_to_check "The quick brown fox jumps over the lazy dog. It is a beautiful day." \
    --secret_key "my_key_123"

# 输出示例：
# --- Detection Result ---
# Result: Watermark DETECTED (Score: 25.34)
# ------------------------
```

### 参数说明

#### 核心参数

| 参数 | 说明 | 默认值 | 必需 |
|------|------|--------|------|
| `--mode` | 运行模式 | - | ✅ |
| `--model_name` | 模型名称 | - | ✅ |
| `--secret_key` | 水印密钥 | `DEFAULT_SECRET_KEY` | ⚠️ 检测时必需 |
| `--prompt` | 输入提示（embed模式） | - | embed模式必需 |
| `--text_to_check` | 待检测文本（detect模式） | - | detect模式必需 |

#### 水印参数

| 参数 | 说明 | 默认值 | 说明 |
|------|------|--------|------|
| `--c_star` | 安全系数 c* | 2.0 | 影响水印强度 ε = c*² × γ |
| `--gamma_design` | 设计攻击强度 γ | 0.03 | 影响水印强度 |
| `--tau_alpha` | LLR 检测阈值 | 5.0 | 应通过H0假设下的实验标定 |

**注意**：检测时的 `c_star` 和 `gamma_design` 应该与嵌入时相同。

## 🔧 环境配置

### 系统要求

- Python 3.10+
- CUDA（可选，用于GPU加速）
- 至少 8GB RAM（推荐 16GB+）
- 至少 10GB 磁盘空间（用于模型缓存）

### 安装步骤

#### 1. Conda 环境（推荐）

```bash
cd experiment
conda env create -f environment.yml
conda activate emb_attack_separa
```

#### 2. Pip 安装

```bash
cd experiment
pip install -r requirements.txt
```

### WSL 环境

如果使用 WSL (Windows Subsystem for Linux)，项目会自动检测环境并使用相应的缓存配置。

**快速测试**：
```bash
python experiment/test_wsl_setup.py
```

详细说明请参考：[`experiment/WSL_TEST_GUIDE.md`](experiment/WSL_TEST_GUIDE.md)

### 缓存配置

项目会自动检测运行环境（Windows/WSL/Linux）并设置相应的缓存路径：

- **Windows**: `D:/Dev/cache/`
- **WSL/Linux**: `~/.cache/emb_attack_separa/`

## 📚 理论文档

### 论文编译

#### 推荐方法：使用 latexmk

```bash
# 编译中文版（使用 XeLaTeX）
latexmk -xelatex moe_paradigm_rigorous_proofs.tex

# 编译英文版（使用 pdfLaTeX）
latexmk -pdf moe_watermark_paraphrase_attack.tex
```

#### 手动编译

```bash
# 中文版
xelatex moe_paradigm_rigorous_proofs.tex
xelatex moe_paradigm_rigorous_proofs.tex  # 第二次编译以生成正确的引用

# 英文版
pdflatex moe_watermark_paraphrase_attack.tex
pdflatex moe_watermark_paraphrase_attack.tex
```

### 技术文档

- **阈值标定理论**: [`experiment/THRESHOLD_EXPLANATION.md`](experiment/THRESHOLD_EXPLANATION.md) - LLR 阈值 τ_α 的理论依据和标定方法
- **检测详细说明**: [`experiment/DETECTION_GUIDE.md`](experiment/DETECTION_GUIDE.md) - 水印检测的详细说明和常见问题

## ❓ 常见问题

### Q1: 为什么检测不到水印？

**可能原因**：
1. **`secret_key` 不匹配**（最常见）
   - 确保检测时使用与嵌入时相同的 `secret_key`
2. **检测的文本不正确**
   - 应使用嵌入时生成的**完整文本**，而不是原始提示
3. **阈值设置过高**
   - 尝试降低 `--tau_alpha` 值，或使用标定模式获得合适的阈值
4. **文本经过攻击**
   - 如果文本被改写或攻击，水印可能被破坏

### Q2: 如何选择合适的阈值？

**推荐方法**：使用标定模式

```bash
# 准备无水印样本，然后标定阈值
python main.py --mode calibrate \
    --model_name google/switch-base-8 \
    --num_calib_samples 100 \
    --secret_key "my_key_123"
```

**临时方法**：根据实际LLR分数调整

如果检测时LLR分数为 8.28，可以设置：
```bash
--tau_alpha 8.0  # 略低于LLR分数
```

详细理论说明请参考：[`experiment/THRESHOLD_EXPLANATION.md`](experiment/THRESHOLD_EXPLANATION.md)

### Q3: 检测时可以使用不同的参数吗？

**不建议**。检测时应该使用与嵌入时相同的参数：
- `--secret_key`: **必须相同**
- `--c_star`: 应该相同
- `--gamma_design`: 应该相同

只有 `--tau_alpha` 可以根据需要调整。

### Q4: WSL 环境下如何配置？

项目会自动检测WSL环境并使用Linux路径。详细说明请参考：
- [`experiment/WSL_TEST_GUIDE.md`](experiment/WSL_TEST_GUIDE.md)

### Q5: 模型下载失败怎么办？

1. 检查网络连接
2. 确认镜像源配置（项目默认使用 `https://hf-mirror.com`）
3. 手动设置环境变量：
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```

## 📝 注意事项

1. **`secret_key` 必须匹配**：这是最重要的！嵌入和检测必须使用相同的密钥
2. **检测完整文本**：使用嵌入时生成的完整文本，而不是原始提示
3. **参数一致性**：检测时的 `c_star` 和 `gamma_design` 应该与嵌入时相同
4. **阈值标定**：建议通过H0假设下的实验标定阈值，而不是随意设置
5. **模型版本**：确保使用相同的模型版本

## 🔬 论文文件

### 核心论文
- **`moe_paradigm_rigorous_proofs.tex`** - 范式之争的严格数学证明（中文版，USENIX SOUPS 格式）

### 其他版本
- `moe_paradigm_rigorous_proofs_soups.tex` - SOUPS 格式版本
- `moe_watermark_paraphrase_attack.tex` - 英文版论文
- `moe_watermark_paraphrase_attack_zh.tex` - 中文版论文

## 📄 依赖要求

### LaTeX 发行版
- TeX Live 2020 或更高版本（推荐）
- MiKTeX 2.9 或更高版本

### Python 依赖
见 `experiment/requirements.txt`

## 🤝 贡献

如有问题或建议，请提交 Issue 或 Pull Request。

## 📄 许可证

请参考项目许可证文件。
