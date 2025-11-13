# MoE水印论文LaTeX编译说明

## 文件说明

- `moe_watermark_paraphrase_attack.tex`: 英文版LaTeX论文文件
- `moe_watermark_paraphrase_attack_zh.tex`: 中文版LaTeX论文文件
- `usenix2020_SOUPS.sty`: USENIX SOUPS 2020模板样式文件

## 编译方法

### 英文版编译

#### 使用pdflatex编译

```bash
pdflatex moe_watermark_paraphrase_attack.tex
pdflatex moe_watermark_paraphrase_attack.tex  # 第二次编译以生成正确的引用
```

#### 使用latexmk（推荐）

```bash
latexmk -pdf moe_watermark_paraphrase_attack.tex
```

### 中文版编译

中文版需要使用XeLaTeX或LuaLaTeX编译，因为需要支持中文字体。

#### 使用XeLaTeX编译（推荐）

```bash
xelatex moe_watermark_paraphrase_attack_zh.tex
xelatex moe_watermark_paraphrase_attack_zh.tex  # 第二次编译以生成正确的引用
```

#### 使用latexmk

```bash
latexmk -xelatex moe_watermark_paraphrase_attack_zh.tex
```

#### 使用LuaLaTeX

```bash
lualatex moe_watermark_paraphrase_attack_zh.tex
lualatex moe_watermark_paraphrase_attack_zh.tex
```

**注意**：中文版使用了`ctexart`文档类，会自动处理中文字体。如果需要指定特定字体，可以取消注释文件中的字体设置行。

## 论文特点

1. **符合USENIX SOUPS 2020格式**：使用官方模板样式
2. **理论完备**：包含完整的信息论推导和数学证明框架
3. **实验标记**：实验部分已标记为"待检验"（Experimental validation pending）
4. **证明标记**：定理证明已标记为"待后续补充"（To be supplemented in future work）
5. **跨学科视角**：以跨学科研究团队身份撰写，逻辑自洽

## 主要章节

1. **Introduction**: 水印的必要性与对抗脆弱性困境
2. **Paradigm A**: Kirchenbauer水印的机理分析（信号-攻击重合）
3. **Paradigm B**: MoE水印的信息论基础（信号-攻击解耦）
4. **Core Derivation**: 次线性衰减边界的理论证明
5. **Robustness Engineering**: 从理论到实践的框架
6. **Conclusion**: 范式转变的机理与意义

## 注意事项

- 参考文献中的部分条目标记为"To be cited"，需要补充完整引用信息
- 实验验证部分已明确标记为待检验
- 定理5.1的详细证明待后续补充

## 依赖包

论文使用了以下LaTeX包（大部分为标准包）：
- `amsmath`, `amssymb`, `amsthm`: 数学公式和定理环境
- `graphicx`: 图形支持
- `hyperref`: 超链接支持
- `usenix2020_SOUPS`: USENIX模板样式

