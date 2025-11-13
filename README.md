# MoE 水印论文 LaTeX 项目

本项目包含关于 MoE（混合专家模型）专家激活水印对抗释义攻击的理论证明论文的 LaTeX 源文件。

## 项目结构

```
.
├── README.md                    # 项目说明文件（本文件）
├── .gitignore                   # Git 忽略文件配置
│
├── styles/                      # LaTeX 样式文件目录
│   └── usenix2020_SOUPS.sty    # USENIX SOUPS 2020 会议模板样式
│
├── build/                       # 编译输出目录
│   └── *.pdf                   # 编译生成的 PDF 文件
│
├── docs/                        # 文档和说明文件
│   ├── README_LaTeX.md         # LaTeX 编译详细说明
│   └── *.md                    # 其他 Markdown 文档
│
└── *.tex                        # LaTeX 源文件（论文主文件）
```

## 主要论文文件

### 核心论文
- **`moe_paradigm_rigorous_proofs.tex`** - 范式之争的严格数学证明（中文版，USENIX SOUPS 格式）
  - 包含完整的理论证明框架
  - 所有公式已编号
  - 符合 USENIX SOUPS 模板格式

### 其他版本
- `moe_paradigm_rigorous_proofs_soups.tex` - SOUPS 格式版本
- `moe_watermark_paraphrase_attack.tex` - 英文版论文
- `moe_watermark_paraphrase_attack_zh.tex` - 中文版论文
- `moe_watermark_paraphrase_attack_zh_simple.tex` - 简化中文版

## 编译方法

### 推荐方法：使用 latexmk

```bash
# 编译中文版（使用 XeLaTeX）
latexmk -xelatex moe_paradigm_rigorous_proofs.tex

# 编译英文版（使用 pdfLaTeX）
latexmk -pdf moe_watermark_paraphrase_attack.tex
```

### 手动编译

#### 中文版（XeLaTeX）
```bash
xelatex moe_paradigm_rigorous_proofs.tex
xelatex moe_paradigm_rigorous_proofs.tex  # 第二次编译以生成正确的引用
```

#### 英文版（pdfLaTeX）
```bash
pdflatex moe_watermark_paraphrase_attack.tex
pdflatex moe_watermark_paraphrase_attack.tex  # 第二次编译
```

## 依赖要求

### LaTeX 发行版
- TeX Live 2020 或更高版本（推荐）
- MiKTeX 2.9 或更高版本

### 必需的 LaTeX 包
- `ctexart` - 中文支持（中文版）
- `amsmath`, `amssymb`, `amsthm` - 数学公式和定理环境
- `graphicx` - 图形支持
- `hyperref` - 超链接支持
- `usenix2020_SOUPS` - USENIX 模板样式（位于 `styles/` 目录）

## 项目特点

1. **严格的数学证明**：包含完整的信息论推导和统计假设检验
2. **公式编号**：所有公式均已编号，便于引用
3. **双栏布局**：符合 USENIX SOUPS 会议格式要求
4. **代码规范**：遵循 LaTeX 最佳实践

## 文件说明

### 源文件（保留在版本控制中）
- `*.tex` - LaTeX 源文件
- `styles/*.sty` - 样式文件

### 编译产物（已忽略，不提交到 Git）
- `*.log` - 编译日志
- `*.aux` - 辅助文件
- `*.out` - 超链接输出
- `*.fls`, `*.fdb_latexmk` - latexmk 相关文件
- `*.synctex*` - SyncTeX 文件
- `build/*.pdf` - PDF 输出（可选，可在 `.gitignore` 中配置）

## 清理项目

项目已按照正式 LaTeX 项目规范整理：
- ✅ 所有编译临时文件已删除
- ✅ 文件已按功能分类到相应目录
- ✅ `.gitignore` 已配置，忽略所有临时文件
- ✅ 样式文件路径已更新

## 注意事项

1. **编译路径**：确保在项目根目录执行编译命令
2. **样式文件**：样式文件位于 `styles/` 目录，LaTeX 会自动查找
3. **中文支持**：中文版需要使用 XeLaTeX 或 LuaLaTeX 编译
4. **多次编译**：某些功能（如交叉引用）需要多次编译才能正确生成

## 贡献

如有问题或建议，请提交 Issue 或 Pull Request。

## 许可证

请参考项目许可证文件。

