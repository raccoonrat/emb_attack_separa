# 开始 OKR 实验

## 快速开始

### 方式1: 运行测试脚本（推荐先运行）

```bash
cd experiment
python test_okr_basic.py
```

这会验证：
- 所有模块能否正常导入
- 配置是否正确
- OKRRouter 核心逻辑是否正常
- 设备（GPU/CPU）状态

### 方式2: 运行完整实验

```bash
cd experiment
python run_okr_experiment.py
```

这会运行完整的 OKR 基础实验，包括：
- 加载模型（google/switch-base-8）
- 注入 OKR 水印
- 生成带水印的文本
- 检测水印
- 保存结果到 `./okr_results/`

### 方式3: 使用示例代码

```bash
cd experiment
python okr_example.py
```

这会运行 `example_run_experiment()`，使用快速测试配置（10个样本）。

## 实验配置

### 快速测试配置（默认）

- 样本数: 10
- 批次大小: 2
- 最大长度: 128
- 适合快速验证功能

### 完整实验配置

修改 `run_okr_experiment.py` 中的配置：

```python
config = get_default_okr_config()  # 改为完整配置
config.experiment.num_samples = 100  # 更多样本
config.experiment.batch_size = 4
```

## 输出文件

实验完成后，会在 `./okr_results/` 目录下生成：

- `experiment.log` - 实验日志
- `results.json` - 实验结果（基础实验）
- `robustness_results.json` - 鲁棒性实验结果（如果运行）

## 常见问题

### 1. 内存不足

如果 GPU 内存不足，可以：
- 使用 FP16: `config.model.torch_dtype = "float16"`
- 限制内存: `config.model.max_memory = {0: "5GB"}`
- 使用 CPU: `config.model.device = "cpu"`（会很慢）

### 2. 模型下载慢

已自动配置使用 Hugging Face 镜像源 (`https://hf-mirror.com`)

### 3. 导入错误

确保在 `experiment/` 目录下运行，或添加路径：

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
```

## 下一步

实验成功后，可以：

1. **调整参数**：修改 `epsilon`、`secret_key` 等参数
2. **运行鲁棒性测试**：测试释义攻击下的表现
3. **分析结果**：查看 `results.json` 中的详细数据
4. **扩展实验**：添加自己的实验类型

## 代码结构

```
experiment/
├── okr_config.py          # 配置类
├── okr_experiment.py      # 实验框架
├── okr_example.py         # 示例代码
├── okr_kernel.py          # 核心路由逻辑
├── okr_patch.py           # 水印注入
├── okr_detector.py        # 水印检测
├── run_okr_experiment.py  # 实验运行脚本
├── test_okr_basic.py      # 基础测试脚本
└── START_OKR_EXPERIMENT.md # 本文档
```

