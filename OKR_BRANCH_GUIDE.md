# OKR 独立分支创建指南

## 概述

本文档说明如何将 OKR (Opportunistic Keyed Routing) 相关代码作为独立分支分离出来。

## OKR 相关文件列表

### 核心代码文件
- `experiment/okr_config.py` - OKR 配置类
- `experiment/okr_experiment.py` - OKR 实验框架
- `experiment/okr_example.py` - 使用示例
- `experiment/okr_kernel.py` - OKR 核心路由逻辑
- `experiment/okr_patch.py` - 水印注入代码
- `experiment/okr_detector.py` - 水印检测器

### 运行脚本
- `experiment/run_okr_experiment.py` - 实验运行脚本
- `experiment/run_okr_with_sudo.sh` - 使用sudo运行的脚本
- `experiment/test_okr_basic.py` - 基础测试脚本
- `experiment/test_okr_deepseek_moe_local.py` - DeepSeek-MoE本地测试

### 文档文件
- `experiment/OKR_README.md` - OKR 实验指南
- `experiment/START_OKR_EXPERIMENT.md` - 快速开始指南
- `okr/` - OKR 算法相关文档目录
- `OKR_Colab_Experiment.ipynb` - Colab 实验笔记本

### 结果目录
- `experiment/okr_results/` - 实验结果目录

## 创建独立分支的步骤

### 方法1: 使用提供的脚本（推荐）

#### 使用 Python 脚本
```bash
python create_okr_branch.py
```

#### 使用 Bash 脚本
```bash
bash create_okr_branch.sh
```

### 方法2: 手动创建分支

#### 步骤1: 检查当前状态
```bash
git status
git branch -a
```

#### 步骤2: 处理未提交的更改

如果有未提交的更改，可以选择：

**选项A: 提交更改**
```bash
git add -A
git commit -m "WIP: OKR相关更改"
```

**选项B: 暂存更改**
```bash
git stash push -m "临时暂存: 创建OKR分支前"
```

**选项C: 放弃更改**
```bash
git restore .
git clean -fd
```

#### 步骤3: 创建并切换到新分支
```bash
git checkout -b okr
```

#### 步骤4: 验证分支创建
```bash
git branch --show-current
# 应该显示: okr
```

#### 步骤5: 推送到远程（可选）
```bash
git push -u origin okr
```

## 分支管理建议

### 分支命名
- 主分支: `okr` 或 `okr-feature`
- 如果需要多个OKR相关分支，可以使用:
  - `okr-dev` - 开发分支
  - `okr-experiment` - 实验分支
  - `okr-docs` - 文档分支

### 分支策略

1. **从 main 分支创建**: 确保基于最新的主分支
   ```bash
   git checkout main
   git pull origin main
   git checkout -b okr
   ```

2. **定期同步 main 分支**: 保持与主分支同步
   ```bash
   git checkout okr
   git merge main
   # 或使用 rebase
   git rebase main
   ```

3. **提交规范**: 使用清晰的提交信息
   ```bash
   git commit -m "feat(okr): 添加新的路由算法"
   git commit -m "fix(okr): 修复检测器bug"
   git commit -m "docs(okr): 更新README"
   ```

## 验证分支创建成功

运行以下命令确认：

```bash
# 查看当前分支
git branch --show-current

# 查看所有分支
git branch -a

# 查看OKR相关文件
ls experiment/okr_*.py
ls experiment/OKR_*.md
ls experiment/run_okr_*.py
ls experiment/test_okr_*.py
```

## 后续操作

### 在OKR分支上工作
```bash
git checkout okr
# 进行开发和修改
git add .
git commit -m "你的提交信息"
git push origin okr
```

### 切换回主分支
```bash
git checkout main
```

### 合并OKR分支到main（如果需要）
```bash
git checkout main
git merge okr
git push origin main
```

## 注意事项

1. **文件依赖**: OKR代码可能依赖 `utils/` 目录下的工具，这些文件应该保留在主分支
2. **配置共享**: 某些配置文件（如 `cache_config.py`）可能被多个模块共享
3. **文档同步**: 如果主分支的文档有更新，可能需要同步到OKR分支

## 故障排除

### 问题1: 分支已存在
```bash
# 删除本地分支
git branch -d okr

# 或切换到已存在的分支
git checkout okr
```

### 问题2: 有冲突的更改
```bash
# 查看冲突
git status

# 解决冲突后
git add .
git commit -m "解决冲突"
```

### 问题3: 无法推送到远程
```bash
# 检查远程配置
git remote -v

# 设置上游分支
git push -u origin okr
```

## 联系和支持

如有问题，请参考：
- `experiment/OKR_README.md` - OKR使用文档
- `experiment/START_OKR_EXPERIMENT.md` - 快速开始指南

