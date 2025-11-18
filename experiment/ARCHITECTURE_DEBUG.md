# Switch-base-8 架构调试指南

## 问题：模型架构验证失败

如果遇到 `模型架构验证失败，请确认使用的是switch-base-8` 错误，请按照以下步骤调试：

## 1. 检查模型是否正确加载

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "google/switch-base-8"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"模型类型: {type(model).__name__}")
print(f"模型配置: {type(model.config).__name__}")
```

## 2. 检查模型结构

```python
# 检查decoder
if hasattr(model, 'decoder'):
    print("✓ 找到 model.decoder")
    decoder = model.decoder
elif hasattr(model, 'model') and hasattr(model.model, 'decoder'):
    print("✓ 找到 model.model.decoder")
    decoder = model.model.decoder
else:
    print("✗ 未找到decoder")
    print("可用属性:", [attr for attr in dir(model) if not attr.startswith('_')])

# 检查decoder blocks
if hasattr(decoder, 'block'):
    print("✓ 找到 decoder.block")
    blocks = decoder.block
elif hasattr(decoder, 'layers'):
    print("✓ 找到 decoder.layers")
    blocks = decoder.layers
else:
    print("✗ 未找到blocks")
    print("decoder可用属性:", [attr for attr in dir(decoder) if not attr.startswith('_')])
```

## 3. 检查MoE层

```python
# 检查第一层结构
if len(blocks) > 0:
    first_layer = blocks[0]
    print(f"第一层类型: {type(first_layer).__name__}")
    print(f"第一层属性: {[attr for attr in dir(first_layer) if not attr.startswith('_')]}")
    
    # 检查layer结构
    if hasattr(first_layer, 'layer'):
        print(f"layer长度: {len(first_layer.layer)}")
        if len(first_layer.layer) > 1:
            ffn = first_layer.layer[1]
            print(f"FFN类型: {type(ffn).__name__}")
            print(f"FFN属性: {[attr for attr in dir(ffn) if not attr.startswith('_')]}")
            
            # 检查router
            if hasattr(ffn, 'mlp'):
                print(f"mlp类型: {type(ffn.mlp).__name__}")
                if hasattr(ffn.mlp, 'router'):
                    print("✓ 找到router!")
                    print(f"router类型: {type(ffn.mlp.router).__name__}")
```

## 4. 常见架构变体

Switch Transformer可能有不同的架构变体：

### 变体1: 标准结构
```
model.decoder.block[i].layer[1].mlp.router
```

### 变体2: 简化结构
```
model.decoder.layers[i].feed_forward.router
```

### 变体3: 直接router
```
model.decoder.block[i].router
```

## 5. 手动验证

运行以下脚本进行详细检查：

```python
python -c "
from transformers import AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained('google/switch-base-8')
print('模型类型:', type(model).__name__)
print('模型属性:', [a for a in dir(model) if not a.startswith('_')][:15])
if hasattr(model, 'decoder'):
    print('decoder类型:', type(model.decoder).__name__)
    print('decoder属性:', [a for a in dir(model.decoder) if not a.startswith('_')][:15])
"
```

## 6. 解决方案

如果验证失败，请：

1. **检查模型名称**: 确认使用的是 `google/switch-base-8`
2. **检查transformers版本**: 可能需要更新
   ```bash
   pip install --upgrade transformers
   ```
3. **查看详细错误信息**: 运行 `deploy_switch_base8.py` 会输出详细的调试信息
4. **手动适配**: 根据输出的调试信息，手动修改 `verify_switch_architecture` 函数

## 7. 报告问题

如果问题持续存在，请提供以下信息：

- transformers版本: `pip show transformers`
- 模型加载方式
- 完整的错误堆栈
- 模型结构调试输出

