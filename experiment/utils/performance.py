"""
性能优化工具

减少重复计算，批量处理，内存管理
"""

import torch
import gc
from typing import Callable, Any
from functools import wraps
import time


def batch_process(
    items: list,
    batch_size: int,
    process_fn: Callable,
    *args,
    **kwargs
) -> list:
    """
    批量处理列表，减少内存峰值
    
    Args:
        items: 待处理列表
        batch_size: 批次大小
        process_fn: 处理函数，接受(batch, *args, **kwargs)
        *args, **kwargs: 传递给process_fn的额外参数
        
    Returns:
        处理结果列表
    """
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = process_fn(batch, *args, **kwargs)
        results.extend(batch_results)
        
        # 清理GPU缓存（如果使用GPU）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


def clear_cache(func: Callable) -> Callable:
    """
    装饰器：函数执行后清理GPU缓存
    
    用于内存受限的场景
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return result
    return wrapper


def timing(func: Callable) -> Callable:
    """
    装饰器：测量函数执行时间
    
    用于性能分析
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} 耗时: {elapsed:.2f}秒")
        return result
    return wrapper


def safe_tensor_operation(
    tensor: torch.Tensor,
    operation: Callable,
    default_value: Any = None,
    error_message: str = "Tensor operation failed"
) -> Any:
    """
    安全的tensor操作，捕获异常并返回默认值
    
    Args:
        tensor: 输入tensor
        operation: 操作函数
        default_value: 失败时的默认返回值
        error_message: 错误消息
        
    Returns:
        操作结果或默认值
    """
    try:
        return operation(tensor)
    except Exception as e:
        print(f"{error_message}: {e}")
        return default_value

