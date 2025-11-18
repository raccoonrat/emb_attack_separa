#!/usr/bin/env python3
"""
诊断程序挂起问题

运行此脚本可以帮助定位程序退出时挂起的原因
"""

import sys
import threading
import time
import gc
import torch

def check_threads():
    """检查所有活跃的线程"""
    print("\n" + "="*60)
    print("线程诊断")
    print("="*60)
    
    threads = threading.enumerate()
    print(f"总线程数: {len(threads)}")
    print(f"主线程: {threading.main_thread().name}")
    
    alive_threads = [t for t in threads if t.is_alive() and t != threading.main_thread()]
    print(f"\n活跃的后台线程数: {len(alive_threads)}")
    
    if alive_threads:
        print("\n后台线程详情:")
        for t in alive_threads:
            print(f"  - 名称: {t.name}")
            print(f"    守护线程: {t.daemon}")
            print(f"    是否存活: {t.is_alive()}")
            print(f"    标识符: {t.ident}")
            print()
    else:
        print("✓ 没有活跃的后台线程")

def check_cuda():
    """检查CUDA状态"""
    print("\n" + "="*60)
    print("CUDA诊断")
    print("="*60)
    
    if torch.cuda.is_available():
        print(f"✓ CUDA可用")
        print(f"  设备数量: {torch.cuda.device_count()}")
        print(f"  当前设备: {torch.cuda.current_device()}")
        print(f"  设备名称: {torch.cuda.get_device_name(0)}")
        
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"  已分配显存: {allocated:.2f} GB")
        print(f"  已保留显存: {reserved:.2f} GB")
        
        if allocated > 0 or reserved > 0:
            print("  ⚠ 警告: 仍有显存未释放")
            print("  尝试清理...")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            allocated_after = torch.cuda.memory_allocated(0) / 1024**3
            reserved_after = torch.cuda.memory_reserved(0) / 1024**3
            print(f"  清理后 - 已分配: {allocated_after:.2f} GB, 已保留: {reserved_after:.2f} GB")
    else:
        print("✗ CUDA不可用")

def check_gc():
    """检查垃圾回收状态"""
    print("\n" + "="*60)
    print("垃圾回收诊断")
    print("="*60)
    
    # 获取垃圾回收统计
    counts = gc.get_count()
    print(f"GC计数: {counts}")
    
    # 强制垃圾回收
    print("\n执行垃圾回收...")
    collected = gc.collect()
    print(f"释放对象数: {collected}")
    
    counts_after = gc.get_count()
    print(f"GC计数（回收后）: {counts_after}")

def test_exit():
    """测试退出行为"""
    print("\n" + "="*60)
    print("退出测试")
    print("="*60)
    
    print("准备退出...")
    print("如果程序在这里挂起，说明有资源未释放")
    
    # 清理资源
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    gc.collect()
    
    # 检查线程
    alive_threads = [t for t in threading.enumerate() 
                     if t.is_alive() and t != threading.main_thread()]
    if alive_threads:
        print(f"\n⚠ 警告: 仍有 {len(alive_threads)} 个线程未退出")
        for t in alive_threads:
            print(f"  - {t.name} (daemon={t.daemon})")
    
    print("\n调用 sys.exit(0)...")
    sys.exit(0)

if __name__ == "__main__":
    print("="*60)
    print("程序挂起诊断工具")
    print("="*60)
    
    # 运行诊断
    check_threads()
    check_cuda()
    check_gc()
    test_exit()

