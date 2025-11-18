"""
快速启动脚本 - switch-base-8 (GTX 4050)

一键部署和测试
"""

import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

def check_environment():
    """检查环境"""
    print("检查环境...")
    
    # 检查CUDA
    try:
        import torch
        if not torch.cuda.is_available():
            print("✗ CUDA不可用")
            return False
        print(f"✓ CUDA可用: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("✗ PyTorch未安装")
        return False
    
    # 检查transformers
    try:
        import transformers
        print(f"✓ transformers版本: {transformers.__version__}")
    except ImportError:
        print("✗ transformers未安装")
        return False
    
    return True

def main():
    print("="*60)
    print("快速启动 - switch-base-8 部署")
    print("="*60)
    
    # 环境检查
    if not check_environment():
        print("\n请先安装依赖:")
        print("  pip install -r requirements.txt")
        return
    
    # 选择模式
    print("\n选择部署模式:")
    print("1. 最小测试（推荐首次使用）")
    print("2. 完整部署（Hook机制）")
    print("3. 完整部署（Patch方式）")
    print("4. 8-bit量化部署（显存不足时）")
    
    choice = input("\n请选择 (1-4): ").strip()
    
    if choice == "1":
        # 最小测试
        print("\n运行最小测试...")
        from test_minimal_switch import main as test_main
        test_main()
        
    elif choice == "2":
        # Hook部署
        print("\n运行完整部署（Hook机制）...")
        from deploy_switch_base8 import main as deploy_main
        sys.argv = ["deploy_switch_base8.py"]
        deploy_main()
        
    elif choice == "3":
        # Patch部署
        print("\n运行完整部署（Patch方式）...")
        from deploy_switch_base8 import main as deploy_main
        sys.argv = ["deploy_switch_base8.py", "--use_patch"]
        deploy_main()
        
    elif choice == "4":
        # 8-bit部署
        print("\n运行8-bit量化部署...")
        from deploy_switch_base8 import main as deploy_main
        sys.argv = ["deploy_switch_base8.py", "--use_8bit"]
        deploy_main()
        
    else:
        print("无效选择")

if __name__ == "__main__":
    main()

