"""
OKR (Opportunistic Keyed Routing) 配置模块

独立于 MVES 配置，专门用于 OKR 算法实验
采用配置驱动设计，保证可复现性
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import json
import os


@dataclass
class OKRModelConfig:
    """OKR 模型配置"""
    model_name: str = "google/switch-base-8"  # 默认使用 Switch Transformers，可以是 HuggingFace ID 或本地路径
    model_type: str = "switch"  # switch, mixtral, deepseek_moe
    device: str = "auto"  # auto, cuda, cpu
    torch_dtype: str = "float32"  # float32, bfloat16, float16
    trust_remote_code: bool = False
    max_memory: Optional[Dict[int, str]] = None  # GPU 内存限制，如 {0: "5GB"}
    local_model_path: Optional[str] = None  # 本地模型路径（如果提供，优先使用本地路径）


@dataclass
class OKRWatermarkConfig:
    """OKR 水印配置"""
    secret_key: str = "OKR_DEFAULT_SECRET_KEY"
    epsilon: float = 1.5  # 质量容忍阈值（Logit 差值）
    num_experts: int = 8  # 专家数量 K
    top_k: int = 1  # Top-k 激活数（Switch-base-8 默认是 1）
    
    def validate(self):
        """验证配置有效性"""
        if self.epsilon <= 0:
            raise ValueError("epsilon 必须大于 0")
        if self.num_experts <= 0:
            raise ValueError("num_experts 必须大于 0")
        if self.top_k > self.num_experts:
            raise ValueError("top_k 不能大于 num_experts")


@dataclass
class OKRDetectionConfig:
    """OKR 检测器配置"""
    hit_rate_threshold: float = 0.8  # 命中率阈值（超过此值判定为有水印）
    min_opportunities: int = 10  # 最小机会窗口数（少于此值无法检测）


@dataclass
class OKRAttackConfig:
    """OKR 攻击配置（用于鲁棒性测试）"""
    attack_type: str = "paraphrase"  # none, paraphrase
    paraphrase_model: str = "Vamsi/T5_Paraphrase"  # 释义模型
    attack_strength: str = "moderate"  # mild, moderate, strong


@dataclass
class OKRExperimentConfig:
    """OKR 实验配置"""
    experiment_name: str = "OKR_Basic"
    output_dir: str = "./okr_results"
    num_samples: int = 100  # 实验样本数
    batch_size: int = 4
    max_length: int = 512
    seed: int = 42  # 随机种子
    
    # 数据集配置
    dataset_name: Optional[str] = None
    dataset_split: Optional[str] = None
    custom_prompts: Optional[List[str]] = field(default_factory=lambda: [
        "The quick brown fox jumps over the lazy dog.",
        "In a world where technology advances rapidly, artificial intelligence plays a crucial role.",
        "Climate change is one of the most pressing issues facing humanity today."
    ])


@dataclass
class OKRConfig:
    """OKR 总配置"""
    model: OKRModelConfig = field(default_factory=OKRModelConfig)
    watermark: OKRWatermarkConfig = field(default_factory=OKRWatermarkConfig)
    detection: OKRDetectionConfig = field(default_factory=OKRDetectionConfig)
    attack: OKRAttackConfig = field(default_factory=OKRAttackConfig)
    experiment: OKRExperimentConfig = field(default_factory=OKRExperimentConfig)
    
    def __post_init__(self):
        """初始化后处理"""
        # 确保输出目录存在
        os.makedirs(self.experiment.output_dir, exist_ok=True)
        
        # 自动设置模型相关参数
        model_name_lower = self.model.model_name.lower()
        if "switch" in model_name_lower:
            self.model.model_type = "switch"
            self.watermark.num_experts = 8
            self.watermark.top_k = 1
        elif "deepseek" in model_name_lower and "moe" in model_name_lower:
            self.model.model_type = "deepseek_moe"
            # DeepSeek-MoE-16B: 64 个专家，top-8 激活
            # 如果配置中没有设置，使用默认值
            if self.watermark.num_experts == 8:  # 默认值，需要更新
                self.watermark.num_experts = 64
            if self.watermark.top_k == 1:  # 默认值，需要更新
                self.watermark.top_k = 8
        
        # 验证配置
        self.watermark.validate()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "model": self.model.__dict__,
            "watermark": self.watermark.__dict__,
            "detection": self.detection.__dict__,
            "attack": self.attack.__dict__,
            "experiment": self.experiment.__dict__
        }
    
    def save(self, filepath: str):
        """保存配置到 JSON 文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"配置已保存到: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'OKRConfig':
        """从 JSON 文件加载配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls(
            model=OKRModelConfig(**data.get("model", {})),
            watermark=OKRWatermarkConfig(**data.get("watermark", {})),
            detection=OKRDetectionConfig(**data.get("detection", {})),
            attack=OKRAttackConfig(**data.get("attack", {})),
            experiment=OKRExperimentConfig(**data.get("experiment", {}))
        )
    
    def validate(self):
        """验证配置有效性"""
        self.watermark.validate()
        
        if not (0 < self.detection.hit_rate_threshold <= 1.0):
            raise ValueError("detection.hit_rate_threshold 必须在 (0, 1] 之间")
        
        if self.detection.min_opportunities < 1:
            raise ValueError("detection.min_opportunities 必须大于等于 1")
        
        print("✓ OKR 配置验证通过")


# 预设配置
def get_default_okr_config() -> OKRConfig:
    """获取默认 OKR 配置"""
    return OKRConfig()


def get_quick_test_okr_config() -> OKRConfig:
    """获取快速测试配置（小样本）"""
    config = OKRConfig()
    config.experiment.num_samples = 10
    config.experiment.batch_size = 2
    config.experiment.max_length = 128
    return config


def get_robustness_test_okr_config() -> OKRConfig:
    """获取鲁棒性测试配置"""
    config = OKRConfig()
    config.experiment.experiment_name = "OKR_Robustness"
    config.attack.attack_type = "paraphrase"
    config.experiment.num_samples = 200
    return config


def get_deepseek_moe_config(local_path: Optional[str] = None) -> OKRConfig:
    """
    获取 DeepSeek-MoE 配置
    
    针对 A800 GPU 优化：
    - 使用 bfloat16 精度（A800 原生支持）
    - 设置合理的 GPU 内存限制
    - 适配 DeepSeek-MoE 的专家数量（通常为 64 个专家，top-8 激活）
    
    Args:
        local_path: 本地模型路径（如果提供，优先使用本地路径）
    """
    config = OKRConfig()
    
    # 模型配置
    if local_path:
        # 使用本地路径
        config.model.local_model_path = local_path
        config.model.model_name = local_path  # 也设置 model_name 以便兼容
        print(f"使用本地模型路径: {local_path}")
    else:
        # 使用 HuggingFace 模型 ID
        config.model.model_name = "deepseek-ai/DeepSeek-MoE-16B"  # 或 "deepseek-ai/DeepSeek-MoE-16B-base"
    
    config.model.model_type = "deepseek_moe"
    config.model.device = "cuda"
    config.model.torch_dtype = "bfloat16"  # A800 原生支持 bfloat16，性能更好
    config.model.trust_remote_code = True  # DeepSeek 可能需要 trust_remote_code
    
    # A800 80GB 显存配置：保守使用，留出余量
    config.model.max_memory = {0: "70GB"}  # 留出 10GB 余量
    
    # 水印配置（DeepSeek-MoE 通常有更多专家）
    # DeepSeek-MoE-16B: 64 个专家，top-8 激活
    config.watermark.num_experts = 64
    config.watermark.top_k = 8
    config.watermark.epsilon = 2.0  # 稍微提高 epsilon，因为专家更多
    
    # 实验配置
    config.experiment.experiment_name = "OKR_DeepSeek_MoE"
    config.experiment.batch_size = 1  # DeepSeek-MoE 较大，batch_size 设为 1
    config.experiment.max_length = 512
    
    return config


def get_deepseek_moe_quick_test_config(local_path: Optional[str] = None) -> OKRConfig:
    """获取 DeepSeek-MoE 快速测试配置（小样本，用于快速验证）"""
    config = get_deepseek_moe_config(local_path=local_path)
    config.experiment.num_samples = 5
    config.experiment.batch_size = 1
    config.experiment.max_length = 128
    config.experiment.experiment_name = "OKR_DeepSeek_MoE_QuickTest"
    return config

