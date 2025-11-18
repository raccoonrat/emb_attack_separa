"""
MVES (最小验证实验) 配置模块

集中管理所有实验参数，采用配置驱动设计，保证可复现性
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import json
import os


@dataclass
class ModelConfig:
    """模型配置"""
    model_name: str = "google/switch-base-8"  # MVES关键：使用公开的MoE模型
    model_type: str = "switch"  # switch, mixtral, llama-moe
    device: str = "auto"  # auto, cuda, cpu
    torch_dtype: str = "float32"  # float32, bfloat16, float16
    trust_remote_code: bool = False


@dataclass
class WatermarkConfig:
    """水印配置 (论文定义5.1)"""
    secret_key: str = "MVES_DEFAULT_KEY"
    epsilon: float = 0.01  # 水印强度 ε = c²γ
    c_star: float = 2.0  # 安全系数 c*
    gamma_design: float = 0.03  # 设计的攻击强度 γ
    num_experts: int = 8  # 专家数量 K (switch-base-8默认)
    k_top: int = 1  # Top-k激活数 (switch-base-8默认)
    
    def __post_init__(self):
        """自动计算epsilon如果未指定"""
        if self.epsilon is None:
            self.epsilon = self.c_star**2 * self.gamma_design


@dataclass
class DetectionConfig:
    """检测器配置 (论文定理3.1)"""
    tau_alpha: float = 20.0  # LLR检测阈值
    alpha: float = 0.01  # 第一类错误率
    use_chernoff: bool = True  # 是否计算Chernoff信息


@dataclass
class AttackConfig:
    """攻击配置 (论文定义1.3)"""
    attack_type: str = "paraphrase"  # none, paraphrase, adversarial
    attack_strength: str = "moderate"  # mild, moderate, strong
    paraphrase_model: str = "Vamsi/T5_Paraphrase"  # 释义模型
    gamma_estimation_method: str = "upper_bound"  # upper_bound, kl_divergence


@dataclass
class ExperimentConfig:
    """实验配置"""
    experiment_name: str = "MVES"
    output_dir: str = "./mves_results"
    num_samples: int = 100  # 实验样本数
    batch_size: int = 4
    max_length: int = 512
    seed: int = 42  # 随机种子，保证可复现性
    
    # 数据集配置
    dataset_name: Optional[str] = None
    dataset_split: Optional[str] = None
    custom_prompts: Optional[List[str]] = field(default_factory=lambda: [
        "The quick brown fox jumps over the lazy dog.",
        "In a world where technology advances rapidly, artificial intelligence plays a crucial role.",
        "Climate change is one of the most pressing issues facing humanity today."
    ])


@dataclass
class MVESConfig:
    """MVES总配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    watermark: WatermarkConfig = field(default_factory=WatermarkConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    def __post_init__(self):
        """初始化后处理"""
        # 确保输出目录存在
        os.makedirs(self.experiment.output_dir, exist_ok=True)
        
        # 自动设置模型相关参数（统一使用switch-base-8）
        if "switch" in self.model.model_name.lower():
            self.model.model_type = "switch"
            # switch-base-8默认配置
            self.watermark.num_experts = 8  # switch-base-8固定8个专家
            self.watermark.k_top = 1  # switch-base-8默认Top-1激活
        
        # 确保模型名称统一
        if self.model.model_name != "google/switch-base-8":
            print(f"警告: 模型名称不是google/switch-base-8，当前: {self.model.model_name}")
            print("建议: 统一使用google/switch-base-8以确保兼容性")
    
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
        """保存配置到JSON文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"配置已保存到: {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'MVESConfig':
        """从JSON文件加载配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls(
            model=ModelConfig(**data.get("model", {})),
            watermark=WatermarkConfig(**data.get("watermark", {})),
            detection=DetectionConfig(**data.get("detection", {})),
            attack=AttackConfig(**data.get("attack", {})),
            experiment=ExperimentConfig(**data.get("experiment", {}))
        )
    
    def validate(self):
        """验证配置有效性"""
        errors = []
        
        # 验证水印配置
        if self.watermark.epsilon <= 0:
            errors.append("watermark.epsilon必须大于0")
        if self.watermark.c_star <= 0:
            errors.append("watermark.c_star必须大于0")
        if self.watermark.gamma_design <= 0:
            errors.append("watermark.gamma_design必须大于0")
        if self.watermark.num_experts <= 0:
            errors.append("watermark.num_experts必须大于0")
        if self.watermark.k_top > self.watermark.num_experts:
            errors.append("watermark.k_top不能大于num_experts")
        
        # 验证检测配置
        if self.detection.tau_alpha <= 0:
            errors.append("detection.tau_alpha必须大于0")
        if not (0 < self.detection.alpha < 1):
            errors.append("detection.alpha必须在(0,1)之间")
        
        if errors:
            raise ValueError("配置验证失败:\n" + "\n".join(f"  - {e}" for e in errors))
        
        print("✓ 配置验证通过")


# 预设配置
def get_default_config() -> MVESConfig:
    """获取默认配置"""
    return MVESConfig()


def get_quick_test_config() -> MVESConfig:
    """获取快速测试配置（小样本）"""
    config = MVESConfig()
    config.experiment.num_samples = 10
    config.experiment.batch_size = 2
    config.experiment.max_length = 128
    return config


def get_full_experiment_config() -> MVESConfig:
    """获取完整实验配置"""
    config = MVESConfig()
    config.experiment.num_samples = 1000
    config.experiment.batch_size = 8
    config.experiment.max_length = 512
    config.watermark.c_star = 2.0
    config.watermark.gamma_design = 0.03
    return config

