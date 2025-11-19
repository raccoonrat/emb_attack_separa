"""
统一异常定义

按照Linus的哲学：简洁、明确、实用
"""


class WatermarkError(Exception):
    """水印相关错误的基类"""
    pass


class CalibrationError(WatermarkError):
    """标定过程错误"""
    pass


class DetectionError(WatermarkError):
    """检测过程错误"""
    pass


class ModelPatchError(WatermarkError):
    """模型patch错误"""
    pass


class ConfigurationError(WatermarkError):
    """配置错误"""
    pass

