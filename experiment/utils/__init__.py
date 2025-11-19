"""
工具模块：提供通用的错误处理、日志、性能优化等功能
"""

from .logger import get_logger, setup_logging
from .exceptions import WatermarkError, CalibrationError, DetectionError

__all__ = [
    'get_logger',
    'setup_logging',
    'WatermarkError',
    'CalibrationError',
    'DetectionError',
]

