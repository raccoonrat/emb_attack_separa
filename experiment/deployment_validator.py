"""
部署验证框架（1118文档第9节）

确保系统的理论约束在实际环境中满足
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
from torch.utils.data import DataLoader

from calibration import calibrate_Lg, compute_chernoff_information
from moe_watermark_enhanced import MoEWatermarkHookWrapper


class DeploymentValidator:
    """
    部署前验证器（1118文档第9.1节）
    
    确保系统的理论约束在实际环境中满足
    """
    
    def __init__(self, model: Any, config: Dict[str, Any]):
        """
        初始化验证器
        
        Args:
            model: 预训练模型
            config: 配置字典，包含：
                - L_g: Lipschitz常数
                - C: 综合常数
                - c: 安全系数
                - validation_data: 验证数据集
                - max_ppl_drop: 最大可接受的PPL下降
        """
        self.model = model
        self.config = config
        self.device = next(model.parameters()).device
    
    def validate_all(self) -> Dict[str, Any]:
        """
        执行所有验证检查
        
        Returns:
            validation_result: 验证结果字典
        """
        checks = []
        
        # 检查1: Lipschitz常数在预期范围
        check1 = self._check_lipschitz_constant()
        checks.append(check1)
        
        # 检查2: 综合常数C的拟合质量
        check2 = self._check_combined_constant()
        checks.append(check2)
        
        # 检查3: 安全系数c的有效性
        check3 = self._check_safety_coefficient()
        checks.append(check3)
        
        # 检查4: 性能成本可接受性
        check4 = self._check_performance_cost()
        checks.append(check4)
        
        # 检查5: Top-k激活的排名稳定性
        check5 = self._check_ranking_stability()
        checks.append(check5)
        
        # 总体判决
        all_passed = all(check['status'] for check in checks)
        
        return {
            'passed': all_passed,
            'checks': checks,
            'deployment_ready': all_passed,
            'issues': [ch for ch in checks if not ch['status']],
            'warnings': [ch for ch in checks if ch.get('warning', False)]
        }
    
    def _check_lipschitz_constant(self) -> Dict[str, Any]:
        """
        检查1: Lipschitz常数在预期范围（1118文档）
        
        预期范围: L_95 ∈ [1.5, 3.0]
        如果L_max > 10，需要梯度裁剪
        """
        if 'validation_data' not in self.config:
            return {
                'name': 'Lipschitz Constant Check',
                'status': False,
                'error': 'validation_data not provided'
            }
        
        try:
            L_g_measured = calibrate_Lg(
                self.model,
                self.config['validation_data'],
                self.device
            )
            
            expected_range = self.config.get('L_g_range', (1.5, 3.0))
            L_max_threshold = self.config.get('L_max_threshold', 10.0)
            
            status = expected_range[0] <= L_g_measured <= expected_range[1]
            
            check = {
                'name': 'Lipschitz Constant Check',
                'measured': L_g_measured,
                'expected_range': expected_range,
                'status': status,
                'action': 'Apply gradient clipping if L_max > 10' if L_g_measured > L_max_threshold else 'OK'
            }
            
            if L_g_measured > L_max_threshold:
                check['warning'] = True
                check['message'] = f'L_g={L_g_measured:.2f} exceeds threshold {L_max_threshold}, consider gradient clipping'
            
            return check
            
        except Exception as e:
            return {
                'name': 'Lipschitz Constant Check',
                'status': False,
                'error': str(e)
            }
    
    def _check_combined_constant(self) -> Dict[str, Any]:
        """
        检查2: 综合常数C的拟合质量（1118文档）
        
        要求: R² > 0.90
        """
        if 'C_prop' not in self.config or 'C_R_squared' not in self.config:
            return {
                'name': 'Combined Constant Calibration',
                'status': False,
                'error': 'C_prop or C_R_squared not provided'
            }
        
        C_prop = self.config['C_prop']
        R_squared = self.config['C_R_squared']
        threshold = self.config.get('R_squared_threshold', 0.90)
        
        status = R_squared > threshold
        
        return {
            'name': 'Combined Constant Calibration',
            'C_prop': C_prop,
            'R_squared': R_squared,
            'threshold': threshold,
            'status': status,
            'action': 'Increase sample size if R² < threshold' if not status else 'OK'
        }
    
    def _check_safety_coefficient(self) -> Dict[str, Any]:
        """
        检查3: 安全系数c的有效性（1118文档）
        
        要求: c > C (最小安全系数)
        建议: margin > 10%
        """
        c_configured = self.config.get('c', None)
        C_minimum = self.config.get('C', None)
        
        if c_configured is None or C_minimum is None:
            return {
                'name': 'Safety Coefficient Validity',
                'status': False,
                'error': 'c or C not provided'
            }
        
        status = c_configured > C_minimum
        margin = ((c_configured - C_minimum) / C_minimum * 100) if C_minimum > 0 else 0
        
        return {
            'name': 'Safety Coefficient Validity',
            'c_configured': c_configured,
            'c_minimum_required': C_minimum,
            'status': status,
            'margin_percent': margin,
            'action': 'Increase c if margin < 10%' if margin < 10 else 'OK',
            'warning': margin < 10
        }
    
    def _check_performance_cost(self) -> Dict[str, Any]:
        """
        检查4: 性能成本可接受性（1118文档）
        
        测量PPL下降，确保在可接受范围内
        """
        max_acceptable_ppl_drop = self.config.get('max_ppl_drop', 2.0)
        
        # 这里需要实际测量PPL，简化实现
        # 实际应该调用measure_ppl函数
        measured_ppl_drop = self.config.get('measured_ppl_drop', None)
        
        if measured_ppl_drop is None:
            return {
                'name': 'Performance Cost Acceptability',
                'status': False,
                'error': 'measured_ppl_drop not provided (need to measure PPL)'
            }
        
        status = measured_ppl_drop <= max_acceptable_ppl_drop
        
        return {
            'name': 'Performance Cost Acceptability',
            'measured_ppl_drop': measured_ppl_drop,
            'max_acceptable': max_acceptable_ppl_drop,
            'status': status,
            'action': 'Reduce c if PPL drop exceeds threshold' if not status else 'OK'
        }
    
    def _check_ranking_stability(self) -> Dict[str, Any]:
        """
        检查5: Top-k激活的排名稳定性（1118文档）
        
        测量排名交叉频率，应该 < 10%
        """
        ranking_exchange_freq = self.config.get('ranking_exchange_frequency', None)
        
        if ranking_exchange_freq is None:
            return {
                'name': 'Ranking Exchange Frequency',
                'status': False,
                'error': 'ranking_exchange_frequency not provided (need to measure)'
            }
        
        threshold = self.config.get('ranking_exchange_threshold', 0.10)
        status = ranking_exchange_freq < threshold
        
        return {
            'name': 'Ranking Exchange Frequency',
            'frequency': ranking_exchange_freq,
            'expected': f'<{threshold*100}%',
            'status': status,
            'action': 'Reduce delta_logit magnitude if too high' if not status else 'OK'
        }


def validate_deployment(
    model: Any,
    config: Dict[str, Any],
    validation_data: Optional[DataLoader] = None
) -> Dict[str, Any]:
    """
    部署验证的便捷函数
    
    Args:
        model: 预训练模型
        config: 配置字典
        validation_data: 验证数据集（可选）
        
    Returns:
        validation_result: 验证结果
    """
    if validation_data is not None:
        config['validation_data'] = validation_data
    
    validator = DeploymentValidator(model, config)
    return validator.validate_all()

