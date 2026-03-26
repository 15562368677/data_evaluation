"""
验证器基类和通用类型

定义验证结果、问题等级、配置基类（严格按照 criterion.md）
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class IssueLevel(Enum):
    """问题等级"""
    CRITICAL = "critical"   # 严重：数据无法使用
    MAJOR = "major"         # 重要：显著影响质量
    MINOR = "minor"         # 轻微：建议优化
    INFO = "info"           # 信息：仅供参考（通过的检查）


@dataclass
class ValidationIssue:
    """验证问题"""
    level: IssueLevel
    check_name: str           # 检查项名称（如 "过曝检测"）
    category: str             # 分类（如 "视觉质量"）
    message: str              # 问题描述
    value: Optional[float] = None       # 实际值
    threshold: Optional[float] = None   # 阈值
    passed: bool = False      # 是否通过
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'level': self.level.value,
            'check_name': self.check_name,
            'category': self.category,
            'message': self.message,
            'value': self.value,
            'threshold': self.threshold,
            'passed': bool(self.passed),
        }


@dataclass
class ValidationResult:
    """验证结果"""
    passed: bool                                # 整体是否通过
    score: float                                # 综合得分 (0-100)
    issues: List[ValidationIssue] = field(default_factory=list)  # 所有检查项
    details: Dict[str, Any] = field(default_factory=dict)        # 额外详情
    
    @property
    def passed_count(self) -> int:
        return sum(1 for i in self.issues if i.passed)
    
    @property
    def failed_count(self) -> int:
        return sum(1 for i in self.issues if not i.passed)
    
    @property
    def critical_issues(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.level == IssueLevel.CRITICAL and not i.passed]
    
    @property
    def major_issues(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.level == IssueLevel.MAJOR and not i.passed]
    
    def get_category_summary(self) -> Dict[str, Dict[str, int]]:
        summary = {}
        for issue in self.issues:
            if issue.category not in summary:
                summary[issue.category] = {'passed': 0, 'failed': 0}
            if issue.passed:
                summary[issue.category]['passed'] += 1
            else:
                summary[issue.category]['failed'] += 1
        return summary
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'passed': self.passed,
            'score': self.score,
            'passed_count': self.passed_count,
            'failed_count': self.failed_count,
            'issues': [i.to_dict() for i in self.issues],
            'category_summary': self.get_category_summary(),
            'details': self.details,
        }


@dataclass
class ValidatorConfig:
    """
    验证器配置（严格按照 criterion.md）
    
    所有阈值可通过前端 UI 动态配置
    """
    
    # ========== 4.1 视觉数据质量 ==========
    # 基础技术指标
    min_resolution_width: int = 640           # 分辨率 ≥640×480
    min_resolution_height: int = 480
    min_frame_rate: float = 20.0              # 帧率：推荐30Hz，最低20Hz
    recommended_frame_rate: float = 30.0
    frame_rate_tolerance: float = 2.0         # 帧率稳定性 ±2FPS
    
    # 色彩质量
    color_shift_max: float = 0.10             # 色彩偏移 <10%
    white_balance_min: float = 0.95           # 白平衡准确性 >95%
    
    # 曝光检测
    overexposure_ratio_max: float = 0.05      # 过曝像素占比 ≤5%
    underexposure_ratio_max: float = 0.10     # 欠曝像素占比 ≤10%
    
    # 异常图像
    abnormal_black_ratio_max: float = 0.95    # 全黑像素占比 ≤95%
    abnormal_white_ratio_max: float = 0.95    # 全白像素占比 ≤95%
    
    # ========== 4.2 深度数据质量 ==========
    depth_precision_cm: float = 2.0           # 测量精度 ±2cm (1米内)
    depth_error_max: float = 0.02             # 深度误差分布 ≤2%
    depth_invalid_pixel_max: float = 0.10     # 无效像素 ≤10%
    depth_continuity_min: float = 0.90        # 深度图连续性 >90%
    
    # ========== 4.3 语言指令质量 ==========
    instruction_min_words: int = 3            # 长度范围: 3-50词
    instruction_max_words: int = 50
    instruction_avg_min: int = 8              # 平均 8-20 词
    instruction_avg_max: int = 20
    instruction_ambiguity_threshold: float = 0.95  # 歧义检测通过率 ≥95%
    
    # ========== 4.4 动作数据质量 ==========
    # 采样精度
    min_sampling_rate: float = 60.0           # 最低采样率 30Hz
    recommended_sampling_rate: float = 60.0   # 推荐采样率 60Hz
    sampling_rate_tolerance: float = 0.05     # 频率稳定性 ±5%
    
    # 静止检测
    static_threshold_all: float = 3.0         # 全身异常静止 >3s
    static_threshold_key: float = 5.0         # 关键关节静止 >5s
    static_ratio_max: float = 0.01            # 异常静止占比 <1%
    
    # 数据中断
    data_interrupt_max: float = 1.0           # 采样中断 >1s 视为异常
    
    # 安全性验证
    max_joint_velocity: float = 3.14          # 速度限制 <180°/s ≈ 3.14 rad/s
    grasp_threshold: float = 0.5              # 抓取判定阈值
    machine_id: str = "gr3"                   # 机器人型号 (gr2/gr3)
    
    # ========== 4.5 时间同步质量 ==========
    timestamp_monotonic_min: float = 0.99     # 时序完整性 ≥99%
    timestamp_gap_tolerance: float = 0.10     # 采样间隔一致性 <10%
    timestamp_stability_min: float = 0.90     # 间隔稳定性 ≥90%
    frequency_tolerance: float = 0.10         # 频率波动 <±10%
    frequency_consistency_min: float = 0.90   # 频率一致性 >90%
    
    # ========== 5. 分层质检 ==========
    single_data_pass_rate: float = 0.98       # 单条数据合格率 ≥98%
    invalid_multimodal_max: float = 0.02      # 无效多模态数据 <2%
    invalid_atomic_skill_max: float = 0.02    # 无效原子技能占比 <2%
    smoothness_threshold: float = 0.05        # 动作平滑度不合格 <5%
    
    # ========== 4.6 时长过滤 ==========
    duration_min: Optional[float] = None      # 最小允许时长 (秒)
    duration_max: Optional[float] = None      # 最大允许时长 (秒)
    duration_percentile_min: Optional[int] = None # 最小允许时长百分位 (0-100)
    duration_percentile_max: Optional[int] = None # 最大允许时长百分位 (0-100)

    # ========== 运行时开关 ==========
    enable_cross_episode_checks: bool = True  # 是否启用跨 Episode 对比类检查（参考轨迹等）
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，方便UI展示"""
        return {
            'visual': {
                'min_resolution_width': self.min_resolution_width,
                'min_resolution_height': self.min_resolution_height,
                'min_frame_rate': self.min_frame_rate,
                'recommended_frame_rate': self.recommended_frame_rate,
                'frame_rate_tolerance': self.frame_rate_tolerance,
                'color_shift_max': self.color_shift_max,
                'overexposure_ratio_max': self.overexposure_ratio_max,
                'underexposure_ratio_max': self.underexposure_ratio_max,
                'abnormal_black_ratio_max': self.abnormal_black_ratio_max,
                'abnormal_white_ratio_max': self.abnormal_white_ratio_max,
            },
            'depth': {
                'depth_precision_cm': self.depth_precision_cm,
                'depth_error_max': self.depth_error_max,
                'depth_invalid_pixel_max': self.depth_invalid_pixel_max,
                'depth_continuity_min': self.depth_continuity_min,
            },
            'action': {
                'min_sampling_rate': self.min_sampling_rate,
                'recommended_sampling_rate': self.recommended_sampling_rate,
                'sampling_rate_tolerance': self.sampling_rate_tolerance,
                'static_threshold_all': self.static_threshold_all,
                'static_threshold_key': self.static_threshold_key,
                'static_ratio_max': self.static_ratio_max,
                'data_interrupt_max': self.data_interrupt_max,
                'max_joint_velocity': self.max_joint_velocity,
            },
            'timing': {
                'timestamp_monotonic_min': self.timestamp_monotonic_min,
                'timestamp_gap_tolerance': self.timestamp_gap_tolerance,
                'timestamp_stability_min': self.timestamp_stability_min,
                'frequency_tolerance': self.frequency_tolerance,
                'frequency_consistency_min': self.frequency_consistency_min,
            },
            'layered': {
                'single_data_pass_rate': self.single_data_pass_rate,
                'invalid_multimodal_max': self.invalid_multimodal_max,
                'invalid_atomic_skill_max': self.invalid_atomic_skill_max,
                'smoothness_threshold': self.smoothness_threshold,
            },
            'runtime': {
                'enable_cross_episode_checks': self.enable_cross_episode_checks,
            }
        }


class BaseValidator(ABC):
    """验证器基类"""
    
    def __init__(self, config: Optional[ValidatorConfig] = None):
        self.config = config or ValidatorConfig()
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @property
    @abstractmethod
    def category(self) -> str:
        pass
    
    @abstractmethod
    def validate(self, episode_id: str, data_path: str) -> ValidationResult:
        pass
    
    def _create_issue(
        self,
        check_name: str,
        message: str,
        passed: bool,
        level: IssueLevel = IssueLevel.MAJOR,
        value: Optional[float] = None,
        threshold: Optional[float] = None,
    ) -> ValidationIssue:
        return ValidationIssue(
            level=level,
            check_name=check_name,
            category=self.category,
            message=message,
            value=value,
            threshold=threshold,
            passed=passed,
        )
