"""验证器基类与通用类型。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class IssueLevel(Enum):
    """问题等级。"""

    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"
    INFO = "info"


@dataclass
class ValidationIssue:
    """单个检查项结果。"""

    level: IssueLevel
    check_name: str
    category: str
    message: str
    value: Optional[float] = None
    threshold: Optional[float] = None
    passed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level.value,
            "check_name": self.check_name,
            "category": self.category,
            "message": self.message,
            "value": self.value,
            "threshold": self.threshold,
            "passed": bool(self.passed),
        }


@dataclass
class ValidationResult:
    """验证结果。"""

    passed: bool
    score: Optional[float] = None
    issues: List[ValidationIssue] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed_count(self) -> int:
        return sum(1 for issue in self.issues if issue.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for issue in self.issues if not issue.passed)

    @property
    def critical_issues(self) -> List[ValidationIssue]:
        return [
            issue
            for issue in self.issues
            if issue.level == IssueLevel.CRITICAL and not issue.passed
        ]

    @property
    def major_issues(self) -> List[ValidationIssue]:
        return [
            issue
            for issue in self.issues
            if issue.level == IssueLevel.MAJOR and not issue.passed
        ]

    def get_category_summary(self) -> Dict[str, Dict[str, int]]:
        summary: Dict[str, Dict[str, int]] = {}
        for issue in self.issues:
            bucket = summary.setdefault(issue.category, {"passed": 0, "failed": 0})
            if issue.passed:
                bucket["passed"] += 1
            else:
                bucket["failed"] += 1
        return summary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "score": self.score,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "issues": [issue.to_dict() for issue in self.issues],
            "category_summary": self.get_category_summary(),
            "details": self.details,
        }


def _default_right_joint_direction_coefficients() -> Dict[str, float]:
    return {
        "R_pinky_proximal_joint": -1.0,
        "R_ring_proximal_joint": -1.0,
        "R_middle_proximal_joint": -1.0,
        "R_index_proximal_joint": -1.0,
        "R_thumb_proximal_pitch_joint": 1.0,
    }


def _default_left_joint_direction_coefficients() -> Dict[str, float]:
    return {
        "L_pinky_proximal_joint": -1.0,
        "L_ring_proximal_joint": -1.0,
        "L_middle_proximal_joint": -1.0,
        "L_index_proximal_joint": -1.0,
        "L_thumb_proximal_pitch_joint": 1.0,
    }


def _default_hand_config() -> Dict[str, Dict[str, Any]]:
    return {
        "right": {
            "finger_joints": [
                "R_pinky_proximal_joint",
                "R_ring_proximal_joint",
                "R_middle_proximal_joint",
                "R_index_proximal_joint",
                "R_thumb_proximal_pitch_joint",
            ],
            "joint_direction_coefficients": _default_right_joint_direction_coefficients(),
        },
        "left": {
            "finger_joints": [
                "L_pinky_proximal_joint",
                "L_ring_proximal_joint",
                "L_middle_proximal_joint",
                "L_index_proximal_joint",
                "L_thumb_proximal_pitch_joint",
            ],
            "joint_direction_coefficients": _default_left_joint_direction_coefficients(),
        },
    }


@dataclass
class ValidatorConfig:
    """
    验证器配置（严格按照 criterion.md）

    所有阈值可通过前端 UI 动态配置
    """

    # ========== PnP 检测参数 ==========
    pick_closure_threshold: float = 0.35      # 抓取闭合阈值：手指闭合度超过该值视为 pick 开始
    pick_start_offset: int = -5               # 抓取起点回看帧数：在闭合触发点前额外回溯的帧数
    place_closure_threshold: float = 0.35     # 放置闭合阈值：用于判定 place 阶段手部仍处于抓持状态
    place_velocity_threshold: float = -0.02   # 放置速度阈值：手部张开速度低于该值视为释放信号
    place_velocity_lookback: int = 5          # 放置速度回看窗口：计算释放速度时向前参考的帧数
    place_velocity_lookahead: int = 0         # 放置速度前看窗口：计算释放速度时向后参考的帧数
    place_diff_lookahead: int = 10            # 放置差分前看窗口：检测放置趋势时向后比较的帧数
    place_end_offset: int = 5                 # 放置终点补偿帧数：在检测到 place 后向后扩展的帧数
    min_segment_duration_seconds: float = 0.5 # 最短有效片段时长：低于该时长的 PnP 片段会被过滤
    negative_diff_threshold: float = -0.08    # 负向差分阈值：用于识别放置/松手阶段的下降变化
    positive_diff_threshold: float = 0.05     # 正向差分阈值：用于识别抓取/闭合阶段的上升变化
    min_joints_for_diff: int = 2              # 最少有效关节数：参与差分判断的最小关节数量
    slope_threshold: float = 0.0005           # 斜率阈值：用于判断动作趋势是否足够明显
    slope_lookahead: int = 10                 # 斜率前看窗口：计算趋势斜率时使用的后续帧数
    hand_config: Dict[str, Dict[str, Any]] = field(default_factory=_default_hand_config)  # 手部关节配置：左右手检测所需关节名与方向系数

    # ========== 4.1 视觉数据质量 ==========
    min_resolution_width: int = 640
    min_resolution_height: int = 480
    min_frame_rate: float = 20.0
    recommended_frame_rate: float = 30.0
    frame_rate_tolerance: float = 2.0
    color_shift_max: float = 0.10
    white_balance_min: float = 0.95
    overexposure_ratio_max: float = 0.05
    underexposure_ratio_max: float = 0.10
    abnormal_black_ratio_max: float = 0.95
    abnormal_white_ratio_max: float = 0.95

    # ========== 4.2 深度数据质量 ==========
    depth_precision_cm: float = 2.0
    depth_error_max: float = 0.02
    depth_invalid_pixel_max: float = 0.10
    depth_continuity_min: float = 0.90

    # ========== 4.3 语言指令质量 ==========
    instruction_min_words: int = 3
    instruction_max_words: int = 50
    instruction_avg_min: int = 8
    instruction_avg_max: int = 20
    instruction_ambiguity_threshold: float = 0.95

    # ========== 4.4 动作数据质量 ==========
    min_sampling_rate: float = 60.0
    recommended_sampling_rate: float = 60.0
    sampling_rate_tolerance: float = 0.05
    static_threshold_all: float = 3.0
    static_threshold_key: float = 5.0
    static_ratio_max: float = 0.01
    static_diff_threshold: float = 0.001
    data_interrupt_max: float = 1.0
    max_joint_velocity: float = 3.14
    grasp_threshold: float = 0.5
    machine_id: str = "gr3"
    min_action_duration: float = 1.0
    max_nan_ratio: float = 0.01

    # ========== 4.5 时间同步质量 ==========
    timestamp_monotonic_min: float = 0.99
    timestamp_gap_tolerance: float = 0.10
    timestamp_stability_min: float = 0.90
    frequency_tolerance: float = 0.10
    frequency_consistency_min: float = 0.90

    # ========== 5. 分层质检 ==========
    single_data_pass_rate: float = 0.98
    invalid_multimodal_max: float = 0.02
    invalid_atomic_skill_max: float = 0.02
    smoothness_threshold: float = 0.05

    # ========== 4.6 时长过滤 ==========
    duration_min: Optional[float] = None
    duration_max: Optional[float] = None
    duration_percentile_min: Optional[int] = None
    duration_percentile_max: Optional[int] = None

    # ========== 运行时开关 ==========
    enable_cross_episode_checks: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "visual": {
                "min_resolution_width": self.min_resolution_width,
                "min_resolution_height": self.min_resolution_height,
                "min_frame_rate": self.min_frame_rate,
                "recommended_frame_rate": self.recommended_frame_rate,
                "frame_rate_tolerance": self.frame_rate_tolerance,
                "color_shift_max": self.color_shift_max,
                "white_balance_min": self.white_balance_min,
                "overexposure_ratio_max": self.overexposure_ratio_max,
                "underexposure_ratio_max": self.underexposure_ratio_max,
                "abnormal_black_ratio_max": self.abnormal_black_ratio_max,
                "abnormal_white_ratio_max": self.abnormal_white_ratio_max,
            },
            "depth": {
                "depth_precision_cm": self.depth_precision_cm,
                "depth_error_max": self.depth_error_max,
                "depth_invalid_pixel_max": self.depth_invalid_pixel_max,
                "depth_continuity_min": self.depth_continuity_min,
            },
            "language": {
                "instruction_min_words": self.instruction_min_words,
                "instruction_max_words": self.instruction_max_words,
                "instruction_avg_min": self.instruction_avg_min,
                "instruction_avg_max": self.instruction_avg_max,
                "instruction_ambiguity_threshold": self.instruction_ambiguity_threshold,
            },
            "pnp_detection": {
                "pick_closure_threshold": self.pick_closure_threshold,
                "pick_start_offset": self.pick_start_offset,
                "place_closure_threshold": self.place_closure_threshold,
                "place_velocity_threshold": self.place_velocity_threshold,
                "place_velocity_lookback": self.place_velocity_lookback,
                "place_velocity_lookahead": self.place_velocity_lookahead,
                "place_diff_lookahead": self.place_diff_lookahead,
                "place_end_offset": self.place_end_offset,
                "min_segment_duration_seconds": self.min_segment_duration_seconds,
                "negative_diff_threshold": self.negative_diff_threshold,
                "positive_diff_threshold": self.positive_diff_threshold,
                "min_joints_for_diff": self.min_joints_for_diff,
                "slope_threshold": self.slope_threshold,
                "slope_lookahead": self.slope_lookahead,
                "hand_config": self.hand_config,
            },
            "action": {
                "min_sampling_rate": self.min_sampling_rate,
                "recommended_sampling_rate": self.recommended_sampling_rate,
                "sampling_rate_tolerance": self.sampling_rate_tolerance,
                "static_threshold_all": self.static_threshold_all,
                "static_threshold_key": self.static_threshold_key,
                "static_ratio_max": self.static_ratio_max,
                "static_diff_threshold": self.static_diff_threshold,
                "data_interrupt_max": self.data_interrupt_max,
                "max_joint_velocity": self.max_joint_velocity,
                "grasp_threshold": self.grasp_threshold,
                "machine_id": self.machine_id,
                "min_action_duration": self.min_action_duration,
                "max_nan_ratio": self.max_nan_ratio,
            },
            "timing": {
                "timestamp_monotonic_min": self.timestamp_monotonic_min,
                "timestamp_gap_tolerance": self.timestamp_gap_tolerance,
                "timestamp_stability_min": self.timestamp_stability_min,
                "frequency_tolerance": self.frequency_tolerance,
                "frequency_consistency_min": self.frequency_consistency_min,
            },
            "layered": {
                "single_data_pass_rate": self.single_data_pass_rate,
                "invalid_multimodal_max": self.invalid_multimodal_max,
                "invalid_atomic_skill_max": self.invalid_atomic_skill_max,
                "smoothness_threshold": self.smoothness_threshold,
            },
            "duration_filter": {
                "duration_min": self.duration_min,
                "duration_max": self.duration_max,
                "duration_percentile_min": self.duration_percentile_min,
                "duration_percentile_max": self.duration_percentile_max,
            },
            "runtime": {
                "enable_cross_episode_checks": self.enable_cross_episode_checks,
            },
        }


class BaseValidator(ABC):
    """验证器基类。"""

    def __init__(self, config: Optional[ValidatorConfig] = None):
        self.config = config or ValidatorConfig()

    @property
    @abstractmethod
    def name(self) -> str:
        """验证器名称。"""

    @property
    @abstractmethod
    def category(self) -> str:
        """验证器分类。"""

    @abstractmethod
    def validate(
        self,
        episode_id: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """执行验证。"""

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
