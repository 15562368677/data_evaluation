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


@dataclass
class ValidatorConfig:
    """验证器配置。"""

    pick_closure_threshold: float = 0.35
    pick_start_offset: int = -5
    place_closure_threshold: float = 0.35
    place_velocity_threshold: float = -0.02
    place_velocity_lookback: int = 5
    place_velocity_lookahead: int = 0
    place_diff_lookahead: int = 10
    place_end_offset: int = 5
    negative_diff_threshold: float = -0.08
    positive_diff_threshold: float = 0.05
    min_joints_for_diff: int = 2
    slope_threshold: float = 0.0005
    slope_lookahead: int = 10

    count_sigma_k: float = 3.0
    duration_iqr_multiplier: float = 1.5
    axis_sigma_k: float = 3.0

    right_hand_fingers: tuple[str, ...] = (
        "R_pinky_proximal_joint",
        "R_ring_proximal_joint",
        "R_middle_proximal_joint",
        "R_index_proximal_joint",
        "R_thumb_proximal_pitch_joint",
    )
    left_hand_fingers: tuple[str, ...] = (
        "L_pinky_proximal_joint",
        "L_ring_proximal_joint",
        "L_middle_proximal_joint",
        "L_index_proximal_joint",
        "L_thumb_proximal_pitch_joint",
    )
    right_joint_direction_coefficients: Dict[str, float] = field(
        default_factory=_default_right_joint_direction_coefficients
    )
    left_joint_direction_coefficients: Dict[str, float] = field(
        default_factory=_default_left_joint_direction_coefficients
    )

    enable_cross_episode_checks: bool = True

    def get_hand_fingers(self, hand: str) -> List[str]:
        hand_name = str(hand).lower()
        if hand_name == "right":
            return list(self.right_hand_fingers)
        if hand_name == "left":
            return list(self.left_hand_fingers)
        raise ValueError(f"Unsupported hand: {hand}")

    def get_hand_direction_coefficients(self, hand: str) -> Dict[str, float]:
        hand_name = str(hand).lower()
        if hand_name == "right":
            return dict(self.right_joint_direction_coefficients)
        if hand_name == "left":
            return dict(self.left_joint_direction_coefficients)
        raise ValueError(f"Unsupported hand: {hand}")

    def get_hand_detection_config(self, hand: str) -> Dict[str, Any]:
        return {
            **self.to_pnp_detection_params(),
            "hand": str(hand).lower(),
            "finger_joints": self.get_hand_fingers(hand),
            "joint_direction_coefficients": self.get_hand_direction_coefficients(hand),
        }

    def get_all_hand_joints(self) -> List[str]:
        return self.get_hand_fingers("right") + self.get_hand_fingers("left")

    def to_pnp_detection_params(self) -> Dict[str, Any]:
        return {
            "pick_closure_threshold": self.pick_closure_threshold,
            "pick_start_offset": self.pick_start_offset,
            "place_closure_threshold": self.place_closure_threshold,
            "place_velocity_threshold": self.place_velocity_threshold,
            "place_velocity_lookback": self.place_velocity_lookback,
            "place_velocity_lookahead": self.place_velocity_lookahead,
            "place_diff_lookahead": self.place_diff_lookahead,
            "place_end_offset": self.place_end_offset,
            "negative_diff_threshold": self.negative_diff_threshold,
            "positive_diff_threshold": self.positive_diff_threshold,
            "min_joints_for_diff": self.min_joints_for_diff,
            "slope_threshold": self.slope_threshold,
            "slope_lookahead": self.slope_lookahead,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pnp_detection": self.to_pnp_detection_params(),
            "ee_validation": {
                "count_sigma_k": self.count_sigma_k,
                "duration_iqr_multiplier": self.duration_iqr_multiplier,
                "axis_sigma_k": self.axis_sigma_k,
                "enable_cross_episode_checks": self.enable_cross_episode_checks,
            },
            "hand_config": {
                "right": {
                    "finger_joints": list(self.right_hand_fingers),
                    "joint_direction_coefficients": self.right_joint_direction_coefficients,
                },
                "left": {
                    "finger_joints": list(self.left_hand_fingers),
                    "joint_direction_coefficients": self.left_joint_direction_coefficients,
                },
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
    def validate(self, episode_id: str, data: Dict[str, Any]) -> ValidationResult:
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
