"""PnP 抓取质检服务。"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from .validators import (
    EEActionValidator,
    IssueLevel,
    ValidationResult,
    ValidatorConfig,
    extract_minimum_grasp_counts,
)

HAND_KEYS = ("right", "left")
LEVEL_PRIORITY = {
    IssueLevel.CRITICAL: 4,
    IssueLevel.MAJOR: 3,
    IssueLevel.MINOR: 2,
    IssueLevel.INFO: 1,
}
LEVEL_MESSAGES = {
    IssueLevel.CRITICAL: "小于最小抓取次数，判定为任务失败",
    IssueLevel.MAJOR: "抓取次数过多，判定为抓取异常",
    IssueLevel.MINOR: "抓取时机异常，判定为抓取行为次优",
    IssueLevel.INFO: "抓取行为正常",
}


def _calc_iqr_bounds(
    values: Iterable[Optional[float]],
    multiplier: float,
) -> Tuple[Optional[float], Optional[float]]:
    arr = [float(value) for value in values if value is not None and float(value) > 0]
    if len(arr) < 4:
        return None, None

    series = pd.Series(arr)
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    if iqr <= 0:
        return None, None
    return q1 - multiplier * iqr, q3 + multiplier * iqr


def _calc_sigma_bounds(
    values: Iterable[Optional[float]],
    sigma_k: float,
) -> Tuple[Optional[float], Optional[float]]:
    arr = [float(value) for value in values if value is not None and float(value) > 0]
    if len(arr) < 2:
        return None, None

    series = pd.Series(arr)
    mean_v = float(series.mean())
    std_v = float(series.std(ddof=0))
    if std_v <= 0:
        return None, None

    low = max(0.0, mean_v - sigma_k * std_v)
    high = min(1.0, mean_v + sigma_k * std_v)
    if high <= low:
        return None, None
    return low, high


def _classify_outlier(
    value: Optional[float],
    low: Optional[float],
    high: Optional[float],
) -> str:
    if value is None or value <= 0:
        return "none"
    if low is None or high is None:
        return "normal"
    if value < low:
        return "low"
    if value > high:
        return "high"
    return "normal"


def _student_t_sigma_multiplier(sample_size: int, sigma_k: float) -> Optional[float]:
    if sample_size <= 1 or sigma_k <= 0:
        return None
    if sample_size > 200:
        return float(sigma_k)

    z_value = float(sigma_k)
    df = float(sample_size - 1)
    z2 = z_value * z_value
    z3 = z2 * z_value
    z5 = z3 * z2
    z7 = z5 * z2

    term1 = (z3 + z_value) / (4.0 * df)
    term2 = (5.0 * z5 + 16.0 * z3 + 3.0 * z_value) / (96.0 * df * df)
    term3 = (3.0 * z7 + 19.0 * z5 + 17.0 * z3 - 15.0 * z_value) / (
        384.0 * df * df * df
    )
    return max(z_value, z_value + term1 + term2 + term3)


def _calc_count_upper_bound(
    values: Iterable[Optional[float]],
    sigma_k: float,
) -> Dict[str, Optional[float]]:
    arr = [float(value) for value in values if value is not None]
    if len(arr) < 4:
        return {
            "sample_size": len(arr),
            "mean": None,
            "std": None,
            "multiplier": None,
            "upper_bound": None,
        }

    series = pd.Series(arr)
    mean_v = float(series.mean())
    std_v = float(series.std(ddof=1))
    if std_v <= 0:
        return {
            "sample_size": len(arr),
            "mean": mean_v,
            "std": std_v,
            "multiplier": None,
            "upper_bound": None,
        }

    multiplier = _student_t_sigma_multiplier(len(arr), sigma_k)
    upper_bound = None if multiplier is None else mean_v + multiplier * std_v
    return {
        "sample_size": len(arr),
        "mean": mean_v,
        "std": std_v,
        "multiplier": multiplier,
        "upper_bound": upper_bound,
    }


class AcceptanceService:
    """组合验证器结果并生成最终 QC 结论。"""

    def __init__(self, config: Optional[ValidatorConfig] = None):
        self.config = config or ValidatorConfig()
        self.validator = EEActionValidator(self.config)

    def build_task_context(
        self,
        task_description_en: str,
        detection_results: List[ValidationResult],
    ) -> Dict[str, Any]:
        minimum_required_grasps = extract_minimum_grasp_counts(task_description_en)
        hand_buckets: Dict[str, Dict[str, List[Optional[float]]]] = {
            hand: {"count": [], "duration": [], "axis_points": []} for hand in HAND_KEYS
        }

        for detection_result in detection_results:
            details = detection_result.details or {}
            hands = details.get("hands") or {}
            for hand in HAND_KEYS:
                hand_metrics = hands.get(hand) or {}
                hand_buckets[hand]["count"].append(hand_metrics.get("count"))
                hand_buckets[hand]["duration"].append(hand_metrics.get("duration_ratio"))
                hand_buckets[hand]["axis_points"].extend(hand_metrics.get("axis_points") or [])

        hand_stats: Dict[str, Dict[str, Any]] = {}
        for hand in HAND_KEYS:
            duration_low, duration_high = _calc_iqr_bounds(
                hand_buckets[hand]["duration"],
                self.config.duration_iqr_multiplier,
            )
            axis_low, axis_high = _calc_sigma_bounds(
                hand_buckets[hand]["axis_points"],
                self.config.axis_sigma_k,
            )
            hand_stats[hand] = {
                "minimum_required_count": minimum_required_grasps[hand],
                "count_stats": _calc_count_upper_bound(
                    hand_buckets[hand]["count"],
                    self.config.count_sigma_k,
                ),
                "duration_bounds": {
                    "low": duration_low,
                    "high": duration_high,
                },
                "axis_bounds": {
                    "low": axis_low,
                    "high": axis_high,
                },
            }

        return {
            "task_description_en": task_description_en,
            "minimum_required_grasps": minimum_required_grasps,
            "hand_stats": hand_stats,
        }

    def _ensure_detection_result(self, episode_id: str, payload: Dict[str, Any]) -> ValidationResult:
        existing_result = payload.get("validation_result")
        if isinstance(existing_result, ValidationResult):
            return existing_result
        return self.validator.validate(episode_id, payload)

    def _evaluate_hand(
        self,
        hand: str,
        hand_metrics: Dict[str, Any],
        task_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        hand_stats = task_context.get("hand_stats", {}).get(hand, {})
        count_stats = hand_stats.get("count_stats", {})
        duration_bounds = hand_stats.get("duration_bounds", {})
        axis_bounds = hand_stats.get("axis_bounds", {})

        count_val = int(hand_metrics.get("count") or 0)
        duration_ratio = hand_metrics.get("duration_ratio")
        axis_score = hand_metrics.get("axis_score") or 0.0

        duration_tag = _classify_outlier(
            duration_ratio,
            duration_bounds.get("low"),
            duration_bounds.get("high"),
        )
        axis_tag = _classify_outlier(
            axis_score,
            axis_bounds.get("low"),
            axis_bounds.get("high"),
        )

        minimum_required_count = int(hand_stats.get("minimum_required_count") or 0)
        upper_bound = count_stats.get("upper_bound")

        level = IssueLevel.INFO
        reason = "normal"
        if count_val < minimum_required_count:
            level = IssueLevel.CRITICAL
            reason = "minimum_required_count"
        elif (
            self.config.enable_cross_episode_checks
            and upper_bound is not None
            and float(count_val) > float(upper_bound)
        ):
            level = IssueLevel.MAJOR
            reason = "count_right_tail_outlier"
        elif self.config.enable_cross_episode_checks and (
            duration_tag in {"low", "high"} or axis_tag in {"low", "high"}
        ):
            level = IssueLevel.MINOR
            reason = "timing_outlier"

        return {
            **hand_metrics,
            "minimum_required_count": minimum_required_count,
            "count_mean": count_stats.get("mean"),
            "count_std": count_stats.get("std"),
            "count_sigma_multiplier": count_stats.get("multiplier"),
            "count_upper_bound": upper_bound,
            "duration_tag": duration_tag,
            "axis_tag": axis_tag,
            "level": level.value,
            "message": LEVEL_MESSAGES[level],
            "reason": reason,
        }

    def validate_episode(
        self,
        episode_id: str,
        payload: Dict[str, Any],
        task_context: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        detection_result = self._ensure_detection_result(episode_id, payload)
        effective_task_context = task_context or self.build_task_context(
            str(payload.get("task_description_en") or detection_result.details.get("task_description_en") or "Unknown task"),
            [detection_result],
        )

        detection_details = dict(detection_result.details or {})
        raw_hands = detection_details.get("hands") or {}
        hand_results = {
            hand: self._evaluate_hand(
                hand=hand,
                hand_metrics=raw_hands.get(hand) or {},
                task_context=effective_task_context,
            )
            for hand in HAND_KEYS
        }

        issue_level = IssueLevel.INFO
        for hand_result in hand_results.values():
            candidate_level = IssueLevel(str(hand_result["level"]))
            if LEVEL_PRIORITY[candidate_level] > LEVEL_PRIORITY[issue_level]:
                issue_level = candidate_level

        issue_passed = issue_level != IssueLevel.CRITICAL
        issue_value = None
        issue_threshold = None
        if issue_level == IssueLevel.CRITICAL:
            blockers = [
                hand_result
                for hand_result in hand_results.values()
                if hand_result["level"] == IssueLevel.CRITICAL.value
            ]
            if blockers:
                issue_value = float(min(item["count"] for item in blockers))
                issue_threshold = float(
                    max(item["minimum_required_count"] for item in blockers)
                )
        elif issue_level == IssueLevel.MAJOR:
            majors = [
                hand_result
                for hand_result in hand_results.values()
                if hand_result["level"] == IssueLevel.MAJOR.value
            ]
            if majors:
                issue_value = float(max(item["count"] for item in majors))
                thresholds = [
                    float(item["count_upper_bound"])
                    for item in majors
                    if item.get("count_upper_bound") is not None
                ]
                if thresholds:
                    issue_threshold = max(thresholds)

        details = {
            **detection_details,
            "validator_name": self.validator.name,
            "check_name": "抓取检测",
            "issue_level": issue_level.value,
            "minimum_required_grasps": effective_task_context.get("minimum_required_grasps", {}),
            "right_pnp_result": hand_results["right"].get("segments", []),
            "left_pnp_result": hand_results["left"].get("segments", []),
            "r_count": hand_results["right"].get("count", 0),
            "l_count": hand_results["left"].get("count", 0),
            "r_duration": hand_results["right"].get("duration_ratio"),
            "l_duration": hand_results["left"].get("duration_ratio"),
            "r_duration_tag": hand_results["right"].get("duration_tag"),
            "l_duration_tag": hand_results["left"].get("duration_tag"),
            "r_axis_score": hand_results["right"].get("axis_score", 0.0),
            "l_axis_score": hand_results["left"].get("axis_score", 0.0),
            "r_axis_tag": hand_results["right"].get("axis_tag"),
            "l_axis_tag": hand_results["left"].get("axis_tag"),
            "hands": hand_results,
        }

        issue = self.validator._create_issue(
            check_name="抓取检测",
            message=LEVEL_MESSAGES[issue_level],
            passed=issue_passed,
            level=issue_level,
            value=issue_value,
            threshold=issue_threshold,
        )
        return ValidationResult(
            passed=issue_passed,
            score=None,
            issues=[issue],
            details=details,
        )

    def validate_batch(
        self,
        task_description_en: str,
        episodes: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        detection_rows = []
        for payload in episodes:
            episode_id = str(payload.get("episode_id", ""))
            if not episode_id:
                continue
            detection_result = self._ensure_detection_result(episode_id, payload)
            detection_rows.append(
                {
                    "episode_id": episode_id,
                    "payload": payload,
                    "detection_result": detection_result,
                }
            )

        task_context = self.build_task_context(
            task_description_en=task_description_en,
            detection_results=[item["detection_result"] for item in detection_rows],
        )

        batch_results = []
        for item in detection_rows:
            result = self.validate_episode(
                episode_id=item["episode_id"],
                payload={
                    **item["payload"],
                    "validation_result": item["detection_result"],
                    "task_description_en": task_description_en,
                },
                task_context=task_context,
            )
            batch_results.append(
                {
                    "episode_id": item["episode_id"],
                    "result": result,
                    "stream_summary": self.build_stream_summary(result),
                }
            )

        return batch_results

    def build_stream_summary(self, result: ValidationResult) -> Dict[str, Any]:
        details = result.details or {}
        hands = details.get("hands") or {}
        return {
            "validator_name": details.get("validator_name"),
            "category": self.validator.category,
            "check_name": details.get("check_name"),
            "passed": bool(result.passed),
            "issue_level": details.get("issue_level"),
            "r_count": details.get("r_count"),
            "l_count": details.get("l_count"),
            "r_duration_tag": details.get("r_duration_tag"),
            "l_duration_tag": details.get("l_duration_tag"),
            "r_axis_tag": details.get("r_axis_tag"),
            "l_axis_tag": details.get("l_axis_tag"),
            "minimum_required_grasps": details.get("minimum_required_grasps"),
            "hand_levels": {
                "right": (hands.get("right") or {}).get("level"),
                "left": (hands.get("left") or {}).get("level"),
            },
        }
