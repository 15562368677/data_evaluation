"""PnP 抓取质检服务。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .validators import (
    ActionValidator,
    EEActionValidator,
    IssueLevel,
    ValidationResult,
    ValidatorConfig,
)


def _build_compact_action_details(details: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "validator_name": details.get("validator_name"),
        "category": details.get("category"),
        "check_name": details.get("check_name"),
        "frame_count": details.get("frame_count"),
        "joint_count": details.get("joint_count"),
        "format": details.get("format"),
        "fps": details.get("fps"),
        "all_static_duration": details.get("all_static_duration"),
        "key_static_duration": details.get("key_static_duration"),
        "max_velocity": details.get("max_velocity"),
        "unsafe_joint_count": details.get("unsafe_joint_count"),
        "duration": details.get("duration"),
        "nan_count": details.get("nan_count"),
        "nan_ratio": details.get("nan_ratio"),
    }


class AcceptanceService:
    """组合验证器结果并生成最终 QC 结论。"""

    def __init__(self, config: Optional[ValidatorConfig] = None):
        self.config = config or ValidatorConfig()
        self.validator = EEActionValidator(self.config)
        self.action_validator = ActionValidator(self.config)

    def _combine_results(
        self,
        ee_result: ValidationResult,
        action_result: ValidationResult,
    ) -> ValidationResult:
        ee_details = dict(ee_result.details or {})
        action_details = _build_compact_action_details(action_result.details or {})

        combined_issues = ee_result.issues + action_result.issues
        combined_passed = all(
            item.passed
            for item in combined_issues
            if item.level in (IssueLevel.CRITICAL, IssueLevel.MAJOR)
        )
        combined_score = (
            round(sum(1 for item in combined_issues if item.passed) / len(combined_issues) * 100.0, 1)
            if combined_issues
            else None
        )

        return ValidationResult(
            passed=combined_passed,
            score=combined_score,
            issues=combined_issues,
            details={
                **ee_details,
                "validator_name": "统一质检",
                "check_name": "统一质检",
                "category_results": {
                    self.validator.category: ee_details,
                    self.action_validator.category: action_details,
                },
            },
        )

    def validate_episode(
        self,
        episode_id: str,
        payload: Dict[str, Any],
    ) -> ValidationResult:
        ee_result = self.validator.validate(episode_id)
        action_result = self.action_validator.validate(episode_id, payload)
        return self._combine_results(ee_result, action_result)

    def validate_batch(self, episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        payload_map = {
            str(payload.get("episode_id", "")): payload
            for payload in episodes
            if str(payload.get("episode_id", ""))
        }
        batch_results = []
        for episode_id in payload_map.keys():
            ee_result = self.validator.validate(episode_id)
            action_result = self.action_validator.validate(
                episode_id,
                payload_map.get(episode_id, {}),
            )
            result = self._combine_results(ee_result, action_result)
            batch_results.append(
                {
                    "episode_id": episode_id,
                    "result": result,
                    "stream_summary": self.validator.build_stream_summary(ee_result),
                }
            )

        return batch_results

    def build_stream_summary(self, result: ValidationResult) -> Dict[str, Any]:
        return self.validator.build_stream_summary(result)
