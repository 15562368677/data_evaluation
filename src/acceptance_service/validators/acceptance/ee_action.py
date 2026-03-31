"""末端执行器抓取检测验证器。"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.data_parser import HAND_JOINT_NAMES, JOINT_NAMES, load_joint_data

from ..core.base import BaseValidator, IssueLevel, ValidationResult

logger = logging.getLogger(__name__)

HAND_KEYS = ("right", "left")
LEVEL_PRIORITY = {
    IssueLevel.CRITICAL: 4,
    IssueLevel.MAJOR: 3,
    IssueLevel.MINOR: 2,
    IssueLevel.INFO: 1,
}
LEVEL_MESSAGES = {
    IssueLevel.CRITICAL: "左右手均未检测到抓取片段，判定为任务失败",
    IssueLevel.MAJOR: "抓取次数小于识别到的最小抓取次数，判定为抓取异常",
    IssueLevel.MINOR: "抓取次数大于识别到的最小抓取次数 3 次及以上，判定为抓取行为次优",
    IssueLevel.INFO: "抓取行为正常",
}
LEVEL_SCORES = {
    IssueLevel.CRITICAL: 0.0,
    IssueLevel.MAJOR: 50.0,
    IssueLevel.MINOR: 80.0,
    IssueLevel.INFO: 100.0,
}
HAND_PATTERN = re.compile(
    r"use\s+(left|right|both)\s+hands?\s+to\s+([^;.!?]+)",
    flags=re.IGNORECASE,
)
ACTION_PATTERN = re.compile(r"\b(pick|grasp)\b", flags=re.IGNORECASE)


def _normalize_segments(raw_val: Any) -> List[List[float]]:
    if not raw_val:
        return []

    parsed = raw_val
    if isinstance(raw_val, str):
        try:
            parsed = json.loads(raw_val)
        except Exception:
            return []

    if not isinstance(parsed, list):
        return []

    segments: List[List[float]] = []
    for seg in parsed:
        if not isinstance(seg, (list, tuple)) or len(seg) < 2:
            continue
        try:
            start_sec = float(seg[0])
            end_sec = float(seg[1])
        except Exception:
            continue
        if end_sec < start_sec:
            start_sec, end_sec = end_sec, start_sec
        segments.append([start_sec, end_sec])
    return segments


def _calc_total_duration(segments: Iterable[Iterable[float]]) -> float:
    return sum(max(0.0, float(end_sec) - float(start_sec)) for start_sec, end_sec in segments)


def _filter_short_segments(
    segments: Iterable[Iterable[float]],
    min_duration_seconds: float,
) -> List[List[float]]:
    filtered: List[List[float]] = []
    threshold = max(0.0, float(min_duration_seconds))
    for segment in segments:
        if not isinstance(segment, (list, tuple)) or len(segment) < 2:
            continue
        start_sec = float(segment[0])
        end_sec = float(segment[1])
        if end_sec < start_sec:
            start_sec, end_sec = end_sec, start_sec
        if (end_sec - start_sec) < threshold:
            continue
        filtered.append([start_sec, end_sec])
    return filtered


def _calc_axis_ratios(
    segments: Iterable[Iterable[float]],
    episode_duration: Optional[float],
) -> List[float]:
    if episode_duration is None or episode_duration <= 0:
        return []

    ratios: List[float] = []
    for start_sec, end_sec in segments:
        center = (float(start_sec) + float(end_sec)) / 2.0
        ratio = center / float(episode_duration)
        ratios.append(max(0.0, min(1.0, ratio)))
    return ratios


def _estimate_duration_from_segments(*raw_values: Any) -> Optional[float]:
    max_end = 0.0
    for raw_val in raw_values:
        for seg in _normalize_segments(raw_val):
            max_end = max(max_end, float(seg[1]))
    return max_end if max_end > 0 else None


def _resolve_episode_duration_from_row(row: pd.Series) -> Optional[float]:
    trajectory_duration = row.get("trajectory_duration")
    try:
        if trajectory_duration is not None:
            duration_val = float(trajectory_duration)
            if duration_val > 0:
                return duration_val
    except Exception:
        pass

    trajectory_start = row.get("trajectory_start")
    trajectory_end = row.get("trajectory_end")
    if pd.notnull(trajectory_start) and pd.notnull(trajectory_end):
        try:
            duration_val = float((trajectory_end - trajectory_start).total_seconds())
            if duration_val > 0:
                return duration_val
        except Exception:
            return None
    return None


def _minimum_detected_count(values: Iterable[Optional[int]]) -> int:
    # Ignore zero counts so CRITICAL remains dedicated to "both hands missing",
    # while MAJOR/MINOR compare against the current episode's positive baseline.
    positive_counts = [
        int(value)
        for value in values
        if value is not None and int(value) > 0
    ]
    return min(positive_counts) if positive_counts else 0


def _to_duration_seconds(start_val: Any, end_val: Any) -> Optional[float]:
    try:
        delta = end_val - start_val
        if hasattr(delta, "total_seconds"):
            total = float(delta.total_seconds())
        else:
            total = float(delta)
    except Exception:
        return None
    return total if total >= 0 else None


def build_task_en(descriptions: Any) -> str:
    task_en = "Unknown task"
    if not descriptions:
        return task_en

    if isinstance(descriptions, str):
        try:
            descriptions = json.loads(descriptions)
        except json.JSONDecodeError:
            descriptions = {}

    if isinstance(descriptions, dict):
        en_content = descriptions.get("en")
        if en_content is not None:
            if isinstance(en_content, list):
                task_en = " ".join(str(item) for item in en_content if item)
            else:
                task_en = str(en_content)

    return task_en


def extract_minimum_grasp_counts(task_description_en: str) -> Dict[str, int]:
    counts = {"left": 0, "right": 0}
    if not task_description_en:
        return counts

    for hand_token, clause in HAND_PATTERN.findall(task_description_en):
        action_hits = ACTION_PATTERN.findall(clause)
        if not action_hits:
            continue

        increment = len(action_hits)
        hand_name = str(hand_token).lower()
        if hand_name == "both":
            counts["left"] += increment
            counts["right"] += increment
        elif hand_name in counts:
            counts[hand_name] += increment

    return counts


def _build_compact_hand_details(hand_details: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "hand": hand_details.get("hand"),
        "count": hand_details.get("count", 0),
        "segments": hand_details.get("segments", []),
        "duration_ratio": hand_details.get("duration_ratio"),
        "axis_points": hand_details.get("axis_points", []),
        "axis_score": hand_details.get("axis_score", 0.0),
        "duration_tag": hand_details.get("duration_tag", "none"),
        "axis_tag": hand_details.get("axis_tag", "none"),
        "level": hand_details.get("level"),
        "reason": hand_details.get("reason"),
        "message": hand_details.get("message"),
        "minimum_detected_count": hand_details.get("minimum_detected_count", 0),
        "task_required_count": hand_details.get("task_required_count", 0),
        "count_delta_from_minimum": hand_details.get("count_delta_from_minimum"),
    }


def _build_compact_ee_details(
    *,
    task_context: Dict[str, Any],
    hand_results: Dict[str, Dict[str, Any]],
    issue_level: str,
    episode_duration: Optional[float],
    validator_name: str,
    category: str,
    check_name: str,
) -> Dict[str, Any]:
    return {
        "validator_name": validator_name,
        "category": category,
        "check_name": check_name,
        "issue_level": issue_level,
        "task_description_en": task_context.get("task_description_en", "Unknown task"),
        "task_required_grasps": task_context.get("task_required_grasps", {}),
        "minimum_detected_grasps": task_context.get("minimum_detected_grasps", {}),
        "episode_duration": episode_duration,
        "right_pnp_result": hand_results["right"]["segments"],
        "left_pnp_result": hand_results["left"]["segments"],
        "r_count": hand_results["right"]["count"],
        "l_count": hand_results["left"]["count"],
        "r_duration": hand_results["right"]["duration_ratio"],
        "l_duration": hand_results["left"]["duration_ratio"],
        "r_axis_score": hand_results["right"]["axis_score"],
        "l_axis_score": hand_results["left"]["axis_score"],
        "hands": {
            hand: _build_compact_hand_details(hand_results.get(hand) or {})
            for hand in HAND_KEYS
        },
    }


def calculate_closure_degree(
    joint_angles: Dict[str, float],
    finger_joints: List[str],
    direction_coefficients: Dict[str, float],
) -> float:
    valid_joints = [
        (name, angle)
        for name, angle in joint_angles.items()
        if name in finger_joints and not np.isnan(angle)
    ]
    if not valid_joints:
        return np.nan

    weighted_sum = 0.0
    total_weight = 0.0
    for name, angle in valid_joints:
        coeff = direction_coefficients.get(name, 0.0)
        weight = 0.40 if "thumb" in name.lower() else 0.15
        weighted_sum += coeff * angle * weight
        total_weight += weight

    if total_weight > 0:
        return weighted_sum / total_weight
    return 0.0


def calculate_closure_velocity(
    closure_degrees: Iterable[float],
    timestamps: Optional[Iterable[Any]] = None,
) -> np.ndarray:
    closure_array = np.asarray(list(closure_degrees), dtype=float)
    if len(closure_array) < 2:
        return np.array([])

    velocity = np.diff(closure_array)

    if timestamps is not None:
        timestamp_series = pd.Series(list(timestamps))
        if len(timestamp_series) != len(closure_array):
            raise ValueError(
                f"Length mismatch: closure_degrees ({len(closure_array)}) "
                f"vs timestamps ({len(timestamp_series)})"
            )

        time_diff_series = timestamp_series.diff().iloc[1:]
        if pd.api.types.is_timedelta64_dtype(time_diff_series):
            time_diff = (time_diff_series / pd.Timedelta(seconds=1)).to_numpy(dtype=float)
        else:
            time_diff = time_diff_series.to_numpy(dtype=float)
        time_diff = np.where(time_diff == 0, 1.0, time_diff)
        velocity = velocity / time_diff

    return velocity


def calculate_closure_metrics_from_dataframe(
    df: pd.DataFrame,
    finger_joints: List[str],
    direction_coefficients: Dict[str, float],
) -> pd.DataFrame:
    closure_degrees = []
    for _, row in df.iterrows():
        joint_angles = {joint: row[joint] for joint in finger_joints if joint in df.columns}
        closure_degrees.append(
            calculate_closure_degree(joint_angles, finger_joints, direction_coefficients)
        )

    result_df = pd.DataFrame(
        {
            "timestamp_utc": df["timestamp_utc"],
            "closure_degree": closure_degrees,
        }
    )
    velocity = calculate_closure_velocity(closure_degrees, df["timestamp_utc"].tolist())
    result_df["closure_velocity"] = [np.nan] + list(velocity)
    return result_df


def check_joint_diff_with_slope(
    frame_idx: int,
    joint_differences: Dict[str, np.ndarray],
    state_df: pd.DataFrame,
    action_df: pd.DataFrame,
    hand_config: Dict[str, Any],
) -> Tuple[bool, int]:
    finger_joints = hand_config["finger_joints"]
    direction_coefficients = hand_config["joint_direction_coefficients"]
    negative_diff_threshold = hand_config["negative_diff_threshold"]
    positive_diff_threshold = hand_config["positive_diff_threshold"]
    min_joints = hand_config["min_joints_for_diff"]
    slope_threshold = hand_config["slope_threshold"]
    slope_lookahead = hand_config["slope_lookahead"]

    joints_satisfied = 0
    for joint in finger_joints:
        if joint not in joint_differences:
            continue

        diffs = joint_differences[joint]
        coeff = direction_coefficients.get(joint, 0.0)
        if frame_idx >= len(diffs) or np.isnan(diffs[frame_idx]):
            continue

        diff_val = diffs[frame_idx]
        diff_condition_met = False
        if coeff < 0 and diff_val < negative_diff_threshold:
            diff_condition_met = True
        elif coeff > 0 and diff_val > positive_diff_threshold:
            diff_condition_met = True

        if not diff_condition_met:
            continue

        slope_stable = True
        if joint in state_df.columns and joint in action_df.columns:
            start_idx = frame_idx
            end_idx = min(len(state_df) - 1, frame_idx + slope_lookahead)
            if end_idx > start_idx:
                diff_step = end_idx - start_idx
                state_end_val = state_df[joint].iloc[end_idx]
                state_start_val = state_df[joint].iloc[start_idx]

                if not np.isnan(state_end_val) and not np.isnan(state_start_val):
                    state_slope = np.abs(state_end_val - state_start_val) / diff_step
                    if state_slope > slope_threshold:
                        slope_stable = False

                if slope_stable:
                    action_end_val = state_end_val + diffs[end_idx]
                    action_start_val = state_start_val + diffs[start_idx]
                    if not np.isnan(action_end_val) and not np.isnan(action_start_val):
                        action_slope = np.abs(action_end_val - action_start_val) / diff_step
                        if action_slope > slope_threshold:
                            slope_stable = False

        if diff_condition_met and slope_stable:
            joints_satisfied += 1

    return joints_satisfied >= min_joints, joints_satisfied


def check_sufficient_joint_differences(
    frame_idx: int,
    joint_differences: Dict[str, np.ndarray],
    hand_config: Dict[str, Any],
) -> bool:
    return (
        count_joints_satisfying_diff(
            frame_idx=frame_idx,
            joint_differences=joint_differences,
            hand_config=hand_config,
        )
        >= hand_config["min_joints_for_diff"]
    )


def count_joints_satisfying_diff(
    frame_idx: int,
    joint_differences: Dict[str, np.ndarray],
    hand_config: Dict[str, Any],
) -> int:
    finger_joints = hand_config["finger_joints"]
    direction_coefficients = hand_config["joint_direction_coefficients"]
    negative_diff_threshold = hand_config["negative_diff_threshold"]
    positive_diff_threshold = hand_config["positive_diff_threshold"]

    joints_satisfied = 0
    for joint in finger_joints:
        if joint not in joint_differences:
            continue

        diffs = joint_differences[joint]
        coeff = direction_coefficients.get(joint, 0.0)
        if frame_idx >= len(diffs) or np.isnan(diffs[frame_idx]):
            continue

        diff_val = diffs[frame_idx]
        if coeff > 0 and diff_val > positive_diff_threshold:
            joints_satisfied += 1
        elif coeff < 0 and diff_val < negative_diff_threshold:
            joints_satisfied += 1

    return joints_satisfied


class EEActionValidator(BaseValidator):
    """末端执行器动作验证器。"""

    @property
    def name(self) -> str:
        return "末端执行器动作验证器"

    @property
    def category(self) -> str:
        return "末端轨迹"

    def _load_episode_context(self, episode_id: str) -> Dict[str, Any]:
        from src.utils.source_db import query_df

        episode_df = query_df(
            """
            SELECT
                e.id,
                e.task_id,
                e.trajectory_duration,
                e.trajectory_start,
                e.trajectory_end,
                t.descriptions,
                s.file_path
            FROM episodes e
            LEFT JOIN tasks t
                ON t.id = e.task_id
            LEFT JOIN streams s
                ON s.episode_id = e.id
               AND s.stream_name = 'rgb'
            WHERE e.id = %(episode_id)s
            LIMIT 1
            """,
            {"episode_id": episode_id},
        )
        if episode_df.empty:
            raise ValueError(f"Episode {episode_id} not found in source DB.")

        row = episode_df.iloc[0]
        file_path = row.get("file_path")
        if not file_path:
            raise ValueError(f"Episode {episode_id} has no rgb stream file_path.")

        return {
            "episode_id": str(row.get("id")),
            "task_id": str(row.get("task_id") or ""),
            "file_path": str(file_path),
            "task_description_en": build_task_en(row.get("descriptions")),
            "episode_duration": _resolve_episode_duration_from_row(row),
        }

    def _normalize_timestamp_df(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        if "timestamp_utc" in result.columns and not pd.api.types.is_datetime64_any_dtype(result["timestamp_utc"]):
            result["timestamp_utc"] = pd.to_datetime(result["timestamp_utc"], unit="s")
        return result

    def _load_joint_data_as_dfs(self, file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        parsed_data = load_joint_data(file_path)
        if not parsed_data:
            raise ValueError(f"Failed to load joint data from file_path={file_path}")

        all_joints = list(JOINT_NAMES) + list(HAND_JOINT_NAMES)

        state_df = pd.DataFrame({"timestamp_utc": parsed_data.get("absolute_timestamps_state", [])})
        for joint in all_joints:
            if joint in parsed_data.get("state", {}):
                state_df[joint] = parsed_data["state"][joint]
            else:
                state_df[joint] = np.nan

        action_df = pd.DataFrame({"timestamp_utc": parsed_data.get("absolute_timestamps_action", [])})
        for joint in all_joints:
            if joint in parsed_data.get("action", {}):
                action_df[joint] = parsed_data["action"][joint]
            else:
                action_df[joint] = np.nan

        if len(state_df) == 0 or len(action_df) == 0:
            raise ValueError(f"Episode joint data is empty for file_path={file_path}")

        return self._normalize_timestamp_df(state_df), self._normalize_timestamp_df(action_df)

    def _load_validation_data(self, episode_id: str) -> Dict[str, Any]:
        context = self._load_episode_context(episode_id)
        state_df, action_df = self._load_joint_data_as_dfs(context["file_path"])
        return {
            **context,
            "state_df": state_df,
            "action_df": action_df,
        }

    def _build_hand_detection_config(self, hand: str) -> Dict[str, Any]:
        hand_name = str(hand).lower()
        hand_details = dict((self.config.hand_config or {}).get(hand_name) or {})
        finger_joints = hand_details.get("finger_joints")
        direction_coefficients = hand_details.get("joint_direction_coefficients")
        if not isinstance(finger_joints, list) or not isinstance(direction_coefficients, dict):
            raise ValueError(f"Unsupported hand: {hand}")
        return {
            "pick_closure_threshold": self.config.pick_closure_threshold,
            "pick_start_offset": self.config.pick_start_offset,
            "place_closure_threshold": self.config.place_closure_threshold,
            "place_velocity_threshold": self.config.place_velocity_threshold,
            "place_velocity_lookback": self.config.place_velocity_lookback,
            "place_velocity_lookahead": self.config.place_velocity_lookahead,
            "place_diff_lookahead": self.config.place_diff_lookahead,
            "place_end_offset": self.config.place_end_offset,
            "min_segment_duration_seconds": self.config.min_segment_duration_seconds,
            "negative_diff_threshold": self.config.negative_diff_threshold,
            "positive_diff_threshold": self.config.positive_diff_threshold,
            "min_joints_for_diff": self.config.min_joints_for_diff,
            "slope_threshold": self.config.slope_threshold,
            "slope_lookahead": self.config.slope_lookahead,
            "hand": hand_name,
            "finger_joints": list(finger_joints),
            "joint_direction_coefficients": dict(direction_coefficients),
        }

    @classmethod
    def detect_pick_segments(
        cls,
        closure_degrees: np.ndarray,
        closure_velocities: np.ndarray,
        state_action_diffs: Dict[str, np.ndarray],
        hand_config: Dict[str, Any],
        state_df: Optional[pd.DataFrame] = None,
        action_df: Optional[pd.DataFrame] = None,
    ) -> List[Tuple[int, int]]:
        picks: List[Tuple[int, int]] = []
        n_frames = len(closure_degrees)

        in_pick = False
        pick_start: Optional[int] = None

        for i in range(n_frames):
            if np.isnan(closure_degrees[i]):
                continue

            if not in_pick:
                if closure_degrees[i] <= hand_config["pick_closure_threshold"]:
                    continue

                if state_df is not None and action_df is not None:
                    diff_condition, joints_count = check_joint_diff_with_slope(
                        i,
                        state_action_diffs,
                        state_df,
                        action_df,
                        hand_config,
                    )
                else:
                    diff_condition = check_sufficient_joint_differences(
                        i,
                        state_action_diffs,
                        hand_config,
                    )
                    joints_count = count_joints_satisfying_diff(
                        i,
                        state_action_diffs,
                        hand_config,
                    )

                if not diff_condition:
                    continue

                pick_start = max(0, i + hand_config["pick_start_offset"])
                in_pick = True
                logger.debug(
                    "[%s] Pick 开始检测: frame=%s start=%s joints=%s",
                    hand_config["hand"],
                    i,
                    pick_start,
                    joints_count,
                )

            if in_pick:
                open_condition_closure = (
                    closure_degrees[i] < hand_config["place_closure_threshold"]
                )
                open_condition_velocity = False
                velocity_start_idx = max(0, i - hand_config["place_velocity_lookback"])
                velocity_end_idx = min(
                    len(closure_velocities),
                    i + hand_config["place_velocity_lookahead"],
                )
                for j in range(velocity_start_idx, velocity_end_idx):
                    if j < len(closure_velocities) and not np.isnan(closure_velocities[j]):
                        if closure_velocities[j] < hand_config["place_velocity_threshold"]:
                            open_condition_velocity = True
                            break

                if not (open_condition_closure or open_condition_velocity):
                    continue

                diff_condition_place = True
                window_end = min(n_frames, i + hand_config["place_diff_lookahead"])
                for j in range(i, window_end):
                    if state_df is not None and action_df is not None:
                        _, joints_count = check_joint_diff_with_slope(
                            j,
                            state_action_diffs,
                            state_df,
                            action_df,
                            hand_config,
                        )
                    else:
                        joints_count = count_joints_satisfying_diff(
                            j,
                            joint_differences=state_action_diffs,
                            hand_config=hand_config,
                        )

                    if joints_count >= hand_config["min_joints_for_diff"]:
                        diff_condition_place = False
                        break

                if not diff_condition_place:
                    continue

                place_end = i + hand_config["place_end_offset"]
                if pick_start is not None:
                    picks.append((pick_start, place_end))
                in_pick = False
                pick_start = None

        return picks

    def _detect_hand_segments(
        self,
        state_df: pd.DataFrame,
        action_df: pd.DataFrame,
        hand: str,
    ) -> List[List[float]]:
        hand_name = str(hand).lower()
        hand_config = self._build_hand_detection_config(hand_name)
        sorted_state_df = state_df.sort_values("timestamp_utc")
        sorted_action_df = action_df.sort_values("timestamp_utc")
        closure_df = calculate_closure_metrics_from_dataframe(
            sorted_state_df,
            hand_config["finger_joints"],
            hand_config["joint_direction_coefficients"],
        )

        action_columns = [
            "timestamp_utc",
            *[col for col in hand_config["finger_joints"] if col in sorted_action_df.columns],
        ]
        action_subset = sorted_action_df[action_columns]
        merged = pd.merge_asof(
            sorted_state_df,
            action_subset,
            on="timestamp_utc",
            direction="nearest",
            suffixes=("", "_action"),
        )

        joint_diffs: Dict[str, np.ndarray] = {}
        for joint in hand_config["finger_joints"]:
            action_col = f"{joint}_action"
            if joint in merged.columns and action_col in merged.columns:
                joint_diffs[joint] = (merged[action_col] - merged[joint]).to_numpy()
            else:
                joint_diffs[joint] = np.full(len(sorted_state_df), np.nan)

        picks = self.detect_pick_segments(
            closure_degrees=closure_df["closure_degree"].to_numpy(),
            closure_velocities=closure_df["closure_velocity"].to_numpy(),
            state_action_diffs=joint_diffs,
            hand_config=hand_config,
            state_df=sorted_state_df,
            action_df=sorted_action_df,
        )
        return self._frame_segments_to_time_segments(picks, sorted_state_df)

    def _frame_segments_to_time_segments(
        self,
        picks: List[Tuple[int, int]],
        state_df: pd.DataFrame,
    ) -> List[List[float]]:
        time_picks: List[List[float]] = []
        if len(state_df) == 0 or "timestamp_utc" not in state_df.columns:
            return [[float(start) / 60.0, float(end) / 60.0] for start, end in picks]

        start_time = state_df["timestamp_utc"].iloc[0]
        for start_idx, end_idx in picks:
            try:
                safe_start_idx = max(0, min(start_idx, len(state_df) - 1))
                safe_end_idx = max(0, min(end_idx, len(state_df) - 1))
                start_sec = _to_duration_seconds(
                    start_time,
                    state_df["timestamp_utc"].iloc[safe_start_idx],
                )
                end_sec = _to_duration_seconds(
                    start_time,
                    state_df["timestamp_utc"].iloc[safe_end_idx],
                )
                if start_sec is None or end_sec is None:
                    raise ValueError("failed to convert frame index to seconds")
                time_picks.append([float(start_sec), float(end_sec)])
            except Exception as exc:
                logger.error("Error converting frames %s to time: %s", (start_idx, end_idx), exc)
                time_picks.append([float(start_idx) / 60.0, float(end_idx) / 60.0])
        return time_picks

    def _build_hand_details(
        self,
        hand: str,
        segments: List[List[float]],
        episode_duration: Optional[float],
    ) -> Dict[str, Any]:
        filtered_segments = _filter_short_segments(
            segments,
            self.config.min_segment_duration_seconds,
        )
        duration_ratio = None
        if episode_duration is not None and episode_duration > 0:
            duration_ratio = _calc_total_duration(filtered_segments) / float(episode_duration)
        axis_points = _calc_axis_ratios(filtered_segments, episode_duration)
        axis_score = sum(axis_points) / len(axis_points) if axis_points else 0.0

        return {
            "hand": hand,
            "segments": filtered_segments,
            "count": len(filtered_segments),
            "duration_ratio": duration_ratio,
            "axis_points": axis_points,
            "axis_score": axis_score,
        }

    def build_task_context(
        self,
        task_description_en: str,
        detection_result: ValidationResult,
    ) -> Dict[str, Any]:
        task_required_grasps = extract_minimum_grasp_counts(task_description_en)
        detection_details = dict(detection_result.details or {})
        hands = detection_details.get("hands") or {}
        episode_minimum_detected_count = _minimum_detected_count(
            [
                int((hands.get(hand) or {}).get("count") or 0)
                for hand in HAND_KEYS
            ]
        )

        hand_stats = {
            hand: {
                "minimum_detected_count": episode_minimum_detected_count,
                "task_required_count": int(task_required_grasps.get(hand) or 0),
            }
            for hand in HAND_KEYS
        }

        return {
            "task_description_en": task_description_en,
            "task_required_grasps": task_required_grasps,
            "minimum_detected_grasps": {
                hand: episode_minimum_detected_count
                for hand in HAND_KEYS
            },
            "hand_stats": hand_stats,
        }

    def _evaluate_hand(
        self,
        hand: str,
        hand_metrics: Dict[str, Any],
        task_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        hand_stats = task_context.get("hand_stats", {}).get(hand, {})
        count_val = int(hand_metrics.get("count") or 0)
        minimum_detected_count = int(hand_stats.get("minimum_detected_count") or 0)
        task_required_count = int(hand_stats.get("task_required_count") or 0)
        count_delta_from_minimum = count_val - minimum_detected_count

        level = IssueLevel.INFO
        reason = "normal"
        if minimum_detected_count > 0 and count_val < minimum_detected_count:
            level = IssueLevel.MAJOR
            reason = "below_minimum_detected_count"
        elif count_delta_from_minimum >= 3:
            level = IssueLevel.MINOR
            reason = "above_minimum_detected_count_by_3"

        return {
            **hand_metrics,
            "duration_tag": "none",
            "axis_tag": "none",
            "minimum_detected_count": minimum_detected_count,
            "task_required_count": task_required_count,
            "count_delta_from_minimum": count_delta_from_minimum,
            "level": level.value,
            "message": LEVEL_MESSAGES[level],
            "reason": reason,
        }

    def _detect_episode_result(self, episode_id: str) -> ValidationResult:
        validation_data = self._load_validation_data(episode_id)
        task_description_en = str(validation_data.get("task_description_en") or "Unknown task")
        state_df = validation_data["state_df"]
        action_df = validation_data["action_df"]
        hand_segments = {
            hand: self._detect_hand_segments(state_df, action_df, hand)
            for hand in HAND_KEYS
        }

        episode_duration = self._resolve_episode_duration(validation_data, hand_segments)
        hand_results = {
            hand: self._build_hand_details(hand, hand_segments[hand], episode_duration)
            for hand in HAND_KEYS
        }

        return ValidationResult(
            passed=True,
            score=None,
            issues=[],
            details={
                "episode_id": episode_id,
                "task_id": validation_data.get("task_id"),
                "file_path": validation_data.get("file_path"),
                "validator_name": self.name,
                "category": self.category,
                "check_name": "抓取检测",
                "task_description_en": task_description_en,
                "episode_duration": episode_duration,
                "right_pnp_result": hand_results["right"]["segments"],
                "left_pnp_result": hand_results["left"]["segments"],
                "r_count": hand_results["right"]["count"],
                "l_count": hand_results["left"]["count"],
                "r_duration": hand_results["right"]["duration_ratio"],
                "l_duration": hand_results["left"]["duration_ratio"],
                "r_axis_score": hand_results["right"]["axis_score"],
                "l_axis_score": hand_results["left"]["axis_score"],
                "hands": hand_results,
            },
        )

    def _finalize_episode_result(
        self,
        detection_result: ValidationResult,
        task_context: Dict[str, Any],
    ) -> ValidationResult:
        detection_details = dict(detection_result.details or {})
        raw_hands = detection_details.get("hands") or {}
        hand_results = {
            hand: self._evaluate_hand(
                hand=hand,
                hand_metrics=raw_hands.get(hand) or {},
                task_context=task_context,
            )
            for hand in HAND_KEYS
        }

        if all(int(hand_results[hand].get("count") or 0) == 0 for hand in HAND_KEYS):
            issue_level = IssueLevel.CRITICAL
            for hand in HAND_KEYS:
                hand_results[hand]["level"] = issue_level.value
                hand_results[hand]["message"] = LEVEL_MESSAGES[issue_level]
                hand_results[hand]["reason"] = "no_detected_segments"
        else:
            issue_level = IssueLevel.INFO
            for hand_result in hand_results.values():
                candidate_level = IssueLevel(str(hand_result.get("level") or IssueLevel.INFO.value))
                if LEVEL_PRIORITY[candidate_level] > LEVEL_PRIORITY[issue_level]:
                    issue_level = candidate_level

        issue_value = None
        issue_threshold = None
        if issue_level == IssueLevel.CRITICAL:
            issue_value = 0.0
        elif issue_level == IssueLevel.MAJOR:
            majors = [
                hand_result
                for hand_result in hand_results.values()
                if hand_result.get("level") == IssueLevel.MAJOR.value
            ]
            if majors:
                issue_value = float(min(int(item.get("count") or 0) for item in majors))
                issue_threshold = float(
                    max(int(item.get("minimum_detected_count") or 0) for item in majors)
                )
        elif issue_level == IssueLevel.MINOR:
            minors = [
                hand_result
                for hand_result in hand_results.values()
                if hand_result.get("level") == IssueLevel.MINOR.value
            ]
            if minors:
                issue_value = float(max(int(item.get("count") or 0) for item in minors))
                issue_threshold = float(
                    max(int(item.get("minimum_detected_count") or 0) + 3 for item in minors)
                )

        details = _build_compact_ee_details(
            task_context=task_context,
            hand_results=hand_results,
            issue_level=issue_level.value,
            episode_duration=detection_details.get("episode_duration"),
            validator_name=self.name,
            category=self.category,
            check_name="抓取检测",
        )
        issue = self._create_issue(
            check_name="抓取检测",
            message=LEVEL_MESSAGES[issue_level],
            passed=issue_level != IssueLevel.CRITICAL,
            level=issue_level,
            value=issue_value,
            threshold=issue_threshold,
        )

        return ValidationResult(
            passed=issue_level != IssueLevel.CRITICAL,
            score=LEVEL_SCORES[issue_level],
            issues=[issue],
            details=details,
        )

    def build_stream_summary(self, result: ValidationResult) -> Dict[str, Any]:
        details = result.details or {}
        hands = details.get("hands") or {}
        right = hands.get("right") or {}
        left = hands.get("left") or {}
        return {
            "validator_name": details.get("validator_name"),
            "category": self.category,
            "check_name": details.get("check_name"),
            "passed": bool(result.passed),
            "issue_level": details.get("issue_level"),
            "r_count": right.get("count"),
            "l_count": left.get("count"),
            "r_duration_tag": right.get("duration_tag"),
            "l_duration_tag": left.get("duration_tag"),
            "r_axis_tag": right.get("axis_tag"),
            "l_axis_tag": left.get("axis_tag"),
            "minimum_detected_grasps": details.get("minimum_detected_grasps"),
            "hand_levels": {
                "right": right.get("level"),
                "left": left.get("level"),
            },
        }

    def _resolve_episode_duration(
        self,
        data: Dict[str, Any],
        hand_segments: Dict[str, List[List[float]]],
    ) -> Optional[float]:
        try:
            episode_duration = data.get("episode_duration")
            if episode_duration is not None and float(episode_duration) > 0:
                return float(episode_duration)
        except Exception:
            pass

        state_df = data.get("state_df")
        if isinstance(state_df, pd.DataFrame) and len(state_df) > 1 and "timestamp_utc" in state_df.columns:
            duration_sec = _to_duration_seconds(
                state_df["timestamp_utc"].iloc[0],
                state_df["timestamp_utc"].iloc[-1],
            )
            if duration_sec is not None:
                return duration_sec

        return _estimate_duration_from_segments(
            hand_segments.get("right"),
            hand_segments.get("left"),
        )

    def validate(
        self,
        episode_id: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        detection_result = self._detect_episode_result(episode_id)
        task_context = (data or {}).get("task_context")
        if not isinstance(task_context, dict):
            task_context = self.build_task_context(
                task_description_en=str(
                    detection_result.details.get("task_description_en")
                    or "Unknown task"
                ),
                detection_result=detection_result,
            )
        return self._finalize_episode_result(
            detection_result=detection_result,
            task_context=task_context,
        )
