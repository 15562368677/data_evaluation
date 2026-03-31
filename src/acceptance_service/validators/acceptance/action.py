"""动作数据质量验证器。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..core.base import BaseValidator, IssueLevel, ValidationResult

KEY_JOINTS = [
    "left_shoulder_pitch_joint",
    "left_elbow_pitch_joint",
    "left_wrist_pitch_joint",
    "right_shoulder_pitch_joint",
    "right_elbow_pitch_joint",
    "right_wrist_pitch_joint",
]


class ActionValidator(BaseValidator):
    """验证动作数据的静止、速度、时长和缺失情况。"""

    @property
    def name(self) -> str:
        return "动作数据验证器"

    @property
    def category(self) -> str:
        return "动作数据"

    def validate(
        self,
        episode_id: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        data = data or {}
        issues = []

        joint_df, joint_columns, format_type, timestamps = self._load_joint_data(data)
        if format_type == "unknown":
            issues.append(
                self._create_issue(
                    check_name="状态数据文件",
                    message="缺少可用的动作状态数据",
                    passed=False,
                    level=IssueLevel.CRITICAL,
                )
            )
            return ValidationResult(
                passed=False,
                score=0.0,
                issues=issues,
                details={
                    "episode_id": episode_id,
                    "validator_name": self.name,
                    "category": self.category,
                    "format": format_type,
                },
            )

        if joint_df is None or not joint_columns:
            issues.append(
                self._create_issue(
                    check_name="关节数据列",
                    message=f"未找到或无法解析关节数据 (格式: {format_type})",
                    passed=False,
                    level=IssueLevel.CRITICAL,
                )
            )
            return ValidationResult(
                passed=False,
                score=0.0,
                issues=issues,
                details={
                    "episode_id": episode_id,
                    "validator_name": self.name,
                    "category": self.category,
                    "format": format_type,
                },
            )

        issues.append(
            self._create_issue(
                check_name="状态数据读取",
                message=f"成功加载 {len(joint_df)} 帧 (格式: {format_type})",
                passed=True,
                level=IssueLevel.INFO,
            )
        )
        issues.append(
            self._create_issue(
                check_name="关节数据列",
                message=f"找到 {len(joint_columns)} 个关节列",
                passed=True,
                level=IssueLevel.INFO,
            )
        )

        fps = self._estimate_fps(timestamps)

        all_static_duration = self._detect_consecutive_static(
            joint_df,
            joint_columns,
            fps,
            threshold=self.config.static_diff_threshold,
        )
        issues.append(
            self._create_issue(
                check_name="全身静止检测",
                message=(
                    f"最长连续静止 {all_static_duration:.1f}s "
                    f"（阈值 {self.config.static_threshold_all}s）"
                ),
                passed=all_static_duration <= self.config.static_threshold_all,
                level=IssueLevel.MAJOR,
                value=all_static_duration,
                threshold=self.config.static_threshold_all,
            )
        )

        key_joints_in_df = [joint for joint in KEY_JOINTS if joint in joint_columns]
        key_static_duration = None
        if key_joints_in_df:
            key_static_duration = self._detect_consecutive_static(
                joint_df,
                key_joints_in_df,
                fps,
                threshold=self.config.static_diff_threshold,
            )
            issues.append(
                self._create_issue(
                    check_name="关键关节静止检测",
                    message=(
                        f"关键关节最长连续静止 {key_static_duration:.1f}s "
                        f"（阈值 {self.config.static_threshold_key}s）"
                    ),
                    passed=key_static_duration <= self.config.static_threshold_key,
                    level=IssueLevel.MAJOR,
                    value=key_static_duration,
                    threshold=self.config.static_threshold_key,
                )
            )

        main_joint_cols = [
            column
            for column in joint_columns
            if not any(
                finger in column.lower()
                for finger in ["thumb", "index", "middle", "ring", "pinky"]
            )
        ]
        time_diffs = self._compute_time_diffs(timestamps)
        velocity_issues = self._check_joint_velocities(joint_df, main_joint_cols, time_diffs)
        unsafe_count = sum(1 for item in velocity_issues.values() if not item["safe"])
        max_velocity = max(
            (item["max_velocity"] for item in velocity_issues.values()),
            default=0.0,
        )
        issues.append(
            self._create_issue(
                check_name="关节速度安全",
                message=(
                    f"最大速度 {max_velocity:.2f} rad/s "
                    f"（限制 {self.config.max_joint_velocity} rad/s），"
                    f"超速关节 {unsafe_count} 个"
                ),
                passed=unsafe_count == 0,
                level=IssueLevel.MAJOR,
                value=max_velocity,
                threshold=self.config.max_joint_velocity,
            )
        )

        duration = (len(joint_df) / fps) if fps > 0 else 0.0
        issues.append(
            self._create_issue(
                check_name="数据时长",
                message=(
                    f"时长 {duration:.1f} 秒 "
                    f"（最少 {self.config.min_action_duration} 秒）"
                ),
                passed=duration >= self.config.min_action_duration,
                level=IssueLevel.MAJOR,
                value=duration,
                threshold=self.config.min_action_duration,
            )
        )

        nan_count = int(joint_df[joint_columns].isna().sum().sum())
        total_values = len(joint_df) * len(joint_columns)
        nan_ratio = (nan_count / total_values) if total_values > 0 else 0.0
        issues.append(
            self._create_issue(
                check_name="NaN 值检查",
                message=(
                    f"NaN 占比 {nan_ratio * 100:.2f}% "
                    f"（限制 {self.config.max_nan_ratio * 100:.2f}%）"
                ),
                passed=nan_ratio < self.config.max_nan_ratio,
                level=IssueLevel.MAJOR,
                value=nan_ratio,
                threshold=self.config.max_nan_ratio,
            )
        )

        passed_count = sum(1 for issue in issues if issue.passed)
        total_count = len(issues)
        score = round((passed_count / total_count * 100.0), 1) if total_count > 0 else 0.0
        overall_passed = all(
            issue.passed
            for issue in issues
            if issue.level in (IssueLevel.CRITICAL, IssueLevel.MAJOR)
        )

        return ValidationResult(
            passed=overall_passed,
            score=score,
            issues=issues,
            details={
                "episode_id": episode_id,
                "validator_name": self.name,
                "category": self.category,
                "check_name": "动作数据质检",
                "frame_count": len(joint_df),
                "joint_count": len(joint_columns),
                "format": format_type,
                "fps": fps,
                "all_static_duration": all_static_duration,
                "key_static_duration": key_static_duration,
                "max_velocity": max_velocity,
                "unsafe_joint_count": unsafe_count,
                "duration": duration,
                "nan_count": nan_count,
                "nan_ratio": nan_ratio,
            },
        )

    def _load_joint_data(
        self,
        data: Dict[str, Any],
    ) -> Tuple[Optional[pd.DataFrame], List[str], str, Optional[np.ndarray]]:
        state_df = data.get("state_df")
        if isinstance(state_df, pd.DataFrame):
            joint_df, joint_columns = self._extract_joint_data(state_df)
            timestamps = state_df["timestamp_utc"].to_numpy() if "timestamp_utc" in state_df.columns else None
            return joint_df, joint_columns, "dataframe", timestamps

        data_path = data.get("data_path")
        if isinstance(data_path, str) and data_path:
            return self._load_joint_data_from_path(Path(data_path))

        return None, [], "unknown", None

    def _load_joint_data_from_path(
        self,
        path: Path,
    ) -> Tuple[Optional[pd.DataFrame], List[str], str, Optional[np.ndarray]]:
        state_file = path / "observation.state.parquet"
        hdf5_file = path / "data.hdf5"

        if state_file.exists():
            try:
                df = pd.read_parquet(state_file)
                joint_df, joint_columns = self._extract_joint_data(df)
                timestamps = df["timestamp_utc"].to_numpy() if "timestamp_utc" in df.columns else None
                return joint_df, joint_columns, "parquet", timestamps
            except Exception:
                pass

        if hdf5_file.exists():
            try:
                import h5py

                with h5py.File(hdf5_file, "r") as handle:
                    timestamps = handle["timestamp"][:] if "timestamp" in handle else None
                    state_robot = (
                        handle["state"]["robot"][:]
                        if "state" in handle and "robot" in handle["state"]
                        else None
                    )
                    state_hand = (
                        handle["state"]["hand"][:]
                        if "state" in handle and "hand" in handle["state"]
                        else None
                    )

                    records: List[Dict[str, float]] = []
                    n_frames = len(timestamps) if timestamps is not None else 0
                    for index in range(n_frames):
                        record: Dict[str, float] = {}
                        if state_robot is not None:
                            for joint_idx, value in enumerate(state_robot[index]):
                                record[f"robot_joint_{joint_idx}"] = float(value)
                        if state_hand is not None:
                            for joint_idx, value in enumerate(state_hand[index]):
                                record[f"hand_joint_{joint_idx}"] = float(value)
                        records.append(record)

                    joint_df = pd.DataFrame(records)
                    return joint_df, list(joint_df.columns), "hdf5", timestamps
            except Exception:
                pass

        return None, [], "unknown", None

    def _extract_joint_data(
        self,
        df: pd.DataFrame,
    ) -> Tuple[Optional[pd.DataFrame], List[str]]:
        joint_columns = [
            column
            for column in df.columns
            if column != "timestamp_utc"
            and (
                "_joint" in column.lower()
                or "proximal" in column.lower()
            )
        ]
        if joint_columns:
            numeric_df = df.copy()
            numeric_df[joint_columns] = numeric_df[joint_columns].apply(
                pd.to_numeric,
                errors="coerce",
            )
            return numeric_df, joint_columns

        nested_column = "observation.state"
        if nested_column not in df.columns:
            return None, []

        records: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            state = row[nested_column]
            record: Dict[str, Any] = {}
            if hasattr(state, "__iter__") and not isinstance(state, (str, dict)):
                for item in state:
                    if isinstance(item, dict) and "name" in item and "value" in item:
                        record[item["name"]] = item["value"]
            elif isinstance(state, dict):
                record = dict(state)
            records.append(record)

        joint_df = pd.DataFrame(records)
        joint_columns = [
            column
            for column in joint_df.columns
            if "_joint" in column.lower() or "proximal" in column.lower()
        ]
        if not joint_columns:
            joint_columns = [
                column
                for column in joint_df.columns
                if pd.api.types.is_numeric_dtype(joint_df[column])
            ]
        if joint_columns:
            joint_df[joint_columns] = joint_df[joint_columns].apply(
                pd.to_numeric,
                errors="coerce",
            )
        return joint_df, joint_columns

    def _estimate_fps(self, timestamps: Optional[np.ndarray]) -> float:
        if timestamps is None or len(timestamps) <= 1:
            return 60.0

        try:
            ts_series = pd.Series(timestamps)
            diffs = ts_series.diff().iloc[1:]
            if pd.api.types.is_timedelta64_dtype(diffs):
                seconds = (diffs / pd.Timedelta(seconds=1)).to_numpy(dtype=float)
            else:
                raw_diffs = diffs.to_numpy(dtype=float)
                median_diff = np.median(raw_diffs[raw_diffs > 0]) if np.any(raw_diffs > 0) else 0.0
                if median_diff > 1e6:
                    seconds = raw_diffs / 1e9
                elif median_diff > 1e3:
                    seconds = raw_diffs / 1e3
                else:
                    seconds = raw_diffs

            valid_seconds = seconds[seconds > 0]
            if len(valid_seconds) == 0:
                return 60.0
            avg_interval = float(np.mean(valid_seconds))
            return 1.0 / avg_interval if avg_interval > 0 else 60.0
        except Exception:
            return 60.0

    def _compute_time_diffs(self, timestamps: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if timestamps is None or len(timestamps) <= 1:
            return None

        try:
            ts_series = pd.Series(timestamps)
            diffs = ts_series.diff().iloc[1:]
            if pd.api.types.is_timedelta64_dtype(diffs):
                return (diffs / pd.Timedelta(seconds=1)).to_numpy(dtype=float)

            raw_diffs = diffs.to_numpy(dtype=float)
            median_diff = np.median(raw_diffs[raw_diffs > 0]) if np.any(raw_diffs > 0) else 0.0
            if median_diff > 1e6:
                return raw_diffs / 1e9
            if median_diff > 1e3:
                return raw_diffs / 1e3
            return raw_diffs
        except Exception:
            return None

    def _detect_consecutive_static(
        self,
        df: pd.DataFrame,
        columns: List[str],
        fps: float,
        threshold: float,
    ) -> float:
        if not columns or len(df) < 2 or fps <= 0:
            return 0.0

        data = df[columns].to_numpy(dtype=float)
        if data.ndim != 2 or len(data) < 2:
            return 0.0

        diffs = np.abs(np.diff(data, axis=0))
        if diffs.size == 0:
            return 0.0
        max_diffs = np.nanmax(diffs, axis=1)
        is_static = np.where(np.isnan(max_diffs), False, max_diffs < threshold)

        max_consecutive = 0
        current_consecutive = 0
        for static_flag in is_static:
            if static_flag:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive / fps

    def _check_joint_velocities(
        self,
        df: pd.DataFrame,
        columns: List[str],
        time_diffs: Optional[np.ndarray] = None,
    ) -> Dict[str, Dict[str, float | bool]]:
        results: Dict[str, Dict[str, float | bool]] = {}
        for column in columns:
            data = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
            if len(data) < 2:
                continue

            pos_diff = np.abs(np.diff(data))
            valid_mask = ~np.isnan(pos_diff)
            if time_diffs is not None and len(time_diffs) == len(pos_diff):
                safe_diffs = np.where(time_diffs > 0, time_diffs, 1e-6)
                velocities = pos_diff[valid_mask] / safe_diffs[valid_mask]
            else:
                velocities = pos_diff[valid_mask] * 60.0

            p99_velocity = float(np.percentile(velocities, 99)) if len(velocities) > 0 else 0.0
            results[column] = {
                "max_velocity": p99_velocity,
                "safe": p99_velocity < self.config.max_joint_velocity,
            }
        return results
