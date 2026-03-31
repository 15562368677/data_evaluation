#!/usr/bin/env python3
"""Generate a formal grasp-detection performance dashboard in HTML."""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

AcceptanceService = None
ValidationResult = None
ValidatorConfig = None
HAND_CONFIG_BASE = None
calculate_closure_metrics_from_dataframe = None
pick_identify = None
parse_hdf5_joints = None
parse_parquet_joints = None
resolve_joint_paths = None
query_pnp_df = None
download_s3_file = None
query_df = None
build_task_en = None

DEFAULT_TASK_IDS = [
    "954", "907", "760", "838", "1042", "1026", "812", "820", "801", "844",
    "824", "780", "919", "930", "1033", "847", "1003", "1027", "1060", "806",
    "750", "1032", "828", "849", "1010", "955", "787", "853", "805", "929",
    "961", "794", "1004", "1041", "797", "786", "761", "851", "783", "1002",
    "908", "1030", "925", "784", "1007", "1008", "1034", "785", "795", "1056",
]

DEFAULT_PARAMS = {
    "pick_closure_threshold": 0.35,
    "pick_start_offset": -5,
    "place_closure_threshold": 0.35,
    "place_velocity_threshold": -0.02,
    "place_velocity_lookback": 5,
    "place_velocity_lookahead": 0,
    "place_diff_lookahead": 10,
    "place_end_offset": 5,
    "negative_diff_threshold": -0.08,
    "positive_diff_threshold": 0.05,
    "min_joints_for_diff": 2,
    "slope_threshold": 0.0005,
    "slope_lookahead": 10,
}


@dataclass
class SampleMeta:
    task_id: str
    episode_id: str
    trajectory_duration_sec: float
    trajectory_start: str | None


@dataclass
class EpisodeProfile:
    task_id: str
    episode_id: str
    trajectory_duration_sec: float
    trajectory_start: str | None
    file_path: str | None
    joint_source_type: str | None
    download_object_count: int
    state_frames: int
    action_frames: int
    stream_lookup_seconds: float
    path_resolve_seconds: float
    download_seconds: float
    parse_seconds: float
    dataframe_build_seconds: float
    normalize_seconds: float
    right_closure_seconds: float
    right_align_diff_seconds: float
    right_identify_seconds: float
    left_closure_seconds: float
    left_align_diff_seconds: float
    left_identify_seconds: float
    compute_seconds: float
    issue_prepare_seconds: float
    issue_context_seconds: float
    issue_eval_seconds: float
    total_seconds: float
    right_segments: list[list[float]]
    left_segments: list[list[float]]
    issue_level: str | None
    issue_passed: bool | None
    issue_message: str | None
    status: str
    error_message: str | None


def ensure_runtime_dependencies() -> None:
    global AcceptanceService
    global ValidationResult
    global ValidatorConfig
    global HAND_CONFIG_BASE
    global calculate_closure_metrics_from_dataframe
    global pick_identify
    global parse_hdf5_joints
    global parse_parquet_joints
    global resolve_joint_paths
    global query_pnp_df
    global download_s3_file
    global query_df
    global build_task_en

    if AcceptanceService is not None:
        return

    from src.acceptance_service import AcceptanceService as _AcceptanceService
    from src.acceptance_service.validators.core.base import (
        ValidationResult as _ValidationResult,
        ValidatorConfig as _ValidatorConfig,
    )
    from src.engines.pnp_detector.data_detector import (
        HAND_CONFIG_BASE as _HAND_CONFIG_BASE,
        calculate_closure_metrics_from_dataframe as _calculate_closure_metrics_from_dataframe,
        pick_identify as _pick_identify,
    )
    from src.utils.data_parser import (
        parse_hdf5_joints as _parse_hdf5_joints,
        parse_parquet_joints as _parse_parquet_joints,
        resolve_joint_paths as _resolve_joint_paths,
    )
    from src.utils.result_db import query_pnp_df as _query_pnp_df
    from src.utils.s3_client import download_s3_file as _download_s3_file
    from src.utils.source_db import query_df as _query_df
    from src.validators.ee_action import build_task_en as _build_task_en

    AcceptanceService = _AcceptanceService
    ValidationResult = _ValidationResult
    ValidatorConfig = _ValidatorConfig
    HAND_CONFIG_BASE = _HAND_CONFIG_BASE
    calculate_closure_metrics_from_dataframe = _calculate_closure_metrics_from_dataframe
    pick_identify = _pick_identify
    parse_hdf5_joints = _parse_hdf5_joints
    parse_parquet_joints = _parse_parquet_joints
    resolve_joint_paths = _resolve_joint_paths
    query_pnp_df = _query_pnp_df
    download_s3_file = _download_s3_file
    query_df = _query_df
    build_task_en = _build_task_en


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-ids", nargs="+", default=DEFAULT_TASK_IDS)
    parser.add_argument("--sample-per-task", type=int, default=10)
    parser.add_argument("--output-html", default="docs/grasp_detection_performance_dashboard.html")
    parser.add_argument("--output-json", default="reports/grasp_detection_performance_dashboard.json")
    parser.add_argument("--top-slowest", type=int, default=15)
    return parser.parse_args()


def render_progress(current: int, total: int, *, label: str) -> None:
    width = 28
    ratio = 1.0 if total <= 0 else max(0.0, min(1.0, current / total))
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    percent = ratio * 100.0
    sys.stdout.write(f"\r{label} [{bar}] {current}/{total} {percent:5.1f}%")
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")


def mean_or_zero(values: list[float]) -> float:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(sum(vals) / len(vals)) if vals else 0.0


def percentile(values: list[float], q: float) -> float:
    vals = np.asarray([float(v) for v in values if v is not None and not math.isnan(float(v))], dtype=float)
    return float(np.percentile(vals, q)) if vals.size else 0.0


def max_or_zero(values: list[float]) -> float:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return max(vals) if vals else 0.0


def format_seconds(value: float) -> str:
    return f"{value:.3f}s"


def format_ratio(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%"


def html_escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def query_sampled_episodes(task_ids: list[str], sample_per_task: int) -> tuple[list[SampleMeta], dict[str, float]]:
    source_started = time.perf_counter()
    episodes_df = query_df(
        """
        SELECT
            e.task_id::text AS task_id,
            e.id::text AS episode_id,
            e.trajectory_duration,
            e.trajectory_start
        FROM episodes e
        WHERE e.task_id::text = ANY(%(task_ids)s)
          AND e.trajectory_duration IS NOT NULL
          AND e.trajectory_duration > 0
        ORDER BY e.task_id, e.trajectory_start NULLS LAST, e.id
        """,
        {"task_ids": [str(task_id) for task_id in task_ids]},
    )
    source_query_seconds = time.perf_counter() - source_started

    invalid_started = time.perf_counter()
    invalid_df = query_pnp_df(
        """
        SELECT DISTINCT episode_id::text AS episode_id
        FROM manual_duration_results
        WHERE duration_result = 'invalid'
          AND task_id::text = ANY(%(task_ids)s)
        """,
        {"task_ids": [str(task_id) for task_id in task_ids]},
    )
    invalid_query_seconds = time.perf_counter() - invalid_started

    filter_started = time.perf_counter()
    invalid_ids = set(invalid_df["episode_id"].astype(str).tolist()) if not invalid_df.empty else set()
    if invalid_ids:
        episodes_df = episodes_df[~episodes_df["episode_id"].astype(str).isin(invalid_ids)].copy()
    episodes_df["sample_rank"] = episodes_df.groupby("task_id").cumcount() + 1
    sampled_df = episodes_df[episodes_df["sample_rank"] <= int(sample_per_task)].copy()
    filter_seconds = time.perf_counter() - filter_started

    sampled = [
        SampleMeta(
            task_id=str(row.task_id),
            episode_id=str(row.episode_id),
            trajectory_duration_sec=float(row.trajectory_duration),
            trajectory_start=row.trajectory_start.isoformat() if pd.notnull(row.trajectory_start) else None,
        )
        for row in sampled_df.itertuples(index=False)
    ]
    return sampled, {
        "source_query_seconds": source_query_seconds,
        "invalid_query_seconds": invalid_query_seconds,
        "filter_seconds": filter_seconds,
    }


def query_task_descriptions(task_ids: list[str]) -> dict[str, str]:
    tasks_df = query_df(
        """
        SELECT id::text AS task_id, descriptions
        FROM tasks
        WHERE id::text = ANY(%(task_ids)s)
        """,
        {"task_ids": [str(task_id) for task_id in task_ids]},
    )
    result: dict[str, str] = {}
    for row in tasks_df.itertuples(index=False):
        result[str(row.task_id)] = build_task_en(row.descriptions)
    return result


def query_stream_paths(episode_ids: list[str]) -> tuple[dict[str, str], float]:
    lookup_started = time.perf_counter()
    stream_df = query_df(
        """
        SELECT DISTINCT ON (episode_id)
            episode_id::text AS episode_id,
            file_path
        FROM streams
        WHERE episode_id::text = ANY(%(episode_ids)s)
          AND stream_name = 'rgb'
        ORDER BY episode_id, id
        """,
        {"episode_ids": [str(episode_id) for episode_id in episode_ids]},
    )
    lookup_seconds = time.perf_counter() - lookup_started
    stream_map = {
        str(row.episode_id): str(row.file_path)
        for row in stream_df.itertuples(index=False)
        if pd.notnull(row.file_path)
    }
    return stream_map, lookup_seconds


def build_hand_configs() -> tuple[dict[str, Any], dict[str, Any], ValidatorConfig]:
    validator_config = ValidatorConfig()
    right = {**HAND_CONFIG_BASE["right"], **DEFAULT_PARAMS, "hand": "right"}
    left = {**HAND_CONFIG_BASE["left"], **DEFAULT_PARAMS, "hand": "left"}
    return right, left, validator_config


def parsed_joint_data_to_dfs(parsed_data: dict[str, Any], config: ValidatorConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_joints = config.get_all_hand_joints()

    state_df = pd.DataFrame({"timestamp_utc": parsed_data.get("absolute_timestamps_state", [])})
    for joint in all_joints:
        state_df[joint] = parsed_data.get("state", {}).get(joint, [np.nan] * len(state_df))

    action_df = pd.DataFrame({"timestamp_utc": parsed_data.get("absolute_timestamps_action", [])})
    for joint in all_joints:
        action_df[joint] = parsed_data.get("action", {}).get(joint, [np.nan] * len(action_df))

    return state_df, action_df


def normalize_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "timestamp_utc" in result.columns and not pd.api.types.is_datetime64_any_dtype(result["timestamp_utc"]):
        result["timestamp_utc"] = pd.to_datetime(result["timestamp_utc"], unit="s", errors="coerce")
    return result


def process_hand_timed(
    state_df: pd.DataFrame,
    action_df: pd.DataFrame,
    hand_config: dict[str, Any],
) -> tuple[list[list[float]], float, float, float]:
    closure_started = time.perf_counter()
    closure_df = calculate_closure_metrics_from_dataframe(
        state_df,
        hand_config["right_hand_fingers"],
        hand_config["joint_direction_coefficients"],
    )
    closure_seconds = time.perf_counter() - closure_started

    align_started = time.perf_counter()
    sorted_state_df = state_df.sort_values("timestamp_utc")
    sorted_action_df = action_df.sort_values("timestamp_utc")
    action_cols = ["timestamp_utc"] + [col for col in hand_config["right_hand_fingers"] if col in sorted_action_df.columns]
    action_subset = sorted_action_df[action_cols]
    merged = pd.merge_asof(
        sorted_state_df,
        action_subset,
        on="timestamp_utc",
        direction="nearest",
        suffixes=("", "_action"),
    )
    diffs: dict[str, np.ndarray] = {}
    for joint in hand_config["right_hand_fingers"]:
        action_col = f"{joint}_action"
        if joint in merged.columns and action_col in merged.columns:
            diffs[joint] = (merged[action_col] - merged[joint]).to_numpy()
        else:
            diffs[joint] = np.full(len(sorted_state_df), np.nan)
    align_diff_seconds = time.perf_counter() - align_started

    identify_started = time.perf_counter()
    picks = pick_identify(
        closure_degrees=closure_df["closure_degree"].to_numpy(),
        closure_velocities=closure_df["closure_velocity"].to_numpy(),
        state_action_diffs=diffs,
        config=hand_config,
        state_df=sorted_state_df,
        action_df=sorted_action_df,
    )
    segments: list[list[float]] = []
    if len(sorted_state_df) > 0 and "timestamp_utc" in sorted_state_df.columns:
        t0 = sorted_state_df["timestamp_utc"].iloc[0]
        for start_idx, end_idx in picks:
            start_idx = max(0, min(int(start_idx), len(sorted_state_df) - 1))
            end_idx = max(0, min(int(end_idx), len(sorted_state_df) - 1))
            start_sec = (sorted_state_df["timestamp_utc"].iloc[start_idx] - t0).total_seconds()
            end_sec = (sorted_state_df["timestamp_utc"].iloc[end_idx] - t0).total_seconds()
            segments.append([float(start_sec), float(end_sec)])
    identify_seconds = time.perf_counter() - identify_started
    return segments, closure_seconds, align_diff_seconds, identify_seconds


def profile_episode(
    sample: SampleMeta,
    validator_config: ValidatorConfig,
    right_config: dict[str, Any],
    left_config: dict[str, Any],
    file_path: str | None,
    stream_lookup_seconds: float = 0.0,
) -> EpisodeProfile:
    if not file_path:
        return EpisodeProfile(
            task_id=sample.task_id,
            episode_id=sample.episode_id,
            trajectory_duration_sec=sample.trajectory_duration_sec,
            trajectory_start=sample.trajectory_start,
            file_path=None,
            joint_source_type=None,
            download_object_count=0,
            state_frames=0,
            action_frames=0,
            stream_lookup_seconds=stream_lookup_seconds,
            path_resolve_seconds=0.0,
            download_seconds=0.0,
            parse_seconds=0.0,
            dataframe_build_seconds=0.0,
            normalize_seconds=0.0,
            right_closure_seconds=0.0,
            right_align_diff_seconds=0.0,
            right_identify_seconds=0.0,
            left_closure_seconds=0.0,
            left_align_diff_seconds=0.0,
            left_identify_seconds=0.0,
            compute_seconds=0.0,
            issue_prepare_seconds=0.0,
            issue_context_seconds=0.0,
            issue_eval_seconds=0.0,
            total_seconds=stream_lookup_seconds,
            right_segments=[],
            left_segments=[],
            issue_level=None,
            issue_passed=None,
            issue_message=None,
            status="failed",
            error_message="No rgb stream path found for episode.",
        )

    resolve_started = time.perf_counter()
    joint_info = resolve_joint_paths(file_path)
    path_resolve_seconds = time.perf_counter() - resolve_started
    if joint_info is None:
        return EpisodeProfile(
            task_id=sample.task_id,
            episode_id=sample.episode_id,
            trajectory_duration_sec=sample.trajectory_duration_sec,
            trajectory_start=sample.trajectory_start,
            file_path=file_path,
            joint_source_type=None,
            download_object_count=0,
            state_frames=0,
            action_frames=0,
            stream_lookup_seconds=stream_lookup_seconds,
            path_resolve_seconds=path_resolve_seconds,
            download_seconds=0.0,
            parse_seconds=0.0,
            dataframe_build_seconds=0.0,
            normalize_seconds=0.0,
            right_closure_seconds=0.0,
            right_align_diff_seconds=0.0,
            right_identify_seconds=0.0,
            left_closure_seconds=0.0,
            left_align_diff_seconds=0.0,
            left_identify_seconds=0.0,
            compute_seconds=0.0,
            issue_prepare_seconds=0.0,
            issue_context_seconds=0.0,
            issue_eval_seconds=0.0,
            total_seconds=stream_lookup_seconds + path_resolve_seconds,
            right_segments=[],
            left_segments=[],
            issue_level=None,
            issue_passed=None,
            issue_message=None,
            status="failed",
            error_message="No joint source resolved from stream path.",
        )

    download_started = time.perf_counter()
    download_object_count = 0
    parsed_data: dict[str, Any] | None = None
    if joint_info["type"] == "parquet":
        action_local = None
        state_local = None
        if "action_key" in joint_info:
            download_object_count += 1
            action_local = download_s3_file(joint_info["action_key"], cache_kind="joints")
        if "state_key" in joint_info:
            download_object_count += 1
            state_local = download_s3_file(joint_info["state_key"], cache_kind="joints")
        download_seconds = time.perf_counter() - download_started

        parse_started = time.perf_counter()
        parsed_data = parse_parquet_joints(action_local, state_local)
        parse_seconds = time.perf_counter() - parse_started
    else:
        download_object_count = 1
        hdf5_local = download_s3_file(joint_info["key"], cache_kind="joints")
        download_seconds = time.perf_counter() - download_started

        parse_started = time.perf_counter()
        parsed_data = parse_hdf5_joints(hdf5_local) if hdf5_local is not None else None
        parse_seconds = time.perf_counter() - parse_started

    if not parsed_data:
        total = stream_lookup_seconds + path_resolve_seconds + download_seconds + parse_seconds
        return EpisodeProfile(
            task_id=sample.task_id,
            episode_id=sample.episode_id,
            trajectory_duration_sec=sample.trajectory_duration_sec,
            trajectory_start=sample.trajectory_start,
            file_path=file_path,
            joint_source_type=joint_info["type"],
            download_object_count=download_object_count,
            state_frames=0,
            action_frames=0,
            stream_lookup_seconds=stream_lookup_seconds,
            path_resolve_seconds=path_resolve_seconds,
            download_seconds=download_seconds,
            parse_seconds=parse_seconds,
            dataframe_build_seconds=0.0,
            normalize_seconds=0.0,
            right_closure_seconds=0.0,
            right_align_diff_seconds=0.0,
            right_identify_seconds=0.0,
            left_closure_seconds=0.0,
            left_align_diff_seconds=0.0,
            left_identify_seconds=0.0,
            compute_seconds=0.0,
            issue_prepare_seconds=0.0,
            issue_context_seconds=0.0,
            issue_eval_seconds=0.0,
            total_seconds=total,
            right_segments=[],
            left_segments=[],
            issue_level=None,
            issue_passed=None,
            issue_message=None,
            status="failed",
            error_message="Joint data parse returned empty payload.",
        )

    dataframe_started = time.perf_counter()
    state_df, action_df = parsed_joint_data_to_dfs(parsed_data, validator_config)
    dataframe_build_seconds = time.perf_counter() - dataframe_started
    if len(state_df) == 0 or len(action_df) == 0:
        total = (
            stream_lookup_seconds + path_resolve_seconds + download_seconds + parse_seconds + dataframe_build_seconds
        )
        return EpisodeProfile(
            task_id=sample.task_id,
            episode_id=sample.episode_id,
            trajectory_duration_sec=sample.trajectory_duration_sec,
            trajectory_start=sample.trajectory_start,
            file_path=file_path,
            joint_source_type=joint_info["type"],
            download_object_count=download_object_count,
            state_frames=int(len(state_df)),
            action_frames=int(len(action_df)),
            stream_lookup_seconds=stream_lookup_seconds,
            path_resolve_seconds=path_resolve_seconds,
            download_seconds=download_seconds,
            parse_seconds=parse_seconds,
            dataframe_build_seconds=dataframe_build_seconds,
            normalize_seconds=0.0,
            right_closure_seconds=0.0,
            right_align_diff_seconds=0.0,
            right_identify_seconds=0.0,
            left_closure_seconds=0.0,
            left_align_diff_seconds=0.0,
            left_identify_seconds=0.0,
            compute_seconds=0.0,
            issue_prepare_seconds=0.0,
            issue_context_seconds=0.0,
            issue_eval_seconds=0.0,
            total_seconds=total,
            right_segments=[],
            left_segments=[],
            issue_level=None,
            issue_passed=None,
            issue_message=None,
            status="failed",
            error_message="State or action dataframe is empty.",
        )

    normalize_started = time.perf_counter()
    state_df = normalize_timestamps(state_df)
    action_df = normalize_timestamps(action_df)
    normalize_seconds = time.perf_counter() - normalize_started

    compute_started = time.perf_counter()
    right_segments, r_closure, r_align, r_identify = process_hand_timed(state_df, action_df, right_config)
    left_segments, l_closure, l_align, l_identify = process_hand_timed(state_df, action_df, left_config)
    compute_seconds = time.perf_counter() - compute_started

    total_seconds = (
        stream_lookup_seconds
        + path_resolve_seconds
        + download_seconds
        + parse_seconds
        + dataframe_build_seconds
        + normalize_seconds
        + compute_seconds
    )

    return EpisodeProfile(
        task_id=sample.task_id,
        episode_id=sample.episode_id,
        trajectory_duration_sec=sample.trajectory_duration_sec,
        trajectory_start=sample.trajectory_start,
        file_path=file_path,
        joint_source_type=joint_info["type"],
        download_object_count=download_object_count,
        state_frames=int(len(state_df)),
        action_frames=int(len(action_df)),
        stream_lookup_seconds=stream_lookup_seconds,
        path_resolve_seconds=path_resolve_seconds,
        download_seconds=download_seconds,
        parse_seconds=parse_seconds,
        dataframe_build_seconds=dataframe_build_seconds,
        normalize_seconds=normalize_seconds,
        right_closure_seconds=r_closure,
        right_align_diff_seconds=r_align,
        right_identify_seconds=r_identify,
        left_closure_seconds=l_closure,
        left_align_diff_seconds=l_align,
        left_identify_seconds=l_identify,
        compute_seconds=compute_seconds,
        issue_prepare_seconds=0.0,
        issue_context_seconds=0.0,
        issue_eval_seconds=0.0,
        total_seconds=total_seconds,
        right_segments=right_segments,
        left_segments=left_segments,
        issue_level=None,
        issue_passed=None,
        issue_message=None,
        status="success",
        error_message=None,
    )


def evaluate_issues(profiles: list[EpisodeProfile], task_descriptions: dict[str, str], validator_config: ValidatorConfig) -> None:
    service = AcceptanceService(config=validator_config)
    _ = task_descriptions
    for profile in profiles:
        if profile.status != "success":
            continue
        issue_eval_started = time.perf_counter()
        final_result = service.validator.validate(profile.episode_id)
        profile.issue_prepare_seconds = 0.0
        profile.issue_context_seconds = 0.0
        profile.issue_eval_seconds = time.perf_counter() - issue_eval_started
        profile.total_seconds += profile.issue_eval_seconds
        profile.issue_level = final_result.details.get("issue_level")
        profile.issue_passed = bool(final_result.passed)
        issue_message = None
        if final_result.issues:
            issue_message = final_result.issues[0].message
        profile.issue_message = issue_message


def sort_task_id(task_id: str) -> tuple[int, int | str]:
    return (0, int(task_id)) if task_id.isdigit() else (1, task_id)


def build_stage_summary(profiles: list[EpisodeProfile]) -> list[dict[str, Any]]:
    successful = [item for item in profiles if item.status == "success"]
    stage_specs = [
        ("stream_lookup_seconds", "Stream Lookup"),
        ("path_resolve_seconds", "Joint Path Resolve"),
        ("download_seconds", "Download"),
        ("parse_seconds", "Parse"),
        ("dataframe_build_seconds", "DataFrame Build"),
        ("normalize_seconds", "Timestamp Normalize"),
        ("right_closure_seconds", "Right Closure"),
        ("right_align_diff_seconds", "Right Align+Diff"),
        ("right_identify_seconds", "Right Identify"),
        ("left_closure_seconds", "Left Closure"),
        ("left_align_diff_seconds", "Left Align+Diff"),
        ("left_identify_seconds", "Left Identify"),
        ("compute_seconds", "Detection Compute"),
        ("issue_prepare_seconds", "Issue Prep"),
        ("issue_context_seconds", "Issue Context"),
        ("issue_eval_seconds", "Issue Evaluate"),
        ("total_seconds", "End-to-End"),
    ]
    rows = []
    for attr, label in stage_specs:
        values = [getattr(item, attr) for item in successful]
        rows.append(
            {
                "label": label,
                "avg": mean_or_zero(values),
                "p50": percentile(values, 50),
                "p95": percentile(values, 95),
                "max": max_or_zero(values),
            }
        )
    return rows


def build_task_rows(profiles: list[EpisodeProfile], task_ids: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_task: dict[str, list[EpisodeProfile]] = {}
    for profile in profiles:
        by_task.setdefault(profile.task_id, []).append(profile)

    for task_id in task_ids:
        items = by_task.get(task_id, [])
        successful = [item for item in items if item.status == "success"]
        rows.append(
            {
                "task_id": task_id,
                "sampled": len(items),
                "success_count": len(successful),
                "failed_count": len(items) - len(successful),
                "avg_duration": mean_or_zero([item.trajectory_duration_sec for item in successful]),
                "avg_download": mean_or_zero([item.download_seconds for item in successful]),
                "avg_compute": mean_or_zero([item.compute_seconds for item in successful]),
                "avg_issue": mean_or_zero(
                    [item.issue_prepare_seconds + item.issue_context_seconds + item.issue_eval_seconds for item in successful]
                ),
                "avg_total": mean_or_zero([item.total_seconds for item in successful]),
                "p95_total": percentile([item.total_seconds for item in successful], 95),
            }
        )
    return rows


def build_issue_rows(profiles: list[EpisodeProfile]) -> list[dict[str, Any]]:
    buckets = {"critical": 0, "major": 0, "minor": 0, "info": 0, "unknown": 0}
    for profile in profiles:
        key = profile.issue_level or "unknown"
        buckets[key] = buckets.get(key, 0) + 1
    return [{"level": level, "count": count} for level, count in buckets.items()]


def build_slowest_rows(profiles: list[EpisodeProfile], top_n: int) -> list[EpisodeProfile]:
    successful = [item for item in profiles if item.status == "success"]
    return sorted(successful, key=lambda item: item.total_seconds, reverse=True)[:top_n]


def summary_cards(profiles: list[EpisodeProfile], sampling_meta: dict[str, float], task_ids: list[str]) -> list[dict[str, str]]:
    successful = [item for item in profiles if item.status == "success"]
    issue_total = [
        item.issue_prepare_seconds + item.issue_context_seconds + item.issue_eval_seconds
        for item in successful
    ]
    return [
        {"title": "任务数", "value": str(len(task_ids)), "hint": "固定 task_id 清单"},
        {"title": "样本总数", "value": str(len(profiles)), "hint": "每任务默认 10 条"},
        {"title": "成功计时", "value": str(len(successful)), "hint": "成功完成下载、计算和 issues 判断"},
        {"title": "平均端到端", "value": format_seconds(mean_or_zero([item.total_seconds for item in successful])), "hint": "单条 episode 总耗时"},
        {"title": "P95 端到端", "value": format_seconds(percentile([item.total_seconds for item in successful], 95)), "hint": "尾部时延"},
        {"title": "平均下载", "value": format_seconds(mean_or_zero([item.download_seconds for item in successful])), "hint": "S3 文件下载"},
        {"title": "平均计算", "value": format_seconds(mean_or_zero([item.compute_seconds for item in successful])), "hint": "抓取检测主计算"},
        {"title": "平均 issues", "value": format_seconds(mean_or_zero(issue_total)), "hint": "issues 准备 + 上下文 + 判定"},
        {
            "title": "采样查询",
            "value": format_seconds(
                sampling_meta["source_query_seconds"]
                + sampling_meta["invalid_query_seconds"]
                + sampling_meta["filter_seconds"]
            ),
            "hint": "episode 采样与 invalid 过滤",
        },
    ]


def render_stage_cards(stage_rows: list[dict[str, Any]]) -> str:
    max_avg = max((row["avg"] for row in stage_rows), default=0.0) or 1.0
    cards: list[str] = []
    for row in stage_rows:
        width = row["avg"] / max_avg * 100.0
        cards.append(
            f"""
            <article class="stage-card panel">
              <div class="stage-head">
                <h3>{html_escape(row["label"])}</h3>
                <strong>{format_seconds(row["avg"])}</strong>
              </div>
              <div class="meter"><span style="width:{width:.2f}%"></span></div>
              <div class="stage-meta">
                <span>P50 {format_seconds(row["p50"])}</span>
                <span>P95 {format_seconds(row["p95"])}</span>
                <span>MAX {format_seconds(row["max"])}</span>
              </div>
            </article>
            """.strip()
        )
    return "\n".join(cards)


def render_task_table(task_rows: list[dict[str, Any]]) -> str:
    lines = []
    for row in task_rows:
        lines.append(
            f"""
            <tr>
              <td>{html_escape(row["task_id"])}</td>
              <td>{row["sampled"]}</td>
              <td>{row["success_count"]}</td>
              <td>{row["failed_count"]}</td>
              <td>{format_seconds(row["avg_download"])}</td>
              <td>{format_seconds(row["avg_compute"])}</td>
              <td>{format_seconds(row["avg_issue"])}</td>
              <td>{format_seconds(row["avg_total"])}</td>
              <td>{format_seconds(row["p95_total"])}</td>
              <td>{row["avg_duration"]:.1f}s</td>
            </tr>
            """.strip()
        )
    return "\n".join(lines)


def render_issue_rows(issue_rows: list[dict[str, Any]]) -> str:
    total = sum(item["count"] for item in issue_rows) or 1
    blocks = []
    for row in issue_rows:
        ratio = row["count"] / total * 100.0
        blocks.append(
            f"""
            <article class="issue-card panel issue-{html_escape(row["level"])}">
              <span class="issue-label">{html_escape(row["level"])}</span>
              <strong>{row["count"]}</strong>
              <div class="meter compact"><span style="width:{ratio:.2f}%"></span></div>
            </article>
            """.strip()
        )
    return "\n".join(blocks)


def render_slowest_rows(slowest: list[EpisodeProfile]) -> str:
    rows = []
    for item in slowest:
        issue_total = item.issue_prepare_seconds + item.issue_context_seconds + item.issue_eval_seconds
        rows.append(
            f"""
            <tr>
              <td>{html_escape(item.task_id)}</td>
              <td>{html_escape(item.episode_id)}</td>
              <td>{item.trajectory_duration_sec:.1f}s</td>
              <td>{format_seconds(item.download_seconds)}</td>
              <td>{format_seconds(item.compute_seconds)}</td>
              <td>{format_seconds(issue_total)}</td>
              <td>{format_seconds(item.total_seconds)}</td>
              <td>{html_escape(item.issue_level or '-')}</td>
            </tr>
            """.strip()
        )
    return "\n".join(rows)


def build_methodology(task_ids: list[str], sample_per_task: int) -> str:
    task_text = "、".join(task_ids)
    return f"""
      <ol class="method-list">
        <li>采样范围固定为 {len(task_ids)} 个 task_id：{html_escape(task_text)}。</li>
        <li>每个 task 按 <code>trajectory_start</code> 升序抽取前 {sample_per_task} 条有效 episode，并过滤 <code>manual_duration_results</code> 中已标记为 <code>invalid</code> 的样本。</li>
        <li>单条 episode 计时按阶段拆分为：<code>Stream Lookup</code>、<code>Joint Path Resolve</code>、<code>Download</code>、<code>Parse</code>、<code>DataFrame Build</code>、<code>Timestamp Normalize</code>、<code>Detection Compute</code>、<code>Issue Prep</code>、<code>Issue Context</code>、<code>Issue Evaluate</code>。</li>
        <li><code>Detection Compute</code> 继续拆成右手与左手的 <code>Closure</code>、<code>Align+Diff</code>、<code>Identify</code>，用于定位抓取检测主瓶颈。</li>
        <li><code>Issue Context</code> 表示同 task 样本共同构建统计上下文的耗时，按任务内 episode 平均分摊。</li>
      </ol>
    """.strip()


def build_html(
    *,
    profiles: list[EpisodeProfile],
    sampling_meta: dict[str, float],
    task_ids: list[str],
    sample_per_task: int,
    output_json: Path,
    generated_at: str,
    top_slowest: int,
) -> str:
    stage_rows = build_stage_summary(profiles)
    task_rows = build_task_rows(profiles, task_ids)
    issue_rows = build_issue_rows(profiles)
    slowest = build_slowest_rows(profiles, top_slowest)
    cards = summary_cards(profiles, sampling_meta, task_ids)

    card_html = "\n".join(
        f"""
        <article class="stat-card panel">
          <span class="stat-title">{html_escape(card["title"])}</span>
          <strong>{html_escape(card["value"])}</strong>
          <p>{html_escape(card["hint"])}</p>
        </article>
        """.strip()
        for card in cards
    )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>抓取检测性能分析看板</title>
  <style>
    :root {{
      --bg: #eef2e8;
      --panel: rgba(252, 253, 248, 0.88);
      --panel-strong: #fbfdf7;
      --ink: #17211b;
      --muted: #57635a;
      --line: rgba(28, 42, 33, 0.12);
      --accent: #2f6c4f;
      --accent-strong: #214d39;
      --accent-soft: rgba(47, 108, 79, 0.12);
      --warm: #b96e32;
      --warm-soft: rgba(185, 110, 50, 0.12);
      --danger: #b24848;
      --shadow: 0 18px 42px rgba(30, 44, 34, 0.08);
    }}

    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "IBM Plex Sans", "Noto Sans SC", "PingFang SC", "Microsoft YaHei", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(47, 108, 79, 0.18), transparent 22%),
        radial-gradient(circle at right 12% top 16%, rgba(185, 110, 50, 0.14), transparent 20%),
        linear-gradient(180deg, #f3f6ee 0%, #ebf0e5 46%, #e5ebdd 100%);
    }}

    .page {{
      max-width: 1480px;
      margin: 0 auto;
      padding: 28px 18px 56px;
      display: grid;
      gap: 18px;
    }}

    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(12px);
    }}

    .hero {{
      padding: 30px 28px;
      display: grid;
      gap: 18px;
      background:
        linear-gradient(135deg, rgba(252, 253, 248, 0.96), rgba(247, 250, 242, 0.88)),
        var(--panel);
    }}

    .eyebrow {{
      display: inline-flex;
      width: fit-content;
      padding: 8px 12px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent-strong);
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }}

    h1 {{
      margin: 0;
      font-size: clamp(30px, 4vw, 50px);
      line-height: 1.04;
    }}

    .hero-copy {{
      display: grid;
      gap: 10px;
      max-width: 900px;
    }}

    .hero-copy p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.7;
    }}

    .hero-tags {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}

    .tag {{
      padding: 10px 14px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.74);
      border: 1px solid rgba(28, 42, 33, 0.08);
      color: var(--muted);
      font-size: 13px;
    }}

    .section {{
      display: grid;
      gap: 14px;
    }}

    .section-head {{
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 12px;
    }}

    .section-head h2 {{
      margin: 0;
      font-size: 24px;
    }}

    .section-head p {{
      margin: 0;
      color: var(--muted);
    }}

    .stats-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
    }}

    .stat-card {{
      padding: 18px 18px 16px;
      display: grid;
      gap: 10px;
    }}

    .stat-title {{
      color: var(--muted);
      font-size: 13px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      font-weight: 700;
    }}

    .stat-card strong {{
      font-size: 28px;
      line-height: 1;
    }}

    .stat-card p {{
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }}

    .stage-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 14px;
    }}

    .stage-card {{
      padding: 18px;
      display: grid;
      gap: 12px;
    }}

    .stage-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
    }}

    .stage-head h3 {{
      margin: 0;
      font-size: 16px;
    }}

    .stage-head strong {{
      font-size: 22px;
    }}

    .meter {{
      width: 100%;
      height: 12px;
      border-radius: 999px;
      overflow: hidden;
      background: rgba(47, 108, 79, 0.08);
    }}

    .meter span {{
      display: block;
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, var(--accent), #5f9a7b);
    }}

    .meter.compact {{
      height: 8px;
    }}

    .stage-meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      color: var(--muted);
      font-size: 13px;
    }}

    .issues-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 14px;
    }}

    .issue-card {{
      padding: 18px;
      display: grid;
      gap: 10px;
    }}

    .issue-card strong {{
      font-size: 30px;
      line-height: 1;
    }}

    .issue-label {{
      text-transform: uppercase;
      font-size: 12px;
      letter-spacing: 0.04em;
      font-weight: 700;
      color: var(--muted);
    }}

    .issue-critical .meter span {{ background: linear-gradient(90deg, #8a2f2f, #c85c5c); }}
    .issue-major .meter span {{ background: linear-gradient(90deg, #9d5a1f, #d0883f); }}
    .issue-minor .meter span {{ background: linear-gradient(90deg, #6f7f29, #9eaf4d); }}
    .issue-info .meter span {{ background: linear-gradient(90deg, var(--accent), #5f9a7b); }}
    .issue-unknown .meter span {{ background: linear-gradient(90deg, #667085, #8b96a8); }}

    .content-panel {{
      padding: 22px;
    }}

    .method-list {{
      margin: 0;
      padding-left: 22px;
      color: var(--muted);
      line-height: 1.8;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}

    th, td {{
      padding: 11px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}

    th {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}

    tr:last-child td {{
      border-bottom: none;
    }}

    .table-wrap {{
      overflow: auto;
    }}

    code {{
      background: rgba(28, 42, 33, 0.06);
      padding: 2px 6px;
      border-radius: 6px;
    }}

    .footnote {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.7;
    }}

    @media (max-width: 840px) {{
      .page {{ padding: 18px 14px 40px; }}
      .hero {{ padding: 22px 20px; }}
      .content-panel {{ padding: 18px; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero panel">
      <div class="hero-copy">
        <span class="eyebrow">Performance Dashboard</span>
        <h1>抓取检测性能分析看板</h1>
        <p>本报告面向指定 task 清单，对每个任务抽取 10 条 episode，按“下载、解析、抓取检测计算、issues 判断”完整链路做阶段化计时，并将结果整理为可直接审阅的卡片式 HTML 仪表板。</p>
      </div>
      <div class="hero-tags">
        <span class="tag">生成时间：{html_escape(generated_at)}</span>
        <span class="tag">输出 JSON：{html_escape(str(output_json))}</span>
        <span class="tag">采样策略：按 trajectory_start 升序取前 {sample_per_task} 条</span>
      </div>
    </section>

    <section class="section">
      <div class="section-head">
        <h2>核心指标</h2>
        <p>聚合所有成功完成检测与 issues 判定的 episode。</p>
      </div>
      <div class="stats-grid">
        {card_html}
      </div>
    </section>

    <section class="section">
      <div class="section-head">
        <h2>阶段耗时</h2>
        <p>按平均耗时排序，便于快速定位链路热点。</p>
      </div>
      <div class="stage-grid">
        {render_stage_cards(stage_rows)}
      </div>
    </section>

    <section class="section">
      <div class="section-head">
        <h2>Issues 分布</h2>
        <p>展示最终质检等级在全量样本中的分布情况。</p>
      </div>
      <div class="issues-grid">
        {render_issue_rows(issue_rows)}
      </div>
    </section>

    <section class="section">
      <div class="section-head">
        <h2>方法说明</h2>
        <p>确保报告口径清晰，方便复用或复核。</p>
      </div>
      <div class="content-panel panel">
        {build_methodology(task_ids, sample_per_task)}
      </div>
    </section>

    <section class="section">
      <div class="section-head">
        <h2>任务级汇总</h2>
        <p>对每个 task 的 10 条样本分别统计下载、计算、issues 和端到端耗时。</p>
      </div>
      <div class="content-panel panel table-wrap">
        <table>
          <thead>
            <tr>
              <th>Task ID</th>
              <th>采样</th>
              <th>成功</th>
              <th>失败</th>
              <th>平均下载</th>
              <th>平均计算</th>
              <th>平均 Issues</th>
              <th>平均总耗时</th>
              <th>P95 总耗时</th>
              <th>平均轨迹时长</th>
            </tr>
          </thead>
          <tbody>
            {render_task_table(task_rows)}
          </tbody>
        </table>
      </div>
    </section>

    <section class="section">
      <div class="section-head">
        <h2>最慢样本</h2>
        <p>优先排查尾部 episode，通常更能反映真实生产热点。</p>
      </div>
      <div class="content-panel panel table-wrap">
        <table>
          <thead>
            <tr>
              <th>Task ID</th>
              <th>Episode ID</th>
              <th>轨迹时长</th>
              <th>下载</th>
              <th>计算</th>
              <th>Issues</th>
              <th>总耗时</th>
              <th>Issue Level</th>
            </tr>
          </thead>
          <tbody>
            {render_slowest_rows(slowest)}
          </tbody>
        </table>
      </div>
      <p class="footnote">说明：<code>Issues</code> 为 <code>Issue Prep</code>、<code>Issue Context</code>、<code>Issue Evaluate</code> 三项之和。若需要进一步拆解某个 task，可直接查看 JSON 明细中的单条 episode 记录。</p>
    </section>
  </main>
</body>
</html>
""".strip()


def main() -> None:
    args = parse_args()
    ensure_runtime_dependencies()
    warnings.filterwarnings(
        "ignore",
        message="pandas only supports SQLAlchemy connectable",
        category=UserWarning,
    )
    output_html = (REPO_ROOT / args.output_html).resolve()
    output_json = (REPO_ROOT / args.output_json).resolve()
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    sampling_started = time.perf_counter()
    sampled, sampling_meta = query_sampled_episodes(list(args.task_ids), args.sample_per_task)
    sampling_meta["sampling_total_seconds"] = time.perf_counter() - sampling_started

    task_descriptions = query_task_descriptions(list(args.task_ids))
    stream_map, stream_lookup_total_seconds = query_stream_paths([sample.episode_id for sample in sampled])
    per_episode_stream_lookup_seconds = stream_lookup_total_seconds / len(sampled) if sampled else 0.0
    sampling_meta["stream_lookup_total_seconds"] = stream_lookup_total_seconds
    right_config, left_config, validator_config = build_hand_configs()

    profiles: list[EpisodeProfile] = []
    total = len(sampled)
    render_progress(0, total, label="Profiling")
    for idx, sample in enumerate(sampled, start=1):
        try:
            profiles.append(
                profile_episode(
                    sample,
                    validator_config,
                    right_config,
                    left_config,
                    file_path=stream_map.get(sample.episode_id),
                    stream_lookup_seconds=per_episode_stream_lookup_seconds,
                )
            )
        except Exception as exc:
            profiles.append(
                EpisodeProfile(
                    task_id=sample.task_id,
                    episode_id=sample.episode_id,
                    trajectory_duration_sec=sample.trajectory_duration_sec,
                    trajectory_start=sample.trajectory_start,
                    file_path=None,
                    joint_source_type=None,
                    download_object_count=0,
                    state_frames=0,
                    action_frames=0,
                    stream_lookup_seconds=0.0,
                    path_resolve_seconds=0.0,
                    download_seconds=0.0,
                    parse_seconds=0.0,
                    dataframe_build_seconds=0.0,
                    normalize_seconds=0.0,
                    right_closure_seconds=0.0,
                    right_align_diff_seconds=0.0,
                    right_identify_seconds=0.0,
                    left_closure_seconds=0.0,
                    left_align_diff_seconds=0.0,
                    left_identify_seconds=0.0,
                    compute_seconds=0.0,
                    issue_prepare_seconds=0.0,
                    issue_context_seconds=0.0,
                    issue_eval_seconds=0.0,
                    total_seconds=0.0,
                    right_segments=[],
                    left_segments=[],
                    issue_level=None,
                    issue_passed=None,
                    issue_message=None,
                    status="failed",
                    error_message=str(exc),
                )
            )
        render_progress(idx, total, label="Profiling")

    evaluate_issues(profiles, task_descriptions, validator_config)

    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "task_ids": list(args.task_ids),
        "sample_per_task": int(args.sample_per_task),
        "sampling_meta": sampling_meta,
        "stage_summary": build_stage_summary(profiles),
        "task_summary": build_task_rows(profiles, list(args.task_ids)),
        "issue_summary": build_issue_rows(profiles),
        "profiles": [asdict(profile) for profile in profiles],
    }
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    html_text = build_html(
        profiles=profiles,
        sampling_meta=sampling_meta,
        task_ids=list(args.task_ids),
        sample_per_task=int(args.sample_per_task),
        output_json=output_json,
        generated_at=payload["generated_at"],
        top_slowest=int(args.top_slowest),
    )
    output_html.write_text(html_text, encoding="utf-8")

    print(f"HTML report written to: {output_html}")
    print(f"JSON report written to: {output_json}")


if __name__ == "__main__":
    main()
