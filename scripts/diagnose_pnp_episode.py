#!/usr/bin/env python3
"""Diagnose why a single episode did or did not trigger PNP pick detection."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.engines.pnp_detector.data_detector import (  # noqa: E402
    HAND_CONFIG_BASE,
    calculate_closure_metrics_from_dataframe,
    pick_identify,
)
from src.workers.pnp_worker import load_joint_data_as_dfs  # noqa: E402


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
class JointCheck:
    joint: str
    diff_value: float | None
    diff_threshold: float | None
    diff_passed: bool
    slope_passed: bool
    state_slope: float | None
    action_slope: float | None
    reason: str


@dataclass
class FrameDiagnosis:
    frame_idx: int
    timestamp_utc: str
    seconds_from_start: float
    closure_degree: float | None
    closure_passed: bool
    diff_passed: bool
    joints_satisfied: int
    pick_frame_passed: bool
    joint_checks: list[JointCheck]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_id", help="Episode ID to diagnose, e.g. 907517")
    parser.add_argument("--hand", choices=["right", "left"], default="right")
    parser.add_argument("--top-k", type=int, default=5, help="How many best candidate frames to print")
    parser.add_argument(
        "--params-json",
        type=str,
        default=None,
        help="Optional JSON string overriding detector params",
    )
    parser.add_argument(
        "--show-joints",
        action="store_true",
        help="Print per-joint details for every reported candidate frame",
    )
    return parser.parse_args()


def normalize_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "timestamp_utc" in result.columns and not pd.api.types.is_datetime64_any_dtype(result["timestamp_utc"]):
        result["timestamp_utc"] = pd.to_datetime(result["timestamp_utc"], unit="s")
    return result


def build_configs(params_override: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    params = dict(DEFAULT_PARAMS)
    if params_override:
        params.update(params_override)
    right = {**HAND_CONFIG_BASE["right"], **params}
    left = {**HAND_CONFIG_BASE["left"], **params}
    load_config = {"right_hand_fingers": right["right_hand_fingers"] + left["right_hand_fingers"]}
    return right, left, load_config


def prepare_hand_inputs(st_df: pd.DataFrame, ac_df: pd.DataFrame, hand_config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, np.ndarray], np.ndarray, np.ndarray]:
    closure_df = calculate_closure_metrics_from_dataframe(
        st_df,
        hand_config["right_hand_fingers"],
        hand_config["joint_direction_coefficients"],
    )

    st = st_df.copy().sort_values("timestamp_utc").reset_index(drop=True)
    ac = ac_df.copy().sort_values("timestamp_utc").reset_index(drop=True)
    ac_cols = ["timestamp_utc"] + [col for col in hand_config["right_hand_fingers"] if col in ac.columns]
    action_subset = ac[ac_cols]
    merged = pd.merge_asof(st, action_subset, on="timestamp_utc", direction="nearest", suffixes=("", "_action"))

    diffs: dict[str, np.ndarray] = {}
    for joint in hand_config["right_hand_fingers"]:
        action_col = f"{joint}_action"
        if joint in merged.columns and action_col in merged.columns:
            diffs[joint] = (merged[action_col] - merged[joint]).to_numpy()
        else:
            diffs[joint] = np.full(len(st), np.nan)

    return (
        closure_df,
        diffs,
        closure_df["closure_degree"].to_numpy(),
        closure_df["closure_velocity"].to_numpy(),
    )


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        fval = float(value)
    except Exception:
        return None
    if math.isnan(fval):
        return None
    return fval


def diagnose_joint(
    frame_idx: int,
    joint: str,
    diffs: dict[str, np.ndarray],
    state_df: pd.DataFrame,
    hand_config: dict[str, Any],
) -> JointCheck:
    coeff = hand_config["joint_direction_coefficients"].get(joint, 0.0)
    negative_threshold = hand_config["negative_diff_threshold"]
    positive_threshold = hand_config["positive_diff_threshold"]
    slope_threshold = hand_config["slope_threshold"]
    slope_lookahead = hand_config["slope_lookahead"]

    diff_series = diffs.get(joint)
    if diff_series is None or frame_idx >= len(diff_series):
        return JointCheck(joint, None, None, False, False, None, None, "missing_diff_series")

    diff_val = _float_or_none(diff_series[frame_idx])
    if diff_val is None:
        return JointCheck(joint, None, None, False, False, None, None, "diff_nan")

    if coeff < 0:
        diff_threshold = negative_threshold
        diff_passed = diff_val < negative_threshold
    else:
        diff_threshold = positive_threshold
        diff_passed = diff_val > positive_threshold

    if not diff_passed:
        return JointCheck(joint, diff_val, diff_threshold, False, False, None, None, "diff_threshold_not_met")

    if joint not in state_df.columns:
        return JointCheck(joint, diff_val, diff_threshold, True, False, None, None, "joint_missing_in_state")

    start_idx = frame_idx
    end_idx = min(len(state_df) - 1, frame_idx + slope_lookahead)
    if end_idx <= start_idx:
        return JointCheck(joint, diff_val, diff_threshold, True, True, None, None, "passed")

    diff_step = end_idx - start_idx
    state_start = _float_or_none(state_df[joint].iloc[start_idx])
    state_end = _float_or_none(state_df[joint].iloc[end_idx])
    end_diff = _float_or_none(diff_series[end_idx])

    if state_start is None or state_end is None or end_diff is None:
        return JointCheck(joint, diff_val, diff_threshold, True, False, None, None, "slope_data_nan")

    state_slope = abs(state_end - state_start) / diff_step
    if state_slope > slope_threshold:
        return JointCheck(
            joint,
            diff_val,
            diff_threshold,
            True,
            False,
            state_slope,
            None,
            "state_slope_too_large",
        )

    action_start = state_start + diff_val
    action_end = state_end + end_diff
    action_slope = abs(action_end - action_start) / diff_step
    if action_slope > slope_threshold:
        return JointCheck(
            joint,
            diff_val,
            diff_threshold,
            True,
            False,
            state_slope,
            action_slope,
            "action_slope_too_large",
        )

    return JointCheck(joint, diff_val, diff_threshold, True, True, state_slope, action_slope, "passed")


def diagnose_frame(
    frame_idx: int,
    st_df: pd.DataFrame,
    closure_degrees: np.ndarray,
    diffs: dict[str, np.ndarray],
    hand_config: dict[str, Any],
) -> FrameDiagnosis:
    closure_degree = _float_or_none(closure_degrees[frame_idx]) if frame_idx < len(closure_degrees) else None
    closure_passed = closure_degree is not None and closure_degree > hand_config["pick_closure_threshold"]

    joint_checks = [
        diagnose_joint(frame_idx, joint, diffs, st_df, hand_config)
        for joint in hand_config["right_hand_fingers"]
    ]
    joints_satisfied = sum(1 for item in joint_checks if item.diff_passed and item.slope_passed)
    diff_passed = joints_satisfied >= hand_config["min_joints_for_diff"]

    t0 = st_df["timestamp_utc"].iloc[0]
    ts = st_df["timestamp_utc"].iloc[frame_idx]
    seconds_from_start = float((ts - t0).total_seconds())

    return FrameDiagnosis(
        frame_idx=frame_idx,
        timestamp_utc=str(ts),
        seconds_from_start=seconds_from_start,
        closure_degree=closure_degree,
        closure_passed=closure_passed,
        diff_passed=diff_passed,
        joints_satisfied=joints_satisfied,
        pick_frame_passed=bool(closure_passed and diff_passed),
        joint_checks=joint_checks,
    )


def frame_sort_key(item: FrameDiagnosis) -> tuple[int, int, float, float]:
    closure = item.closure_degree if item.closure_degree is not None else float("-inf")
    return (
        1 if item.pick_frame_passed else 0,
        item.joints_satisfied,
        closure,
        -float(item.frame_idx),
    )


def summarize_blocker(frame: FrameDiagnosis, hand_config: dict[str, Any]) -> str:
    if not frame.closure_passed:
        closure = "nan" if frame.closure_degree is None else f"{frame.closure_degree:.6f}"
        threshold = hand_config["pick_closure_threshold"]
        return f"pick closure not met: {closure} <= {threshold:.6f}"

    if not frame.diff_passed:
        reasons: list[str] = []
        for item in frame.joint_checks:
            if item.reason == "passed":
                continue
            if item.reason == "diff_threshold_not_met":
                if item.diff_value is None or item.diff_threshold is None:
                    reasons.append(f"{item.joint}: diff threshold not met")
                else:
                    reasons.append(
                        f"{item.joint}: diff={item.diff_value:.6f}, threshold={item.diff_threshold:.6f}"
                    )
            elif item.reason == "state_slope_too_large":
                reasons.append(
                    f"{item.joint}: state_slope={item.state_slope:.6f} > {hand_config['slope_threshold']:.6f}"
                )
            elif item.reason == "action_slope_too_large":
                reasons.append(
                    f"{item.joint}: action_slope={item.action_slope:.6f} > {hand_config['slope_threshold']:.6f}"
                )
            else:
                reasons.append(f"{item.joint}: {item.reason}")
        detail = "; ".join(reasons[:4])
        return (
            f"only {frame.joints_satisfied} joints passed, need >= {hand_config['min_joints_for_diff']}; "
            f"{detail}"
        )

    return "pick frame conditions satisfied"


def print_frame(frame: FrameDiagnosis, hand_config: dict[str, Any], show_joints: bool) -> None:
    closure = "nan" if frame.closure_degree is None else f"{frame.closure_degree:.6f}"
    print(
        f"frame={frame.frame_idx} time={frame.seconds_from_start:.3f}s ts={frame.timestamp_utc} "
        f"closure={closure} closure_passed={frame.closure_passed} "
        f"joints_satisfied={frame.joints_satisfied}/{hand_config['min_joints_for_diff']} "
        f"pick_frame_passed={frame.pick_frame_passed}"
    )
    print(f"  blocker: {summarize_blocker(frame, hand_config)}")
    if show_joints:
        for item in frame.joint_checks:
            diff_text = "nan" if item.diff_value is None else f"{item.diff_value:.6f}"
            threshold_text = "nan" if item.diff_threshold is None else f"{item.diff_threshold:.6f}"
            state_slope = "nan" if item.state_slope is None else f"{item.state_slope:.6f}"
            action_slope = "nan" if item.action_slope is None else f"{item.action_slope:.6f}"
            print(
                f"    joint={item.joint} diff={diff_text} threshold={threshold_text} "
                f"diff_passed={item.diff_passed} slope_passed={item.slope_passed} "
                f"state_slope={state_slope} action_slope={action_slope} reason={item.reason}"
            )


def main() -> int:
    args = parse_args()
    params_override = json.loads(args.params_json) if args.params_json else None
    right_config, left_config, load_config = build_configs(params_override)
    hand_config = right_config if args.hand == "right" else left_config

    state_df, action_df = load_joint_data_as_dfs(str(args.episode_id), load_config)
    if state_df is None or action_df is None or len(state_df) == 0:
        print(f"failed to load joint data for episode_id={args.episode_id}", file=sys.stderr)
        return 1

    state_df = normalize_timestamps(state_df)
    action_df = normalize_timestamps(action_df)

    _, diffs, closure_degrees, closure_velocities = prepare_hand_inputs(state_df, action_df, hand_config)
    picks = pick_identify(
        closure_degrees=closure_degrees,
        closure_velocities=closure_velocities,
        state_action_diffs=diffs,
        config=hand_config,
        state_df=state_df,
        action_df=action_df,
    )

    diagnoses = [
        diagnose_frame(frame_idx, state_df, closure_degrees, diffs, hand_config)
        for frame_idx in range(len(state_df))
    ]
    ranked = sorted(diagnoses, key=frame_sort_key, reverse=True)
    best = ranked[0] if ranked else None

    print(f"episode_id={args.episode_id}")
    print(f"hand={args.hand}")
    print(f"state_frames={len(state_df)} action_frames={len(action_df)}")
    print(f"detected_picks={len(picks)}")
    print(f"pick_segments={json.dumps(picks, ensure_ascii=True)}")
    print(f"params={json.dumps({k: hand_config[k] for k in DEFAULT_PARAMS}, ensure_ascii=True, sort_keys=True)}")
    print()

    if best is None:
        print("no frames available")
        return 1

    print("best_candidate_frame:")
    print_frame(best, hand_config, True)
    print()

    if not picks:
        print("diagnosis:")
        print(f"  no pick detected for {args.hand} hand")
        print(f"  closest frame blocker: {summarize_blocker(best, hand_config)}")
        print()

    print(f"top_{args.top_k}_candidate_frames:")
    for frame in ranked[: max(1, int(args.top_k))]:
        print_frame(frame, hand_config, args.show_joints)

    if picks:
        print()
        print("detected_pick_windows:")
        t0 = state_df["timestamp_utc"].iloc[0]
        for start_idx, end_idx in picks:
            s_idx = max(0, min(start_idx, len(state_df) - 1))
            e_idx = max(0, min(end_idx, len(state_df) - 1))
            start_sec = float((state_df["timestamp_utc"].iloc[s_idx] - t0).total_seconds())
            end_sec = float((state_df["timestamp_utc"].iloc[e_idx] - t0).total_seconds())
            print(
                f"  frames=({start_idx},{end_idx}) "
                f"seconds=({start_sec:.3f},{end_sec:.3f})"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
