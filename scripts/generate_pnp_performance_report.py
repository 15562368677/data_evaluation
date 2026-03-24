#!/usr/bin/env python3
"""Generate a time-focused PNP performance report in Markdown or HTML."""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

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
from src.utils.result_db import query_pnp_df  # noqa: E402
from src.utils.source_db import query_df  # noqa: E402
from src.workers.pnp_worker import load_joint_data_as_dfs  # noqa: E402

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
class EpisodeTiming:
    task_id: str
    episode_id: str
    trajectory_duration_sec: float
    state_frames: int
    action_frames: int
    load_seconds: float
    normalize_seconds: float
    right_closure_seconds: float
    right_align_diff_seconds: float
    right_identify_seconds: float
    left_closure_seconds: float
    left_align_diff_seconds: float
    left_identify_seconds: float
    total_detect_seconds: float
    total_seconds: float
    status: str
    error_message: str | None


@dataclass
class SamplingTiming:
    source_query_seconds: float
    invalid_query_seconds: float
    filter_seconds: float
    sampled_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-ids", nargs="+", default=DEFAULT_TASK_IDS)
    parser.add_argument("--sample-per-task", type=int, default=3)
    parser.add_argument("--max-total-episodes", type=int, default=120)
    parser.add_argument("--format", choices=["md", "html", "both"], default="both")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--report-name", default="pnp_time_performance_report")
    parser.add_argument("--top-slowest", type=int, default=12)
    parser.add_argument("--bucket-size", type=int, default=20)
    return parser.parse_args()


def mean_or_zero(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return float(sum(vals) / len(vals)) if vals else 0.0


def percentile(values: Iterable[float], q: float) -> float:
    vals = np.asarray([float(v) for v in values if v is not None and not math.isnan(float(v))], dtype=float)
    return float(np.percentile(vals, q)) if vals.size else 0.0


def max_or_zero(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    return max(vals) if vals else 0.0


def svg_escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def query_sampled_episodes(task_ids: list[str], sample_per_task: int, max_total_episodes: int) -> tuple[pd.DataFrame, SamplingTiming]:
    if sample_per_task <= 0:
        raise ValueError("sample_per_task must be > 0")

    source_started = time.perf_counter()
    episodes_df = query_df(
        """
        SELECT
            e.task_id::text AS task_id,
            e.id::text AS episode_id,
            e.trajectory_start,
            e.trajectory_duration
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
        FROM duration_results
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
    episodes_df["rn"] = episodes_df.groupby("task_id").cumcount() + 1
    sampled_df = episodes_df[episodes_df["rn"] <= int(sample_per_task)].copy()
    if len(sampled_df) > max_total_episodes:
        sampled_df = sampled_df.head(int(max_total_episodes)).copy()
    filter_seconds = time.perf_counter() - filter_started

    return sampled_df, SamplingTiming(
        source_query_seconds=source_query_seconds,
        invalid_query_seconds=invalid_query_seconds,
        filter_seconds=filter_seconds,
        sampled_count=int(len(sampled_df)),
    )


def build_hand_configs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    right = {**HAND_CONFIG_BASE["right"], **DEFAULT_PARAMS}
    left = {**HAND_CONFIG_BASE["left"], **DEFAULT_PARAMS}
    load_config = {"right_hand_fingers": right["right_hand_fingers"] + left["right_hand_fingers"]}
    return right, left, load_config


def normalize_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "timestamp_utc" in result.columns and not pd.api.types.is_datetime64_any_dtype(result["timestamp_utc"]):
        result["timestamp_utc"] = pd.to_datetime(result["timestamp_utc"], unit="s")
    return result


def process_hand_timed(st_df: pd.DataFrame, ac_df: pd.DataFrame, hand_config: dict[str, Any]) -> tuple[list[list[float]], float, float, float]:
    closure_started = time.perf_counter()
    closure_df = calculate_closure_metrics_from_dataframe(
        st_df,
        hand_config["right_hand_fingers"],
        hand_config["joint_direction_coefficients"],
    )
    closure_seconds = time.perf_counter() - closure_started

    align_started = time.perf_counter()
    st = st_df.copy().sort_values("timestamp_utc")
    ac = ac_df.copy().sort_values("timestamp_utc")
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
    align_diff_seconds = time.perf_counter() - align_started

    identify_started = time.perf_counter()
    picks = pick_identify(
        closure_degrees=closure_df["closure_degree"].to_numpy(),
        closure_velocities=closure_df["closure_velocity"].to_numpy(),
        state_action_diffs=diffs,
        config=hand_config,
        state_df=st_df,
        action_df=ac_df,
    )
    segments: list[list[float]] = []
    if len(st_df) > 0 and "timestamp_utc" in st_df.columns:
        t0 = st_df["timestamp_utc"].iloc[0]
        for start_idx, end_idx in picks:
            start_idx = max(0, min(int(start_idx), len(st_df) - 1))
            end_idx = max(0, min(int(end_idx), len(st_df) - 1))
            start_sec = (st_df["timestamp_utc"].iloc[start_idx] - t0).total_seconds()
            end_sec = (st_df["timestamp_utc"].iloc[end_idx] - t0).total_seconds()
            segments.append([float(start_sec), float(end_sec)])
    identify_seconds = time.perf_counter() - identify_started
    return segments, closure_seconds, align_diff_seconds, identify_seconds


def run_timing(samples_df: pd.DataFrame) -> list[EpisodeTiming]:
    config_right, config_left, config_load = build_hand_configs()
    results: list[EpisodeTiming] = []

    for row in samples_df.itertuples(index=False):
        task_id = str(row.task_id)
        episode_id = str(row.episode_id)
        trajectory_duration_sec = float(row.trajectory_duration)
        load_started = time.perf_counter()
        try:
            state_df, action_df = load_joint_data_as_dfs(episode_id, config_load)
            load_seconds = time.perf_counter() - load_started
            if state_df is None or action_df is None or len(state_df) == 0:
                results.append(EpisodeTiming(
                    task_id=task_id,
                    episode_id=episode_id,
                    trajectory_duration_sec=trajectory_duration_sec,
                    state_frames=0,
                    action_frames=0,
                    load_seconds=load_seconds,
                    normalize_seconds=0.0,
                    right_closure_seconds=0.0,
                    right_align_diff_seconds=0.0,
                    right_identify_seconds=0.0,
                    left_closure_seconds=0.0,
                    left_align_diff_seconds=0.0,
                    left_identify_seconds=0.0,
                    total_detect_seconds=0.0,
                    total_seconds=load_seconds,
                    status="failed",
                    error_message="No valid joint data returned.",
                ))
                continue

            normalize_started = time.perf_counter()
            state_df = normalize_timestamps(state_df)
            action_df = normalize_timestamps(action_df)
            normalize_seconds = time.perf_counter() - normalize_started

            detect_started = time.perf_counter()
            _, r_closure, r_align, r_identify = process_hand_timed(state_df, action_df, config_right)
            _, l_closure, l_align, l_identify = process_hand_timed(state_df, action_df, config_left)
            total_detect_seconds = time.perf_counter() - detect_started

            total_seconds = load_seconds + normalize_seconds + total_detect_seconds
            results.append(EpisodeTiming(
                task_id=task_id,
                episode_id=episode_id,
                trajectory_duration_sec=trajectory_duration_sec,
                state_frames=int(len(state_df)),
                action_frames=int(len(action_df)),
                load_seconds=load_seconds,
                normalize_seconds=normalize_seconds,
                right_closure_seconds=r_closure,
                right_align_diff_seconds=r_align,
                right_identify_seconds=r_identify,
                left_closure_seconds=l_closure,
                left_align_diff_seconds=l_align,
                left_identify_seconds=l_identify,
                total_detect_seconds=total_detect_seconds,
                total_seconds=total_seconds,
                status="success",
                error_message=None,
            ))
        except Exception as exc:
            load_seconds = time.perf_counter() - load_started
            results.append(EpisodeTiming(
                task_id=task_id,
                episode_id=episode_id,
                trajectory_duration_sec=trajectory_duration_sec,
                state_frames=0,
                action_frames=0,
                load_seconds=load_seconds,
                normalize_seconds=0.0,
                right_closure_seconds=0.0,
                right_align_diff_seconds=0.0,
                right_identify_seconds=0.0,
                left_closure_seconds=0.0,
                left_align_diff_seconds=0.0,
                left_identify_seconds=0.0,
                total_detect_seconds=0.0,
                total_seconds=load_seconds,
                status="failed",
                error_message=str(exc),
            ))
    return results


def build_time_metric_rows(results: list[EpisodeTiming]) -> list[dict[str, Any]]:
    successful = [item for item in results if item.status == "success"]
    metrics = [
        ("load_seconds", "Data Load"),
        ("normalize_seconds", "Timestamp Normalize"),
        ("right_closure_seconds", "Right Closure"),
        ("right_align_diff_seconds", "Right Align+Diff"),
        ("right_identify_seconds", "Right Identify"),
        ("left_closure_seconds", "Left Closure"),
        ("left_align_diff_seconds", "Left Align+Diff"),
        ("left_identify_seconds", "Left Identify"),
        ("total_detect_seconds", "Total Detect"),
        ("total_seconds", "End-to-End Total"),
    ]
    rows = []
    for key, label in metrics:
        values = [getattr(item, key) for item in successful]
        rows.append({
            "metric": label,
            "avg_seconds": mean_or_zero(values),
            "p50_seconds": percentile(values, 50),
            "p95_seconds": percentile(values, 95),
            "max_seconds": max_or_zero(values),
        })
    return rows


def build_task_timing_rows(results: list[EpisodeTiming]) -> list[dict[str, Any]]:
    rows = []
    successful = [item for item in results if item.status == "success"]
    by_task: dict[str, list[EpisodeTiming]] = {}
    for item in successful:
        by_task.setdefault(item.task_id, []).append(item)
    for task_id in sorted(by_task.keys(), key=lambda v: (0, int(v)) if v.isdigit() else (1, v)):
        items = by_task[task_id]
        rows.append({
            "task_id": task_id,
            "sampled": len(items),
            "avg_load_seconds": mean_or_zero(item.load_seconds for item in items),
            "avg_detect_seconds": mean_or_zero(item.total_detect_seconds for item in items),
            "avg_total_seconds": mean_or_zero(item.total_seconds for item in items),
            "p95_total_seconds": percentile((item.total_seconds for item in items), 95),
            "avg_trajectory_duration_sec": mean_or_zero(item.trajectory_duration_sec for item in items),
        })
    return rows


def build_slowest_rows(results: list[EpisodeTiming], top_n: int) -> list[dict[str, Any]]:
    rows = []
    for item in sorted(results, key=lambda x: x.total_seconds, reverse=True)[:top_n]:
        rows.append({
            "task_id": item.task_id,
            "episode_id": item.episode_id,
            "trajectory_duration_sec": item.trajectory_duration_sec,
            "load_seconds": item.load_seconds,
            "normalize_seconds": item.normalize_seconds,
            "total_detect_seconds": item.total_detect_seconds,
            "total_seconds": item.total_seconds,
            "status": item.status,
        })
    return rows


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    headers = [title for _, title in columns]
    keys = [key for key, _ in columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        rendered = []
        for key in keys:
            value = row.get(key, "")
            rendered.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def regression_series(x_values: list[float], y_values: list[float]) -> tuple[list[tuple[float, float]], str | None, float | None]:
    if len(x_values) < 2:
        return [], None, None
    degree = 2 if len(x_values) >= 3 else 1
    coeffs = np.polyfit(x_values, y_values, degree)
    poly = np.poly1d(coeffs)
    xs = np.linspace(min(x_values), max(x_values), 120) if not math.isclose(min(x_values), max(x_values)) else np.asarray([x_values[0]])
    series = [(float(x), float(poly(x))) for x in xs]
    y_pred = poly(x_values)
    ss_res = float(np.sum((np.asarray(y_values) - y_pred) ** 2))
    y_mean = float(np.mean(y_values))
    ss_tot = float(np.sum((np.asarray(y_values) - y_mean) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    if degree == 2:
        equation = f"y = {coeffs[0]:.5f}x^2 + {coeffs[1]:.5f}x + {coeffs[2]:.5f}"
    else:
        equation = f"y = {coeffs[0]:.5f}x + {coeffs[1]:.5f}"
    return series, equation, r2


def write_text(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def render_scatter_svg(title: str, x_label: str, y_label: str, points: list[dict[str, Any]], regression: list[tuple[float, float]], regression_label: str | None, output_path: Path) -> Path:
    width, height = 920, 560
    margin_left, margin_top, margin_right, margin_bottom = 92, 76, 36, 78
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    x_values = [float(p["x"]) for p in points] or [0.0, 1.0]
    y_values = [float(p["y"]) for p in points] or [0.0, 1.0]
    if regression:
        x_values.extend(float(x) for x, _ in regression)
        y_values.extend(float(y) for _, y in regression)
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(0.0, min(y_values)), max(y_values)
    if math.isclose(x_min, x_max):
        x_min -= 1.0
        x_max += 1.0
    else:
        pad = (x_max - x_min) * 0.08
        x_min -= pad
        x_max += pad
    if math.isclose(y_min, y_max):
        y_min -= 1.0
        y_max += 1.0
    else:
        pad = max((y_max - y_min) * 0.12, 0.05)
        y_min = max(0.0, y_min - pad)
        y_max += pad

    def map_x(v: float) -> float:
        return margin_left + (v - x_min) / (x_max - x_min) * plot_width

    def map_y(v: float) -> float:
        return margin_top + plot_height - (v - y_min) / (y_max - y_min) * plot_height

    grid, ticks = [], []
    for idx in range(6):
        gx = margin_left + idx / 5 * plot_width
        gy = margin_top + idx / 5 * plot_height
        xv = x_min + idx / 5 * (x_max - x_min)
        yv = y_max - idx / 5 * (y_max - y_min)
        grid.append(f'<line x1="{gx:.1f}" y1="{margin_top}" x2="{gx:.1f}" y2="{margin_top + plot_height}" stroke="#dbe2ea" stroke-dasharray="4 4"/>')
        grid.append(f'<line x1="{margin_left}" y1="{gy:.1f}" x2="{margin_left + plot_width}" y2="{gy:.1f}" stroke="#dbe2ea" stroke-dasharray="4 4"/>')
        ticks.append(f'<text x="{gx:.1f}" y="{margin_top + plot_height + 24}" font-size="12" font-family="Times New Roman, Georgia, serif" text-anchor="middle" fill="#4b5563">{xv:.1f}</text>')
        ticks.append(f'<text x="{margin_left - 12}" y="{gy + 4:.1f}" font-size="12" font-family="Times New Roman, Georgia, serif" text-anchor="end" fill="#4b5563">{yv:.2f}</text>')

    regression_svg = ""
    if regression:
        path_d = " ".join([f"M {map_x(regression[0][0]):.2f} {map_y(regression[0][1]):.2f}"] + [f"L {map_x(x):.2f} {map_y(y):.2f}" for x, y in regression[1:]])
        regression_svg = f'<path d="{path_d}" fill="none" stroke="#c41e3a" stroke-width="2.4"/>'

    point_svg = []
    for point in points:
        px = map_x(float(point["x"]))
        py = map_y(float(point["y"]))
        point_svg.append(f'<circle cx="{px:.2f}" cy="{py:.2f}" r="5.2" fill="#1d4ed8" stroke="#ffffff" stroke-width="1.3"/>')
        point_svg.append(f'<text x="{px:.2f}" y="{py - 10:.2f}" font-size="10.5" font-family="Times New Roman, Georgia, serif" text-anchor="middle" fill="#111827">{svg_escape(point["label"])}</text>')

    note = ""
    if regression_label:
        note = (
            f'<rect x="{width - 360}" y="{margin_top + 8}" width="324" height="36" rx="6" fill="#ffffff" stroke="#cbd5e1"/>'
            f'<text x="{width - 198}" y="{margin_top + 30}" font-size="11.5" font-family="Times New Roman, Georgia, serif" text-anchor="middle" fill="#334155">{svg_escape(regression_label)}</text>'
        )
    svg = f"""
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#fff"/>
  <text x="{width / 2}" y="36" text-anchor="middle" font-size="24" font-weight="700" font-family="Times New Roman, Georgia, serif" fill="#111827">{svg_escape(title)}</text>
  <text x="{width / 2}" y="58" text-anchor="middle" font-size="12" font-family="Times New Roman, Georgia, serif" fill="#64748b">Bucketed means with polynomial fit</text>
  <rect x="{margin_left}" y="{margin_top}" width="{plot_width}" height="{plot_height}" fill="#fbfdff" stroke="#cbd5e1"/>
  {''.join(grid)}
  <line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#0f172a" stroke-width="1.2"/>
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#0f172a" stroke-width="1.2"/>
  {''.join(ticks)}
  {regression_svg}
  {''.join(point_svg)}
  {note}
  <text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-size="14" font-family="Times New Roman, Georgia, serif" fill="#334155">{svg_escape(x_label)}</text>
  <text x="26" y="{height / 2}" transform="rotate(-90 26 {height / 2})" text-anchor="middle" font-size="14" font-family="Times New Roman, Georgia, serif" fill="#334155">{svg_escape(y_label)}</text>
</svg>
""".strip()
    return write_text(output_path, svg)


def render_bar_svg(title: str, x_label: str, y_label: str, rows: list[dict[str, Any]], output_path: Path) -> Path:
    width, height = 920, 560
    margin_left, margin_top, margin_right, margin_bottom = 92, 76, 36, 108
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    if not rows:
        return write_text(output_path, f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><text x="50%" y="50%" text-anchor="middle">No data</text></svg>')
    max_value = max(float(row["value"]) for row in rows)
    max_value = max(max_value * 1.15, 1.0)
    gap = 12
    bar_width = max(18.0, (plot_width - gap * (len(rows) - 1)) / max(1, len(rows)))

    def map_y(v: float) -> float:
        return margin_top + plot_height - v / max_value * plot_height

    grid, ticks, bars, labels = [], [], [], []
    for idx in range(6):
        gy = margin_top + idx / 5 * plot_height
        gv = max_value - idx / 5 * max_value
        grid.append(f'<line x1="{margin_left}" y1="{gy:.1f}" x2="{margin_left + plot_width}" y2="{gy:.1f}" stroke="#dbe2ea" stroke-dasharray="4 4"/>')
        ticks.append(f'<text x="{margin_left - 12}" y="{gy + 4:.1f}" font-size="12" font-family="Times New Roman, Georgia, serif" text-anchor="end" fill="#4b5563">{gv:.2f}</text>')
    for idx, row in enumerate(rows):
        x = margin_left + idx * (bar_width + gap)
        value = float(row["value"])
        y = map_y(value)
        h = margin_top + plot_height - y
        bars.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{h:.2f}" fill="#315b8a"/>')
        labels.append(f'<text x="{x + bar_width / 2:.2f}" y="{y - 8:.2f}" font-size="10.5" font-family="Times New Roman, Georgia, serif" text-anchor="middle" fill="#111827">{value:.2f}</text>')
        labels.append(f'<text x="{x + bar_width / 2:.2f}" y="{margin_top + plot_height + 18:.2f}" font-size="10.5" font-family="Times New Roman, Georgia, serif" text-anchor="middle" fill="#334155" transform="rotate(25 {x + bar_width / 2:.2f} {margin_top + plot_height + 18:.2f})">{svg_escape(row["label"])}</text>')
    svg = f"""
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#fff"/>
  <text x="{width / 2}" y="36" text-anchor="middle" font-size="24" font-weight="700" font-family="Times New Roman, Georgia, serif" fill="#111827">{svg_escape(title)}</text>
  <text x="{width / 2}" y="58" text-anchor="middle" font-size="12" font-family="Times New Roman, Georgia, serif" fill="#64748b">Standardized timing summary</text>
  <rect x="{margin_left}" y="{margin_top}" width="{plot_width}" height="{plot_height}" fill="#fbfdff" stroke="#cbd5e1"/>
  {''.join(grid)}
  <line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#0f172a" stroke-width="1.2"/>
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#0f172a" stroke-width="1.2"/>
  {''.join(ticks)}
  {''.join(bars)}
  {''.join(labels)}
  <text x="{width / 2}" y="{height - 16}" text-anchor="middle" font-size="14" font-family="Times New Roman, Georgia, serif" fill="#334155">{svg_escape(x_label)}</text>
  <text x="26" y="{height / 2}" transform="rotate(-90 26 {height / 2})" text-anchor="middle" font-size="14" font-family="Times New Roman, Georgia, serif" fill="#334155">{svg_escape(y_label)}</text>
</svg>
""".strip()
    return write_text(output_path, svg)


def generate_figures(results: list[EpisodeTiming], output_dir: Path, report_name: str, bucket_size: int, top_slowest: int) -> dict[str, str]:
    assets_dir = output_dir / f"{report_name}_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    successful = [item for item in results if item.status == "success"]

    bucket_records = []
    for item in successful:
        bucket_start = int(item.trajectory_duration_sec // bucket_size) * bucket_size
        bucket_records.append({
            "bucket_start": bucket_start,
            "bucket_label": f"{bucket_start}s-{bucket_start + bucket_size}s",
            "trajectory_duration_sec": item.trajectory_duration_sec,
            "total_seconds": item.total_seconds,
        })
    bucket_df = pd.DataFrame(bucket_records)
    bucket_points: list[dict[str, Any]] = []
    regression, regression_label = [], None
    if not bucket_df.empty:
        grouped = bucket_df.groupby(["bucket_start", "bucket_label"], as_index=False).agg(
            avg_duration=("trajectory_duration_sec", "mean"),
            avg_runtime=("total_seconds", "mean"),
            sample_count=("total_seconds", "count"),
        ).sort_values("bucket_start")
        bucket_points = [
            {"x": float(row.avg_duration), "y": float(row.avg_runtime), "label": f"{row.bucket_label} (n={int(row.sample_count)})"}
            for row in grouped.itertuples(index=False)
        ]
        regression_series_points, equation, r2 = regression_series(
            [float(row.avg_duration) for row in grouped.itertuples(index=False)],
            [float(row.avg_runtime) for row in grouped.itertuples(index=False)],
        )
        regression = regression_series_points
        if equation:
            regression_label = equation if r2 is None else f"{equation}; R^2 = {r2:.4f}"
    duration_chart = render_scatter_svg(
        title="PNP Runtime Versus Trajectory Duration",
        x_label="Average trajectory duration (s)",
        y_label="Average processing time per episode (s)",
        points=bucket_points,
        regression=regression,
        regression_label=regression_label,
        output_path=assets_dir / "duration_vs_runtime.svg",
    )

    stage_rows = []
    for label, value in [
        ("Load", mean_or_zero(item.load_seconds for item in successful)),
        ("Normalize", mean_or_zero(item.normalize_seconds for item in successful)),
        ("R Closure", mean_or_zero(item.right_closure_seconds for item in successful)),
        ("R Align+Diff", mean_or_zero(item.right_align_diff_seconds for item in successful)),
        ("R Identify", mean_or_zero(item.right_identify_seconds for item in successful)),
        ("L Closure", mean_or_zero(item.left_closure_seconds for item in successful)),
        ("L Align+Diff", mean_or_zero(item.left_align_diff_seconds for item in successful)),
        ("L Identify", mean_or_zero(item.left_identify_seconds for item in successful)),
    ]:
        stage_rows.append({"label": label, "value": value})
    stage_chart = render_bar_svg(
        title="Average Runtime by Pipeline Stage",
        x_label="Pipeline step",
        y_label="Average time (s)",
        rows=stage_rows,
        output_path=assets_dir / "average_step_time.svg",
    )

    slow_rows = [
        {"label": f"{item.task_id}-{item.episode_id}", "value": item.total_seconds}
        for item in sorted(successful, key=lambda entry: entry.total_seconds, reverse=True)[:top_slowest]
    ]
    slow_chart = render_bar_svg(
        title="Longest Runtime Episodes",
        x_label="Task-Episode",
        y_label="Total time (s)",
        rows=slow_rows,
        output_path=assets_dir / "top_slowest_episodes.svg",
    )

    return {
        "duration_vs_runtime": str(duration_chart.relative_to(output_dir)),
        "average_step_time": str(stage_chart.relative_to(output_dir)),
        "top_slowest_episodes": str(slow_chart.relative_to(output_dir)),
    }


def build_markdown_report(results: list[EpisodeTiming], sampling: SamplingTiming, figure_paths: dict[str, str], report_wall_seconds: float, top_slowest: int) -> str:
    successful = [item for item in results if item.status == "success"]
    failed = [item for item in results if item.status != "success"]
    metric_rows = build_time_metric_rows(results)
    task_rows = build_task_timing_rows(results)
    slow_rows = build_slowest_rows(results, top_slowest)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# PNP 时间性能评测报告",
        "",
        f"生成时间：`{generated_at}`",
        "",
        "## 1. 评测范围",
        "",
        f"- 样本数：`{len(results)}`",
        f"- 成功完成计时的 episode 数：`{len(successful)}`",
        f"- 数据采样耗时：源库查询 ` {sampling.source_query_seconds:.3f}s `，invalid 过滤查询 ` {sampling.invalid_query_seconds:.3f}s `，本地筛选 ` {sampling.filter_seconds:.3f}s `",
        f"- 报告生成总耗时（含绘图）：`{report_wall_seconds:.3f}s`",
        f"- 采样策略：按 `trajectory_start, id` 顺序抽取每个任务前 N 条有效 episode，非随机抽样",
        "",
        "## 2. 核心时间结论",
        "",
        f"- 平均端到端耗时：`{mean_or_zero(item.total_seconds for item in successful):.3f}s/episode`",
        f"- P95 端到端耗时：`{percentile((item.total_seconds for item in successful), 95):.3f}s/episode`",
        f"- 平均检测阶段耗时：`{mean_or_zero(item.total_detect_seconds for item in successful):.3f}s/episode`",
        f"- 平均加载阶段耗时：`{mean_or_zero(item.load_seconds for item in successful):.3f}s/episode`",
        "",
        "## 3. 图表",
        "",
        f"![Trajectory Runtime]({figure_paths['duration_vs_runtime']})",
        "",
        f"![Average Step Time]({figure_paths['average_step_time']})",
        "",
        f"![Top Slowest Episodes]({figure_paths['top_slowest_episodes']})",
        "",
        "## 4. 分步骤时间统计",
        "",
        markdown_table(metric_rows, [
            ("metric", "步骤"),
            ("avg_seconds", "平均(s)"),
            ("p50_seconds", "P50(s)"),
            ("p95_seconds", "P95(s)"),
            ("max_seconds", "最大(s)"),
        ]),
        "",
        "## 5. 分任务时间统计",
        "",
        markdown_table(task_rows, [
            ("task_id", "任务ID"),
            ("sampled", "采样数"),
            ("avg_trajectory_duration_sec", "平均轨迹时长(s)"),
            ("avg_load_seconds", "平均加载(s)"),
            ("avg_detect_seconds", "平均检测(s)"),
            ("avg_total_seconds", "平均总耗时(s)"),
            ("p95_total_seconds", "P95总耗时(s)"),
        ]),
        "",
        "## 6. 最慢样本",
        "",
        markdown_table(slow_rows, [
            ("task_id", "任务ID"),
            ("episode_id", "Episode"),
            ("trajectory_duration_sec", "轨迹时长(s)"),
            ("load_seconds", "加载(s)"),
            ("normalize_seconds", "归一化(s)"),
            ("total_detect_seconds", "检测(s)"),
            ("total_seconds", "总耗时(s)"),
            ("status", "状态"),
        ]),
        "",
        "## 7. 异常样本",
        "",
    ]
    if failed:
        lines.append(markdown_table([
            {"task_id": item.task_id, "episode_id": item.episode_id, "error_message": item.error_message or ""}
            for item in failed
        ], [("task_id", "任务ID"), ("episode_id", "Episode"), ("error_message", "错误信息")]))
    else:
        lines.append("- 本次采样未出现异常样本。")
    return "\n".join(lines)


def html_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "".join(f"<th>{svg_escape(title)}</th>" for _, title in columns)
    body_parts = []
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key, "")
            rendered = f"{value:.3f}" if isinstance(value, float) else str(value)
            cells.append(f"<td>{svg_escape(rendered)}</td>")
        body_parts.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body_parts)}</tbody></table>"


def build_html_report(results: list[EpisodeTiming], sampling: SamplingTiming, figure_paths: dict[str, str], report_wall_seconds: float, top_slowest: int) -> str:
    successful = [item for item in results if item.status == "success"]
    failed = [item for item in results if item.status != "success"]
    metric_rows = build_time_metric_rows(results)
    task_rows = build_task_timing_rows(results)
    slow_rows = build_slowest_rows(results, top_slowest)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    failed_html = html_table([
        {"task_id": item.task_id, "episode_id": item.episode_id, "error_message": item.error_message or ""}
        for item in failed
    ], [("task_id", "任务ID"), ("episode_id", "Episode"), ("error_message", "错误信息")]) if failed else "<p>本次采样未出现异常样本。</p>"

    return f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>PNP 时间性能评测报告</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #1f2937; }}
    h1, h2 {{ color: #111827; }}
    .meta {{ color: #6b7280; margin-bottom: 24px; }}
    table {{ width: 100%; border-collapse: collapse; margin: 12px 0 24px; font-size: 14px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f4f6; }}
    ul {{ line-height: 1.7; }}
    img {{ width: 100%; max-width: 920px; border: 1px solid #e5e7eb; border-radius: 8px; margin: 10px 0 20px; }}
    code {{ background: #f3f4f6; padding: 2px 6px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>PNP 时间性能评测报告</h1>
  <div class="meta">生成时间：{svg_escape(generated_at)}</div>
  <h2>评测范围</h2>
  <ul>
    <li>样本数：<code>{len(results)}</code></li>
    <li>成功完成计时的 episode 数：<code>{len(successful)}</code></li>
    <li>数据采样耗时：源库查询 <code>{sampling.source_query_seconds:.3f}s</code>，invalid 过滤查询 <code>{sampling.invalid_query_seconds:.3f}s</code>，本地筛选 <code>{sampling.filter_seconds:.3f}s</code></li>
    <li>报告生成总耗时（含绘图）：<code>{report_wall_seconds:.3f}s</code></li>
  </ul>
  <h2>核心时间结论</h2>
  <ul>
    <li>平均端到端耗时：<code>{mean_or_zero(item.total_seconds for item in successful):.3f}s/episode</code></li>
    <li>P95 端到端耗时：<code>{percentile((item.total_seconds for item in successful), 95):.3f}s/episode</code></li>
    <li>平均检测阶段耗时：<code>{mean_or_zero(item.total_detect_seconds for item in successful):.3f}s/episode</code></li>
    <li>平均加载阶段耗时：<code>{mean_or_zero(item.load_seconds for item in successful):.3f}s/episode</code></li>
  </ul>
  <h2>图表</h2>
  <img src="{svg_escape(figure_paths['duration_vs_runtime'])}" alt="duration vs runtime" />
  <img src="{svg_escape(figure_paths['average_step_time'])}" alt="average step time" />
  <img src="{svg_escape(figure_paths['top_slowest_episodes'])}" alt="top slowest episodes" />
  <h2>分步骤时间统计</h2>
  {html_table(metric_rows, [("metric", "步骤"), ("avg_seconds", "平均(s)"), ("p50_seconds", "P50(s)"), ("p95_seconds", "P95(s)"), ("max_seconds", "最大(s)")])}
  <h2>分任务时间统计</h2>
  {html_table(task_rows, [("task_id", "任务ID"), ("sampled", "采样数"), ("avg_trajectory_duration_sec", "平均轨迹时长(s)"), ("avg_load_seconds", "平均加载(s)"), ("avg_detect_seconds", "平均检测(s)"), ("avg_total_seconds", "平均总耗时(s)"), ("p95_total_seconds", "P95总耗时(s)")])}
  <h2>最慢样本</h2>
  {html_table(slow_rows, [("task_id", "任务ID"), ("episode_id", "Episode"), ("trajectory_duration_sec", "轨迹时长(s)"), ("load_seconds", "加载(s)"), ("normalize_seconds", "归一化(s)"), ("total_detect_seconds", "检测(s)"), ("total_seconds", "总耗时(s)"), ("status", "状态")])}
  <h2>异常样本</h2>
  {failed_html}
</body>
</html>
""".strip()


def ensure_output_dir(path_str: str) -> Path:
    path = Path(path_str)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_report(output_dir: Path, report_name: str, report_format: str, markdown_text: str, html_text: str) -> list[Path]:
    paths = []
    if report_format in {"md", "both"}:
        md_path = output_dir / f"{report_name}.md"
        md_path.write_text(markdown_text, encoding="utf-8")
        paths.append(md_path)
    if report_format in {"html", "both"}:
        html_path = output_dir / f"{report_name}.html"
        html_path.write_text(html_text, encoding="utf-8")
        paths.append(html_path)
    return paths


def main() -> int:
    args = parse_args()
    report_started = time.perf_counter()
    sampled_df, sampling_timing = query_sampled_episodes(args.task_ids, args.sample_per_task, args.max_total_episodes)
    if sampled_df.empty:
        raise RuntimeError("No eligible episodes found for the requested task list.")
    results = run_timing(sampled_df)
    output_dir = ensure_output_dir(args.output_dir)
    figure_paths = generate_figures(results, output_dir, args.report_name, args.bucket_size, args.top_slowest)
    report_wall_seconds = time.perf_counter() - report_started
    markdown_text = build_markdown_report(results, sampling_timing, figure_paths, report_wall_seconds, args.top_slowest)
    html_text = build_html_report(results, sampling_timing, figure_paths, report_wall_seconds, args.top_slowest)
    written_paths = write_report(output_dir, args.report_name, args.format, markdown_text, html_text)
    print(json.dumps({
        "sampled_episodes": len(results),
        "successful_episodes": sum(1 for item in results if item.status == "success"),
        "avg_total_seconds": mean_or_zero(item.total_seconds for item in results if item.status == "success"),
        "report_paths": [str(path) for path in written_paths],
        "figure_paths": figure_paths,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
