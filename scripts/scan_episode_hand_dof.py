#!/usr/bin/env python3
"""扫描指定 episodes 的手部关节自由度，并导出 12+12 DoF 的 episode id。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.utils.data_parser import resolve_joint_paths
from src.utils.s3_client import download_s3_file
from src.utils.source_db import query_df

# 项目当前已知的 6x2 手部关节名
KNOWN_HAND_JOINTS = {
    "L_pinky_proximal_joint",
    "L_ring_proximal_joint",
    "L_middle_proximal_joint",
    "L_index_proximal_joint",
    "L_thumb_proximal_pitch_joint",
    "L_thumb_proximal_yaw_joint",
    "R_pinky_proximal_joint",
    "R_ring_proximal_joint",
    "R_middle_proximal_joint",
    "R_index_proximal_joint",
    "R_thumb_proximal_pitch_joint",
    "R_thumb_proximal_yaw_joint",
}


@dataclass
class EpisodeRecord:
    episode_id: str
    range_tag: str
    trajectory_start: str | None


@dataclass
class ProbeResult:
    episode_id: str
    file_path: str | None
    joint_type: str | None
    joint_key: str | None
    matched_known_joint_names: list[str]
    hand_dims: dict[str, Any]
    detected_dof_per_side: int | None


def fetch_target_episodes() -> list[EpisodeRecord]:
    sql = """
        SELECT id::text AS episode_id,
               'station_id=14@2025-11-16~2026-01-13'::text AS range_tag,
               trajectory_start
        FROM episodes
        WHERE station_id = 14
          AND trajectory_start >= DATE '2025-11-16'
          AND trajectory_start < DATE '2026-01-14'

        ORDER BY trajectory_start, episode_id
    """
    df = query_df(sql)
    if df.empty:
        return []

    records: list[EpisodeRecord] = []
    for _, row in df.iterrows():
        ts = row.get("trajectory_start")
        ts_str = None if pd.isna(ts) else str(ts)
        records.append(
            EpisodeRecord(
                episode_id=str(row["episode_id"]),
                range_tag=str(row["range_tag"]),
                trajectory_start=ts_str,
            )
        )
    return records


def fetch_rgb_file_paths(episode_ids: list[str]) -> dict[str, str]:
    if not episode_ids:
        return {}

    sql = """
        SELECT DISTINCT ON (s.episode_id)
               s.episode_id::text AS episode_id,
               s.file_path
        FROM streams s
        WHERE s.episode_id::text = ANY(%(episode_ids)s)
          AND s.stream_name = 'rgb'
        ORDER BY
          s.episode_id,
          CASE
            WHEN POSITION('camera_top.parquet' IN s.file_path) > 0 THEN 0
            WHEN POSITION('depth' IN s.file_path) > 0 THEN 2
            ELSE 1
          END,
          s.file_path
    """
    df = query_df(sql, {"episode_ids": episode_ids})
    if df.empty:
        return {}

    return {str(row["episode_id"]): str(row["file_path"]) for _, row in df.iterrows()}


def _detect_hdf5_hand_dof(local_hdf5: Path) -> tuple[dict[str, Any], int | None]:
    import h5py

    hand_dims: dict[str, Any] = {}
    detected_dof_per_side: int | None = None

    with h5py.File(local_hdf5, "r") as f:
        action_shape = tuple(f["action/hand"].shape) if "action/hand" in f else None
        state_shape = tuple(f["state/hand"].shape) if "state/hand" in f else None

        hand_dims["action_hand_shape"] = action_shape
        hand_dims["state_hand_shape"] = state_shape

        cand_dim = None
        if action_shape and len(action_shape) >= 2:
            cand_dim = int(action_shape[1])
        elif state_shape and len(state_shape) >= 2:
            cand_dim = int(state_shape[1])

        if cand_dim is not None and cand_dim > 0:
            # 默认左右拼接在第二维
            detected_dof_per_side = cand_dim // 2 if cand_dim % 2 == 0 else None

        # 额外记录可能的名称数据集，方便后续人工核对
        name_like_keys = [k for k in f.keys() if "name" in k.lower() or "joint" in k.lower()]
        if name_like_keys:
            hand_dims["root_name_like_keys"] = sorted(name_like_keys)

    return hand_dims, detected_dof_per_side


def _collect_parquet_joint_names(local_parquet: Path, col_name: str) -> set[str]:
    names: set[str] = set()
    try:
        df = pd.read_parquet(local_parquet, columns=[col_name])
    except Exception:
        return names

    if col_name not in df.columns:
        return names

    for val in df[col_name].head(200):
        if val is None:
            continue
        try:
            for item in val:
                if isinstance(item, dict) and "name" in item:
                    names.add(str(item["name"]))
        except Exception:
            continue
    return names


def _detect_parquet_hand_dof(action_local: Path | None, state_local: Path | None) -> tuple[list[str], dict[str, Any], int | None]:
    names: set[str] = set()

    if action_local and action_local.exists():
        names.update(_collect_parquet_joint_names(action_local, "action"))
    if state_local and state_local.exists():
        names.update(_collect_parquet_joint_names(state_local, "observation.state"))

    matched_known = sorted(n for n in names if n in KNOWN_HAND_JOINTS)

    left_known = [n for n in matched_known if n.startswith("L_")]
    right_known = [n for n in matched_known if n.startswith("R_")]

    hand_tokens = ("thumb", "index", "middle", "ring", "pinky", "finger", "hand")
    left_candidates = {
        n
        for n in names
        if ("left" in n.lower() or n.startswith("L_") or n.startswith("l_"))
        and any(tok in n.lower() for tok in hand_tokens)
    }
    right_candidates = {
        n
        for n in names
        if ("right" in n.lower() or n.startswith("R_") or n.startswith("r_"))
        and any(tok in n.lower() for tok in hand_tokens)
    }

    # parquet 无固定二维 hand 数组，采用命名启发式估计是否左右各12自由度
    dof_per_side = 12 if len(left_candidates) >= 12 and len(right_candidates) >= 12 else None

    meta = {
        "parquet_joint_name_count": len(names),
        "matched_known_joint_count": len(matched_known),
        "left_known_count": len(left_known),
        "right_known_count": len(right_known),
        "left_hand_candidate_count": len(left_candidates),
        "right_hand_candidate_count": len(right_candidates),
    }
    return matched_known, meta, dof_per_side


def probe_episode(episode_id: str, file_path: str | None) -> ProbeResult:
    if not file_path:
        return ProbeResult(
            episode_id=episode_id,
            file_path=None,
            joint_type=None,
            joint_key=None,
            matched_known_joint_names=[],
            hand_dims={"error": "missing_rgb_file_path"},
            detected_dof_per_side=None,
        )

    joint_info = resolve_joint_paths(file_path)
    if not joint_info:
        return ProbeResult(
            episode_id=episode_id,
            file_path=file_path,
            joint_type=None,
            joint_key=None,
            matched_known_joint_names=[],
            hand_dims={"error": "joint_path_not_found"},
            detected_dof_per_side=None,
        )

    jtype = str(joint_info.get("type"))

    if jtype == "hdf5":
        key = str(joint_info.get("key"))
        local = download_s3_file(key)
        if local is None:
            return ProbeResult(
                episode_id=episode_id,
                file_path=file_path,
                joint_type=jtype,
                joint_key=key,
                matched_known_joint_names=[],
                hand_dims={"error": "s3_download_failed"},
                detected_dof_per_side=None,
            )

        hand_dims, dof_per_side = _detect_hdf5_hand_dof(local)
        # hdf5 的手部通常无名称，这里仅保留项目已知 12 个名字做对齐说明
        matched_known = sorted(KNOWN_HAND_JOINTS) if dof_per_side is not None and dof_per_side >= 6 else []

        return ProbeResult(
            episode_id=episode_id,
            file_path=file_path,
            joint_type=jtype,
            joint_key=key,
            matched_known_joint_names=matched_known,
            hand_dims=hand_dims,
            detected_dof_per_side=dof_per_side,
        )

    action_key = str(joint_info.get("action_key")) if joint_info.get("action_key") else None
    state_key = str(joint_info.get("state_key")) if joint_info.get("state_key") else None

    action_local = download_s3_file(action_key) if action_key else None
    state_local = download_s3_file(state_key) if state_key else None

    matched_known, hand_dims, dof_per_side = _detect_parquet_hand_dof(action_local, state_local)
    key_summary = ",".join([k for k in [action_key, state_key] if k])

    return ProbeResult(
        episode_id=episode_id,
        file_path=file_path,
        joint_type=jtype,
        joint_key=key_summary,
        matched_known_joint_names=matched_known,
        hand_dims=hand_dims,
        detected_dof_per_side=dof_per_side,
    )


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def render_progress(current: int, total: int, hit_12dof: int, width: int = 24) -> None:
    if total <= 0:
        print("\r  [无数据可扫描]", end="", flush=True)
        return
    done = max(0, min(current, total))
    ratio = done / total
    filled = int(ratio * width)
    bar = "#" * filled + "-" * (width - filled)
    print(
        f"\r  [{bar}] {done}/{total} | 12dof_per_side={hit_12dof}",
        end="",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="扫描指定时间段 episodes 的手部关节自由度")
    parser.add_argument(
        "--probe-episode-id",
        default="652465",
        help="先用于探测手部关节格式的 episode id（默认: 652465）",
    )
    parser.add_argument(
        "--output-ids-json",
        type=Path,
        default=PROJECT_ROOT / "hand_12dof_episode_ids.json",
        help="输出 12+12 自由度 episode id 列表（默认: 项目根目录 hand_12dof_episode_ids.json）",
    )
    parser.add_argument(
        "--output-report-json",
        type=Path,
        default=PROJECT_ROOT / "scripts" / "episode_hand_scan_report.json",
        help="输出完整扫描报告（默认: scripts/episode_hand_scan_report.json）",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="仅扫描前 N 条目标 episode（0 表示全部）",
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="只执行 probe episode 探测，不执行全量扫描",
    )
    args = parser.parse_args()

    print("[1/4] 查询目标 episodes ...")
    records = fetch_target_episodes()
    if args.limit > 0:
        records = records[: args.limit]
    print(f"  命中 episodes: {len(records)}")

    all_ids = [r.episode_id for r in records]
    print("[2/4] 查询 rgb file_path ...")
    id_to_file = fetch_rgb_file_paths(all_ids + [str(args.probe_episode_id)])
    print(f"  命中 rgb file_path: {len(id_to_file)}")

    print(f"[3/4] 先探测 episode_id={args.probe_episode_id} 的手部结构 ...")
    probe = probe_episode(str(args.probe_episode_id), id_to_file.get(str(args.probe_episode_id)))
    probe_payload = {
        "episode_id": probe.episode_id,
        "file_path": probe.file_path,
        "joint_type": probe.joint_type,
        "joint_key": probe.joint_key,
        "matched_known_joint_names": probe.matched_known_joint_names,
        "hand_dims": probe.hand_dims,
        "detected_dof_per_side": probe.detected_dof_per_side,
    }
    print(json.dumps(probe_payload, ensure_ascii=False, indent=2))

    if args.probe_only:
        atomic_write_json(
            args.output_report_json,
            {"probe": probe_payload, "total_episodes": len(records), "rows": []},
        )
        print(f"\n仅 probe 完成，报告已输出: {args.output_report_json}")
        return

    print("[4/4] 扫描目标 episodes 的手部 DoF ...")
    report_rows: list[dict[str, Any]] = []
    ids_12dof: list[str] = []
    render_progress(0, len(records), 0)

    for idx, rec in enumerate(records, start=1):
        file_path = id_to_file.get(rec.episode_id)
        result = probe_episode(rec.episode_id, file_path)

        is_12dof_per_side = result.detected_dof_per_side == 12
        if is_12dof_per_side:
            ids_12dof.append(rec.episode_id)

        row = {
            "episode_id": rec.episode_id,
            "range_tag": rec.range_tag,
            "trajectory_start": rec.trajectory_start,
            "file_path": result.file_path,
            "joint_type": result.joint_type,
            "joint_key": result.joint_key,
            "matched_known_joint_names": result.matched_known_joint_names,
            "hand_dims": result.hand_dims,
            "detected_dof_per_side": result.detected_dof_per_side,
            "is_12dof_per_side": is_12dof_per_side,
        }
        report_rows.append(row)
        render_progress(idx, len(records), len(ids_12dof))
    print()

    atomic_write_json(args.output_ids_json, ids_12dof)
    atomic_write_json(
        args.output_report_json,
        {
            "probe": probe_payload,
            "total_episodes": len(records),
            "episodes_with_12dof_per_side": len(ids_12dof),
            "ids_12dof_per_side": ids_12dof,
            "rows": report_rows,
        },
    )

    print(f"\n已输出 12DoF(左右各12) id 列表: {args.output_ids_json}")
    print(f"已输出完整报告: {args.output_report_json}")


if __name__ == "__main__":
    main()
