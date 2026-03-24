#!/usr/bin/env python3
"""将 report 中缺失于 12dof_ids.json 的 id 追加回去。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_existing_path(preferred: Path, fallbacks: list[Path]) -> Path:
    if preferred.exists():
        return preferred
    for p in fallbacks:
        if p.exists():
            return p
    return preferred


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def extract_report_ids(report: dict[str, Any]) -> list[str]:
    # 优先取明确的 12dof 列表
    if isinstance(report.get("ids_12dof_per_side"), list):
        return [str(x) for x in report["ids_12dof_per_side"] if str(x).strip()]

    rows = report.get("rows")
    if not isinstance(rows, list):
        return []

    # 其次取 rows 中标记为 12dof 的项
    marked = [
        str(row.get("episode_id"))
        for row in rows
        if isinstance(row, dict)
        and row.get("episode_id") is not None
        and bool(row.get("is_12dof_per_side"))
    ]
    marked = [x for x in marked if x.strip()]
    if marked:
        return marked

    # 最后兜底：rows 里所有 episode_id
    return [
        str(row.get("episode_id"))
        for row in rows
        if isinstance(row, dict) and row.get("episode_id") is not None and str(row.get("episode_id")).strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 episode_hand_scan_report.json 中缺失的 12dof id 追加到 12dof_ids.json"
    )
    parser.add_argument(
        "--ids-json",
        type=Path,
        default=PROJECT_ROOT / "12dof_ids.json",
        help="12dof id 列表文件（默认: 项目根目录 12dof_ids.json）",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=PROJECT_ROOT / "episode_hand_scan_report.json",
        help="扫描报告文件（默认优先项目根目录，不存在则自动尝试 scripts/episode_hand_scan_report.json）",
    )
    args = parser.parse_args()

    ids_data = load_json(args.ids_json)
    if not isinstance(ids_data, list):
        raise ValueError(f"{args.ids_json} 格式错误，期望 JSON 数组")

    report_path = resolve_existing_path(
        args.report_json,
        [PROJECT_ROOT / "scripts" / "episode_hand_scan_report.json"],
    )
    report_data = load_json(report_path)
    if not isinstance(report_data, dict):
        raise ValueError(f"{args.report_json} 格式错误，期望 JSON 对象")

    current_ids = [str(x) for x in ids_data if str(x).strip()]
    current_set = set(current_ids)

    report_ids = extract_report_ids(report_data)

    to_append: list[str] = []
    for eid in report_ids:
        if eid not in current_set:
            current_set.add(eid)
            to_append.append(eid)

    updated_ids = current_ids + to_append
    atomic_write_json(args.ids_json, updated_ids)

    print(f"追加了 {len(to_append)} 条 id 到 {args.ids_json}（report: {report_path}）")


if __name__ == "__main__":
    main()
