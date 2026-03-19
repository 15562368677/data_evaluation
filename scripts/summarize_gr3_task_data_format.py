import argparse
import json
import re
import sys
import time
import warnings
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_parser import resolve_joint_paths
from src.utils.source_db import query_df


TASK_ID_PATTERN = re.compile(r"task_id\s*=\s*['\"]?([^'\"\s]+)['\"]?", re.IGNORECASE)


class TaskProgress:
    def __init__(self, task_id: str, total: int, width: int = 24):
        self.task_id = task_id
        self.total = max(0, total)
        self.width = width
        self.last_render_at = 0.0

    def render(self, current: int, parquet_count: int, hdf5_count: int, force: bool = False):
        now = time.time()
        if not force and (now - self.last_render_at) < 0.2:
            return
        self.last_render_at = now

        if self.total <= 0:
            line = f"\r  task_id={self.task_id} [无记录]"
            print(line, end="", flush=True)
            return

        done = min(max(current, 0), self.total)
        ratio = done / self.total
        filled = int(ratio * self.width)
        bar = "#" * filled + "-" * (self.width - filled)
        line = (
            f"\r  task_id={self.task_id} [{bar}] {done}/{self.total} "
            f"(parquet={parquet_count}, hdf5={hdf5_count})"
        )
        print(line, end="", flush=True)

    def finish(self, current: int, parquet_count: int, hdf5_count: int):
        self.render(current, parquet_count, hdf5_count, force=True)
        print()


def extract_task_ids(config_path: Path) -> list[str]:
    with config_path.open("r", encoding="utf-8") as f:
        items = json.load(f)

    task_ids: list[str] = []
    seen: set[str] = set()

    if not isinstance(items, list):
        raise ValueError(f"{config_path} 必须是数组格式")

    for item in items:
        if not isinstance(item, dict):
            continue
        query = str(item.get("query", "")).strip()
        if not query:
            continue
        match = TASK_ID_PATTERN.search(query)
        if not match:
            continue
        task_id = match.group(1).strip()
        if task_id and task_id not in seen:
            seen.add(task_id)
            task_ids.append(task_id)

    return task_ids


def fetch_episode_file_paths(task_id: str) -> list[tuple[str, str]]:
    sql = """
        SELECT e.id::text AS episode_id, s.file_path
        FROM episodes e
        JOIN streams s ON s.episode_id = e.id
        WHERE e.task_id::text = %(task_id)s
          AND s.stream_name = 'rgb'
        ORDER BY e.id
    """
    df = query_df(sql, {"task_id": task_id})
    if df.empty:
        return []
    return [(str(row["episode_id"]), str(row["file_path"])) for _, row in df.iterrows()]


def classify_episode(file_path: str) -> str:
    joint_info = resolve_joint_paths(file_path)
    if not joint_info:
        return "unknown"
    return str(joint_info.get("type", "unknown"))


def summarize_task(task_id: str) -> dict:
    rows = fetch_episode_file_paths(task_id)
    total = len(rows)
    parquet_count = 0
    hdf5_count = 0
    unknown_count = 0
    progress = TaskProgress(task_id=task_id, total=total)
    progress.render(0, parquet_count, hdf5_count, force=True)

    for i, (_, file_path) in enumerate(rows, start=1):
        fmt = classify_episode(file_path)
        if fmt == "parquet":
            parquet_count += 1
        elif fmt == "hdf5":
            hdf5_count += 1
        else:
            unknown_count += 1
        progress.render(i, parquet_count, hdf5_count)

    progress.finish(total, parquet_count, hdf5_count)

    result = {
        "task_id": task_id,
        "parquet": parquet_count,
        "hdf5": hdf5_count,
    }
    if unknown_count:
        result["unknown"] = unknown_count
    return result


def _atomic_write_json(path: Path, data: dict | list):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def load_checkpoint(checkpoint_path: Path) -> list[dict]:
    if not checkpoint_path.exists():
        return []
    with checkpoint_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"checkpoint 文件格式错误: {checkpoint_path}")
    return data


def save_checkpoint(checkpoint_path: Path, results: list[dict]):
    _atomic_write_json(checkpoint_path, results)


def main():
    warnings.filterwarnings(
        "ignore",
        message=(
            "pandas only supports SQLAlchemy connectable "
            r"\(engine/connection\) or database string URI"
        ),
        category=UserWarning,
    )

    parser = argparse.ArgumentParser(
        description="根据 gr3_task_configs.json 统计每个 task_id 的 parquet/hdf5 数据条数"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "gr3_task_configs.json",
        help="任务配置文件路径，默认 gr3_task_configs.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "scripts" / "gr3_task_data_format_summary.json",
        help="输出结果路径",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="仅处理前 N 个 task_id（0 表示全部）",
    )
    parser.add_argument(
        "--task-id",
        dest="task_ids",
        action="append",
        default=[],
        help="仅处理指定 task_id；可重复传入，如 --task-id 752 --task-id 839",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "scripts" / "gr3_task_data_format_summary.checkpoint.json",
        help="断点续跑文件路径",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="忽略 checkpoint，强制从头开始",
    )
    args = parser.parse_args()

    task_ids = extract_task_ids(args.config)
    if args.task_ids:
        selected = {str(t).strip() for t in args.task_ids if str(t).strip()}
        task_ids = [task_id for task_id in task_ids if task_id in selected]
    if args.limit > 0:
        task_ids = task_ids[: args.limit]

    print(f"将处理 task_id 数量: {len(task_ids)}")

    results: list[dict] = []
    if not args.restart:
        try:
            results = load_checkpoint(args.checkpoint)
            if results:
                print(f"检测到 checkpoint，已恢复 {len(results)} 个 task_id: {args.checkpoint}")
        except Exception as e:
            print(f"读取 checkpoint 失败，将从头开始: {e}")
            results = []
    else:
        print("已指定 --restart，忽略历史 checkpoint。")

    task_ids_set = set(task_ids)
    filtered_results: list[dict] = []
    done_task_ids: set[str] = set()
    for item in results:
        if not isinstance(item, dict):
            continue
        task_id = str(item.get("task_id", "")).strip()
        if task_id and task_id in task_ids_set and task_id not in done_task_ids:
            filtered_results.append(item)
            done_task_ids.add(task_id)
    results = filtered_results

    for idx, task_id in enumerate(task_ids, start=1):
        if task_id in done_task_ids:
            print(f"[{idx}/{len(task_ids)}] task_id={task_id} (已完成，跳过)")
            continue
        print(f"[{idx}/{len(task_ids)}] task_id={task_id}")
        result = summarize_task(task_id)
        results.append(result)
        done_task_ids.add(task_id)
        save_checkpoint(args.checkpoint, results)

    results_by_task_id = {str(item["task_id"]): item for item in results if isinstance(item, dict) and "task_id" in item}
    ordered_results = [results_by_task_id[task_id] for task_id in task_ids if task_id in results_by_task_id]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(ordered_results, f, ensure_ascii=False, indent=2)

    print(f"已输出: {args.output}")
    if args.checkpoint.exists():
        try:
            args.checkpoint.unlink()
            print(f"checkpoint 已清理: {args.checkpoint}")
        except Exception as e:
            print(f"checkpoint 清理失败，可手动删除: {args.checkpoint} ({e})")


if __name__ == "__main__":
    main()
