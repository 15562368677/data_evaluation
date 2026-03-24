#!/usr/bin/env python3
"""统计 assets/id.txt 中 episodes 的 quest+pedal / quest 数量。

规则:
- 对每个 episode_id, 从 streams 表取 rgb 对应 file_path
- 解析出对应 S3 前缀
- 若该前缀下任意对象名(文件名)包含子串 "base" (不区分大小写), 归类为 quest+pedal
- 否则归类为 quest
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.source_db import query_df
from src.utils.s3_client import _get_minio_client, S3_BUCKET


def read_episode_ids(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"id 文件不存在: {path}")
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#"):
            continue
        # 支持逗号/中文逗号/空白混合分隔
        parts = re.split(r"[,\uFF0C\s]+", raw)
        for p in parts:
            s = p.strip()
            if not s or s.startswith("#"):
                continue
            ids.append(s)
    return ids


def get_rgb_file_paths_bulk(episode_ids: list[str], chunk_size: int = 1000) -> dict[str, str]:
    """批量查询 episode_id -> rgb file_path，减少 DB 往返次数。"""
    result: dict[str, str] = {}
    if not episode_ids:
        return result

    # 优先 camera_top.parquet，尽量避开 depth parquet
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

    for i in range(0, len(episode_ids), chunk_size):
        chunk = episode_ids[i : i + chunk_size]
        df = query_df(sql, {"episode_ids": chunk})
        if df.empty:
            continue
        for _, row in df.iterrows():
            result[str(row["episode_id"])] = str(row["file_path"])
    return result


def s3_prefix_from_file_path(file_path: str) -> str:
    # 与 data_parser 的规则对齐，统一补 factory/ 前缀
    path = file_path.strip().rstrip("/")

    if path.endswith("/top/rgb") or path.endswith("/top/depth"):
        if path.endswith("/top/rgb"):
            base = path[: -len("/top/rgb")]
        else:
            base = path[: -len("/top/depth")]
    elif "/" in path:
        base = path.rsplit("/", 1)[0]
    else:
        base = path

    if not base.startswith("factory/"):
        base = f"factory/{base}"
    return base.rstrip("/") + "/"


def has_base_object(prefix: str) -> bool:
    client = _get_minio_client()
    objs: Iterable = client.list_objects(S3_BUCKET, prefix=prefix, recursive=True)
    for obj in objs:
        name = Path(obj.object_name).name.lower()
        if "base" in name:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="统计 id 列表中 episodes 在 S3 上是否命中 base 文件名"
    )
    parser.add_argument(
        "--id-file",
        default="assets/id.txt",
        help="episode id 列表文件路径（默认: assets/id.txt）",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="每处理多少条打印一次进度（默认: 50）",
    )
    parser.add_argument(
        "--db-chunk-size",
        type=int,
        default=1000,
        help="数据库批量查询 chunk 大小（默认: 1000）",
    )
    args = parser.parse_args()

    id_file = Path(args.id_file)
    episode_ids = read_episode_ids(id_file)
    id_to_file = get_rgb_file_paths_bulk(episode_ids, chunk_size=args.db_chunk_size)

    quest_pedal = 0
    quest = 0
    missing_stream = 0
    failed = 0
    prefix_cache: dict[str, bool] = {}

    try:
        for idx, episode_id in enumerate(episode_ids, start=1):
            file_path = id_to_file.get(episode_id)
            if not file_path:
                missing_stream += 1
                if args.progress_every > 0 and idx % args.progress_every == 0:
                    print(
                        f"[progress] {idx}/{len(episode_ids)} | "
                        f"quest+pedal={quest_pedal}, quest={quest}, missing_stream={missing_stream}, failed={failed}"
                    )
                continue

            prefix = s3_prefix_from_file_path(file_path)
            if prefix not in prefix_cache:
                prefix_cache[prefix] = has_base_object(prefix)

            if prefix_cache[prefix]:
                quest_pedal += 1
            else:
                quest += 1

            if args.progress_every > 0 and idx % args.progress_every == 0:
                print(
                    f"[progress] {idx}/{len(episode_ids)} | "
                    f"quest+pedal={quest_pedal}, quest={quest}, missing_stream={missing_stream}, failed={failed}"
                )
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] 用户中断，以下为当前已完成统计：")
    except Exception as e:
        failed += 1
        print(f"[ERROR] 运行失败: {e}")

    print("=== 统计结果 ===")
    print(f"总id数: {len(episode_ids)}")
    print(f"quest+pedal: {quest_pedal}")
    print(f"quest: {quest}")
    print(f"missing_stream: {missing_stream}")
    print(f"failed: {failed}")


if __name__ == "__main__":
    main()
