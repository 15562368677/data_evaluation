"""数据库连接模块 (PnP Result)"""

import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def get_pnp_connection():
    """获取pnp_result数据库连接"""
    return psycopg2.connect(
        host=os.getenv("PNP_DB_HOST"),
        port=int(os.getenv("PNP_DB_PORT")),
        user=os.getenv("PNP_DB_USER"),
        password=os.getenv("PNP_DB_PASSWORD"),
        database=os.getenv("PNP_DB_NAME")
    )

def query_pnp_df(sql: str, params=None):
    """执行pnp查询并返回 DataFrame"""
    import pandas as pd

    with get_pnp_connection() as conn:
        return pd.read_sql_query(sql, conn, params=params)

def init_pnp_db():
    import logging
    try:
        conn = get_pnp_connection()
        with conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS pnp_batches (
                uniq_id VARCHAR(255) PRIMARY KEY,
                task_id VARCHAR(255),
                sample_ratio INT,
                is_overwrite BOOLEAN,
                parameters JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            cur.execute("""
            CREATE TABLE IF NOT EXISTS pnp_streams (
                id SERIAL PRIMARY KEY,
                episode_id VARCHAR(255) NOT NULL,
                batch_id VARCHAR(255) REFERENCES pnp_batches(uniq_id),
                pnp_result JSONB,
                right_pnp_result JSONB,
                left_pnp_result JSONB,
                checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(episode_id, batch_id)
            );
            """)
            
            # If the table already existed without these columns, add them
            cur.execute("""
            ALTER TABLE pnp_streams 
            ADD COLUMN IF NOT EXISTS right_pnp_result JSONB,
            ADD COLUMN IF NOT EXISTS left_pnp_result JSONB;
            """)
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"Failed to initialize pnp_result database: {e}")


def init_duration_result_db():
    """初始化 duration_results 表"""
    import logging
    try:
        conn = get_pnp_connection()
        with conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS duration_results (
                id SERIAL PRIMARY KEY,
                episode_id VARCHAR(255) NOT NULL,
                task_id VARCHAR(255),
                duration_result VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(episode_id)
            );
            """)
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"Failed to initialize duration_results table: {e}")


def save_duration_results(records: list):
    """批量保存 duration 检测结果到 duration_results 表。

    每条 record 结构: {"episode_id": str, "task_id": str, "label": str}
    label 取值: "pass" / "fast" / "slow" / "invalid"
    """
    if not records:
        return 0

    conn = get_pnp_connection()
    count = 0
    try:
        with conn.cursor() as cur:
            for rec in records:
                ep_id = str(rec.get("episode_id", ""))
                task_id = str(rec.get("task_id", ""))
                label = rec.get("label", "")
                
                cur.execute("""
                    INSERT INTO duration_results (episode_id, task_id, duration_result)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (episode_id) DO UPDATE SET
                        task_id = EXCLUDED.task_id,
                        duration_result = EXCLUDED.duration_result,
                        created_at = CURRENT_TIMESTAMP
                """, (ep_id, task_id, label))
                count += 1
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return count


def query_checked_episodes(episode_ids: list):
    """查询已检测的 episode 及其标签结果。

    参数: episode_ids — 待检查的 episode_id 列表
    返回: dict, key=episode_id(str), value=label(str: pass/fast/slow/invalid)
    """
    if not episode_ids:
        return {}

    conn = get_pnp_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(episode_ids))
            cur.execute(
                f"SELECT episode_id, duration_result "
                f"FROM duration_results WHERE episode_id IN ({placeholders})",
                [str(eid) for eid in episode_ids],
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    result = {}
    for row in rows:
        ep_id = str(row[0])
        label = row[1] if row[1] else "pass"
        result[ep_id] = label
    return result

def init_pnp_result_db():
    """初始化 pnp_results 表"""
    import logging
    try:
        conn = get_pnp_connection()
        with conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS pnp_results (
                id SERIAL PRIMARY KEY,
                episode_id VARCHAR(255) NOT NULL,
                task_id VARCHAR(255),
                pnp_result VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(episode_id)
            );
            """)
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"Failed to initialize pnp_results table: {e}")

def save_pnp_results(records: list):
    """批量保存 pnp 检测结果到 pnp_results 表。

    每条 record 结构: {"episode_id": str, "task_id": str, "label": str}
    label 取值: "pass" / "multi_pick" / "fail_pick" / "invalid"
    """
    if not records:
        return 0

    conn = get_pnp_connection()
    count = 0
    try:
        with conn.cursor() as cur:
            for rec in records:
                ep_id = str(rec.get("episode_id", ""))
                task_id = str(rec.get("task_id", ""))
                label = rec.get("label", "")
                
                cur.execute("""
                    INSERT INTO pnp_results (episode_id, task_id, pnp_result)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (episode_id) DO UPDATE SET
                        task_id = EXCLUDED.task_id,
                        pnp_result = EXCLUDED.pnp_result,
                        created_at = CURRENT_TIMESTAMP
                """, (ep_id, task_id, label))
                count += 1
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return count

def query_checked_pnp_episodes(episode_ids: list):
    """查询已检测的 episode 及其标签结果。

    参数: episode_ids — 待检查的 episode_id 列表
    返回: dict, key=episode_id(str), value=label(str: pass/multi_pick/fail_pick/invalid)
    """
    if not episode_ids:
        return {}

    conn = get_pnp_connection()
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(episode_ids))
            cur.execute(
                f"SELECT episode_id, pnp_result "
                f"FROM pnp_results WHERE episode_id IN ({placeholders})",
                [str(eid) for eid in episode_ids],
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    result = {}
    for row in rows:
        ep_id = str(row[0])
        label = row[1] if row[1] else "pass"
        result[ep_id] = label
    return result
