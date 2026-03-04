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
        return pd.read_sql(sql, conn, params=params)

def init_pnp_db():
    import logging
    try:
        conn = get_pnp_connection()
        with conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS batches (
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
                batch_id VARCHAR(255) REFERENCES batches(uniq_id),
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
    """初始化 duration_result 表"""
    import logging
    try:
        conn = get_pnp_connection()
        with conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS duration_result (
                id SERIAL PRIMARY KEY,
                episode_id VARCHAR(255) NOT NULL,
                task_id VARCHAR(255),
                "pass" BOOLEAN DEFAULT FALSE,
                fast BOOLEAN DEFAULT FALSE,
                slow BOOLEAN DEFAULT FALSE,
                invalid BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(episode_id)
            );
            """)
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"Failed to initialize duration_result table: {e}")


def save_duration_results(records: list):
    """批量保存 duration 检测结果到 duration_result 表。

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
                is_pass = label == "pass"
                is_fast = label == "fast"
                is_slow = label == "slow"
                is_invalid = label == "invalid"
                cur.execute("""
                    INSERT INTO duration_result (episode_id, task_id, "pass", fast, slow, invalid)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (episode_id) DO UPDATE SET
                        task_id = EXCLUDED.task_id,
                        "pass" = EXCLUDED."pass",
                        fast = EXCLUDED.fast,
                        slow = EXCLUDED.slow,
                        invalid = EXCLUDED.invalid,
                        created_at = CURRENT_TIMESTAMP
                """, (ep_id, task_id, is_pass, is_fast, is_slow, is_invalid))
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
                f'SELECT episode_id, "pass", fast, slow, invalid '
                f"FROM duration_result WHERE episode_id IN ({placeholders})",
                [str(eid) for eid in episode_ids],
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    result = {}
    for row in rows:
        ep_id = str(row[0])
        if row[1]:
            result[ep_id] = "pass"
        elif row[2]:
            result[ep_id] = "fast"
        elif row[3]:
            result[ep_id] = "slow"
        elif row[4]:
            result[ep_id] = "invalid"
        else:
            result[ep_id] = "pass"  # fallback
    return result
