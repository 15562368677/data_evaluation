"""数据库连接模块 (PnP Result)"""

import os

import psycopg2
from psycopg2.extras import Json
from dotenv import load_dotenv

from src.validators.base import IssueLevel, ValidationResult

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
                status VARCHAR(20) DEFAULT 'queued',
                total_episodes INT DEFAULT 0,
                processed_episodes INT DEFAULT 0,
                failed_episodes INT DEFAULT 0,
                last_heartbeat TIMESTAMP,
                error_message TEXT,
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
            cur.execute("""
            CREATE TABLE IF NOT EXISTS pnp_failures (
                id SERIAL PRIMARY KEY,
                episode_id VARCHAR(255) NOT NULL,
                batch_id VARCHAR(255) REFERENCES pnp_batches(uniq_id),
                error_message TEXT,
                failed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(episode_id, batch_id)
            );
            """)
            
            # If the table already existed without these columns, add them
            cur.execute("""
            ALTER TABLE pnp_streams 
            ADD COLUMN IF NOT EXISTS right_pnp_result JSONB,
            ADD COLUMN IF NOT EXISTS left_pnp_result JSONB;
            """)

            # Backward-compatible migration for existing pnp_batches
            cur.execute("""
            ALTER TABLE pnp_batches
            ADD COLUMN IF NOT EXISTS status VARCHAR(20) DEFAULT 'queued',
            ADD COLUMN IF NOT EXISTS total_episodes INT DEFAULT 0,
            ADD COLUMN IF NOT EXISTS processed_episodes INT DEFAULT 0,
            ADD COLUMN IF NOT EXISTS failed_episodes INT DEFAULT 0,
            ADD COLUMN IF NOT EXISTS last_heartbeat TIMESTAMP,
            ADD COLUMN IF NOT EXISTS error_message TEXT;
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


def delete_duration_results(episode_ids: list):
    """按 episode_id 列表删除 duration_results 记录，返回删除条数。"""
    if not episode_ids:
        return 0

    conn = get_pnp_connection()
    deleted = 0
    try:
        with conn.cursor() as cur:
            placeholders = ",".join(["%s"] * len(episode_ids))
            cur.execute(
                f"DELETE FROM duration_results WHERE episode_id IN ({placeholders})",
                [str(eid) for eid in episode_ids],
            )
            deleted = cur.rowcount or 0
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return deleted

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


def init_qc_result_db():
    """初始化统一 QC 结果表。"""
    import logging

    try:
        conn = get_pnp_connection()
        with conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS test_qc_results (
                id BIGSERIAL PRIMARY KEY,
                episode_id VARCHAR(255) NOT NULL UNIQUE,
                passed BOOLEAN,
                overall_score DOUBLE PRECISION,
                passed_count INTEGER,
                failed_count INTEGER,
                has_major_issue BOOLEAN,
                has_blocker_issue BOOLEAN,
                category_summary JSONB,
                raw_json JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_test_qc_results_episode
            ON test_qc_results (episode_id);
            """)
            cur.execute("""
            CREATE TABLE IF NOT EXISTS test_qc_categories (
                id BIGSERIAL PRIMARY KEY,
                qc_result_id BIGINT REFERENCES test_qc_results(id) ON DELETE CASCADE,
                category VARCHAR(64),
                passed BOOLEAN,
                score DOUBLE PRECISION,
                passed_count INTEGER,
                failed_count INTEGER,
                details JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            cur.execute("""
            CREATE TABLE IF NOT EXISTS test_qc_issues (
                id BIGSERIAL PRIMARY KEY,
                qc_result_id BIGINT REFERENCES test_qc_results(id) ON DELETE CASCADE,
                category_id BIGINT REFERENCES test_qc_categories(id) ON DELETE CASCADE,
                category VARCHAR(64),
                check_name VARCHAR(256),
                level VARCHAR(16),
                passed BOOLEAN,
                value DOUBLE PRECISION,
                threshold DOUBLE PRECISION,
                message TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """)
            cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_test_qc_issues_result
            ON test_qc_issues (qc_result_id);
            """)
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"Failed to initialize qc result tables: {e}")

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


def save_qc_results(records: list):
    """批量保存统一 QC 结果。"""
    if not records:
        return 0

    conn = get_pnp_connection()
    count = 0
    try:
        with conn.cursor() as cur:
            for rec in records:
                episode_id = str(rec.get("episode_id", ""))
                result = rec.get("result")
                if not episode_id or not isinstance(result, ValidationResult):
                    continue

                raw_json = result.to_dict()
                category_summary = result.get_category_summary()
                has_major_issue = any(
                    issue.level == IssueLevel.MAJOR
                    for issue in result.issues
                )
                has_blocker_issue = any(
                    issue.level == IssueLevel.CRITICAL
                    for issue in result.issues
                )

                cur.execute(
                    """
                    INSERT INTO test_qc_results (
                        episode_id,
                        passed,
                        overall_score,
                        passed_count,
                        failed_count,
                        has_major_issue,
                        has_blocker_issue,
                        category_summary,
                        raw_json,
                        updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (episode_id) DO UPDATE SET
                        passed = EXCLUDED.passed,
                        overall_score = EXCLUDED.overall_score,
                        passed_count = EXCLUDED.passed_count,
                        failed_count = EXCLUDED.failed_count,
                        has_major_issue = EXCLUDED.has_major_issue,
                        has_blocker_issue = EXCLUDED.has_blocker_issue,
                        category_summary = EXCLUDED.category_summary,
                        raw_json = EXCLUDED.raw_json,
                        updated_at = CURRENT_TIMESTAMP
                    RETURNING id
                    """,
                    (
                        episode_id,
                        result.passed,
                        result.score,
                        result.passed_count,
                        result.failed_count,
                        has_major_issue,
                        has_blocker_issue,
                        Json(category_summary),
                        Json(raw_json),
                    ),
                )
                qc_result_id = cur.fetchone()[0]

                cur.execute(
                    "DELETE FROM test_qc_categories WHERE qc_result_id = %s",
                    (qc_result_id,),
                )

                category_name = None
                if result.issues:
                    category_name = result.issues[0].category
                elif result.details:
                    category_name = str(result.details.get("category", ""))

                cur.execute(
                    """
                    INSERT INTO test_qc_categories (
                        qc_result_id,
                        category,
                        passed,
                        score,
                        passed_count,
                        failed_count,
                        details
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        qc_result_id,
                        category_name,
                        result.passed,
                        result.score,
                        result.passed_count,
                        result.failed_count,
                        Json(result.details),
                    ),
                )
                category_id = cur.fetchone()[0]

                for issue in result.issues:
                    cur.execute(
                        """
                        INSERT INTO test_qc_issues (
                            qc_result_id,
                            category_id,
                            category,
                            check_name,
                            level,
                            passed,
                            value,
                            threshold,
                            message
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            qc_result_id,
                            category_id,
                            issue.category,
                            issue.check_name,
                            issue.level.value,
                            issue.passed,
                            issue.value,
                            issue.threshold,
                            issue.message,
                        ),
                    )
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
