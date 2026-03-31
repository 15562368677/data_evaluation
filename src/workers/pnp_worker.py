"""Background worker for PnP detection."""

import time
import json
import logging
import pandas as pd
import numpy as np

from src.acceptance_service import AcceptanceService
from src.utils.source_db import query_df
from src.utils.result_db import (
    get_pnp_connection,
    init_pnp_db,
    init_qc_result_db,
    query_pnp_df,
    save_qc_results,
)
from src.utils.data_parser import HAND_JOINT_NAMES, JOINT_NAMES, load_joint_data
from src.acceptance_service.validators import EEActionValidator, ValidatorConfig


def load_joint_data_as_dfs(episode_id: str, config: ValidatorConfig):
    """
    Find remote files, download and parse into DataFrames using utils/data_parser.
    """
    # Find file path from streams
    sql = "SELECT file_path FROM streams WHERE episode_id = %s AND stream_name = 'rgb' LIMIT 1"
    df = query_df(sql, (episode_id,))
    if df.empty:
        return None, None
    file_path = str(df.iloc[0]["file_path"])

    parsed_data = load_joint_data(file_path)
    if not parsed_data:
        return None, None

    all_joints = list(JOINT_NAMES) + list(HAND_JOINT_NAMES)
    
    # 构建 state_df
    state_df = pd.DataFrame({'timestamp_utc': parsed_data.get('absolute_timestamps_state', [])})
    for joint in all_joints:
        if joint in parsed_data.get('state', {}):
            state_df[joint] = parsed_data['state'][joint]
        else:
            state_df[joint] = np.nan
            
    # 构建 action_df
    action_df = pd.DataFrame({'timestamp_utc': parsed_data.get('absolute_timestamps_action', [])})
    for joint in all_joints:
        if joint in parsed_data.get('action', {}):
            action_df[joint] = parsed_data['action'][joint]
        else:
            action_df[joint] = np.nan

    return state_df, action_df

def _rebuild_batch_qc_results(batch_id: str, validator_config: ValidatorConfig):
    batch_df = query_pnp_df(
        """
        SELECT
            s.episode_id
        FROM pnp_streams s
        WHERE s.batch_id = %s
        ORDER BY s.episode_id
        """,
        (batch_id,),
    )
    if batch_df.empty:
        return []

    episode_payloads = []
    for _, row in batch_df.iterrows():
        episode_id = str(row["episode_id"])
        state_df, action_df = load_joint_data_as_dfs(episode_id, validator_config)
        episode_payloads.append(
            {
                "episode_id": episode_id,
                "state_df": state_df,
                "action_df": action_df,
            }
        )

    acceptance_service = AcceptanceService(config=validator_config)
    return acceptance_service.validate_batch(episodes=episode_payloads)


def run_pnp_task(uniq_id, task_id, sample_ratio, overwrite, params_dict):
    logging.info(f"Starting PNP Task: {uniq_id} for Task {task_id}")
    init_pnp_db()
    init_qc_result_db()

    conn = get_pnp_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO pnp_batches (
                    uniq_id, task_id, sample_ratio, is_overwrite, parameters, status,
                    total_episodes, processed_episodes, failed_episodes, last_heartbeat, error_message
                )
                VALUES (%s, %s, %s, %s, %s, 'queued', 0, 0, 0, CURRENT_TIMESTAMP, NULL)
                ON CONFLICT (uniq_id) DO UPDATE SET
                    task_id = EXCLUDED.task_id,
                    sample_ratio = EXCLUDED.sample_ratio,
                    is_overwrite = EXCLUDED.is_overwrite,
                    parameters = EXCLUDED.parameters
            """, (uniq_id, task_id, sample_ratio, overwrite, json.dumps(params_dict)))
            cur.execute("""
                UPDATE pnp_batches
                SET status = 'running',
                    last_heartbeat = CURRENT_TIMESTAMP,
                    error_message = NULL
                WHERE uniq_id = %s
            """, (uniq_id,))
        conn.commit()
    except Exception as e:
        conn.close()
        raise e

    validator_overrides = {}
    base_validator_config = ValidatorConfig()
    for key, value in (params_dict or {}).items():
        if hasattr(base_validator_config, key):
            validator_overrides[key] = value
    validator_config = ValidatorConfig(**validator_overrides)

    # Retrieve episodes
    episodes_df = query_df(
        """
        SELECT id, trajectory_start
        FROM episodes
        WHERE task_id = %s
          AND trajectory_duration IS NOT NULL
          AND trajectory_duration > 0
        ORDER BY trajectory_start NULLS LAST, id
        """,
        (task_id,),
    )
    total_episodes = [str(e) for e in episodes_df['id'].tolist()]
    
    logging.info(
        f"[PNP] task_id={task_id}: duration_result='invalid' episodes will be kept for PnP validation "
        f"(episode_count={len(total_episodes)})"
    )
    
    # Apply overwrite rule
    if not overwrite:
        before_overwrite_filter = len(total_episodes)
        with conn.cursor() as cur:
            # 继续同一批次时，不应把本批次已完成记录当作“已存在”过滤掉。
            cur.execute(
                "SELECT DISTINCT episode_id FROM pnp_streams WHERE batch_id <> %s",
                (uniq_id,),
            )
            existing = set(row[0] for row in cur.fetchall())
        total_episodes = [ep for ep in total_episodes if ep not in existing]
        logging.info(
            f"[PNP] task_id={task_id}: overwrite=false excluded existing episodes = "
            f"{before_overwrite_filter - len(total_episodes)} (after={len(total_episodes)})"
        )

    # 按 trajectory_start 排序后顺序质检；sample_ratio 仅做“取前 N%”的顺序截断。
    if sample_ratio == 0:
        sample_count = 1
    else:
        sample_count = max(1, int(len(total_episodes) * sample_ratio / 100.0))
    
    if sample_count < len(total_episodes):
        sampled_episodes = total_episodes[:sample_count]
    else:
        sampled_episodes = total_episodes

    with conn.cursor() as cur:
        cur.execute(
            "SELECT episode_id FROM pnp_streams WHERE batch_id = %s",
            (uniq_id,),
        )
        completed_in_batch = set(row[0] for row in cur.fetchall())

    sampled_set = set(sampled_episodes)
    already_done = len(sampled_set & completed_in_batch)
    remaining_episodes = [ep for ep in sampled_episodes if ep not in completed_in_batch]

    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE pnp_batches
            SET total_episodes = %s,
                processed_episodes = %s,
                failed_episodes = 0,
                last_heartbeat = CURRENT_TIMESTAMP
            WHERE uniq_id = %s
            """,
            (len(sampled_episodes), already_done, uniq_id),
        )
    conn.commit()

    if not sampled_episodes:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE pnp_batches
                SET status = 'success',
                    last_heartbeat = CURRENT_TIMESTAMP,
                    error_message = NULL
                WHERE uniq_id = %s
                """,
                (uniq_id,),
            )
        conn.commit()
        conn.close()
        logging.info("No episodes left to process for this batch.")
        return

    logging.info(
        f"Will process {len(remaining_episodes)} episodes "
        f"(already_done={already_done}, total_in_batch={len(sampled_episodes)})"
    )

    failed_episodes = 0
    last_error_message = None
    processed_episodes = already_done
    stream_validator = EEActionValidator(config=validator_config)

    def _record_failure(cur, ep_id: str, err_msg: str):
        cur.execute(
            """
            INSERT INTO pnp_failures (episode_id, batch_id, error_message, failed_at)
            VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (episode_id, batch_id)
            DO UPDATE SET error_message = EXCLUDED.error_message, failed_at = CURRENT_TIMESTAMP
            """,
            (ep_id, uniq_id, err_msg),
        )

    stop_status = None
    for episode_id in remaining_episodes:
        # 支持页面控制：paused / stopping / stopped
        with conn.cursor() as cur:
            cur.execute("SELECT status FROM pnp_batches WHERE uniq_id = %s", (uniq_id,))
            status_row = cur.fetchone()
            current_status = str(status_row[0]) if status_row and status_row[0] is not None else "running"
        if current_status == "paused":
            stop_status = "paused"
            break
        if current_status in {"stopping", "stopped"}:
            stop_status = "stopped"
            break

        try:
            detection_result = stream_validator.validate(episode_id)
            detection_details = detection_result.details or {}
            hands = detection_details.get("hands") or {}
            right_segments = (hands.get("right") or {}).get("segments") or []
            left_segments = (hands.get("left") or {}).get("segments") or []
            right_json = json.dumps(right_segments)
            left_json = json.dumps(left_segments)

            # Insert into database
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO pnp_streams (episode_id, batch_id, pnp_result, right_pnp_result, left_pnp_result)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT(episode_id, batch_id) 
                    DO UPDATE SET pnp_result = %s, right_pnp_result = %s, left_pnp_result = %s, checked_at = CURRENT_TIMESTAMP
                """, (episode_id, uniq_id, None, right_json, left_json, None, right_json, left_json))
                cur.execute(
                    "DELETE FROM pnp_failures WHERE episode_id = %s AND batch_id = %s",
                    (episode_id, uniq_id),
                )
                cur.execute(
                    """
                    UPDATE pnp_batches
                    SET processed_episodes = COALESCE(processed_episodes, 0) + 1,
                        last_heartbeat = CURRENT_TIMESTAMP
                    WHERE uniq_id = %s
                    """,
                    (uniq_id,),
                )
            conn.commit()
            processed_episodes += 1

            logging.info(
                f"Processed episode {episode_id}, found R:{len(right_segments)} "
                f"L:{len(left_segments)} pick-place operations."
            )

        except Exception as e:
            logging.error(f"Error processing episode {episode_id}: {e}")
            conn.rollback()
            failed_episodes += 1
            last_error_message = str(e)
            with conn.cursor() as cur:
                _record_failure(cur, episode_id, last_error_message)
                cur.execute(
                    """
                    UPDATE pnp_batches
                    SET failed_episodes = COALESCE(failed_episodes, 0) + 1,
                        last_heartbeat = CURRENT_TIMESTAMP,
                        error_message = %s
                    WHERE uniq_id = %s
                    """,
                    (last_error_message, uniq_id),
                )
            conn.commit()
            continue

    if stop_status in {"paused", "stopped"}:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE pnp_batches
                SET status = %s,
                    last_heartbeat = CURRENT_TIMESTAMP
                WHERE uniq_id = %s
                """,
                (stop_status, uniq_id),
            )
        conn.commit()
        conn.close()
        logging.info(f"Stopped PNP task {uniq_id} with status={stop_status}.")
        return

    final_status = "success"
    if failed_episodes > 0 and processed_episodes > 0:
        final_status = "partial"
    elif failed_episodes > 0 and processed_episodes == 0:
        final_status = "failed"

    if final_status in {"success", "partial"} and processed_episodes > 0:
        try:
            batch_results = _rebuild_batch_qc_results(str(uniq_id), validator_config)
            saved_qc_count = save_qc_results(
                [{"episode_id": item["episode_id"], "result": item["result"]} for item in batch_results]
            )
            with conn.cursor() as cur:
                for item in batch_results:
                    cur.execute(
                        """
                        UPDATE pnp_streams
                        SET pnp_result = %s,
                            checked_at = CURRENT_TIMESTAMP
                        WHERE episode_id = %s
                          AND batch_id = %s
                        """,
                        (
                            json.dumps(item["stream_summary"], ensure_ascii=False),
                            item["episode_id"],
                            uniq_id,
                        ),
                    )
            conn.commit()
            logging.info(
                f"Rebuilt EEActionValidator QC results for batch {uniq_id}, task {task_id}: "
                f"{saved_qc_count} episodes saved."
            )
        except Exception as exc:
            conn.rollback()
            logging.error(f"Failed to rebuild QC results for batch {uniq_id}, task {task_id}: {exc}")
            last_error_message = f"QC rebuild failed: {exc}"
            if processed_episodes > 0:
                final_status = "partial"
            else:
                final_status = "failed"

    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE pnp_batches
            SET status = %s,
                last_heartbeat = CURRENT_TIMESTAMP,
                error_message = %s
            WHERE uniq_id = %s
            """,
            (final_status, last_error_message if failed_episodes > 0 else None, uniq_id),
        )
    conn.commit()
    conn.close()
    logging.info(f"Finished PNP task {uniq_id} with status={final_status}.")
