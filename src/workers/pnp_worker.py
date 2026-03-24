"""Background worker for PnP detection."""

import time
import json
import logging
import psycopg2
import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.source_db import query_df
from src.utils.result_db import get_pnp_connection, init_pnp_db
from src.utils.data_parser import load_joint_data

from src.engines.pnp_detector.data_detector import (
    HAND_CONFIG_BASE,
    calculate_closure_metrics_from_dataframe,
    pick_identify
)


def load_joint_data_as_dfs(episode_id: str, config: dict):
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

    all_joints = config['right_hand_fingers']
    
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


def run_pnp_task(uniq_id, task_id, sample_ratio, overwrite, params_dict):
    logging.info(f"Starting PNP Task: {uniq_id} for Task {task_id}")
    init_pnp_db()

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

    # Configs for processing
    config_right = {
        **HAND_CONFIG_BASE['right'],
        **params_dict
    }

    config_left = {
        **HAND_CONFIG_BASE['left'],
        **params_dict
    }

    # Config combined for loading data frames for both hands.
    config_load = {
        'right_hand_fingers': config_right['right_hand_fingers'] + config_left['right_hand_fingers'],
    }

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

    def process_hand(st_df, ac_df, hand_config):
        closure_df = calculate_closure_metrics_from_dataframe(
            st_df, 
            hand_config['right_hand_fingers'], 
            hand_config['joint_direction_coefficients']
        )
        
        st = st_df.copy()
        ac = ac_df.copy()
        st = st.sort_values('timestamp_utc')
        ac = ac.sort_values('timestamp_utc')
        ac_cols = ['timestamp_utc'] + [col for col in hand_config['right_hand_fingers'] if col in ac.columns]
        action_subset = ac[ac_cols]
        merged = pd.merge_asof(st, action_subset, on='timestamp_utc', direction='nearest', suffixes=('', '_action'))
        
        diffs = {}
        for joint in hand_config['right_hand_fingers']:
            a_col = f"{joint}_action"
            if joint in merged.columns and a_col in merged.columns:
                diffs[joint] = (merged[a_col] - merged[joint]).to_numpy()
            else:
                diffs[joint] = np.full(len(st), np.nan)
        
        closure_degrees = closure_df['closure_degree'].to_numpy()
        closure_velocities = closure_df['closure_velocity'].to_numpy()
        picks = pick_identify(
            closure_degrees=closure_degrees,
            closure_velocities=closure_velocities,
            state_action_diffs=diffs,
            config=hand_config,
            state_df=st_df,
            action_df=ac_df
        )
        time_picks = []
        for p in picks:
            try:
                if len(st_df) > 0 and 'timestamp_utc' in st_df.columns:
                    t0 = st_df['timestamp_utc'].iloc[0]
                    s_idx = max(0, min(p[0], len(st_df)-1))
                    e_idx = max(0, min(p[1], len(st_df)-1))
                    st_sec = (st_df['timestamp_utc'].iloc[s_idx] - t0).total_seconds()
                    ed_sec = (st_df['timestamp_utc'].iloc[e_idx] - t0).total_seconds()
                    time_picks.append([float(st_sec), float(ed_sec)])
                else:
                    time_picks.append([float(p[0])/60.0, float(p[1])/60.0])
            except Exception as e:
                logging.error(f"Error converting frames {p} to time: {e}")
                time_picks.append([float(p[0])/60.0, float(p[1])/60.0])
        return json.dumps(time_picks)

    failed_episodes = 0
    last_error_message = None
    processed_episodes = already_done

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
            state_df, action_df = load_joint_data_as_dfs(episode_id, config_load)
            if state_df is None or action_df is None or len(state_df) == 0:
                failed_episodes += 1
                last_error_message = f"Episode {episode_id} has no valid joint data."
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
                
            if 'timestamp_utc' in state_df.columns and not pd.api.types.is_datetime64_any_dtype(state_df['timestamp_utc']):
                state_df['timestamp_utc'] = pd.to_datetime(state_df['timestamp_utc'], unit='s')
            if 'timestamp_utc' in action_df.columns and not pd.api.types.is_datetime64_any_dtype(action_df['timestamp_utc']):
                action_df['timestamp_utc'] = pd.to_datetime(action_df['timestamp_utc'], unit='s')

            # Process both hands
            right_json = process_hand(state_df, action_df, config_right)
            left_json = process_hand(state_df, action_df, config_left)

            # Insert into database
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO pnp_streams (episode_id, batch_id, right_pnp_result, left_pnp_result)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT(episode_id, batch_id) 
                    DO UPDATE SET right_pnp_result = %s, left_pnp_result = %s, checked_at = CURRENT_TIMESTAMP
                """, (episode_id, uniq_id, right_json, left_json, right_json, left_json))
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

            logging.info(f"Processed episode {episode_id}, found R:{len(json.loads(right_json))} L:{len(json.loads(left_json))} pick-place operations.")

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
