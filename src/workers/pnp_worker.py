"""Background worker for PnP detection."""

import time
import json
import logging
import random
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

    all_joints = config['right_hand_fingers'] + config['additional_joints']
    
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
                INSERT INTO pnp_batches (uniq_id, task_id, sample_ratio, is_overwrite, parameters)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (uniq_id) DO NOTHING
            """, (uniq_id, task_id, sample_ratio, overwrite, json.dumps(params_dict)))
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

    # Config combined for loading data frames (which checks right_hand_fingers + additional_joints)
    config_load = {
        'right_hand_fingers': config_right['right_hand_fingers'] + config_left['right_hand_fingers'],
        'additional_joints': config_right['additional_joints'] + config_left['additional_joints']
    }

    # Retrieve episodes
    episodes_df = query_df("SELECT id FROM episodes WHERE task_id = %s", (task_id,))
    total_episodes = [str(e) for e in episodes_df['id'].tolist()]
    
    # Exclude episodes marked as invalid in duration_results
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT episode_id FROM duration_results WHERE task_id = %s AND duration_result = 'invalid'", (task_id,))
            invalid_duration_episodes = set(row[0] for row in cur.fetchall())
        if invalid_duration_episodes:
            logging.info(f"Excluding {len(invalid_duration_episodes)} episodes marked as invalid in duration check.")
            total_episodes = [ep for ep in total_episodes if ep not in invalid_duration_episodes]
    except Exception as e:
        logging.warning(f"Failed to filter invalid duration episodes: {e}")
    
    # Apply overwrite rule
    if not overwrite:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT episode_id FROM pnp_streams")
            existing = set(row[0] for row in cur.fetchall())
        total_episodes = [ep for ep in total_episodes if ep not in existing]

    if not total_episodes:
        logging.info("No episodes left to process.")
        conn.close()
        return

    # Sampling
    if sample_ratio == 0:
        sample_count = 1
    else:
        sample_count = max(1, int(len(total_episodes) * sample_ratio / 100.0))
    
    if sample_count < len(total_episodes):
        sampled_episodes = random.sample(total_episodes, sample_count)
    else:
        sampled_episodes = total_episodes

    logging.info(f"Will process {len(sampled_episodes)} episodes out of {len(total_episodes)}")

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
        elbow_angles = np.zeros(len(closure_degrees))

        picks = pick_identify(
            closure_degrees=closure_degrees,
            closure_velocities=closure_velocities,
            elbow_angles=elbow_angles,
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

    for episode_id in sampled_episodes:
        try:
            state_df, action_df = load_joint_data_as_dfs(episode_id, config_load)
            if state_df is None or action_df is None or len(state_df) == 0:
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
            conn.commit()

            logging.info(f"Processed episode {episode_id}, found R:{len(json.loads(right_json))} L:{len(json.loads(left_json))} pick-place operations.")

        except Exception as e:
            logging.error(f"Error processing episode {episode_id}: {e}")
            conn.rollback()
            continue

    conn.close()
    logging.info(f"Finished PNP task {uniq_id}.")

