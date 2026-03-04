from typing import Dict, List, Union, Tuple
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Hand configurations
HAND_CONFIG_BASE = {
    'right': {
        'right_hand_fingers': [
            'R_pinky_proximal_joint', 'R_ring_proximal_joint',
            'R_middle_proximal_joint', 'R_index_proximal_joint',
            'R_thumb_proximal_pitch_joint'
        ],
        'additional_joints': ['right_elbow_pitch_joint'],
        'joint_direction_coefficients': {
            'R_pinky_proximal_joint': -1.0,
            'R_ring_proximal_joint': -1.0,
            'R_middle_proximal_joint': -1.0,
            'R_index_proximal_joint': -1.0,
            'R_thumb_proximal_pitch_joint': 1.0,
        },
    },
    'left': {
        'right_hand_fingers': [
            'L_pinky_proximal_joint', 'L_ring_proximal_joint',
            'L_middle_proximal_joint', 'L_index_proximal_joint',
            'L_thumb_proximal_pitch_joint'
        ],
        'additional_joints': ['left_elbow_pitch_joint'],
        'joint_direction_coefficients': {
            'L_pinky_proximal_joint': -1.0,
            'L_ring_proximal_joint': -1.0,
            'L_middle_proximal_joint': -1.0,
            'L_index_proximal_joint': -1.0,
            'L_thumb_proximal_pitch_joint': 1.0,
        },
    }
}

def calculate_closure_degree(
    joint_angles: Dict[str, float],
    finger_joints: List[str],
    direction_coefficients: Dict[str, float] = None
) -> float:
    if direction_coefficients is None:
        raise ValueError("direction_coefficients must be provided")

    valid_joints = [
        (name, angle)
        for name, angle in joint_angles.items()
        if name in finger_joints and not np.isnan(angle)
    ]
    if not valid_joints:
        return np.nan

    weighted_sum = 0.0
    total_weight = 0.0
    
    for name, angle in valid_joints:
        coeff = direction_coefficients.get(name, 0)
        # 确定权重，拇指 40%，其余四指各 15%
        if "thumb" in name.lower():
            weight = 0.40
        else:
            weight = 0.15
            
        weighted_sum += (coeff * angle * weight)
        total_weight += weight
        
    # 如果有效关节总权重不足 1.0 (例如有NaN)，进行归一化
    if total_weight > 0:
        return weighted_sum / total_weight
    return 0.0


def calculate_closure_velocity(
    closure_degrees: Union[List[float], np.ndarray],
    timestamps: Union[List[float], np.ndarray] = None
) -> np.ndarray:
    closure_degrees = np.array(closure_degrees)
    if len(closure_degrees) < 2:
        return np.array([])

    velocity = np.diff(closure_degrees)

    if timestamps is not None:
        timestamps = np.array(timestamps)
        if len(timestamps) != len(closure_degrees):
            raise ValueError(
                f"Length mismatch: closure_degrees ({len(closure_degrees)}) "
                f"vs timestamps ({len(timestamps)})"
            )
        time_diff = np.diff(timestamps)
        # 判断规则只关注闭合速度的大值，60帧/秒下，时间间隔不可能小于1/60=0.01667s，除1只会避免除0,不影响判断
        time_diff = np.where(time_diff == 0, 1.0, time_diff)
        velocity = velocity / time_diff

    return velocity


def calculate_closure_metrics_from_dataframe(
    df: pd.DataFrame,
    finger_joints: List[str],
    direction_coefficients: Dict[str, float] = None
) -> pd.DataFrame:
    
    if direction_coefficients is None:
        raise ValueError("direction_coefficients must be provided")

    closure_degrees = []
    for _, row in df.iterrows():
        joint_angles = {joint: row[joint] for joint in finger_joints if joint in df.columns}
        closure_degrees.append(calculate_closure_degree(joint_angles, finger_joints, direction_coefficients))

    result_df = pd.DataFrame({
        "timestamp_utc": df["timestamp_utc"],
        "closure_degree": closure_degrees,
    })

    # 检测过采样均匀，所以这里没有考虑归一化处理
    velocity = calculate_closure_velocity(closure_degrees)

    result_df["closure_velocity"] = [np.nan] + list(velocity)

    logger.info(f"Calculated closure metrics for {len(result_df)} records")
    logger.info(f"Closure degree range: [{result_df['closure_degree'].min():.4f}, {result_df['closure_degree'].max():.4f}]")
    logger.info(f"Closure velocity range: [{result_df['closure_velocity'].min():.4f}, {result_df['closure_velocity'].max():.4f}]")
    return result_df



def check_joint_diff_with_slope(
    frame_idx: int,
    joint_differences: dict,
    state_df: pd.DataFrame,
    action_df: pd.DataFrame,
    config: dict,
    finger_joints: List[str] = None,
    direction_coefficients: Dict[str, float] = None
) -> Tuple[bool, int]:
    """
    检查指定帧是否有足够数量的关节的差值满足条件，并验证斜率稳定性
    
    注意：由于 state 和 action 的采样频率不同，需要使用时间戳对齐来获取对应的 action 值
    
    新的差值条件：
    - 方向系数为 -1 的关节（四个手指）：差值 < -0.2
    - 方向系数为 +1 的关节（拇指pitch）：差值 > 0.1
    - 同时需要 state 和 action 的斜率绝对值小于阈值（筛选掉瞬时差）
    
    Args:
        frame_idx: 要检查的帧索引（基于 state 帧）
        joint_differences: 关节差值字典 {"joint_name": np.ndarray}（已按 state 帧对齐）
        state_df: state 数据帧，包含关节角度和时间戳
        action_df: action 数据帧，包含关节角度和时间戳
        config: 配置字典
        finger_joints: 手指关节名称列表
        direction_coefficients: 方向系数字典，若为 None 使用默认值
        min_joints: 最少需要多少个关节满足条件（默认2个）
        slope_threshold: 斜率阈值，state和action斜率绝对值都需要小于此值
        slope_lookahead: 计算斜率使用的跨度（帧数）
        
    Returns:
        Tuple[bool, int]: (是否满足条件, 满足条件的关节数量)
    """
    if direction_coefficients is None:
        raise ValueError("direction_coefficients must be provided")
    
    if finger_joints is None:
        finger_joints = config['right_hand_fingers']
    
    joints_satisfied = 0
    
    # 使用配置的阈值
    NEGATIVE_DIFF_THRESHOLD = config['negative_diff_threshold']
    POSITIVE_DIFF_THRESHOLD = config['positive_diff_threshold']
    min_joints = config['min_joints_for_diff']
    slope_threshold = config['slope_threshold']
    slope_lookahead = config['slope_lookahead']
    
    # 预先转换时间戳用于对齐
    state_timestamps = pd.to_datetime(state_df['timestamp_utc'])
    action_timestamps = pd.to_datetime(action_df['timestamp_utc'])
    
    for joint in finger_joints:
        if joint not in joint_differences:
            continue
            
        diffs = joint_differences[joint]
        coeff = direction_coefficients.get(joint, 0)
        
        if frame_idx >= len(diffs) or np.isnan(diffs[frame_idx]):
            continue
            
        diff_val = diffs[frame_idx]
        
        # 检查差值条件
        diff_condition_met = False
        if coeff < 0:  # 四个手指
            if diff_val < NEGATIVE_DIFF_THRESHOLD:
                diff_condition_met = True
        elif coeff > 0:  # 拇指
            if diff_val > POSITIVE_DIFF_THRESHOLD:
                diff_condition_met = True
        
        if not diff_condition_met:
            continue
        
        # 检查斜率稳定性
        # 计算 state 和 action 在当前帧和未来第n帧的斜率 (Secant method)
        slope_stable = True
        
        if joint in state_df.columns and joint in action_df.columns:
            # 计算 state 斜率: |val[i+n] - val[i]| / n
            start_idx = frame_idx
            end_idx = min(len(state_df) - 1, frame_idx + slope_lookahead)
            
            if end_idx > start_idx:
                diff_step = end_idx - start_idx
                # 防止除以0
                if diff_step > 0:
                    # State slope
                    state_end_val = state_df[joint].iloc[end_idx]
                    state_start_val = state_df[joint].iloc[start_idx]
                    
                    if not np.isnan(state_end_val) and not np.isnan(state_start_val):
                        state_slope = np.abs(state_end_val - state_start_val) / diff_step
                        if state_slope > slope_threshold:
                            slope_stable = False
                    
                    # Action slope (timestamp aligned)
                    if slope_stable:
                        state_end_ts = state_timestamps.iloc[end_idx]
                        state_start_ts = state_timestamps.iloc[start_idx]
                        
                        # Find closest action frames
                        action_end_idx = np.argmin(np.abs(action_timestamps - state_end_ts))
                        action_start_idx = np.argmin(np.abs(action_timestamps - state_start_ts))
                        
                        action_end_val = action_df[joint].iloc[action_end_idx]
                        action_start_val = action_df[joint].iloc[action_start_idx]
                        
                        if not np.isnan(action_end_val) and not np.isnan(action_start_val):
                            action_slope = np.abs(action_end_val - action_start_val) / diff_step
                            if action_slope > slope_threshold:
                                slope_stable = False
        
        if diff_condition_met and slope_stable:
            joints_satisfied += 1
    
    return joints_satisfied >= min_joints, joints_satisfied


def check_sufficient_joint_differences(
    frame_idx: int,
    joint_differences: dict,
    config: dict,
    direction_coefficients: Dict[str, float] = None
) -> bool:
    """
    检查指定帧是否有足够数量的关节的有向差值满足条件（不检查斜率，用于向后兼容）
    
    有向差值条件（根据关节方向系数）：
    - 对于方向系数为 +1 的关节：差值（action - state）应为 > 阈值 的正数
    - 对于方向系数为 -1 的关节：差值（action - state）应为 < -阈值 的负数
    
    Args:
        frame_idx: 要检查的帧索引
        joint_differences: 关节差值字典 {"joint_name": np.ndarray}
        state_action_diff_thresholds: 每个关节的阈值字典 {"joint_name": float}（阈值为正数）
        direction_coefficients: 方向系数字典，若为 None 使用默认值
        min_joints: 最少需要多少个关节满足条件（默认2个）
        
    Returns:
        True 如果有 >= min_joints 个关节的有向差值满足各自的条件，否则 False
    """
    if direction_coefficients is None:
        raise ValueError("direction_coefficients must be provided from config")
    
    # 使用配置的阈值
    NEGATIVE_DIFF_THRESHOLD = config['negative_diff_threshold']
    POSITIVE_DIFF_THRESHOLD = config['positive_diff_threshold']
    min_joints = config['min_joints_for_diff']
    finger_joints = config['right_hand_fingers']
    
    joints_satisfied = 0
    
    for joint in finger_joints:
        if joint in joint_differences:
            diffs = joint_differences[joint]
            coeff = direction_coefficients.get(joint, 0)
            
            if frame_idx < len(diffs) and not np.isnan(diffs[frame_idx]):
                diff_val = diffs[frame_idx]
                
                # 根据方向系数和固定阈值判定
                if coeff > 0:
                    # 正方向关节（拇指）：差值应 > 0.1
                    if diff_val > POSITIVE_DIFF_THRESHOLD:
                        joints_satisfied += 1
                elif coeff < 0:
                    # 负方向关节（四个手指）：差值应 < -0.2
                    if diff_val < NEGATIVE_DIFF_THRESHOLD:
                        joints_satisfied += 1
    
    return joints_satisfied >= min_joints


def count_joints_satisfying_diff(
    frame_idx: int,
    joint_differences: dict,
    config: dict,
    direction_coefficients: Dict[str, float] = None
) -> int:
    """
    统计满足差值条件的关节数量（不检查斜率）
    
    Args:
        frame_idx: 要检查的帧索引
        joint_differences: 关节差值字典
        direction_coefficients: 方向系数字典
        
    Returns:
        满足条件的关节数量
    """
    if direction_coefficients is None:
        raise ValueError("direction_coefficients must be provided from config")
    
    # 使用配置的阈值
    NEGATIVE_DIFF_THRESHOLD = config['negative_diff_threshold']
    POSITIVE_DIFF_THRESHOLD = config['positive_diff_threshold']
    finger_joints = config['right_hand_fingers']
    
    joints_satisfied = 0
    
    for joint in finger_joints:
        if joint in joint_differences:
            diffs = joint_differences[joint]
            coeff = direction_coefficients.get(joint, 0)
            
            if frame_idx < len(diffs) and not np.isnan(diffs[frame_idx]):
                diff_val = diffs[frame_idx]
                
                if coeff > 0 and diff_val > POSITIVE_DIFF_THRESHOLD:
                    joints_satisfied += 1
                elif coeff < 0 and diff_val < NEGATIVE_DIFF_THRESHOLD:
                    joints_satisfied += 1
    
    return joints_satisfied


def pick_identify(
    closure_degrees: np.ndarray,
    closure_velocities: np.ndarray,
    elbow_angles: np.ndarray,
    state_action_diffs: dict,
    config: dict,
    state_df: pd.DataFrame = None,
    action_df: pd.DataFrame = None
) -> List[Tuple[int, int]]:
    """
    Pick-Place 识别 - 新逻辑

    Pick 判定条件（同时满足）：
    1) i 帧手指闭合度 > pick_closure_threshold (默认 0.35)
    2) i 帧至少有 min_joints_for_diff 个关节满足差值条件：
       - 四个手指（系数=-1）：state_action_diff < negative_diff_threshold (默认 -0.022)
       - 拇指（系数=+1）：state_action_diff > positive_diff_threshold (默认 0.03)
       - 需要验证 state/action 斜率稳定性：|val[i+lookahead] - val[i]| / lookahead ≤ slope_threshold
    -> pick_start = i + pick_start_offset (默认 -10)

    Place 判定条件（同时满足）：
    1) 帧范围 [i, i + place_diff_lookahead) 内所有帧的满足差值条件的关节数都 < min_joints_for_diff
    2) 以下条件之一成立（OR）：
       2a) i 帧闭合度 < place_closure_threshold (默认 0.35)
       2b) 帧范围 [i - place_velocity_lookback, i + place_velocity_lookahead) 内存在速度 < place_velocity_threshold (默认 -0.02)
    -> place_end = i + place_end_offset (默认 5)

    返回：(pick_start, place_end) 对
    """
    picks: List[Tuple[int, int]] = []
    n_frames = len(closure_degrees)
    
    # 从配置获取阈值
    PICK_CLOSURE_THRESHOLD = config['pick_closure_threshold']
    PLACE_CLOSURE_THRESHOLD = config['place_closure_threshold']
    PLACE_VELOCITY_THRESHOLD = config['place_velocity_threshold']
    PICK_START_OFFSET = config['pick_start_offset']
    PLACE_VELOCITY_LOOKBACK = config['place_velocity_lookback']
    PLACE_VELOCITY_LOOKAHEAD = config['place_velocity_lookahead']
    PLACE_LOOKAHEAD = config['place_diff_lookahead']
    MIN_JOINTS = config['min_joints_for_diff']
    SLOPE_THRESHOLD = config['slope_threshold']
    SLOPE_LOOKAHEAD = config['slope_lookahead']

    in_pick = False
    pick_start: int | None = None

    for i in range(n_frames):
        # 跳过 NaN
        if np.isnan(closure_degrees[i]):
            continue

        # ========== Pick 开始检测 ==========
        if not in_pick:
            # 条件1：i 帧手指闭合度大于阈值
            closure_condition = closure_degrees[i] > PICK_CLOSURE_THRESHOLD
            
            if not closure_condition:
                continue
            
            # 条件2：i 帧 n 个以上手指差值满足条件（带斜率稳定性检查）
            if state_df is not None and action_df is not None:
                diff_condition, joints_count = check_joint_diff_with_slope(
                    i, state_action_diffs, state_df, action_df, config,
                    direction_coefficients=config['joint_direction_coefficients']
                )
            else:
                # 如果没有提供 state_df 和 action_df，使用简单的差值检查
                diff_condition = check_sufficient_joint_differences(
                    i, state_action_diffs, config,
                    direction_coefficients=config['joint_direction_coefficients']
                )
                joints_count = count_joints_satisfying_diff(i, state_action_diffs, config)
            
            if not diff_condition:
                continue
            
            # 所有条件满足，记录 Pick 开始
            pick_start = max(0, i + PICK_START_OFFSET)
            in_pick = True
            logger.info(f"🔍 Pick 开始检测: 帧 {i} -> 开始帧 {pick_start}")
            logger.debug(f"   条件1: 闭合度 {closure_degrees[i]:.4f} > {PICK_CLOSURE_THRESHOLD}")
            logger.debug(f"   条件2: {joints_count} 个关节满足差值条件（>={MIN_JOINTS}）")
        
        # ========== Place 结束检测（需要先进入 Pick 状态）==========
        if in_pick:
            # 条件1：闭合度小于阈值 OR (i-lookback, i+lookahead)帧内存在闭合速度小于阈值的帧
            open_condition_closure = closure_degrees[i] < PLACE_CLOSURE_THRESHOLD
            
            # 检查闭合速度条件：(i-lookback, i+lookahead)帧内存在速度小于阈值的帧
            open_condition_velocity = False
            velocity_start_idx = max(0, i - PLACE_VELOCITY_LOOKBACK)
            velocity_end_idx = min(len(closure_velocities), i + PLACE_VELOCITY_LOOKAHEAD)
            for j in range(velocity_start_idx, velocity_end_idx):
                if j < len(closure_velocities) and not np.isnan(closure_velocities[j]):
                    if closure_velocities[j] < PLACE_VELOCITY_THRESHOLD:
                        open_condition_velocity = True
                        break
            
            # 条件1：闭合度 OR 闭合速度满足其中一个即可
            open_condition = open_condition_closure or open_condition_velocity
            
            if not open_condition:
                continue
            
            # 条件2：未来 n 帧内没有关节或只有少于 MIN_JOINTS 个关节满足差值条件
            diff_condition_place = True
            window_end = min(n_frames, i + PLACE_LOOKAHEAD)
            for j in range(i, window_end):
                if state_df is not None and action_df is not None:
                    # 使用带斜率检查的差值计算 (过滤掉不稳定/噪声导致的 Active 状态)
                    _, joints_count = check_joint_diff_with_slope(
                        j, state_action_diffs, state_df, action_df, config,
                        direction_coefficients=config['joint_direction_coefficients']
                    )
                else:
                    joints_count = count_joints_satisfying_diff(j, state_action_diffs, config, 
                                                                 direction_coefficients=config['joint_direction_coefficients'])
                
                if joints_count >= MIN_JOINTS:
                    diff_condition_place = False
                    break
            
            if not diff_condition_place:
                continue
            

            
            # 所有条件满足，记录 Place 结束
            place_end = i + config['place_end_offset']
            picks.append((pick_start, place_end))
            logger.info(f"✅ Pick-Place 完成: 帧 {pick_start} -> {place_end} (时长: {place_end - pick_start + 1} 帧)")
            logger.debug(f"   条件1a: 闭合度 {closure_degrees[i]:.4f} < {PLACE_CLOSURE_THRESHOLD} = {open_condition_closure}")
            logger.debug(f"   条件1b: 帧 ({velocity_start_idx}, {velocity_end_idx}) 内存在速度 < {PLACE_VELOCITY_THRESHOLD} = {open_condition_velocity}")
            logger.debug(f"   条件2: 帧 ({i}, {window_end}) 内满足差值条件的关节数 < {MIN_JOINTS}")

            in_pick = False
            pick_start = None

    return picks



