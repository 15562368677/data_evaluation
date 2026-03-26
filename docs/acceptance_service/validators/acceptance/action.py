"""
动作数据验证器

验证采样频率、静止检测、关节安全性
支持新格式 (Parquet) 和旧格式 (HDF5) 数据
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Tuple

from ..core.base import (
    BaseValidator,
    ValidationResult,
    ValidationIssue,
    ValidatorConfig,
    IssueLevel,
)


# 关键关节列表（用于静止检测）
KEY_JOINTS = [
    'left_shoulder_pitch_joint',
    'left_elbow_pitch_joint',
    'left_wrist_pitch_joint',
    'right_shoulder_pitch_joint',
    'right_elbow_pitch_joint',
    'right_wrist_pitch_joint',
]

# HDF5 旧格式的关节映射（索引 -> 名称）
HDF5_ROBOT_JOINTS = [
    f'robot_joint_{i}' for i in range(32)
]
HDF5_HAND_JOINTS = [
    f'hand_joint_{i}' for i in range(12)
]


class ActionValidator(BaseValidator):
    """动作数据验证器"""
    
    @property
    def name(self) -> str:
        return "动作数据验证"
    
    @property
    def category(self) -> str:
        return "动作数据"
    
    def _load_joint_data(self, path: Path) -> Tuple[Optional[pd.DataFrame], List[str], str, Optional[np.ndarray]]:
        """
        加载关节数据，支持 Parquet 和 HDF5 格式
        
        Returns:
            (joint_df, joint_columns, format_type, timestamps): 关节数据、列名、格式类型、时间戳
        """
        state_file = path / 'observation.state.parquet'
        hdf5_file = path / 'data.hdf5'
        
        # 优先尝试 Parquet 格式
        if state_file.exists():
            try:
                df = pd.read_parquet(state_file)
                joint_df, joint_columns = self._extract_joint_data(df)
                timestamps = df['timestamp_utc'].values if 'timestamp_utc' in df.columns else None
                return joint_df, joint_columns, 'parquet', timestamps
            except Exception:
                pass
        
        # 尝试 HDF5 格式
        if hdf5_file.exists():
            try:
                import h5py
                with h5py.File(hdf5_file, 'r') as f:
                    records = []
                    timestamps = f['timestamp'][:] if 'timestamp' in f else None
                    n_frames = len(timestamps) if timestamps is not None else 0
                    
                    # 读取机器人状态
                    state_robot = f['state']['robot'][:] if 'state' in f and 'robot' in f['state'] else None
                    state_hand = f['state']['hand'][:] if 'state' in f and 'hand' in f['state'] else None
                    
                    for i in range(n_frames):
                        record = {}
                        if state_robot is not None:
                            for j, val in enumerate(state_robot[i]):
                                record[f'robot_joint_{j}'] = float(val)
                        if state_hand is not None:
                            for j, val in enumerate(state_hand[i]):
                                record[f'hand_joint_{j}'] = float(val)
                        records.append(record)
                    
                    joint_df = pd.DataFrame(records)
                    joint_columns = list(joint_df.columns)
                    return joint_df, joint_columns, 'hdf5', timestamps
            except Exception as e:
                print(f"[ActionValidator] HDF5 加载失败: {e}")
                pass
        
        return None, [], 'unknown', None
    
    def validate(self, episode_id: str, data_path: str) -> ValidationResult:
        """验证动作数据质量"""
        issues: List[ValidationIssue] = []
        path = Path(data_path)
        
        # 加载关节数据（支持 Parquet 和 HDF5）
        joint_df, joint_columns, format_type, timestamps = self._load_joint_data(path)
        
        if format_type == 'unknown':
            issues.append(self._create_issue(
                check_name="状态数据文件",
                message="缺少 observation.state.parquet 或 data.hdf5",
                passed=False,
                level=IssueLevel.CRITICAL,
            ))
            return ValidationResult(passed=False, score=0, issues=issues)
        
        if joint_df is None or not joint_columns:
            issues.append(self._create_issue(
                check_name="关节数据列",
                message=f"未找到或无法解析关节数据 (格式: {format_type})",
                passed=False,
                level=IssueLevel.CRITICAL,
            ))
            return ValidationResult(passed=False, score=0, issues=issues)
        
        issues.append(self._create_issue(
            check_name="状态数据读取",
            message=f"成功加载 {len(joint_df)} 帧 (格式: {format_type})",
            passed=True,
        ))
        
        issues.append(self._create_issue(
            check_name="关节数据列",
            message=f"找到 {len(joint_columns)} 个关节列",
            passed=True,
        ))
        
        # 动态计算实际采样率（基于时间戳）
        fps = 60.0  # 默认值
        if timestamps is not None and len(timestamps) > 1:
            try:
                # 处理不同格式的时间戳
                ts_float = timestamps.astype(float)
                diffs = np.diff(ts_float)
                
                # 判断时间戳单位
                median_diff = np.median(diffs[diffs > 0])
                if median_diff > 1e6:
                    # 纳秒 (datetime64[ns])
                    diffs = diffs / 1e9
                elif median_diff > 1e3:
                    # 毫秒
                    diffs = diffs / 1e3
                # 否则假设是秒
                
                avg_interval = np.mean(diffs[diffs > 0])
                if avg_interval > 0:
                    fps = 1.0 / avg_interval
            except Exception as e:
                print(f"[ActionValidator] 计算采样率失败: {e}")
                fps = 60.0  # 回退默认值
        
        # 1. 全身异常静止检测（所有关节同时静止 > threshold s）- 使用配置阈值
        all_static_duration = self._detect_consecutive_static(joint_df, joint_columns, fps)
        static_threshold_sec = self.config.static_threshold_all  # 默认 3.0
        static_passed = all_static_duration <= static_threshold_sec
        
        issues.append(self._create_issue(
            check_name="全身静止检测",
            message=f"最长连续静止 {all_static_duration:.1f}s（阈值 {static_threshold_sec}s）",
            passed=static_passed,
            level=IssueLevel.MAJOR,
            value=all_static_duration,
            threshold=static_threshold_sec,
        ))
        
        # 2. 关键关节静止检测（任务相关关节静止 > threshold s）- 使用配置阈值
        key_joints_in_df = [j for j in KEY_JOINTS if j in joint_columns]
        if key_joints_in_df:
            key_static_duration = self._detect_consecutive_static(joint_df, key_joints_in_df, fps)
            key_threshold_sec = self.config.static_threshold_key  # 默认 5.0
            key_static_passed = key_static_duration <= key_threshold_sec
            
            issues.append(self._create_issue(
                check_name="关键关节静止检测",
                message=f"关键关节最长连续静止 {key_static_duration:.1f}s（阈值 {key_threshold_sec}s）",
                passed=key_static_passed,
                level=IssueLevel.MAJOR,
                value=key_static_duration,
                threshold=key_threshold_sec,
            ))
        
        # 3. 关节速度安全检查 - 使用实际时间差
        # 只检测主要关节（排除手指关节，手指关节速度可以很快）
        main_joint_cols = [c for c in joint_columns if not any(
            finger in c.lower() for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']
        )]
        
        time_diffs = None
        if timestamps is not None and len(timestamps) > 1:
            try:
                ts_float = timestamps.astype(float)
                raw_diffs = np.diff(ts_float)
                
                # 判断时间戳单位并转换为秒
                median_diff = np.median(raw_diffs[raw_diffs > 0]) if np.any(raw_diffs > 0) else 0
                if median_diff > 1e6:
                    # 纳秒
                    time_diffs = raw_diffs / 1e9
                elif median_diff > 1e3:
                    # 毫秒
                    time_diffs = raw_diffs / 1e3
                else:
                    # 秒
                    time_diffs = raw_diffs
            except Exception as e:
                print(f"[ActionValidator] 计算时间差失败: {e}")
                pass
        velocity_issues = self._check_joint_velocities(joint_df, main_joint_cols, time_diffs)
        unsafe_count = sum(1 for v in velocity_issues.values() if not v['safe'])
        velocity_passed = unsafe_count == 0
        
        max_velocity = max((v['max_velocity'] for v in velocity_issues.values()), default=0)
        issues.append(self._create_issue(
            check_name="关节速度安全",
            message=f"最大速度 {max_velocity:.2f} rad/s（限制 {self.config.max_joint_velocity} rad/s），超速关节 {unsafe_count} 个",
            passed=velocity_passed,
            level=IssueLevel.MAJOR,
            value=max_velocity,
            threshold=self.config.max_joint_velocity,
        ))
        
        # 4. 数据时长检查
        duration = len(joint_df) / fps  # 使用实际采样率
        min_duration = 1.0  # 至少 1 秒
        duration_passed = duration >= min_duration
        
        issues.append(self._create_issue(
            check_name="数据时长",
            message=f"时长 {duration:.1f} 秒（最少 {min_duration} 秒）",
            passed=duration_passed,
            level=IssueLevel.MAJOR,
            value=duration,
            threshold=min_duration,
        ))
        
        # 5. NaN 检查
        nan_count = joint_df[joint_columns].isna().sum().sum()
        nan_ratio = nan_count / (len(joint_df) * len(joint_columns)) if len(joint_df) > 0 and len(joint_columns) > 0 else 0
        nan_passed = nan_ratio < 0.01  # NaN 不超过 1%
        
        issues.append(self._create_issue(
            check_name="NaN 值检查",
            message=f"NaN 占比 {nan_ratio*100:.2f}%（限制 1%）",
            passed=nan_passed,
            level=IssueLevel.MAJOR,
            value=nan_ratio,
            threshold=0.01,
        ))
        
        # 计算得分
        passed_count = sum(1 for i in issues if i.passed)
        total_count = len(issues)
        score = (passed_count / total_count * 100) if total_count > 0 else 0
        overall_passed = all(
            i.passed for i in issues 
            if i.level in (IssueLevel.CRITICAL, IssueLevel.MAJOR)
        )
        
        return ValidationResult(
            passed=overall_passed,
            score=round(score, 1),
            issues=issues,
            details={
                'episode_id': episode_id,
                'frame_count': len(joint_df),
                'joint_count': len(joint_columns),
                'format': format_type,
            }
        )
    
    def _extract_joint_data(self, df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], List[str]]:
        """
        从 DataFrame 中提取关节数据
        
        支持两种数据格式：
        1. 关节作为独立列（如 left_shoulder_pitch_joint）
        2. 关节嵌套在 observation.state 列中
        
        Returns:
            (joint_df, joint_columns): 展平后的 DataFrame 和关节列名列表
        """
        # 方式1: 检查是否有直接的关节列
        joint_columns = [c for c in df.columns if '_joint' in c.lower()]
        if joint_columns:
            return df, joint_columns
        
        # 方式2: 尝试从 observation.state 解析嵌套数据
        if 'observation.state' not in df.columns:
            return None, []
        
        try:
            # 解析嵌套结构并展平
            records = []
            for idx, row in df.iterrows():
                state = row['observation.state']
                record = {}
                
                # 支持 list/ndarray 和 dict
                if hasattr(state, '__iter__') and not isinstance(state, (str, dict)):
                    for item in state:
                        if isinstance(item, dict) and 'name' in item and 'value' in item:
                            record[item['name']] = item['value']
                elif isinstance(state, dict):
                    record = state
                
                records.append(record)
            
            joint_df = pd.DataFrame(records)
            joint_columns = [c for c in joint_df.columns if '_joint' in c.lower() or 'proximal' in c.lower()]
            
            if not joint_columns:
                # 如果没有明确的关节列，使用所有数值列
                joint_columns = [c for c in joint_df.columns if joint_df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
            
            return joint_df, joint_columns
            
        except Exception as e:
            print(f"解析 observation.state 失败: {e}")
            return None, []
    
    def _detect_consecutive_static(self, df: pd.DataFrame, columns: List[str], fps: float = 60.0, threshold: float = 0.001) -> float:
        """
        检测最长连续静止时长
        
        Args:
            df: 包含关节数据的 DataFrame
            columns: 关节列名列表
            fps: 采样频率（Hz）
            threshold: 静止阈值（所有关节变化都小于此值视为静止）
        
        Returns:
            最长连续静止时长（秒）
        """
        data = df[columns].values
        if len(data) < 2:
            return 0.0
        
        # 计算帧间差异
        diffs = np.abs(np.diff(data, axis=0))
        max_diffs = np.max(diffs, axis=1)
        
        # 标记静止帧（所有关节变化都小于阈值）
        is_static = max_diffs < threshold
        
        # 找最长连续静止段
        max_consecutive = 0
        current_consecutive = 0
        
        for static in is_static:
            if static:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        # 转换为秒
        return max_consecutive / fps
    
    def _detect_static_frames(self, df: pd.DataFrame, columns: List[str], threshold: float = 0.001) -> int:
        """检测静止帧数量"""
        data = df[columns].values
        if len(data) < 2:
            return 0
        
        # 计算帧间差异
        diffs = np.abs(np.diff(data, axis=0))
        max_diffs = np.max(diffs, axis=1)
        
        # 静止帧：所有关节变化都小于阈值
        static_frames = np.sum(max_diffs < threshold)
        return int(static_frames)
    
    def _check_joint_velocities(self, df: pd.DataFrame, columns: List[str], time_diffs: np.ndarray = None) -> Dict[str, Dict]:
        """检查关节速度
        
        Args:
            df: 关节数据DataFrame
            columns: 关节列名
            time_diffs: 每帧的实际时间差（秒），如果None则使用默认采样率
        """
        results = {}
        max_velocity = self.config.max_joint_velocity
        
        for col in columns:
            data = df[col].values
            if len(data) < 2:
                continue
            
            # 位置差分
            pos_diff = np.abs(np.diff(data))
            
            # 使用实际时间差计算速度
            if time_diffs is not None and len(time_diffs) == len(pos_diff):
                # 避免除零
                safe_diffs = np.where(time_diffs > 0, time_diffs, 1e-6)
                velocities = pos_diff / safe_diffs  # rad/s
            else:
                # 回退到默认采样率
                velocities = pos_diff * 60.0
            
            # 使用P99而不是最大值，避免极少数异常跳变帧影响判定
            p99_vel = np.percentile(velocities, 99) if len(velocities) > 0 else 0
            
            results[col] = {
                'max_velocity': float(p99_vel),  # 使用P99作为"最大速度"
                'safe': p99_vel < max_velocity,
            }
        
        return results
