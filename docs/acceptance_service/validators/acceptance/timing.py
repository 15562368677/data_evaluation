"""
时间同步验证器

验证时间戳单调性、采样间隔一致性、频率稳定性
支持新格式 (Parquet) 和旧格式 (HDF5) 数据
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Tuple

from ..core.base import (
    BaseValidator,
    ValidationResult,
    ValidationIssue,
    ValidatorConfig,
    IssueLevel,
)


class TimingValidator(BaseValidator):
    """时间同步验证器"""
    
    @property
    def name(self) -> str:
        return "时间同步验证"
    
    @property
    def category(self) -> str:
        return "时间同步"
    
    def _normalize_timestamps_to_seconds(self, diffs: np.ndarray) -> np.ndarray:
        """
        将时间戳差值归一化为秒
        
        自动检测时间戳单位（纳秒、毫秒、秒）并转换
        """
        if len(diffs) == 0:
            return diffs
        
        median_diff = np.median(diffs[diffs > 0]) if np.any(diffs > 0) else 0
        
        if median_diff > 1e15:
            # 纳秒 (datetime64[ns] 的 int64 表示)
            return diffs / 1e9
        elif median_diff > 1e6:
            # 可能是纳秒（较小的值）
            return diffs / 1e9
        elif median_diff > 1e3:
            # 毫秒
            return diffs / 1e3
        else:
            # 秒
            return diffs
    
    def _load_timestamps(self, path: Path) -> Tuple[Optional[np.ndarray], str, int]:
        """
        加载时间戳数据，支持 Parquet 和 HDF5 格式
        
        Returns:
            (timestamps, format_type, frame_count): 时间戳数组、格式类型、帧数
        """
        state_file = path / 'observation.state.parquet'
        hdf5_file = path / 'data.hdf5'
        
        # 优先尝试 Parquet 格式
        if state_file.exists():
            try:
                df = pd.read_parquet(state_file)
                # 获取时间戳列
                for col in ['timestamp_utc', 'user_timestamp_utc', 'timestamp_uhlc']:
                    if col in df.columns:
                        return df[col].values, 'parquet', len(df)
                return None, 'parquet', len(df)
            except Exception:
                pass
        
        # 尝试 HDF5 格式
        if hdf5_file.exists():
            try:
                import h5py
                with h5py.File(hdf5_file, 'r') as f:
                    if 'timestamp' in f:
                        timestamps = f['timestamp'][:]
                        return timestamps, 'hdf5', len(timestamps)
                return None, 'hdf5', 0
            except Exception:
                pass
        
        return None, 'unknown', 0
    
    def validate(self, episode_id: str, data_path: str) -> ValidationResult:
        """验证时间同步质量"""
        issues: List[ValidationIssue] = []
        path = Path(data_path)
        
        # 加载时间戳数据（支持 Parquet 和 HDF5）
        timestamps, format_type, frame_count = self._load_timestamps(path)
        
        if format_type == 'unknown':
            issues.append(self._create_issue(
                check_name="状态数据文件",
                message="缺少 observation.state.parquet 或 data.hdf5",
                passed=False,
                level=IssueLevel.CRITICAL,
            ))
            return ValidationResult(passed=False, score=0, issues=issues)
        
        if timestamps is None:
            issues.append(self._create_issue(
                check_name="时间戳字段",
                message=f"未找到时间戳字段 (格式: {format_type})",
                passed=False,
                level=IssueLevel.CRITICAL,
            ))
            return ValidationResult(passed=False, score=0, issues=issues)
        
        issues.append(self._create_issue(
            check_name="状态数据文件",
            message=f"成功加载 {frame_count} 帧数据 (格式: {format_type})",
            passed=True,
        ))
        
        issues.append(self._create_issue(
            check_name="时间戳字段",
            message=f"使用 {'timestamp' if format_type == 'hdf5' else 'timestamp_utc'} 作为时间戳",
            passed=True,
        ))
        
        # 1. 时间戳单调递增检查
        raw_diffs = np.diff(timestamps.astype(float))
        non_monotonic = np.sum(raw_diffs <= 0)
        monotonic_ratio = 1 - (non_monotonic / len(raw_diffs)) if len(raw_diffs) > 0 else 1
        monotonic_passed = monotonic_ratio >= 0.99  # 99% 单调
        
        issues.append(self._create_issue(
            check_name="时间戳单调递增",
            message=f"单调性 {monotonic_ratio*100:.1f}%，非单调点 {non_monotonic} 个",
            passed=monotonic_passed,
            level=IssueLevel.MAJOR,
            value=monotonic_ratio,
            threshold=0.99,
        ))
        
        # 将时间差归一化为秒
        diffs_sec = self._normalize_timestamps_to_seconds(raw_diffs)
        
        # 2. 采样间隔一致性
        if len(diffs_sec) > 0:
            positive_diffs = diffs_sec[diffs_sec > 0]
            if len(positive_diffs) > 0:
                median_interval_sec = np.median(positive_diffs)
                interval_std = np.std(positive_diffs)
                interval_cv = interval_std / median_interval_sec if median_interval_sec > 0 else 0
                
                gap_tolerance = self.config.timestamp_gap_tolerance
                interval_passed = interval_cv < gap_tolerance
                
                issues.append(self._create_issue(
                    check_name="采样间隔一致性",
                    message=f"变异系数 {interval_cv*100:.2f}%（阈值 {gap_tolerance*100:.0f}%）",
                    passed=interval_passed,
                    level=IssueLevel.MAJOR,
                    value=interval_cv,
                    threshold=gap_tolerance,
                ))
                
                # 3. 估算采样频率
                estimated_freq = 1.0 / median_interval_sec if median_interval_sec > 0 else 0
                
                min_freq = self.config.min_sampling_rate
                freq_passed = estimated_freq >= min_freq
                
                issues.append(self._create_issue(
                    check_name="采样频率",
                    message=f"估算频率 {estimated_freq:.1f} Hz（最低 {min_freq} Hz）",
                    passed=freq_passed,
                    level=IssueLevel.MAJOR,
                    value=estimated_freq,
                    threshold=min_freq,
                ))
        
        # 4. 检测大的时间跳跃（数据中断检测：采样中断>1s）
        if len(diffs_sec) > 0:
            # 标准：采样中断>1s 视为异常
            large_gaps = np.sum(diffs_sec > 1.0)
            gap_ratio = large_gaps / len(diffs_sec)
            gap_passed = gap_ratio < 0.01  # 不超过1%
            
            issues.append(self._create_issue(
                check_name="数据中断检测",
                message=f"中断>1s 共 {large_gaps} 个（占比 {gap_ratio*100:.2f}%）",
                passed=gap_passed,
                level=IssueLevel.MAJOR,
                value=gap_ratio,
                threshold=0.01,
            ))
        
        # 5. 频率一致性检测
        # 核心是计算时间间隔(Δt)的标准差，标准差越小，一致性越好
        if len(diffs_sec) > 0:
            positive_diffs = diffs_sec[diffs_sec > 0]
            if len(positive_diffs) > 10:
                # 排除开头和结尾各10%的数据（可能有启动/停止抖动）
                n = len(positive_diffs)
                start_idx = int(n * 0.1)
                end_idx = int(n * 0.9)
                intervals_sec = positive_diffs[start_idx:end_idx]
                
                # 计算统计指标
                mean_interval = np.mean(intervals_sec)  # 平均间隔
                std_dev = np.std(intervals_sec)         # 标准差（一致性核心指标）
                cv = std_dev / mean_interval if mean_interval > 0 else 0  # 变异系数
                actual_freq = 1.0 / mean_interval if mean_interval > 0 else 0  # 实际频率
                
                # 最大抖动范围
                max_jitter = np.max(intervals_sec) - np.min(intervals_sec)
                
                # 使用配置中的阈值
                freq_tolerance = self.config.frequency_tolerance  # 默认 0.02 (2%)
                variation_passed = cv <= freq_tolerance
                issues.append(self._create_issue(
                    check_name="频率波动",
                    message=f"CV={cv*100:.2f}% (阈值 {freq_tolerance*100:.1f}%)",
                    passed=variation_passed,
                    level=IssueLevel.MAJOR,
                    value=cv,
                    threshold=freq_tolerance,
                ))
                
                # 频率一致性：使用 frequency_consistency_min
                # 一致性定义为 1 - CV (变异系数)
                consistency_score = 1.0 - cv
                min_consistency = self.config.frequency_consistency_min # 默认 0.98 (98%)
                
                consistency_passed = consistency_score >= min_consistency
                issues.append(self._create_issue(
                    check_name="频率一致性",
                    message=f"一致性 {consistency_score*100:.2f}% (要求 > {min_consistency*100:.1f}%)",
                    passed=consistency_passed,
                    level=IssueLevel.MAJOR,
                    value=consistency_score,
                    threshold=min_consistency,
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
            details={'episode_id': episode_id, 'frame_count': frame_count}
        )
