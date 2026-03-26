"""
末端执行器轨迹验证器

验证抓取事件和 EE 轨迹
"""

from typing import Optional
from arts_analysis.services.acceptance_service.validators.core.base import (
    BaseValidator, 
    ValidationResult, 
    ValidationIssue,
    IssueLevel,
    ValidatorConfig,
)
from arts_analysis.services.trajectory_service.ee_trajectory_service import get_ee_trajectory_service


class EETrajectoryValidator(BaseValidator):
    """末端执行器轨迹验证器"""
    
    @property
    def name(self) -> str:
        return "EE 轨迹验证器"
    
    @property
    def category(self) -> str:
        return "末端轨迹"
    
    def _is_hdf5_format(self, data_path: str) -> bool:
        """检测是否为 HDF5 旧格式数据"""
        from pathlib import Path
        path = Path(data_path)
        hdf5_file = path / 'data.hdf5'
        metadata_file = path / 'metadata.json'
        return hdf5_file.exists() and not metadata_file.exists()
    
    def _validate_hdf5_trajectory(self, data_path: str, issues: list, details: dict) -> bool:
        """验证 HDF5 格式的轨迹数据"""
        from pathlib import Path
        import h5py
        import numpy as np
        
        path = Path(data_path)
        hdf5_file = path / 'data.hdf5'
        
        try:
            with h5py.File(hdf5_file, 'r') as f:
                # 检查是否有机器人状态数据
                if 'state' not in f or 'robot' not in f['state']:
                    issues.append(self._create_issue(
                        check_name="EE 轨迹读取",
                        message="HDF5 格式：缺少 state/robot 数据",
                        passed=False,
                        level=IssueLevel.MINOR,
                    ))
                    return False
                
                robot_state = f['state']['robot'][:]
                timestamps = f['timestamp'][:] if 'timestamp' in f else None
                
                n_frames = len(robot_state)
                n_joints = robot_state.shape[1] if robot_state.ndim > 1 else 0
                
                issues.append(self._create_issue(
                    check_name="EE 轨迹读取",
                    message=f"HDF5 格式：{n_frames} 帧, {n_joints} 关节",
                    passed=True,
                ))
                
                # 检测关节运动（简化版抓取检测）
                # 对于旧格式数据，我们检查关节是否有明显运动
                if n_frames > 10 and n_joints > 0:
                    # 计算关节变化范围
                    joint_ranges = np.max(robot_state, axis=0) - np.min(robot_state, axis=0)
                    max_range = np.max(joint_ranges)
                    
                    # 如果关节有明显运动（> 0.1 rad），认为有动作
                    has_motion = max_range > 0.1
                    
                    issues.append(self._create_issue(
                        check_name="关节运动检测",
                        message=f"最大关节变化范围: {max_range:.2f} rad",
                        passed=has_motion,
                        level=IssueLevel.MINOR,
                        value=max_range,
                        threshold=0.1,
                    ))
                    
                    details["has_motion"] = has_motion
                    details["max_joint_range"] = float(max_range)
                
                # 检查手部数据（如果有）
                if 'hand' in f['state']:
                    hand_state = f['state']['hand'][:]
                    hand_ranges = np.max(hand_state, axis=0) - np.min(hand_state, axis=0)
                    max_hand_range = np.max(hand_ranges)
                    
                    # 手部有明显运动可能表示抓取动作
                    has_grasp_motion = max_hand_range > 0.1
                    
                    issues.append(self._create_issue(
                        check_name="手部运动检测",
                        message=f"最大手部变化范围: {max_hand_range:.2f} rad",
                        passed=True,  # 信息性检查
                        level=IssueLevel.INFO,
                        value=max_hand_range,
                    ))
                    
                    details["has_grasp_motion"] = has_grasp_motion
                    details["max_hand_range"] = float(max_hand_range)
                
                return True
                
        except Exception as e:
            issues.append(self._create_issue(
                check_name="EE 轨迹读取",
                message=f"HDF5 读取失败: {str(e)[:50]}",
                passed=False,
                level=IssueLevel.MINOR,
            ))
            return False
    
    def validate(self, episode_id: str, data_path: str) -> ValidationResult:
        """
        验证 EE 轨迹和抓取事件
        
        检查项：
        1. 是否检测到抓取事件
        
        输出详情 (details):
        - grasp_positions: {left: [x,y,z], right: [x,y,z]}
        - place_positions: {left: [x,y,z], right: [x,y,z]}
        
        注：密度标签在数据导出时批量计算
        """
        issues = []
        
        # 初始化详情字典
        details = {
            "has_grasp_events": False,
            "grasp_positions": {},
            "place_positions": {},
        }
        
        # 检测是否为 HDF5 旧格式
        if self._is_hdf5_format(data_path):
            # 旧格式数据使用 HDF5 进行轨迹验证
            details["format"] = "hdf5"
            self._validate_hdf5_trajectory(data_path, issues, details)
            
            # 计算得分
            passed_count = sum(1 for i in issues if i.passed)
            score = (passed_count / len(issues) * 100) if issues else 0
            
            return ValidationResult(
                passed=all(i.passed for i in issues if i.level != IssueLevel.INFO),
                score=round(score, 1),
                issues=issues,
                details=details
            )
        
        ee_service = get_ee_trajectory_service()
        
        # 获取配置
        threshold = getattr(self.config, 'grasp_threshold', 0.5)
        machine_id = getattr(self.config, 'machine_id', 'gr2')
        
        # 检测抓取事件
        grasp_result = ee_service.detect_grasp_events(data_path, strategy='action_threshold', threshold=threshold)
        
        # 更新详情字典
        details["machine_id"] = machine_id
        
        # 检查是否有错误
        if "error" in grasp_result:
            issues.append(self._create_issue(
                check_name="EE 轨迹读取",
                message=f"读取失败: {grasp_result['error'][:50]}",
                passed=False,
                level=IssueLevel.MINOR,
            ))
            details["error"] = grasp_result["error"]
            return ValidationResult(
                passed=False,
                score=0,
                issues=issues,
                details=details
            )
        
        # 检查是否有抓取事件
        has_events = grasp_result.get("has_grasp_events", False)
        details["has_grasp_events"] = has_events
        
        issues.append(self._create_issue(
            check_name="抓取事件检测",
            message="检测到抓取事件" if has_events else "未检测到抓取事件",
            passed=has_events,
            level=IssueLevel.MINOR,
            value=1 if has_events else 0,
            threshold=1,
        ))
        
        # 获取抓取/放置坐标 (仅当检测到事件时)
        if has_events:
            try:
                # 计算抓取和放置位置坐标
                ee_positions = ee_service.compute_ee_positions_at_events(
                    data_path, 
                    machine_id=machine_id,
                    strategy='action_threshold', 
                    threshold=threshold
                )
                
                if "error" not in ee_positions:
                    # 提取抓取位置 (first)
                    if "left_first" in ee_positions:
                        details["grasp_positions"]["left"] = ee_positions["left_first"]
                    if "right_first" in ee_positions:
                        details["grasp_positions"]["right"] = ee_positions["right_first"]
                    
                    # 提取放置位置 (last)
                    if "left_last" in ee_positions:
                        details["place_positions"]["left"] = ee_positions["left_last"]
                    if "right_last" in ee_positions:
                        details["place_positions"]["right"] = ee_positions["right_last"]
                        
            except Exception as e:
                print(f"[WARN] Position extraction failed: {e}")
        
        # 计算得分
        passed_count = sum(1 for i in issues if i.passed)
        score = (passed_count / len(issues) * 100) if issues else 0
        
        return ValidationResult(
            passed=all(i.passed for i in issues if i.level != IssueLevel.MINOR),
            score=round(score, 1),
            issues=issues,
            details=details
        )

