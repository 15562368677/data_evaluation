"""
元数据验证器

验证 Episode 的元数据完整性和文件结构
支持新格式 (Parquet) 和旧格式 (HDF5) 数据
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any

from ..core.base import (
    BaseValidator,
    ValidationResult,
    ValidationIssue,
    ValidatorConfig,
    IssueLevel,
)


class MetadataValidator(BaseValidator):
    """元数据和文件结构验证器"""
    
    # 必需的元数据字段 (新格式)
    REQUIRED_FIELDS = [
        'episode_index',
        'task_id',
        'pilot',
        'machine_id',
    ]
    
    # 推荐的元数据字段
    RECOMMENDED_FIELDS = [
        'start_time',
        'end_time',
        'session_id',
        'station_id',
    ]
    
    # 必需的数据文件 (新格式 Parquet)
    REQUIRED_FILES = [
        'metadata.json',
        'observation.state.parquet',
    ]
    
    # 可选的数据文件（至少有一个）
    OPTIONAL_FILES = [
        'action.parquet',
        'action.base.parquet',
        'observation.base_state.parquet',
        'observation.images.camera_top.parquet',
    ]
    
    # HDF5 旧格式文件
    HDF5_FILE = 'data.hdf5'
    
    @property
    def name(self) -> str:
        return "元数据验证"
    
    @property
    def category(self) -> str:
        return "元数据"
    
    def _detect_data_format(self, path: Path) -> str:
        """检测数据格式: 'parquet' (新格式) 或 'hdf5' (旧格式)"""
        metadata_file = path / 'metadata.json'
        hdf5_file = path / self.HDF5_FILE
        
        if metadata_file.exists():
            return 'parquet'
        elif hdf5_file.exists():
            return 'hdf5'
        else:
            return 'unknown'
    
    def _load_hdf5_metadata(self, hdf5_path: Path) -> Dict[str, Any]:
        """从 HDF5 文件中提取元数据"""
        import h5py
        
        metadata = {}
        try:
            with h5py.File(hdf5_path, 'r') as f:
                # 尝试从 HDF5 属性中读取元数据
                for key in f.attrs.keys():
                    val = f.attrs[key]
                    # 处理 numpy 类型
                    if hasattr(val, 'item'):
                        val = val.item()
                    elif hasattr(val, 'decode'):
                        val = val.decode('utf-8')
                    metadata[key] = val
                
                # 从数据集推断一些信息
                if 'timestamp' in f:
                    timestamps = f['timestamp'][:]
                    metadata['total_frames'] = len(timestamps)
                    if len(timestamps) > 1:
                        # 估算 FPS
                        duration = timestamps[-1] - timestamps[0]
                        if duration > 0:
                            metadata['fps'] = len(timestamps) / duration
                            metadata['duration'] = duration
                
                # 检查数据集结构
                metadata['_has_state_robot'] = 'state' in f and 'robot' in f['state']
                metadata['_has_state_hand'] = 'state' in f and 'hand' in f['state']
                metadata['_has_timestamp'] = 'timestamp' in f
                
        except Exception as e:
            metadata['_load_error'] = str(e)
        
        return metadata
    
    def _validate_hdf5_format(self, episode_id: str, path: Path) -> ValidationResult:
        """验证 HDF5 旧格式数据"""
        issues: List[ValidationIssue] = []
        hdf5_file = path / self.HDF5_FILE
        
        # 1. 检查 HDF5 文件存在性
        if not hdf5_file.exists():
            issues.append(self._create_issue(
                check_name="data.hdf5 存在性",
                message="缺少 data.hdf5 文件",
                passed=False,
                level=IssueLevel.CRITICAL,
            ))
            return ValidationResult(
                passed=False,
                score=0,
                issues=issues,
                details={'episode_id': episode_id, 'format': 'hdf5'}
            )
        
        issues.append(self._create_issue(
            check_name="data.hdf5 存在性",
            message="data.hdf5 文件存在 (旧格式)",
            passed=True,
        ))
        
        # 2. 加载并验证 HDF5 内容
        metadata = self._load_hdf5_metadata(hdf5_file)
        
        if '_load_error' in metadata:
            issues.append(self._create_issue(
                check_name="HDF5 文件格式",
                message=f"HDF5 文件读取失败: {metadata['_load_error'][:50]}",
                passed=False,
                level=IssueLevel.CRITICAL,
            ))
            return ValidationResult(passed=False, score=0, issues=issues)
        
        issues.append(self._create_issue(
            check_name="HDF5 文件格式",
            message="HDF5 文件格式正确",
            passed=True,
        ))
        
        # 3. 验证 HDF5 数据结构
        if metadata.get('_has_timestamp'):
            issues.append(self._create_issue(
                check_name="时间戳数据",
                message="包含时间戳数据",
                passed=True,
            ))
        else:
            issues.append(self._create_issue(
                check_name="时间戳数据",
                message="缺少时间戳数据",
                passed=False,
                level=IssueLevel.MAJOR,
            ))
        
        if metadata.get('_has_state_robot'):
            issues.append(self._create_issue(
                check_name="机器人状态数据",
                message="包含机器人状态数据 (state/robot)",
                passed=True,
            ))
        else:
            issues.append(self._create_issue(
                check_name="机器人状态数据",
                message="缺少机器人状态数据",
                passed=False,
                level=IssueLevel.MAJOR,
            ))
        
        if metadata.get('_has_state_hand'):
            issues.append(self._create_issue(
                check_name="手部状态数据",
                message="包含手部状态数据 (state/hand)",
                passed=True,
            ))
        else:
            issues.append(self._create_issue(
                check_name="手部状态数据",
                message="缺少手部状态数据",
                passed=True,  # 手部数据可选
                level=IssueLevel.MINOR,
            ))
        
        # 4. 验证时长 (如果有)
        duration = metadata.get('duration')
        if duration is not None:
            if self.config.duration_min is not None:
                passed = duration >= self.config.duration_min
                issues.append(self._create_issue(
                    check_name="最小时长限制",
                    message=f"时长 {duration:.1f}s >= {self.config.duration_min}s" if passed else f"时长 {duration:.1f}s < {self.config.duration_min}s",
                    passed=passed,
                    level=IssueLevel.CRITICAL,
                    value=duration,
                    threshold=self.config.duration_min
                ))
            
            if self.config.duration_max is not None:
                passed = duration <= self.config.duration_max
                issues.append(self._create_issue(
                    check_name="最大时长限制",
                    message=f"时长 {duration:.1f}s <= {self.config.duration_max}s" if passed else f"时长 {duration:.1f}s > {self.config.duration_max}s",
                    passed=passed,
                    level=IssueLevel.CRITICAL,
                    value=duration,
                    threshold=self.config.duration_max
                ))
        
        # 5. 检查视频文件（旧格式视频可能在子目录中）
        video_files = [
            'top/rgb/video.mp4',
            'video.mp4',
            'camera_top/rgb/video.mp4',
            '0/rgb/video.mp4',
        ]
        has_video = any((path / vf).exists() for vf in video_files)
        video_msg = "存在视频文件" if has_video else "缺少视频文件（旧格式数据可能未下载视频）"
        issues.append(self._create_issue(
            check_name="视频文件",
            message=video_msg,
            passed=True,  # 视频文件对于旧格式数据是可选的，不影响验证结果
            level=IssueLevel.INFO if not has_video else IssueLevel.MINOR,
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
                'format': 'hdf5',
                'metadata': metadata,
            }
        )
    
    def validate(self, episode_id: str, data_path: str) -> ValidationResult:
        """验证元数据完整性，支持新旧两种格式"""
        issues: List[ValidationIssue] = []
        path = Path(data_path)
        
        # 检测数据格式
        data_format = self._detect_data_format(path)
        
        # 如果是 HDF5 旧格式，使用专门的验证逻辑
        if data_format == 'hdf5':
            return self._validate_hdf5_format(episode_id, path)
        
        # 以下是新格式 (Parquet) 的验证逻辑
        
        # 1. 检查 metadata.json 是否存在
        metadata_file = path / 'metadata.json'
        if not metadata_file.exists():
            # 检查是否有 HDF5 文件（可能下载时出错）
            hdf5_file = path / self.HDF5_FILE
            if hdf5_file.exists():
                return self._validate_hdf5_format(episode_id, path)
            
            issues.append(self._create_issue(
                check_name="metadata.json 存在性",
                message="缺少 metadata.json 文件",
                passed=False,
                level=IssueLevel.CRITICAL,
            ))
            # 无法继续验证
            return ValidationResult(
                passed=False,
                score=0,
                issues=issues,
                details={'episode_id': episode_id, 'format': 'unknown'}
            )
        else:
            issues.append(self._create_issue(
                check_name="metadata.json 存在性",
                message="metadata.json 文件存在",
                passed=True,
            ))
        
        # 2. 读取并验证 metadata
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except Exception as e:
            issues.append(self._create_issue(
                check_name="metadata.json 格式",
                message=f"metadata.json 解析失败: {str(e)[:50]}",
                passed=False,
                level=IssueLevel.CRITICAL,
            ))
            return ValidationResult(passed=False, score=0, issues=issues)
        
        issues.append(self._create_issue(
            check_name="metadata.json 格式",
            message="metadata.json 格式正确",
            passed=True,
        ))
        
        # 3. 验证必需字段
        for field in self.REQUIRED_FIELDS:
            if field in metadata and metadata[field] is not None:
                issues.append(self._create_issue(
                    check_name=f"必需字段: {field}",
                    message=f"字段 {field} = {metadata[field]}",
                    passed=True,
                ))
            elif field in metadata:
                # 字段存在但值为空，降级为警告
                issues.append(self._create_issue(
                    check_name=f"必需字段: {field}",
                    message=f"字段 {field} 值为空",
                    passed=True,  # 改为通过，但记录警告
                    level=IssueLevel.MINOR,
                ))
            else:
                issues.append(self._create_issue(
                    check_name=f"必需字段: {field}",
                    message=f"缺少必需字段 {field}",
                    passed=False,
                    level=IssueLevel.MAJOR,
                ))
        
        # 4. 验证推荐字段
        for field in self.RECOMMENDED_FIELDS:
            if field in metadata and metadata[field] is not None:
                issues.append(self._create_issue(
                    check_name=f"推荐字段: {field}",
                    message=f"字段 {field} 存在",
                    passed=True,
                    level=IssueLevel.MINOR,
                ))
            else:
                issues.append(self._create_issue(
                    check_name=f"推荐字段: {field}",
                    message=f"缺少推荐字段 {field}",
                    passed=False,
                    level=IssueLevel.MINOR,
                ))
        
        # 5. 验证必需数据文件
        for filename in self.REQUIRED_FILES:
            file_path = path / filename
            if file_path.exists():
                issues.append(self._create_issue(
                    check_name=f"必需文件: {filename}",
                    message=f"文件 {filename} 存在",
                    passed=True,
                ))
            else:
                issues.append(self._create_issue(
                    check_name=f"必需文件: {filename}",
                    message=f"缺少必需文件 {filename}",
                    passed=False,
                    level=IssueLevel.CRITICAL,
                ))
        
        # 6. 验证可选文件（至少有一个 action 文件）
        action_files = ['action.parquet', 'action.base.parquet']
        has_action = any((path / f).exists() for f in action_files)
        issues.append(self._create_issue(
            check_name="Action 数据文件",
            message="存在 Action 数据文件" if has_action else "缺少所有 Action 数据文件",
            passed=has_action,
            level=IssueLevel.MAJOR,
        ))
        
        # 7. 验证时长 (如果配置了阈值)
        # 尝试从 metadata 获取时长信息
        duration = None
        if 'total_frames' in metadata:
            fps = metadata.get('fps', 30.0)
            if fps > 0:
                duration = metadata['total_frames'] / fps
        elif 'duration' in metadata:
            duration = metadata['duration']
            
        if duration is not None:
            # 7.1 绝对时长检查
            if self.config.duration_min is not None:
                passed = duration >= self.config.duration_min
                issues.append(self._create_issue(
                    check_name="最小时长限制",
                    message=f"时长 {duration:.1f}s >= {self.config.duration_min}s",
                    passed=passed,
                    level=IssueLevel.CRITICAL,
                    value=duration,
                    threshold=self.config.duration_min
                ))
            
            if self.config.duration_max is not None:
                passed = duration <= self.config.duration_max
                issues.append(self._create_issue(
                    check_name="最大时长限制",
                    message=f"时长 {duration:.1f}s <= {self.config.duration_max}s",
                    passed=passed,
                    level=IssueLevel.CRITICAL,
                    value=duration,
                    threshold=self.config.duration_max
                ))
                
            # 7.2 百分位时长检查
            if (self.config.duration_percentile_min is not None or 
                self.config.duration_percentile_max is not None) and 'task_id' in metadata:
                try:
                    from arts_analysis.services.database.data_service import get_fetcher
                    import numpy as np
                    
                    fetcher = get_fetcher()
                    # 注意：这里 metadata['task_id'] 可能是 string，需要转 int
                    # 如果是 task_template_id 还是 task_instance_id? 
                    # 通常 metadata 里的 task_id 对应数据库 tasks.id (instance_id)
                    # 但有时可能是 template_id。这里假设是 instance_id，因为 data_fetcher 已修正为用 instance_id
                    task_id_val = int(metadata['task_id'])
                    
                    # 获取历史时长分布
                    history_durations = fetcher.get_task_durations(task_id_val)
                    
                    if history_durations and len(history_durations) > 5:
                        if self.config.duration_percentile_min is not None:
                            p_min = self.config.duration_percentile_min
                            th_min = float(np.percentile(history_durations, p_min))
                            passed = duration >= th_min
                            issues.append(self._create_issue(
                                check_name=f"时长百分位 > P{p_min}",
                                message=f"时长 {duration:.1f}s >= P{p_min}({th_min:.1f}s)",
                                passed=passed,
                                level=IssueLevel.CRITICAL,
                                value=duration,
                                threshold=th_min
                            ))
                            
                        if self.config.duration_percentile_max is not None:
                            p_max = self.config.duration_percentile_max
                            th_max = float(np.percentile(history_durations, p_max))
                            passed = duration <= th_max
                            issues.append(self._create_issue(
                                check_name=f"时长百分位 < P{p_max}",
                                message=f"时长 {duration:.1f}s <= P{p_max}({th_max:.1f}s)",
                                passed=passed,
                                level=IssueLevel.CRITICAL,
                                value=duration,
                                threshold=th_max
                            ))
                except Exception as e:
                    print(f"[MetadataValidator] 时长百分位检查失败: {e}")
                    # 不阻断，仅记录警告
                    issues.append(self._create_issue(
                        check_name="时长百分位检查",
                        message=f"检查失败: {str(e)}",
                        passed=True,
                        level=IssueLevel.INFO
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
                'metadata': metadata,
            }
        )
