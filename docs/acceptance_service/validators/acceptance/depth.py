"""
深度数据验证器

验证深度图像质量：无效像素、连续性、精度等
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional

from ..core.base import (
    BaseValidator,
    ValidationResult,
    ValidationIssue,
    ValidatorConfig,
    IssueLevel,
)


class DepthValidator(BaseValidator):
    """深度数据验证器"""
    
    @property
    def name(self) -> str:
        return "深度数据验证"
    
    @property
    def category(self) -> str:
        return "深度数据"
    
    def _is_hdf5_format(self, path: Path) -> bool:
        """检测是否为 HDF5 旧格式数据"""
        hdf5_file = path / 'data.hdf5'
        metadata_file = path / 'metadata.json'
        return hdf5_file.exists() and not metadata_file.exists()
    
    def _validate_hdf5_depth(self, path: Path, issues: List[ValidationIssue]) -> bool:
        """验证 HDF5 格式的深度视频数据"""
        import cv2
        
        # 查找深度视频文件
        depth_paths = [
            path / 'top' / 'depth' / 'video.mkv',
            path / 'depth_video.mkv',
            path / 'camera_top' / 'depth' / 'video.mkv',
        ]
        
        depth_file = None
        for dp in depth_paths:
            if dp.exists():
                depth_file = dp
                break
        
        if not depth_file:
            issues.append(self._create_issue(
                check_name="深度数据文件",
                message="HDF5 格式：未找到深度视频文件",
                passed=False,
                level=IssueLevel.MAJOR,
            ))
            return False
        
        try:
            cap = cv2.VideoCapture(str(depth_file))
            if not cap.isOpened():
                issues.append(self._create_issue(
                    check_name="深度数据文件",
                    message="无法打开深度视频文件",
                    passed=False,
                    level=IssueLevel.MAJOR,
                ))
                return False
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            issues.append(self._create_issue(
                check_name="深度数据文件",
                message=f"HDF5 格式：{frame_count} 帧, {fps:.1f} FPS, {width}×{height}",
                passed=True,
            ))
            
            # 采样检查深度质量
            invalid_ratios = []
            continuity_ratios = []
            sample_step = max(1, frame_count // 10)
            prev_depth_img = None
            
            for idx in range(0, frame_count, sample_step):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # MKV 深度视频通常是单通道或三通道相同
                if frame.ndim == 3:
                    depth_img = frame[:, :, 0]
                else:
                    depth_img = frame
                
                # 验证单帧
                self._validate_depth_frame(depth_img, invalid_ratios)
                # 验证连续性
                self._validate_continuity(depth_img, prev_depth_img, continuity_ratios)
                prev_depth_img = depth_img
            
            cap.release()
            
            # 汇总结果
            self._summarize_results("depth", invalid_ratios, continuity_ratios, issues)
            return True
            
        except Exception as e:
            issues.append(self._create_issue(
                check_name="深度数据文件",
                message=f"读取失败: {str(e)[:50]}",
                passed=False,
                level=IssueLevel.MAJOR,
            ))
            return False
    
    def validate(self, episode_id: str, data_path: str) -> ValidationResult:
        """验证深度数据质量"""
        issues: List[ValidationIssue] = []
        path = Path(data_path)
        
        # 检测是否为 HDF5 旧格式
        if self._is_hdf5_format(path):
            # 旧格式数据使用深度视频文件进行验证
            self._validate_hdf5_depth(path, issues)
            
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
                details={'episode_id': episode_id, 'format': 'hdf5'}
            )
        
        # 检查深度数据文件
        # 假设深度文件命名模式为 observation.images.*_depth.parquet
        depth_files = list(path.glob('observation.images.*_depth.parquet'))
        
        if not depth_files:
            # 尝试查找 camera_top_depth
            top_depth = path / 'observation.images.camera_top_depth.parquet'
            if top_depth.exists():
                depth_files = [top_depth]
                
        # 尝试查找 depth_video.mkv
        depth_video = path / 'depth_video.mkv'
        if depth_video.exists():
            depth_files.append(depth_video)
        
        if not depth_files:
            issues.append(self._create_issue(
                check_name="深度数据文件",
                message="未找到深度数据文件",
                passed=False,
                level=IssueLevel.MAJOR,
            ))
            return ValidationResult(
                passed=False,
                score=0,
                issues=issues,
                details={'episode_id': episode_id}
            )
        
        issues.append(self._create_issue(
            check_name="深度数据文件",
            message=f"找到 {len(depth_files)} 个深度数据文件",
            passed=True,
        ))
        
        # 验证每个深度文件
        for depth_file in depth_files:
            camera_name = depth_file.stem.replace('observation.images.', '').replace('_depth', '')
            if depth_file.name == 'depth_video.mkv':
                camera_name = 'camera_top'
            
            try:
                invalid_ratios = []
                continuity_ratios = []
                frame_count = 0
                
                if depth_file.suffix == '.parquet':
                    df = pd.read_parquet(depth_file)
                    frame_count = len(df)
                    
                    issues.append(self._create_issue(
                        check_name=f"深度 {camera_name} 读取",
                        message=f"成功加载 {frame_count} 帧 (Parquet)",
                        passed=True,
                    ))
                    
                    # 检查深度数据列
                    depth_col = None
                    for col in df.columns:
                        if 'depth' in col.lower() or col == 'image':
                            depth_col = col
                            break
                    
                    if not depth_col:
                        issues.append(self._create_issue(
                            check_name=f"深度 {camera_name} 格式",
                            message="未找到深度数据列",
                            passed=True,
                            level=IssueLevel.MINOR,
                        ))
                        continue
                    
                    # 采样检查
                    sample_step = max(1, frame_count // 10)
                    sample_indices = list(range(0, frame_count, sample_step))
                    prev_depth_img = None
                    
                    for idx in sample_indices:
                        img_data = df[depth_col].iloc[idx]
                        if hasattr(img_data, 'shape'):
                            depth_img = img_data
                            # 验证单帧
                            self._validate_depth_frame(depth_img, invalid_ratios)
                            # 验证连续性
                            self._validate_continuity(depth_img, prev_depth_img, continuity_ratios)
                            prev_depth_img = depth_img
                            
                elif depth_file.suffix == '.mkv':
                    import cv2
                    cap = cv2.VideoCapture(str(depth_file))
                    if not cap.isOpened():
                        raise RuntimeError("无法打开视频文件")
                        
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    issues.append(self._create_issue(
                        check_name=f"深度 {camera_name} 读取",
                        message=f"成功加载 {frame_count} 帧 (MKV)",
                        passed=True,
                    ))
                    
                    # 采样检查
                    sample_step = max(1, frame_count // 10)
                    prev_depth_img = None
                    
                    for idx in range(0, frame_count, sample_step):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                        ret, frame = cap.read()
                        if not ret:
                            continue
                            
                        # MKV 读取的是 BGR，如果是单通道深度图被存为视频，通常三个通道相同
                        # 取第一个通道作为深度值
                        if frame.ndim == 3:
                            depth_img = frame[:, :, 0]
                        else:
                            depth_img = frame
                            
                        # 验证单帧
                        self._validate_depth_frame(depth_img, invalid_ratios)
                        # 验证连续性
                        self._validate_continuity(depth_img, prev_depth_img, continuity_ratios)
                        prev_depth_img = depth_img
                        
                    cap.release()
                
                # 汇总结果
                self._summarize_results(camera_name, invalid_ratios, continuity_ratios, issues)

            except Exception as e:
                issues.append(self._create_issue(
                    check_name=f"深度 {camera_name} 读取",
                    message=f"读取失败: {str(e)[:50]}",
                    passed=False,
                    level=IssueLevel.MAJOR,
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
                'depth_files': [f.name for f in depth_files],
            }
        )

    def _validate_depth_frame(self, depth_img, invalid_ratios):
        """验证单帧深度图"""
        total_pixels = depth_img.size
        if total_pixels > 0:
            # 对于 uint8 (MKV)，0 通常是无效值
            # 对于 float (Parquet)，0 或 NaN 是无效值
            if depth_img.dtype == np.uint8:
                invalid_pixels = np.sum(depth_img == 0)
            else:
                invalid_pixels = np.sum(depth_img == 0) + np.sum(np.isnan(depth_img))
            invalid_ratios.append(invalid_pixels / total_pixels)

    def _validate_continuity(self, depth_img, prev_depth_img, continuity_ratios):
        """验证深度图连续性"""
        if prev_depth_img is not None and prev_depth_img.shape == depth_img.shape:
            if depth_img.dtype == np.uint8:
                valid_curr = (depth_img > 0)
                valid_prev = (prev_depth_img > 0)
            else:
                valid_curr = (depth_img > 0) & (~np.isnan(depth_img))
                valid_prev = (prev_depth_img > 0) & (~np.isnan(prev_depth_img))
                
            overlap = np.sum(valid_curr & valid_prev)
            union = np.sum(valid_curr | valid_prev)
            if union > 0:
                continuity_ratios.append(overlap / union)

    def _summarize_results(self, camera_name, invalid_ratios, continuity_ratios, issues):
        """汇总验证结果"""
        if invalid_ratios:
            avg_invalid = np.mean(invalid_ratios)
            max_invalid = self.config.depth_invalid_pixel_max
            passed = avg_invalid <= max_invalid
            
            issues.append(self._create_issue(
                check_name=f"深度 {camera_name} 完整性",
                message=f"无效像素 {avg_invalid*100:.1f}%（阈值 {max_invalid*100:.0f}%）",
                passed=passed,
                level=IssueLevel.MAJOR,
                value=avg_invalid,
                threshold=max_invalid,
            ))
            
        if continuity_ratios:
            avg_continuity = np.mean(continuity_ratios)
            min_continuity = self.config.depth_continuity_min
            continuity_passed = avg_continuity >= min_continuity
            
            issues.append(self._create_issue(
                check_name=f"深度 {camera_name} 连续性",
                message=f"连续性 {avg_continuity*100:.1f}%（阈值 {min_continuity*100:.0f}%）",
                passed=continuity_passed,
                level=IssueLevel.MAJOR,
                value=avg_continuity,
                threshold=min_continuity,
            ))
