"""
视觉数据验证器

验证图像质量：过曝、欠曝、异常图像等
注：由于视觉数据通常存储为 parquet 中的编码数据，这里提供基础验证框架
"""

import numpy as np
import pandas as pd
import io
from PIL import Image
from pathlib import Path
from typing import List, Optional, Dict

from ..core.base import (
    BaseValidator,
    ValidationResult,
    ValidationIssue,
    ValidatorConfig,
    IssueLevel,
)


class VisualValidator(BaseValidator):
    """视觉数据验证器"""
    
    @property
    def name(self) -> str:
        return "视觉数据验证"
    
    @property
    def category(self) -> str:
        return "视觉质量"
    
    def _is_hdf5_format(self, path: Path) -> bool:
        """检测是否为 HDF5 旧格式数据"""
        hdf5_file = path / 'data.hdf5'
        metadata_file = path / 'metadata.json'
        return hdf5_file.exists() and not metadata_file.exists()
    
    def _validate_hdf5_video(self, path: Path, issues: List[ValidationIssue]) -> bool:
        """验证 HDF5 格式的视频数据"""
        import cv2
        
        # 查找视频文件
        video_paths = [
            path / 'top' / 'rgb' / 'video.mp4',
            path / 'video.mp4',
            path / 'camera_top' / 'rgb' / 'video.mp4',
        ]
        
        video_file = None
        for vp in video_paths:
            if vp.exists():
                video_file = vp
                break
        
        if not video_file:
            issues.append(self._create_issue(
                check_name="视频文件",
                message="HDF5 格式：未找到视频文件",
                passed=False,
                level=IssueLevel.MAJOR,
            ))
            return False
        
        try:
            cap = cv2.VideoCapture(str(video_file))
            if not cap.isOpened():
                issues.append(self._create_issue(
                    check_name="视频文件",
                    message="无法打开视频文件",
                    passed=False,
                    level=IssueLevel.MAJOR,
                ))
                return False
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            issues.append(self._create_issue(
                check_name="视频文件",
                message=f"HDF5 格式：{frame_count} 帧, {fps:.1f} FPS, {width}×{height}",
                passed=True,
            ))
            
            # 分辨率检查
            min_w = self.config.min_resolution_width
            min_h = self.config.min_resolution_height
            resolution_passed = width >= min_w and height >= min_h
            issues.append(self._create_issue(
                check_name="视频分辨率",
                message=f"{width}×{height}（最低 {min_w}×{min_h}）",
                passed=resolution_passed,
                level=IssueLevel.MAJOR,
                value=width * height,
                threshold=min_w * min_h,
            ))
            
            # 时长检查
            min_duration = 1.0
            duration_passed = duration >= min_duration
            issues.append(self._create_issue(
                check_name="视频时长",
                message=f"{duration:.2f} 秒（最少 {min_duration} 秒）",
                passed=duration_passed,
                level=IssueLevel.MAJOR,
                value=duration,
                threshold=min_duration,
            ))
            
            # FPS 检查
            min_fps = self.config.min_frame_rate
            fps_passed = fps >= min_fps
            issues.append(self._create_issue(
                check_name="视频 FPS",
                message=f"{fps:.1f} Hz（最低 {min_fps} Hz）",
                passed=fps_passed,
                level=IssueLevel.MAJOR,
                value=fps,
                threshold=min_fps,
            ))
            
            # 采样检查图像质量
            self._check_video_frames(cap, frame_count, issues)
            
            cap.release()
            return True
            
        except Exception as e:
            issues.append(self._create_issue(
                check_name="视频文件",
                message=f"读取失败: {str(e)[:50]}",
                passed=False,
                level=IssueLevel.MAJOR,
            ))
            return False
    
    def _check_video_frames(self, cap, frame_count: int, issues: List[ValidationIssue]):
        """检查视频帧质量"""
        sample_step = max(1, frame_count // 10)
        
        stats = {
            'overexposure': [],
            'underexposure': [],
            'black': [],
            'white': [],
        }
        
        for idx in range(0, frame_count, sample_step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # 转换为灰度计算亮度
            gray = np.mean(frame, axis=2) if frame.ndim == 3 else frame
            
            # 过曝 (像素值 > 250)
            stats['overexposure'].append(np.mean(gray > 250))
            # 欠曝 (像素值 < 5)
            stats['underexposure'].append(np.mean(gray < 5))
            # 全黑/全白
            stats['black'].append(np.mean(gray < 2))
            stats['white'].append(np.mean(gray > 253))
        
        if stats['overexposure']:
            avg_over = np.mean(stats['overexposure'])
            th_over = self.config.overexposure_ratio_max
            issues.append(self._create_issue(
                check_name="过曝检测",
                message=f"过曝率 {avg_over*100:.1f}%（阈值 {th_over*100:.0f}%）",
                passed=avg_over <= th_over,
                level=IssueLevel.MAJOR,
                value=avg_over,
                threshold=th_over
            ))
            
            avg_under = np.mean(stats['underexposure'])
            th_under = self.config.underexposure_ratio_max
            issues.append(self._create_issue(
                check_name="欠曝检测",
                message=f"欠曝率 {avg_under*100:.1f}%（阈值 {th_under*100:.0f}%）",
                passed=avg_under <= th_under,
                level=IssueLevel.MAJOR,
                value=avg_under,
                threshold=th_under
            ))
            
            avg_black = np.mean(stats['black'])
            avg_white = np.mean(stats['white'])
            th_abnormal = self.config.abnormal_black_ratio_max
            abnormal_passed = avg_black < th_abnormal and avg_white < th_abnormal
            issues.append(self._create_issue(
                check_name="异常图像检测",
                message=f"全黑 {avg_black*100:.1f}%, 全白 {avg_white*100:.1f}%（阈值 {th_abnormal*100:.0f}%）",
                passed=abnormal_passed,
                level=IssueLevel.MAJOR,
                value=max(avg_black, avg_white),
                threshold=th_abnormal
            ))
    
    def validate(self, episode_id: str, data_path: str) -> ValidationResult:
        """验证视觉数据质量"""
        issues: List[ValidationIssue] = []
        path = Path(data_path)
        
        # 检测是否为 HDF5 旧格式
        if self._is_hdf5_format(path):
            # 旧格式数据使用视频文件进行验证
            self._validate_hdf5_video(path, issues)
            
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
        
        # 检查视觉数据文件
        camera_files = list(path.glob('observation.images.*.parquet'))
        
        if not camera_files:
            issues.append(self._create_issue(
                check_name="相机数据文件",
                message="未找到相机数据文件",
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
            check_name="相机数据文件",
            message=f"找到 {len(camera_files)} 个相机数据文件",
            passed=True,
        ))
        
        # 验证每个相机文件
        for camera_file in camera_files:
            camera_name = camera_file.stem.replace('observation.images.', '')
            
            try:
                df = pd.read_parquet(camera_file)
                frame_count = len(df)
                
                issues.append(self._create_issue(
                    check_name=f"相机 {camera_name} 数据读取",
                    message=f"成功加载 {frame_count} 帧",
                    passed=True,
                ))
                
                # 检查是否有 parameters 列（包含分辨率信息）
                if 'parameters' in df.columns:
                    try:
                        params = df['parameters'].iloc[0] if len(df) > 0 else {}
                        if isinstance(params, dict):
                            width = params.get('width', {}).get('Integer', 0)
                            height = params.get('height', {}).get('Integer', 0)
                            
                            min_w = self.config.min_resolution_width
                            min_h = self.config.min_resolution_height
                            
                            resolution_passed = width >= min_w and height >= min_h
                            issues.append(self._create_issue(
                                check_name=f"相机 {camera_name} 分辨率",
                                message=f"{width}×{height}（最低 {min_w}×{min_h}）",
                                passed=resolution_passed,
                                level=IssueLevel.MAJOR,
                                value=width * height,
                                threshold=min_w * min_h,
                            ))
                    except Exception:
                        pass
                
                # 获取时间戳并计算时长
                timestamps = None
                duration = 0.0
                
                if 'timestamp_utc' in df.columns and frame_count > 0:
                    timestamps = df['timestamp_utc'].values
                    # 确保是纳秒 (pandas 默认) 或 转换为秒
                    if timestamps.dtype == 'datetime64[ns]':
                        timestamps = timestamps.astype('int64') / 1e9
                    
                    if len(timestamps) > 1:
                        duration = timestamps[-1] - timestamps[0]
                
                # 数据时长检查 (替代原有的帧数检查)
                min_duration = 1.0  # 至少 1 秒
                duration_passed = duration >= min_duration
                issues.append(self._create_issue(
                    check_name=f"相机 {camera_name} 数据时长",
                    message=f"{duration:.2f} 秒（最少 {min_duration} 秒）",
                    passed=duration_passed,
                    level=IssueLevel.MAJOR,
                    value=duration,
                    threshold=min_duration,
                ))

                # FPS 计算 (排除首尾1秒)
                fps = 0.0
                if timestamps is not None and len(timestamps) > 2:
                    if duration > 2.0:
                        # 排除首尾 1 秒
                        start_time = timestamps[0] + 1.0
                        end_time = timestamps[-1] - 1.0
                        mask = (timestamps >= start_time) & (timestamps <= end_time)
                        valid_timestamps = timestamps[mask]
                        
                        if len(valid_timestamps) > 1:
                            valid_duration = valid_timestamps[-1] - valid_timestamps[0]
                            if valid_duration > 0:
                                fps = (len(valid_timestamps) - 1) / valid_duration
                    elif duration > 0:
                        # 持续时间短于2秒，使用全部数据
                        fps = (frame_count - 1) / duration
                        
                    # 验证 FPS
                    min_fps = self.config.min_frame_rate
                    fps_passed = fps >= min_fps
                    issues.append(self._create_issue(
                        check_name=f"相机 {camera_name} FPS",
                        message=f"{fps:.1f} Hz（最低 {min_fps} Hz）",
                        passed=fps_passed,
                        level=IssueLevel.MAJOR,
                        value=fps,
                        threshold=min_fps,
                    ))
                
                # 图像质量采样检查
                if 'image' in df.columns:
                    self._check_image_content(df, camera_name, issues)
                
            except Exception as e:
                issues.append(self._create_issue(
                    check_name=f"相机 {camera_name} 数据读取",
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
                'camera_files': [f.name for f in camera_files],
            }
        )
    
    def _check_image_content(self, df: pd.DataFrame, camera_name: str, issues: List[ValidationIssue]):
        """检查图像内容质量
        
        标准：
        - 过曝：过饱和像素比例≤5%
        - 欠曝：欠曝像素比例≤10%
        - 异常图像：全黑/全白像素比例≤95%
        - 色彩偏移：<10%
        """
        frame_count = len(df)
        sample_step = max(1, frame_count // 10)  # 采样约10帧
        sample_indices = range(0, frame_count, sample_step)
        
        stats = {
            'overexposure': [],
            'underexposure': [],
            'black': [],
            'white': [],
            'color_shift': []
        }
        
        for idx in sample_indices:
            try:
                img_data = df['image'].iloc[idx]
                img = None
                
                # 解码图像
                if isinstance(img_data, dict) and 'bytes' in img_data:
                    img = Image.open(io.BytesIO(img_data['bytes']))
                elif isinstance(img_data, bytes):
                    img = Image.open(io.BytesIO(img_data))
                
                if img:
                    # 转换为 numpy 数组
                    img_arr = np.array(img)
                    if len(img_arr.shape) == 3:
                        # 亮度 (简化计算)
                        gray = np.mean(img_arr, axis=2)
                        
                        # 过曝 (像素值 > 250)
                        over = np.mean(gray > 250)
                        stats['overexposure'].append(over)
                        
                        # 欠曝 (像素值 < 5)
                        under = np.mean(gray < 5)
                        stats['underexposure'].append(under)
                        
                        # 全黑/全白
                        stats['black'].append(np.mean(gray < 2))
                        stats['white'].append(np.mean(gray > 253))
                        
                        # 色偏 (RGB 均值差异)
                        means = np.mean(img_arr, axis=(0, 1))
                        std_dev = np.std(means)
                        stats['color_shift'].append(std_dev / 255.0)
            except Exception:
                pass
        
        # 汇总统计并生成 issues - 使用配置中的阈值
        if stats['overexposure']:
            # 过曝检测
            avg_over = np.mean(stats['overexposure'])
            th_over = self.config.overexposure_ratio_max  # 默认 0.05
            issues.append(self._create_issue(
                check_name=f"相机 {camera_name} 过曝检测",
                message=f"过曝率 {avg_over*100:.1f}%（阈值 {th_over*100:.0f}%）",
                passed=avg_over <= th_over,
                level=IssueLevel.MAJOR,
                value=avg_over,
                threshold=th_over
            ))
            
            # 欠曝检测
            avg_under = np.mean(stats['underexposure'])
            th_under = self.config.underexposure_ratio_max  # 默认 0.10
            issues.append(self._create_issue(
                check_name=f"相机 {camera_name} 欠曝检测",
                message=f"欠曝率 {avg_under*100:.1f}%（阈值 {th_under*100:.0f}%）",
                passed=avg_under <= th_under,
                level=IssueLevel.MAJOR,
                value=avg_under,
                threshold=th_under
            ))
            
            # 异常图像检测：全黑/全白
            avg_black = np.mean(stats['black'])
            avg_white = np.mean(stats['white'])
            th_black = self.config.abnormal_black_ratio_max  # 默认 0.95
            th_white = self.config.abnormal_white_ratio_max  # 默认 0.95
            abnormal_passed = avg_black < th_black and avg_white < th_white
            issues.append(self._create_issue(
                check_name=f"相机 {camera_name} 异常图像检测",
                message=f"全黑 {avg_black*100:.1f}%, 全白 {avg_white*100:.1f}%（阈值 {th_black*100:.0f}%）",
                passed=abnormal_passed,
                level=IssueLevel.MAJOR,
                value=max(avg_black, avg_white),
                threshold=th_black
            ))
            
            # 色彩偏移检测
            avg_shift = np.mean(stats['color_shift'])
            th_shift = self.config.color_shift_max  # 默认 0.10
            issues.append(self._create_issue(
                check_name=f"相机 {camera_name} 色偏检测",
                message=f"色偏度 {avg_shift*100:.1f}%（阈值 {th_shift*100:.0f}%）",
                passed=avg_shift <= th_shift,
                level=IssueLevel.MAJOR,
                value=avg_shift,
                threshold=th_shift
            ))
