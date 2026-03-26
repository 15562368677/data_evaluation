"""
质检服务

组合所有验证器，执行完整的 Episode 质检流程
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import numpy as np
import psycopg2

from arts_analysis.config import DB_CONFIG, REVIEW_DB_CONFIG

from .validators import (
    ValidationResult,
    ValidationIssue,
    ValidatorConfig,
    IssueLevel,
    MetadataValidator,
    TimingValidator,
    ActionValidator,
    VisualValidator,
    DepthValidator,
    EETrajectoryValidator,
)


@dataclass
class AcceptanceResult:
    """质检结果"""
    episode_id: str
    passed: bool
    overall_score: float
    category_results: Dict[str, ValidationResult]  # 按分类存储结果
    all_issues: List[ValidationIssue] = field(default_factory=list)
    
    @property
    def passed_count(self) -> int:
        return sum(1 for i in self.all_issues if i.passed)
    
    @property
    def failed_count(self) -> int:
        return sum(1 for i in self.all_issues if not i.passed)
    
    @property
    def category_summary(self) -> Dict[str, Dict[str, Any]]:
        """按分类统计"""
        summary = {}
        for cat, result in self.category_results.items():
            summary[cat] = {
                'passed': result.passed,
                'score': result.score,
                'passed_count': result.passed_count,
                'failed_count': result.failed_count,
            }
        return summary
    
    def get_bar_chart_data(self) -> Dict[str, List[Dict]]:
        """获取柱状图数据（按分类分组）"""
        chart_data = {}
        for issue in self.all_issues:
            if issue.category not in chart_data:
                chart_data[issue.category] = []
            chart_data[issue.category].append({
                'check_name': issue.check_name,
                'passed': issue.passed,
                'value': issue.value,
                'threshold': issue.threshold,
                'level': issue.level.value,
            })
        return chart_data
    
    def to_dict(self) -> Dict[str, Any]:
        # 构建详细结果字典
        category_details = {}
        for cat, res in self.category_results.items():
            category_details[cat] = {
                'passed': res.passed,
                'score': res.score,
                'issues': [i.to_dict() for i in res.issues],
                'details': res.details  # 包含详细信息（如抓取坐标）
            }

        return {
            'episode_id': self.episode_id,
            'passed': self.passed,
            'overall_score': self.overall_score,
            'passed_count': self.passed_count,
            'failed_count': self.failed_count,
            'category_summary': self.category_summary,
            'category_details': category_details,
            'all_issues': [i.to_dict() for i in self.all_issues],
            'bar_chart_data': self.get_bar_chart_data(),
        }


class NumpyEncoder(json.JSONEncoder):
    """处理 Numpy 类型的 JSON 编码器"""
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class AcceptanceService:
    """质检服务"""
    
    def __init__(self, config: Optional[ValidatorConfig] = None):
        self.config = config or ValidatorConfig()
        
        # 初始化所有验证器
        self.validators = [
            MetadataValidator(self.config),
            TimingValidator(self.config),
            ActionValidator(self.config),
            VisualValidator(self.config),
            DepthValidator(self.config),
            EETrajectoryValidator(self.config),  # EE 轨迹验证器 (已启用)
        ]
    
    def validate_episode(self, episode_id: str, data_path: Optional[str] = None, check_exists: bool = False, reference_tube: Optional[Any] = None) -> AcceptanceResult:
        """
        质检单个 Episode
        
        Args:
            episode_id: Episode ID
            data_path: 数据目录路径，默认为 data_cache/{episode_id}
            check_exists: 是否检查已存在的结果（避免重复质检）
            reference_tube: (可选) HighResReferenceTube 实例，用于轨迹质量检测
        
        Returns:
            AcceptanceResult
        """
        # 如果开启检查，先查询数据库（使用 ValidationStateManager + 布隆过滤器）
        if check_exists:
            from arts_analysis.services.data_validation.validation_state_manager import get_validation_manager
            manager = get_validation_manager()
            
            # 1. 先查布隆过滤器 (O(1))
            if manager.is_validated(episode_id):
                # 2. 如果布隆过滤器说可能存在，再查数据库确认
                if manager.is_validated_confirmed(episode_id):
                    print(f"Episode {episode_id} 已质检（布隆过滤器+DB确认），跳过。")
                    return None
            
            # 如果布隆过滤器说不存在，那就一定不存在，直接继续

        if data_path is None:
            data_path = f"data_cache/{episode_id}"
        
        category_results = {}
        all_issues = []
        
        # 运行所有验证器
        # 1. 运行 MetadataValidator (获取基础信息)
        metadata_validator = self.validators[0]
        try:
            metadata_result = metadata_validator.validate(episode_id, data_path)
            category_results[metadata_validator.category] = metadata_result
            all_issues.extend(metadata_result.issues)
            
            # 2. 检查时长是否在允许范围内
            duration = metadata_result.details.get('duration')
            if duration is not None:
                is_outlier = False
                msg = ""
                threshold = 0.0
                
                if self.config.duration_min is not None and duration < self.config.duration_min:
                    is_outlier = True
                    msg = f"时长 {duration:.2f}s 小于阈值 {self.config.duration_min:.2f}s，跳过后续质检"
                    threshold = self.config.duration_min
                elif self.config.duration_max is not None and duration > self.config.duration_max:
                    is_outlier = True
                    msg = f"时长 {duration:.2f}s 大于阈值 {self.config.duration_max:.2f}s，跳过后续质检"
                    threshold = self.config.duration_max
                
                if is_outlier:
                    all_issues.append(ValidationIssue(
                        level=IssueLevel.MAJOR,
                        check_name="时长异常过滤",
                        category="时长过滤",
                        message=msg,
                        value=duration,
                        threshold=threshold,
                        passed=False
                    ))
                    return AcceptanceResult(
                        episode_id=episode_id,
                        passed=False,
                        overall_score=0.0,
                        category_results=category_results,
                        all_issues=all_issues
                    )

        except Exception as e:
            # MetadataValidator 失败
            error_issue = ValidationIssue(
                level=IssueLevel.CRITICAL,
                check_name=f"{metadata_validator.name} 执行",
                category=metadata_validator.category,
                message=f"验证器执行失败: {str(e)[:50]}",
                passed=False,
            )
            all_issues.append(error_issue)
        
        # 3. 运行其他验证器
        for validator in self.validators[1:]:
            try:
                result = validator.validate(episode_id, data_path)
                category_results[validator.category] = result
                all_issues.extend(result.issues)
            except Exception as e:
                # 验证器执行失败
                error_issue = ValidationIssue(
                    level=IssueLevel.CRITICAL,
                    check_name=f"{validator.name} 执行",
                    category=validator.category,
                    message=f"验证器执行失败: {str(e)[:50]}",
                    passed=False,
                )
                error_result = ValidationResult(
                    passed=False,
                    score=0,
                    issues=[error_issue],
                )
                category_results[validator.category] = error_result
                all_issues.append(error_issue)
        
        # 集成 AnomalyDetector 进行轨迹质量检测（可按配置关闭跨 Episode 对比）
        cross_enabled = getattr(self.config, 'enable_cross_episode_checks', True)
        if reference_tube and cross_enabled:
            try:
                from arts_analysis.services.duration_detector.miner_logic import AnomalyDetector
                from arts_analysis.services.trajectory_service.ee_state_change_time import to_py
                import pandas as pd
                import os
                
                state_file = os.path.join(data_path, "observation.state.parquet")
                if os.path.exists(state_file):
                    print(f"[DEBUG] 开始轨迹检测: {episode_id[:12]}... state_file存在")
                    
                    # 直接从 data_path 读取 parquet（不使用 fetcher，避免重新下载）
                    df = pd.read_parquet(state_file)
                    
                    # 降采样
                    sample_rate = 5
                    if len(df) > sample_rate:
                        df = df.iloc[::sample_rate]
                    
                    # 解析 observation.state 为 DataFrame（参考 data_fetcher.load_state_trajectory 逻辑）
                    records = []
                    for idx, row in df.iterrows():
                        record = {
                            'frame_idx': idx,
                        }
                        # 添加 timestamp
                        if 'timestamp_utc' in df.columns:
                            record['timestamp'] = row['timestamp_utc']
                        else:
                            record['timestamp'] = idx / 60.0  # 假设 60Hz
                        
                        entry = row.get("observation.state", None)
                        py_entry = to_py(entry)
                        
                        if isinstance(py_entry, (list, tuple)):
                            for item in py_entry:
                                if isinstance(item, dict) and "name" in item:
                                    record[item["name"]] = float(item.get("value", 0))
                        records.append(record)
                    
                    current_traj = pd.DataFrame(records)
                    
                    if not current_traj.empty and len(current_traj.columns) > 3:
                        detector = AnomalyDetector(reference_tube)
                        quality_score, issues = detector.evaluate(current_traj)
                        
                        # 调试日志
                        print(f"[DEBUG] 轨迹检测完成: episode={episode_id[:12]}... 分数={quality_score:.1f}, 问题数={len(issues)}")
                        if issues:
                            for i, issue in enumerate(issues[:3]):
                                print(f"[DEBUG]   问题 {i+1}: {issue[:80]}...")
                        
                        # 将检测结果转换为 ValidationIssue
                        traj_issues = []
                        for issue_text in issues:
                            # 提取异常类型作为检查项名称，例如 "轨迹异常-Stall"
                            label = issue_text.split(":")[0] if ":" in issue_text else "通用"
                            traj_issues.append(ValidationIssue(
                                level=IssueLevel.MAJOR, # 轨迹异常通常比较严重
                                check_name=f"轨迹异常-{label}",
                                category="动作数据",
                                message=issue_text,
                                passed=False,
                                value=None,  # Set to None to show message in UI
                                threshold=60.0 # 假设阈值
                            ))
                        
                        if not issues:
                            # 通过
                            traj_issues.append(ValidationIssue(
                                level=IssueLevel.INFO,
                                check_name="轨迹质量检测",
                                category="动作数据",
                                message="轨迹符合标准动作模式",
                                passed=True,
                                value=quality_score,
                                threshold=60.0
                            ))
                            
                        # 合并结果
                        if "动作数据" in category_results:
                            # 更新现有结果
                            existing_res = category_results["动作数据"]
                            existing_res.issues.extend(traj_issues)
                            # 重新计算分数 (简单平均)
                            passed = sum(1 for i in existing_res.issues if i.passed)
                            existing_res.score = passed / len(existing_res.issues) * 100
                            existing_res.passed = all(i.passed for i in existing_res.issues if i.level in (IssueLevel.CRITICAL, IssueLevel.MAJOR))
                        else:
                            # 新建结果
                            passed = not bool(issues)
                            category_results["动作数据"] = ValidationResult(
                                passed=passed,
                                score=quality_score,
                                issues=traj_issues
                            )
                        
                        all_issues.extend(traj_issues)
                        
            except Exception as e:
                print(f"轨迹质量检测失败: {e}")
                import traceback
                traceback.print_exc()

        # 计算整体得分
        if all_issues:
            passed_count = sum(1 for i in all_issues if i.passed)
            overall_score = passed_count / len(all_issues) * 100
        else:
            overall_score = 0
        
        # 整体是否通过（所有 CRITICAL 和 MAJOR 问题必须通过）
        critical_major_passed = all(
            i.passed for i in all_issues 
            if i.level in (IssueLevel.CRITICAL, IssueLevel.MAJOR)
        )
        
        return AcceptanceResult(
            episode_id=episode_id,
            passed=critical_major_passed,
            overall_score=round(overall_score, 1),
            category_results=category_results,
            all_issues=all_issues,
        )
    
    def batch_validate(
        self, 
        episode_ids: List[str], 
        progress_callback: Optional[callable] = None
    ) -> List[AcceptanceResult]:
        """
        批量质检
        
        Args:
            episode_ids: Episode ID 列表
            progress_callback: 进度回调 (current, total, episode_id)
        
        Returns:
            AcceptanceResult 列表
        """
        results = []
        total = len(episode_ids)
        
        for i, episode_id in enumerate(episode_ids):
            if progress_callback:
                progress_callback(i + 1, total, episode_id)
            
            # 开启重复检查
            result = self.validate_episode(episode_id, check_exists=True)
            if result:
                results.append(result)
        
        return results
    
    def get_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return self.config.to_dict()
    
    def update_config(self, **kwargs) -> None:
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def _get_db_connection(self):
        """获取数据库连接"""
        return psycopg2.connect(**REVIEW_DB_CONFIG)
    
    def _get_duration_classification(self, episode_id: str) -> Optional[tuple]:
        """
        获取 episode 的时长分类信息
        
        Args:
            episode_id: Episode ID
            
        Returns:
            (duration_info, metadata)
            - duration_info: 字典包含 duration, category, label, task_stats 等信息
            - metadata: 字典包含 task/pilot/machine/创建时间等元数据
        """
        try:
            from arts_analysis.services.duration_detector.duration_analyzer import get_duration_analyzer
            import psycopg2
            
            # 1. 从 episodes 表获取时长和任务/元数据
            conn = psycopg2.connect(**DB_CONFIG)
            cursor = conn.cursor()
            cursor.execute(
                """SELECT t.id, t.task_template_id, t.task_template, t.descriptions,
                          e.trajectory_duration, e.pilot, e.machine_id, e.create_time, e.cloud_path
                   FROM episodes e
                   LEFT JOIN tasks t ON e.task_id = t.id
                   WHERE e.uniq_id = %s""",
                (episode_id,)
            )
            row = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if not row:
                return None, None
            
            task_instance_id, task_template_id, task_template, descriptions, duration, pilot, machine_id, create_time, cloud_path = row
            task_name = task_template or ""
            if descriptions:
                try:
                    if isinstance(descriptions, str):
                        descriptions = json.loads(descriptions)
                    if isinstance(descriptions, dict) and 'zh' in descriptions and descriptions['zh']:
                        full_desc = descriptions['zh'][0]
                        task_name = full_desc.split('；')[0].split('。')[0][:20]
                except Exception:
                    pass

            metadata = {
                'task_instance_id': task_instance_id,
                'task_template_id': task_template_id,
                'task_name': task_name,
                'pilot': pilot,
                'machine_id': machine_id,
                'episode_create_time': create_time,
                'cloud_path': cloud_path
            }

            if not task_template_id or duration is None:
                return None, metadata
            
            # 2. 获取任务统计并分类
            analyzer = get_duration_analyzer()
            task_stats = analyzer.get_task_stats(task_template_id)
            
            if not task_stats:
                # 没有预计算的统计，返回基本信息
                return {
                    'duration': float(duration),
                    'category': 'unknown',
                    'label': '⚪ 未统计',
                    'task_id': task_template_id,
                }, metadata
            
            # 3. 分类
            category = analyzer.classify_duration(float(duration), task_stats)
            label = analyzer.get_classification_label(category)
            
            return {
                'duration': float(duration),
                'category': category,
                'label': label,
                'task_id': task_template_id,
                'is_bimodal': task_stats.get('is_bimodal', False),
                'p10': task_stats.get('p10'),
                'p50': task_stats.get('p50'),
                'p90': task_stats.get('p90'),
                'valley_point': task_stats.get('valley_point'),
            }, metadata
            
        except Exception as e:
            print(f"获取时长分类失败 ({episode_id}): {e}")
            return None, None
    
    def save_result(self, result: 'AcceptanceResult', config_id: int = 1) -> Optional[int]:
        """
        保存质检结果到数据库 (validation_results 表)
        
        Args:
            result: AcceptanceResult 对象
            config_id: 配置 ID（默认1为default，暂未使用）
            
        Returns:
            插入的 result_id，失败返回 None
        """
        import json
        from datetime import datetime
        
        try:
            from arts_analysis.services.database.review_db import get_review_db
            
            db = get_review_db()
            
            # 确保表存在
            db.create_validation_results_table()
            
            # 获取各维度得分
            scores = {cat: r.score for cat, r in result.category_results.items()}
            
            # 序列化 issues 为 JSON
            issues_list = []
            for issue in result.all_issues:
                issue_dict = issue.to_dict()
                # 处理 numpy 类型
                if hasattr(issue_dict.get('value'), 'item'):
                    issue_dict['value'] = issue_dict['value'].item()
                if hasattr(issue_dict.get('threshold'), 'item'):
                    issue_dict['threshold'] = issue_dict['threshold'].item()
                issues_list.append(issue_dict)
            
            issues_json = json.dumps(issues_list, ensure_ascii=False, cls=NumpyEncoder)
            
            # 添加时长分类信息
            result_dict = result.to_dict()
            duration_info, metadata = self._get_duration_classification(result.episode_id)
            if duration_info:
                result_dict['duration_info'] = duration_info
            
            result_json = json.dumps(result_dict, ensure_ascii=False, default=str, cls=NumpyEncoder)
            config_json = json.dumps(self.config.to_dict(), ensure_ascii=False, cls=NumpyEncoder)
            
            # 构建保存数据字典
            save_data = {
                'episode_id': result.episode_id,
                'task_template_id': duration_info.get('task_id') if duration_info else None,
                'visual_score': float(scores.get('视觉质量', 0)),
                'action_score': float(scores.get('动作数据', 0)),
                'timing_score': float(scores.get('时间同步', 0)),
                'depth_score': float(scores.get('深度数据', 0)),
                'language_score': None,  # 暂无语言数据
                'overall_score': float(result.overall_score),
                'is_valid': bool(result.passed),
                'issues': issues_json,
                'has_depth_data': '深度数据' in scores,
                'config_json': config_json,
                'result_json': result_json,
                'validated_by': 'auto',
                'validated_at': datetime.now(),
            }

            # 补充元数据（用于导出/筛选加速）
            if metadata:
                for key, value in metadata.items():
                    if value is not None:
                        save_data[key] = value
            
            # 保存到 validation_results 表
            result_id = db.insert_validation_result(save_data)
            
            # 更新质检状态（Redis + 布隆过滤器）
            try:
                from arts_analysis.services.data_validation.validation_state_manager import get_validation_manager
                manager = get_validation_manager()
                manager.mark_as_validated(result.episode_id)
            except Exception as e:
                print(f"更新质检状态失败: {e}")
            
            return result_id
            
        except Exception as e:
            print(f"保存质检结果失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def save_result_v2(self, result: 'AcceptanceResult') -> Optional[int]:
        """
        新版结构化入库（qc_results / qc_categories / qc_issues）

        Returns:
            qc_result_id
        """
        import json
        from datetime import datetime

        try:
            from arts_analysis.services.database.review_db import get_review_db
            db = get_review_db()

            db.create_qc_tables()

            with db.get_connection() as conn:
                with conn.cursor() as cursor:

                    # ========================
                    # 1️⃣ qc_results
                    # ========================
                    raw_dict = result.to_dict()

                    has_major_issue = any(
                        (not i.passed and i.level.value == "major")
                        for i in result.all_issues
                    )

                    qc_result_data = {
                        "episode_id": result.episode_id,
                        "passed": result.passed,
                        "overall_score": float(result.overall_score),
                        "passed_count": result.passed_count,
                        "failed_count": result.failed_count,
                        "has_major_issue": has_major_issue,
                        "has_blocker_issue": False,
                        "category_summary": json.dumps(result.category_summary, ensure_ascii=False),
                        "raw_json": r'{}',
                        # "raw_json": json.dumps(raw_dict, ensure_ascii=False),
                    }

                    insert_qc_result_sql = """
                    INSERT INTO qc_results (
                        episode_id, passed, overall_score,
                        passed_count, failed_count,
                        has_major_issue, has_blocker_issue,
                        category_summary, raw_json, updated_at
                    )
                    VALUES (
                        %(episode_id)s, %(passed)s, %(overall_score)s,
                        %(passed_count)s, %(failed_count)s,
                        %(has_major_issue)s, %(has_blocker_issue)s,
                        %(category_summary)s::jsonb,
                        %(raw_json)s::jsonb,
                        NOW()
                    )
                    ON CONFLICT (episode_id) DO UPDATE SET
                        passed = EXCLUDED.passed,
                        overall_score = EXCLUDED.overall_score,
                        passed_count = EXCLUDED.passed_count,
                        failed_count = EXCLUDED.failed_count,
                        has_major_issue = EXCLUDED.has_major_issue,
                        has_blocker_issue = EXCLUDED.has_blocker_issue,
                        category_summary = EXCLUDED.category_summary,
                        raw_json = EXCLUDED.raw_json,
                        updated_at = NOW()
                    RETURNING id
                    """

                    cursor.execute(insert_qc_result_sql, qc_result_data)
                    qc_result_id = cursor.fetchone()[0]

                    # ========================
                    # 2️⃣ 清理旧数据（关键）
                    # ========================
                    cursor.execute("DELETE FROM qc_categories WHERE qc_result_id = %s", (qc_result_id,))
                    cursor.execute("DELETE FROM qc_issues WHERE qc_result_id = %s", (qc_result_id,))

                    # ========================
                    # 3️⃣ qc_categories + qc_issues
                    # ========================
                    category_id_map = {}

                    for cat, cat_result in result.category_results.items():

                        # 插入 category
                        category_data = {
                            "qc_result_id": qc_result_id,
                            "category": cat,
                            "passed": cat_result.passed,
                            "score": float(cat_result.score),
                            "passed_count": cat_result.passed_count,
                            "failed_count": cat_result.failed_count,
                            "details": json.dumps(cat_result.details or {}, ensure_ascii=False),
                        }

                        insert_category_sql = """
                        INSERT INTO qc_categories (
                            qc_result_id, category, passed, score,
                            passed_count, failed_count, details
                        )
                        VALUES (
                            %(qc_result_id)s, %(category)s, %(passed)s, %(score)s,
                            %(passed_count)s, %(failed_count)s,
                            %(details)s::jsonb
                        )
                        RETURNING id
                        """

                        cursor.execute(insert_category_sql, category_data)
                        category_id = cursor.fetchone()[0]
                        category_id_map[cat] = category_id

                        # ========================
                        # 插入 issues（批量）
                        # ========================
                        issue_rows = []

                        for issue in cat_result.issues:
                            value = issue.value
                            threshold = issue.threshold

                            if hasattr(value, "item"):
                                value = value.item()
                            if hasattr(threshold, "item"):
                                threshold = threshold.item()

                            issue_rows.append((
                                qc_result_id,
                                category_id,
                                cat,
                                issue.check_name,
                                issue.level.value,
                                bool(issue.passed),
                                value,
                                threshold,
                                issue.message,
                            ))

                        insert_issue_sql = """
                        INSERT INTO qc_issues (
                            qc_result_id, category_id, category,
                            check_name, level, passed,
                            value, threshold, message
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """

                        cursor.executemany(insert_issue_sql, issue_rows)

                    conn.commit()

                    return qc_result_id

        except Exception as e:
            print(f"[save_result_v2] 保存失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_history(self, episode_id: Optional[str] = None, limit: int = 100) -> List[Dict]:
        """
        获取历史质检记录
        
        Args:
            episode_id: 可选，指定 Episode
            limit: 返回数量限制
            
        Returns:
            质检记录列表
        """
        try:
            conn = self._get_db_connection()
            cursor = conn.cursor()
            
            if episode_id:
                cursor.execute("""
                    SELECT episode_id, passed, overall_score, passed_count, failed_count, 
                           visual_score, action_score, timing_score, metadata_score, created_at
                    FROM acceptance_results 
                    WHERE episode_id = %s 
                    ORDER BY created_at DESC LIMIT %s
                """, (episode_id, limit))
            else:
                cursor.execute("""
                    SELECT episode_id, passed, overall_score, passed_count, failed_count, 
                           visual_score, action_score, timing_score, metadata_score, created_at
                    FROM acceptance_results 
                    ORDER BY created_at DESC LIMIT %s
                """, (limit,))
            
            rows = cursor.fetchall()
            cursor.close()
            conn.close()
            
            return [
                {
                    'episode_id': r[0], 'passed': r[1], 'overall_score': r[2],
                    'passed_count': r[3], 'failed_count': r[4],
                    'visual_score': r[5], 'action_score': r[6], 
                    'timing_score': r[7], 'metadata_score': r[8],
                    'created_at': r[9].isoformat() if r[9] else None
                }
                for r in rows
            ]
            
        except Exception as e:
            print(f"获取质检历史失败: {e}")
            return []
