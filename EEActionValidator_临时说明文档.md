# EEActionValidator 临时说明文档

下面这份文档基于 `src/acceptance_service/validators/acceptance/ee_action.py`、`src/acceptance_service/validators/core/base.py` 和调用入口 `src/acceptance_service/acceptance_service.py`。

## 1. 文件定位与整体职责

`EEActionValidator` 是一个“末端执行器抓取检测验证器”。它的职责不是做整套动作质量检查，而是专门判断：

- 左右手有没有检测到抓取片段
- 每只手抓取了多少次
- 当前 episode 的抓取次数相对“该 episode 内检测到的最小正次数”是偏少、正常还是偏多
- 最终给出 `CRITICAL / MAJOR / MINOR / INFO` 等级和一个分数

它对外的主要公开入口是：

- `validate(episode_id)`：单条 episode 验证

外部主要通过这里调用它：

- `src/acceptance_service/acceptance_service.py:40`
- `src/acceptance_service/acceptance_service.py:73`
- `src/acceptance_service/acceptance_service.py:93`
- `src/workers/pnp_worker.py:222`

## 2. `ee_action.py` 里一共有几个函数

这个文件里目前一共 33 个“可调用单元”：

- 模块级函数 16 个
- `EEActionValidator` 类里的方法 17 个

从当前代码看，没有明显“完全没被调用”的死函数；大多数 helper 都在文件内部调用，少数作为公共导出被外部用到。此前仅用于备用分支、但在当前 `validate()` 实际路径中不会执行到的差分判断 helper 已删除。

## 3. 模块级函数说明

### 3.1 `_normalize_segments`

位置：`ee_action.py:45`

功能：把原始片段数据统一转成 `[[start_sec, end_sec], ...]`。支持输入字符串 JSON 或列表。

是否调用：有，供 `_estimate_duration_from_segments` 内部调用。

### 3.2 `_calc_total_duration`

位置：`ee_action.py:74`

功能：计算多个片段总时长。

是否调用：有，供 `_build_hand_details` 调用。

### 3.3 `_filter_short_segments`

位置：`ee_action.py:78`

功能：过滤掉太短的抓取片段。

是否调用：有，供 `_build_hand_details` 调用。

### 3.4 `_calc_axis_ratios`

位置：`ee_action.py:97`

功能：把每个片段中心点映射到 episode 全时长中的相对位置 `[0,1]`。

是否调用：有，供 `_build_hand_details` 调用。

### 3.5 `_estimate_duration_from_segments`

位置：`ee_action.py:112`

功能：如果拿不到 episode duration，就从左右手片段里估算总时长。

是否调用：有，供 `_resolve_episode_duration` 调用。

### 3.6 `_resolve_episode_duration_from_row`

位置：`ee_action.py:120`

功能：从数据库查询结果里拿 episode 时长，优先 `trajectory_duration`，否则用 `trajectory_end - trajectory_start`。

是否调用：有，供 `_load_episode_context` 调用。

### 3.7 `_minimum_detected_count`

位置：`ee_action.py:142`

功能：取左右手中“正数抓取次数”的最小值，忽略 0。

是否调用：有，供 `build_task_context` 调用。

说明：这个值是后续 `MAJOR/MINOR` 判定的基线。

### 3.8 `_to_duration_seconds`

位置：`ee_action.py:153`

功能：把两个时间点或数值差转成秒。

是否调用：有，供 `_frame_segments_to_time_segments` 和 `_resolve_episode_duration` 调用。

### 3.9 `build_task_en`

位置：`ee_action.py:165`

功能：从任务描述 `descriptions` 里抽英文说明。

是否调用：有。

内部调用：`_load_episode_context`。

外部调用：作为公共函数被导出。

### 3.10 `extract_minimum_grasp_counts`

位置：`ee_action.py:187`

功能：从英文任务描述里解析左右手最少抓取次数。

是否调用：有。

内部调用：`build_task_context`。

外部调用：作为公共函数被导出。

说明：当前它解析了 `task_required_count`，但最终等级判定并没有直接用这个值。

### 3.11 `_build_compact_hand_details`

位置：`ee_action.py:208`

功能：压缩单手结果详情。

是否调用：有，供 `_build_compact_ee_details` 调用。

### 3.12 `_build_compact_ee_details`

位置：`ee_action.py:227`

功能：生成最终 `details` 输出结构。

是否调用：有，供 `_finalize_episode_result` 调用。

### 3.13 `calculate_closure_degree`

位置：`ee_action.py:261`

功能：根据手指关节角度和方向系数，算单帧“闭合度”。

是否调用：有，供 `calculate_closure_metrics_from_dataframe` 调用。

### 3.14 `calculate_closure_velocity`

位置：`ee_action.py:287`

功能：计算闭合度变化速度。

是否调用：有，供 `calculate_closure_metrics_from_dataframe` 调用。

### 3.15 `calculate_closure_metrics_from_dataframe`

位置：`ee_action.py:316`

功能：把整段关节时序转换成 `closure_degree + closure_velocity` DataFrame。

是否调用：有，供 `_detect_hand_segments` 调用。

### 3.16 `check_joint_diff_with_slope`

位置：`ee_action.py:339`

功能：判断当前帧是否有足够多关节满足“动作差分阈值”，并且趋势足够平稳。

是否调用：有，供 `detect_pick_segments` 调用。

### 3.17 `check_sufficient_joint_differences`

位置：`ee_action.py:402`

功能：简化版差分判断，只判断数量是否达标。

是否调用：有，供 `detect_pick_segments` 的 fallback 分支调用。

### 3.18 `count_joints_satisfying_diff`

位置：`ee_action.py:417`

功能：统计某一帧有多少关节满足差分阈值。

是否调用：有，供 `check_sufficient_joint_differences` 和 `detect_pick_segments` 调用。

## 4. `EEActionValidator` 类的方法说明

### 4.1 `name`

位置：`ee_action.py:450`

功能：返回验证器名称。

是否调用：有，多个结果结构里都用到。

### 4.2 `category`

位置：`ee_action.py:454`

功能：返回分类 `"末端轨迹"`。

是否调用：有。

### 4.3 `_load_episode_context`

功能：查 DB，拿 episode 基本信息、任务描述和 rgb 文件路径。

是否调用：有，供 `_load_validation_data`。

### 4.4 `_normalize_timestamp_df`

功能：把 `timestamp_utc` 统一转成 datetime。

是否调用：有，供 `_load_joint_data_as_dfs`。

### 4.5 `_load_joint_data_as_dfs`

功能：从 joint 数据文件里构造 `state_df` 和 `action_df`。

是否调用：有，供 `_load_validation_data`。

### 4.6 `_load_validation_data`

功能：整合 episode context 和 joint DataFrame。

是否调用：有，供 `_detect_episode_result`。

### 4.7 `_build_hand_detection_config`

功能：把 `ValidatorConfig` 中的阈值和单手关节配置拼成检测配置。

是否调用：有，供 `_detect_hand_segments`。

### 4.8 `detect_pick_segments`

功能：核心检测算法。按帧扫描，找抓取开始和放下结束，输出 `(start_frame, end_frame)` 列表。

是否调用：有，供 `_detect_hand_segments`。

说明：这是核心算法之一。

### 4.9 `_detect_hand_segments`

功能：对单只手完成闭合度计算、state/action 对齐、关节差分构造、抓取片段检测。

是否调用：有，供 `_detect_episode_result`。

### 4.10 `_frame_segments_to_time_segments`

功能：把帧区间转成秒区间。

是否调用：有，供 `_detect_hand_segments`。

### 4.11 `_build_hand_details`

功能：给单手片段补充统计信息：次数、时长占比、轴向位置分数。

是否调用：有，供 `_detect_episode_result`。

### 4.12 `build_task_context`

功能：根据任务描述和检测结果，建立后续判级所需上下文。

是否调用：有，供 `validate` 和 `validate_batch`。

说明：这里生成了 `minimum_detected_grasps` 和 `task_required_grasps`。

### 4.13 `_evaluate_hand`

功能：对单手结果打等级。

是否调用：有，供 `_finalize_episode_result`。

说明：这里决定单手是 `INFO / MAJOR / MINOR`。

### 4.14 `_detect_episode_result`

功能：先做“纯检测”，不下最终结论；返回中间版 `ValidationResult`。

是否调用：有，供 `validate` 和 `validate_batch`。

### 4.15 `_finalize_episode_result`

功能：把中间检测结果转成最终 QC 结果。

是否调用：有，供 `validate` 和 `validate_batch`。

说明：这里决定最终总等级和总分。

### 4.16 `validate`

功能：单条 episode 的正式入口。

是否调用：有，外部在 `AcceptanceService.validate_episode()` 和 `pnp_worker.py` 中调用。

说明：这是最主要的公共 API。

## 5. `EEActionValidator` 的结构关系

继承关系如下：

`BaseValidator`

-> `EEActionValidator`

相关基础类型还包括：

- `IssueLevel`
- `ValidationIssue`
- `ValidationResult`
- `ValidatorConfig`

都定义在 `src/acceptance_service/validators/core/base.py`。

## 6. 基类 `BaseValidator` 的属性和方法

位置：`base.py` 中 `BaseValidator` 类定义部分。

`BaseValidator` 提供的是“验证器统一接口”：

- `self.config`
  - 在 `__init__` 里初始化
  - 类型是 `ValidatorConfig`
  - 所有阈值都从这里读取

抽象属性 / 抽象方法：

- `name`
  - 验证器名称
- `category`
  - 验证器分类
- `validate(episode_id, data=None) -> ValidationResult`
  - 所有子类都必须实现的主入口

通用辅助方法：

- `_create_issue(...) -> ValidationIssue`
  - 用统一格式构建 issue 对象
  - `EEActionValidator` 在 `_finalize_episode_result` 里用它创建最终 issue

## 7. `ValidatorConfig` 在这个验证器里真正用到的关键字段

虽然 `ValidatorConfig` 很大，但 `EEActionValidator` 主要依赖的是 PnP 检测相关参数：

- `pick_closure_threshold`
- `pick_start_offset`
- `place_closure_threshold`
- `place_velocity_threshold`
- `place_velocity_lookback`
- `place_velocity_lookahead`
- `place_diff_lookahead`
- `place_end_offset`
- `min_segment_duration_seconds`
- `negative_diff_threshold`
- `positive_diff_threshold`
- `min_joints_for_diff`
- `slope_threshold`
- `slope_lookahead`
- `hand_config`

其中 `hand_config` 包含：

- `right.finger_joints`
- `right.joint_direction_coefficients`
- `left.finger_joints`
- `left.joint_direction_coefficients`

## 8. `IssueLevel` 等级定义

位置：`base.py:11`

共有 4 级：

- `CRITICAL = "critical"`
- `MAJOR = "major"`
- `MINOR = "minor"`
- `INFO = "info"`

在 `ee_action.py` 里优先级是：

- `CRITICAL` 最高
- `MAJOR`
- `MINOR`
- `INFO` 最低

对应消息：

- `CRITICAL`: 左右手均未检测到抓取片段，任务失败
- `MAJOR`: 抓取次数小于识别到的最小抓取次数，判定为抓取异常
- `MINOR`: 抓取次数大于识别到的最小抓取次数 3 次及以上，判定为抓取行为次优
- `INFO`: 抓取行为正常

对应分数：

- `CRITICAL -> 0.0`
- `MAJOR -> 50.0`
- `MINOR -> 80.0`
- `INFO -> 100.0`

## 9. `validate()` 的核心流程

`validate()` 执行链路是：

1. `validate(episode_id)`
2. `_detect_episode_result(episode_id)`
3. `_load_validation_data()`
4. `_load_episode_context()` + `_load_joint_data_as_dfs()`
5. 对左右手分别 `_detect_hand_segments()`
6. `detect_pick_segments()` 找抓取区间
7. `_build_hand_details()` 生成每只手统计信息
8. `build_task_context()` 建立判级上下文
9. `_finalize_episode_result()` 输出最终 `ValidationResult`

可以概括成两阶段：

- 第一阶段：检测抓取片段
- 第二阶段：根据抓取次数做等级判定并包装结果

## 10. `validate()` 输出结果 `ValidationResult` 的结构

`ValidationResult` 字段有：

- `passed: bool`
- `score: Optional[float]`
- `issues: List[ValidationIssue]`
- `details: Dict[str, Any]`

另外还有派生属性：

- `passed_count`
- `failed_count`
- `critical_issues`
- `major_issues`

`to_dict()` 输出格式是：

```python
{
    "passed": self.passed,
    "score": self.score,
    "passed_count": self.passed_count,
    "failed_count": self.failed_count,
    "issues": [issue.to_dict() for issue in self.issues],
    "category_summary": self.get_category_summary(),
    "details": self.details,
}
```

## 11. `EEActionValidator.validate()` 最终返回的 `issues` 结构

这个验证器最终只塞 1 个 issue。

这个 issue 的结构如下：

```python
{
    "level": "critical" | "major" | "minor" | "info",
    "check_name": "抓取检测",
    "category": "末端轨迹",
    "message": "...",
    "value": float | None,
    "threshold": float | None,
    "passed": bool
}
```

几个关键点：

- `CRITICAL` 时：
  - `passed = False`
  - `value = 0.0`
- `MAJOR` 时：
  - `passed = True`
  - `value = 触发 major 的手中最小 count`
  - `threshold = 对应 minimum_detected_count`
- `MINOR` 时：
  - `passed = True`
  - `value = 触发 minor 的手中最大 count`
  - `threshold = minimum_detected_count + 3`
- `INFO` 时：
  - `passed = True`

这里有一个很重要的实现细节：

`EEActionValidator` 里只有 `CRITICAL` 会让整个 `ValidationResult.passed = False`。`MAJOR` 和 `MINOR` 虽然是问题等级，但这个验证器自己仍然返回 `passed=True`。

## 12. `details` 的核心输出结构

最终 `details` 的结构大致是：

```python
{
    "validator_name": "末端执行器动作验证器",
    "category": "末端轨迹",
    "check_name": "抓取检测",
    "issue_level": "critical|major|minor|info",
    "task_description_en": "...",
    "task_required_grasps": {
        "left": int,
        "right": int
    },
    "minimum_detected_grasps": {
        "left": int,
        "right": int
    },
    "episode_duration": float | None,
    "right_pnp_result": [[start_sec, end_sec], ...],
    "left_pnp_result": [[start_sec, end_sec], ...],
    "r_count": int,
    "l_count": int,
    "r_duration": float | None,
    "l_duration": float | None,
    "r_axis_score": float,
    "l_axis_score": float,
    "hands": {
        "right": {...},
        "left": {...}
    }
}
```

单手 `hands[hand]` 里包含：

```python
{
    "hand": "right|left",
    "count": int,
    "segments": [[start_sec, end_sec], ...],
    "duration_ratio": float | None,
    "axis_points": [float, ...],
    "axis_score": float,
    "duration_tag": "none",
    "axis_tag": "none",
    "level": "critical|major|minor|info",
    "reason": str,
    "message": str,
    "minimum_detected_count": int,
    "task_required_count": int,
    "count_delta_from_minimum": int
}
```

## 13. `level` 的判定规则

这是这个验证器最核心的业务逻辑。

先看单手判定 `_evaluate_hand()` 的规则：

1. 默认 `INFO`
2. 如果 `minimum_detected_count > 0` 且当前手 `count < minimum_detected_count`
   - 判 `MAJOR`
3. 否则如果 `count - minimum_detected_count >= 3`
   - 判 `MINOR`
4. 否则
   - 仍是 `INFO`

这里的 `minimum_detected_count` 不是来自任务文本，而是来自当前 episode 左右手检测出的“最小正抓取次数”。

例子：

- 右手 5 次，左手 2 次
- `minimum_detected_count = 2`

则：

- 左手：`2 == 2`，`INFO`
- 右手：`5 - 2 = 3`，`MINOR`

再看总等级 `_finalize_episode_result()` 的规则：

1. 如果左右手 `count` 都是 0
   - 总等级 `CRITICAL`
2. 否则从两只手的等级里取最高优先级
   - `MAJOR` 压过 `MINOR`
   - `MINOR` 压过 `INFO`

所以总等级逻辑是：

- 双手都没抓到 -> `CRITICAL`
- 至少一只手偏少 -> `MAJOR`
- 没有偏少，但有一只手偏多 3 次以上 -> `MINOR`
- 否则 -> `INFO`

## 14. 一个容易误解但很重要的点

虽然 `build_task_context()` 解析了任务描述里的 `task_required_grasps`，并且单手结果里也保留了 `task_required_count`，但当前版本的等级判定并没有直接拿“任务要求次数”来判 `level`。

也就是说，现在的核心判定基线是：

- 当前 episode 检测到的左右手最小正抓取次数

而不是：

- 任务文本明确要求左手/右手至少抓几次

这是当前实现的真实行为。

## 15. 一句话总结

`EEActionValidator` 的工作流就是：先从 DB 和关节文件里恢复左右手时序数据，检测每只手的抓取片段，再把每只手的抓取次数做对比，最后按 `CRITICAL / MAJOR / MINOR / INFO` 输出一个 `ValidationResult`；其中只有“双手都没检测到抓取片段”会让最终结果 `passed=False`。
