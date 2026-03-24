"""数采员页面回调。"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, ctx, dcc, html, ALL, no_update
from plotly.subplots import make_subplots

from src.utils.source_db import query_df
from src.utils.data_parser import get_video_url
from src.utils import clear_cache, get_cache, set_cache


PILOT_SUMMARY_CACHE_KEY = "pilot_summary_table_v5"


def _calc_bench_and_ratio(group_df: pd.DataFrame):
    """计算单个（pilot, task）组的 bench 与倒序占比。"""
    if group_df.empty:
        return None, None, 0

    g = group_df.sort_values("trajectory_start", na_position="last").copy()
    vals = pd.to_numeric(g["duration_sec"], errors="coerce").dropna().to_numpy()
    count = int(len(vals))
    if count == 0:
        return None, None, 0

    first3 = vals[:3]
    if len(first3) < 3:
        bench = float(first3[0])
    else:
        bench = float(np.mean(first3))

    asc_ratio = float(np.sum(vals <= bench) / count * 100)
    desc_ratio = float(100.0 - asc_ratio)
    return bench, desc_ratio, count


def _hex_to_rgba(hex_color, alpha):
    """将 HEX 颜色转换为 rgba 字符串。"""
    color = str(hex_color).lstrip("#")
    if len(color) != 6:
        return f"rgba(134, 239, 172, {alpha})"
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


def _silverman_dispersion(values: np.ndarray) -> float:
    """Silverman 鲁棒离散度，用 min(std, IQR/1.34) 抵抗多峰对离散度的膨胀。"""
    if len(values) < 2:
        return 0.0
    std = float(np.std(values, ddof=1))
    iqr_val = float(np.subtract(*np.percentile(values, [75, 25])))
    if iqr_val <= 0:
        return std
    return min(std, iqr_val / 1.34)


def register_callbacks(app):
    """注册数采员页面的回调。"""

    # 回调1：点击查询按钮时查询数据，存入 Store
    @app.callback(
        Output("pilot-full-data", "data"),
        Input("search-btn", "n_clicks"),
        [
            State("filter-pilot", "value"),
            State("filter-date-range", "start_date"),
            State("filter-date-range", "end_date"),
            State("filter-task", "value"),
        ],
    )
    def fetch_pilot_data(n_clicks, pilot, start_date, end_date, task_id):
        if not n_clicks or not pilot:
            return None

        conditions = ["pilot = %(pilot)s", "valid = true"]
        params = {"pilot": pilot}

        if start_date:
            conditions.append("trajectory_start >= %(start_date)s")
            params["start_date"] = start_date
        if end_date:
            conditions.append(
                "trajectory_start < %(end_date)s::date + interval '1 day'"
            )
            params["end_date"] = end_date
        if task_id:
            conditions.append("task_id = %(task_id)s")
            params["task_id"] = task_id

        where_clause = " AND ".join(conditions)

        sql = f"""
            SELECT
                DATE(trajectory_start) AS date,
                COUNT(*) AS episode_count,
                COALESCE(SUM(trajectory_duration), 0) AS total_duration_sec
            FROM episodes
            WHERE {where_clause}
            GROUP BY DATE(trajectory_start)
            ORDER BY date
        """

        try:
            df = query_df(sql, params)
        except Exception as e:
            return {"error": str(e)}

        if df.empty:
            return {"empty": True}

        # Ensure date is datetime and sort
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        df["total_hours"] = df["total_duration_sec"] / 3600.0
        # Convert back to string for frontend
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")

        return {
            "dates": df["date"].tolist(),
            "counts": df["episode_count"].tolist(),
            "hours": df["total_hours"].tolist(),
            "pilot": pilot,
        }

    # 回调2：数据变化时动态更新滑块的 max 值
    @app.callback(
        [
            Output("pilot-date-slider", "max"),
            Output("pilot-date-slider", "marks"),
            Output("pilot-date-slider", "value"),
        ],
        Input("pilot-full-data", "data"),
        State("pilot-date-slider", "value"),
    )
    def update_slider_range(data, current_value):
        if not data or not isinstance(data, dict) or "error" in data or "empty" in data or "dates" not in data:
            return 60, {i: str(i) for i in range(5, 65, 5)}, 10

        total_days = len(data["dates"])
        max_val = max(total_days, 5)

        # 生成合理的 marks
        if max_val <= 20:
            step = 5
        elif max_val <= 50:
            step = 10
        else:
            step = 20
        marks = {i: str(i) for i in range(step, max_val + 1, step)}
        if max_val not in marks:
            marks[max_val] = str(max_val)

        # 保持当前值，但不超过 max
        new_value = min(current_value, max_val) if current_value else min(10, max_val)

        return max_val, marks, new_value

    # 回调3：数据或滑块变化时更新图表
    @app.callback(
        Output("pilot-chart-container", "children"),
        [
            Input("pilot-full-data", "data"),
            Input("pilot-date-slider", "value"),
        ],
    )
    def update_pilot_chart(data, max_bars):
        if not data or not isinstance(data, dict):
            return html.Div(
                "请选择数采员并点击查询",
                style={"textAlign": "center", "padding": "60px", "color": "#999"},
            )

        if "error" in data:
            return html.Div(
                f"查询出错: {data['error']}",
                style={"color": "red", "padding": "20px"},
            )

        if "empty" in data or "dates" not in data:
            return html.Div(
                "该数采员在所选条件下无采集记录",
                style={"textAlign": "center", "padding": "60px", "color": "#999"},
            )

        dates = data["dates"]
        counts = data["counts"]
        hours = data["hours"]
        pilot_name = data["pilot"]
        # Slice data to show only the last max_bars entries
        if len(dates) > max_bars:
            dates = dates[-max_bars:]
            counts = counts[-max_bars:]
            hours = hours[-max_bars:]

        # 使用唯一的日期标签作为 x 轴，避免 multicategory 导致排序问题
        x_labels = [d[5:].replace("-", "月") + "日" for d in dates]  # "MM月DD日"

        # 双 Y 轴图表
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # 条形图：采集数量 —— 用堆叠薄片模拟从下到上 白→蓝 渐变
        n_gradient = 60  # 段数越多过渡越自然
        for i in range(n_gradient):
            t = i / (n_gradient - 1)  # 0(底部)→1(顶部)
            r = int(255 - (255 - 55) * t)
            g = int(255 - (255 - 100) * t)
            b = int(255 - (255 - 220) * t)
            seg_color = f"rgba({r}, {g}, {b}, 0.9)"
            # 每段略微增高，消除段间抗锯齿缝隙
            segment_y = [c / n_gradient * 1.005 for c in counts]
            is_top = i == n_gradient - 1

            fig.add_trace(
                go.Bar(
                    x=x_labels,
                    y=segment_y,
                    name="采集数量" if is_top else "",
                    showlegend=is_top,
                    legendgroup="采集数量",
                    marker=dict(
                        color=seg_color,
                        line=dict(width=0),
                        cornerradius=3 if is_top else 0,
                    ),
                    customdata=dates if is_top else None,
                    hovertemplate="日期: %{customdata}<br>采集数量: %{text}<extra></extra>" if is_top else None,
                    text=[str(c) for c in counts] if is_top else None,
                    hoverinfo="skip" if not is_top else None,
                ),
                secondary_y=False,
            )

        # 折线图：采集时长
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=hours,
                name="采集时长 (小时)",
                mode="lines+markers",
                line=dict(color="#86efac", width=2.5, shape="spline"),
                marker=dict(size=5, color="#86efac", line=dict(width=1.5, color="#fff")),
                customdata=dates,
                hovertemplate="日期: %{customdata}<br>时长: %{y:.2f}h<extra></extra>",
            ),
            secondary_y=True,
        )

        fig.update_layout(
            title=dict(
                text=f"数采员 {pilot_name} 采集统计",
                font=dict(size=16, color="#1a1a1a"),
                x=0.01,
            ),
            xaxis=dict(
                title_text="",
                categoryorder="array",
                categoryarray=x_labels,  # 显式按数据顺序排列，确保折线正确连接
                gridcolor="rgba(0,0,0,0.04)",
                linecolor="rgba(0,0,0,0.08)",
                tickfont=dict(size=11, color="#888"),
            ),
            yaxis=dict(
                title_text="采集数量",
                title_font=dict(size=12, color="#1a73e8"),
                tickfont=dict(size=11, color="#888"),
                gridcolor="rgba(0,0,0,0.05)",
                zeroline=False,
            ),
            yaxis2=dict(
                title_text="采集时长 (小时)",
                title_font=dict(size=12, color="#86efac"),
                tickfont=dict(size=11, color="#888"),
                gridcolor="rgba(0,0,0,0.05)",
                showgrid=True,
                zeroline=False,
            ),
            barmode="stack",
            height=480,
            margin=dict(l=56, r=56, t=52, b=44),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=12, color="#666"),
                bgcolor="rgba(0,0,0,0)",
            ),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            hoverlabel=dict(
                bgcolor="#fff",
                bordercolor="#eaeaea",
                font=dict(size=12, color="#1a1a1a"),
            ),
        )

        return dcc.Graph(figure=fig, style={"width": "100%"})

    # 回调4：数采员维度任务时长箱形图（每个任务一个箱体）+ 每个箱体基准线
    @app.callback(
        Output("pilot-task-box-chart-container", "children"),
        [
            Input("pilot-full-data", "data"),
            Input("pilot-date-slider", "value"),
        ],
        [
            State("filter-date-range", "start_date"),
            State("filter-date-range", "end_date"),
        ],
    )
    def update_pilot_task_box_chart(data, max_tasks, start_date, end_date):
        if not data or not isinstance(data, dict):
            return html.Div(
                "请选择数采员并点击查询以查看任务箱形图",
                style={"textAlign": "center", "padding": "48px", "color": "#999"},
            )

        if "error" in data:
            return html.Div(
                f"查询出错: {data['error']}",
                style={"color": "red", "padding": "20px"},
            )

        if "empty" in data or "pilot" not in data:
            return html.Div(
                "该数采员在所选条件下无采集记录",
                style={"textAlign": "center", "padding": "48px", "color": "#999"},
            )

        pilot_name = str(data["pilot"])
        conditions = [
            "pilot = %(pilot)s",
            "valid = true",
            "task_id IS NOT NULL",
            "task_id != '0'",
            "trajectory_duration IS NOT NULL",
            "trajectory_duration > 0",
        ]
        params = {"pilot": pilot_name}

        if start_date:
            conditions.append("trajectory_start >= %(start_date)s")
            params["start_date"] = start_date
        if end_date:
            conditions.append("trajectory_start < %(end_date)s::date + interval '1 day'")
            params["end_date"] = end_date

        where_clause = " AND ".join(conditions)
        sql = f"""
            SELECT
                task_id,
                trajectory_duration,
                trajectory_start
            FROM episodes
            WHERE {where_clause}
        """

        try:
            df = query_df(sql, params)
        except Exception as e:
            return html.Div(
                f"查询出错: {e}",
                style={"color": "red", "padding": "20px"},
            )

        if df.empty:
            return html.Div(
                "该数采员在当前筛选条件下无任务时长记录",
                style={"textAlign": "center", "padding": "48px", "color": "#999"},
            )

        df = df.copy()
        df["task_id"] = df["task_id"].astype(str)
        df["duration_sec"] = pd.to_numeric(df["trajectory_duration"], errors="coerce")
        df["trajectory_start"] = pd.to_datetime(df["trajectory_start"], errors="coerce")
        df = df[df["duration_sec"].notna() & (df["duration_sec"] > 0)]

        if df.empty:
            return html.Div(
                "该数采员缺少有效 trajectory_duration 数据",
                style={"textAlign": "center", "padding": "48px", "color": "#999"},
            )

        tasks_all = sorted(
            df["task_id"].unique().tolist(),
            key=lambda x: (0, int(str(x))) if str(x).isdigit() else (1, str(x)),
        )

        # 限制显示的任务数量（受滑块控制）用于 Box Chart
        tasks_sorted = tasks_all
        if max_tasks and len(tasks_all) > max_tasks:
            tasks_sorted = tasks_all[-max_tasks:]

        color_palette = [
            "#60a5fa",
            "#f59e0b",
            "#a78bfa",
            "#34d399",
            "#f87171",
            "#22d3ee",
            "#fb7185",
            "#4ade80",
            "#fbbf24",
            "#818cf8",
        ]

        # 1. 计算 Bench Ratio (对所有任务)
        bench_ratio_x = []
        bench_ratio_y = []
        bench_ratio_hover = []
        df_for_baseline = df.sort_values(["task_id", "trajectory_start"], na_position="last")

        for task_id in tasks_all:
            task_values = df.loc[df["task_id"] == task_id, "duration_sec"].to_numpy()
            if len(task_values) == 0:
                continue

            task_count = int(len(task_values))
            if task_count <= 40:
                continue

            task_rows = df_for_baseline.loc[df_for_baseline["task_id"] == task_id, "duration_sec"].head(3)
            if task_rows.empty:
                continue
            if len(task_rows) < 3:
                baseline = float(task_rows.iloc[0])
            else:
                baseline = float(task_rows.mean())

            asc_ratio = float(np.sum(task_values <= baseline) / task_count * 100)
            bench_ratio = float(100.0 - asc_ratio)
            bench_ratio_x.append(str(task_id))
            bench_ratio_y.append(bench_ratio)
            bench_ratio_hover.append(
                f"task_id: {task_id}<br>bench: {baseline:.1f} s<br>count: {task_count}<br>倒序比例: {bench_ratio:.1f}%"
            )

        # 2. 生成 Box Chart (仅对过滤后的任务 tasks_sorted)
        fig = go.Figure()

        for idx, task_id in enumerate(tasks_sorted):
            task_values = df.loc[df["task_id"] == task_id, "duration_sec"].to_numpy()
            if len(task_values) == 0:
                continue

            task_rows = df_for_baseline.loc[df_for_baseline["task_id"] == task_id, "duration_sec"].head(3)
            if task_rows.empty:
                continue
            if len(task_rows) < 3:
                baseline = float(task_rows.iloc[0])
            else:
                baseline = float(task_rows.mean())

            task_count = int(len(task_values))
            median_val = float(np.median(task_values))
            min_val = float(np.min(task_values))
            max_val = float(np.max(task_values))
            q1_box = float(np.percentile(task_values, 25))
            q3_box = float(np.percentile(task_values, 75))
            
            hover_text = (
                f"median: {median_val:.1f} s<br>"
                f"min: {min_val:.1f} s<br>"
                f"max: {max_val:.1f} s<br>"
                f"count: {task_count}<br>"
                f"bench: {baseline:.1f} s"
            )

            base_color = color_palette[idx % len(color_palette)]

            fig.add_trace(
                go.Box(
                    x=[idx] * task_count,
                    y=task_values,
                    name=str(task_id),
                    boxpoints=False,
                    jitter=0.22,
                    pointpos=0,
                    hoverinfo="skip",
                    hovertemplate="<extra></extra>",
                    marker=dict(color=_hex_to_rgba(base_color, 0.35), size=4),
                    line=dict(color=_hex_to_rgba(base_color, 0.9), width=1.8),
                    fillcolor=_hex_to_rgba(base_color, 0.22),
                    showlegend=False,
                )
            )

            if q3_box > q1_box:
                hover_base = q1_box
                hover_height = q3_box - q1_box
            else:
                hover_base = max(q1_box - 0.15, 0)
                hover_height = 0.3

            fig.add_trace(
                go.Bar(
                    x=[idx],
                    y=[hover_height],
                    base=[hover_base],
                    width=[0.56],
                    marker=dict(color="rgba(0,0,0,0.001)", line=dict(width=0)),
                    hovertemplate=hover_text + "<extra></extra>",
                    showlegend=False,
                )
            )

            if task_count >= 4:
                iqr = q3_box - q1_box
                lower_fence = q1_box - 1.5 * iqr
                upper_fence = q3_box + 1.5 * iqr
                outlier_values = task_values[(task_values < lower_fence) | (task_values > upper_fence)]
            else:
                outlier_values = np.array([])

            if len(outlier_values) > 0:
                if len(outlier_values) == 1:
                    outlier_x = np.array([float(idx)])
                else:
                    outlier_x = np.linspace(idx - 0.16, idx + 0.16, len(outlier_values))

                fig.add_trace(
                    go.Scatter(
                        x=outlier_x,
                        y=outlier_values,
                        mode="markers",
                        marker=dict(color=_hex_to_rgba(base_color, 0.6), size=4),
                        hovertemplate=(
                            f"task_id: {task_id}<br>"
                            "时长: %{y:.1f} s<extra></extra>"
                        ),
                        showlegend=False,
                    )
                )

            fig.add_shape(
                type="line",
                xref="x",
                yref="y",
                x0=idx - 0.28,
                x1=idx + 0.28,
                y0=baseline,
                y1=baseline,
                line=dict(color="#ef4444", width=2.4),
                layer="above",
            )

        fig.update_layout(
            title=dict(
                text=f"{pilot_name} 采集任务数：{len(tasks_sorted)}  采集数量：{len(df)}",
                font=dict(size=15, color="#1a1a1a"),
                x=0.01,
            ),
            xaxis=dict(
                title_text="任务 task_id",
                tickmode="array",
                tickvals=list(range(len(tasks_sorted))),
                ticktext=tasks_sorted,
                tickangle=-20,
                tickfont=dict(size=11, color="#666"),
                title_font=dict(size=12, color="#666"),
                gridcolor="rgba(0,0,0,0.03)",
                zeroline=False,
            ),
            yaxis=dict(
                title_text="trajectory_duration（s）",
                tickfont=dict(size=11, color="#888"),
                title_font=dict(size=12, color="#666"),
                gridcolor="rgba(0,0,0,0.05)",
                zeroline=False,
            ),
            height=420,
            margin=dict(l=56, r=32, t=54, b=68),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            hoverlabel=dict(
                bgcolor="#fff",
                bordercolor="#eaeaea",
                font=dict(size=12, color="#1a1a1a"),
            ),
            annotations=[
                dict(
                    x=0,
                    y=1.08,
                    xref="paper",
                    yref="paper",
                    text="红线 = 基准线（前3条平均；不足3条取第1条）",
                    showarrow=False,
                    font=dict(size=11, color="#ef4444"),
                    xanchor="left",
                )
            ],
        )

        bench_fig = go.Figure()
        if bench_ratio_x:
            bench_fig.add_trace(
                go.Scatter(
                    x=bench_ratio_x,
                    y=bench_ratio_y,
                    mode="lines+markers",
                    line=dict(color="#3b82f6", width=2.4),
                    marker=dict(size=6, color="#3b82f6"),
                    text=bench_ratio_hover,
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=False,
                )
            )
            bench_fig.add_shape(
                type="line",
                xref="paper",
                yref="y",
                x0=0,
                x1=1,
                y0=60,
                y1=60,
                line=dict(color="#ef4444", width=2, dash="dash"),
            )
            bench_fig.add_annotation(
                x=0,
                y=60,
                xref="paper",
                yref="y",
                text="y=60% 参考线",
                showarrow=False,
                font=dict(size=11, color="#ef4444"),
                xanchor="left",
                yanchor="bottom",
            )
        else:
            bench_fig.add_annotation(
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                text="无满足条件的数据（仅展示采集数量 > 40 的任务）",
                showarrow=False,
                font=dict(size=12, color="#999"),
            )

        bench_fig.update_layout(
            title=dict(
                text="bench 在任务采集分布中的占比（仅 count > 40）",
                font=dict(size=14, color="#1a1a1a"),
                x=0.01,
            ),
            xaxis=dict(
                title_text="任务 task_id",
                tickangle=-20,
                tickfont=dict(size=11, color="#666"),
                title_font=dict(size=12, color="#666"),
                gridcolor="rgba(0,0,0,0.03)",
                zeroline=False,
            ),
            yaxis=dict(
                title_text="bench 排位占比（%）",
                range=[0, 100],
                ticksuffix="%",
                tickfont=dict(size=11, color="#888"),
                title_font=dict(size=12, color="#666"),
                gridcolor="rgba(0,0,0,0.05)",
                zeroline=False,
            ),
            height=280,
            margin=dict(l=56, r=32, t=52, b=58),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            hoverlabel=dict(
                bgcolor="#fff",
                bordercolor="#eaeaea",
                font=dict(size=12, color="#1a1a1a"),
            ),
        )

        return html.Div(
            [
                dcc.Graph(figure=fig, style={"width": "100%"}),
                dcc.Graph(figure=bench_fig, style={"width": "100%", "marginTop": "8px"}),
            ]
        )

    # 回调5：任务时长分布面积图（按 pilot 分组）
    @app.callback(
        Output("task-duration-chart-container", "children"),
        [
            Input("filter-task", "value"),
            Input("filter-pilot", "value"),
            Input("filter-date-range", "start_date"),
            Input("filter-date-range", "end_date"),
        ],
    )
    def update_task_duration_chart(task_id, selected_pilot, start_date, end_date):
        if not task_id:
            return html.Div(
                "请选择数采员后再次选择任务",
                style={
                    "height": "360px",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "backgroundColor": "#f9fafb",
                    "color": "#9ca3af",
                    "fontSize": "16px",
                    "border": "1px dashed #e5e7eb",
                    "borderRadius": "8px",
                },
            )

        conditions = [
            "task_id = %(task_id)s",
            "valid = true",
            "trajectory_duration IS NOT NULL",
            "trajectory_duration > 0",
        ]
        params = {"task_id": task_id}

        if start_date:
            conditions.append("trajectory_start >= %(start_date)s")
            params["start_date"] = start_date
        if end_date:
            conditions.append("trajectory_start < %(end_date)s::date + interval '1 day'")
            params["end_date"] = end_date

        where_clause = " AND ".join(conditions)
        sql = f"""
            SELECT
                pilot,
                trajectory_duration
            FROM episodes
            WHERE {where_clause}
        """

        try:
            df = query_df(sql, params)
        except Exception as e:
            return html.Div(
                f"查询出错: {e}",
                style={"color": "red", "padding": "20px"},
            )

        if df.empty:
            return html.Div(
                "该任务在当前筛选条件下无记录",
                style={"textAlign": "center", "padding": "48px", "color": "#999"},
            )

        df = df.copy()
        df["pilot"] = df["pilot"].astype(str)
        df["duration_sec"] = pd.to_numeric(df["trajectory_duration"], errors="coerce")
        df = df[df["duration_sec"].notna() & (df["duration_sec"] > 0)]

        if df.empty:
            return html.Div(
                "该任务缺少有效 trajectory_duration 数据",
                style={"textAlign": "center", "padding": "48px", "color": "#999"},
            )

        # 对极端长尾做轻微裁剪，避免主分布被压扁
        low_q = float(df["duration_sec"].quantile(0.01))
        high_q = float(df["duration_sec"].quantile(0.99))
        if high_q <= low_q:
            low_q = float(df["duration_sec"].min())
            high_q = float(df["duration_sec"].max())
        if high_q <= low_q:
            high_q = low_q + 1.0

        clipped_seconds = df["duration_sec"].clip(lower=low_q, upper=high_q)

        # 根据样本量动态设置分箱数
        bin_count = min(40, max(15, int(np.sqrt(len(clipped_seconds)))))
        bin_edges = np.linspace(float(clipped_seconds.min()), float(clipped_seconds.max()), bin_count + 1)
        if len(bin_edges) < 2 or np.isclose(bin_edges[0], bin_edges[-1]):
            center = float(clipped_seconds.iloc[0])
            left = max(center - 0.5, 0)
            right = center + 0.5
            bin_edges = np.array([left, right])

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        pilots_sorted = (
            df.groupby("pilot")
            .size()
            .sort_values(ascending=False)
            .index
            .tolist()
        )

        color_palette = [
            "#60a5fa",
            "#f59e0b",
            "#a78bfa",
            "#34d399",
            "#f87171",
            "#22d3ee",
            "#fb7185",
            "#4ade80",
            "#fbbf24",
            "#818cf8",
        ]

        fig = go.Figure()

        selected_pilot_str = str(selected_pilot) if selected_pilot else None
        plot_order = [p for p in pilots_sorted if str(p) != selected_pilot_str]
        if selected_pilot_str:
            plot_order.extend([p for p in pilots_sorted if str(p) == selected_pilot_str])

        color_map = {
            pilot_name: color_palette[idx % len(color_palette)]
            for idx, pilot_name in enumerate(pilots_sorted)
        }

        for pilot_name in plot_order:
            pilot_mask = df["pilot"] == pilot_name
            pilot_vals = clipped_seconds[pilot_mask].to_numpy()
            hist_counts, _ = np.histogram(pilot_vals, bins=bin_edges)

            base_color = color_map[pilot_name]
            is_selected = bool(selected_pilot) and str(selected_pilot) == str(pilot_name)

            fig.add_trace(
                go.Scatter(
                    x=bin_centers,
                    y=hist_counts,
                    name=f"{pilot_name} ({len(pilot_vals)})",
                    mode="lines",
                    line=dict(
                        color=_hex_to_rgba(base_color, 1.0 if is_selected else (0.4 if selected_pilot else 0.75)),
                        width=4.2 if is_selected else (1.1 if selected_pilot else 1.8),
                        shape="spline",
                    ),
                    fill="tozeroy",
                    fillcolor=_hex_to_rgba(base_color, 0.5 if is_selected else (0.04 if selected_pilot else 0.14)),
                    opacity=1.0 if is_selected else (0.42 if selected_pilot else 0.9),
                    hovertemplate=(
                        f"数采员: {pilot_name}<br>"
                        "时长: %{x:.1f} s<br>"
                        "采集条数: %{y}<extra></extra>"
                    ),
                )
            )

        title_suffix = f"（高亮: {selected_pilot}）" if selected_pilot else ""
        fig.update_layout(
            title=dict(
                text=f"任务 {task_id} 采集时长分布（按数采员）{title_suffix}",
                font=dict(size=15, color="#1a1a1a"),
                x=0.01,
            ),
            xaxis=dict(
                title_text="trajectory_duration（s）",
                tickfont=dict(size=11, color="#888"),
                title_font=dict(size=12, color="#666"),
                gridcolor="rgba(0,0,0,0.05)",
                zeroline=False,
            ),
            yaxis=dict(
                title_text="采集条数",
                tickfont=dict(size=11, color="#888"),
                title_font=dict(size=12, color="#666"),
                gridcolor="rgba(0,0,0,0.05)",
                zeroline=False,
            ),
            height=360,
            margin=dict(l=56, r=32, t=54, b=48),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=11, color="#666"),
                bgcolor="rgba(0,0,0,0)",
            ),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            hoverlabel=dict(
                bgcolor="#fff",
                bordercolor="#eaeaea",
                font=dict(size=12, color="#1a1a1a"),
            ),
        )

        if selected_pilot and str(selected_pilot) not in pilots_sorted:
            fig.add_annotation(
                text=f"当前选中的数采员 {selected_pilot} 在该任务下无记录",
                xref="paper",
                yref="paper",
                x=0.5,
                y=1.08,
                showarrow=False,
                font=dict(size=11, color="#999"),
            )

        return dcc.Graph(figure=fig, style={"width": "100%"})

    # 回调5：任务时长箱形图（按 pilot）+ 每个箱体基准线
    @app.callback(
        Output("task-duration-box-chart-container", "children"),
        [
            Input("filter-task", "value"),
            Input("filter-pilot", "value"),
            Input("filter-date-range", "start_date"),
            Input("filter-date-range", "end_date"),
        ],
    )
    def update_task_duration_box_chart(task_id, selected_pilot, start_date, end_date):
        if not task_id:
            return html.Div(
                "请选择数采员后再次选择任务",
                style={
                    "height": "420px",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "backgroundColor": "#f9fafb",
                    "color": "#9ca3af",
                    "fontSize": "16px",
                    "border": "1px dashed #e5e7eb",
                    "borderRadius": "8px",
                },
            )

        conditions = [
            "task_id = %(task_id)s",
            "valid = true",
            "trajectory_duration IS NOT NULL",
            "trajectory_duration > 0",
        ]
        params = {"task_id": task_id}

        if start_date:
            conditions.append("trajectory_start >= %(start_date)s")
            params["start_date"] = start_date
        if end_date:
            conditions.append("trajectory_start < %(end_date)s::date + interval '1 day'")
            params["end_date"] = end_date

        where_clause = " AND ".join(conditions)
        sql = f"""
            SELECT
                pilot,
                trajectory_duration,
                trajectory_start
            FROM episodes
            WHERE {where_clause}
        """

        try:
            df = query_df(sql, params)
        except Exception as e:
            return html.Div(
                f"查询出错: {e}",
                style={"color": "red", "padding": "20px"},
            )

        if df.empty:
            return html.Div(
                "该任务在当前筛选条件下无记录",
                style={"textAlign": "center", "padding": "48px", "color": "#999"},
            )

        df = df.copy()
        df["pilot"] = df["pilot"].astype(str)
        df["duration_sec"] = pd.to_numeric(df["trajectory_duration"], errors="coerce")
        df["trajectory_start"] = pd.to_datetime(df["trajectory_start"], errors="coerce")
        df = df[df["duration_sec"].notna() & (df["duration_sec"] > 0)]

        if df.empty:
            return html.Div(
                "该任务缺少有效 trajectory_duration 数据",
                style={"textAlign": "center", "padding": "48px", "color": "#999"},
            )

        pilots_sorted = sorted(
            df["pilot"].unique().tolist(),
            key=lambda x: (0, int(str(x))) if str(x).isdigit() else (1, str(x)),
        )

        color_palette = [
            "#60a5fa",
            "#f59e0b",
            "#a78bfa",
            "#34d399",
            "#f87171",
            "#22d3ee",
            "#fb7185",
            "#4ade80",
            "#fbbf24",
            "#818cf8",
        ]

        fig = go.Figure()
        df_for_baseline = df.sort_values(["pilot", "trajectory_start"], na_position="last")

        for idx, pilot_name in enumerate(pilots_sorted):
            pilot_values = df.loc[df["pilot"] == pilot_name, "duration_sec"].to_numpy()
            if len(pilot_values) == 0:
                continue

            pilot_rows = df_for_baseline.loc[df_for_baseline["pilot"] == pilot_name, "duration_sec"].head(3)
            if pilot_rows.empty:
                continue
            if len(pilot_rows) < 3:
                baseline = float(pilot_rows.iloc[0])
            else:
                baseline = float(pilot_rows.mean())

            pilot_count = int(len(pilot_values))
            median_val = float(np.median(pilot_values))
            min_val = float(np.min(pilot_values))
            max_val = float(np.max(pilot_values))
            q1_box = float(np.percentile(pilot_values, 25))
            q3_box = float(np.percentile(pilot_values, 75))
            hover_text = (
                f"median: {median_val:.1f} s<br>"
                f"min: {min_val:.1f} s<br>"
                f"max: {max_val:.1f} s<br>"
                f"count: {pilot_count}<br>"
                f"bench: {baseline:.1f} s"
            )

            base_color = color_palette[idx % len(color_palette)]

            fig.add_trace(
                go.Box(
                    x=[idx] * pilot_count,
                    y=pilot_values,
                    name=str(pilot_name),
                    boxpoints=False,
                    jitter=0.22,
                    pointpos=0,
                    hoverinfo="skip",
                    hovertemplate="<extra></extra>",
                    marker=dict(color=_hex_to_rgba(base_color, 0.35), size=4),
                    line=dict(color=_hex_to_rgba(base_color, 0.9), width=1.8),
                    fillcolor=_hex_to_rgba(base_color, 0.22),
                    showlegend=False,
                )
            )

            if q3_box > q1_box:
                hover_base = q1_box
                hover_height = q3_box - q1_box
            else:
                hover_base = max(q1_box - 0.15, 0)
                hover_height = 0.3

            fig.add_trace(
                go.Bar(
                    x=[idx],
                    y=[hover_height],
                    base=[hover_base],
                    width=[0.56],
                    marker=dict(color="rgba(0,0,0,0.001)", line=dict(width=0)),
                    hovertemplate=hover_text + "<extra></extra>",
                    showlegend=False,
                )
            )

            if pilot_count >= 4:
                iqr = q3_box - q1_box
                lower_fence = q1_box - 1.5 * iqr
                upper_fence = q3_box + 1.5 * iqr
                outlier_values = pilot_values[(pilot_values < lower_fence) | (pilot_values > upper_fence)]
            else:
                outlier_values = np.array([])

            if len(outlier_values) > 0:
                if len(outlier_values) == 1:
                    outlier_x = np.array([float(idx)])
                else:
                    outlier_x = np.linspace(idx - 0.16, idx + 0.16, len(outlier_values))

                fig.add_trace(
                    go.Scatter(
                        x=outlier_x,
                        y=outlier_values,
                        mode="markers",
                        marker=dict(color=_hex_to_rgba(base_color, 0.6), size=4),
                        hovertemplate=(
                            f"task_id: {task_id}<br>"
                            "时长: %{y:.1f} s<extra></extra>"
                        ),
                        showlegend=False,
                    )
                )

            fig.add_shape(
                type="line",
                xref="x",
                yref="y",
                x0=idx - 0.28,
                x1=idx + 0.28,
                y0=baseline,
                y1=baseline,
                line=dict(color="#ef4444", width=2.4),
                layer="above",
            )

        highlight_pilot = str(selected_pilot) if selected_pilot else "无"
        fig.update_layout(
            title=dict(
                text=f"{task_id} 采集总数：{len(df)}（高亮：{highlight_pilot}）",
                font=dict(size=15, color="#1a1a1a"),
                x=0.01,
            ),
            xaxis=dict(
                title_text="数采员 pilot 编号",
                tickmode="array",
                tickvals=list(range(len(pilots_sorted))),
                ticktext=pilots_sorted,
                tickangle=-20,
                tickfont=dict(size=11, color="#666"),
                title_font=dict(size=12, color="#666"),
                gridcolor="rgba(0,0,0,0.03)",
                zeroline=False,
            ),
            yaxis=dict(
                title_text="trajectory_duration（s）",
                tickfont=dict(size=11, color="#888"),
                title_font=dict(size=12, color="#666"),
                gridcolor="rgba(0,0,0,0.05)",
                zeroline=False,
            ),
            height=420,
            margin=dict(l=56, r=32, t=54, b=68),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            hoverlabel=dict(
                bgcolor="#fff",
                bordercolor="#eaeaea",
                font=dict(size=12, color="#1a1a1a"),
            ),
            annotations=[
                dict(
                    x=0,
                    y=1.08,
                    xref="paper",
                    yref="paper",
                    text="红线 = 基准线（前3条平均；不足3条取第1条）",
                    showarrow=False,
                    font=dict(size=11, color="#ef4444"),
                    xanchor="left",
                )
            ],
        )

        return dcc.Graph(figure=fig, style={"width": "100%"})

    # 回调6：页面底部汇总表（不受筛选框影响）
    @app.callback(
        Output("pilot-summary-table-container", "children"),
        Input("url", "pathname"),
        Input("pilot-summary-cache-clear-btn", "n_clicks"),
    )
    def update_pilot_summary_table(_pathname, clear_clicks):
        triggered_id = getattr(ctx, "triggered_id", None)
        if triggered_id == "pilot-summary-cache-clear-btn" and clear_clicks:
            clear_cache(PILOT_SUMMARY_CACHE_KEY)

        cached_component = get_cache(PILOT_SUMMARY_CACHE_KEY)
        if cached_component is not None:
            return cached_component

        def _cache_and_return(component):
            set_cache(PILOT_SUMMARY_CACHE_KEY, component)
            return component

        sql = """
            SELECT
                e.pilot,
                e.task_id,
                e.trajectory_duration,
                e.trajectory_start
            FROM episodes e
            WHERE e.valid = true
              AND e.pilot IS NOT NULL
              AND e.task_id IS NOT NULL
              AND e.task_id != '0'
              AND e.trajectory_duration IS NOT NULL
              AND e.trajectory_duration > 0
        """

        try:
            df = query_df(sql)
        except Exception as e:
            return html.Div(
                f"统计表查询出错: {e}",
                style={"color": "red", "padding": "20px"},
            )

        if df.empty:
            return _cache_and_return(
                html.Div(
                    "暂无可用于统计的数据",
                    style={"textAlign": "center", "padding": "48px", "color": "#999"},
                )
            )

        df = df.copy()
        df["pilot"] = df["pilot"].astype(str)
        df["task_id"] = df["task_id"].astype(str)
        df["duration_sec"] = pd.to_numeric(df["trajectory_duration"], errors="coerce")
        df["trajectory_start"] = pd.to_datetime(df["trajectory_start"], errors="coerce")
        df = df[df["duration_sec"].notna() & (df["duration_sec"] > 0)]

        if df.empty:
            return _cache_and_return(
                html.Div(
                    "暂无可用于统计的有效时长数据",
                    style={"textAlign": "center", "padding": "48px", "color": "#999"},
                )
            )

        # 每个任务的全局中位数（用于离散度偏度修正）
        task_global_median = df.groupby("task_id")["duration_sec"].median().to_dict()

        # 先按（pilot, task）聚合，计算每个组合的 bench、倒序占比、count、离散度
        combo_records = []
        for (pilot, task_id), g in df.groupby(["pilot", "task_id"], sort=False):
            bench, desc_ratio, count = _calc_bench_and_ratio(g)
            vals = pd.to_numeric(g["duration_sec"], errors="coerce").dropna().to_numpy()
            raw_disp = _silverman_dispersion(vals) if len(vals) >= 2 else 0.0

            # 偏度修正：偏左（用时短）打折，偏右（用时长）加罚
            task_med = task_global_median.get(str(task_id), 0)
            if task_med > 0 and len(vals) >= 2:
                pilot_median = float(np.median(vals))
                shift_ratio = (pilot_median - task_med) / task_med
                skew_factor = 1.0 + float(np.clip(shift_ratio, -0.3, 0.3))
                dispersion = raw_disp * skew_factor
            else:
                dispersion = raw_disp

            combo_records.append(
                {
                    "pilot": str(pilot),
                    "task_id": str(task_id),
                    "count": int(count),
                    "bench": bench,
                    "desc_ratio": desc_ratio,
                    "dispersion": float(dispersion),
                    "is_unqualified": bool(desc_ratio is not None and desc_ratio > 60),
                }
            )

        combo_df = pd.DataFrame(combo_records)
        if combo_df.empty:
            return _cache_and_return(
                html.Div(
                    "暂无可用于统计的任务组合",
                    style={"textAlign": "center", "padding": "48px", "color": "#999"},
                )
            )

        # 计算"同任务其他数采员中，bench 倒序占比 <60% 的占比 >50%"的门槛
        task_peer_stats = {}
        for task_id, task_g in combo_df[combo_df["desc_ratio"].notna()].groupby("task_id"):
            task_peer_stats[str(task_id)] = task_g[["pilot", "desc_ratio"]].copy()

        def _peer_majority_unqualified(row):
            task_id = str(row["task_id"])
            pilot = str(row["pilot"])
            cur_unqualified = bool(row["is_unqualified"])
            if not cur_unqualified:
                return False
            peers = task_peer_stats.get(task_id)
            if peers is None:
                return False
            peers = peers[peers["pilot"] != pilot]
            if peers.empty:
                return False
            peer_under_60_ratio = float((peers["desc_ratio"] < 60).mean() * 100)
            return peer_under_60_ratio > 50.0

        combo_df["peer_gate_pass"] = combo_df.apply(_peer_majority_unqualified, axis=1)
        combo_df["count_gt_40"] = combo_df["count"] > 40
        combo_df["final_unqualified"] = (
            combo_df["is_unqualified"] & combo_df["peer_gate_pass"] & combo_df["count_gt_40"]
        )

        # 计算离散度相对位次：仅 count>40 的组合参与，且任务需 ≥2 人
        disp_eligible = combo_df[combo_df["count"] > 40].copy()
        # 按 task 过滤：只保留 ≥2 个数采员的任务
        task_pilot_cnt = disp_eligible.groupby("task_id")["pilot"].nunique()
        multi_pilot_tasks = set(task_pilot_cnt[task_pilot_cnt >= 2].index)
        disp_eligible = disp_eligible[disp_eligible["task_id"].isin(multi_pilot_tasks)]

        disp_ranks = []
        for _tid, task_g in disp_eligible.groupby("task_id", sort=False):
            n = len(task_g)
            task_g = task_g.copy()
            if n <= 1:
                continue  # 过滤后仍只剩1人则跳过
            task_g["disp_relative_rank"] = (
                task_g["dispersion"].rank(method="average", ascending=True) - 1
            ) / (n - 1)
            disp_ranks.append(task_g)

        if disp_ranks:
            disp_ranked_df = pd.concat(disp_ranks, ignore_index=True)
            pilot_disp_map = disp_ranked_df.groupby("pilot")["disp_relative_rank"].mean().to_dict()
        else:
            pilot_disp_map = {}

        # user_name 映射（users.user_name -> episodes.pilot）
        try:
            users_df = query_df("SELECT user_name FROM users ORDER BY user_name")
            valid_user_names = set(users_df["user_name"].astype(str).tolist())
        except Exception:
            valid_user_names = set()

        pilot_group = combo_df.groupby("pilot", sort=True)
        table_rows = []
        for pilot_name, g in pilot_group:
            collect_task_count = int(g["task_id"].nunique())
            collect_episode_count = int(g["count"].sum())
            unqualified_count = int(g["final_unqualified"].sum())
            unqualified_ratio = (
                float(unqualified_count / collect_task_count * 100) if collect_task_count > 0 else 0.0
            )
            disp_rank_val = pilot_disp_map.get(pilot_name)
            disp_rank_str = f"{disp_rank_val * 100:.1f}%" if disp_rank_val is not None else "—"

            display_user = pilot_name if (not valid_user_names or pilot_name in valid_user_names) else pilot_name

            table_rows.append(
                {
                    "users.user_name": display_user,
                    "采集任务数目": collect_task_count,
                    "采集任务条数": collect_episode_count,
                    "基准线不达标": unqualified_count,
                    "基准线不达标比例": f"{unqualified_ratio:.1f}%",
                    "离散度位次": disp_rank_str,
                }
            )

        if not table_rows:
            return _cache_and_return(
                html.Div(
                    "暂无可展示的统计结果",
                    style={"textAlign": "center", "padding": "48px", "color": "#999"},
                )
            )

        # 表格样式定义
        th_style = {
            "padding": "12px",
            "textAlign": "left",
            "borderBottom": "1px solid #e5e7eb",
            "fontSize": "13px",
            "fontWeight": "600",
            "color": "#374151",
            "backgroundColor": "#f9fafb",
        }
        td_style = {
            "padding": "12px",
            "textAlign": "left",
            "borderBottom": "1px solid #f3f4f6",
            "fontSize": "13px",
            "color": "#4b5563",
        }

        header_cells = [
            html.Th("users.user_name", style=th_style),
            html.Th("采集任务数目", style=th_style),
            html.Th("采集任务条数", style=th_style),
            html.Th("基准线不达标", style=th_style),
            html.Th("基准线不达标比例", style=th_style),
            html.Th("离散度位次", style=th_style),
        ]

        body_rows = []
        for row in table_rows:
            body_rows.append(
                html.Tr(
                    [
                        html.Td(row["users.user_name"], style=td_style),
                        html.Td(row["采集任务数目"], style=td_style),
                        html.Td(row["采集任务条数"], style=td_style),
                        html.Td(row["基准线不达标"], style=td_style),
                        html.Td(row["基准线不达标比例"], style=td_style),
                        html.Td(row["离散度位次"], style=td_style),
                    ]
                )
            )

        summary_component = html.Div(
            [
                html.H5(
                    "数采员全局统计（不受筛选框影响）",
                    style={
                        "margin": "0 0 16px 0",
                        "fontWeight": "600",
                        "fontSize": "16px",
                        "color": "#111827",
                    },
                ),
                html.Div(
                    html.Table(
                        [
                            html.Thead(html.Tr(header_cells)),
                            html.Tbody(body_rows),
                        ],
                        style={
                            "width": "100%",
                            "borderCollapse": "collapse",
                            "background": "#fff",
                        },
                    ),
                    style={
                        "overflowX": "auto",
                        "border": "1px solid #e5e7eb",
                        "borderRadius": "8px",
                        "marginBottom": "12px",
                    },
                ),
                html.Div(
                    "说明：'基准线不达标'按任务维度统计，需当前数采员该任务 count>40 且该任务 bench 倒序占比 >60%，并且去除本人后其余数采员中 bench 倒序占比 <60% 的占比 >50% 才计入。",
                    style={"fontSize": "12px", "color": "#6b7280", "lineHeight": "1.5"},
                ),
                html.Div(
                    "说明：'离散度位次'衡量数采员在每个任务中采集时长的集中程度相对排名（Silverman 鲁棒离散度，允许多峰）。仅 count>40 且 ≥2 人采集的任务参与计算，值越小表示越集中稳定，'—'表示无符合条件的任务。",
                    style={"fontSize": "12px", "color": "#6b7280", "lineHeight": "1.5", "marginTop": "4px"},
                ),
            ],
            style={
                "backgroundColor": "#fff",
                "padding": "24px",
                "borderRadius": "12px",
                "boxShadow": "0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)",
                "border": "1px solid #e5e7eb",
                "marginTop": "24px",
            },
        )
        return _cache_and_return(summary_component)
