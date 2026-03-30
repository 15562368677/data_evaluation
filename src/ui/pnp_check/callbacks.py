"""PnP 筛选页面回调。"""

from urllib.parse import quote

import json
import logging
import pandas as pd
from dash import Input, Output, State, ALL, MATCH, ctx, html, no_update

from src.utils.source_db import query_df
from src.utils.result_db import query_pnp_df, save_pnp_results, query_checked_pnp_episodes
from src.utils.data_parser import get_video_url

# ── 状态 / 标签常量 ──
PNP_STATUS_ORDER = ["pass", "multi_pick", "fail_pick", "invalid"]
PNP_STATUS_LABEL = {
    "pass": "合格",
    "multi_pick": "多次抓取",
    "fail_pick": "抓取不合格",
    "invalid": "无效",
}
PNP_STATUS_COLOR = {
    "pass": "#059669",
    "multi_pick": "#d97706",
    "fail_pick": "#ef4444",
    "invalid": "#6b7280",
}

OUTLIER_ORDER = ["none", "low", "normal", "high"]
OUTLIER_LABEL = {
    "none": "无PnP",
    "low": "低离群",
    "normal": "正常",
    "high": "高离群",
}


def _normalize_segments(raw_val):
    if not raw_val:
        return []
    parsed = raw_val
    if isinstance(raw_val, str):
        try:
            parsed = json.loads(raw_val)
        except Exception:
            return []
    if not isinstance(parsed, list):
        return []

    segments = []
    for seg in parsed:
        if not isinstance(seg, (list, tuple)) or len(seg) < 2:
            continue
        try:
            st = float(seg[0])
            ed = float(seg[1])
        except Exception:
            continue
        if ed < st:
            st, ed = ed, st
        segments.append([st, ed])
    return segments


def _calc_total_duration(segments):
    return sum(max(0.0, float(ed) - float(st)) for st, ed in segments)


def _calc_last_end(segments):
    if not segments:
        return 0.0
    return max(float(seg[1]) for seg in segments)


def _calc_iqr_bounds(values):
    arr = [float(v) for v in values if v is not None and float(v) > 0]
    if len(arr) < 4:
        return None, None
    s = pd.Series(arr)
    q1 = float(s.quantile(0.25))
    q3 = float(s.quantile(0.75))
    iqr = q3 - q1
    if iqr <= 0:
        return None, None
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def _classify_outlier(value, low, high):
    if value is None or value <= 0:
        return "none"
    if low is None or high is None:
        return "normal"
    if value < low:
        return "low"
    if value > high:
        return "high"
    return "normal"


def _calc_sigma_bounds(values, sigma_k=3.0):
    arr = [float(v) for v in values if v is not None and float(v) > 0]
    if len(arr) < 2:
        return None, None
    s = pd.Series(arr)
    mean_v = float(s.mean())
    std_v = float(s.std(ddof=0))
    if std_v <= 0:
        return None, None
    low = mean_v - sigma_k * std_v
    high = mean_v + sigma_k * std_v
    # 比例轴映射到 [0,1]
    low = max(0.0, low)
    high = min(1.0, high)
    if high <= low:
        return None, None
    return low, high


def _pnp_sort_key(value):
    sval = str(value)
    if sval.isdigit():
        return 0, int(sval)
    return 1, sval


def _build_checked_card(row: dict, label: str, selected_episode: str = ""):
    """构建已检测数据的只读卡片。"""
    ep_id = str(row.get("episode_id", ""))
    task_id = str(row.get("task_id", ""))
    r_count = row.get("r_count", 0)
    l_count = row.get("l_count", 0)

    label_text = PNP_STATUS_LABEL.get(label, label)
    label_color = PNP_STATUS_COLOR.get(label, "#6b7280")

    is_selected = str(ep_id) == str(selected_episode or "")

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Button(
                                f"Episode: {ep_id}",
                                id={"type": "pnp-check-open-video-title", "episode_id": ep_id},
                                n_clicks=0,
                                style={
                                    "border": "none",
                                    "background": "transparent",
                                    "color": "#2563eb",
                                    "padding": "0",
                                    "fontSize": "13px",
                                    "fontWeight": "600",
                                    "cursor": "pointer",
                                },
                            ),
                            html.Span(
                                f"task_id: {task_id} ｜ 右手: {r_count}次 ｜ 左手: {l_count}次",
                                style={"fontSize": "12px", "color": "#6b7280", "marginLeft": "8px"},
                            ),
                        ],
                        style={"display": "flex", "alignItems": "center", "flexWrap": "wrap", "gap": "4px"},
                    ),
                    html.Span(
                        label_text,
                        style={
                            "fontSize": "12px",
                            "color": "#fff",
                            "background": label_color,
                            "padding": "3px 10px",
                            "borderRadius": "6px",
                            "fontWeight": "600",
                            "marginLeft": "8px",
                            "whiteSpace": "nowrap",
                        },
                    ),
                ],
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "gap": "8px"},
            )
        ],
        style={
            "border": "1px solid #3b82f6" if is_selected else f"1px solid {label_color}30",
            "borderLeft": f"4px solid {label_color}",
            "borderRadius": "8px",
            "padding": "10px 12px",
            "background": "#e0f2fe" if is_selected else f"{label_color}08",
            "marginBottom": "8px",
        },
    )


def _build_pnp_card(row: dict, status_map: dict, selected_episode: str = ""):
    """构建单条数据的卡片（用于左侧表格区域）。"""
    ep_id = str(row.get("episode_id", ""))
    task_id = str(row.get("task_id", ""))
    r_count = row.get("r_count", 0)
    l_count = row.get("l_count", 0)
    
    current_status = status_map.get(ep_id)

    def _status_btn(key, label, color):
        is_active = current_status == key
        return html.Button(
            label,
            id={"type": "pnp-check-row-status-btn", "episode_id": ep_id, "status": key},
            n_clicks=0,
            style={
                "border": f"1px solid {color}",
                "background": color if is_active else "#fff",
                "color": "#fff" if is_active else color,
                "padding": "3px 8px",
                "borderRadius": "6px",
                "fontSize": "12px",
                "cursor": "pointer",
                "marginLeft": "6px",
            },
        )

    is_selected = str(ep_id) == str(selected_episode or "")

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Button(
                                f"Episode: {ep_id}",
                                id={"type": "pnp-check-open-video-title", "episode_id": ep_id},
                                n_clicks=0,
                                style={
                                    "border": "none",
                                    "background": "transparent",
                                    "color": "#2563eb",
                                    "padding": "0",
                                    "fontSize": "13px",
                                    "fontWeight": "600",
                                    "cursor": "pointer",
                                },
                            ),
                            html.Span(
                                f"task_id: {task_id} ｜ 右手: {r_count}次 ｜ 左手: {l_count}次",
                                style={"fontSize": "12px", "color": "#6b7280", "marginLeft": "8px"},
                            ),
                        ],
                        style={"display": "flex", "alignItems": "center", "flexWrap": "wrap", "gap": "4px"},
                    ),
                    html.Div(
                        [
                            _status_btn("pass", "合格", PNP_STATUS_COLOR["pass"]),
                            _status_btn("multi_pick", "多次抓取", PNP_STATUS_COLOR["multi_pick"]),
                            _status_btn("fail_pick", "抓取不合格", PNP_STATUS_COLOR["fail_pick"]),
                            _status_btn("invalid", "无效", PNP_STATUS_COLOR["invalid"]),
                        ],
                        style={"display": "flex", "alignItems": "center", "marginLeft": "8px"},
                    ),
                ],
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "gap": "8px"},
            )
        ],
        style={
            "border": "1px solid #3b82f6" if is_selected else "1px solid #e5e7eb",
            "borderRadius": "8px",
            "padding": "10px 12px",
            "background": "#e0f2fe" if is_selected else "#fff",
            "marginBottom": "8px",
        },
    )


def _build_sidebar_row(item: dict, group_status: str):
    ep_id = str(item.get("episode_id", ""))
    task_id = str(item.get("task_id", ""))
    r_count = item.get("r_count", 0)
    l_count = item.get("l_count", 0)
    label_color = PNP_STATUS_COLOR.get(group_status, "#6b7280")

    return html.Div(
        [
            html.Button(
                f"Episode {ep_id}",
                id={"type": "pnp-check-open-video-btn", "episode_id": ep_id},
                n_clicks=0,
                style={
                    "border": "none",
                    "background": "transparent",
                    "color": "#2563eb",
                    "padding": "0",
                    "fontSize": "12px",
                    "cursor": "pointer",
                    "textAlign": "left",
                },
            ),
            html.Div(f"task_id: {task_id} ｜ R:{r_count} / L:{l_count}", style={"fontSize": "11px", "color": "#6b7280"}),
            html.Div(
                [
                    html.Span(
                        PNP_STATUS_LABEL.get(group_status, ""),
                        style={
                            "fontSize": "11px",
                            "color": label_color,
                            "border": f"1px solid {label_color}",
                            "borderRadius": "10px",
                            "padding": "1px 6px",
                            "marginRight": "6px",
                        },
                    ),
                    html.Button(
                        "撤销",
                        id={"type": "pnp-check-undo-btn", "episode_id": ep_id},
                        n_clicks=0,
                        style={
                            "border": "1px solid #d1d5db",
                            "background": "#fff",
                            "color": "#374151",
                            "padding": "1px 6px",
                            "borderRadius": "6px",
                            "fontSize": "11px",
                            "cursor": "pointer",
                        },
                    ),
                ],
                style={"display": "flex", "alignItems": "center", "marginTop": "4px"},
            ),
        ],
        style={
            "padding": "8px 10px",
            "border": "1px solid #eef2f7",
            "borderRadius": "6px",
            "background": "#fff",
            "marginBottom": "6px",
        },
    )


def register_callbacks(app):

    # 1. 加载所有 PnP task_id
    @app.callback(
        Output("pnp-check-batch-dropdown", "options"),
        Input("pnp-check-batch-dropdown", "search_value"),
    )
    def load_pnp_tasks(search_value):
        try:
            sql = """
                SELECT DISTINCT task_id
                FROM pnp_batches
                WHERE task_id IS NOT NULL
                ORDER BY task_id DESC
            """
            params = None
            if search_value:
                sql = """
                    SELECT DISTINCT task_id
                    FROM pnp_batches
                    WHERE task_id IS NOT NULL
                      AND CAST(task_id AS TEXT) ILIKE %(search)s
                    ORDER BY task_id DESC
                """
                params = {"search": f"%{search_value}%"}
            df = query_pnp_df(sql, params)
            if df.empty:
                return []
            return [{"label": str(t), "value": str(t)} for t in df["task_id"] if pd.notnull(t)]
        except Exception as e:
            logging.error(f"Failed to load pnp tasks: {e}")
            return []

    # 2. 点击“加载”按钮，获取当前批次的信息，解析 PnP 此数，动态设置 Slider Maximum，并重置 page 和 visible ids 等
    @app.callback(
        [
            Output("pnp-check-query-data", "data"),
            Output("pnp-check-query-message", "children"),
            Output("pnp-check-limits", "data"),
            Output("pnp-check-right-filter", "options"),
            Output("pnp-check-right-filter", "value"),
            Output("pnp-check-right-chart", "figure"),
            Output("pnp-check-left-filter", "options"),
            Output("pnp-check-left-filter", "value"),
            Output("pnp-check-left-chart", "figure"),
            Output("pnp-check-right-duration-filter", "options"),
            Output("pnp-check-right-duration-filter", "value"),
            Output("pnp-check-right-duration-chart", "figure"),
            Output("pnp-check-left-duration-filter", "options"),
            Output("pnp-check-left-duration-filter", "value"),
            Output("pnp-check-left-duration-chart", "figure"),
            Output("pnp-check-right-axis-filter", "options"),
            Output("pnp-check-right-axis-filter", "value"),
            Output("pnp-check-right-axis-chart", "figure"),
            Output("pnp-check-left-axis-filter", "options"),
            Output("pnp-check-left-axis-filter", "value"),
            Output("pnp-check-left-axis-chart", "figure"),
            Output("pnp-check-applied-filters", "data"),
        ],
        Input("pnp-check-load-batch-btn", "n_clicks"),
        State("pnp-check-batch-dropdown", "value")
    )
    def load_batch_data(n_clicks, batch_id):
        import plotly.graph_objects as go
        empty_fig = go.Figure()
        empty_fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=140, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        empty_small_fig = go.Figure()
        empty_small_fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=120, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

        selected_task_id = str(batch_id) if batch_id is not None else None

        if not n_clicks or not selected_task_id:
            return (
                no_update,
                "请选择一个 TASK_ID",
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
            )

        # fetch data for selected task_id (latest result per episode)
        sql = """
            SELECT DISTINCT ON (s.episode_id)
                   s.episode_id, s.right_pnp_result, s.left_pnp_result, b.task_id
            FROM pnp_streams s
            JOIN pnp_batches b ON s.batch_id = b.uniq_id
            WHERE CAST(b.task_id AS TEXT) = %s
            ORDER BY s.episode_id, s.checked_at DESC
        """
        try:
            df = query_pnp_df(sql, (selected_task_id,))
        except Exception as e:
            return [], f"查询异常: {e}", {}, [], [], empty_fig, [], [], empty_fig, [], [], empty_small_fig, [], [], empty_small_fig, [], [], empty_small_fig, [], [], empty_small_fig, {"r_count": [], "l_count": [], "r_duration": [], "l_duration": [], "r_axis": [], "l_axis": []}
        
        if df.empty:
            return [], "该任务下无 Episode 数据", {}, [], [], empty_fig, [], [], empty_fig, [], [], empty_small_fig, [], [], empty_small_fig, [], [], empty_small_fig, [], [], empty_small_fig, {"r_count": [], "l_count": [], "r_duration": [], "l_duration": [], "r_axis": [], "l_axis": []}

        episode_duration_map = {}
        episode_ids = tuple(str(x) for x in df["episode_id"].tolist())
        if episode_ids:
            try:
                meta_df = query_df(
                    "SELECT id, trajectory_duration, trajectory_start, trajectory_end FROM episodes WHERE id IN %s",
                    (episode_ids,),
                )
                for _, mrow in meta_df.iterrows():
                    ep_id = str(mrow.get("id"))
                    total_sec = None
                    td = mrow.get("trajectory_duration")
                    try:
                        if td is not None:
                            td_float = float(td)
                            if td_float > 0:
                                total_sec = td_float
                    except Exception:
                        total_sec = None
                    if (total_sec is None or total_sec <= 0) and pd.notnull(mrow.get("trajectory_start")) and pd.notnull(mrow.get("trajectory_end")):
                        try:
                            total_sec = float((mrow["trajectory_end"] - mrow["trajectory_start"]).total_seconds())
                        except Exception:
                            total_sec = None
                    if total_sec is not None and total_sec > 0:
                        episode_duration_map[ep_id] = total_sec
            except Exception:
                episode_duration_map = {}

        parsed_data = []
        max_r, max_l = 0, 0
        r_counts = {}
        l_counts = {}
        r_axis_points = []
        l_axis_points = []
        for _, row in df.iterrows():
            ep_id = str(row["episode_id"])
            row_task_id = str(row["task_id"])

            r_val = row['right_pnp_result']
            l_val = row['left_pnp_result']
            r_res = _normalize_segments(r_val)
            l_res = _normalize_segments(l_val)
            r_count = len(r_res)
            l_count = len(l_res)
            r_duration_abs = _calc_total_duration(r_res)
            l_duration_abs = _calc_total_duration(l_res)
            max_end = max(_calc_last_end(r_res), _calc_last_end(l_res))
            total_sec = episode_duration_map.get(ep_id)
            if total_sec is None or total_sec <= 0:
                total_sec = max_end if max_end > 0 else None

            if total_sec and total_sec > 0:
                r_duration = r_duration_abs / float(total_sec)
                l_duration = l_duration_abs / float(total_sec)
            else:
                r_duration = 0.0
                l_duration = 0.0

            def _axis_ratios(segments, duration_sec):
                if not duration_sec or duration_sec <= 0:
                    return []
                ratios = []
                for st, ed in segments:
                    center = (float(st) + float(ed)) / 2.0
                    ratio = center / float(duration_sec)
                    ratio = max(0.0, min(1.0, ratio))
                    ratios.append(ratio)
                return ratios

            r_ratios = _axis_ratios(r_res, total_sec)
            l_ratios = _axis_ratios(l_res, total_sec)
            r_axis_score = sum(r_ratios) / len(r_ratios) if r_ratios else 0.0
            l_axis_score = sum(l_ratios) / len(l_ratios) if l_ratios else 0.0
            r_axis_points.extend(r_ratios)
            l_axis_points.extend(l_ratios)

            if r_count > max_r:
                max_r = r_count
            if l_count > max_l:
                max_l = l_count

            r_counts[r_count] = r_counts.get(r_count, 0) + 1
            l_counts[l_count] = l_counts.get(l_count, 0) + 1

            parsed_data.append({
                "episode_id": ep_id,
                "task_id": row_task_id,
                "r_count": r_count,
                "l_count": l_count,
                "r_duration": r_duration,
                "l_duration": l_duration,
                "r_axis_score": r_axis_score,
                "l_axis_score": l_axis_score,
                "right_pnp_result": r_res,
                "left_pnp_result": l_res,
                "task_filter_id": selected_task_id
            })

        r_duration_low, r_duration_high = _calc_iqr_bounds([x["r_duration"] for x in parsed_data])
        l_duration_low, l_duration_high = _calc_iqr_bounds([x["l_duration"] for x in parsed_data])
        r_axis_low, r_axis_high = _calc_sigma_bounds(r_axis_points, sigma_k=3.0)
        l_axis_low, l_axis_high = _calc_sigma_bounds(l_axis_points, sigma_k=3.0)

        r_duration_counts = {k: 0 for k in OUTLIER_ORDER}
        l_duration_counts = {k: 0 for k in OUTLIER_ORDER}
        r_axis_counts = {k: 0 for k in OUTLIER_ORDER}
        l_axis_counts = {k: 0 for k in OUTLIER_ORDER}

        for item in parsed_data:
            r_d_tag = _classify_outlier(item["r_duration"], r_duration_low, r_duration_high)
            l_d_tag = _classify_outlier(item["l_duration"], l_duration_low, l_duration_high)
            r_a_tag = _classify_outlier(item["r_axis_score"], r_axis_low, r_axis_high)
            l_a_tag = _classify_outlier(item["l_axis_score"], l_axis_low, l_axis_high)
            item["r_duration_tag"] = r_d_tag
            item["l_duration_tag"] = l_d_tag
            item["r_axis_tag"] = r_a_tag
            item["l_axis_tag"] = l_a_tag
            r_duration_counts[r_d_tag] += 1
            l_duration_counts[l_d_tag] += 1
            r_axis_counts[r_a_tag] += 1
            l_axis_counts[l_a_tag] += 1

        def _make_opts(c_dict):
            return [{"label": f"{k}次 ({c_dict[k]}个)", "value": k} for k in sorted(c_dict.keys())]

        def _make_outlier_opts(c_dict):
            opts = []
            for key in OUTLIER_ORDER:
                count = int(c_dict.get(key, 0))
                if count > 0:
                    opts.append({"label": f"{OUTLIER_LABEL[key]} ({count}个)", "value": key})
            return opts
            
        def _make_fig(c_dict, title, color):
            x_vals = sorted(c_dict.keys())
            y_vals = [c_dict[x] for x in x_vals]
            fig = go.Figure(data=[go.Bar(
                x=[str(x) for x in x_vals], 
                y=y_vals, 
                marker_color=color,
                text=y_vals,
                textposition='auto',
                hovertemplate=f"{title}: %{{x}}次<br>数量: %{{y}}个<extra></extra>"
            )])
            fig.update_layout(
                margin=dict(l=20, r=20, t=10, b=20),
                xaxis=dict(type='category', title=""),
                yaxis=dict(visible=False),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=140
            )
            return fig

        def _make_outlier_fig(c_dict, title, color):
            x_vals = [k for k in OUTLIER_ORDER if int(c_dict.get(k, 0)) > 0]
            y_vals = [int(c_dict.get(k, 0)) for k in x_vals]
            labels = [OUTLIER_LABEL[k] for k in x_vals]
            if not x_vals:
                return empty_small_fig
            fig = go.Figure(data=[go.Bar(
                x=labels,
                y=y_vals,
                marker_color=color,
                text=y_vals,
                textposition='auto',
                hovertemplate=f"{title}: %{{x}}<br>数量: %{{y}}个<extra></extra>"
            )])
            fig.update_layout(
                margin=dict(l=20, r=20, t=10, b=20),
                xaxis=dict(type='category', title=""),
                yaxis=dict(visible=False),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=120
            )
            return fig

        r_opts = _make_opts(r_counts)
        l_opts = _make_opts(l_counts)
        r_duration_opts = _make_outlier_opts(r_duration_counts)
        l_duration_opts = _make_outlier_opts(l_duration_counts)
        r_axis_opts = _make_outlier_opts(r_axis_counts)
        l_axis_opts = _make_outlier_opts(l_axis_counts)

        r_fig = _make_fig(r_counts, "右手", "#3b82f6") if r_counts else empty_fig
        l_fig = _make_fig(l_counts, "左手", "#10b981") if l_counts else empty_fig
        r_duration_fig = _make_outlier_fig(r_duration_counts, "右手时长离群", "#2563eb")
        l_duration_fig = _make_outlier_fig(l_duration_counts, "左手时长离群", "#059669")
        r_axis_fig = _make_outlier_fig(r_axis_counts, "右手时间轴离群", "#4f46e5")
        l_axis_fig = _make_outlier_fig(l_axis_counts, "左手时间轴离群", "#0d9488")

        task_info = "未知"
        try:
            task_df = query_df("SELECT descriptions FROM tasks WHERE id = %(id)s", {"id": selected_task_id})
            if not task_df.empty:
                desc_val = task_df.iloc[0]["descriptions"]
                if isinstance(desc_val, str):
                    desc_val = json.loads(desc_val)
                if isinstance(desc_val, dict):
                    task_info = desc_val.get("zh", str(desc_val))
                else:
                    task_info = str(desc_val)
        except Exception as e:
            task_info = f"获取失败: {e}"

        # 统计：已执行PnP质检数量（含失败）和总数量
        success_count = len({str(x) for x in df["episode_id"].tolist()})
        failed_count = 0
        total_count = 0
        executed_including_failed = 0
        try:
            failed_df = query_pnp_df(
                """
                SELECT COUNT(DISTINCT f.episode_id) AS failed_count
                FROM pnp_failures f
                JOIN pnp_batches b ON f.batch_id = b.uniq_id
                WHERE CAST(b.task_id AS TEXT) = %s
                """,
                (selected_task_id,),
            )
            if not failed_df.empty:
                failed_count = int(failed_df.iloc[0].get("failed_count") or 0)
        except Exception:
            failed_count = 0

        try:
            executed_df = query_pnp_df(
                """
                SELECT COUNT(DISTINCT episode_id) AS executed_count
                FROM (
                    SELECT s.episode_id
                    FROM pnp_streams s
                    JOIN pnp_batches b ON s.batch_id = b.uniq_id
                    WHERE CAST(b.task_id AS TEXT) = %s
                    UNION
                    SELECT f.episode_id
                    FROM pnp_failures f
                    JOIN pnp_batches b2 ON f.batch_id = b2.uniq_id
                    WHERE CAST(b2.task_id AS TEXT) = %s
                ) t
                """,
                (selected_task_id, selected_task_id),
            )
            if not executed_df.empty:
                executed_including_failed = int(executed_df.iloc[0].get("executed_count") or 0)
        except Exception:
            executed_including_failed = success_count + failed_count

        try:
            total_df = query_df(
                """
                SELECT COUNT(*) AS total_count
                FROM episodes
                WHERE task_id = %(task_id)s
                  AND trajectory_duration IS NOT NULL
                  AND trajectory_duration > 0
                """,
                {"task_id": selected_task_id},
            )
            if not total_df.empty:
                total_count = int(total_df.iloc[0].get("total_count") or 0)
        except Exception:
            total_count = 0

        msg_element = html.Div([
            html.Div(f"共加载 {len(parsed_data)} 条 Episode 数据。当前任务最高 右手PnP次数: {max_r}, 左手PnP次数: {max_l}。"),
            html.Div(f"已执行PnP质检（含失败）：{executed_including_failed}/{total_count}（成功: {success_count}, 失败: {failed_count}）", style={"marginTop": "2px"}),
            html.Div(f"任务内容：{task_info}", style={"marginTop": "5px"})
        ])

        return (
            parsed_data,
            msg_element,
            {"right_max": max_r, "left_max": max_l},
            r_opts, [], r_fig,
            l_opts, [], l_fig,
            r_duration_opts, [], r_duration_fig,
            l_duration_opts, [], l_duration_fig,
            r_axis_opts, [], r_axis_fig,
            l_axis_opts, [], l_axis_fig,
            {"r_count": [], "l_count": [], "r_duration": [], "l_duration": [], "r_axis": [], "l_axis": []},
        )

    @app.callback(
        [
            Output("pnp-check-filter-modal", "is_open"),
            Output("pnp-check-applied-filters", "data", allow_duplicate=True),
        ],
        [
            Input("pnp-check-open-filter-modal-btn", "n_clicks"),
            Input("pnp-check-filter-modal-cancel-btn", "n_clicks"),
            Input("pnp-check-filter-modal-confirm-btn", "n_clicks"),
        ],
        [
            State("pnp-check-filter-modal", "is_open"),
            State("pnp-check-right-filter", "value"),
            State("pnp-check-left-filter", "value"),
            State("pnp-check-right-duration-filter", "value"),
            State("pnp-check-left-duration-filter", "value"),
            State("pnp-check-right-axis-filter", "value"),
            State("pnp-check-left-axis-filter", "value"),
        ],
        prevent_initial_call=True,
    )
    def toggle_filter_modal(open_clicks, cancel_clicks, confirm_clicks, is_open, r_vis, l_vis, r_d_vis, l_d_vis, r_a_vis, l_a_vis):
        trigger = ctx.triggered_id
        if trigger == "pnp-check-open-filter-modal-btn":
            return True, no_update
        if trigger == "pnp-check-filter-modal-cancel-btn":
            return False, no_update
        if trigger == "pnp-check-filter-modal-confirm-btn":
            applied = {
                "r_count": r_vis or [],
                "l_count": l_vis or [],
                "r_duration": r_d_vis or [],
                "l_duration": l_d_vis or [],
                "r_axis": r_a_vis or [],
                "l_axis": l_a_vis or [],
            }
            return False, applied
        return is_open, no_update

    # 3. 核心表格渲染与分页
    @app.callback(
        [
            Output("pnp-check-table-container", "children"),
            Output("pnp-check-visible-ids", "data"),
            Output("pnp-check-selected-summary", "children"),
            Output("pnp-check-toggle-checked-btn", "children"),
            Output("pnp-check-toggle-checked-btn", "outline"),
            Output("pnp-check-page", "data"),
        ],
        [
            Input("pnp-check-query-data", "data"),
            Input("pnp-check-applied-filters", "data"),
            Input("pnp-check-submitted", "data"),
            Input("pnp-check-show-checked", "data"),
            Input("pnp-check-load-more-btn", "n_clicks"),
            Input("pnp-check-selected-video", "data"),
            Input("pnp-check-episode-search", "value"),
        ],
        [
            State("pnp-check-row-status", "data"),
            State("pnp-check-page", "data"),
        ],
    )
    def update_table(all_data, applied_filters, submitted, show_checked, load_more, selected_episode, search_value, row_status, page):
        submitted = submitted or {"pass": [], "multi_pick": [], "fail_pick": [], "invalid": []}
        show_checked = bool(show_checked)
        applied_filters = applied_filters or {}
        search_value = str(search_value or "").strip()
        r_vis = applied_filters.get("r_count", []) or []
        l_vis = applied_filters.get("l_count", []) or []
        r_d_vis = applied_filters.get("r_duration", []) or []
        l_d_vis = applied_filters.get("l_duration", []) or []
        r_a_vis = applied_filters.get("r_axis", []) or []
        l_a_vis = applied_filters.get("l_axis", []) or []

        trigger = ctx.triggered_id
        if trigger == "pnp-check-load-more-btn":
            page = (page or 1) + 1
        elif trigger in [
            "pnp-check-query-data",
            "pnp-check-applied-filters",
            "pnp-check-show-checked",
            "pnp-check-submitted",
            "pnp-check-episode-search",
        ]:
            page = 1
        else:
            page = page or 1

        submitted_ids = set()
        for v in submitted.values():
            if isinstance(v, list):
                for item in v:
                    submitted_ids.add(str(item.get("episode_id")))

        all_data = all_data or []
        all_ids = [str(r.get("episode_id")) for r in all_data]

        # 查询已检数据
        checked_map = {}
        if all_ids:
            try:
                checked_map = query_checked_pnp_episodes(all_ids)
            except Exception:
                checked_map = {}

        checked_count = len(checked_map)
        total_count = len(all_data)
        btn_label = f"查看已检数据：{checked_count}/{total_count}"

        MAX_RENDER = page * 20

        # Filter limits
        r_vis_set = set(r_vis) if r_vis else None
        l_vis_set = set(l_vis) if l_vis else None
        r_d_vis_set = set(r_d_vis) if r_d_vis else None
        l_d_vis_set = set(l_d_vis) if l_d_vis else None
        r_a_vis_set = set(r_a_vis) if r_a_vis else None
        l_a_vis_set = set(l_a_vis) if l_a_vis else None

        def _row_match_filters(row):
            ep_id = str(row.get("episode_id", ""))
            rc = int(row.get("r_count", 0))
            lc = int(row.get("l_count", 0))
            r_d_tag = str(row.get("r_duration_tag", "none"))
            l_d_tag = str(row.get("l_duration_tag", "none"))
            r_a_tag = str(row.get("r_axis_tag", "none"))
            l_a_tag = str(row.get("l_axis_tag", "none"))
            search_ok = (not search_value) or (search_value.lower() in ep_id.lower())
            if not search_ok:
                return False

            filter_matches = []
            if r_vis_set is not None:
                filter_matches.append(rc in r_vis_set)
            if l_vis_set is not None:
                filter_matches.append(lc in l_vis_set)
            if r_d_vis_set is not None:
                filter_matches.append(r_d_tag in r_d_vis_set)
            if l_d_vis_set is not None:
                filter_matches.append(l_d_tag in l_d_vis_set)
            if r_a_vis_set is not None:
                filter_matches.append(r_a_tag in r_a_vis_set)
            if l_a_vis_set is not None:
                filter_matches.append(l_a_tag in l_a_vis_set)

            if not filter_matches:
                return True
            return any(filter_matches)

        if not show_checked:
            # Show unchecked & unsubmitted
            visible_rows = []
            for r in all_data:
                if _row_match_filters(r):
                    ep_id = str(r.get("episode_id"))
                    if ep_id not in submitted_ids and ep_id not in checked_map:
                        visible_rows.append(r)

            visible_ids = [str(x.get("episode_id")) for x in visible_rows]

            if not visible_rows:
                table_ui = html.Div(
                    "当前范围无未提交数据",
                    style={"padding": "20px", "textAlign": "center", "color": "#9ca3af"},
                )
            else:
                shown = visible_rows[:MAX_RENDER]
                cards = [_build_pnp_card(r, row_status or {}, selected_episode) for r in shown]
                if len(visible_rows) > MAX_RENDER:
                    cards.append(html.Div("往下滚动加载更多...", style={"textAlign": "center", "color": "#6b7280", "padding": "10px", "fontSize": "12px", "marginTop": "10px"}))
                table_ui = html.Div(cards)

            sel_r_str = ",".join(map(str, sorted(r_vis_set))) if r_vis_set else "全部"
            sel_l_str = ",".join(map(str, sorted(l_vis_set))) if l_vis_set else "全部"
            sel_r_d = ",".join(OUTLIER_LABEL.get(x, x) for x in sorted(r_d_vis_set)) if r_d_vis_set else "全部"
            sel_l_d = ",".join(OUTLIER_LABEL.get(x, x) for x in sorted(l_d_vis_set)) if l_d_vis_set else "全部"
            sel_r_a = ",".join(OUTLIER_LABEL.get(x, x) for x in sorted(r_a_vis_set)) if r_a_vis_set else "全部"
            sel_l_a = ",".join(OUTLIER_LABEL.get(x, x) for x in sorted(l_a_vis_set)) if l_a_vis_set else "全部"
            search_text = search_value if search_value else "全部"
            summary = (
                f"当前范围内包含 {len(visible_rows)} 条未检测数据"
                f"（ID搜索: {search_text}；"
                f"次数: 右{sel_r_str}/左{sel_l_str}；"
                f"时长离群: 右{sel_r_d}/左{sel_l_d}；"
                f"时间轴离群: 右{sel_r_a}/左{sel_l_a}；"
                f"筛选关系: OR）。"
            )
            return table_ui, visible_ids, summary, btn_label, True, page

        else:
            # Show checked records
            checked_rows = [r for r in all_data if str(r.get("episode_id")) in checked_map and _row_match_filters(r)]
            
            if not checked_rows:
                table_ui = html.Div(
                    "没有已检测的数据",
                    style={"padding": "20px", "textAlign": "center", "color": "#9ca3af"},
                )
            else:
                shown = checked_rows[:MAX_RENDER]
                cards = [_build_checked_card(r, checked_map.get(str(r.get("episode_id")), "pass"), selected_episode) for r in shown]
                if len(checked_rows) > MAX_RENDER:
                    cards.append(html.Div("往下滚动加载更多...", style={"textAlign": "center", "color": "#6b7280", "padding": "10px", "fontSize": "12px", "marginTop": "10px"}))
                table_ui = html.Div(cards)

            if search_value:
                summary = f"当前任务中包含 {len(checked_rows)} 条已检测数据（ID搜索: {search_value}）。"
            else:
                summary = f"当前任务中包含 {len(checked_rows)} 条已检测数据。"
            return table_ui, [], summary, btn_label, False, page

    # 4. 点击卡片上的按钮播放对应视频及渲染 PnP 时间轴
    @app.callback(
        Output("pnp-check-selected-video", "data", allow_duplicate=True),
        [
            Input("pnp-check-query-data", "data"),
            Input("pnp-check-applied-filters", "data"),
            Input("pnp-check-show-checked", "data"),
            Input("pnp-check-episode-search", "value"),
        ],
        prevent_initial_call=True,
    )
    def reset_selected_video_on_dataset_change(_all_data, _filters, _show_checked, _search_value):
        # 数据集或筛选条件变化时，清空已选视频，避免视频/时间轴与当前卡片列表错位
        return None

    @app.callback(
        Output("pnp-check-selected-video", "data"),
        Input({"type": "pnp-check-open-video-btn", "episode_id": ALL}, "n_clicks"),
        Input({"type": "pnp-check-open-video-title", "episode_id": ALL}, "n_clicks"),
        prevent_initial_call=True
    )
    def update_selected_video_data(btn_clicks, title_clicks):
        if not ctx.triggered:
            return no_update
            
        trigger_val = ctx.triggered[0].get("value")
        if not trigger_val:
            return no_update

        trigger_id = ctx.triggered_id
        if not isinstance(trigger_id, dict):
            return no_update
        episode_id = trigger_id.get("episode_id")
        if not episode_id:
            return no_update
        return episode_id

    @app.callback(
        [
            Output("pnp-check-video-container", "children"),
            Output("pnp-check-timeline-container", "children")
        ],
        Input("pnp-check-selected-video", "data"),
        State("pnp-check-query-data", "data"),
        prevent_initial_call=True
    )
    def render_video_and_timeline(episode_id, all_data):
        if not episode_id:
            return (
                html.Div(
                    "点击下方数据卡片播放对应视频",
                    style={
                        "height": "360px",
                        "display": "flex",
                        "alignItems": "center",
                        "justifyContent": "center",
                        "backgroundColor": "#000",
                        "color": "#9ca3af",
                        "fontSize": "16px",
                        "borderRadius": "8px",
                    },
                ),
                html.Div(
                    "暂无时间轴数据",
                    style={
                        "height": "56px",
                        "display": "flex",
                        "alignItems": "center",
                        "justifyContent": "center",
                        "backgroundColor": "#f9fafb",
                        "color": "#9ca3af",
                        "fontSize": "14px",
                        "border": "1px dashed #e5e7eb",
                        "borderRadius": "6px",
                    },
                ),
            )
            
        sql = """
            SELECT s.file_path
            FROM streams s
            WHERE s.episode_id = %(episode_id)s
              AND s.stream_name = 'rgb'
            LIMIT 1
        """
        try:
            df = query_df(sql, {"episode_id": episode_id})
            if df.empty:
                video_elem = html.Div("未找到视频数据", style={"color": "red"})
            else:
                file_path = str(df.iloc[0]["file_path"])
                video_url = get_video_url(file_path)
                if video_url:
                    source_key = str(file_path or video_url)
                    if video_url.startswith("http"):
                        separator = "&" if "?" in video_url else "?"
                        final_src = f"{video_url}{separator}pnp_src_key={quote(source_key, safe='')}"
                    else:
                        final_src = f"/pnp_video?path={quote(video_url, safe='')}&pnp_src_key={quote(source_key, safe='')}"

                    video_elem = html.Div(
                        key=episode_id,
                        style={"width": "100%", "display": "flex", "justifyContent": "center"},
                        children=[
                            html.Video(
                                id="pnp-check-video-player",
                                autoPlay=True,
                                src=final_src,
                                controls=True,
                                style={"width": "100%", "backgroundColor": "#000", "maxHeight": "400px"}
                            )
                        ]
                    )
                else:
                    video_elem = html.Div("视频解析失败", style={"color": "red"})
        except Exception as e:
            video_elem = html.Div(f"视频查询异常 {e}", style={"color": "red"})

        # PnP 时间轴渲染
        r_res = []
        l_res = []
        all_data = all_data or []
        for r in all_data:
            if str(r.get("episode_id")) == str(episode_id):
                r_res = r.get("right_pnp_result", [])
                l_res = r.get("left_pnp_result", [])
                break

        timeline_elem = html.Div(
            id="pnp-check-custom-timeline-wrapper",
            **{"data-right": json.dumps(r_res), "data-left": json.dumps(l_res)},
            style={
                "position": "relative", "width": "100%", "height": "56px", 
                "backgroundColor": "#f9fafb", "borderRadius": "6px", 
                "border": "1px solid #e5e7eb", "cursor": "pointer",
                "marginTop": "10px", "overflow": "hidden"
            },
            children=[
                html.Div("右手", style={"position": "absolute", "left": "5px", "top": "7px", "fontSize": "11px", "fontWeight": "600", "color": "#6b7280", "pointerEvents": "none", "zIndex": 10}),
                html.Div("左手", style={"position": "absolute", "left": "5px", "top": "31px", "fontSize": "11px", "fontWeight": "600", "color": "#6b7280", "pointerEvents": "none", "zIndex": 10}),
                html.Div(id="pnp-check-right-hand-tracks", style={"position": "absolute", "left": "0", "width": "100%", "top": "5px", "height": "18px", "pointerEvents": "none"}),
                html.Div(id="pnp-check-left-hand-tracks", style={"position": "absolute", "left": "0", "width": "100%", "top": "29px", "height": "18px", "pointerEvents": "none"}),
                html.Div(id="pnp-check-timeline-playhead", style={"position": "absolute", "top": "0", "bottom": "0", "left": "0%", "width": "2px", "backgroundColor": "#ef4444", "pointerEvents": "none", "zIndex": 20})
            ]
        )

        return video_elem, timeline_elem

    # clientside callback for video timeline sync
    app.clientside_callback(
        """function(video_children) {
            setTimeout(function() {
                var video = document.getElementById('pnp-check-video-player');
                var wrapper = document.getElementById('pnp-check-custom-timeline-wrapper');
                var playhead = document.getElementById('pnp-check-timeline-playhead');
                
                if (video && wrapper && playhead && !video.dataset.syncBound) {
                    video.dataset.syncBound = '1';
                    
                    var updatePlayhead = function() {
                        var duration = video.duration;
                        if (!Number.isFinite(duration) || duration <= 0) {
                            playhead.style.left = '0%';
                            return;
                        }
                        var pct = (video.currentTime / duration) * 100;
                        playhead.style.left = Math.min(100, Math.max(0, pct)) + '%';
                    };

                    var drawBlocks = function() {
                        var duration = video.duration;
                        var r_tracks = document.getElementById('pnp-check-right-hand-tracks');
                        var l_tracks = document.getElementById('pnp-check-left-hand-tracks');
                        if (r_tracks && l_tracks) {
                            if (!Number.isFinite(duration) || duration <= 0) {
                                r_tracks.innerHTML = '';
                                l_tracks.innerHTML = '';
                                updatePlayhead();
                                return;
                            }
                            try {
                                var r_data = JSON.parse(wrapper.dataset.right || "[]");
                                var l_data = JSON.parse(wrapper.dataset.left || "[]");
                                var clampPct = function(value) {
                                    return Math.min(100, Math.max(0, value));
                                };
                                var r_html = "";
                                for(var i=0; i<r_data.length; i++){
                                    var st_sec = Number(r_data[i][0]);
                                    var ed_sec = Number(r_data[i][1]);
                                    if (!Number.isFinite(st_sec) || !Number.isFinite(ed_sec)) { continue; }
                                    st_sec = Math.min(duration, Math.max(0, st_sec));
                                    ed_sec = Math.min(duration, Math.max(st_sec, ed_sec));
                                    var left_pct = clampPct((st_sec / duration) * 100);
                                    var width_pct = clampPct(((ed_sec - st_sec) / duration) * 100);
                                    r_html += "<div style='position:absolute; left:" + left_pct + "%; width:" + width_pct + "%; height:100%; background:rgba(59, 130, 246, 0.7); border-radius:3px;'></div>";
                                }
                                r_tracks.innerHTML = r_html;

                                var l_html = "";
                                for(var i=0; i<l_data.length; i++){
                                    var st_sec = Number(l_data[i][0]);
                                    var ed_sec = Number(l_data[i][1]);
                                    if (!Number.isFinite(st_sec) || !Number.isFinite(ed_sec)) { continue; }
                                    st_sec = Math.min(duration, Math.max(0, st_sec));
                                    ed_sec = Math.min(duration, Math.max(st_sec, ed_sec));
                                    var left_pct = clampPct((st_sec / duration) * 100);
                                    var width_pct = clampPct(((ed_sec - st_sec) / duration) * 100);
                                    l_html += "<div style='position:absolute; left:" + left_pct + "%; width:" + width_pct + "%; height:100%; background:rgba(16, 185, 129, 0.7); border-radius:3px;'></div>";
                                }
                                l_tracks.innerHTML = l_html;
                            } catch(e) { console.error("Parse data error", e); }
                        }
                    };

                    video.addEventListener('loadedmetadata', drawBlocks);
                    video.addEventListener('durationchange', drawBlocks);
                    if (video.readyState >= 1) { drawBlocks(); }
                    
                    video.addEventListener('timeupdate', updatePlayhead);
                    video.addEventListener('seeking', updatePlayhead);
                    video.addEventListener('seeked', function(){
                        updatePlayhead();
                        drawBlocks();
                    });
                    
                    wrapper.addEventListener('click', function(e){
                        var rect = wrapper.getBoundingClientRect();
                        var pct = (e.clientX - rect.left) / rect.width;
                        if(Number.isFinite(video.duration) && video.duration > 0) {
                            video.currentTime = Math.min(1, Math.max(0, pct)) * video.duration;
                        }
                    });
                }
            }, 300);
            return window.dash_clientside.no_update;
        }""",
        Output("pnp-check-video-container", "data-sync-bound"),
        Input("pnp-check-video-container", "children")
    )

    # 行按钮状态客户端同步样式
    app.clientside_callback(
        """
        function(status_map) {
            if (!status_map) return window.dash_clientside.no_update;
            var buttons = document.querySelectorAll('button[id*="pnp-check-row-status-btn"]');
            var colorMap = {
                'pass': '#059669',
                'multi_pick': '#d97706',
                'fail_pick': '#ef4444',
                'invalid': '#6b7280'
            };
            buttons.forEach(function(btn) {
                try {
                    var id_str = btn.id;
                    var id_obj = JSON.parse(id_str);
                    var ep_id = String(id_obj.episode_id);
                    var status = id_obj.status;
                    var color = colorMap[status] || '#000';
                    var isActive = (status_map[ep_id] === status);
                    btn.style.background = isActive ? color : '#fff';
                    btn.style.color = isActive ? '#fff' : color;
                } catch(e) {}
            });
            return window.dash_clientside.no_update;
        }
        """,
        Output("pnp-check-query-message", "data-dummy"),
        Input("pnp-check-row-status", "data"),
        prevent_initial_call=True,
    )

    # 处理客户端数据滚动到底部
    app.clientside_callback(
        """
        function(id_table) {
            var tableContainer = document.getElementById(id_table);
            if(tableContainer && !tableContainer.dataset.scrollBound) {
                tableContainer.dataset.scrollBound = '1';
                var isFetching = false;
                tableContainer.addEventListener('scroll', function() {
                    /* 如果内容还没填满容器，就不可能“滚动到底部加载更多” */
                    if (tableContainer.scrollHeight <= tableContainer.clientHeight + 5) {
                        return;
                    }
                    /* 当滚动到底部 */
                    if(tableContainer.scrollTop + tableContainer.clientHeight >= Math.floor(tableContainer.scrollHeight) - 10) {
                        if (!isFetching) {
                            isFetching = true;
                            var btn = document.getElementById('pnp-check-load-more-btn');
                            if (btn) {
                                btn.click();
                                setTimeout(function(){ isFetching = false; }, 800);
                            } else {
                                isFetching = false;
                            }
                        }
                    }
                });
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output("pnp-check-table-container", "data-scroll-bound"),
        Input("pnp-check-table-container", "id")
    )

    # 5. Sidebar logic handling (purely clientside logic converted to python to manage `pnp-check-submitted` dict)
    # Similar to duration_check, we'll implement serverside handling for buttons here for simplicity,
    # or follow it directly from duration_check: Handle "All pass" / "row status btn" clicks.
    
    @app.callback(
        Output("pnp-check-row-status", "data"),
        [
            Input("pnp-check-all-pass-btn", "n_clicks"),
            Input("pnp-check-all-multi-btn", "n_clicks"),
            Input("pnp-check-all-fail-btn", "n_clicks"),
            Input("pnp-check-all-invalid-btn", "n_clicks"),
            Input({"type": "pnp-check-row-status-btn", "episode_id": ALL, "status": ALL}, "n_clicks"),
            Input("pnp-check-query-data", "data")
        ],
        [
            State("pnp-check-row-status", "data"),
            State("pnp-check-visible-ids", "data"),
        ]
    )
    def update_row_status(ap_clk, am_clk, af_clk, ai_clk, row_clks, query_data, row_status, visible_ids):
        trigger = ctx.triggered_id
        if not trigger:
            return no_update

        row_status = row_status or {}

        if trigger == "pnp-check-query-data":
            # Just clean non-existent
            return {}

        visible_ids = set([str(x) for x in (visible_ids or [])])

        if trigger == "pnp-check-all-pass-btn":
            for eid in visible_ids: row_status[eid] = "pass"
        elif trigger == "pnp-check-all-multi-btn":
            for eid in visible_ids: row_status[eid] = "multi_pick"
        elif trigger == "pnp-check-all-fail-btn":
            for eid in visible_ids: row_status[eid] = "fail_pick"
        elif trigger == "pnp-check-all-invalid-btn":
            for eid in visible_ids: row_status[eid] = "invalid"
        elif isinstance(trigger, dict) and trigger.get("type") == "pnp-check-row-status-btn":
            ep_id = str(trigger.get("episode_id"))
            status = trigger.get("status")
            if ep_id in row_status and row_status[ep_id] == status:
                del row_status[ep_id]
            else:
                row_status[ep_id] = status

        return row_status

    
    # "Submit to Sidebar" logic
    @app.callback(
        [
            Output("pnp-check-submitted", "data"),
            Output("pnp-check-action-message", "children"),
            Output("pnp-check-row-status", "data", allow_duplicate=True),
        ],
        [
            Input("pnp-check-submit-btn", "n_clicks"),
            Input({"type": "pnp-check-undo-all-btn", "status": ALL}, "n_clicks"),
            Input({"type": "pnp-check-undo-btn", "episode_id": ALL}, "n_clicks"),
        ],
        [
            State("pnp-check-row-status", "data"),
            State("pnp-check-query-data", "data"),
            State("pnp-check-submitted", "data"),
        ],
        prevent_initial_call=True,
    )
    def handle_submit_and_undo(submit_clicks, undo_all_clicks, undo_clicks, row_status, query_data, submitted):
        trigger = ctx.triggered_id
        if not trigger:
            return no_update, no_update, no_update

        submitted = submitted or {"pass": [], "multi_pick": [], "fail_pick": [], "invalid": []}
        msg = ""

        if trigger == "pnp-check-submit-btn":
            if not row_status:
                return no_update, html.Span("无可提交的数据", style={"color": "red"}), no_update

            appended_count = 0
            all_dict = {str(r.get("episode_id")): r for r in (query_data or [])}

            for ep_id, status in row_status.items():
                if status not in submitted:
                    submitted[status] = []
                # Check exist
                exists = any(str(r.get("episode_id")) == ep_id for r in submitted[status])
                if not exists and ep_id in all_dict:
                    submitted[status].append(all_dict[ep_id])
                    appended_count += 1

            msg = html.Span(f"成功将 {appended_count} 条数据加入左侧面板", style={"color": "#10b981"})
            row_status = {}

        elif isinstance(trigger, dict):
            t_type = trigger.get("type")
            if t_type == "pnp-check-undo-all-btn":
                status = trigger.get("status")
                c = len(submitted.get(status, []))
                submitted[status] = []
                msg = html.Span(f"已撤销所有 {PNP_STATUS_LABEL.get(status, status)} 的待提交数据 ({c}条)", style={"color": "#6b7280"})
            elif t_type == "pnp-check-undo-btn":
                ep_id = str(trigger.get("episode_id"))
                for st in PNP_STATUS_ORDER:
                    initial_len = len(submitted.get(st, []))
                    submitted[st] = [r for r in submitted.get(st, []) if str(r.get("episode_id")) != ep_id]
                    if len(submitted[st]) < initial_len:
                        msg = html.Span(f"已撤销 Episode: {ep_id}", style={"color": "#6b7280"})
                        break

        return submitted, msg, row_status

    # Render sidebar
    @app.callback(
        [
            Output("pnp-check-sidebar-container", "children"),
            Output("pnp-check-sidebar-task-filter", "options")
        ],
        [
            Input("pnp-check-submitted", "data"),
            Input("pnp-check-sidebar-task-filter", "value")
        ]
    )
    def render_sidebar(submitted, filter_task_ids):
        submitted = submitted or {}
        filter_task_ids = set(filter_task_ids) if filter_task_ids else None
        
        all_task_ids = set()
        
        sections = []
        MAX_SHOW = 50
        for st in PNP_STATUS_ORDER:
            rows = submitted.get(st, [])
            for r in rows:
                tid = str(r.get("task_id", ""))
                if tid:
                    all_task_ids.add(tid)
                    
            if filter_task_ids:
                rows = [r for r in rows if str(r.get("task_id", "")) in filter_task_ids]
                
            total = len(rows)
            folder_title = f"{PNP_STATUS_LABEL.get(st, st)} ({total})"
            
            shown_rows = rows[:MAX_SHOW]
            row_children = [_build_sidebar_row(item, st) for item in shown_rows]
            if total > MAX_SHOW:
                row_children.append(
                    html.Div(f"… 仅展示前 {MAX_SHOW} 条，共 {total} 条", style={"fontSize": "11px", "color": "#9ca3af", "textAlign": "center", "padding": "4px"})
                )
            
            sections.append(
                html.Div(
                    [
                        html.Div(
                            [
                                html.Button("▸ " + folder_title, id={"type": "pnp-check-folder-toggle", "status": st}, n_clicks=0, className="pnp-check-folder-toggle-btn", style={"border": "none", "background": "transparent", "color": PNP_STATUS_COLOR.get(st, "#374151"), "fontWeight": "600", "fontSize": "13px", "padding": "0", "cursor": "pointer", "textAlign": "left", "flex": "1"}),
                                html.Button("全部撤销", id={"type": "pnp-check-undo-all-btn", "status": st}, n_clicks=0, style={"border": "1px solid #e5e7eb", "background": "#fff", "color": "#ef4444", "fontSize": "11px", "padding": "1px 8px", "borderRadius": "4px", "cursor": "pointer", "flexShrink": "0"}) if total > 0 else html.Span(),
                            ],
                            style={"marginBottom": "6px", "display": "flex", "alignItems": "center", "gap": "8px"}
                        ),
                        html.Div(row_children, className="dc-folder-content", style={"paddingLeft": "8px", "display": "none"})
                    ],
                    style={"border": "1px solid #e5e7eb", "borderRadius": "8px", "padding": "8px", "background": "#f9fafb", "marginBottom": "8px"}
                )
            )

        task_id_opts = [{"label": f"Task {tid}", "value": tid} for tid in sorted(all_task_ids)]

        if all(len(submitted.get(k, [])) == 0 for k in PNP_STATUS_ORDER):
            return html.Div("暂无已提交数据", style={"textAlign": "center", "padding": "24px 10px", "color": "#9ca3af", "fontSize": "13px"}), task_id_opts

        return html.Div(sections), task_id_opts

    # Sidebar open toggle
    app.clientside_callback(
        """
        function(n_clicks) {
            if (!dash_clientside.callback_context.triggered.length) return window.dash_clientside.no_update;
            var trigger_id_str = dash_clientside.callback_context.triggered[0].prop_id.split('.')[0];
            try {
                var trigger_data = JSON.parse(trigger_id_str);
                var btn = document.getElementById(trigger_id_str);
                if (btn) {
                    var container = btn.parentElement.parentElement;
                    var content = container.querySelector('.dc-folder-content');
                    if (content) {
                        if (content.style.display === 'none') {
                            content.style.display = 'block';
                            btn.innerText = btn.innerText.replace('▸', '▾');
                        } else {
                            content.style.display = 'none';
                            btn.innerText = btn.innerText.replace('▾', '▸');
                        }
                    }
                }
            } catch(e) {}
            return window.dash_clientside.no_update;
        }
        """,
        Output("pnp-check-folder-open", "data-dummy"),
        Input({"type": "pnp-check-folder-toggle", "status": ALL}, "n_clicks")
    )
    
    # Save to DB
    @app.callback(
        [Output("pnp-check-save-db-message", "children"), Output("pnp-check-submitted", "data", allow_duplicate=True)],
        Input("pnp-check-save-db-btn", "n_clicks"),
        State("pnp-check-submitted", "data"),
        prevent_initial_call=True
    )
    def save_pnp_to_db(n_clicks, submitted):
        if not n_clicks or not submitted:
            return no_update, no_update
        
        records = []
        for st, items in submitted.items():
            for item in items:
                records.append({
                    "episode_id": item.get("episode_id"),
                    "task_id": item.get("task_id"),
                    "label": st
                })
        
        if not records:
            return html.Div("没有需要保存的数据", style={"color": "red"}), no_update
            
        try:
            count = save_pnp_results(records)
            return html.Div(f"成功保存 {count} 条数据到 manual_pnp_results!", style={"color": "#10b981", "fontWeight": "bold"}), {"pass": [], "multi_pick": [], "fail_pick": [], "invalid": []}
        except Exception as e:
            return html.Div(f"保存失败: {e}", style={"color": "red"}), no_update

    # Toggle Checked / Unchecked Mode
    @app.callback(
        Output("pnp-check-show-checked", "data"),
        Input("pnp-check-toggle-checked-btn", "n_clicks"),
        State("pnp-check-show-checked", "data"),
        prevent_initial_call=True
    )
    def toggle_show_checked(n_clicks, current_state):
        if not n_clicks: return no_update
        return not current_state
