"""时长检测页面回调。

性能优化说明：
- 拖动时间轴 → 仅通过 clientside callback 在浏览器端过滤数据，不回服务端
- 表格渲染、侧边栏渲染均在 clientside callback 完成
- 只有「查询」和「打开视频」会触发服务端回调
- 侧边栏数据通过 localStorage 持久化在浏览器端
- 「保存结果到数据库」才会写入 result 库
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, ctx, dcc, html, ALL, no_update

from src.utils.source_db import query_df
from src.utils.data_parser import get_video_url
from src.utils.redis_cache import get_cache, set_cache
from src.utils.result_db import save_duration_results, query_checked_episodes

# ── 状态 / 标签常量 ──
DURATION_STATUS_ORDER = ["pass", "fast", "slow", "invalid"]
DURATION_STATUS_LABEL = {
    "pass": "合格",
    "fast": "过快",
    "slow": "过慢",
    "invalid": "无效",
}
DURATION_STATUS_COLOR = {
    "pass": "#059669",
    "fast": "#d97706",
    "slow": "#2563eb",
    "invalid": "#6b7280",
}


def _duration_sort_key(value):
    sval = str(value)
    if sval.isdigit():
        return 0, int(sval)
    return 1, sval


def _build_checked_card(row:dict,label:str):
    """构建已检测数据的只读卡片。"""
    episode_id = str(row.get("id", ""))
    task_id = str(row.get("task_id", ""))
    duration_val = row.get("trajectory_duration", 0)
    try:
        duration_text = f"{float(duration_val):.2f}s"
    except Exception:
        duration_text = "未知"

    start_val = row.get("trajectory_start")
    if pd.notnull(start_val):
        try:
            start_text = pd.to_datetime(start_val).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            start_text = str(start_val)
    else:
        start_text = "未知"

    label_text = DURATION_STATUS_LABEL.get(label, label)
    label_color = DURATION_STATUS_COLOR.get(label, "#6b7280")

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Button(
                                f"Episode: {episode_id}",
                                id={"type": "duration-check-open-video-title", "episode_id": episode_id},
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
                                f"task_id: {task_id} ｜ 时长: {duration_text} ｜ 开始: {start_text}",
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
            "border": f"1px solid {label_color}30",
            "borderLeft": f"4px solid {label_color}",
            "borderRadius": "8px",
            "padding": "10px 12px",
            "background": f"{label_color}08",
            "marginBottom": "8px",
        },
    )


def _build_duration_card(row:dict, status_map:dict):
    """构建单条数据的卡片（用于左侧表格区域）。"""
    episode_id = str(row.get("id", ""))
    task_id = str(row.get("task_id", ""))
    duration_val = row.get("trajectory_duration", 0)
    start_val = row.get("trajectory_start")
    try:
        duration_text = f"{float(duration_val):.2f}s"
    except Exception:
        duration_text = "未知"

    if pd.notnull(start_val):
        try:
            start_text = pd.to_datetime(start_val).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            start_text = str(start_val)
    else:
        start_text = "未知"

    current_status = status_map.get(episode_id)

    def _status_btn(key, label, color):
        is_active = current_status == key
        return html.Button(
            label,
            id={"type": "duration-check-row-status-btn", "episode_id": episode_id, "status": key},
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

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Button(
                                f"Episode: {episode_id}",
                                id={"type": "duration-check-open-video-title", "episode_id": episode_id},
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
                                f"task_id: {task_id} ｜ 时长: {duration_text} ｜ 开始: {start_text}",
                                style={"fontSize": "12px", "color": "#6b7280", "marginLeft": "8px"},
                            ),
                        ],
                        style={"display": "flex", "alignItems": "center", "flexWrap": "wrap", "gap": "4px"},
                    ),
                    html.Div(
                        [
                            _status_btn("pass", "合格", DURATION_STATUS_COLOR["pass"]),
                            _status_btn("fast", "过快", DURATION_STATUS_COLOR["fast"]),
                            _status_btn("slow", "过慢", DURATION_STATUS_COLOR["slow"]),
                            _status_btn("invalid", "无效", DURATION_STATUS_COLOR["invalid"]),
                        ],
                        style={"display": "flex", "alignItems": "center", "marginLeft": "8px"},
                    ),
                ],
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "gap": "8px"},
            )
        ],
        style={
            "border": "1px solid #e5e7eb",
            "borderRadius": "8px",
            "padding": "10px 12px",
            "background": "#fff",
            "marginBottom": "8px",
        },
    )


def _build_sidebar_row(item:dict, group_status:str):
    ep_id = str(item.get("id", ""))
    task_id = str(item.get("task_id", ""))
    duration_sec = item.get("duration_sec", 0)
    try:
        duration_text = f"{float(duration_sec):.2f}s"
    except Exception:
        duration_text = "未知"

    label_color = DURATION_STATUS_COLOR.get(group_status, "#6b7280")

    return html.Div(
        [
            html.Button(
                f"Episode {ep_id}",
                id={"type": "duration-check-open-video-btn", "episode_id": ep_id},
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
            html.Div(f"task_id: {task_id} ｜ 时长: {duration_text}", style={"fontSize": "11px", "color": "#6b7280"}),
            html.Div(
                [
                    html.Span(
                        DURATION_STATUS_LABEL.get(group_status, ""),
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
                        id={"type": "duration-check-undo-btn", "episode_id": ep_id},
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


def _build_submitted_sidebar(submitted_map):
    """构建侧边栏。
    
    - 每个分组默认折叠（仅显示标题行+数量）
    - 展开/折叠由 clientside JS 直接操控 CSS display，不回服务端
    - 每组最多展示 50 条，超出提示
    """
    safe_submitted = submitted_map if isinstance(submitted_map, dict) else {}
    MAX_SHOW = 50

    sections = []
    for status_key in DURATION_STATUS_ORDER:
        rows = safe_submitted.get(status_key, []) if isinstance(safe_submitted.get(status_key, []), list) else []
        rows_sorted = sorted(rows, key=lambda x: _duration_sort_key(x.get("id", "")))
        total = len(rows_sorted)
        folder_title = f"{DURATION_STATUS_LABEL.get(status_key, status_key)} ({total})"

        # 构建行内容（限制数量）
        shown_rows = rows_sorted[:MAX_SHOW]
        row_children = [_build_sidebar_row(item, status_key) for item in shown_rows]
        if total > MAX_SHOW:
            row_children.append(
                html.Div(
                    f"… 仅展示前 {MAX_SHOW} 条，共 {total} 条",
                    style={"fontSize": "11px", "color": "#9ca3af", "textAlign": "center", "padding": "4px"},
                )
            )

        sections.append(
            html.Div(
                [
                    html.Div(
                        [
                            html.Button(
                                "▸ " + folder_title,
                                id={"type": "duration-check-folder-toggle", "status": status_key},
                                n_clicks=0,
                                className="dc-folder-toggle-btn",
                                style={
                                    "border": "none",
                                    "background": "transparent",
                                    "color": DURATION_STATUS_COLOR.get(status_key, "#374151"),
                                    "fontWeight": "600",
                                    "fontSize": "13px",
                                    "padding": "0",
                                    "cursor": "pointer",
                                    "textAlign": "left",
                                    "flex": "1",
                                },
                            ),
                            html.Button(
                                "全部撤销",
                                id={"type": "duration-check-undo-all-btn", "status": status_key},
                                n_clicks=0,
                                style={
                                    "border": "1px solid #e5e7eb",
                                    "background": "#fff",
                                    "color": "#ef4444",
                                    "fontSize": "11px",
                                    "padding": "1px 8px",
                                    "borderRadius": "4px",
                                    "cursor": "pointer",
                                    "flexShrink": "0",
                                },
                            ) if total > 0 else html.Span(),
                        ],
                        style={"marginBottom": "6px", "display": "flex", "alignItems": "center", "gap": "8px"},
                    ),
                    # 默认折叠（display:none），由 clientside JS 切换
                    html.Div(
                        row_children,
                        className="dc-folder-content",
                        style={"paddingLeft": "8px", "display": "none"},
                    ),
                ],
                style={
                    "border": "1px solid #e5e7eb",
                    "borderRadius": "8px",
                    "padding": "8px",
                    "background": "#f9fafb",
                    "marginBottom": "8px",
                },
            )
        )

    if all(len((safe_submitted.get(k, []) if isinstance(safe_submitted.get(k, []), list) else [])) == 0 for k in DURATION_STATUS_ORDER):
        return html.Div(
            "暂无已提交数据",
            style={
                "textAlign": "center",
                "padding": "24px 10px",
                "color": "#9ca3af",
                "fontSize": "13px",
                "border": "1px dashed #e5e7eb",
                "borderRadius": "8px",
                "background": "#fff",
            },
        )

    return html.Div(sections)


def _build_duration_distribution_figure(df, selected_range=None):
    duration_series = pd.to_numeric(df["duration_sec"], errors="coerce")
    duration_series = duration_series[duration_series.notna()]
    if duration_series.empty:
        return None, None

    values = duration_series.to_numpy(dtype=float)
    min_v = float(np.min(values))
    max_v = float(np.max(values))
    if max_v <= min_v:
        max_v = min_v + 1.0

    q01 = float(np.quantile(values, 0.01))
    q99 = float(np.quantile(values, 0.99))
    clip_low = min(min_v, q01)
    clip_high = max(max_v, q99)
    clipped = np.clip(values, clip_low, clip_high)

    bin_count = int(min(60, max(18, np.sqrt(len(clipped)))))
    hist_counts, bin_edges = np.histogram(clipped, bins=bin_count)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    if selected_range and isinstance(selected_range, (list, tuple)) and len(selected_range) == 2:
        lo = float(selected_range[0])
        hi = float(selected_range[1])
    else:
        try:
            import scipy.stats as stats
            if len(values) > 2:
                # 使用 t 分布进行稳健估计 (获取自由度, 定位参数 loc, 尺度参数 scale)
                df_est, loc, scale = stats.t.fit(values)
                # 左侧红线使用 p1（第 1% 分位数）
                lo = float(np.quantile(values, 0.01))
                # 右侧仍然使用 t分布稳健估计的 3-sigma
                hi = float(loc + 3 * scale)
            else:
                mean = np.mean(values)
                std = np.std(values) if len(values) > 1 else 0
                lo = float(np.quantile(values, 0.01))
                hi = float(mean + 3 * std)
        except Exception:
            mean = np.mean(values)
            std = np.std(values) if len(values) > 1 else 0
            lo = float(np.quantile(values, 0.01))
            hi = float(mean + 3 * std)

    lo = float(np.clip(lo, min_v, max_v))
    hi = float(np.clip(hi, min_v, max_v))
    if hi <= lo:
        hi = min(max_v, lo + max((max_v - min_v) * 0.05, 0.1))

    slider_range = [lo, hi]

    x_coords = [float(bin_edges[0])] + centers.tolist() + [float(bin_edges[-1])]
    y_coords = [0.0] + hist_counts.tolist() + [0.0]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_coords,
            y=y_coords,
            mode="lines",
            line=dict(color="#3b82f6", width=2.2, shape="spline"),
            fill="tozeroy",
            fillcolor="rgba(59, 130, 246, 0.18)",
            hovertemplate="时长: %{x:.2f}s<br>条数: %{y}<extra></extra>",
            name="时长分布",
        )
    )
    # 添加透明定界点，强制让绘图区域（以及底部的rangeslider）至少包裹所有数据的最小最大值，防止边缘数据被卡出范围
    fig.add_trace(
        go.Scatter(
            x=[min_v, max_v],
            y=[0, 0],
            mode="markers",
            marker=dict(color="rgba(0,0,0,0)", size=0.1),
            showlegend=False,
            hoverinfo="skip"
        )
    )

    fig.update_layout(
        xaxis=dict(
            title_text="trajectory_duration（s）",
            tickfont=dict(size=11, color="#666"),
            title_font=dict(size=12, color="#666"),
            gridcolor="rgba(0,0,0,0.05)",
            zeroline=False,
            rangeslider=dict(
                visible=True,
                thickness=0.12,
                bordercolor="#d1d5db",
                borderwidth=1,
            ),
            range=slider_range,
        ),
        yaxis=dict(
            title_text="采集条数",
            tickfont=dict(size=11, color="#666"),
            title_font=dict(size=12, color="#666"),
            gridcolor="rgba(0,0,0,0.05)",
            zeroline=False,
        ),
        height=360,
        margin=dict(l=56, r=24, t=52, b=45),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        dragmode="zoom",
        hoverlabel=dict(bgcolor="#fff", bordercolor="#eaeaea", font=dict(size=12, color="#1a1a1a")),
    )

    fig.add_vline(x=slider_range[0], line_width=2, line_color="#ef4444", line_dash="dash")
    fig.add_vline(x=slider_range[1], line_width=2, line_color="#ef4444", line_dash="dash")

    return fig, slider_range, [min_v, lo, hi, max_v]


def register_callbacks(app):

    # ─────────────────────────────────────────────
    # clientside callback: 按钮高亮 - 纯浏览器端
    # ─────────────────────────────────────────────
    app.clientside_callback(
        """
        function(status_map) {
            if (!status_map) return window.dash_clientside.no_update;

            var buttons = document.querySelectorAll('button[id*="duration-check-row-status-btn"]');
            var colorMap = {
                'pass': '#059669',
                'fast': '#d97706',
                'slow': '#2563eb',
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
        Output("duration-check-query-message", "data-dummy"),
        Input("duration-check-row-status", "data"),
        prevent_initial_call=True,
    )

    # ─────────────────────────────────────────────
    # 加载任务列表（仅查 source_pg）
    # ─────────────────────────────────────────────
    @app.callback(
        Output("duration-check-task", "options"),
        Input("duration-check-task", "search_value"),
    )
    def update_task_options(search_value):
        try:
            df = query_df("SELECT DISTINCT task_id FROM episodes WHERE valid=true ORDER BY task_id")
            return [{"label": str(t), "value": str(t)} for t in df["task_id"] if pd.notnull(t) and str(t) != "0"]
        except Exception:
            return []

    # ─────────────────────────────────────────────
    # 查询按钮 → 拉取数据 + 生成分布图
    # ─────────────────────────────────────────────
    @app.callback(
        [
            Output("duration-check-query-data", "data"),
            Output("duration-check-query-message", "children"),
            Output("duration-check-dist-chart-container", "children"),
            Output("duration-check-selected-range", "data"),
            Output("duration-check-limits", "data"),
        ],
        Input("duration-check-search-btn", "n_clicks"),
        [
            State("duration-check-date", "date"),
            State("duration-check-task", "value"),
        ],
    )
    def fetch_data(n_clicks, date_val, task_val):
        if not n_clicks:
            return no_update, no_update, no_update, no_update, no_update

        if not date_val and not task_val:
            return [], "请输入日期或任务ID", html.Div("请选择任务并点击查询", style={"textAlign": "center"}), None, None

        # Redis 缓存
        cache_key = f"duration_check_fast_{date_val or 'all'}_{task_val or 'all'}"
        cached_data = get_cache(cache_key)

        if cached_data:
            df = pd.DataFrame(cached_data)
        else:
            conditions = ["valid = true", "trajectory_duration IS NOT NULL", "trajectory_duration > 0"]
            params = {}
            if date_val:
                conditions.append("DATE(trajectory_start) = %(date)s")
                params["date"] = date_val
            if task_val:
                conditions.append("task_id = %(task_id)s")
                params["task_id"] = task_val

            where = " AND ".join(conditions)
            sql = f"SELECT id, task_id, trajectory_duration, trajectory_start FROM episodes WHERE {where}"

            try:
                df = query_df(sql, params)
            except Exception as e:
                return [], f"查询出错: {e}", html.Div("查询出错", style={"textAlign": "center"}), None, None

            if not df.empty:
                df["id"] = df["id"].astype(str)
                df["task_id"] = df["task_id"].astype(str)
                df["trajectory_start"] = df["trajectory_start"].astype(str)
                set_cache(cache_key, df.to_dict("records"), 1800)

        df["duration_sec"] = pd.to_numeric(df["trajectory_duration"], errors="coerce")

        fig, rng, limits = _build_duration_distribution_figure(df)
        if not fig:
            return [], "有效时长数据为空", html.Div("暂无统计图", style={"textAlign": "center"}), None, None

        chart_layout = html.Div(
            [
                html.Div(
                    [
                        html.Span("任务时长分布（trajectory_duration）", style={"fontSize": "15px", "fontWeight": "600", "color": "#1a1a1a"}),
                        html.Div(
                            [
                                html.Button("检测过快异常", id="duration-check-fast-anomaly-btn", n_clicks=0, style={"padding": "4px 10px", "border": "1px solid #d97706", "color": "#d97706", "background": "transparent", "borderRadius": "4px", "cursor": "pointer", "fontSize": "12px", "fontWeight": "bold"}),
                                html.Button("检测过慢异常", id="duration-check-slow-anomaly-btn", n_clicks=0, style={"padding": "4px 10px", "border": "1px solid #2563eb", "color": "#2563eb", "background": "transparent", "borderRadius": "4px", "cursor": "pointer", "fontSize": "12px", "fontWeight": "bold"}),
                            ],
                            style={"display": "flex", "gap": "8px"}
                        )
                    ],
                    style={"display": "flex", "justifyContent": "flex-start", "gap": "16px", "alignItems": "center", "marginBottom": "6px", "paddingLeft": "15px", "paddingRight": "15px"}
                ),
                dcc.Graph(
                    id="duration-check-dist-chart",
                    figure=fig,
                    style={"width": "100%"},
                    config={"editable": False},
                )
            ]
        )

        msg = f"共查询到 {len(df)} 条记录。"
        return (
            df.to_dict("records"),
            msg,
            chart_layout,
            rng,
            limits
        )

    # ─────────────────────────────────────────────
    # 拖动时间轴 → 更新 selected_range（纯 clientside）
    # ─────────────────────────────────────────────
    app.clientside_callback(
        """
        function(relayoutData) {
            if (!relayoutData) return window.dash_clientside.no_update;
            // 响应双击复原：返回一个极限大的范围，让Python端认为包含全部数据
            if (relayoutData['xaxis.autorange']) {
                return [-9999999, 9999999];
            }
            if (relayoutData['xaxis.range[0]'] != null && relayoutData['xaxis.range[1]'] != null) {
                var x0 = relayoutData['xaxis.range[0]'];
                var x1 = relayoutData['xaxis.range[1]'];
                return [Math.min(x0, x1), Math.max(x0, x1)]; 
            }
            if (relayoutData['xaxis.range']) {
                var rng = relayoutData['xaxis.range'];
                return [Math.min(rng[0], rng[1]), Math.max(rng[0], rng[1])];
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output("duration-check-selected-range", "data", allow_duplicate=True),
        Input("duration-check-dist-chart", "relayoutData"),
        prevent_initial_call=True,
    )

    # ─────────────────────────────────────────────
    # 核心：更新左侧表格 + 可见 IDs + 摘要
    # ★ 支持两种模式：未检数据（默认） / 已检数据（toggle）
    # ─────────────────────────────────────────────
    @app.callback(
        [
            Output("duration-check-table-container", "children"),
            Output("duration-check-visible-ids", "data"),
            Output("duration-check-selected-summary", "children"),
            Output("duration-check-toggle-checked-btn", "children"),
            Output("duration-check-toggle-checked-btn", "outline"),
            Output("duration-check-page", "data"),
        ],
        [
            Input("duration-check-query-data", "data"),
            Input("duration-check-selected-range", "data"),
            Input("duration-check-submitted", "data"),
            Input("duration-check-show-checked", "data"),
            Input("duration-check-load-more-btn", "n_clicks"),
        ],
        [
            State("duration-check-row-status", "data"),
            State("duration-check-page", "data"),
        ],
    )
    def update_table(all_data, sel_range, submitted, show_checked, load_more, row_status, page):
        submitted = submitted or {"pass": [], "fast": [], "slow": [], "invalid": []}
        show_checked = bool(show_checked)

        trigger = ctx.triggered_id
        if trigger == "duration-check-load-more-btn":
            page = (page or 1) + 1
        elif trigger in ["duration-check-query-data", "duration-check-selected-range", "duration-check-show-checked", "duration-check-submitted"]:
            page = 1
        else:
            page = page or 1

        submitted_ids = set()
        for v in submitted.values():
            if isinstance(v, list):
                for item in v:
                    submitted_ids.add(str(item.get("id")))

        all_data = all_data or []
        all_ids = [str(r.get("id")) for r in all_data]

        # 查询全局已检数据
        checked_map = {}
        if all_ids:
            try:
                checked_map = query_checked_episodes(all_ids)
            except Exception:
                checked_map = {}

        checked_count = len(checked_map)
        total_count = len(all_data)
        btn_label = f"查看已检数据：{checked_count}/{total_count}"

        MAX_RENDER = page * 20

        if not show_checked:
            # ── 正常模式：显示未检测+未提交的数据（受时间轴筛选管控） ──
            range_rows = []
            if sel_range:
                # 增加极小的容差避免浮点数精度丢失边缘数据
                lo, hi = sel_range[0] - 0.05, sel_range[1] + 0.05
                for r in all_data:
                    d = r.get("duration_sec")
                    if d is not None and lo <= d <= hi:
                        range_rows.append(r)
            else:
                range_rows = all_data

            visible_rows = [
                r for r in range_rows
                if str(r.get("id")) not in submitted_ids
                and str(r.get("id")) not in checked_map
            ]
            visible_rows.sort(key=lambda x: float(x.get("duration_sec", 0) or 0))

            visible_ids = [str(x.get("id")) for x in visible_rows]

            if not visible_rows:
                table_ui = html.Div(
                    "当前范围无未提交数据",
                    style={"padding": "20px", "textAlign": "center", "color": "#9ca3af"},
                )
            else:
                shown = visible_rows[:MAX_RENDER]
                cards = [_build_duration_card(r, row_status or {}) for r in shown]
                if len(visible_rows) > MAX_RENDER:
                    cards.append(html.Div("往下滚动加载更多...", style={"textAlign": "center", "color": "#6b7280", "padding": "10px", "fontSize": "12px", "marginTop": "10px"}))
                table_ui = html.Div(cards)

            summary = (
                f"当前范围内包含 {len(visible_rows)} 条未检测数据（{sel_range[0]:.2f}s - {sel_range[1]:.2f}s）。"
                if sel_range
                else ""
            )
            return table_ui, visible_ids, summary, btn_label, True, page

        else:
            # ── 已检模式：显示所有已检测的数据（不受时间轴限制） ──
            checked_rows = [r for r in all_data if str(r.get("id")) in checked_map]
            checked_rows.sort(key=lambda x: float(x.get("duration_sec", 0) or 0))

            if not checked_rows:
                table_ui = html.Div(
                    "当前选中任务下没有已检测的数据",
                    style={"padding": "20px", "textAlign": "center", "color": "#9ca3af"},
                )
            else:
                shown = checked_rows[:MAX_RENDER]
                cards = [_build_checked_card(r, checked_map.get(str(r.get("id")), "pass")) for r in shown]
                if len(checked_rows) > MAX_RENDER:
                    cards.append(html.Div("往下滚动加载更多...", style={"textAlign": "center", "color": "#6b7280", "padding": "10px", "fontSize": "12px", "marginTop": "10px"}))
                table_ui = html.Div(cards)

            summary = f"当前数据源中包含 {len(checked_rows)} 条已检测数据。"
            return table_ui, [], summary, btn_label, False, page

    app.clientside_callback(
        """
        function(id_table) {
            var tableContainer = document.getElementById(id_table);
            if(tableContainer && !tableContainer.dataset.scrollBound) {
                tableContainer.dataset.scrollBound = '1';
                tableContainer.addEventListener('scroll', function() {
                    // 当滚动到底部
                    if(tableContainer.scrollTop + tableContainer.clientHeight >= Math.floor(tableContainer.scrollHeight) - 2) {
                        var btn = document.getElementById('duration-check-load-more-btn');
                        if(btn) btn.click();
                    }
                });
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output("duration-check-table-container", "data-scroll-bound"),
        Input("duration-check-table-container", "id")
    )

    # 侧边栏独立回调 — 仅依赖 submitted 数据变更
    # folder 展开/折叠由 clientside JS 直接操控，不触发此回调
    @app.callback(
        Output("duration-check-sidebar-container", "children"),
        Input("duration-check-submitted", "data"),
    )
    def update_sidebar(submitted):
        submitted = submitted or {"pass": [], "fast": [], "slow": [], "invalid": []}
        return _build_submitted_sidebar(submitted)

    # ─────────────────────────────────────────────
    # 切换 查看已检数据 / 未检数据
    # ─────────────────────────────────────────────
    @app.callback(
        Output("duration-check-show-checked", "data"),
        [
            Input("duration-check-toggle-checked-btn", "n_clicks"),
            Input("duration-check-query-data", "data"),
        ],
        State("duration-check-show-checked", "data"),
        prevent_initial_call=True,
    )
    def toggle_checked_view(n_clicks, query_data, current):
        if ctx.triggered_id == "duration-check-query-data":
            return False
        if not n_clicks:
            return no_update
        return not bool(current)

    # ─────────────────────────────────────────────
    # 单行标记状态
    # ─────────────────────────────────────────────
    @app.callback(
        Output("duration-check-row-status", "data", allow_duplicate=True),
        Input({"type": "duration-check-row-status-btn", "episode_id": ALL, "status": ALL}, "n_clicks"),
        State("duration-check-row-status", "data"),
        prevent_initial_call=True,
    )
    def update_single_row_status(n_clicks_list, current_status):
        if not ctx.triggered_id:
            return no_update

        # 防止组件重建时误触发
        trigger_val = ctx.triggered[0].get("value") if ctx.triggered else None
        if not trigger_val or trigger_val == 0:
            return no_update

        ep_id = str(ctx.triggered_id["episode_id"])
        status = ctx.triggered_id["status"]

        current_status = current_status or {}
        if current_status.get(ep_id) == status:
            del current_status[ep_id]
        else:
            current_status[ep_id] = status

        return current_status

    # ─────────────────────────────────────────────
    # 批量标记：全部合格 / 过快 / 过慢 / 无效
    # ─────────────────────────────────────────────
    @app.callback(
        Output("duration-check-row-status", "data", allow_duplicate=True),
        [
            Input("duration-check-all-pass-btn", "n_clicks"),
            Input("duration-check-all-fast-btn", "n_clicks"),
            Input("duration-check-all-slow-btn", "n_clicks"),
            Input("duration-check-all-invalid-btn", "n_clicks"),
        ],
        [
            State("duration-check-visible-ids", "data"),
            State("duration-check-row-status", "data"),
        ],
        prevent_initial_call=True,
    )
    def batch_update_status(p_clicks, f_clicks, s_clicks, i_clicks, visible_ids, current_status):
        if not ctx.triggered_id:
            return no_update

        current_status = current_status or {}
        btn_map = {
            "duration-check-all-pass-btn": "pass",
            "duration-check-all-fast-btn": "fast",
            "duration-check-all-slow-btn": "slow",
            "duration-check-all-invalid-btn": "invalid",
        }
        status_to_set = btn_map.get(ctx.triggered_id, "")

        if not status_to_set or not visible_ids:
            return no_update

        for vid in visible_ids:
            current_status[str(vid)] = status_to_set

        return current_status

    # ─────────────────────────────────────────────
    # 提交到侧边栏（浏览器 localStorage）
    # ─────────────────────────────────────────────
    @app.callback(
        [
            Output("duration-check-submitted", "data", allow_duplicate=True),
            Output("duration-check-row-status", "data", allow_duplicate=True),
            Output("duration-check-action-message", "children"),
        ],
        Input("duration-check-submit-btn", "n_clicks"),
        [
            State("duration-check-row-status", "data"),
            State("duration-check-submitted", "data"),
            State("duration-check-query-data", "data"),
        ],
        prevent_initial_call=True,
    )
    def submit_to_sidebar(n_clicks, row_status, submitted, all_data):
        if not n_clicks or not row_status:
            return no_update, no_update, no_update

        submitted = submitted or {"pass": [], "fast": [], "slow": [], "invalid": []}
        data_map = {str(d.get("id")): d for d in all_data or []}

        # 已有的 episode_id 集合，用于去重
        existing_ids = set()
        for v in submitted.values():
            if isinstance(v, list):
                for item in v:
                    existing_ids.add(str(item.get("id")))

        count = 0
        rem_status = {}
        for ep_id, status in row_status.items():
            if status in submitted and ep_id in data_map and ep_id not in existing_ids:
                row = data_map[ep_id]
                # 只存必要字段，减小 localStorage 体积
                submitted[status].append({
                    "id": str(row.get("id", "")),
                    "task_id": str(row.get("task_id", "")),
                    "duration_sec": row.get("duration_sec", 0),
                })
                existing_ids.add(ep_id)
                count += 1
            elif status not in submitted:
                rem_status[ep_id] = status

        msg = html.Div(
            f"成功提交 {count} 条数据到侧边栏（浏览器本地暂存）。",
            style={"color": "#10b981", "fontSize": "13px"},
        )
        return submitted, rem_status, msg

    # ─────────────────────────────────────────────
    # 撤销侧边栏中的某条
    # ─────────────────────────────────────────────
    @app.callback(
        Output("duration-check-submitted", "data", allow_duplicate=True),
        Input({"type": "duration-check-undo-btn", "episode_id": ALL}, "n_clicks"),
        State("duration-check-submitted", "data"),
        prevent_initial_call=True,
    )
    def undo_submit(n_clicks_list, submitted):
        if not ctx.triggered_id:
            return no_update

        # 防止组件重建时误触发：检查实际点击值
        trigger_val = ctx.triggered[0].get("value") if ctx.triggered else None
        if not trigger_val or trigger_val == 0:
            return no_update

        ep_id = str(ctx.triggered_id["episode_id"])
        submitted = submitted or {"pass": [], "fast": [], "slow": [], "invalid": []}

        for k in submitted:
            if isinstance(submitted[k], list):
                submitted[k] = [x for x in submitted[k] if str(x.get("id")) != ep_id]

        return submitted

    # ─────────────────────────────────────────────
    # 全部撤销某一分组
    # ─────────────────────────────────────────────
    @app.callback(
        Output("duration-check-submitted", "data", allow_duplicate=True),
        Input({"type": "duration-check-undo-all-btn", "status": ALL}, "n_clicks"),
        State("duration-check-submitted", "data"),
        prevent_initial_call=True,
    )
    def undo_all_in_group(n_clicks_list, submitted):
        if not ctx.triggered_id:
            return no_update

        trigger_val = ctx.triggered[0].get("value") if ctx.triggered else None
        if not trigger_val or trigger_val == 0:
            return no_update

        status_key = str(ctx.triggered_id["status"])
        submitted = submitted or {"pass": [], "fast": [], "slow": [], "invalid": []}

        if status_key in submitted:
            submitted[status_key] = []

        return submitted

    # ─────────────────────────────────────────────
    # 文件夹展开/折叠（纯 clientside，不回服务端）
    # ─────────────────────────────────────────────
    app.clientside_callback(
        """
        function() {
            var triggered = window.dash_clientside.callback_context.triggered;
            if (!triggered || triggered.length === 0) return window.dash_clientside.no_update;
            var trig = triggered[0];
            if (!trig.value || trig.value === 0) return window.dash_clientside.no_update;

            // 获取被点击的按钮 DOM
            var btnId;
            try { btnId = JSON.parse(trig.prop_id.split('.')[0]); } catch(e) { return window.dash_clientside.no_update; }

            // 找到按钮元素 → 父容器 → 下一个兄弟元素（dc-folder-content）
            var btns = document.querySelectorAll('.dc-folder-toggle-btn');
            for (var i = 0; i < btns.length; i++) {
                var btn = btns[i];
                try {
                    var id = JSON.parse(btn.id);
                    if (id.status === btnId.status) {
                        var contentDiv = btn.parentElement.nextElementSibling;
                        if (contentDiv && contentDiv.classList.contains('dc-folder-content')) {
                            var isHidden = contentDiv.style.display === 'none';
                            contentDiv.style.display = isHidden ? 'block' : 'none';
                            // 更新箭头
                            var text = btn.textContent;
                            if (isHidden) {
                                btn.textContent = text.replace('▸', '▾');
                            } else {
                                btn.textContent = text.replace('▾', '▸');
                            }
                        }
                        break;
                    }
                } catch(e) {}
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output("duration-check-folder-open", "data"),
        Input({"type": "duration-check-folder-toggle", "status": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )

    # ─────────────────────────────────────────────
    # 保存结果到数据库（成功后清空侧边栏）
    # ─────────────────────────────────────────────
    @app.callback(
        [
            Output("duration-check-save-db-message", "children"),
            Output("duration-check-submitted", "data", allow_duplicate=True),
        ],
        Input("duration-check-save-db-btn", "n_clicks"),
        State("duration-check-submitted", "data"),
        prevent_initial_call=True,
    )
    def save_to_database(n_clicks, submitted):
        if not n_clicks:
            return no_update, no_update

        submitted = submitted or {"pass": [], "fast": [], "slow": [], "invalid": []}

        records = []
        group_counts = {}
        for label_key in DURATION_STATUS_ORDER:
            items = submitted.get(label_key, [])
            if not isinstance(items, list):
                continue
            cnt = 0
            for item in items:
                records.append({
                    "episode_id": str(item.get("id", "")),
                    "task_id": str(item.get("task_id", "")),
                    "label": label_key,
                })
                cnt += 1
            if cnt > 0:
                group_counts[label_key] = cnt

        if not records:
            return html.Div(
                "侧边栏中没有数据需要保存。",
                style={"color": "#d97706", "fontSize": "12px"},
            ), no_update

        try:
            count = save_duration_results(records)
            detail = "、".join(
                f"{DURATION_STATUS_LABEL.get(k, k)} {v} 条"
                for k, v in group_counts.items()
            )
            msg = html.Div(
                f"✅ 成功保存 {count} 条结果到数据库！（{detail}）",
                style={"color": "#059669", "fontSize": "12px", "fontWeight": "600"},
            )
            # 保存成功 → 清空侧边栏
            empty_submitted = {"pass": [], "fast": [], "slow": [], "invalid": []}
            return msg, empty_submitted
        except Exception as e:
            return html.Div(
                f"❌ 保存失败: {e}",
                style={"color": "#dc2626", "fontSize": "12px"},
            ), no_update

    # ─────────────────────────────────────────────
    # 视频弹窗
    # ─────────────────────────────────────────────
    @app.callback(
        [
            Output("duration-check-video-modal", "is_open"),
            Output("duration-check-video-modal-title", "children"),
            Output("duration-check-video-modal-body", "children"),
        ],
        [
            Input({"type": "duration-check-open-video-title", "episode_id": ALL}, "n_clicks"),
            Input({"type": "duration-check-open-video-btn", "episode_id": ALL}, "n_clicks"),
            Input("duration-check-video-modal-close", "n_clicks"),
        ],
        prevent_initial_call=True,
    )
    def toggle_video_modal(*args):
        if not ctx.triggered_id or not ctx.triggered:
            return no_update, no_update, no_update

        trigger_val = ctx.triggered[0].get("value")
        if not trigger_val:
            return no_update, no_update, no_update

        if ctx.triggered_id == "duration-check-video-modal-close":
            return False, "", html.Div()

        ep_id = str(ctx.triggered_id["episode_id"])

        try:
            sql = """
                SELECT s.file_path
                FROM streams s
                WHERE s.episode_id = %(episode_id)s
                  AND s.stream_name = 'rgb'
                LIMIT 1
            """
            df = query_df(sql, {"episode_id": ep_id})
            if df.empty:
                body = html.Div("未找到该 episode 的 rgb 流数据记录", style={"color": "red", "padding": "20px"})
                return True, f"查看视频 - Episode {ep_id}", body

            file_path = str(df.iloc[0]["file_path"])
            url = get_video_url(file_path)
        except Exception as e:
            body = html.Div(f"数据库查询失败: {e}", style={"color": "red", "padding": "20px"})
            return True, f"查看视频 - Episode {ep_id}", body

        if not url:
            body = html.Div("无法通过数据路径获取或转换该 episode 的视频 URL", style={"color": "red", "padding": "20px"})
        else:
            video_src = url if url.startswith("http") else f"/pnp_video?path={url}"
            body = html.Video(
                src=video_src,
                controls=True,
                autoPlay=True,
                style={"width": "100%", "maxHeight": "70vh", "backgroundColor": "#000"},
            )

        return True, f"查看视频 - Episode {ep_id}", body

    # ─────────────────────────────────────────────
    # 跳转异常数据范围的按钮事件
    # ─────────────────────────────────────────────
    @app.callback(
        [
            Output("duration-check-selected-range", "data", allow_duplicate=True),
            Output("duration-check-dist-chart", "figure", allow_duplicate=True),
        ],
        [
            Input("duration-check-fast-anomaly-btn", "n_clicks"),
            Input("duration-check-slow-anomaly-btn", "n_clicks"),
        ],
        State("duration-check-limits", "data"),
        prevent_initial_call=True,
    )
    def jump_to_anomaly(fast_clicks, slow_clicks, limits):
        from dash import Patch
        
        if not limits or not ctx.triggered_id:
            return no_update, no_update
        
        min_v, lo, hi, max_v = limits
        patched_fig = Patch()
        
        margin_v = max((max_v - min_v) * 0.05, 0.5)
        
        if ctx.triggered_id == "duration-check-fast-anomaly-btn":
            if not fast_clicks:
                return no_update, no_update
            new_range = [min_v - margin_v, lo]
        elif ctx.triggered_id == "duration-check-slow-anomaly-btn":
            if not slow_clicks:
                return no_update, no_update
            new_range = [hi, max_v + margin_v]
        else:
            return no_update, no_update
            
        patched_fig["layout"]["xaxis"]["range"] = new_range
        return new_range, patched_fig
