"""Dash 应用主入口"""

import dash
from flask import send_file, request, abort
from dash import html, dcc, Input, Output, State, no_update, ctx
import dash_bootstrap_components as dbc

from src.utils.source_db import query_df

from src.utils.result_db import init_duration_result_db, init_pnp_result_db
from src.ui import pilot, pnp, pnp_result, duration_check, pnp_check

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="数据采集评估工具",
)

# 初始化 duration_result 表
init_duration_result_db()
init_pnp_result_db()


def load_initial_pilots():
    try:
        df = query_df("SELECT user_name FROM users ORDER BY user_name")
        return [{"label": str(r), "value": str(r)} for r in df["user_name"]]
    except Exception:
        return []


def load_initial_tasks():
    try:
        df = query_df("SELECT id FROM tasks ORDER BY id")
        return [{"label": str(r), "value": str(r)} for r in df["id"]]
    except Exception:
        return []


initial_pilots = load_initial_pilots()
initial_tasks = load_initial_tasks()

# ── 侧边栏 ──
sidebar = html.Div(
    [
        html.Div("数采评估", className="sidebar-brand"),
        dbc.Nav(
        [
            dbc.NavLink(
                "数采员统计",
                href="/pilot",
                id="nav-pilot",
                active="exact",
            ),
            dbc.NavLink(
                "时长检测",
                href="/duration_check",
                id="nav-duration-check",
                active="exact",
            ),
            html.Div(
                [
                    dbc.NavLink(
                        "PnP 质检",
                        id="pnp-folder-toggle",
                        n_clicks=0,
                        style={"cursor": "pointer", "marginBottom": "5px"}
                    ),
                    dbc.Collapse(
                        dbc.Nav(
                            [
                                dbc.NavLink(
                                    "PnP 检测",
                                    href="/pnp",
                                    id="nav-pnp",
                                    active="exact",
                                    style={"paddingLeft": "2.5rem"},
                                ),
                                dbc.NavLink(
                                    "PnP 结果",
                                    href="/pnp_result",
                                    id="nav-pnp-result",
                                    active="exact",
                                    style={"paddingLeft": "2.5rem"},
                                ),
                                dbc.NavLink(
                                    "PnP 筛选",
                                    href="/pnp_check",
                                    id="nav-pnp-check",
                                    active="exact",
                                    style={"paddingLeft": "2.5rem"},
                                ),
                            ],
                            vertical=True,
                            pills=True,
                        ),
                        id="pnp-folder-collapse",
                        is_open=True,
                    )
                ]
            ),
        ],
        vertical=True,
        pills=True,
    ),
    ],
    className="sidebar",
    style={
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "210px",
        "padding": "20px 16px",
        "zIndex": 1000,
    },
)

# ── 筛选栏 ──
filter_bar = html.Div(
    [
        # 隐藏 Store：记录上一次的筛选状态，用于判断哪个筛选器发生了变化
        dcc.Store(id="filter-prev-state", data={"pilot": None, "task": None, "start_date": None, "end_date": None}),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Label("日期范围"),
                        html.Div(
                            [
                                dcc.DatePickerRange(
                                    id="filter-date-range",
                                    display_format="YYYY-MM-DD",
                                ),
                                html.Button(
                                    "×",
                                    id="date-clear-btn",
                                    n_clicks=0,
                                    style={
                                        "position": "absolute",
                                        "right": "28px",
                                        "display": "none",
                                        "top": "50%",
                                        "transform": "translateY(-50%)",
                                        "border": "none",
                                        "background": "transparent",
                                        "fontSize": "16px",
                                        "color": "#999",
                                        "cursor": "pointer",
                                        "padding": "0 4px",
                                        "zIndex": "2",
                                    },
                                ),
                            ],
                            style={"position": "relative", "width": "100%"},
                        ),
                    ],
                    md=4,
                ),
                dbc.Col(
                    [
                        dbc.Label("数采员"),
                        dcc.Dropdown(
                            id="filter-pilot",
                            options=initial_pilots,
                            placeholder="搜索或选择...",
                            clearable=True,
                            searchable=True,
                        ),
                    ],
                    md=4,
                ),
                dbc.Col(
                    [
                        dbc.Label("\u00a0"),  # 占位
                        dbc.Button(
                            "查询",
                            id="search-btn",
                            className="w-100 search-btn",
                            n_clicks=0,
                            style={
                                "height": "38px",
                                "borderRadius": "8px",
                                "fontWeight": "500",
                                "fontSize": "14px",
                            },
                        ),
                    ],
                    md=3,
                ),
            ],
            className="g-3 align-items-end",
        ),
    ],
    className="filter-bar",
)

# ── 内容区 ──
content = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        html.Div(id="filter-bar-container", children=filter_bar),
        html.Div(pilot.layout(), id="page-content"),
    ],
    style={
        "marginLeft": "226px",
        "padding": "24px 28px",
        "minHeight": "100vh",
    },
)

app.layout = html.Div([sidebar, content])


@app.callback(
    [Output("page-content", "children"), Output("filter-bar-container", "style")],
    Input("url", "pathname"),
)
def render_page(pathname):
    if pathname in (None, "/", "/pilot"):
        return pilot.layout(), {"display": "block"}
    if pathname == "/duration_check":
        return duration_check.layout(), {"display": "none"}
    if pathname == "/pnp":
        return pnp.layout(), {"display": "none"}
    if pathname == "/pnp_result":
        return pnp_result.layout(), {"display": "none"}
    if pathname == "/pnp_check":
        return pnp_check.layout(), {"display": "none"}
    return html.Div(
        "页面未找到",
        style={"padding": "60px", "textAlign": "center", "color": "#999"},
    ), {"display": "none"}


def _build_where(pilot_val, start_date, end_date, task_val):
    """根据当前筛选值构建 WHERE 子句和参数"""
    conditions = ["valid = true"]
    params = {}
    if pilot_val:
        conditions.append("pilot = %(pilot)s")
        params["pilot"] = pilot_val
    if start_date:
        conditions.append("DATE(trajectory_start) >= %(start_date)s")
        params["start_date"] = start_date
    if end_date:
        conditions.append("DATE(trajectory_start) <= %(end_date)s")
        params["end_date"] = end_date
    if task_val:
        conditions.append("task_id = %(task_id)s")
        params["task_id"] = task_val
    return conditions, params


@app.callback(
    [
        Output("filter-pilot", "options"),
        Output("filter-pilot", "value"),
        Output("filter-task", "options"),
        Output("filter-task", "value"),
        Output("filter-date-range", "min_date_allowed"),
        Output("filter-date-range", "max_date_allowed"),
        Output("filter-date-range", "start_date"),
        Output("filter-date-range", "end_date"),
        Output("filter-prev-state", "data"),
    ],
    [
        Input("filter-pilot", "value"),
        Input("filter-task", "value"),
        Input("filter-date-range", "start_date"),
        Input("filter-date-range", "end_date"),
    ],
    [State("filter-prev-state", "data")],
)
def cross_filter(pilot_val, task_val, start_date, end_date, prev_state):
    """筛选器联动：pilot/日期会联动更新三者；task 变化仅更新 task 自身"""

    prev_state = prev_state or {"pilot": None, "task": None, "start_date": None, "end_date": None}

    # 判断是哪个筛选器触发了变化
    triggered = ctx.triggered_id if ctx.triggered_id else None

    # 如果值没有实际变化（由本回调自身更新引起的二次触发），直接跳过
    # 但初次加载（triggered is None）仍需走初始化逻辑
    if (triggered is not None
            and pilot_val == prev_state.get("pilot")
            and task_val == prev_state.get("task")
            and start_date == prev_state.get("start_date")
            and end_date == prev_state.get("end_date")):
        return (no_update, no_update, no_update, no_update,
                no_update, no_update, no_update, no_update,
                no_update)

    # 所有筛选器都为空时，恢复初始状态
    if not pilot_val and not task_val and not start_date and not end_date:
        new_state = {"pilot": None, "task": None, "start_date": None, "end_date": None}
        return (initial_pilots, None, initial_tasks, None,
                None, None, None, None, new_state)

    # task 的变化不反向影响 pilot/日期
    if triggered == "filter-task":
        new_state = {
            "pilot": pilot_val,
            "task": task_val,
            "start_date": start_date,
            "end_date": end_date,
        }
        return (
            no_update, no_update,
            no_update, task_val,
            no_update, no_update, no_update, no_update,
            new_state,
        )

    # ── 根据当前已选条件，查询其他筛选器的可选范围 ──

    # 1) 查询可选的 pilot 列表（仅受日期约束，排除 pilot 与 task 自身条件）
    conds_p, params_p = _build_where(None, start_date, end_date, None)
    where_p = ("WHERE " + " AND ".join(conds_p)) if conds_p else ""
    try:
        pilot_df = query_df(
            f"SELECT DISTINCT pilot FROM episodes {where_p} ORDER BY pilot", params_p
        )
        pilot_options = [{"label": str(r), "value": str(r)} for r in pilot_df["pilot"]]
    except Exception:
        pilot_options = initial_pilots

    # 2) 查询可选的 task 列表（排除 task 自身条件）
    conds_t, params_t = _build_where(pilot_val, start_date, end_date, None)
    where_t = ("WHERE " + " AND ".join(conds_t)) if conds_t else ""
    try:
        task_df = query_df(
            f"SELECT DISTINCT task_id FROM episodes {where_t} ORDER BY task_id", params_t
        )
        task_options = [{"label": str(t), "value": str(t)} for t in task_df["task_id"]]
    except Exception:
        task_options = initial_tasks

    # 3) 查询可选的日期范围（排除日期自身条件）
    conds_d, params_d = _build_where(pilot_val, None, None, None)
    where_d = ("WHERE " + " AND ".join(conds_d)) if conds_d else ""
    try:
        date_df = query_df(
            f"SELECT MIN(DATE(trajectory_start)) AS min_date, MAX(DATE(trajectory_start)) AS max_date FROM episodes {where_d}",
            params_d,
        )
        if not date_df.empty and date_df["min_date"].iloc[0] is not None:
            min_date = str(date_df["min_date"].iloc[0])
            max_date = str(date_df["max_date"].iloc[0])
        else:
            min_date = None
            max_date = None
    except Exception:
        min_date = None
        max_date = None

    # ── 校验当前选中值是否仍在可选范围内 ──
    valid_pilots = [o["value"] for o in pilot_options]
    new_pilot = pilot_val if pilot_val in valid_pilots else None

    valid_tasks = [o["value"] for o in task_options]
    new_task = task_val if task_val in valid_tasks else None

    # 日期范围策略：
    # - 日期控件自身触发：保留用户手动选择
    # - 数采员切换触发：重置为该数采员的最早/最晚日期（保证覆盖全量）
    # - 其他触发：保留已有选择；若为空则回填可选范围
    if triggered == "filter-date-range":
        new_start = start_date
        new_end = end_date
    elif triggered == "filter-pilot":
        new_start = min_date
        new_end = max_date
    else:
        new_start = start_date if start_date else min_date
        new_end = end_date if end_date else max_date

    new_state = {
        "pilot": new_pilot,
        "task": new_task,
        "start_date": new_start,
        "end_date": new_end,
    }

    return (
        pilot_options, new_pilot,
        task_options, new_task,
        min_date, max_date, new_start, new_end,
        new_state,
    )


# ── 日期清除按钮回调 ──
@app.callback(
    [
        Output("filter-date-range", "start_date", allow_duplicate=True),
        Output("filter-date-range", "end_date", allow_duplicate=True),
    ],
    Input("date-clear-btn", "n_clicks"),
    prevent_initial_call=True,
)
def clear_date_range(n_clicks):
    if not n_clicks:
        return no_update, no_update
    return None, None


@app.callback(
    Output("date-clear-btn", "style"),
    [Input("filter-date-range", "start_date"), Input("filter-date-range", "end_date")],
    prevent_initial_call=False,
)
def toggle_clear_btn(start, end):
    base_style = {
        "position": "absolute",
        "right": "28px",
        "top": "50%",
        "transform": "translateY(-50%)",
        "border": "none",
        "background": "transparent",
        "fontSize": "16px",
        "color": "#999",
        "cursor": "pointer",
        "padding": "0 4px",
        "zIndex": "2",
    }
    if start or end:
        return {**base_style, "display": "block"}
    return {**base_style, "display": "none"}


@app.callback(
    Output("search-btn", "className"),
    Input("search-btn", "n_clicks"),
    prevent_initial_call=False,
)
def toggle_search_btn_class(n_clicks):
    base_class = "w-100 search-btn"
    if n_clicks and n_clicks > 0:
        return f"{base_class} search-btn-clicked"
    return base_class


pilot.register_callbacks(app)
pnp.register_callbacks(app)
pnp_result.register_callbacks(app)
duration_check.register_callbacks(app)
pnp_check.register_callbacks(app)

@app.callback(
    Output("pnp-folder-collapse", "is_open"),
    [Input("pnp-folder-toggle", "n_clicks")],
    [State("pnp-folder-collapse", "is_open")],
)
def toggle_pnp_folder(n, is_open):
    if n:
        return not is_open
    return is_open

server = app.server


# ── Flask 路由：提供本地转换的视频文件 ──
@server.route("/pnp_video")
def serve_pnp_video():
    """提供本地缓存的视频文件。"""
    import os

    path = request.args.get("path", "")
    if not path or not os.path.isfile(path):
        abort(404)
    return send_file(path, mimetype="video/mp4")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8051)
