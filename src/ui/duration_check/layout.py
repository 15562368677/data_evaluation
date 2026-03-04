"""时长检测页面布局。"""

from dash import html, dcc
import dash_bootstrap_components as dbc


def layout():
    """时长检测页面布局"""
    return html.Div(
        [
            # ── 状态存储 ──
            dcc.Store(id="duration-check-query-data", data=[]),
            dcc.Store(id="duration-check-selected-range"),
            dcc.Store(id="duration-check-row-status", data={}),
            dcc.Store(id="duration-check-visible-ids", data=[]),
            dcc.Store(
                id="duration-check-folder-open",
                data={"pass": True, "fast": True, "slow": True, "invalid": True},
            ),
            # 侧边栏已提交数据 - 使用 localStorage 持久化在浏览器端
            dcc.Store(
                id="duration-check-submitted",
                storage_type="local",
                data={"pass": [], "fast": [], "slow": [], "invalid": []},
            ),
            # 保存到数据库的触发器
            dcc.Store(id="duration-check-save-db-trigger", data=0),
            # 查看已检数据的 toggle 状态（False=显示未检，True=显示已检）
            dcc.Store(id="duration-check-show-checked", data=False),
            # 当前页码
            dcc.Store(id="duration-check-page", data=1),
            dcc.Store(id="duration-check-limits", data=None),
            html.Button(id="duration-check-load-more-btn", style={"display": "none"}, n_clicks=0),
            # 时长检测界面
            html.Div(
                [
                    html.H5(
                        "时长检测界面",
                        style={
                            "margin": "0 0 14px 0",
                            "fontWeight": "600",
                            "fontSize": "16px",
                            "color": "#111827",
                        },
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("日期", style={"fontWeight": "bold", "fontSize": "14px", "marginBottom": "6px"}),
                                    dcc.DatePickerSingle(
                                        id="duration-check-date",
                                        display_format="YYYY-MM-DD",
                                        placeholder="选择日期",
                                    ),
                                ],
                                style={"flex": "0 0 260px"},
                            ),
                            html.Div(
                                [
                                    html.Label("任务 ID", style={"fontWeight": "bold", "fontSize": "14px", "marginBottom": "6px"}),
                                    dcc.Dropdown(
                                        id="duration-check-task",
                                        options=[],
                                        placeholder="搜索或选择任务 ID",
                                        clearable=True,
                                        searchable=True,
                                    ),
                                ],
                                style={"flex": "1"},
                            ),
                            html.Div(
                                dbc.Button(
                                    "查询",
                                    id="duration-check-search-btn",
                                    n_clicks=0,
                                    className="w-100 search-btn",
                                    style={
                                        "height": "38px",
                                        "borderRadius": "8px",
                                        "fontWeight": "500",
                                        "fontSize": "14px",
                                    },
                                ),
                                style={"width": "110px"},
                            ),
                        ],
                        style={
                            "display": "flex",
                            "gap": "12px",
                            "alignItems": "end",
                            "marginBottom": "12px",
                        },
                    ),
                    html.Div(id="duration-check-query-message", style={"marginBottom": "8px", "fontSize": "12px", "color": "#6b7280"}),
                    dcc.Loading(
                        id="duration-check-dist-loading",
                        type="circle",
                        children=html.Div(
                            id="duration-check-dist-chart-container",
                            children=html.Div(
                                "请选择任务并点击查询",
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
                            ),
                        ),
                    ),
                    html.Div(
                        id="duration-check-selected-summary",
                        style={"margin": "10px 0 8px", "fontSize": "12px", "color": "#6b7280"},
                    ),
                    # 批量操作按钮行
                    html.Div(
                        [
                            dbc.Button(
                                "全部合格",
                                id="duration-check-all-pass-btn",
                                color="success",
                                outline=True,
                                size="sm",
                                style={"marginRight": "8px"},
                            ),
                            dbc.Button(
                                "全部过快",
                                id="duration-check-all-fast-btn",
                                color="warning",
                                outline=True,
                                size="sm",
                                style={"marginRight": "8px"},
                            ),
                            dbc.Button(
                                "全部过慢",
                                id="duration-check-all-slow-btn",
                                color="info",
                                outline=True,
                                size="sm",
                                style={"marginRight": "8px"},
                            ),
                            dbc.Button(
                                "全部无效",
                                id="duration-check-all-invalid-btn",
                                color="secondary",
                                outline=True,
                                size="sm",
                                style={"marginRight": "8px"},
                            ),
                            dbc.Button(
                                "提交到侧边栏",
                                id="duration-check-submit-btn",
                                color="primary",
                                outline=True,
                                size="sm",
                                title="将已标记的数据提交到右侧侧边栏（保存在浏览器本地）",
                                style={"marginRight": "8px"},
                            ),
                            dbc.Button(
                                "查看已检数据：0/0",
                                id="duration-check-toggle-checked-btn",
                                color="dark",
                                outline=True,
                                size="sm",
                                title="切换查看已检测/未检测数据",
                            ),
                        ],
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "marginBottom": "10px",
                        },
                    ),
                    html.Div(id="duration-check-action-message", style={"marginBottom": "10px"}),
                    # 主内容区：左侧表格 + 右侧侧边栏
                    html.Div(
                        [
                            html.Div(
                                id="duration-check-table-container",
                                style={
                                    "border": "1px solid #e5e7eb",
                                    "borderRadius": "8px",
                                    "padding": "10px",
                                    "backgroundColor": "#fff",
                                    "height": "920px",
                                    "overflowY": "auto",
                                },
                            ),
                            html.Div(
                                [
                                    # 侧边栏顶部：保存到数据库按钮
                                    html.Div(
                                        [
                                            dbc.Button(
                                                "💾 保存结果到数据库",
                                                id="duration-check-save-db-btn",
                                                color="success",
                                                size="sm",
                                                className="w-100",
                                                style={
                                                    "fontWeight": "600",
                                                    "borderRadius": "8px",
                                                },
                                            ),
                                        ],
                                        style={"marginBottom": "10px"},
                                    ),
                                    html.Div(id="duration-check-save-db-message", style={"marginBottom": "8px"}),
                                    # 侧边栏内容
                                    html.Div(id="duration-check-sidebar-container"),
                                ],
                                style={
                                    "border": "1px solid #e5e7eb",
                                    "borderRadius": "8px",
                                    "padding": "10px",
                                    "backgroundColor": "#fff",
                                    "height": "920px",
                                    "overflowY": "auto",
                                },
                            ),
                        ],
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "1fr 320px",
                            "gap": "12px",
                        },
                    ),
                    dbc.Modal(
                        [
                            dbc.ModalHeader(
                                dbc.ModalTitle(id="duration-check-video-modal-title"),
                                close_button=True,
                            ),
                            dbc.ModalBody(id="duration-check-video-modal-body"),
                            dbc.ModalFooter(
                                dbc.Button("关闭", id="duration-check-video-modal-close", color="secondary", size="sm")
                            ),
                        ],
                        id="duration-check-video-modal",
                        is_open=False,
                        size="xl",
                        centered=True,
                    ),
                ],
                className="card-container",
            ),
        ]
    )
