"""PnP 筛选页面布局。"""

from dash import html, dcc
import dash_bootstrap_components as dbc

def layout():
    """PnP 筛选页面布局"""
    return html.Div(
        [
            # ── 状态存储 ──
            dcc.Store(id="pnp-check-query-data", data=[]),
            dcc.Store(id="pnp-check-row-status", data={}),
            dcc.Store(id="pnp-check-visible-ids", data=[]),
            dcc.Store(
                id="pnp-check-folder-open",
                data={"pass": True, "multi_pick": True, "fail_pick": True, "invalid": True},
            ),
            # 侧边栏已提交数据 - 使用 localStorage 持久化在浏览器端
            dcc.Store(
                id="pnp-check-submitted",
                storage_type="local",
                data={"pass": [], "multi_pick": [], "fail_pick": [], "invalid": []},
            ),
            # 保存到数据库的触发器
            dcc.Store(id="pnp-check-save-db-trigger", data=0),
            # 查看已检数据的 toggle 状态（False=显示未检，True=显示已检）
            dcc.Store(id="pnp-check-show-checked", data=False),
            # 当前页码
            dcc.Store(id="pnp-check-page", data=1),
            # 右手最大/最小限制 和 左手最大/最小限制
            dcc.Store(id="pnp-check-limits", data={"right_max": 10, "left_max": 10}),
            html.Button(id="pnp-check-load-more-btn", style={"display": "none"}, n_clicks=0),
            dcc.Store(id="pnp-check-selected-video", data=None),

            # PnP 筛选界面
            html.Div(
                [
                    html.H5(
                        "PnP 筛选界面",
                        style={
                            "margin": "0 0 14px 0",
                            "fontWeight": "600",
                            "fontSize": "16px",
                            "color": "#111827",
                        },
                    ),
                    # 顶部筛选框：选择 BATCH_ID
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("批次 BATCH_ID", style={"fontWeight": "bold", "fontSize": "14px", "marginBottom": "6px"}),
                                    dcc.Dropdown(
                                        id="pnp-check-batch-dropdown",
                                        options=[],
                                        placeholder="请选择 BATCH_ID（按时间排序）",
                                        clearable=True,
                                        searchable=True,
                                    ),
                                ],
                                style={"flex": "1"},
                            ),
                            html.Div(
                                dbc.Button(
                                    "加载批次",
                                    id="pnp-check-load-batch-btn",
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
                            "marginBottom": "20px",
                        },
                    ),

                    # 左右手 PnP 次数筛选功能
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("右手 PnP 次数筛选", style={"fontWeight": "bold", "fontSize": "14px"}),
                                    html.Div(
                                        dcc.Dropdown(
                                            id="pnp-check-right-filter",
                                            multi=True,
                                            placeholder="请选择次数 (留空表示全部)",
                                        ),
                                        style={"paddingTop": "10px"}
                                    ),
                                    dcc.Graph(
                                        id="pnp-check-right-chart",
                                        config={"displayModeBar": False},
                                        style={"height": "140px", "marginTop": "10px"}
                                    )
                                ],
                                style={"flex": "1", "paddingRight": "20px"}
                            ),
                            html.Div(
                                [
                                    html.Label("左手 PnP 次数筛选", style={"fontWeight": "bold", "fontSize": "14px"}),
                                    html.Div(
                                        dcc.Dropdown(
                                            id="pnp-check-left-filter",
                                            multi=True,
                                            placeholder="请选择次数 (留空表示全部)",
                                        ),
                                        style={"paddingTop": "10px"}
                                    ),
                                    dcc.Graph(
                                        id="pnp-check-left-chart",
                                        config={"displayModeBar": False},
                                        style={"height": "140px", "marginTop": "10px"}
                                    )
                                ],
                                style={"flex": "1", "paddingLeft": "20px"}
                            ),
                        ],
                        style={"display": "flex", "width": "100%", "marginBottom": "20px"}
                    ),

                    html.Div(id="pnp-check-query-message", style={"marginBottom": "8px", "fontSize": "12px", "color": "#6b7280"}),

                    # 视频和时间轴展示区域 (原本时长分布图的位置)
                    html.Div(
                        [
                            html.Div(
                                id="pnp-check-video-container",
                                children=html.Div(
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
                                style={"width": "100%", "backgroundColor": "#000", "borderRadius": "8px", "overflow": "hidden"}
                            ),
                            html.Div(id="pnp-check-timeline-container", style={"marginTop": "10px"})
                        ],
                        style={"marginBottom": "20px", "padding": "15px", "border": "1px solid #e5e7eb", "borderRadius": "8px", "backgroundColor": "#fff"}
                    ),

                    html.Div(
                        id="pnp-check-selected-summary",
                        style={"margin": "10px 0 8px", "fontSize": "12px", "color": "#6b7280"},
                    ),

                    # 批量操作按钮行：合格，多次抓取，抓取不合格，无效
                    html.Div(
                        [
                            dbc.Button("全部合格", id="pnp-check-all-pass-btn", color="success", outline=True, size="sm", style={"marginRight": "8px"}),
                            dbc.Button("全部多次抓取", id="pnp-check-all-multi-btn", color="warning", outline=True, size="sm", style={"marginRight": "8px"}),
                            dbc.Button("全部抓取不合格", id="pnp-check-all-fail-btn", color="danger", outline=True, size="sm", style={"marginRight": "8px"}),
                            dbc.Button("全部无效", id="pnp-check-all-invalid-btn", color="secondary", outline=True, size="sm", style={"marginRight": "8px"}),
                            dbc.Button("提交到侧边栏", id="pnp-check-submit-btn", color="primary", outline=True, size="sm", title="将已标记的数据提交到右侧侧边栏", style={"marginRight": "8px"}),
                            dbc.Button("查看已检数据：0/0", id="pnp-check-toggle-checked-btn", color="dark", outline=True, size="sm", title="切换查看已检测/未检测数据"),
                        ],
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "marginBottom": "10px",
                        },
                    ),

                    html.Div(id="pnp-check-action-message", style={"marginBottom": "10px"}),

                    # 主内容区：左侧表格 + 右侧侧边栏
                    html.Div(
                        [
                            html.Div(
                                id="pnp-check-table-container",
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
                                                id="pnp-check-save-db-btn",
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
                                    html.Div(id="pnp-check-save-db-message", style={"marginBottom": "8px"}),
                                    # Task filter
                                    html.Div(
                                        [
                                            dcc.Dropdown(
                                                id="pnp-check-sidebar-task-filter",
                                                options=[],
                                                multi=True,
                                                placeholder="筛选任务 (task_id)...",
                                                style={"fontSize": "12px"}
                                            ),
                                        ],
                                        style={"marginBottom": "10px"},
                                    ),
                                    # 侧边栏内容
                                    html.Div(id="pnp-check-sidebar-container"),
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
                ],
                className="card-container",
            ),
        ]
    )
