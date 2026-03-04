"""PnP 检测页面布局。"""

from dash import html, dcc
import dash_bootstrap_components as dbc

# PnP 默认参数
PNP_DEFAULT_PARAMS = {
    "pick_closure_threshold": 0.35,
    "pick_start_offset": -10,
    "place_closure_threshold": 0.35,
    "place_velocity_threshold": -0.02,
    "place_velocity_lookback": 5,
    "place_velocity_lookahead": 0,
    "place_diff_lookahead": 10,
    "place_end_offset": 5,
    "negative_diff_threshold": -0.08,
    "positive_diff_threshold": 0.05,
    "min_joints_for_diff": 2,
    "slope_threshold": 0.0005,
    "slope_lookahead": 10,
}

def layout():
    """PnP 检测页面布局"""
    # PNP 分组显示结构
    from dash import html, dcc
    
    param_groups = [
        {
            "title": "🎯 PICK 检测条件",
            "categories": [
                {
                    "sub_title": "闭合度判断",
                    "items": [
                        {"id": "pick_closure_threshold", "label": "Pick 闭合度阈值", "desc": "判定抓取的最小闭合程度 (>)"},
                    ]
                },
                {
                    "sub_title": "关节差值与斜率判断",
                    "items": [
                        {"id": "negative_diff_threshold", "label": "四指差值阈值", "desc": "四指角度比过往变小的幅度 (<0)"},
                        {"id": "positive_diff_threshold", "label": "拇指差值阈值", "desc": "拇指角度比过往增大的幅度 (>0)"},
                        {"id": "min_joints_for_diff", "label": "最少满足差值关节数", "desc": "须有几个关节同时满足差值要求"},
                        {"id": "slope_threshold", "label": "斜率(抖动)阈值", "desc": "动作稳定性的斜率上限"},
                        {"id": "slope_lookahead", "label": "斜率计算跨度", "desc": "计算斜率时的帧跨度"},
                    ]
                },
                {
                    "sub_title": "截取帧参数",
                    "items": [
                        {"id": "pick_start_offset", "label": "Pick 截取偏移帧数", "desc": "检测到Pick后向前偏置的帧数 (通常<0)"},
                    ]
                }
            ]
        },
        {
            "title": "🫳 PLACE 检测条件",
            "categories": [
                {
                    "sub_title": "闭合度判断",
                    "items": [
                        {"id": "place_closure_threshold", "label": "Place 闭合度阈值", "desc": "判定松开的最大闭合程度 (<)"},
                    ]
                },
                {
                    "sub_title": "关节差值窗口",
                    "items": [
                        {"id": "place_diff_lookahead", "label": "Place 差值前看帧数", "desc": "往未来多少帧检测是否差值解开"},
                    ]
                },
                {
                    "sub_title": "手指速度判断",
                    "items": [
                        {"id": "place_velocity_threshold", "label": "Place 速度阈值", "desc": "松开动作的允许速度阈值"},
                        {"id": "place_velocity_lookback", "label": "速度回溯帧数", "desc": "往前回溯帧数"},
                        {"id": "place_velocity_lookahead", "label": "速度前看帧数", "desc": "往后前看帧数"},
                    ]
                },
                {
                    "sub_title": "截取帧参数",
                    "items": [
                        {"id": "place_end_offset", "label": "Place 截取偏移帧数", "desc": "检测到Place后向后偏置的帧数 (通常>0)"},
                    ]
                }
            ]
        }
    ]

    param_inputs = []
    
    for group in param_groups:
        category_inputs = []
        for category in group["categories"]:
            items_components = []
            for item in category["items"]:
                key = item["id"]
                value = PNP_DEFAULT_PARAMS.get(key, 0)
                items_components.append(
                    html.Div(
                        [
                            html.Label(item["label"], style={"fontSize": "11px", "color": "#333", "fontWeight": "600", "marginBottom": "2px", "display": "block"}),
                            html.Div(item["desc"], style={"fontSize": "9px", "color": "#888", "marginBottom": "4px"}),
                            dcc.Input(
                                id=f"pnp-param-{key}",
                                type="number",
                                value=value,
                                style={
                                    "width": "100%",
                                    "height": "28px",
                                    "padding": "0 6px",
                                    "borderRadius": "4px",
                                    "border": "1px solid #d1d5db",
                                    "fontSize": "11px",
                                }
                            )
                        ],
                        style={"marginBottom": "8px", "padding": "6px", "backgroundColor": "#fff", "borderRadius": "4px", "border": "1px solid #e5e7eb"}
                    )
                )
            category_inputs.append(
                html.Div(
                    [
                        html.Div(category["sub_title"], style={"fontSize": "12px", "fontWeight": "600", "color": "#10b981", "marginBottom": "6px"}),
                        html.Div(items_components)
                    ],
                    style={"marginBottom": "12px", "paddingLeft": "8px", "borderLeft": "2px solid #34d399"}
                )
            )

        param_inputs.append(
            html.Div(
                [
                    html.Div(group["title"], style={"fontSize": "13px", "fontWeight": "bold", "color": "#4f46e5", "marginBottom": "10px", "marginTop": "4px", "borderBottom": "2px solid #c7d2fe", "paddingBottom": "4px"}),
                    html.Div(category_inputs)
                ],
                style={"marginBottom": "16px", "padding": "10px", "backgroundColor": "#f8fafc", "borderRadius": "6px", "border": "1px solid #e2e8f0"}
            )
        )

    return html.Div(
        [
            # 隐藏 Store
            dcc.Store(id="pnp-episode-data"),
            dcc.Store(id="pnp-joint-data"),
            dcc.Store(id="pnp-video-url"),
            dcc.Interval(id="pnp-sync-interval", interval=120, n_intervals=0),

            # 弹窗 (Modal)
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("提交 PNP 检测", style={"fontSize": "16px"})),
                    dbc.ModalBody(
                        [
                            html.Div(
                                [
                                    html.Label("Uniq ID (为空则自动生成)", style={"fontSize": "12px", "fontWeight": "600"}),
                                    dcc.Input(id="pnp-modal-uniq-id", type="text", placeholder="请输入...",
                                              style={"width": "100%", "marginBottom": "16px", "height": "32px", "borderRadius": "4px", "border": "1px solid #ccc", "padding": "0 8px"}),
                                    
                                    html.Label("任务 ID", style={"fontSize": "12px", "fontWeight": "600"}),
                                    dcc.Dropdown(
                                        id="pnp-modal-task-search",
                                        options=[],
                                        placeholder="搜索或选择任务...",
                                        clearable=True,
                                        searchable=True,
                                    ),
                                    html.Div(style={"marginBottom": "16px"}),

                                    html.Label("检测比例 (0-100, 0表示最少检测1条)", style={"fontSize": "12px", "fontWeight": "600"}),
                                    dcc.Input(id="pnp-modal-sample-ratio", type="number", min=0, max=100, value=0,
                                              style={"width": "100%", "marginBottom": "16px", "height": "32px", "borderRadius": "4px", "border": "1px solid #ccc", "padding": "0 8px"}),
                                    
                                    dbc.Checkbox(
                                        id="pnp-modal-overwrite",
                                        label="覆盖检测",
                                        value=False,
                                        style={"marginBottom": "16px"}
                                    ),

                                    html.Label("使用的检测参数", style={"fontSize": "12px", "fontWeight": "600"}),
                                    html.Pre(
                                        id="pnp-modal-params-display",
                                        style={
                                            "backgroundColor": "#f4f4f4",
                                            "padding": "10px",
                                            "borderRadius": "4px",
                                            "fontSize": "12px",
                                            "maxHeight": "150px",
                                            "overflowY": "auto"
                                        }
                                    )
                                ]
                            )
                        ]
                    ),
                    dbc.ModalFooter(
                        [
                            dbc.Button("取消", id="pnp-modal-close-btn", color="secondary", className="me-2", size="sm"),
                            dbc.Button("确认提交", id="pnp-modal-confirm-btn", color="primary", size="sm"),
                        ]
                    ),
                ],
                id="pnp-submit-modal",
                is_open=False,
            ),

            # 页面提示
            html.Div(id="pnp-global-toast-container", style={"position": "fixed", "top": "20px", "right": "20px", "zIndex": 9999}),

            # 顶部搜索栏
            html.Div(
                [
                    # 左侧：任务搜索
                    html.Div(
                        [
                            html.Label("任务搜索", style={"fontSize": "12px", "fontWeight": "600", "color": "#888", "marginBottom": "6px"}),
                            dcc.Dropdown(
                                id="pnp-task-search",
                                options=[],
                                placeholder="搜索或选择任务...",
                                clearable=True,
                                searchable=True,
                            ),
                        ],
                        style={"flex": "1", "marginRight": "16px"},
                    ),
                    # 右侧：记录 ID 搜索
                    html.Div(
                        [
                            html.Label("记录 ID", style={"fontSize": "12px", "fontWeight": "600", "color": "#888", "marginBottom": "6px"}),
                            dcc.Dropdown(
                                id="pnp-episode-search",
                                options=[],
                                placeholder="搜索或选择记录 ID...",
                                clearable=True,
                                searchable=True,
                            ),
                        ],
                        style={"flex": "1", "marginRight": "16px"},
                    ),
                    # 加载按钮
                    html.Div(
                        [
                            html.Label("\u00a0", style={"fontSize": "12px", "marginBottom": "6px", "display": "block"}),
                            html.Button(
                                "加载数据",
                                id="pnp-load-btn",
                                n_clicks=0,
                                style={
                                    "height": "38px",
                                    "padding": "0 24px",
                                    "borderRadius": "8px",
                                    "border": "1px solid #d1d5db",
                                    "background": "#fff",
                                    "color": "#1a1a1a",
                                    "fontWeight": "500",
                                    "fontSize": "14px",
                                    "cursor": "pointer",
                                    "whiteSpace": "nowrap",
                                },
                            ),
                        ],
                    ),
                ],
                style={
                    "display": "flex",
                    "alignItems": "flex-end",
                    "gap": "8px",
                    "marginBottom": "20px",
                },
            ),

            # 加载状态提示
            html.Div(id="pnp-status-msg", style={"marginBottom": "12px"}),

            # 主内容区容器 (左侧图表视频 + 右侧参数栏)
            html.Div(
                [
                    # 左侧主内容区
                    html.Div(
                        [
                            # 视频播放区
                            html.Div(
                                [
                                    html.Div(
                                        "头部 RGB 画面",
                                        style={
                                            "fontSize": "14px",
                                            "fontWeight": "600",
                                            "color": "#374151",
                                            "marginBottom": "8px",
                                        },
                                    ),
                                    dcc.Loading(
                                        type="circle",
                                        children=html.Div(
                                            id="pnp-video-container",
                                            children=html.Div(
                                                "请选择任务和记录后点击加载",
                                                style={
                                                    "height": "400px",
                                                    "display": "flex",
                                                    "alignItems": "center",
                                                    "justifyContent": "center",
                                                    "backgroundColor": "#f9fafb",
                                                    "color": "#9ca3af",
                                                    "fontSize": "14px",
                                                    "border": "1px dashed #e5e7eb",
                                                    "borderRadius": "8px",
                                                },
                                            ),
                                        ),
                                    ),
                                ],
                                style={"marginBottom": "16px"},
                            ),

                            # 共用进度条
                            html.Div(
                                [
                                    dcc.Slider(
                                        id="pnp-progress-slider",
                                        min=0,
                                        max=1,
                                        step=0.01,
                                        value=0,
                                        updatemode="drag",
                                        marks=None,
                                        tooltip={"placement": "bottom", "always_visible": False},
                                        className="pnp-progress-slider",
                                    ),
                                ],
                                style={"marginBottom": "20px", "padding": "0 4px"},
                            ),

                            # 关节数据图表区
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div(
                                                "关节数据",
                                                style={
                                                    "fontSize": "14px",
                                                    "fontWeight": "600",
                                                    "color": "#374151",
                                                },
                                            ),
                                            html.Div(
                                                id="pnp-progress-text",
                                                children="0.00s / 0.00s",
                                                style={
                                                    "fontSize": "12px",
                                                    "color": "#6b7280",
                                                    "fontVariantNumeric": "tabular-nums",
                                                },
                                            ),
                                        ],
                                        style={
                                            "display": "flex",
                                            "alignItems": "center",
                                            "justifyContent": "space-between",
                                            "marginBottom": "8px",
                                        },
                                    ),
                                    dcc.Loading(
                                        type="circle",
                                        children=html.Div(
                                            id="pnp-joint-chart-wrapper",
                                            children=[
                                                html.Div(id="pnp-joint-chart-container"),
                                            ],
                                            style={"position": "relative"},
                                        ),
                                    ),
                                ],
                            ),
                        ],
                        className="card-container",
                        style={"flex": "1", "minWidth": "0"}
                    ),
                    
                    # 右侧边栏：检测参数输入
                    html.Div(
                        [
                            html.Button(
                                "提交检测",
                                id="pnp-open-modal-btn",
                                style={
                                    "width": "100%",
                                    "height": "40px",
                                    "backgroundColor": "#3b82f6",
                                    "color": "white",
                                    "borderRadius": "8px",
                                    "border": "none",
                                    "fontWeight": "bold",
                                    "cursor": "pointer",
                                    "marginBottom": "20px"
                                }
                            ),
                            html.Div(
                                "检测参数",
                                style={"fontSize": "14px", "fontWeight": "600", "marginBottom": "12px", "borderBottom": "1px solid #eee", "paddingBottom": "8px"}
                            ),
                            html.Div(
                                param_inputs,
                                style={"maxHeight": "700px", "overflowY": "auto"}
                            )
                        ],
                        className="card-container",
                        style={"width": "300px", "marginLeft": "20px", "flexShrink": "0"}
                    )
                ],
                style={"display": "flex", "alignItems": "flex-start"}
            ),
        ]
    )
