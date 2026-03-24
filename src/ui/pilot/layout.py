"""数采员页面布局。"""

from dash import html, dcc
import dash_bootstrap_components as dbc


def layout():
    """数采员页面布局"""
    return html.Div(
        [
            # 存储完整数据的隐藏 Store
            dcc.Store(id="pilot-full-data"),
            # Charts section wrapped in a card
            html.Div(
                [
                    # 滑块：控制显示的柱状图条数（图表上方，宽度 50%）
                    html.Div(
                        [
                            html.Label("显示柱状图条数：", style={"fontWeight": "bold", "fontSize": "14px"}),
                            dcc.Slider(
                                id="pilot-date-slider",
                                min=5,
                                max=60,
                                step=1,
                                value=10,
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                        ],
                        style={"marginBottom": "10px", "padding": "10px", "width": "50%"},
                    ),
                    # 上方统计图
                    dcc.Loading(
                        id="pilot-loading",
                        type="circle",
                        children=html.Div(id="pilot-chart-container"),
                    ),
                    # 新增：柱状图下方，数采员-任务时长箱形图
                    dcc.Loading(
                        id="pilot-task-box-loading",
                        type="circle",
                        children=html.Div(id="pilot-task-box-chart-container"),
                    ),
                    # 任务筛选框（移动到主图下方）
                    html.Div(
                        [
                            html.Label("任务", style={"fontWeight": "bold", "fontSize": "14px", "marginBottom": "6px"}),
                            dcc.Dropdown(
                                id="filter-task",
                                options=[],
                                placeholder="搜索或选择任务...",
                                clearable=True,
                                searchable=True,
                            ),
                        ],
                        style={"marginTop": "20px", "marginBottom": "12px", "maxWidth": "560px"},
                    ),
                    # 下方面积分布图
                    dcc.Loading(
                        id="task-duration-loading",
                        type="circle",
                        children=html.Div(
                            html.Div(
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
                            ),
                            id="task-duration-chart-container",
                        ),
                    ),
                    # 新增：任务时长箱形图
                    dcc.Loading(
                        id="task-duration-box-loading",
                        type="circle",
                        children=html.Div(
                            html.Div(
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
                            ),
                            id="task-duration-box-chart-container",
                        ),
                    ),
                ],
                className="card-container",
            ),
            # 页面底部统计表（不受筛选框影响）
            dcc.Loading(
                id="pilot-summary-table-loading",
                type="circle",
                children=html.Div(id="pilot-summary-table-container"),
            ),
            html.Div(
                html.Button(
                    "清除全局统计缓存",
                    id="pilot-summary-cache-clear-btn",
                    n_clicks=0,
                    style={
                        "border": "1px solid #d1d5db",
                        "background": "#fff",
                        "color": "#374151",
                        "padding": "6px 12px",
                        "borderRadius": "8px",
                        "cursor": "pointer",
                        "fontSize": "13px",
                    },
                ),
                style={"marginTop": "6px", "marginBottom": "2px"},
            ),
        ]
    )
