"""SQL 查询页面布局。"""

from dash import dcc, html
import dash_bootstrap_components as dbc


def layout():
    """SQL 查询页面布局。"""
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H5(
                                "结果库表",
                                style={
                                    "margin": "0 0 10px 0",
                                    "fontWeight": "600",
                                    "fontSize": "16px",
                                    "color": "#111827",
                                },
                            ),
                            html.Div(
                                "点击表名会自动查询前 100 条。",
                                style={"fontSize": "12px", "color": "#6b7280", "marginBottom": "8px"},
                            ),
                            html.Div(
                                id="sql-query-table-list",
                                children=html.Div(
                                    "加载中...",
                                    style={"fontSize": "12px", "color": "#9ca3af", "padding": "6px"},
                                ),
                                style={"overflowY": "auto", "maxHeight": "76vh"},
                            ),
                        ],
                        style={
                            "width": "280px",
                            "flexShrink": "0",
                            "border": "1px solid #e5e7eb",
                            "borderRadius": "10px",
                            "padding": "12px",
                            "backgroundColor": "#fff",
                            "height": "fit-content",
                        },
                    ),
                    html.Div(
                        [
                            html.H5(
                                "SQL 查询",
                                style={
                                    "margin": "0 0 14px 0",
                                    "fontWeight": "600",
                                    "fontSize": "16px",
                                    "color": "#111827",
                                },
                            ),
                            html.Div(
                                "仅支持只读查询（SELECT / WITH），默认查询结果库。",
                                style={"fontSize": "12px", "color": "#6b7280", "marginBottom": "10px"},
                            ),
                            dcc.Textarea(
                                id="sql-query-input",
                                placeholder="输入 SQL，例如: SELECT * FROM pnp_results LIMIT 100",
                                style={
                                    "width": "100%",
                                    "height": "170px",
                                    "fontFamily": "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
                                    "fontSize": "13px",
                                    "lineHeight": "1.5",
                                    "padding": "10px",
                                    "border": "1px solid #d1d5db",
                                    "borderRadius": "8px",
                                    "backgroundColor": "#fafafa",
                                    "resize": "vertical",
                                },
                            ),
                            html.Div(
                                [
                                    dbc.Button(
                                        "执行查询",
                                        id="sql-query-run-btn",
                                        color="primary",
                                        size="sm",
                                        className="search-btn",
                                        style={"marginRight": "8px"},
                                    ),
                                    dbc.Button(
                                        "清空",
                                        id="sql-query-clear-btn",
                                        color="secondary",
                                        outline=True,
                                        size="sm",
                                    ),
                                ],
                                style={"marginTop": "10px", "marginBottom": "10px"},
                            ),
                            html.Div(id="sql-query-message", style={"fontSize": "12px", "marginBottom": "10px"}),
                            html.Div(
                                id="sql-query-result-container",
                                children=html.Div(
                                    "输入 SQL 后点击“执行查询”。",
                                    style={
                                        "height": "180px",
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
                        ],
                        style={
                            "flex": "1",
                            "minWidth": "0",
                            "background": "#fff",
                            "border": "1px solid #e5e7eb",
                            "borderRadius": "10px",
                            "padding": "14px",
                        },
                    ),
                ],
                style={"display": "flex", "gap": "12px", "alignItems": "flex-start"},
            ),
        ]
    )
