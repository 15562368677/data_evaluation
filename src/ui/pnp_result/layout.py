from dash import html, dcc
import dash_bootstrap_components as dbc

def layout():
    return html.Div([
        # Stores for pagination and state
        dcc.Store(id="pnp-res-batch-page", data=1),
        dcc.Store(id="pnp-res-episode-page", data=1),
        dcc.Store(id="pnp-res-selected-batch", data=None),
        dcc.Store(id="pnp-res-selected-episode", data=None),
        
        # Hidden buttons to trigger load more
        html.Button(id="pnp-res-batch-load-more-btn", style={"display": "none"}, n_clicks=0),
        html.Button(id="pnp-res-episode-load-more-btn", style={"display": "none"}, n_clicks=0),
        
        # Main flex container
        html.Div([
            
            # Left side: Main Content
            html.Div([
                
                # Video Player section
                html.Div([
                    html.H5("视频画面", style={"marginBottom": "15px", "color": "#111827", "fontWeight": "600", "fontSize": "16px"}),
                    html.Div(id="pnp-res-video-container", children=[
                        html.Div(
                            "请在列表选择一个批次与 Episode 进行查看",
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
                        )
                    ]),
                ], style={"marginBottom": "20px"}),
                
                # Timeline section
                html.Div([
                    html.H5("Pick/Place 时间轴", style={"marginBottom": "15px", "color": "#111827", "fontWeight": "600", "fontSize": "16px"}),
                    dcc.Loading(
                        id="loading-timeline",
                        type="circle",
                        children=html.Div(id="pnp-res-timeline-container", children=[
                            html.Div(
                                "暂无时间轴数据",
                                style={
                                    "height": "200px",
                                    "display": "flex",
                                    "alignItems": "center",
                                    "justifyContent": "center",
                                    "backgroundColor": "#f9fafb",
                                    "color": "#9ca3af",
                                    "fontSize": "14px",
                                    "border": "1px dashed #e5e7eb",
                                    "borderRadius": "8px",
                                }
                            )
                        ])
                    )
                ], style={"marginBottom": "20px"}),
                
                # Episode List section
                html.Div([
                    html.Div([
                        html.H5("批次内 Episode 检测记录", style={"margin": "0", "color": "#111827", "fontWeight": "600", "fontSize": "16px"}),
                        html.Span("每次加载 20 条，滚动到底部继续加载", style={"fontSize": "12px", "color": "#6b7280"}),
                    ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "15px"}),
                    
                    html.Div(
                        id="pnp-res-episode-list", 
                        style={
                            "overflowY": "auto", 
                            "height": "850px", 
                            "border": "1px solid #e5e7eb", 
                            "borderRadius": "8px",
                            "padding": "10px",
                            "backgroundColor": "#ffffff"
                        },
                        children=[],
                        className="scroll-container"
                    ),
                ]),
                
            ], style={"flex": "1", "minWidth": "0", "paddingRight": "24px"}),
            
            # Right side: Batches List
            html.Div([
                html.Div([
                    html.H5("检测批次查询", style={"margin": "0", "color": "#111827", "fontWeight": "600", "fontSize": "16px"}),
                    html.Span("每次加载 10 条", style={"fontSize": "12px", "color": "#6b7280"}),
                ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "15px"}),
                
                # Search controls
                html.Div([
                    dbc.Input(id="pnp-res-task-search-input", placeholder="输入任务ID搜索批次...", type="text", style={"marginBottom": "10px"}),
                    dbc.Button("刷新/查询", id="pnp-res-batch-refresh-btn", color="primary", size="sm", className="w-100"),
                ], style={"marginBottom": "15px"}),

                html.Div(
                    id="pnp-res-batch-list", 
                    style={
                        "overflowY": "auto", 
                        "height": "calc(100vh - 200px)", 
                        "paddingRight": "5px"
                    },
                    children=[],
                    className="scroll-container"
                ),
            ], style={"width": "350px", "flexShrink": "0", "borderLeft": "1px solid #e5e7eb", "paddingLeft": "24px"}),
            
        ], style={"display": "flex", "flexDirection": "row", "width": "100%", "height": "100%"}),

    ], style={"padding": "10px"})
