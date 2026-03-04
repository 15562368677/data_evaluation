"""PnP 检测页面回调。"""

import plotly.graph_objects as go
from dash import Input, Output, State, ctx, dcc, html, no_update
from plotly.subplots import make_subplots

from src.utils.source_db import query_df
from src.utils.data_parser import (
    JOINT_NAMES,
    HAND_JOINT_NAMES,
    get_video_url,
    load_joint_data,
)


def register_callbacks(app):
    """注册 PnP 检测页面的回调。"""

    # ── 回调1：任务搜索 → 加载任务列表 ──
    @app.callback(
        Output("pnp-task-search", "options"),
        Input("pnp-task-search", "search_value"),
        prevent_initial_call=False,
    )
    def load_task_options(search_value):
        """加载任务列表（无 valid/machine_id 限制）。"""
        try:
            if search_value:
                sql = """
                    SELECT DISTINCT task_id
                    FROM episodes
                    WHERE CAST(task_id AS TEXT) LIKE %(search)s
                    ORDER BY task_id
                    LIMIT 100
                """
                df = query_df(sql, {"search": f"%{search_value}%"})
            else:
                sql = """
                    SELECT DISTINCT task_id
                    FROM episodes
                    ORDER BY task_id
                    LIMIT 100
                """
                df = query_df(sql)
            return [{"label": str(t), "value": str(t)} for t in df["task_id"]]
        except Exception:
            return []

    # ── 回调1.5：弹窗任务搜索 → 加载任务列表 ──
    @app.callback(
        Output("pnp-modal-task-search", "options"),
        Input("pnp-modal-task-search", "search_value"),
        prevent_initial_call=False,
    )
    def load_modal_task_options(search_value):
        """弹窗内加载任务列表。"""
        try:
            if search_value:
                sql = """
                    SELECT DISTINCT task_id
                    FROM episodes
                    WHERE CAST(task_id AS TEXT) LIKE %(search)s
                    ORDER BY task_id
                    LIMIT 100
                """
                df = query_df(sql, {"search": f"%{search_value}%"})
            else:
                sql = """
                    SELECT DISTINCT task_id
                    FROM episodes
                    ORDER BY task_id
                    LIMIT 100
                """
                df = query_df(sql)
            return [{"label": str(t), "value": str(t)} for t in df["task_id"]]
        except Exception:
            return []

    # ── 回调2：记录 ID 搜索 → 加载记录列表（受任务筛选） ──
    @app.callback(
        Output("pnp-episode-search", "options"),
        [
            Input("pnp-task-search", "value"),
            Input("pnp-episode-search", "search_value"),
        ],
    )
    def load_episode_options(task_id, search_value):
        """加载记录 ID 列表（无 valid/machine_id 限制，支持全局搜索）。"""
        try:
            conditions = []
            params = {}

            if task_id:
                conditions.append("task_id = %(task_id)s")
                params["task_id"] = task_id

            if search_value:
                conditions.append("CAST(id AS TEXT) LIKE %(search)s")
                params["search"] = f"%{search_value}%"

            where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
            sql = f"""
                SELECT id, task_id
                FROM episodes
                {where}
                ORDER BY id DESC
                LIMIT 200
            """
            df = query_df(sql, params)
            return [
                {"label": f"{row['id']} (task: {row['task_id']})", "value": str(row["id"])}
                for _, row in df.iterrows()
            ]
        except Exception:
            return []

    # ── 回调3：选择记录 ID 时自动填充任务 ──
    @app.callback(
        Output("pnp-task-search", "value"),
        Input("pnp-episode-search", "value"),
        State("pnp-task-search", "value"),
        prevent_initial_call=True,
    )
    def auto_fill_task(episode_id, current_task):
        """选择记录 ID 后自动填充对应的任务 ID。"""
        if not episode_id:
            return no_update

        try:
            sql = """
                SELECT task_id FROM episodes WHERE id = %(id)s LIMIT 1
            """
            df = query_df(sql, {"id": episode_id})
            if not df.empty:
                return str(df.iloc[0]["task_id"])
        except Exception:
            pass
        return no_update

    # ── 回调4：点击加载按钮 → 下载数据 ──
    @app.callback(
        [
            Output("pnp-video-url", "data"),
            Output("pnp-joint-data", "data"),
            Output("pnp-status-msg", "children"),
        ],
        Input("pnp-load-btn", "n_clicks"),
        State("pnp-episode-search", "value"),
        prevent_initial_call=True,
    )
    def load_episode_data(n_clicks, episode_id):
        """加载选中记录的视频和关节数据。"""
        if not n_clicks or not episode_id:
            return no_update, no_update, html.Div(
                "请先选择一条记录",
                style={"color": "#f59e0b", "fontSize": "13px"},
            )

        # 从 streams 表查找 file_path
        try:
            sql = """
                SELECT s.file_path
                FROM streams s
                WHERE s.episode_id = %(episode_id)s
                  AND s.stream_name = 'rgb'
                LIMIT 1
            """
            df = query_df(sql, {"episode_id": episode_id})
            if df.empty:
                return None, None, html.Div(
                    f"未找到记录 {episode_id} 的 rgb 流数据",
                    style={"color": "#ef4444", "fontSize": "13px"},
                )

            file_path = str(df.iloc[0]["file_path"])
        except Exception as e:
            return None, None, html.Div(
                f"查询数据库失败: {e}",
                style={"color": "#ef4444", "fontSize": "13px"},
            )

        # 获取视频 URL
        status_parts = []
        video_url = None
        try:
            video_url = get_video_url(file_path)
            if video_url:
                status_parts.append("✅ 视频数据已加载")
            else:
                status_parts.append("⚠️ 未找到视频数据")
        except Exception as e:
            status_parts.append(f"❌ 视频加载失败: {e}")

        # 获取关节数据
        joint_data = None
        try:
            joint_data = load_joint_data(file_path)
            if joint_data:
                action_count = len(joint_data.get("action", {}))
                state_count = len(joint_data.get("state", {}))
                status_parts.append(f"✅ 关节数据已加载 (action: {action_count}, state: {state_count})")
            else:
                status_parts.append("⚠️ 未找到关节数据")
        except Exception as e:
            status_parts.append(f"❌ 关节数据加载失败: {e}")

        status_msg = html.Div(
            [html.Div(s, style={"fontSize": "13px", "marginBottom": "2px"}) for s in status_parts],
            style={"color": "#374151"},
        )

        return video_url, joint_data, status_msg

    # ── 回调5：视频 URL 变化 → 更新视频播放器 ──
    @app.callback(
        Output("pnp-video-container", "children"),
        Input("pnp-video-url", "data"),
    )
    def update_video_player(video_url):
        """更新视频播放器。"""
        if not video_url:
            return html.Div(
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
            )

        # 判断是 URL 还是本地路径
        if video_url.startswith("http"):
            return html.Video(
                src=video_url,
                controls=True,
                id="pnp-video-element",
                style={
                    "width": "100%",
                    "maxHeight": "500px",
                    "borderRadius": "8px",
                    "backgroundColor": "#000",
                },
            )
        else:
            # 本地文件通过 /pnp_video 路由提供
            return html.Video(
                src=f"/pnp_video?path={video_url}",
                controls=True,
                id="pnp-video-element",
                style={
                    "width": "100%",
                    "maxHeight": "500px",
                    "borderRadius": "8px",
                    "backgroundColor": "#000",
                },
            )

    # ── 回调6：关节数据变化 → 更新图表 ──
    @app.callback(
        Output("pnp-joint-chart-container", "children"),
        Input("pnp-joint-data", "data"),
    )
    def update_joint_charts(joint_data):
        """更新关节数据图表。"""
        if not joint_data:
            return html.Div(
                "暂无关节数据",
                style={
                    "height": "300px",
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

        action_data = joint_data.get("action", {})
        state_data = joint_data.get("state", {})
        ts_action = joint_data.get("timestamps_action", [])
        ts_state = joint_data.get("timestamps_state", [])

        charts = []

        # 按关节分组绘制（左手、右手放在顶部）
        from collections import OrderedDict
        joint_groups = OrderedDict([
            ("左手关节", [n for n in HAND_JOINT_NAMES if n.startswith("L_")]),
            ("右手关节", [n for n in HAND_JOINT_NAMES if n.startswith("R_")]),
            ("左臂关节", [n for n in JOINT_NAMES if "left" in n and ("shoulder" in n or "elbow" in n or "wrist" in n)]),
            ("右臂关节", [n for n in JOINT_NAMES if "right" in n and ("shoulder" in n or "elbow" in n or "wrist" in n)]),
            ("躯干关节", [n for n in JOINT_NAMES if "waist" in n or "head" in n]),
        ])

        COLS = 4  # 每行4个子图

        for group_name, joint_names in joint_groups.items():
            # 检查该组是否有数据
            has_data = any(
                n in action_data or n in state_data for n in joint_names
            )
            if not has_data:
                continue

            n_joints = len(joint_names)
            n_rows = (n_joints + COLS - 1) // COLS  # 向上取整

            fig = make_subplots(
                rows=n_rows,
                cols=COLS,
                shared_xaxes=True,
                vertical_spacing=0.08,
                horizontal_spacing=0.05,
                subplot_titles=[n.replace("_joint", "").replace("_", " ") for n in joint_names],
            )

            first_trace = True
            for idx, jname in enumerate(joint_names):
                row = idx // COLS + 1
                col = idx % COLS + 1

                short_name = jname.replace("_joint", "").replace("_", " ")
                if jname in action_data:
                    x_vals = ts_action if ts_action else list(range(len(action_data[jname])))
                    fig.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=action_data[jname],
                            name="action",
                            line=dict(color="#3b82f6", width=1.2),
                            showlegend=first_trace,
                            legendgroup="action",
                            hovertemplate=f"{short_name} action: %{{y:.4f}}<extra></extra>",
                        ),
                        row=row,
                        col=col,
                    )
                if jname in state_data:
                    x_vals = ts_state if ts_state else list(range(len(state_data[jname])))
                    fig.add_trace(
                        go.Scatter(
                            x=x_vals,
                            y=state_data[jname],
                            name="state",
                            line=dict(color="#ef4444", width=1.2),
                            showlegend=first_trace,
                            legendgroup="state",
                            hovertemplate=f"{short_name} state: %{{y:.4f}}<extra></extra>",
                        ),
                        row=row,
                        col=col,
                    )
                first_trace = False

            fig.update_layout(
                title=dict(
                    text=group_name,
                    font=dict(size=14, color="#1a1a1a"),
                    x=0.01,
                ),
                height=max(220 * n_rows, 280),
                margin=dict(l=40, r=20, t=60, b=30),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(size=11),
                ),
                hovermode="x unified",
            )

            # 更新所有子图的轴样式
            fig.update_yaxes(
                gridcolor="rgba(0,0,0,0.05)",
                zeroline=False,
                tickfont=dict(size=9, color="#888"),
            )
            fig.update_xaxes(
                gridcolor="rgba(0,0,0,0.05)",
                tickfont=dict(size=9, color="#888"),
            )

            charts.append(
                dcc.Graph(
                    figure=fig,
                    style={"width": "100%", "marginBottom": "16px"},
                )
            )

        if not charts:
            return html.Div(
                "关节数据为空",
                style={
                    "textAlign": "center",
                    "padding": "48px",
                    "color": "#999",
                },
            )

        return html.Div(charts)

    # ── 回调7：前端同步视频时间、进度条与关节图进度线 ──
    app.clientside_callback(
        """
        function(n_intervals, video_url, joint_data) {
            var video = document.getElementById('pnp-video-element');

            var getJointDuration = function(data) {
                if (!data) return 0;
                var ta = Array.isArray(data.timestamps_action) ? data.timestamps_action : [];
                var ts = Array.isArray(data.timestamps_state) ? data.timestamps_state : [];
                var maxA = ta.length ? Number(ta[ta.length - 1]) : 0;
                var maxS = ts.length ? Number(ts[ts.length - 1]) : 0;
                var v = Math.max(maxA, maxS, 0);
                return Number.isFinite(v) ? v : 0;
            };

            var jointDuration = getJointDuration(joint_data);
            var duration = jointDuration > 0 ? jointDuration : 1;
            var currentTime = 0;

            if (video) {
                var videoDuration = Number(video.duration);
                if (Number.isFinite(videoDuration) && videoDuration > 0) {
                    duration = videoDuration;
                }
                var t = Number(video.currentTime);
                currentTime = Number.isFinite(t) ? t : 0;
            }

            if (!Number.isFinite(duration) || duration <= 0) {
                duration = 1;
            }

            currentTime = Math.max(0, Math.min(duration, currentTime));
            var ratio = duration > 0 ? (currentTime / duration) : 0;
            ratio = Math.max(0, Math.min(1, ratio));

            var toFixed2 = function(v) {
                return (Number.isFinite(v) ? v : 0).toFixed(2);
            };
            var text = toFixed2(currentTime) + "s / " + toFixed2(duration) + "s";

            var plots = document.querySelectorAll('#pnp-joint-chart-container .js-plotly-plot');
            if (window.Plotly && plots && plots.length) {
                for (var i = 0; i < plots.length; i++) {
                    try {
                        var gd = plots[i];
                        var layout = gd.layout || {};
                        var xaxisKeys = Object.keys(layout).filter(function(k) {
                            return /^xaxis\\d*$/.test(k);
                        });
                        if (!xaxisKeys.length) {
                            xaxisKeys = ['xaxis'];
                        }

                        var shapes = [];
                        for (var j = 0; j < xaxisKeys.length; j++) {
                            var xKey = xaxisKeys[j];
                            var idx = xKey === 'xaxis' ? '' : xKey.replace('xaxis', '');
                            var yKey = 'yaxis' + idx;
                            if (!layout[yKey]) {
                                continue;
                            }

                            shapes.push({
                                type: 'line',
                                xref: idx ? ('x' + idx + ' domain') : 'x domain',
                                yref: idx ? ('y' + idx + ' domain') : 'y domain',
                                x0: ratio,
                                x1: ratio,
                                y0: 0,
                                y1: 1,
                                line: {
                                    color: '#8b5cf6',
                                    width: 2
                                },
                                layer: 'above'
                            });
                        }

                        window.Plotly.relayout(gd, {shapes: shapes});
                    } catch (e) {}
                }
            }

            return [currentTime, duration, text];
        }
        """,
        [
            Output("pnp-progress-slider", "value"),
            Output("pnp-progress-slider", "max"),
            Output("pnp-progress-text", "children"),
        ],
        [
            Input("pnp-sync-interval", "n_intervals"),
            Input("pnp-video-url", "data"),
            Input("pnp-joint-data", "data"),
        ],
    )

    app.clientside_callback(
        """
        function(sliderValue) {
            var video = document.getElementById('pnp-video-element');
            if (!video) {
                return window.dash_clientside.no_update;
            }

            var duration = Number(video.duration);
            if (!Number.isFinite(duration) || duration <= 0) {
                return window.dash_clientside.no_update;
            }

            var target = Number(sliderValue);
            if (!Number.isFinite(target)) {
                return window.dash_clientside.no_update;
            }

            target = Math.max(0, Math.min(duration, target));
            if (Math.abs(video.currentTime - target) > 0.12) {
                video.currentTime = target;
            }

            return target;
        }
        """,
        Output("pnp-video-container", "data-seek-sync"),
        Input("pnp-progress-slider", "value"),
    )

    # ── 回调8：模态框相关（打开/关闭/同步参数） ──
    import json
    import time
    from redis import Redis
    from rq import Queue

    param_keys = [
        "pick_closure_threshold", "pick_start_offset", "place_closure_threshold",
        "place_velocity_threshold", "place_velocity_lookback", "place_velocity_lookahead",
        "place_diff_lookahead", "place_end_offset", "negative_diff_threshold",
        "positive_diff_threshold", "min_joints_for_diff", "slope_threshold", "slope_lookahead"
    ]
    param_states = [State(f"pnp-param-{k}", "value") for k in param_keys]

    @app.callback(
        [
            Output("pnp-submit-modal", "is_open"),
            Output("pnp-modal-params-display", "children"),
            Output("pnp-modal-task-search", "value"),
        ],
        [
            Input("pnp-open-modal-btn", "n_clicks"),
            Input("pnp-modal-close-btn", "n_clicks"),
            Input("pnp-modal-confirm-btn", "n_clicks"),
        ],
        [
            State("pnp-submit-modal", "is_open"),
            State("pnp-task-search", "value"),
        ] + param_states,
        prevent_initial_call=True
    )
    def toggle_modal(open_clicks, close_clicks, confirm_clicks, is_open, current_task, *param_values):
        trigger_id = ctx.triggered_id

        if trigger_id == "pnp-open-modal-btn":
            # 组合参数用于显示
            params_dict = dict(zip(param_keys, param_values))
            display_text = json.dumps(params_dict, indent=2, ensure_ascii=False)
            return True, display_text, current_task
        
        if trigger_id == "pnp-modal-close-btn":
            return False, no_update, no_update

        # 对于 confirm 按钮的点击，这里也负责关闭弹窗
        if trigger_id == "pnp-modal-confirm-btn":
            return False, no_update, no_update

        return is_open, no_update, no_update

    # ── 回调9：提交检测到 Worker Queue ──
    import dash_bootstrap_components as dbc
    @app.callback(
        Output("pnp-global-toast-container", "children"),
        Input("pnp-modal-confirm-btn", "n_clicks"),
        [
            State("pnp-modal-uniq-id", "value"),
            State("pnp-modal-task-search", "value"),
            State("pnp-modal-sample-ratio", "value"),
            State("pnp-modal-overwrite", "value"),
        ] + param_states,
        prevent_initial_call=True
    )
    def submit_pnp_detection(n_clicks, uniq_id, task_id, sample_ratio, overwrite, *param_values):
        if not n_clicks:
            return no_update

        # 输入检查
        if not task_id:
            return dbc.Toast(
                "请先选择需要检测的任务 ID",
                id="toast-error",
                header="提交失败",
                is_open=True,
                dismissable=True,
                icon="danger",
                duration=4000
            )

        if not uniq_id:
            uniq_id = f"{task_id}_{int(time.time())}"

        if sample_ratio is None:
            sample_ratio = 0

        # 获取参数字典
        params_dict = dict(zip(param_keys, param_values))

        # 将任务提交到 Queue
        try:
            from rq import Queue
            from redis import Redis
            import os
            from dotenv import load_dotenv
            from src.workers.pnp_worker import run_pnp_task

            load_dotenv()
            redis_host = os.environ.get("REDIS_HOST", "localhost")
            redis_port = int(os.environ.get("REDIS_PORT", 6379))
            redis_db = int(os.environ.get("REDIS_DB", 1))
            redis_password = os.environ.get("REDIS_PASSWORD", None)

            q = Queue('pnp_tasks', connection=Redis(
                host=redis_host, 
                port=redis_port, 
                db=redis_db, 
                password=redis_password
            ))
            job = q.enqueue(
                run_pnp_task,
                args=(uniq_id, task_id, sample_ratio, overwrite, params_dict),
                job_timeout=3600 # 默认为1小时，防超时
            )
            return dbc.Toast(
                f"任务提交成功！(Uniq ID: {uniq_id})\n后台检测中，请稍后查看结果库。",
                id=f"toast-success-{uniq_id}",
                header="提交成功",
                is_open=True,
                dismissable=True,
                icon="success",
                duration=5000
            )
        except Exception as e:
            return dbc.Toast(
                f"提交任务到 Worker 失败: {str(e)}",
                id="toast-error-submit",
                header="系统错误",
                is_open=True,
                dismissable=True,
                icon="danger",
                duration=5000
            )


