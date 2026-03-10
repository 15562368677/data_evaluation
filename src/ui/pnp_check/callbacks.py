"""PnP 筛选页面回调。"""

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


def _pnp_sort_key(value):
    sval = str(value)
    if sval.isdigit():
        return 0, int(sval)
    return 1, sval


def _build_checked_card(row: dict, label: str):
    """构建已检测数据的只读卡片。"""
    ep_id = str(row.get("episode_id", ""))
    task_id = str(row.get("task_id", ""))
    r_count = row.get("r_count", 0)
    l_count = row.get("l_count", 0)

    label_text = PNP_STATUS_LABEL.get(label, label)
    label_color = PNP_STATUS_COLOR.get(label, "#6b7280")

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
            "border": f"1px solid {label_color}30",
            "borderLeft": f"4px solid {label_color}",
            "borderRadius": "8px",
            "padding": "10px 12px",
            "background": f"{label_color}08",
            "marginBottom": "8px",
        },
    )


def _build_pnp_card(row: dict, status_map: dict):
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
            "border": "1px solid #e5e7eb",
            "borderRadius": "8px",
            "padding": "10px 12px",
            "background": "#fff",
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

    # 1. 加载所有 PnP Batches
    @app.callback(
        Output("pnp-check-batch-dropdown", "options"),
        Input("pnp-check-batch-dropdown", "search_value"),
    )
    def load_pnp_batches(search_value):
        try:
            sql = "SELECT uniq_id, task_id, created_at FROM pnp_batches ORDER BY created_at DESC"
            df = query_pnp_df(sql)
            if df.empty:
                return []
            opts = []
            for _, row in df.iterrows():
                b_id = str(row['uniq_id'])
                t_id = str(row['task_id'])
                cd = row['created_at'].strftime("%Y-%m-%d %H:%M") if pd.notnull(row['created_at']) else ""
                label = f"{b_id} (Task: {t_id}) [{cd}]"
                opts.append({"label": label, "value": b_id})
            return opts
        except Exception as e:
            logging.error(f"Failed to load pnp batches: {e}")
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
        ],
        Input("pnp-check-load-batch-btn", "n_clicks"),
        State("pnp-check-batch-dropdown", "value")
    )
    def load_batch_data(n_clicks, batch_id):
        import plotly.graph_objects as go
        empty_fig = go.Figure()
        empty_fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=140, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

        if not n_clicks or not batch_id:
            return no_update, "请选择一个 BATCH_ID", no_update, no_update, no_update, no_update, no_update, no_update, no_update

        # fetch data 
        sql = """
            SELECT s.episode_id, s.right_pnp_result, s.left_pnp_result, b.task_id 
            FROM pnp_streams s
            JOIN pnp_batches b ON s.batch_id = b.uniq_id
            WHERE s.batch_id = %s
        """
        try:
            df = query_pnp_df(sql, (batch_id,))
        except Exception as e:
            return [], f"查询异常: {e}", {}, [], [], empty_fig, [], [], empty_fig
        
        if df.empty:
            return [], "该批次下无 Episode 数据", {}, [], [], empty_fig, [], [], empty_fig

        parsed_data = []
        max_r, max_l = 0, 0
        r_counts = {}
        l_counts = {}
        for _, row in df.iterrows():
            ep_id = str(row["episode_id"])
            task_id = str(row["task_id"])
            
            r_val = row['right_pnp_result']
            l_val = row['left_pnp_result']
            r_res = r_val if isinstance(r_val, list) else (json.loads(r_val) if r_val else [])
            l_res = l_val if isinstance(l_val, list) else (json.loads(l_val) if l_val else [])
            r_count = len(r_res)
            l_count = len(l_res)
            
            if r_count > max_r: max_r = r_count
            if l_count > max_l: max_l = l_count
            
            r_counts[r_count] = r_counts.get(r_count, 0) + 1
            l_counts[l_count] = l_counts.get(l_count, 0) + 1
            
            parsed_data.append({
                "episode_id": ep_id,
                "task_id": task_id,
                "r_count": r_count,
                "l_count": l_count,
                "right_pnp_result": r_res,
                "left_pnp_result": l_res,
                "batch_id": batch_id
            })

        def _make_opts(c_dict):
            return [{"label": f"{k}次 ({c_dict[k]}个)", "value": k} for k in sorted(c_dict.keys())]
            
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

        r_opts = _make_opts(r_counts)
        l_opts = _make_opts(l_counts)

        r_fig = _make_fig(r_counts, "右手", "#3b82f6") if r_counts else empty_fig
        l_fig = _make_fig(l_counts, "左手", "#10b981") if l_counts else empty_fig

        task_info = "未知"
        if parsed_data:
            t_id = parsed_data[0]["task_id"]
            try:
                # Need to use query_df from source_db which is already imported
                task_df = query_df("SELECT descriptions FROM tasks WHERE id = %(id)s", {"id": t_id})
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

        msg_element = html.Div([
            html.Div(f"共加载 {len(parsed_data)} 条 Episode 数据。当前批次最高 右手PnP次数: {max_r}, 左手PnP次数: {max_l}。"),
            html.Div(f"任务内容：{task_info}", style={"marginTop": "5px"})
        ])

        return (
            parsed_data,
            msg_element,
            {"right_max": max_r, "left_max": max_l},
            r_opts, [], r_fig,
            l_opts, [], l_fig,
        )

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
            Input("pnp-check-right-filter", "value"),
            Input("pnp-check-left-filter", "value"),
            Input("pnp-check-submitted", "data"),
            Input("pnp-check-show-checked", "data"),
            Input("pnp-check-load-more-btn", "n_clicks"),
        ],
        [
            State("pnp-check-row-status", "data"),
            State("pnp-check-page", "data"),
        ],
    )
    def update_table(all_data, r_vis, l_vis, submitted, show_checked, load_more, row_status, page):
        submitted = submitted or {"pass": [], "multi_pick": [], "fail_pick": [], "invalid": []}
        show_checked = bool(show_checked)

        trigger = ctx.triggered_id
        if trigger == "pnp-check-load-more-btn":
            page = (page or 1) + 1
        elif trigger in ["pnp-check-query-data", "pnp-check-right-filter", "pnp-check-left-filter", "pnp-check-show-checked", "pnp-check-submitted"]:
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

        if not show_checked:
            # Show unchecked & unsubmitted
            visible_rows = []
            for r in all_data:
                rc = int(r.get("r_count", 0))
                lc = int(r.get("l_count", 0))
                
                r_ok = (r_vis_set is None) or (rc in r_vis_set)
                l_ok = (l_vis_set is None) or (lc in l_vis_set)
                
                if r_ok and l_ok:
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
                cards = [_build_pnp_card(r, row_status or {}) for r in shown]
                if len(visible_rows) > MAX_RENDER:
                    cards.append(html.Div("往下滚动加载更多...", style={"textAlign": "center", "color": "#6b7280", "padding": "10px", "fontSize": "12px", "marginTop": "10px"}))
                table_ui = html.Div(cards)

            sel_r_str = ",".join(map(str, sorted(r_vis_set))) if r_vis_set else "全部"
            sel_l_str = ",".join(map(str, sorted(l_vis_set))) if l_vis_set else "全部"
            summary = f"当前范围内包含 {len(visible_rows)} 条未检测数据（右: {sel_r_str}, 左: {sel_l_str}）。"
            return table_ui, visible_ids, summary, btn_label, True, page

        else:
            # Show checked records
            checked_rows = [r for r in all_data if str(r.get("episode_id")) in checked_map]
            
            if not checked_rows:
                table_ui = html.Div(
                    "没有已检测的数据",
                    style={"padding": "20px", "textAlign": "center", "color": "#9ca3af"},
                )
            else:
                shown = checked_rows[:MAX_RENDER]
                cards = [_build_checked_card(r, checked_map.get(str(r.get("episode_id")), "pass")) for r in shown]
                if len(checked_rows) > MAX_RENDER:
                    cards.append(html.Div("往下滚动加载更多...", style={"textAlign": "center", "color": "#6b7280", "padding": "10px", "fontSize": "12px", "marginTop": "10px"}))
                table_ui = html.Div(cards)

            summary = f"当前批次中包含 {len(checked_rows)} 条已检测数据。"
            return table_ui, [], summary, btn_label, False, page

    # 4. 点击卡片上的按钮播放对应视频及渲染 PnP 时间轴
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
            
        trigger_id_str = ctx.triggered[0]["prop_id"].split(".")[0]
        try:
            trigger_dict = json.loads(trigger_id_str)
            episode_id = trigger_dict.get("episode_id")
            if not episode_id: return no_update
            return episode_id
        except Exception:
            return no_update

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
            return no_update, no_update
            
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
                    if video_url.startswith("http"):
                        video_elem = html.Video(id="pnp-check-video-player", key=episode_id, autoPlay=True, src=video_url, controls=True, style={"width": "100%", "backgroundColor": "#000", "maxHeight": "400px"})
                    else:
                        video_elem = html.Video(id="pnp-check-video-player", key=episode_id, autoPlay=True, src=f"/pnp_video?path={video_url}", controls=True, style={"width": "100%", "backgroundColor": "#000", "maxHeight": "400px"})
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
                    
                    var drawBlocks = function() {
                        var duration = video.duration || 1; 
                        var r_tracks = document.getElementById('pnp-check-right-hand-tracks');
                        var l_tracks = document.getElementById('pnp-check-left-hand-tracks');
                        if (r_tracks && l_tracks) {
                            try {
                                var r_data = JSON.parse(wrapper.dataset.right || "[]");
                                var l_data = JSON.parse(wrapper.dataset.left || "[]");
                                
                                var determine_sec_format = function(data) {
                                    if(data.length === 0) return true;
                                    return data[data.length-1][1] < 500;
                                };
                                var is_r_sec = determine_sec_format(r_data);
                                var is_l_sec = determine_sec_format(l_data);
                                var approx_fps = 62.512;
                                
                                var r_html = "";
                                for(var i=0; i<r_data.length; i++){
                                    var st_sec = is_r_sec ? r_data[i][0] : (r_data[i][0] / approx_fps);
                                    var ed_sec = is_r_sec ? r_data[i][1] : (r_data[i][1] / approx_fps);
                                    var left_pct = (st_sec / duration) * 100;
                                    var width_pct = ((ed_sec - st_sec) / duration) * 100;
                                    r_html += "<div style='position:absolute; left:" + left_pct + "%; width:" + width_pct + "%; height:100%; background:rgba(59, 130, 246, 0.7); border-radius:3px;'></div>";
                                }
                                r_tracks.innerHTML = r_html;
                                
                                var l_html = "";
                                for(var i=0; i<l_data.length; i++){
                                    var st_sec = is_l_sec ? l_data[i][0] : (l_data[i][0] / approx_fps);
                                    var ed_sec = is_l_sec ? l_data[i][1] : (l_data[i][1] / approx_fps);
                                    var left_pct = (st_sec / duration) * 100;
                                    var width_pct = ((ed_sec - st_sec) / duration) * 100;
                                    l_html += "<div style='position:absolute; left:" + left_pct + "%; width:" + width_pct + "%; height:100%; background:rgba(16, 185, 129, 0.7); border-radius:3px;'></div>";
                                }
                                l_tracks.innerHTML = l_html;
                            } catch(e) { console.error("Parse data error", e); }
                        }
                    };

                    video.addEventListener('loadedmetadata', drawBlocks);
                    if (video.readyState >= 1) { drawBlocks(); }
                    
                    video.addEventListener('timeupdate', function(){
                        var pct = (video.currentTime / (video.duration || 1)) * 100;
                        playhead.style.left = Math.min(100, Math.max(0, pct)) + '%';
                    });
                    
                    wrapper.addEventListener('click', function(e){
                        var rect = wrapper.getBoundingClientRect();
                        var pct = (e.clientX - rect.left) / rect.width;
                        if(video.duration) {
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
            return html.Div(f"成功保存 {count} 条数据到 pnp_results!", style={"color": "#10b981", "fontWeight": "bold"}), {"pass": [], "multi_pick": [], "fail_pick": [], "invalid": []}
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

