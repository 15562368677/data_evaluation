"""Callbacks for PnP Result Page"""

import json
import re
from dash import Input, Output, State, ALL, ctx, html, no_update
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from src.utils.source_db import query_df
from src.utils.result_db import query_pnp_df, get_pnp_connection
from src.utils.data_parser import get_video_url

def register_callbacks(app):
    
    app.clientside_callback(
        """
        function(id_batch, id_episode) {
            // 为批量列表绑定滚动
            var batchList = document.getElementById(id_batch);
            if(batchList && !batchList.dataset.scrollBound) {
                batchList.dataset.scrollBound = '1';
                batchList.addEventListener('scroll', function() {
                    // 当滚动到底部
                    if(batchList.scrollTop + batchList.clientHeight >= Math.floor(batchList.scrollHeight) - 2) {
                        var btn = document.getElementById('pnp-res-batch-load-more-btn');
                        if(btn) btn.click();
                    }
                });
            }
            
            // 为 Episode 列表绑定滚动
            var epList = document.getElementById(id_episode);
            if(epList && !epList.dataset.scrollBound) {
                epList.dataset.scrollBound = '1';
                epList.addEventListener('scroll', function() {
                    if(epList.scrollTop + epList.clientHeight >= Math.floor(epList.scrollHeight) - 2) {
                        var btn = document.getElementById('pnp-res-episode-load-more-btn');
                        if(btn) btn.click();
                    }
                });
            }
            return window.dash_clientside.no_update;
        }
        """,
        Output("pnp-res-batch-list", "data-scroll-bound"),
        Input("pnp-res-batch-list", "id"),
        Input("pnp-res-episode-list", "id")
    )

    # --- 视频进度到时间轴同步监听 ---
    # --- 视频及自定义时间轴同步控制 ---
    app.clientside_callback(
        """function(video_children) {
            setTimeout(function() {
                var video = document.getElementById('pnp-res-video-player');
                var wrapper = document.getElementById('custom-timeline-wrapper');
                var playhead = document.getElementById('timeline-playhead');
                
                if (video && wrapper && playhead && !video.dataset.syncBound) {
                    video.dataset.syncBound = '1';
                    
                    var drawBlocks = function() {
                        var duration = video.duration || 1; 
                        var r_tracks = document.getElementById('right-hand-tracks');
                        var l_tracks = document.getElementById('left-hand-tracks');
                        if (r_tracks && l_tracks) {
                            try {
                                var r_data = JSON.parse(wrapper.dataset.right || "[]");
                                var l_data = JSON.parse(wrapper.dataset.left || "[]");
                                
                                var determine_sec_format = function(data) {
                                    if(data.length === 0) return true;
                                    // if the value is unreasonably large (like frame 5000) vs a 150s video
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
        Output("pnp-res-video-container", "data-sync-bound"),
        Input("pnp-res-video-container", "children")
    )

    # --- 1. 加载或追加批次列表 (右侧边栏) ---
    @app.callback(
        Output("pnp-res-task-filter", "options"),
        Input("pnp-res-task-filter", "search_value"),
        State("pnp-res-task-filter", "value"),
    )
    def load_task_filter_options(search_value, selected_value):
        try:
            params = {}
            where = ""
            if search_value:
                where = "WHERE CAST(task_id AS TEXT) ILIKE %(search)s"
                params["search"] = f"%{search_value}%"
            df = query_pnp_df(
                f"""
                SELECT DISTINCT task_id
                FROM pnp_batches
                {where}
                ORDER BY task_id DESC
                LIMIT 100
                """,
                params,
            )
            options = [{"label": str(t), "value": str(t)} for t in df["task_id"] if pd.notnull(t)]
            if selected_value is not None and str(selected_value) not in {str(x["value"]) for x in options}:
                options.append({"label": str(selected_value), "value": str(selected_value)})
            return options
        except Exception:
            return []

    @app.callback(
        [Output("pnp-res-batch-list", "children"),
         Output("pnp-res-batch-page", "data")],
        [Input("pnp-res-batch-refresh-btn", "n_clicks"),
         Input("pnp-res-batch-load-more-btn", "n_clicks")],
        [State("pnp-res-task-filter", "value"),
         State("pnp-res-batch-page", "data"),
         State("pnp-res-batch-list", "children"),
         State("pnp-res-selected-batch", "data")]
    )
    def update_batch_list(refresh_clicks, load_more_clicks, selected_task_id, page, current_children, selected_batch):
        trigger = ctx.triggered_id
        
        # 如果是点击刷新/查询，或者是初次加载，重置分页
        if trigger == "pnp-res-batch-refresh-btn" or trigger is None:
            page = 1
            current_children = []
        elif trigger == "pnp-res-batch-load-more-btn":
            page += 1
            
        limit = 10
        offset = (page - 1) * limit
        
        # 构造查询
        where_clause = ""
        params = {}
        if selected_task_id:
            where_clause = "WHERE task_id = %(task_id)s"
            params["task_id"] = str(selected_task_id)
            
        sql = f"""
            SELECT b.uniq_id, b.task_id, b.sample_ratio, b.created_at,
                   COALESCE(b.status, 'queued') AS status,
                   COALESCE(b.total_episodes, 0) AS total_episodes,
                   COALESCE(b.processed_episodes, 0) AS processed_episodes,
                   COALESCE(b.failed_episodes, 0) AS failed_episodes,
                   b.last_heartbeat
            FROM pnp_batches b
            {where_clause}
            ORDER BY b.created_at DESC
            LIMIT {limit} OFFSET {offset}
        """
        
        try:
            df = query_pnp_df(sql, params)
        except Exception as e:
            # Backward-compatible fallback when new columns are not migrated yet.
            legacy_sql = f"""
                SELECT b.uniq_id, b.task_id, b.sample_ratio, b.created_at,
                       COUNT(p.episode_id) as processed_count
                FROM pnp_batches b
                LEFT JOIN pnp_streams p ON b.uniq_id = p.batch_id
                {where_clause}
                GROUP BY b.uniq_id, b.task_id, b.sample_ratio, b.created_at
                ORDER BY b.created_at DESC
                LIMIT {limit} OFFSET {offset}
            """
            try:
                df = query_pnp_df(legacy_sql, params)
                if not df.empty:
                    df["status"] = "queued"
                    df["total_episodes"] = df["processed_count"]
                    df["processed_episodes"] = df["processed_count"]
                    df["failed_episodes"] = 0
                    df["last_heartbeat"] = None
            except Exception:
                return current_children + [html.Div(f"加载出错: {e}", style={"color": "red"})], page
            
        if df.empty and page == 1:
            return [html.Div("暂无批次记录", style={"color": "#9ca3af", "textAlign": "center", "padding": "20px"})], page
        elif df.empty:
            return current_children, page # 到底了

        def _safe_int(value):
            try:
                if pd.isna(value):
                    return 0
                return int(value)
            except Exception:
                return 0

        def _safe_dt_text(value):
            try:
                if pd.isna(value):
                    return "-"
                return pd.to_datetime(value).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                return str(value) if value is not None else "-"

        # 渲染新的批次卡片
        new_cards = []
        for _, row in df.iterrows():
            uniq_id = str(row["uniq_id"])
            t_id = str(row["task_id"])
            total_eps = _safe_int(row.get("total_episodes"))
            processed_eps = _safe_int(row.get("processed_episodes"))
            failed_eps = _safe_int(row.get("failed_episodes"))
            status = str(row.get("status") or "queued")
            progress = 0.0 if total_eps <= 0 else min(100.0, (processed_eps / total_eps) * 100.0)
            if total_eps <= 0:
                progress_text = "进度: -"
            else:
                progress_text = f"进度: {processed_eps}/{total_eps} ({progress:.1f}%)"
            status_color = {
                "queued": "#6b7280",
                "running": "#2563eb",
                "paused": "#f59e0b",
                "stopping": "#f97316",
                "stopped": "#ef4444",
                "success": "#059669",
                "partial": "#d97706",
                "failed": "#dc2626",
            }.get(status, "#6b7280")
            show_actions = status in {"running", "paused"}
            is_running = status == "running"
            heartbeat = row.get("last_heartbeat")
            heartbeat_text = _safe_dt_text(heartbeat)
            created_at_text = _safe_dt_text(row.get("created_at"))
            card = html.Div(
                [
                    html.Div(f"Batch: {uniq_id}", style={"fontWeight": "600", "fontSize": "13px", "wordBreak": "break-all"}),
                    html.Div(f"Task ID: {t_id}", style={"fontSize": "12px", "color": "#4b5563", "marginTop": "4px"}),
                    html.Div(f"状态: {status}", style={"fontSize": "12px", "color": status_color, "marginTop": "2px", "fontWeight": "600"}),
                    html.Div(progress_text, style={"fontSize": "12px", "color": "#10b981", "marginTop": "2px"}),
                    html.Div(f"失败条数: {failed_eps}", style={"fontSize": "12px", "color": "#ef4444", "marginTop": "2px"}),
                    html.Div(f"心跳: {heartbeat_text}", style={"fontSize": "11px", "color": "#6b7280", "marginTop": "2px"}),
                    html.Div(f"时间: {created_at_text}", style={"fontSize": "11px", "color": "#9ca3af", "marginTop": "4px"}),
                    html.Div(
                        [
                            html.Button(
                                "暂停",
                                id={"type": "pnp-res-batch-action-btn", "index": uniq_id, "action": "pause"},
                                n_clicks=0,
                                disabled=not is_running,
                                style={
                                    "padding": "2px 8px",
                                    "fontSize": "11px",
                                    "borderRadius": "4px",
                                    "border": "1px solid #f59e0b",
                                    "background": "#fff",
                                    "color": "#f59e0b",
                                    "cursor": "pointer",
                                },
                            ),
                            html.Button(
                                "继续",
                                id={"type": "pnp-res-batch-action-btn", "index": uniq_id, "action": "resume"},
                                n_clicks=0,
                                disabled=is_running,
                                style={
                                    "padding": "2px 8px",
                                    "fontSize": "11px",
                                    "borderRadius": "4px",
                                    "border": "1px solid #2563eb",
                                    "background": "#fff",
                                    "color": "#2563eb",
                                    "cursor": "pointer",
                                },
                            ),
                            html.Button(
                                "结束",
                                id={"type": "pnp-res-batch-action-btn", "index": uniq_id, "action": "stop"},
                                n_clicks=0,
                                style={
                                    "padding": "2px 8px",
                                    "fontSize": "11px",
                                    "borderRadius": "4px",
                                    "border": "1px solid #ef4444",
                                    "background": "#fff",
                                    "color": "#ef4444",
                                    "cursor": "pointer",
                                },
                            ),
                        ],
                        style={
                            "display": "flex",
                            "gap": "6px",
                            "marginTop": "8px",
                        },
                    ) if show_actions else html.Div(),
                ],
                id={'type': 'pnp-res-batch-card', 'index': uniq_id},
                style={
                    "padding": "12px",
                    "marginBottom": "10px",
                    "backgroundColor": "#e0f2fe" if uniq_id == selected_batch else "#f9fafb",
                    "border": f"1px solid {'#3b82f6' if uniq_id == selected_batch else '#e5e7eb'}",
                    "borderRadius": "6px",
                    "cursor": "pointer",
                    "transition": "all 0.2s"
                },
                className="hover-card" # 可在后续加 CSS :hover
            )
            new_cards.append(card)
            
        return current_children + new_cards, page

    # --- 2. 选择某个批次 ---
    @app.callback(
        Output("pnp-res-selected-batch", "data"),
        Input({'type': 'pnp-res-batch-card', 'index': ALL}, "n_clicks"),
        prevent_initial_call=True
    )
    def select_batch(n_clicks_list):
        if not ctx.triggered:
            return no_update
            
        trigger = ctx.triggered[0]
        if trigger['value'] is None:
            return no_update
        
        trigger_id = trigger['prop_id'].split('.')[0]
        try:
            trigger_dict = json.loads(trigger_id)
            selected_uniq_id = trigger_dict['index']
            return selected_uniq_id
        except:
            return no_update

    @app.callback(
        Output({'type': 'pnp-res-batch-card', 'index': ALL}, 'style'),
        Input("pnp-res-selected-batch", "data"),
        State({'type': 'pnp-res-batch-card', 'index': ALL}, 'id'),
        State({'type': 'pnp-res-batch-card', 'index': ALL}, 'style'),
        prevent_initial_call=True
    )
    def update_batch_card_styles(selected_batch, ids, styles):
        if not ids or not styles:
             return no_update
        new_styles = []
        for id_dict, style in zip(ids, styles):
            new_style = style.copy() if style else {}
            if id_dict['index'] == selected_batch:
                new_style['backgroundColor'] = '#e0f2fe'
                new_style['border'] = '1px solid #3b82f6'
            else:
                new_style['backgroundColor'] = '#f9fafb'
                new_style['border'] = '1px solid #e5e7eb'
            new_styles.append(new_style)
        return new_styles

    @app.callback(
        [
            Output("pnp-res-action-msg", "children"),
            Output("pnp-res-batch-refresh-btn", "n_clicks", allow_duplicate=True),
        ],
        Input({"type": "pnp-res-batch-action-btn", "index": ALL, "action": ALL}, "n_clicks"),
        State("pnp-res-batch-refresh-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def control_batch_action(_clicks, refresh_clicks):
        trigger = ctx.triggered_id
        if not isinstance(trigger, dict):
            return no_update, no_update
        trigger_val = ctx.triggered[0].get("value") if ctx.triggered else 0
        if not trigger_val:
            return no_update, no_update

        batch_id = str(trigger.get("index", ""))
        action = str(trigger.get("action", ""))
        if not batch_id or action not in {"pause", "resume", "stop"}:
            return no_update, no_update

        try:
            info_df = query_pnp_df(
                """
                SELECT uniq_id, task_id, sample_ratio, is_overwrite, parameters, status
                FROM pnp_batches
                WHERE uniq_id = %s
                LIMIT 1
                """,
                (batch_id,),
            )
            if info_df.empty:
                return html.Span(f"未找到批次 {batch_id}", style={"color": "#ef4444"}), refresh_clicks or 0

            row = info_df.iloc[0]
            current_status = str(row.get("status") or "")

            if action == "pause" and current_status != "running":
                return html.Span("仅 running 批次可暂停", style={"color": "#f59e0b"}), refresh_clicks or 0
            if action == "resume" and current_status != "paused":
                return html.Span("仅 paused 批次可继续", style={"color": "#f59e0b"}), refresh_clicks or 0
            if action == "stop" and current_status not in {"running", "paused", "stopping"}:
                return html.Span("该批次当前不可结束", style={"color": "#f59e0b"}), refresh_clicks or 0

            with get_pnp_connection() as conn:
                with conn.cursor() as cur:
                    if action == "pause":
                        cur.execute(
                            "UPDATE pnp_batches SET status = 'paused', last_heartbeat = CURRENT_TIMESTAMP WHERE uniq_id = %s",
                            (batch_id,),
                        )
                        msg = html.Span(f"批次 {batch_id} 已请求暂停", style={"color": "#2563eb"})
                    elif action == "stop":
                        target_status = "stopping" if current_status == "running" else "stopped"
                        cur.execute(
                            "UPDATE pnp_batches SET status = %s, last_heartbeat = CURRENT_TIMESTAMP WHERE uniq_id = %s",
                            (target_status, batch_id),
                        )
                        msg = html.Span(f"批次 {batch_id} 已请求结束", style={"color": "#ef4444"})
                    else:
                        cur.execute(
                            "UPDATE pnp_batches SET status = 'running', last_heartbeat = CURRENT_TIMESTAMP WHERE uniq_id = %s",
                            (batch_id,),
                        )

                if action == "resume":
                    from rq import Queue
                    from redis import Redis
                    import os
                    from dotenv import load_dotenv
                    from src.workers.pnp_worker import run_pnp_task

                    load_dotenv()
                    redis_conn = Redis(
                        host=os.environ.get("REDIS_HOST", "localhost"),
                        port=int(os.environ.get("REDIS_PORT", 6379)),
                        db=int(os.environ.get("REDIS_DB", 1)),
                        password=os.environ.get("REDIS_PASSWORD", None),
                    )
                    q = Queue("pnp_tasks", connection=redis_conn)
                    params = row.get("parameters")
                    if isinstance(params, str):
                        params = json.loads(params)
                    if not isinstance(params, dict):
                        params = {}
                    overwrite_val = row.get("is_overwrite")
                    overwrite = str(overwrite_val).lower() in {"1", "true", "t", "yes"} if overwrite_val is not None else False
                    q.enqueue(
                        run_pnp_task,
                        args=(
                            batch_id,
                            str(row.get("task_id")),
                            int(row.get("sample_ratio") or 0),
                            overwrite,
                            params,
                        ),
                        job_timeout=3600,
                    )
                    msg = html.Span(f"批次 {batch_id} 已继续执行", style={"color": "#059669"})

            return msg, (refresh_clicks or 0) + 1
        except Exception as e:
            return html.Span(f"批次操作失败: {e}", style={"color": "#ef4444"}), refresh_clicks or 0

    @app.callback(
        Output("pnp-res-show-failed", "data"),
        Input("pnp-res-failed-toggle-btn", "n_clicks"),
        State("pnp-res-show-failed", "data"),
        prevent_initial_call=True,
    )
    def toggle_failed_records(n_clicks, show_failed):
        if not n_clicks:
            return no_update
        return not bool(show_failed)

    @app.callback(
        [
            Output("pnp-res-failed-toggle-btn", "children"),
            Output("pnp-res-failed-toggle-btn", "outline"),
            Output("pnp-res-selected-episode", "data", allow_duplicate=True),
        ],
        Input("pnp-res-show-failed", "data"),
        prevent_initial_call=True,
    )
    def sync_failed_toggle_ui(show_failed):
        if show_failed:
            return "返回正常记录", False, None
        return "失败记录", True, None

    @app.callback(
        Output("pnp-res-selected-episode", "data", allow_duplicate=True),
        [
            Input("pnp-res-selected-batch", "data"),
            Input("pnp-res-show-failed", "data"),
            Input("pnp-res-episode-search", "value"),
        ],
        prevent_initial_call=True,
    )
    def reset_selected_episode_on_filters_change(_selected_batch, _show_failed, _search_value):
        return None

    @app.callback(
        Output("pnp-res-show-failed", "data", allow_duplicate=True),
        Input("pnp-res-selected-batch", "data"),
        prevent_initial_call=True,
    )
    def reset_failed_toggle_on_batch_change(_selected_batch):
        return False

    # --- 3. 加载选定批次内的 Episode (中下列表) ---
    @app.callback(
        [Output("pnp-res-episode-list", "children"),
         Output("pnp-res-episode-page", "data")],
        [Input("pnp-res-selected-batch", "data"),
         Input("pnp-res-episode-load-more-btn", "n_clicks"),
         Input("pnp-res-show-failed", "data"),
         Input("pnp-res-episode-search", "value")],
        [State("pnp-res-episode-page", "data"),
         State("pnp-res-episode-list", "children"),
         State("pnp-res-selected-episode", "data")],
        prevent_initial_call=True
    )
    def update_episode_list(selected_batch, load_more_clicks, show_failed, search_value, page, current_children, selected_episode):
        trigger = ctx.triggered_id
        
        if not selected_batch:
            return [], 1
            
        if trigger in {"pnp-res-selected-batch", "pnp-res-show-failed", "pnp-res-episode-search"}:
            page = 1
            current_children = []
        elif trigger == "pnp-res-episode-load-more-btn":
            page += 1
            
        limit = 20
        offset = (page - 1) * limit
        search_value = (search_value or "").strip()
        
        if show_failed:
            where_clause = "WHERE batch_id = %s"
            params = [selected_batch]
            if search_value:
                where_clause += " AND CAST(episode_id AS TEXT) ILIKE %s"
                params.append(f"%{search_value}%")
            sql = f"""
                SELECT episode_id, error_message, failed_at
                FROM pnp_failures
                {where_clause}
                ORDER BY failed_at DESC
                LIMIT {limit} OFFSET {offset}
            """
        else:
            where_clause = "WHERE batch_id = %s"
            params = [selected_batch]
            if search_value:
                where_clause += " AND CAST(episode_id AS TEXT) ILIKE %s"
                params.append(f"%{search_value}%")
            # 查询批次下的正常检测记录
            sql = f"""
                SELECT episode_id, right_pnp_result, left_pnp_result, checked_at
                FROM pnp_streams
                {where_clause}
                ORDER BY checked_at DESC
                LIMIT {limit} OFFSET {offset}
            """
        try:
            df = query_pnp_df(sql, tuple(params))
        except Exception as e:
            return current_children + [html.Div(f"加载出错: {e}", style={"color": "red"})], page
            
        if df.empty and page == 1:
            if show_failed:
                try:
                    batch_df = query_pnp_df(
                        """
                        SELECT failed_episodes, error_message
                        FROM pnp_batches
                        WHERE uniq_id = %s
                        LIMIT 1
                        """,
                        (selected_batch,),
                    )
                    if not batch_df.empty and int(batch_df.iloc[0].get("failed_episodes") or 0) > 0:
                        err = str(batch_df.iloc[0].get("error_message") or "未知错误")
                        m = re.search(r"Episode\\s+(\\d+)", err)
                        ep_text = m.group(1) if m else "未知"
                        fallback_item = html.Div(
                            [
                                html.Div(
                                    [
                                        html.Span(f"Episode: {ep_text}", style={"fontWeight": "600", "fontSize": "14px"}),
                                        html.Span("时长: 未知", style={"fontSize": "12px", "color": "#6b7280"}),
                                    ],
                                    style={"display": "flex", "justifyContent": "space-between", "marginBottom": "6px"},
                                ),
                                html.Div(
                                    f"失败原因: {err}",
                                    style={"fontSize": "12px", "color": "#ef4444", "wordBreak": "break-all"},
                                ),
                                html.Div(
                                    "说明: 该批次失败明细未落表（可能是旧worker或历史批次），当前为回退展示",
                                    style={"fontSize": "11px", "color": "#9ca3af", "marginTop": "4px"},
                                ),
                            ],
                            style={
                                "padding": "12px",
                                "marginBottom": "8px",
                                "backgroundColor": "#fef2f2",
                                "border": "1px solid #fecaca",
                                "borderRadius": "6px",
                            },
                        )
                        return [fallback_item], page
                except Exception:
                    pass
            if search_value:
                empty_text = f"未找到 ID 包含 “{search_value}” 的失败记录" if show_failed else f"未找到 ID 包含 “{search_value}” 的检测记录"
            else:
                empty_text = "该批次暂无失败记录" if show_failed else "该批次暂无检测记录"
            return [html.Div(empty_text, style={"color": "#9ca3af", "textAlign": "center"})], page
        elif df.empty:
            return current_children, page
            
        # 查询原数据库获取每个 episode 的视频时长或附加信息（比如从 streams 或 episodes 表中关联）
        # 为了高效，我们可以用这个批量去查
        episode_ids = tuple(str(x) for x in df['episode_id'].tolist())
        episode_meta = {}
        if episode_ids:
            try:
                # 尝试获取轨迹时长等信息
                meta_sql = f"SELECT id, trajectory_start, trajectory_end FROM episodes WHERE id IN %s"
                meta_df = query_df(meta_sql, (episode_ids,))
                for _, mrow in meta_df.iterrows():
                    dur = "未知"
                    if pd.notnull(mrow['trajectory_start']) and pd.notnull(mrow['trajectory_end']):
                        dur = f"{(mrow['trajectory_end'] - mrow['trajectory_start']).total_seconds():.1f}s"
                    episode_meta[str(mrow['id'])] = dur
            except:
                pass

        new_items = []
        for _, row in df.iterrows():
            ep_id = str(row['episode_id'])
            
            duration = episode_meta.get(ep_id, "未知")

            if show_failed:
                error_message = str(row.get("error_message") or "未知错误")
                ts = row.get("failed_at")
                ts_text = ts.strftime("%Y-%m-%d %H:%M") if pd.notnull(ts) else "未知时间"
                item = html.Div(
                    [
                        html.Div(
                            [
                                html.Span(f"Episode: {ep_id}", style={"fontWeight": "600", "fontSize": "14px"}),
                                html.Span(f"时长: {duration}", style={"fontSize": "12px", "color": "#6b7280"}),
                            ],
                            style={"display": "flex", "justifyContent": "space-between", "marginBottom": "6px"},
                        ),
                        html.Div(
                            f"失败原因: {error_message}",
                            style={"fontSize": "12px", "color": "#ef4444", "wordBreak": "break-all"},
                        ),
                        html.Div(f"失败时间: {ts_text}", style={"fontSize": "11px", "color": "#9ca3af", "marginTop": "4px"}),
                    ],
                    id={"type": "pnp-res-episode-card", "index": ep_id},
                    style={
                        "padding": "12px",
                        "marginBottom": "8px",
                        "backgroundColor": "#fee2e2" if ep_id == selected_episode else "#fef2f2",
                        "border": f"1px solid {'#ef4444' if ep_id == selected_episode else '#fecaca'}",
                        "borderRadius": "6px",
                        "cursor": "pointer",
                    },
                )
                new_items.append(item)
                continue

            # 简单统计一下左右手分别发现了多少次
            r_val = row['right_pnp_result']
            l_val = row['left_pnp_result']
            r_res = r_val if isinstance(r_val, list) else (json.loads(r_val) if r_val else [])
            l_res = l_val if isinstance(l_val, list) else (json.loads(l_val) if l_val else [])
            r_count = len(r_res)
            l_count = len(l_res)

            item = html.Div(
                [
                    html.Div(
                        [
                            html.Span(f"Episode: {ep_id}", style={"fontWeight": "600", "fontSize": "14px"}),
                            html.Span(f"时长: {duration}", style={"fontSize": "12px", "color": "#6b7280"}),
                        ],
                        style={"display": "flex", "justifyContent": "space-between", "marginBottom": "6px"}
                    ),
                    html.Div(
                        [
                            html.Span(f"右手检测到 {r_count} 次 PnP", style={"fontSize": "12px", "color": "#3b82f6", "marginRight": "12px"}),
                            html.Span(f"左手检测到 {l_count} 次 PnP", style={"fontSize": "12px", "color": "#10b981"}),
                        ]
                    ),
                    html.Div(f"检测时间: {row['checked_at'].strftime('%Y-%m-%d %H:%M')}", style={"fontSize": "11px", "color": "#9ca3af", "marginTop": "4px"}),
                ],
                id={'type': 'pnp-res-episode-card', 'index': ep_id},
                style={
                    "padding": "12px",
                    "marginBottom": "8px",
                    "backgroundColor": "#e0f2fe" if ep_id == selected_episode else "#f9fafb",
                    "border": f"1px solid {'#3b82f6' if ep_id == selected_episode else '#e5e7eb'}",
                    "borderRadius": "6px",
                    "cursor": "pointer"
                }
            )
            new_items.append(item)
            
        return current_children + new_items, page

    # --- 4. 选择某个 Episode ---
    @app.callback(
        Output("pnp-res-selected-episode", "data"),
        Input({'type': 'pnp-res-episode-card', 'index': ALL}, "n_clicks"),
        prevent_initial_call=True
    )
    def select_episode(n_clicks_list):
        if not ctx.triggered:
            return no_update
            
        trigger = ctx.triggered[0]
        if trigger['value'] is None:
            return no_update
            
        trigger_id = trigger['prop_id'].split('.')[0]
        try:
            trigger_dict = json.loads(trigger_id)
            return trigger_dict['index']
        except:
            return no_update

    @app.callback(
        Output({'type': 'pnp-res-episode-card', 'index': ALL}, 'style'),
        Input("pnp-res-selected-episode", "data"),
        State("pnp-res-show-failed", "data"),
        State({'type': 'pnp-res-episode-card', 'index': ALL}, 'id'),
        State({'type': 'pnp-res-episode-card', 'index': ALL}, 'style'),
        prevent_initial_call=True
    )
    def update_episode_card_styles(selected_episode, show_failed, ids, styles):
        if not ids or not styles:
             return no_update
        new_styles = []
        for id_dict, style in zip(ids, styles):
            new_style = style.copy() if style else {}
            if id_dict['index'] == selected_episode:
                if show_failed:
                    new_style['backgroundColor'] = '#fee2e2'
                    new_style['border'] = '1px solid #ef4444'
                else:
                    new_style['backgroundColor'] = '#e0f2fe'
                    new_style['border'] = '1px solid #3b82f6'
            else:
                if show_failed:
                    new_style['backgroundColor'] = '#fef2f2'
                    new_style['border'] = '1px solid #fecaca'
                else:
                    new_style['backgroundColor'] = '#f9fafb'
                    new_style['border'] = '1px solid #e5e7eb'
            new_styles.append(new_style)
        return new_styles

    # --- 5. 更新视频和时间轴 ---
    @app.callback(
        [Output("pnp-res-video-container", "children"),
         Output("pnp-res-timeline-container", "children")],
        Input("pnp-res-selected-episode", "data"),
        State("pnp-res-selected-batch", "data"),
        prevent_initial_call=True
    )
    def update_video_and_timeline(episode_id, batch_id):
        if not episode_id or not batch_id:
            return no_update, no_update
            
        # 1) 查视频URL
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
                        video_elem = html.Video(id="pnp-res-video-player", src=video_url, controls=True, style={"width": "100%", "maxHeight": "500px", "backgroundColor": "#000"})
                    else:
                        video_elem = html.Video(id="pnp-res-video-player", src=f"/pnp_video?path={video_url}", controls=True, style={"width": "100%", "maxHeight": "500px", "backgroundColor": "#000"})
                else:
                    video_elem = html.Div("视频解析失败", style={"color": "red"})
        except Exception as e:
            video_elem = html.Div(f"视频查询异常 {e}", style={"color": "red"})
            
        # 2) 查 PnP 结果画时间轴
        pnp_sql = "SELECT right_pnp_result, left_pnp_result FROM pnp_streams WHERE episode_id = %s AND batch_id = %s"
        try:
            pnp_df = query_pnp_df(pnp_sql, (episode_id, batch_id))
            if pnp_df.empty:
                timeline_elem = html.Div("未查询到 PnP 结果")
            else:
                r_val = pnp_df.iloc[0]['right_pnp_result']
                l_val = pnp_df.iloc[0]['left_pnp_result']
                right_res = r_val if isinstance(r_val, list) else (json.loads(r_val) if r_val else [])
                left_res = l_val if isinstance(l_val, list) else (json.loads(l_val) if l_val else [])
                
                timeline_elem = html.Div(
                    id="custom-timeline-wrapper",
                    **{"data-right": json.dumps(right_res), "data-left": json.dumps(left_res)},
                    style={
                        "position": "relative", "width": "100%", "height": "56px", 
                        "backgroundColor": "#f9fafb", "borderRadius": "6px", 
                        "border": "1px solid #e5e7eb", "cursor": "pointer",
                        "marginTop": "10px", "overflow": "hidden"
                    },
                    children=[
                        html.Div("右手", style={"position": "absolute", "left": "5px", "top": "7px", "fontSize": "11px", "fontWeight": "600", "color": "#6b7280", "pointerEvents": "none", "zIndex": 10}),
                        html.Div("左手", style={"position": "absolute", "left": "5px", "top": "31px", "fontSize": "11px", "fontWeight": "600", "color": "#6b7280", "pointerEvents": "none", "zIndex": 10}),
                        html.Div(id="right-hand-tracks", style={"position": "absolute", "left": "0", "width": "100%", "top": "5px", "height": "18px", "pointerEvents": "none"}),
                        html.Div(id="left-hand-tracks", style={"position": "absolute", "left": "0", "width": "100%", "top": "29px", "height": "18px", "pointerEvents": "none"}),
                        html.Div(id="timeline-playhead", style={"position": "absolute", "top": "0", "bottom": "0", "left": "0%", "width": "2px", "backgroundColor": "#ef4444", "pointerEvents": "none", "zIndex": 20})
                    ]
                )
        except Exception as e:
            timeline_elem = html.Div(f"时间轴异常 {e}", style={"color": "red"})
            
        return video_elem, timeline_elem
