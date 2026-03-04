"""Callbacks for PnP Result Page"""

import json
from dash import Input, Output, State, ALL, MATCH, ctx, html, dcc, no_update
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from src.utils.source_db import query_df
from src.utils.result_db import query_pnp_df
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
        [Output("pnp-res-batch-list", "children"),
         Output("pnp-res-batch-page", "data")],
        [Input("pnp-res-batch-refresh-btn", "n_clicks"),
         Input("pnp-res-batch-load-more-btn", "n_clicks")],
        [State("pnp-res-task-search-input", "value"),
         State("pnp-res-batch-page", "data"),
         State("pnp-res-batch-list", "children"),
         State("pnp-res-selected-batch", "data")]
    )
    def update_batch_list(refresh_clicks, load_more_clicks, search_val, page, current_children, selected_batch):
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
        if search_val:
            where_clause = "WHERE task_id ILIKE %(search)s OR uniq_id ILIKE %(search)s"
            params["search"] = f"%{search_val}%"
            
        sql = f"""
            SELECT b.uniq_id, b.task_id, b.sample_ratio, b.created_at,
                   COUNT(p.episode_id) as processed_count
            FROM batches b
            LEFT JOIN pnp_streams p ON b.uniq_id = p.batch_id
            {where_clause}
            GROUP BY b.uniq_id, b.task_id, b.sample_ratio, b.created_at
            ORDER BY b.created_at DESC
            LIMIT {limit} OFFSET {offset}
        """
        
        try:
            df = query_pnp_df(sql, params)
        except Exception as e:
            return current_children + [html.Div(f"加载出错: {e}", style={"color": "red"})], page
            
        if df.empty and page == 1:
            return [html.Div("暂无批次记录", style={"color": "#9ca3af", "textAlign": "center", "padding": "20px"})], page
        elif df.empty:
            return current_children, page # 到底了

        # 渲染新的批次卡片
        new_cards = []
        for _, row in df.iterrows():
            uniq_id = str(row["uniq_id"])
            t_id = str(row["task_id"])
            
            card = html.Div(
                [
                    html.Div(f"Batch: {uniq_id}", style={"fontWeight": "600", "fontSize": "13px", "wordBreak": "break-all"}),
                    html.Div(f"Task ID: {t_id}", style={"fontSize": "12px", "color": "#4b5563", "marginTop": "4px"}),
                    html.Div(f"已检测条数: {row['processed_count']}", style={"fontSize": "12px", "color": "#10b981", "marginTop": "2px"}),
                    html.Div(f"时间: {row['created_at'].strftime('%Y-%m-%d %H:%M:%S')}", style={"fontSize": "11px", "color": "#9ca3af", "marginTop": "4px"}),
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

    # --- 3. 加载选定批次内的 Episode (中下列表) ---
    @app.callback(
        [Output("pnp-res-episode-list", "children"),
         Output("pnp-res-episode-page", "data")],
        [Input("pnp-res-selected-batch", "data"),
         Input("pnp-res-episode-load-more-btn", "n_clicks")],
        [State("pnp-res-episode-page", "data"),
         State("pnp-res-episode-list", "children"),
         State("pnp-res-selected-episode", "data")],
        prevent_initial_call=True
    )
    def update_episode_list(selected_batch, load_more_clicks, page, current_children, selected_episode):
        trigger = ctx.triggered_id
        
        if not selected_batch:
            return [], 1
            
        if trigger == "pnp-res-selected-batch":
            page = 1
            current_children = []
        elif trigger == "pnp-res-episode-load-more-btn":
            page += 1
            
        limit = 20
        offset = (page - 1) * limit
        
        # 查询批次下的 episodes
        sql = f"""
            SELECT episode_id, right_pnp_result, left_pnp_result, checked_at 
            FROM pnp_streams 
            WHERE batch_id = %s
            ORDER BY checked_at DESC
            LIMIT {limit} OFFSET {offset}
        """
        try:
            df = query_pnp_df(sql, (selected_batch,))
        except Exception as e:
            return current_children + [html.Div(f"加载出错: {e}", style={"color": "red"})], page
            
        if df.empty and page == 1:
            return [html.Div("该批次暂无检测记录", style={"color": "#9ca3af", "textAlign": "center"})], page
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
            
            # 简单统计一下左右手分别发现了多少次
            r_val = row['right_pnp_result']
            l_val = row['left_pnp_result']
            r_res = r_val if isinstance(r_val, list) else (json.loads(r_val) if r_val else [])
            l_res = l_val if isinstance(l_val, list) else (json.loads(l_val) if l_val else [])
            r_count = len(r_res)
            l_count = len(l_res)
            
            duration = episode_meta.get(ep_id, "未知")
            
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
        State({'type': 'pnp-res-episode-card', 'index': ALL}, 'id'),
        State({'type': 'pnp-res-episode-card', 'index': ALL}, 'style'),
        prevent_initial_call=True
    )
    def update_episode_card_styles(selected_episode, ids, styles):
        if not ids or not styles:
             return no_update
        new_styles = []
        for id_dict, style in zip(ids, styles):
            new_style = style.copy() if style else {}
            if id_dict['index'] == selected_episode:
                new_style['backgroundColor'] = '#e0f2fe'
                new_style['border'] = '1px solid #3b82f6'
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
