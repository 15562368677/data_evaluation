import re

with open('src/ui/pnp/layout.py', 'r') as f:
    content = f.read()

grouped_layout = """    # PNP 分组显示结构
    from dash import html, dcc
    
    # 互锁关联配置
    # 注意：这里的字典 key 仅用于标识
    param_groups = [
        {
            "title": "🔍 Pick 检测条件",
            "items": [
                {"id": "pick_closure_threshold", "label": "Pick 闭合度阈值", "desc": "判定抓取成功的最小闭合程度"},
                {"id": "pick_start_offset", "label": "Pick 截取偏移帧数", "desc": "从判定点向前推的帧数"},
            ]
        },
        {
            "title": "🫳 Place 检测条件",
            "items": [
                {"id": "place_closure_threshold", "label": "Place 闭合度阈值", "desc": "判定松开的闭合程度上限"},
                {"id": "place_end_offset", "label": "Place 截取偏移帧数", "desc": "从判定点向后推的帧数"},
                {"id": "place_diff_lookahead", "label": "Place 差值前看帧数", "desc": "判断未来多少帧内差值不足以构成抓取"},
            ]
        },
        {
            "title": "⚙️ 速度检测条件 (Pick & Place 互锁)",
            "items": [
                {"id": "pick_velocity_threshold", "label": "Pick 动作速度阈值", "desc": "暂未使用，填 0"},
                {"id": "place_velocity_threshold", "label": "Place 动作速度阈值", "desc": "松开动作的最小速度"},
                {"id": "pick_velocity_lookback", "label": "Pick 速度回溯帧数", "desc": "计算速度往前回溯的窗口"},
                {"id": "place_velocity_lookback", "label": "Place 速度回溯帧数", "desc": "计算速度往前回溯的窗口(互锁)"},
                {"id": "pick_velocity_lookahead", "label": "Pick 速度前看帧数", "desc": "计算速度往后查看的窗口"},
                {"id": "place_velocity_lookahead", "label": "Place 速度前看帧数", "desc": "计算速度往后查看的窗口(互锁)"},
            ]
        },
        {
            "title": "📐 关节差值与斜率 (基础条件)",
            "items": [
                {"id": "negative_diff_threshold", "label": "四指差值阈值 (<0)", "desc": "四指(方向系数-1)的差值需小于此值"},
                {"id": "positive_diff_threshold", "label": "拇指差值阈值 (>0)", "desc": "拇指(方向系数+1)的差值需大于此值"},
                {"id": "min_joints_for_diff", "label": "最少满足差值关节数", "desc": "几个关节满足差值才算有效"},
                {"id": "slope_threshold", "label": "斜率阈值", "desc": "判定关节角度稳定的最大斜率变化"},
                {"id": "slope_lookahead", "label": "斜率计算跨度", "desc": "计算斜率时前后使用的帧数"},
            ]
        }
    ]

    param_inputs = []
    
    for group in param_groups:
        group_items = []
        for item in group["items"]:
            key = item["id"]
            value = PNP_DEFAULT_PARAMS.get(key, 0)
            group_items.append(
                html.Div(
                    [
                        html.Label(item["label"], style={"fontSize": "12px", "color": "#333", "fontWeight": "600", "marginBottom": "2px", "display": "block"}),
                        html.Div(item["desc"], style={"fontSize": "10px", "color": "#888", "marginBottom": "6px"}),
                        dcc.Input(
                            id=f"pnp-param-{key}",
                            type="number",
                            value=value,
                            style={
                                "width": "100%",
                                "height": "30px",
                                "padding": "0 8px",
                                "borderRadius": "4px",
                                "border": "1px solid #d1d5db",
                                "fontSize": "12px",
                            }
                        )
                    ],
                    style={"marginBottom": "12px", "padding": "8px", "backgroundColor": "#fff", "borderRadius": "4px", "border": "1px solid #f0f0f0"}
                )
            )
            
        param_inputs.append(
            html.Div(
                [
                    html.Div(group["title"], style={"fontSize": "13px", "fontWeight": "bold", "color": "#4f46e5", "marginBottom": "10px", "marginTop": "8px"}),
                    html.Div(group_items)
                ],
                style={"marginBottom": "16px", "padding": "10px", "backgroundColor": "#f8fafc", "borderRadius": "6px", "border": "1px solid #e2e8f0"}
            )
        )
"""

content = re.sub(r'    # 生成右侧参数表单.*?    return html\.Div\(', grouped_layout + '\n    return html.Div(', content, flags=re.DOTALL)

with open('src/ui/pnp/layout.py', 'w') as f:
    f.write(content)
