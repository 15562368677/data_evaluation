"""SQL 查询页面回调。"""

import re
import time
import json
from datetime import date, datetime

import pandas as pd
from dash import Input, Output, State, ctx, dash_table, html, ALL
import dash_bootstrap_components as dbc

from src.utils.result_db import query_pnp_df

_READONLY_START_RE = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)
_BLOCKED_KEYWORDS_RE = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|create|grant|revoke|comment|vacuum|analyze)\b",
    re.IGNORECASE,
)
_MAX_ROWS = 1000


def _quote_ident(name):
    parts = [p for p in str(name).split(".") if p]
    return ".".join('"' + p.replace('"', '""') + '"' for p in parts)


def _placeholder(text):
    return html.Div(
        text,
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
    )


def _to_dash_cell(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, (list, tuple, dict, set)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _validate_readonly_sql(raw_sql):
    if not raw_sql or not raw_sql.strip():
        return False, "SQL 不能为空。"

    statements = [seg.strip() for seg in raw_sql.split(";") if seg.strip()]
    if len(statements) != 1:
        return False, "仅允许执行一条 SQL 语句。"

    sql = statements[0]
    if not _READONLY_START_RE.match(sql):
        return False, "仅允许 SELECT / WITH 开头的只读查询。"

    if _BLOCKED_KEYWORDS_RE.search(sql):
        return False, "检测到写入/DDL 关键字，已阻止执行。"

    return True, sql


def _build_table(df):
    if df.empty:
        return _placeholder("查询成功，返回 0 行。")

    truncated = len(df) > _MAX_ROWS
    view_df = df.head(_MAX_ROWS).copy() if truncated else df.copy()
    view_df.columns = [str(c) for c in view_df.columns]
    view_df = view_df.astype(object).where(pd.notna(view_df), None)
    for col in view_df.columns:
        view_df[col] = view_df[col].map(_to_dash_cell)

    tip = None
    if truncated:
        tip = html.Div(
            f"结果共 {len(df)} 行，仅展示前 {_MAX_ROWS} 行。",
            style={"fontSize": "12px", "color": "#b45309", "marginBottom": "8px"},
        )

    table = dash_table.DataTable(
        columns=[{"name": c, "id": c} for c in view_df.columns],
        data=view_df.to_dict("records"),
        page_action="native",
        page_size=20,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto", "maxHeight": "70vh", "overflowY": "auto"},
        style_cell={
            "textAlign": "left",
            "fontSize": "12px",
            "padding": "8px",
            "minWidth": "100px",
            "maxWidth": "500px",
            "whiteSpace": "normal",
            "height": "auto",
        },
        style_header={
            "backgroundColor": "#f3f4f6",
            "fontWeight": "600",
            "border": "1px solid #e5e7eb",
        },
        style_data={"border": "1px solid #f1f5f9"},
    )

    if tip:
        return html.Div([tip, table])
    return table


def register_callbacks(app):
    @app.callback(
        Output("sql-query-table-list", "children"),
        Input("url", "pathname"),
    )
    def load_result_tables(pathname):
        if pathname != "/sql_query":
            return html.Div("加载中...", style={"fontSize": "12px", "color": "#9ca3af", "padding": "6px"})

        sql = """
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_type = 'BASE TABLE'
              AND table_schema NOT IN ('pg_catalog', 'information_schema')
            ORDER BY table_schema, table_name
        """
        try:
            df = query_pnp_df(sql)
        except Exception as exc:
            return html.Div(
                f"读取结果库表失败: {exc}",
                style={"fontSize": "12px", "color": "#dc2626", "padding": "6px"},
            )

        if df.empty:
            return html.Div("结果库中没有可用表。", style={"fontSize": "12px", "color": "#6b7280", "padding": "6px"})

        cards = []
        for _, row in df.iterrows():
            schema = str(row["table_schema"])
            table = str(row["table_name"])
            query_target = table if schema == "public" else f"{schema}.{table}"
            cards.append(
                dbc.Button(
                    query_target,
                    id={"type": "sql-query-table-btn", "table": query_target},
                    n_clicks=0,
                    color="light",
                    className="w-100",
                    style={
                        "textAlign": "left",
                        "marginBottom": "8px",
                        "border": "1px solid #e5e7eb",
                        "backgroundColor": "#fff",
                        "fontSize": "12px",
                        "color": "#111827",
                    },
                )
            )
        return cards

    @app.callback(
        Output("sql-query-message", "children"),
        Output("sql-query-result-container", "children"),
        Output("sql-query-input", "value"),
        Input("sql-query-run-btn", "n_clicks"),
        Input("sql-query-clear-btn", "n_clicks"),
        Input({"type": "sql-query-table-btn", "table": ALL}, "n_clicks"),
        State("sql-query-input", "value"),
        prevent_initial_call=True,
    )
    def run_sql_query(run_clicks, clear_clicks, _table_clicks, raw_sql):
        trigger = ctx.triggered_id

        if trigger == "sql-query-clear-btn":
            return (
                html.Span("已清空。", style={"color": "#6b7280"}),
                _placeholder("输入 SQL 后点击“执行查询”。"),
                "",
            )

        sql = raw_sql
        if isinstance(trigger, dict) and trigger.get("type") == "sql-query-table-btn":
            table_name = str(trigger.get("table", "")).strip()
            if not table_name:
                return (
                    html.Span("表名无效。", style={"color": "#dc2626"}),
                    _placeholder("SQL 未执行。"),
                    raw_sql,
                )
            sql = f"SELECT * FROM {_quote_ident(table_name)} LIMIT 100"
        elif trigger == "sql-query-run-btn":
            if not run_clicks:
                return (
                    html.Span("请输入 SQL 并执行。", style={"color": "#6b7280"}),
                    _placeholder("输入 SQL 后点击“执行查询”。"),
                    raw_sql,
                )
        else:
            return (
                html.Span("请输入 SQL 并执行。", style={"color": "#6b7280"}),
                _placeholder("输入 SQL 后点击“执行查询”。"),
                raw_sql,
            )

        ok, parsed_or_msg = _validate_readonly_sql(sql)
        if not ok:
            return (
                html.Span(parsed_or_msg, style={"color": "#dc2626"}),
                _placeholder("SQL 未执行。"),
                sql,
            )

        sql = parsed_or_msg
        start = time.perf_counter()
        try:
            df = query_pnp_df(sql)
        except Exception as exc:
            return (
                html.Span(f"执行失败: {exc}", style={"color": "#dc2626"}),
                _placeholder("查询失败。"),
                sql,
            )

        elapsed_ms = (time.perf_counter() - start) * 1000
        msg = html.Span(
            f"执行成功：{len(df)} 行，{len(df.columns)} 列，耗时 {elapsed_ms:.1f} ms。",
            style={"color": "#059669"},
        )
        return msg, _build_table(df), sql
