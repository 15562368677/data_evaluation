"""数据库连接模块 (Source Data)"""

import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "dataplatform-backend-pgsql.fftaicorp.com"),
    "port": int(os.getenv("DB_PORT", "30673")),
    "user": os.getenv("DB_USER", "readonly_user"),
    "password": os.getenv("DB_PASSWORD", "fftai2015"),
    "database": os.getenv("DB_NAME", "data_collection"),
}


def get_connection():
    """获取数据库连接"""
    return psycopg2.connect(**DB_CONFIG)


def query_df(sql: str, params=None):
    """执行查询并返回 DataFrame"""
    import pandas as pd

    with get_connection() as conn:
        return pd.read_sql(sql, conn, params=params)
