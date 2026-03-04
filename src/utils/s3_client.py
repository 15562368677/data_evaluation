"""S3 (MinIO) 客户端工具，用于从 farts-data 桶下载数据。"""

import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from minio import Minio

load_dotenv()

S3_ENDPOINT = os.getenv("S3_FAST_MINIO_ENDPOINT", "s3.fftaicorp.com")
S3_ACCESS_KEY = os.getenv("S3_FAST_MINIO_ACCESS_KEY", "wd9vaB5TTxdcVpQpMPCNXEvxkLF2ayt0")
S3_SECRET_KEY = os.getenv("S3_FAST_MINIO_SECRET_KEY", "arjpox9otwxd53qppgxzuwcbtqrmfq45")
S3_REGION = os.getenv("S3_FAST_MINIO_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_FAST_MINIO_BUCKET", "farts-data")
S3_SECURE = os.getenv("S3_FAST_SECURE", "true").lower() == "true"

# 本地缓存目录
CACHE_DIR = Path(tempfile.gettempdir()) / "pnp_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _get_minio_client() -> Minio:
    """获取 MinIO 客户端实例。"""
    return Minio(
        endpoint=S3_ENDPOINT,
        access_key=S3_ACCESS_KEY,
        secret_key=S3_SECRET_KEY,
        region=S3_REGION,
        secure=S3_SECURE,
    )


def s3_object_exists(key: str) -> bool:
    """检查 S3 对象是否存在。"""
    client = _get_minio_client()
    try:
        client.stat_object(S3_BUCKET, key)
        return True
    except Exception:
        return False


def download_s3_file(key: str, force: bool = False) -> Path | None:
    """从 S3 下载文件到本地缓存，返回本地路径。已缓存则跳过。"""
    local_path = CACHE_DIR / key.replace("/", os.sep)
    if local_path.exists() and not force:
        return local_path

    client = _get_minio_client()
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        client.fget_object(S3_BUCKET, key, str(local_path))
        return local_path
    except Exception as e:
        print(f"[S3] 下载失败 {key}: {e}")
        return None


def generate_presigned_url(key: str, expires_in: int = 3600) -> str | None:
    """生成 S3 预签名 URL，用于前端直接播放视频。"""
    from datetime import timedelta

    client = _get_minio_client()
    try:
        url = client.presigned_get_object(
            S3_BUCKET,
            key,
            expires=timedelta(seconds=expires_in),
        )
        return url
    except Exception as e:
        print(f"[S3] 生成预签名 URL 失败 {key}: {e}")
        return None
