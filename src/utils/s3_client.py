"""S3 (MinIO) 客户端工具，用于从 farts-data 桶下载数据。"""

import os
import shutil
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
JOINT_CACHE_DIR = CACHE_DIR / "joints"
VIDEO_CACHE_DIR = CACHE_DIR / "video"
CACHE_KIND_DIRS = {
    "joints": JOINT_CACHE_DIR,
    "video": VIDEO_CACHE_DIR,
}
CACHE_MAX_EPISODES = {
    "joints": 1000,
    "video": 50,
}

for cache_dir in CACHE_KIND_DIRS.values():
    cache_dir.mkdir(parents=True, exist_ok=True)


def _get_cache_root(cache_kind: str) -> Path:
    if cache_kind not in CACHE_KIND_DIRS:
        raise ValueError(f"Unsupported cache kind: {cache_kind}")
    return CACHE_KIND_DIRS[cache_kind]


def _get_episode_relative_dir(key: str) -> Path:
    normalized = key.strip("/")
    suffixes = [
        "/top/rgb/video.mp4",
        "/top/depth/video.mp4",
        "/action.parquet",
        "/observation.state.parquet",
        "/observation.images.camera_top.parquet",
        "/observation.images.camera_top_depth.parquet",
        "/data.hdf5",
    ]
    for suffix in suffixes:
        if normalized.endswith(suffix):
            return Path(normalized[: -len(suffix)])

    path = Path(normalized)
    return path.parent if path.parent != Path(".") else path


def _touch_episode_cache(cache_kind: str, key: str) -> Path:
    episode_dir = _get_cache_root(cache_kind) / _get_episode_relative_dir(key)
    episode_dir.mkdir(parents=True, exist_ok=True)
    access_marker = episode_dir / ".last_access"
    access_marker.touch()
    return episode_dir


def _iter_episode_dirs(cache_root: Path) -> list[Path]:
    marker_files = list(cache_root.rglob(".last_access"))
    if marker_files:
        return [marker.parent for marker in marker_files]

    episode_dirs = []
    for file_path in cache_root.rglob("*"):
        if file_path.is_file():
            episode_dirs.append(file_path.parent)
    return list({path for path in episode_dirs})


def _prune_cache(cache_kind: str) -> None:
    cache_root = _get_cache_root(cache_kind)
    max_episodes = CACHE_MAX_EPISODES[cache_kind]
    episode_dirs = _iter_episode_dirs(cache_root)

    if len(episode_dirs) <= max_episodes:
        return

    episode_dirs.sort(
        key=lambda path: (path / ".last_access").stat().st_mtime if (path / ".last_access").exists() else path.stat().st_mtime
    )
    remove_count = len(episode_dirs) - max_episodes
    for episode_dir in episode_dirs[:remove_count]:
        shutil.rmtree(episode_dir, ignore_errors=True)


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


def download_s3_file(key: str, force: bool = False, cache_kind: str = "joints") -> Path | None:
    """从 S3 下载文件到本地缓存，返回本地路径。已缓存则跳过。"""
    cache_root = _get_cache_root(cache_kind)
    local_path = cache_root / key.replace("/", os.sep)
    if local_path.exists() and not force:
        _touch_episode_cache(cache_kind, key)
        return local_path

    client = _get_minio_client()
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        client.fget_object(S3_BUCKET, key, str(local_path))
        _touch_episode_cache(cache_kind, key)
        _prune_cache(cache_kind)
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
