"""简易内存缓存工具。"""

from __future__ import annotations

import time
from threading import Lock
from typing import Any

_CACHE: dict[str, dict[str, Any]] = {}
_CACHE_LOCK = Lock()


def set_cache(key: str, value: Any, ttl_seconds: int | None = None) -> Any:
    """写入缓存，支持可选 TTL（秒）。"""
    expires_at = None
    if ttl_seconds is not None and ttl_seconds > 0:
        expires_at = time.time() + ttl_seconds

    with _CACHE_LOCK:
        _CACHE[key] = {
            "value": value,
            "created_at": time.time(),
            "expires_at": expires_at,
        }
    return value


def get_cache(key: str) -> Any | None:
    """读取缓存；若已过期会自动清理并返回 None。"""
    with _CACHE_LOCK:
        item = _CACHE.get(key)
        if item is None:
            return None

        expires_at = item.get("expires_at")
        if expires_at is not None and expires_at <= time.time():
            _CACHE.pop(key, None)
            return None

        return item.get("value")


def clear_cache(key: str | None = None) -> int:
    """清理缓存。传 key 仅清一个；不传则清空全部。返回清理条数。"""
    with _CACHE_LOCK:
        if key is None:
            count = len(_CACHE)
            _CACHE.clear()
            return count

        existed = 1 if key in _CACHE else 0
        _CACHE.pop(key, None)
        return existed
