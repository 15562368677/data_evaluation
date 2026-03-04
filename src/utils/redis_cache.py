from typing import Optional, Any
import json
import os
import redis
from dotenv import load_dotenv

load_dotenv()

redis_host = os.environ.get("REDIS_HOST", "localhost")
redis_port = int(os.environ.get("REDIS_PORT", 6379))
redis_db = int(os.environ.get("REDIS_DB", 1))
redis_password = os.environ.get("REDIS_PASSWORD", None)

_redis_client = None

def get_redis_client() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            decode_responses=True
        )
    return _redis_client

def get_cache(key: str) -> Optional[Any]:
    try:
        client = get_redis_client()
        val = client.get(key)
        if val:
            return json.loads(val)
    except Exception as e:
        print(f"Redis get cache exception: {e}")
    return None

def set_cache(key: str, value: Any, expire_seconds: int = 3600):
    try:
        client = get_redis_client()
        client.setex(key, expire_seconds, json.dumps(value))
    except Exception as e:
        print(f"Redis set cache exception: {e}")
