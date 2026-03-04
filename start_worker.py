import os
from dotenv import load_dotenv
from redis import Redis
from rq import Worker, Queue

if __name__ == '__main__':
    # 从 .env 文件中加载环境变量
    load_dotenv()
    
    redis_host = os.environ.get("REDIS_HOST", "localhost")
    redis_port = int(os.environ.get("REDIS_PORT", 6379))
    redis_db = int(os.environ.get("REDIS_DB", 1))
    redis_password = os.environ.get("REDIS_PASSWORD", None)

    redis_conn = Redis(
        host=redis_host, 
        port=redis_port, 
        db=redis_db, 
        password=redis_password
    )

    print(f"Starting PNP Worker on Redis {redis_host}:{redis_port}/{redis_db}")
    
    worker = Worker(['pnp_tasks'], connection=redis_conn)
    worker.work()
