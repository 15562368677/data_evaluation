#!/bin/bash

echo "================================"
echo "Starting evaluation components"
echo "================================"

# 启动 Worker 进程 (后台)
echo "[1/2] Starting Background Worker Process..."
uv run python start_worker.py &
WORKER_PID=$!
echo "Worker started with PID: $WORKER_PID"

echo "--------------------------------"

# 启动 Web 前端服务
echo "[2/2] Starting Web Application..."
uv run main.py

# 如果主进程退出（比如按了 Ctrl+C），确保一并结束 Worker 进程
trap "echo 'Shutting down Worker...'; kill $WORKER_PID; exit 0" INT TERM EXIT
