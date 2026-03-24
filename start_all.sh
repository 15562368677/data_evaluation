#!/bin/bash

echo "================================"
echo "Starting evaluation components"
echo "================================"

# 启动 Worker 进程 (后台)
echo "[1/2] Starting Background Worker Process..."
uv run python start_worker.py &
WORKER_PID=$!
echo "Worker started with PID: $WORKER_PID"

cleanup() {
    echo "Shutting down Worker..."
    if [ -n "$WORKER_PID" ] && kill -0 "$WORKER_PID" 2>/dev/null; then
        kill "$WORKER_PID"
        wait "$WORKER_PID" 2>/dev/null
    fi
}

trap cleanup INT TERM EXIT

echo "--------------------------------"

# 启动 Web 前端服务
echo "[2/2] Starting Web Application..."
uv run main.py
