#!/bin/bash
# 每日 Docker 服務重啟腳本
# 執行時間：每天早上 8:40

LOG_FILE="/srv/live_test/logs/docker_restart.log"
DATE=$(date '+%Y-%m-%d %H:%M:%S')

# 確保 logs 目錄存在
mkdir -p /srv/live_test/logs

echo "[$DATE] 開始執行 Docker 重啟..." >> $LOG_FILE

# 切換到正確的目錄
cd /srv/live_test

# 檢查是否在正確的目錄
if [ ! -f "docker-compose.yaml" ]; then
    echo "[$DATE] 錯誤：找不到 docker-compose.yaml 文件" >> $LOG_FILE
    exit 1
fi

# 清理未使用的 volumes
echo "[$DATE] 清理未使用的 Docker volumes..." >> $LOG_FILE
docker volume prune -f >> $LOG_FILE 2>&1

# 停止服務
echo "[$DATE] 停止 Docker 服務..." >> $LOG_FILE
docker compose down >> $LOG_FILE 2>&1
DOWN_STATUS=$?

if [ $DOWN_STATUS -ne 0 ]; then
    echo "[$DATE] 警告：Docker 停止過程中出現問題，狀態碼：$DOWN_STATUS" >> $LOG_FILE
fi

# 等待幾秒確保完全停止
echo "[$DATE] 等待服務完全停止..." >> $LOG_FILE
sleep 5

# 重新啟動服務
echo "[$DATE] 重新啟動 Docker 服務..." >> $LOG_FILE
docker compose up -d >> $LOG_FILE 2>&1
UP_STATUS=$?

if [ $UP_STATUS -eq 0 ]; then
    echo "[$DATE] Docker 重啟成功完成" >> $LOG_FILE
    
    # 檢查服務狀態
    echo "[$DATE] 檢查服務狀態..." >> $LOG_FILE
    docker compose ps >> $LOG_FILE 2>&1
else
    echo "[$DATE] Docker 重啟失敗，狀態碼：$UP_STATUS" >> $LOG_FILE
fi

echo "[$DATE] ===========================================" >> $LOG_FILE
