#!/bin/bash

# 設置錯誤處理
set -e

# 設置環境變量
export PYTHONUNBUFFERED=1

# 設置日誌相關變量
LOG_DIR="/app/logs"
CURRENT_DATE=$(date +%Y%m%d)
LOG_FILE="$LOG_DIR/app_${CURRENT_DATE}.log"

# 確保日誌目錄存在
mkdir -p "$LOG_DIR"

# 清理超過7天的日誌
find "$LOG_DIR" -name "app_*.log" -type f -mtime +7 -exec rm {} \;

# 運行應用並將輸出導向到日誌文件
{
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ===== Starting applications ====="
    
    # 啟動主應用
    python app.py \
        --transport ${TRANSPORT:-rtcpush} \
        --model ${MODEL:-musetalk} \
        --avatar_id ${AVATAR_ID:-avator_1} \
        --tts ${TTS} \
        --TTS_SERVER ${TTS_SERVER:-http://52.69.235.212:9880} \
        --max_session ${MAX_SESSION:-5} \
        --avatar_list ${AVATAR_LIST:-"avator_10,avator_3"} 2>&1 & \
        # --customvideo_config ${CONFIG:-"data/custom_config.json"} \

    # 啟動 edgetts_api.py
    python edgetts_api.py 2>&1 &


    # 啟動 api_v3_live.py
    (cd /app/streaming_tts && python api_v3_live.py -p ${TTS_PORT:-9885} -a 0.0.0.0) 2>&1 &

    
    
    
    
    # 等待所有背景程序
    wait
} | tee -a "$LOG_FILE"


# 檢查退出狀態
if [ $? -ne 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Application crashed with exit code $?" | tee -a "$LOG_FILE"
    exit 1
fi
