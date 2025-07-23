# WebSocket 實時音頻傳輸優化方案

## 1. 問題描述

### 當前延遲情況
- 停止錄音到獲得結果的時間差異約3秒（2025-04-11T01:43:22.827Z 到 2025-04-11T01:43:25.972Z）
- 後端處理時間實際很快（記錄顯示處理迅速）
- 主要延遲來自前端的音頻傳輸過程

### 性能瓶頸分析
- 當前使用 HTTP POST 請求傳輸完整音頻文件
- 等待整個錄音完成才開始傳輸
- FormData 封包和傳輸效率較低

## 2. 優化方案

### WebSocket 實時傳輸
- 建立持久性 WebSocket 連接
- 使用小時間片段（200ms）即時傳輸音頻數據
- 邊錄製邊處理，無需等待完整錄音

### 代碼結構調整
- 前端代碼分為兩部分：
  - rtcpushapi_part1.html：基礎結構和樣式
  - rtcpushapi_part2.html：WebSocket和音頻處理邏輯

- 後端代碼分為兩部分：
  - api_v3_live_part1.py：主要服務和基礎功能
  - api_v3_live_part2.py：WebSocket處理和音頻流處理

## 3. 實現細節

### 前端修改要點
```javascript
// WebSocket 連接設置
const wsUrl = `wss://${window.location.hostname}:9880/ws/audio/${sessionId}`;
const ws = new WebSocket(wsUrl);

// 音頻錄製配置
mediaRecorder = new MediaRecorder(audioStream, {
    ...options,
    timeslice: 200  // 每200ms發送一次數據
});

// 實時數據處理
mediaRecorder.ondataavailable = (event) => {
    if (event.data.size > 0) {
        ws.send(event.data);
    }
};
```

### 後端修改要點
```python
# WebSocket 端點
@app.websocket("/ws/audio/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: int):
    await websocket.accept()
    
    async def process_audio():
        while True:
            audio_chunk = await websocket.receive_bytes()
            result = await process_audio_chunk(audio_chunk, session_id)
            if result:
                await websocket.send_json(result)
```

## 4. 部署說明

### 文件結構
```
web/
├── rtcpushapi_part1.html      # 基礎結構和樣式
└── rtcpushapi_part2.html      # WebSocket和音頻處理邏輯

streaming_tts/
├── api_v3_live_part1.py       # 主要服務和基礎功能
└── api_v3_live_part2.py       # WebSocket處理和音頻流處理
```

### 配置更新
1. 更新 docker-compose.yaml：
```yaml
services:
  app:
    ports:
      - "9880:9880"
      - "9881:9881/tcp"  # WebSocket端口
```

2. 環境變量配置：
```env
WEBSOCKET_PORT=9881
```

### 合併步驟
1. 前端文件合併：
```bash
cat web/rtcpushapi_part1.html web/rtcpushapi_part2.html > web/rtcpushapi.html
```

2. 後端文件合併：
```bash
cat streaming_tts/api_v3_live_part1.py streaming_tts/api_v3_live_part2.py > streaming_tts/api_v3_live.py
```

## 5. 預期效果

### 性能提升
- 音頻傳輸延遲從3秒降至500ms以內
- 實時反饋更加流暢
- 系統資源使用更優化

### 注意事項
1. WebSocket連接管理
   - 確保連接斷開時正確清理資源
   - 實現自動重連機制
   - 處理錯誤情況

2. 音頻數據處理
   - 確保音頻片段正確串接
   - 處理可能的數據丟失情況
   - 監控音頻質量

3. 系統監控
   - 添加性能監控指標
   - 記錄關鍵時間點
   - 異常情況告警

## 6. 後續優化建議
1. 引入音頻數據壓縮
2. 實現斷點續傳
3. 添加數據緩存機制
4. 優化錯誤處理流程
