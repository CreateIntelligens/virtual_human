# WebRTC 實現優化記錄 (2025/04/29-30)

## 1. 問題分析
- 發現 Session ID 管理和實例準備狀態的問題
- 出現 "Nerfreal instance not ready for session 0" 警告
- WebRTC 連接和 sessionId 分配的時序問題

## 2. rtcpush 到 webrtc 的轉換
### 2.1 主要架構改變
1. 連接方式改變：
   - rtcpush: 使用 SRS 服務器推流
   - webrtc: 直接使用瀏覽器的 WebRTC API

2. SessionID 處理：
   - rtcpush: 預設使用固定的 sessionId
   - webrtc: 動態分配 sessionId，由後端生成並返回

3. 媒體流處理：
   - rtcpush: 
     ```javascript
     // 舊版推流方式
     await post(push_url, pc.localDescription.sdp)
     ```
   - webrtc:
     ```javascript
     // 新版直接使用 WebRTC
     pc.createOffer()
       .then(offer => pc.setLocalDescription(offer))
       .then(() => {
         // 與後端交換 SDP
         return fetch('/offer', {...})
       })
     ```

### 2.2 媒體處理改變
1. 音視頻控制：
   ```javascript
   // 新版支持更靈活的媒體控制
   pc.addTransceiver('video', { direction: 'recvonly' });
   pc.addTransceiver('audio', { direction: 'recvonly' });
   ```

2. 連接管理：
   ```javascript
   // 新增連接狀態管理
   @pc.on("connectionstatechange")
   async def on_connectionstatechange():
       if pc.connectionState == "failed":
           await pc.close()
   ```

### 2.3 檔案結構改變
1. 原始 rtcpush 實現：
   - 主要檔案：web/rtcpushapi.html
   - 相關檔案：
     - rtcpushapi_google.html (Google STT 版本)
     - rtcpushapi_demo.html (示範版本)

2. 新版 webrtc 實現：
   - 主要檔案：web/webrtcapi_no_condition.html
   - 相關檔案：
     - webrtcapi_demo.html (新版示範)
     - webrtcapi.html (基礎版本)

3. 關鍵差異：
   - rtcpush 版本需要 SRS 服務器支援
   - webrtc 版本直接使用瀏覽器 API
   - webrtc 版本支援更靈活的 session 管理
   - webrtc 版本有更完整的錯誤處理機制

## 3. 已完成的修改

### 3.1 前端修改 (webrtcapi_no_condition.html)
1. 優化 sessionId 處理機制：
```javascript
function negotiate() {
    // ... 
    .then((answer) => {
        document.getElementById('sessionid').value = answer.sessionid;
        document.getElementById('sessionInfo').textContent = `Session ID: ${answer.sessionid}`;
        console.log(`${getTimestamp()} Session ID updated to: ${answer.sessionid}`);
        return new Promise(resolve => {
            const sid = getSessionId();
            if (sid && sid !== 0) {
                resolve(pc.setRemoteDescription(answer));
            }
        });
    })
}
```

2. 初始化順序優化：
```javascript
$(document).ready(async function() {
    videoElement.muted = true;
    try {
        await startPlay();  // 先執行連接
        console.log(`${getTimestamp()} WebRTC connection established with sessionId: ${getSessionId()}`);
        
        if (getSessionId() === 0) {
            throw new Error('Failed to get valid sessionId');
        }
        
        await initializeUIWithConfig();  // 再初始化 UI
        startCheckingSpeakingStatus();   // 最後開始檢查狀態
        console.log(`${getTimestamp()} Application initialized`);
    } catch (error) {
        console.error(`${getTimestamp()} Initialization failed:`, error);
        showError('初始化失敗，請重新載入頁面');
    }
});
```

3. 新增時間戳記錄：
```javascript
function getTimestamp() {
    return `${new Date().toISOString()}.${(performance.now() % 1000).toFixed(3).padStart(3, '0')}`;
}
```

## 4. 注意事項
1. 保持 rtcpush 和 webrtc 兩種模式的兼容性
2. 避免破壞現有的功能
3. 確保錯誤處理和日誌記錄的完整性
4. 維持程式碼的可維護性
5. 確保 session 管理的一致性

## 5. 實現細節
### 5.1 Session 管理
- 使用隱藏的 input 欄位儲存 sessionId
- 所有 API 調用統一使用 getSessionId() 函數
- sessionId 在 WebRTC 連接建立時由後端分配

### 5.2 日誌記錄
- 新增時間戳記錄機制
- 統一的日誌格式
- 包含毫秒級精確度

### 5.3 錯誤處理
- 連接失敗自動重試
- UI 錯誤提示
- 資源清理機制
