###############################################################################
# EdgeTTS API Service - 提供 HTTP API 給前端使用
###############################################################################
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import edge_tts
import asyncio
from io import BytesIO
from datetime import datetime
import uvicorn
import os
import json
import re

# 載入所有替換規則
REPLACE_RULES = []
try:
    replacements_path = os.path.join(os.path.dirname(__file__), 'replacements.json')
    with open(replacements_path, 'r', encoding='utf-8') as f:
        REPLACE_RULES = json.load(f)
    print(f'[{datetime.now()}] Loaded {len(REPLACE_RULES)} replacement rules from replacements.json')
except FileNotFoundError:
    print(f'[{datetime.now()}] Warning: replacements.json not found, text replacement disabled')
except Exception as e:
    print(f'[{datetime.now()}] Error loading replacements.json: {str(e)}')

app = FastAPI(title="EdgeTTS API Service")

# 允許跨域請求(CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生產環境請改為具體的前端網址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TTSRequest(BaseModel):
    text: str
    voice: str = "zh-TW-HsiaoChenNeural"
    rate: str = "+0%"  # 語速調整 -50% 到 +100%
    volume: str = "+0%"  # 音量調整 -50% 到 +100%
    pitch: str = "+0Hz"  # 音調調整
    format: str = "mp3"  # 音檔格式: mp3 或 wav

class VoiceInfo(BaseModel):
    name: str
    short_name: str
    gender: str
    locale: str

def apply_text_replacements(text: str) -> str:
    """
    根據 replacements.json 來做一系列替換
    """
    if not REPLACE_RULES:
        return text
    
    original_text = text
    for rule in REPLACE_RULES:
        flag_val = 0
        for f in rule.get('flags', []):
            flag_val |= getattr(re, f, 0)
        text = re.sub(rule['pattern'], rule['replacement'], text, flags=flag_val)
    
    if text != original_text:
        print(f'[{datetime.now()}] Text after replacements: {text}')
    
    return text

@app.get("/")
async def root():
    return {
        "service": "EdgeTTS API",
        "version": "1.0",
        "endpoints": {
            "/tts": "POST - 文字轉語音(返回完整音檔)",
            "/tts/stream": "POST - 文字轉語音(串流模式)",
            "/voices": "GET - 取得可用語音列表",
            "/voices/{locale}": "GET - 取得特定語言的語音列表"
        }
    }

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """
    將文字轉換為語音並返回完整的音檔
    
    參數:
    - text: 要轉換的文字
    - voice: 語音名稱(預設: zh-TW-HsiaoChenNeural)
    - rate: 語速調整(預設: +0%)
    - volume: 音量調整(預設: +0%)
    - pitch: 音調調整(預設: +0Hz)
    - format: 音檔格式(預設: mp3, 可選: wav)
    
    返回: MP3 或 WAV 音檔
    """
    try:
        print(f'[{datetime.now()}] TTS Request - Text: {request.text[:50]}..., Voice: {request.voice}, Format: {request.format}')
        
        # 應用文字替換規則
        processed_text = apply_text_replacements(request.text)
        
        # 根據格式選擇不同的輸出格式
        # EdgeTTS 支援的格式: audio-24khz-48kbitrate-mono-mp3, audio-24khz-48khz-mono-pcm (WAV)
        if request.format.lower() == "wav":
            output_format = "audio-24khz-48khz-mono-pcm"
            media_type = "audio/wav"
            filename = "speech.wav"
        else:
            output_format = "audio-24khz-48kbitrate-mono-mp3"
            media_type = "audio/mpeg"
            filename = "speech.mp3"
        
        # 建立 communicate 物件
        communicate = edge_tts.Communicate(
            text=processed_text,
            voice=request.voice,
            rate=request.rate,
            volume=request.volume,
            pitch=request.pitch
        )
        
        # 收集所有音訊資料
        audio_data = BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.write(chunk["data"])
        
        audio_data.seek(0)
        
        print(f'[{datetime.now()}] TTS Success - Generated {audio_data.getbuffer().nbytes} bytes ({request.format})')
        
        return Response(
            content=audio_data.read(),
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )
    
    except Exception as e:
        print(f'[{datetime.now()}] TTS Error: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/tts/stream")
async def text_to_speech_stream(request: TTSRequest):
    """
    將文字轉換為語音並以串流方式返回
    
    適合長文本或需要即時播放的場景
    """
    try:
        print(f'[{datetime.now()}] TTS Stream Request - Text: {request.text[:50]}..., Voice: {request.voice}, Format: {request.format}')
        
        # 應用文字替換規則
        processed_text = apply_text_replacements(request.text)
        
        # 根據格式選擇 media type
        if request.format.lower() == "wav":
            media_type = "audio/wav"
        else:
            media_type = "audio/mpeg"
        
        async def generate():
            communicate = edge_tts.Communicate(
                text=processed_text,
                voice=request.voice,
                rate=request.rate,
                volume=request.volume,
                pitch=request.pitch
            )
            
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]
        
        return StreamingResponse(
            generate(),
            media_type=media_type,
            headers={
                "Cache-Control": "no-cache",
                "Transfer-Encoding": "chunked"
            }
        )
    
    except Exception as e:
        print(f'[{datetime.now()}] TTS Stream Error: {str(e)}')
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/voices")
async def get_all_voices():
    """
    取得所有可用的語音列表
    """
    try:
        voices = await edge_tts.list_voices()
        return {
            "total": len(voices),
            "voices": [
                {
                    "name": v["Name"],
                    "short_name": v["ShortName"],
                    "gender": v["Gender"],
                    "locale": v["Locale"],
                    "suggested_codec": v.get("SuggestedCodec", "audio-24khz-48kbitrate-mono-mp3"),
                    "friendly_name": v.get("FriendlyName", ""),
                    "status": v.get("Status", "")
                }
                for v in voices
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/voices/{locale}")
async def get_voices_by_locale(locale: str):
    """
    取得特定語言的語音列表
    
    常用語言代碼:
    - zh-TW: 繁體中文(台灣)
    - zh-CN: 簡體中文(中國)
    - en-US: 英語(美國)
    - ja-JP: 日語
    - ko-KR: 韓語
    """
    try:
        voices = await edge_tts.list_voices()
        filtered = [
            {
                "name": v["Name"],
                "short_name": v["ShortName"],
                "gender": v["Gender"],
                "locale": v["Locale"],
                "friendly_name": v.get("FriendlyName", "")
            }
            for v in voices
            if v["Locale"].lower().startswith(locale.lower())
        ]
        
        return {
            "locale": locale,
            "total": len(filtered),
            "voices": filtered
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # 啟動服務於 8765 port
    print(f'[{datetime.now()}] Starting EdgeTTS API Service on http://0.0.0.0:8765')
    uvicorn.run(app, host="0.0.0.0", port=8765)
