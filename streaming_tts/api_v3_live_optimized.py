import signal
import os
import sys
import traceback
from typing import Generator, Dict, List, Union, Optional
import logging
import time
import json
import threading
import yaml
import torch
import asyncio
import io
import aiohttp
from concurrent.futures import ThreadPoolExecutor
now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))
print(sys.path)

from starlette.middleware.cors import CORSMiddleware 
from fastapi import FastAPI, Request, Response, Depends, UploadFile, File,HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, JSONResponse, Response
from io import BytesIO
from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.TTS_infer_pack.TTS_new import TTS, TTS_Config
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import get_method_names as get_cut_method_names
from pydantic import BaseModel
import numpy as np
import soundfile as sf
import nltk
import opencc
import argparse
import wave
import subprocess
import requests
from faster_whisper import WhisperModel
import pyaudio
from datetime import datetime
from google.cloud import speech
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(now_dir, "wonderland-nft-5072ee803fcd.json")

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
nltk.download('averaged_perceptron_tagger_eng')

# 基礎設置
origins = ["*"]
i18n = I18nAuto()
cut_method_names = get_cut_method_names()
converter = opencc.OpenCC('s2t')

# 請求模型
class TTS_Request(BaseModel):
    model_id: str = None  # 改為可選字段
    text: str = None
    text_lang: str = None
    ref_audio_path: str = None
    prompt_lang: str = None
    prompt_text: str = ""
    top_k: int = 5
    top_p: float = 1
    temperature: float = 1
    text_split_method: str = "cut5"
    batch_size: int = 1
    batch_threshold: float = 0.75
    split_bucket: bool = True
    speed_factor: float = 1.0
    fragment_interval: float = 0.3
    seed: int = -1
    media_type: str = "wav"
    streaming_mode: bool = False
    parallel_infer: bool = True
    repetition_penalty: float = 1.35



# 初始化參數解析
parser = argparse.ArgumentParser(description="GPT-SoVITS api")
parser.add_argument("-c", "--tts_config", type=str, default="GPT_SoVITS/configs/tts_infer.yaml", help="tts_infer路径")
parser.add_argument("-a", "--bind_addr", type=str, default="0.0.0.0", help="default: 127.0.0.1")
parser.add_argument("-p", "--port", type=int, default="9880", help="default: 9880")
args = parser.parse_args()

config_path = args.tts_config if args.tts_config not in [None, ""] else "GPT_SoVITS/configs/tts_infer.yaml"
port = args.port
host = args.bind_addr
argv = sys.argv

# FastAPI 應用
APP = FastAPI()
APP.mount("/srt", StaticFiles(directory="./srt"), name="srt")
APP.mount("/audio", StaticFiles(directory="./audio"), name="audio")
APP.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# 初始化配置
tts_config = TTS_Config(config_path)
tts_pipeline = TTS(tts_config)
tts_pipeline1 = TTS(TTS_Config("GPT_SoVITS/configs/tts_infer1.yaml"))


# 工具函數
def pack_ogg(io_buffer:BytesIO, data:np.ndarray, rate:int):
    with sf.SoundFile(io_buffer, mode='w', samplerate=rate, channels=1, format='ogg') as audio_file:
        audio_file.write(data)
    return io_buffer

def pack_raw(io_buffer:BytesIO, data:np.ndarray, rate:int):
    io_buffer.write(data.tobytes())
    return io_buffer

def pack_wav(io_buffer:BytesIO, data:np.ndarray, rate:int):
    io_buffer = BytesIO()
    sf.write(io_buffer, data, rate, format='wav')
    return io_buffer

def pack_aac(io_buffer:BytesIO, data:np.ndarray, rate:int):
    process = subprocess.Popen([
        'ffmpeg',
        '-f', 's16le',
        '-ar', str(rate),
        '-ac', '1',
        '-i', 'pipe:0',
        '-c:a', 'aac',
        '-b:a', '192k',
        '-vn',
        '-f', 'adts',
        'pipe:1'
    ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, _ = process.communicate(input=data.tobytes())
    io_buffer.write(out)
    return io_buffer

def pack_audio(io_buffer:BytesIO, data:np.ndarray, rate:int, media_type:str):
    if media_type == "ogg":
        io_buffer = pack_ogg(io_buffer, data, rate)
    elif media_type == "aac":
        io_buffer = pack_aac(io_buffer, data, rate)
    elif media_type == "wav":
        io_buffer = pack_wav(io_buffer, data, rate)
    else:
        io_buffer = pack_raw(io_buffer, data, rate)
    io_buffer.seek(0)
    return io_buffer

def wave_header_chunk(frame_input=b"", channels=1, sample_width=2, sample_rate=32000):
    wav_buf = BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)
    wav_buf.seek(0)
    return wav_buf.read()




# 參數檢查
def check_params(req:dict):
    text:str = req.get("text", "")
    text_lang:str = req.get("text_lang", "")
    ref_audio_path:str = req.get("ref_audio_path", "")
    streaming_mode:bool = req.get("streaming_mode", False)
    media_type:str = req.get("media_type", "wav")
    prompt_lang:str = req.get("prompt_lang", "")
    text_split_method:str = req.get("text_split_method", "cut5")

    if ref_audio_path in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "ref_audio_path is required"})
    if text in [None, ""]:
        return JSONResponse(status_code=400, content={"message": "text is required"})
    if (text_lang in [None, ""]):
        return JSONResponse(status_code=400, content={"message": "text_lang is required"})
    elif text_lang.lower() not in tts_config.languages:
        return JSONResponse(status_code=400, content={"message": "text_lang is not supported"})
    if (prompt_lang in [None, ""]):
        return JSONResponse(status_code=400, content={"message": "prompt_lang is required"})
    elif prompt_lang.lower() not in tts_config.languages:
        return JSONResponse(status_code=400, content={"message": "prompt_lang is not supported"})
    if media_type not in ["wav", "raw", "ogg", "aac"]:
        return JSONResponse(status_code=400, content={"message": "media_type is not supported"})
    elif media_type == "ogg" and not streaming_mode:
        return JSONResponse(status_code=400, content={"message": "ogg format is not supported in non-streaming mode"})
    
    if text_split_method not in cut_method_names:
        return JSONResponse(status_code=400, content={"message": f"text_split_method:{text_split_method} is not supported"})
    
    return None


# TTS 處理函數
async def tts_handle(req:dict):
    streaming_mode = req.get("streaming_mode", False)
    media_type = req.get("media_type", "wav")

    check_res = check_params(req)
    if check_res is not None:
        return check_res

    if streaming_mode:
        req["return_fragment"] = True
    
    try:
        tts_generator = tts_pipeline.run(req)
        
        if streaming_mode:
            def streaming_generator(tts_generator: Generator, media_type: str):
                start_time = time.time()
                total_chunks = 0
                text = req.get("text", "")
                text_length = len(text)
                total_audio_samples = 0
                
                if media_type == "wav":
                    # 收集所有音頻塊
                    audio_chunks = []
                    first_sr = None
                    
                    # 收集並顯示進度
                    for sr, chunk in tts_generator:
                        if first_sr is None:
                            first_sr = sr
                        audio_chunks.append(chunk)
                        total_chunks += 1
                        total_audio_samples += len(chunk)
                        audio_duration = total_audio_samples / sr
                        
                        current_time = time.time()
                        elapsed_time = current_time - start_time
                        chars_per_audio_second = text_length / audio_duration if audio_duration > 0 else 0
                        chars_per_process_second = text_length / elapsed_time if elapsed_time > 0 else 0
                        
                        print(f"\r處理進度：已處理{total_chunks}個片段，總字數={text_length}字，"
                            f"已用時間={elapsed_time:.2f}秒，音頻長度={audio_duration:.2f}秒，"
                            f"當前生成速度={chars_per_process_second:.2f}字/秒，"
                            f"音頻每秒字數={chars_per_audio_second:.2f}字/秒", end="", flush=True)
                        
                    # 合併所有塊並生成完整WAV
                    full_audio = np.concatenate(audio_chunks)
                    bio = BytesIO()
                    with wave.open(bio, 'wb') as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(first_sr)
                        wav_file.writeframes(full_audio.tobytes())
                        
                    bio.seek(0)
                    print(f"\n處理完成！")
                    print(f"最終統計：")
                    print(f"- 總字數：{text_length}字")
                    print(f"- 音頻長度：{audio_duration:.2f}秒")
                    print(f"- 處理耗時：{elapsed_time:.2f}秒")
                    print(f"- 當前生成速度={chars_per_process_second:.2f}字/秒")
                    print(f"- 音頻每秒字數={chars_per_audio_second:.2f}字/秒")
                    
                    yield bio.read()
            return StreamingResponse(
                streaming_generator(tts_generator, media_type),
                media_type=f"audio/{media_type}"
            )
    
        else:
            sr, audio_data = next(tts_generator)
            audio_data = pack_audio(BytesIO(), audio_data, sr, media_type).getvalue()
            return Response(audio_data, media_type=f"audio/{media_type}")
            
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"message": f"tts failed", "Exception": str(e)}
        )

async def tts_handle1(req:dict):
    streaming_mode = req.get("streaming_mode", False)
    media_type = req.get("media_type", "wav")

    check_res = check_params(req)
    if check_res is not None:
        return check_res

    if streaming_mode:
        req["return_fragment"] = True
    
    try:
        tts_generator = tts_pipeline1.run(req)
        
        if streaming_mode:
            def streaming_generator(tts_generator: Generator, media_type: str):
                start_time = time.time()
                total_chunks = 0
                text = req.get("text", "")
                text_length = len(text)
                total_audio_samples = 0
                
                if media_type == "wav":
                    # 收集所有音頻塊
                    audio_chunks = []
                    first_sr = None
                    
                    # 收集並顯示進度
                    for sr, chunk in tts_generator:
                        if first_sr is None:
                            first_sr = sr
                        audio_chunks.append(chunk)
                        total_chunks += 1
                        total_audio_samples += len(chunk)
                        audio_duration = total_audio_samples / sr
                        
                        current_time = time.time()
                        elapsed_time = current_time - start_time
                        chars_per_audio_second = text_length / audio_duration if audio_duration > 0 else 0
                        chars_per_process_second = text_length / elapsed_time if elapsed_time > 0 else 0
                        
                        print(f"\r處理進度：已處理{total_chunks}個片段，總字數={text_length}字，"
                            f"已用時間={elapsed_time:.2f}秒，音頻長度={audio_duration:.2f}秒，"
                            f"當前生成速度={chars_per_process_second:.2f}字/秒，"
                            f"音頻每秒字數={chars_per_audio_second:.2f}字/秒", end="", flush=True)
                        
                    # 合併所有塊並生成完整WAV
                    full_audio = np.concatenate(audio_chunks)
                    bio = BytesIO()
                    with wave.open(bio, 'wb') as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(first_sr)
                        wav_file.writeframes(full_audio.tobytes())
                        
                    bio.seek(0)
                    print(f"\n處理完成！")
                    print(f"最終統計：")
                    print(f"- 總字數：{text_length}字")
                    print(f"- 音頻長度：{audio_duration:.2f}秒")
                    print(f"- 處理耗時：{elapsed_time:.2f}秒")
                    print(f"- 當前生成速度={chars_per_process_second:.2f}字/秒")
                    print(f"- 音頻每秒字數={chars_per_audio_second:.2f}字/秒")
                    
                    yield bio.read()
            return StreamingResponse(
                streaming_generator(tts_generator, media_type),
                media_type=f"audio/{media_type}"
            )
    
        else:
            sr, audio_data = next(tts_generator)
            audio_data = pack_audio(BytesIO(), audio_data, sr, media_type).getvalue()
            return Response(audio_data, media_type=f"audio/{media_type}")
            
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"message": f"tts failed", "Exception": str(e)}
        )

async def tts_handle_srt(req:dict, request:Request):
    try:
        tts_generator = tts_pipeline.run(req)
        sr, audio_data = next(tts_generator)
        return JSONResponse({
            "code":"200", 
            "srt":f"http://{request.url.hostname}:{request.url.port}/srt/tts-out.srt",
            "audio":f"http://{request.url.hostname}:{request.url.port}/audio/audio.wav"
        })
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"tts failed", "Exception": str(e)})

@APP.get("/srt")
async def tts_get_endpoint_srt(
    request: Request,
    text: str = None,
    text_lang: str = None,
    ref_audio_path: str = None,
    prompt_lang: str = None,
    prompt_text: str = "",
    top_k: int = 5,
    top_p: float = 1,
    temperature: float = 1,
    text_split_method: str = "cut5",
    batch_size: int = 10,
    batch_threshold: float = 0.75,
    split_bucket: bool = True,
    speed_factor: float = 1.0,
    fragment_interval: float = 0.3,
    seed: int = -1,
    media_type: str = "wav",
    streaming_mode: bool = False,
    parallel_infer: bool = True,
    repetition_penalty: float = 1.35
):
    req = {
        "text": text,
        "text_lang": text_lang.lower(),
        "ref_audio_path": ref_audio_path,
        "prompt_text": prompt_text,
        "prompt_lang": prompt_lang.lower(),
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": text_split_method,
        "batch_size": int(batch_size),
        "batch_threshold": float(batch_threshold),
        "speed_factor": float(speed_factor),
        "split_bucket": split_bucket,
        "fragment_interval": fragment_interval,
        "seed": seed,
        "media_type": media_type,
        "streaming_mode": streaming_mode,
        "parallel_infer": parallel_infer,
        "repetition_penalty": float(repetition_penalty)
    }
    return await tts_handle_srt(req, request)

@APP.post("/srt")
async def tts_post_endpoint_srt(request: TTS_Request, req1: Request):
    req = request.dict()
    return await tts_handle_srt(req, req1)

@APP.get("/")
async def tts_get_endpoint_default(
    text: str = None,
    text_lang: str = "auto",
    ref_audio_path: str = "./output/2025/02/19/c819eb70a3aa0d8b79bcb11beb3752bb/ref_audios/Lula_voice_45_7794560_7971840.wav",
    prompt_lang: str = "zh",
    prompt_text: str = "到底在哪裡?帥哥在哪裡?我沒看到這個是真實的這是真實",
    top_k: int = 5,
    top_p: float = 1,
    temperature: float = 1,
    text_split_method: str = "cut1",
    batch_size: int = 20,
    batch_threshold: float = 0.75,
    split_bucket: bool = True,
    speed_factor: float = 0.9,
    fragment_interval: float = 0.3,
    seed: int = 1413775942,
    media_type: str = "wav",
    streaming_mode: bool = True,
    parallel_infer: bool = True,
    repetition_penalty: float = 1.8
):
    req = {
        "text": text,
        "text_lang": text_lang.lower(),
        "ref_audio_path": ref_audio_path,
        "prompt_text": prompt_text,
        "prompt_lang": prompt_lang.lower(),
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": text_split_method,
        "batch_size": int(batch_size),
        "batch_threshold": float(batch_threshold),
        "speed_factor": float(speed_factor),
        "split_bucket": split_bucket,
        "fragment_interval": fragment_interval,
        "seed": seed,
        "media_type": media_type,
        "streaming_mode": streaming_mode,
        "parallel_infer": parallel_infer,
        "repetition_penalty": float(repetition_penalty)
    }
    return await tts_handle(req)

@APP.get("/test")
async def tts_get_endpoint_test(
    text: str = None,
    text_lang: str = "auto",
    ref_audio_path: str = "./output/2024/09/18/4c22dbd4462eb8623bce49674dc0b684/ref_audios/sweet_11_2300160_2405120.wav",
    prompt_lang: str = "zh",
    prompt_text: str = "不妨深入體會在地的文化熱情",
    top_k: int = 5,
    top_p: float = 1,
    temperature: float = 1,
    text_split_method: str = "cut1",
    batch_size: int = 20,
    batch_threshold: float = 0.75,
    split_bucket: bool = True,
    speed_factor: float = 1.0,
    fragment_interval: float = 0.3,
    seed: int = 717357706,
    media_type: str = "wav",
    streaming_mode: bool = True,
    parallel_infer: bool = True,
    repetition_penalty: float = 1.8
):
    req = {
        "text": text,
        "text_lang": text_lang.lower(),
        "ref_audio_path": ref_audio_path,
        "prompt_text": prompt_text,
        "prompt_lang": prompt_lang.lower(),
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": text_split_method,
        "batch_size": int(batch_size),
        "batch_threshold": float(batch_threshold),
        "speed_factor": float(speed_factor),
        "split_bucket": split_bucket,
        "fragment_interval": fragment_interval,
        "seed": seed,
        "media_type": media_type,
        "streaming_mode": streaming_mode,
        "parallel_infer": parallel_infer,
        "repetition_penalty": float(repetition_penalty)
    }
    return await tts_handle1(req)

@APP.post("/")
async def tts_post_endpoint_default(request: TTS_Request):
    req = request.dict()
    return await tts_handle(req)

@APP.post("/test")
async def tts_post_endpoint_test(request: TTS_Request):
    req = request.dict()
    return await tts_handle1(req)

model_path = "tools/asr/models/faster-whisper-large-v3-turbo"
device = "cuda"  # 或 "cuda" 如果你有 GPU
model = WhisperModel(model_path, device=device, compute_type="float32")

# 優化的音頻處理函數
async def convert_audio(audio_data: bytes) -> bytes:
    """使用FFmpeg進行音頻轉換的優化函數"""
    process = await asyncio.create_subprocess_exec(
        'ffmpeg',
        '-i', 'pipe:0',
        '-ar', '16000',
        '-ac', '1',
        '-c:a', 'pcm_s16le',
        '-f', 'wav',
        'pipe:1',
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout, stderr = await process.communicate(input=audio_data)
    
    if process.returncode != 0:
        print(f"FFmpeg錯誤輸出: {stderr.decode()}")
        raise RuntimeError("FFmpeg轉換失敗")
        
    return stdout

async def transcribe_audio(audio_data: bytes) -> str:
    """執行音頻轉錄的優化函數"""
    audio_stream = io.BytesIO(audio_data)
    segments, _ = model.transcribe(
        audio=audio_stream,
        beam_size=3,
        temperature=0.0,
        compression_ratio_threshold=2.4,
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500,
            threshold=0.3
        ),
        language="zh"
    )
    
    return " ".join(segment.text for segment in segments)

@APP.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    try:
        print("開始辨識")
        start_time = time.time()
        
        # 讀取音頻數據
        audio_data = await audio.read()
        print(f"接收到的音頻檔案大小: {len(audio_data)} 字節")
        
        # 並行執行音頻轉換和轉錄
        converted_audio = await convert_audio(audio_data)
        print(f"音頻轉換完成，耗時: {time.time() - start_time:.2f}秒")
        
        # 執行轉錄
        input_text = await transcribe_audio(converted_audio)
        print(f"轉錄完成，文本長度: {len(input_text)}，耗時: {time.time() - start_time:.2f}秒")
        
        # 調用聊天機器人
        url = "https://ddgsrvchat.aicreate360.com/custom_service"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoxLCJlbWFpbCI6ImFkbWluQGFpY3JlYXRlMzYwLmNvbSIsInJvbGUiOiJhZG1pbiIsImlhdCI6MTYwNjQ4NjU5Mn0.0j5z1E1l1z8Vr6ZtX9Dz4J6G1m6wD3jV5b8W7Xw5Y2c"
        }
        
        payload = {
            "type": "message",
            "message": {
                "type": "text",
                "id": "516483564309840636",
                "quoteToken": "y-3002SPcbDa-XFl-rsAqsG3bJpRryEIkcqI3mVWv-6eyDLpQsZGKYdsk8A66RSXu-FHoPzdADkeThELij-f_duq6OOr5lqPcoTXeF6_ilfoU4WydySk3GvizPX9Q01xTOtmYOaIRpVxV4FwfhbhSw",
                "text": converter.convert(input_text)
            },
            "webhookEventId": "01J2G7V0RXWR040AK1Q8PQD2R0",
            "deliveryContext": {
                "isRedelivery": False
            },
            "timestamp": 1720679498017,
            "source": {
                "type": "user",
                "userId": "U03cd17c02ef4a297c2c2a910e5b6219f"
            },
            "replyToken": "d6e582b0f794446c980b50653439d9f3",
            "mode": "active"
        }
        
        print("呼叫聊天機器人")
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                response_json = await response.json()
                message = response_json['message']
                return_json = {
                    "input_text": converter.convert(input_text),
                    "text": converter.convert(message)
                }
        
        print(f"總處理完成，總耗時: {time.time() - start_time:.2f}秒")
        return JSONResponse(return_json)
    
    except Exception as e:
        print(f"處理錯誤: {str(e)}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@APP.post("/transcribe2")
async def transcribe(audio: UploadFile = File(...)):
    try:
        # 初始化Google Speech client
        client = speech.SpeechClient()
        
        # 讀取上傳的音頻文件
        audio_content = await audio.read()
        
        # 配置識別參數
        audio = speech.RecognitionAudio(content=audio_content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,  # 改為 LINEAR16
            sample_rate_hertz=16000,
            language_code="zh-TW",
            enable_automatic_punctuation=True
        )
        
        # 執行識別
        response = client.recognize(config=config, audio=audio)
        
        # 提取識別結果
        input_text = ""
        for result in response.results:
            input_text += result.alternatives[0].transcript
        
        # 調用聊天機器人API
        url = "https://ddgsrvchat.aicreate360.com/custom_service"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoxLCJlbWFpbCI6ImFkbWluQGFpY3JlYXRlMzYwLmNvbSIsInJvbGUiOiJhZG1pbiIsImlhdCI6MTYwNjQ4NjU5Mn0.0j5z1E1l1z8Vr6ZtX9Dz4J6G1m6wD3jV5b8W7Xw5Y2c"
        }
        
        payload = {
            "type": "message",
            "message": {
                "type": "text",
                "id": "516483564309840636",
                "quoteToken": "y-3002SPcbDa-XFl-rsAqsG3bJpRryEIkcqI3mVWv-6eyDLpQsZGKYdsk8A66RSXu-FHoPzdADkeThELij-f_duq6OOr5lqPcoTXeF6_ilfoU4WydySk3GvizPX9Q01xTOtmYOaIRpVxV4FwfhbhSw",
                "text": converter.convert(input_text)
            },
            "webhookEventId": "01J2G7V0RXWR040AK1Q8PQD2R0",
            "deliveryContext": {
                "isRedelivery": False
            },
            "timestamp": 1720679498017,
            "source": {
                "type": "user",
                "userId": "U03cd17c02ef4a297c2c2a910e5b6219f"
            },
            "replyToken": "d6e582b0f794446c980b50653439d9f3",
            "mode": "active"
        }
        
        print("呼叫聊天機器人")
        response = requests.post(url, headers=headers, json=payload)
        response_json = response.json()
        message = response_json['message']
        
        return_json = {
            "input_text": converter.convert(input_text),
            "text": converter.convert(message)
        }
        return JSONResponse(return_json)
        
    except Exception as e:
        print(f"Error in transcribe: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )



import uvicorn


# 設置日誌級別
logging.getLogger("multipart").setLevel(logging.WARNING)
logging.getLogger("faster_whisper").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

if __name__ == "__main__":
    try:
        uvicorn.run(app=APP, host=host, port=port, workers=1, log_level="info")
    except Exception as e:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)
