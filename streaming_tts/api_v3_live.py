import signal
import os
import sys
import traceback
from typing import Generator, Dict, List,Union,Optional
import logging
import time
import json
import threading
import yaml
import torch
import asyncio
import re
now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))
print(sys.path)

from starlette.middleware.cors import CORSMiddleware 
from fastapi import FastAPI, Request, Response, Depends, UploadFile, File, Form, HTTPException
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
import random
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
tts_pipeline2 = TTS(TTS_Config("GPT_SoVITS/configs/tts_infer2.yaml"))
tts_pipeline3 = TTS(TTS_Config("GPT_SoVITS/configs/tts_infer_aika2.yaml"))
tts_pipeline4 = TTS(TTS_Config("GPT_SoVITS/configs/tts_infer_aika3.yaml"))


def load_speech_phrases(file_path: str) -> List[str]:
    """
    從文本文件中讀取語音識別關鍵詞
    
    Args:
        file_path: 關鍵詞文件路徑
        
    Returns:
        關鍵詞列表，如果讀取失敗則返回空列表
    """
    try:
        phrases = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                phrase = line.strip()  # 去除前後空白字符
                if phrase:  # 忽略空行
                    phrases.append(phrase)
        
        print(f"[{datetime.now()}] 成功載入 {len(phrases)} 個語音識別關鍵詞: {phrases}")
        return phrases
        
    except Exception as e:
        print(f"[{datetime.now()}] 錯誤：讀取關鍵詞文件失敗 {str(e)}，返回空列表")
        return []

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
                        
                        print(f"[{datetime.now()}] \r處理進度：已處理{total_chunks}個片段，總字數={text_length}字，"
                            f"[{datetime.now()}] 已用時間={elapsed_time:.2f}秒，音頻長度={audio_duration:.2f}秒，"
                            f"[{datetime.now()}] 當前生成速度={chars_per_process_second:.2f}字/秒，"
                            f"[{datetime.now()}] 音頻每秒字數={chars_per_audio_second:.2f}字/秒", end="", flush=True)
                        
                    # 合併所有塊並生成完整WAV
                    full_audio = np.concatenate(audio_chunks)
                    bio = BytesIO()
                    with wave.open(bio, 'wb') as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(first_sr)
                        wav_file.writeframes(full_audio.tobytes())
                        
                    bio.seek(0)
                    print(f"[{datetime.now()}] \n處理完成！")
                    print(f"[{datetime.now()}] 最終統計：")
                    print(f"[{datetime.now()}] - 總字數：{text_length}字")
                    print(f"[{datetime.now()}] - 音頻長度：{audio_duration:.2f}秒")
                    print(f"[{datetime.now()}] - 處理耗時：{elapsed_time:.2f}秒")
                    print(f"[{datetime.now()}] - 當前生成速度={chars_per_process_second:.2f}字/秒")
                    print(f"[{datetime.now()}] - 音頻每秒字數={chars_per_audio_second:.2f}字/秒")
                    
                    yield bio.read()
                elif media_type == "ogg":
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
                        
                        print(f"[{datetime.now()}] \r處理進度：已處理{total_chunks}個片段，總字數={text_length}字，"
                            f"[{datetime.now()}] 已用時間={elapsed_time:.2f}秒，音頻長度={audio_duration:.2f}秒，"
                            f"[{datetime.now()}] 當前生成速度={chars_per_process_second:.2f}字/秒，"
                            f"[{datetime.now()}] 音頻每秒字數={chars_per_audio_second:.2f}字/秒", end="", flush=True)
                    
                    # 合併所有塊並生成完整OGG
                    full_audio = np.concatenate(audio_chunks)
                    bio = BytesIO()
                    
                    # 使用soundfile代替wave處理OGG格式
                    with sf.SoundFile(bio, mode='w', samplerate=first_sr, channels=1, format='ogg') as ogg_file:
                        ogg_file.write(full_audio)
                    
                    bio.seek(0)
                    print(f"[{datetime.now()}] \n處理完成！")
                    print(f"[{datetime.now()}] 最終統計：")
                    print(f"[{datetime.now()}] - 總字數：{text_length}字")
                    print(f"[{datetime.now()}] - 音頻長度：{audio_duration:.2f}秒")
                    print(f"[{datetime.now()}] - 處理耗時：{elapsed_time:.2f}秒")
                    print(f"[{datetime.now()}] - 當前生成速度={chars_per_process_second:.2f}字/秒")
                    print(f"[{datetime.now()}] - 音頻每秒字數={chars_per_audio_second:.2f}字/秒")
                    
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
                        
                        print(f"[{datetime.now()}] \r處理進度：已處理{total_chunks}個片段，總字數={text_length}字，"
                            f"[{datetime.now()}] 已用時間={elapsed_time:.2f}秒，音頻長度={audio_duration:.2f}秒，"
                            f"[{datetime.now()}] 當前生成速度={chars_per_process_second:.2f}字/秒，"
                            f"[{datetime.now()}] 音頻每秒字數={chars_per_audio_second:.2f}字/秒", end="", flush=True)
                        
                    # 合併所有塊並生成完整WAV
                    full_audio = np.concatenate(audio_chunks)
                    bio = BytesIO()
                    with wave.open(bio, 'wb') as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(first_sr)
                        wav_file.writeframes(full_audio.tobytes())
                        
                    bio.seek(0)
                    print(f"[{datetime.now()}] \n處理完成！")
                    print(f"[{datetime.now()}] 最終統計：")
                    print(f"[{datetime.now()}] - 總字數：{text_length}字")
                    print(f"[{datetime.now()}] - 音頻長度：{audio_duration:.2f}秒")
                    print(f"[{datetime.now()}] - 處理耗時：{elapsed_time:.2f}秒")
                    print(f"[{datetime.now()}] - 當前生成速度={chars_per_process_second:.2f}字/秒")
                    print(f"[{datetime.now()}] - 音頻每秒字數={chars_per_audio_second:.2f}字/秒")
                    
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

async def tts_handle2(req:dict):
    streaming_mode = req.get("streaming_mode", False)
    media_type = req.get("media_type", "wav")

    check_res = check_params(req)
    if check_res is not None:
        return check_res

    if streaming_mode:
        req["return_fragment"] = True
    
    try:
        tts_generator = tts_pipeline2.run(req)
        
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
                        
                        print(f"[{datetime.now()}] \r處理進度：已處理{total_chunks}個片段，總字數={text_length}字，"
                            f"[{datetime.now()}] 已用時間={elapsed_time:.2f}秒，音頻長度={audio_duration:.2f}秒，"
                            f"[{datetime.now()}] 當前生成速度={chars_per_process_second:.2f}字/秒，"
                            f"[{datetime.now()}] 音頻每秒字數={chars_per_audio_second:.2f}字/秒", end="", flush=True)
                        
                    # 合併所有塊並生成完整WAV
                    full_audio = np.concatenate(audio_chunks)
                    bio = BytesIO()
                    with wave.open(bio, 'wb') as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(first_sr)
                        wav_file.writeframes(full_audio.tobytes())
                        
                    bio.seek(0)
                    print(f"[{datetime.now()}] \n處理完成！")
                    print(f"[{datetime.now()}] 最終統計：")
                    print(f"[{datetime.now()}] - 總字數：{text_length}字")
                    print(f"[{datetime.now()}] - 音頻長度：{audio_duration:.2f}秒")
                    print(f"[{datetime.now()}] - 處理耗時：{elapsed_time:.2f}秒")
                    print(f"[{datetime.now()}] - 當前生成速度={chars_per_process_second:.2f}字/秒")
                    print(f"[{datetime.now()}] - 音頻每秒字數={chars_per_audio_second:.2f}字/秒")
                    
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

# TTS 處理函數
async def tts_handle_fun(req:dict, tts_pipeline):
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
                        
                        print(f"[{datetime.now()}] \r處理進度：已處理{total_chunks}個片段，總字數={text_length}字，"
                            f"[{datetime.now()}] 已用時間={elapsed_time:.2f}秒，音頻長度={audio_duration:.2f}秒，"
                            f"[{datetime.now()}] 當前生成速度={chars_per_process_second:.2f}字/秒，"
                            f"[{datetime.now()}] 音頻每秒字數={chars_per_audio_second:.2f}字/秒", end="", flush=True)
                        
                    # 合併所有塊並生成完整WAV
                    full_audio = np.concatenate(audio_chunks)
                    bio = BytesIO()
                    with wave.open(bio, 'wb') as wav_file:
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(first_sr)
                        wav_file.writeframes(full_audio.tobytes())
                        
                    bio.seek(0)
                    print(f"[{datetime.now()}] \n處理完成！")
                    print(f"[{datetime.now()}] 最終統計：")
                    print(f"[{datetime.now()}] - 總字數：{text_length}字")
                    print(f"[{datetime.now()}] - 音頻長度：{audio_duration:.2f}秒")
                    print(f"[{datetime.now()}] - 處理耗時：{elapsed_time:.2f}秒")
                    print(f"[{datetime.now()}] - 當前生成速度={chars_per_process_second:.2f}字/秒")
                    print(f"[{datetime.now()}] - 音頻每秒字數={chars_per_audio_second:.2f}字/秒")
                    
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
    # 首先檢查文本是否為空
    if text is None:
        raise HTTPException(status_code=400, detail="必須提供文本參數")
        
    # 確認有文本後才進行替換處理
    if re.search(r"oem\s*/\s*odm", text, flags=re.IGNORECASE):
        text = re.sub(r"\s*/\s*", "或", text, flags=re.IGNORECASE)
    elif re.search(r"odm\s*/\s*oem", text, flags=re.IGNORECASE):
        text = re.sub(r"\s*/\s*", "或", text, flags=re.IGNORECASE)
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
    # 首先檢查文本是否為空
    if request.text is None:
        raise HTTPException(status_code=400, detail="必須提供文本參數")
        
    # 確認有文本後才進行替換處理
    text = request.text
    if re.search(r"oem\s*/\s*odm", text, flags=re.IGNORECASE):
        request.text = re.sub(r"\s*/\s*", "或", text, flags=re.IGNORECASE)
    elif re.search(r"odm\s*/\s*oem", text, flags=re.IGNORECASE):
        request.text = re.sub(r"\s*/\s*", "或", text, flags=re.IGNORECASE)
        
    req = request.dict()
    return await tts_handle_srt(req, req1)

@APP.get("/")
async def tts_get_endpoint_default(
    text: str = None,
    text_lang: str = "auto",
    ref_audio_path: str = "./output/2025/05/20/08de02774602371870aeb28139bb672f/ref_audios/20250520064316_48_8183360_8283520.wav",
    # ref_audio_path: str = "./output/2025/02/19/c819eb70a3aa0d8b79bcb11beb3752bb/ref_audios/Lula_voice_45_7794560_7971840.wav",
    prompt_lang: str = "zh",
    prompt_text: str = "我們隊在最後一刻反敗為勝",
    # prompt_text: str = "到底在哪裡?帥哥在哪裡?我沒看到這個是真實的這是真實",
    top_k: int = 5,
    top_p: float = 1,
    temperature: float = 1,
    text_split_method: str = "cut1",
    batch_size: int = 20,
    batch_threshold: float = 0.75,
    split_bucket: bool = True,
    speed_factor: float = 1.0,
    fragment_interval: float = 0.3,
    seed: int = 2150594710,
    media_type: str = "wav",
    streaming_mode: bool = True,
    parallel_infer: bool = True,
    repetition_penalty: float = 1.8
):
    # # 首先檢查文本是否為空
    # if text is None:
    #     raise HTTPException(status_code=400, detail="必須提供文本參數")
        
    # # 確認有文本後才進行替換處理
    # if re.search(r"oem\s*/\s*odm", text, flags=re.IGNORECASE):
    #     text = re.sub(r"\s*/\s*", "或", text, flags=re.IGNORECASE)
    # elif re.search(r"odm\s*/\s*oem", text, flags=re.IGNORECASE):
    #     text = re.sub(r"\s*/\s*", "或", text, flags=re.IGNORECASE)


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
    # # 首先檢查文本是否為空
    # if text is None:
    #     raise HTTPException(status_code=400, detail="必須提供文本參數")
        
    # # 確認有文本後才進行替換處理
    # if re.search(r"oem\s*/\s*odm", text, flags=re.IGNORECASE):
    #     text = re.sub(r"\s*/\s*", "或", text, flags=re.IGNORECASE)
    # elif re.search(r"odm\s*/\s*oem", text, flags=re.IGNORECASE):
    #     text = re.sub(r"\s*/\s*", "或", text, flags=re.IGNORECASE)


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

@APP.get("/maxgut")
async def tts_get_endpoint_test(
    text: str = None,
    text_lang: str = "auto",
    ref_audio_path: str = "./output/2025/05/12/fa08e17adea18c4292c0e4e6bea3b7f8/ref_audios/20250512071334_33_5771520_5927360.wav",
    prompt_lang: str = "zh",
    prompt_text: str = "餐廳服務生的態度實在太差了讓人感覺被冒犯",
    top_k: int = 5,
    top_p: float = 1,
    temperature: float = 1,
    text_split_method: str = "cut1",
    batch_size: int = 20,
    batch_threshold: float = 0.75,
    split_bucket: bool = True,
    speed_factor: float = 1.0,
    fragment_interval: float = 0.3,
    seed: int = 1771600266,
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

@APP.get("/aika2")
async def tts_get_endpoint_default(
    text: str = None,
    text_lang: str = "auto",
    ref_audio_path: str = "./output/2025/05/20/af1c448616622e9693d4662ac8d77c89/ref_audios/20250520064937_3_477440_698240.wav",
    # ref_audio_path: str = "./output/2025/02/19/c819eb70a3aa0d8b79bcb11beb3752bb/ref_audios/Lula_voice_45_7794560_7971840.wav",
    prompt_lang: str = "zh",
    prompt_text: str = "因為相同的地方每天都不一樣啊！時光就像這條河流，看似平靜，卻暗藏湍急",
    # prompt_text: str = "到底在哪裡?帥哥在哪裡?我沒看到這個是真實的這是真實",
    top_k: int = 5,
    top_p: float = 1,
    temperature: float = 1,
    text_split_method: str = "cut1",
    batch_size: int = 20,
    batch_threshold: float = 0.75,
    split_bucket: bool = True,
    speed_factor: float = 1.0,
    fragment_interval: float = 0.3,
    seed: int = 2751940178,
    media_type: str = "wav",
    streaming_mode: bool = True,
    parallel_infer: bool = True,
    repetition_penalty: float = 1.8
):
    # # 首先檢查文本是否為空
    # if text is None:
    #     raise HTTPException(status_code=400, detail="必須提供文本參數")
        
    # # 確認有文本後才進行替換處理
    # if re.search(r"oem\s*/\s*odm", text, flags=re.IGNORECASE):
    #     text = re.sub(r"\s*/\s*", "或", text, flags=re.IGNORECASE)
    # elif re.search(r"odm\s*/\s*oem", text, flags=re.IGNORECASE):
    #     text = re.sub(r"\s*/\s*", "或", text, flags=re.IGNORECASE)


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
    return await tts_handle_fun(req, tts_pipeline3)


@APP.get("/aika3")
async def tts_get_endpoint_default(
    text: str = None,
    text_lang: str = "auto",
    ref_audio_path: str = "./output/2025/05/20/e56ef08edf563a30bfbfb4bfdfe3ad73/ref_audios/20250520072700_12_2083840_2210240.wav",
    # ref_audio_path: str = "./output/2025/02/19/c819eb70a3aa0d8b79bcb11beb3752bb/ref_audios/Lula_voice_45_7794560_7971840.wav",
    prompt_lang: str = "zh",
    prompt_text: str = "大学是人生的重要階段我會嚴謹規劃",
    # prompt_text: str = "到底在哪裡?帥哥在哪裡?我沒看到這個是真實的這是真實",
    top_k: int = 5,
    top_p: float = 1,
    temperature: float = 1,
    text_split_method: str = "cut1",
    batch_size: int = 20,
    batch_threshold: float = 0.75,
    split_bucket: bool = True,
    speed_factor: float = 1.0,
    fragment_interval: float = 0.3,
    seed: int = 1600615455,
    media_type: str = "wav",
    streaming_mode: bool = True,
    parallel_infer: bool = True,
    repetition_penalty: float = 1.8
):
    # # 首先檢查文本是否為空
    # if text is None:
    #     raise HTTPException(status_code=400, detail="必須提供文本參數")
        
    # # 確認有文本後才進行替換處理
    # if re.search(r"oem\s*/\s*odm", text, flags=re.IGNORECASE):
    #     text = re.sub(r"\s*/\s*", "或", text, flags=re.IGNORECASE)
    # elif re.search(r"odm\s*/\s*oem", text, flags=re.IGNORECASE):
    #     text = re.sub(r"\s*/\s*", "或", text, flags=re.IGNORECASE)


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
    return await tts_handle_fun(req, tts_pipeline4)

@APP.post("/")
async def tts_post_endpoint_default(request: TTS_Request):
    req = request.dict()
    return await tts_handle(req)

@APP.post("/test")
async def tts_post_endpoint_test(request: TTS_Request):
    req = request.dict()
    return await tts_handle1(req)

@APP.post("/maxgut")
async def tts_post_endpoint_test(request: TTS_Request):
    req = request.dict()
    return await tts_handle2(req)
@APP.post("/aika2")
async def tts_post_endpoint_test(request: TTS_Request):
    req = request.dict()
    return await tts_handle_fun(req, tts_pipeline3)
@APP.post("/aika3")
async def tts_post_endpoint_test(request: TTS_Request):
    req = request.dict()
    return await tts_handle_fun(req, tts_pipeline4)

model_path = "tools/asr/models/faster-whisper-large-v3-turbo"
device = "cuda"  # 或 "cuda" 如果你有 GPU
model = WhisperModel(model_path, device=device, compute_type="float32")

@APP.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    input_text = ""
    print(f"[{datetime.now()}] 開始辨識")
    
    # 使用臨時檔案來處理音頻
    import tempfile
    import os
    import subprocess
    
    # 創建兩個臨時文件：一個用於原始上傳，一個用於轉換後的文件
    original_file = tempfile.NamedTemporaryFile(delete=False, suffix=".tmp")
    original_file_name = original_file.name
    
    converted_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    converted_file_name = converted_file.name
    
    try:
        # 寫入接收的音頻數據到臨時檔案
        audio_data = await audio.read()
        print(f"[{datetime.now()}] 接收到的音頻檔案大小: {len(audio_data)} 字節")
        
        with open(original_file_name, "wb") as f:
            f.write(audio_data)
        
        # 嘗試使用FFmpeg轉換音頻為標準格式
        try:
            # 使用FFmpeg將任何音頻格式轉換為16kHz單聲道WAV
            print(f"[{datetime.now()}] 使用FFmpeg轉換音頻格式...")
            subprocess.run([
                "ffmpeg", 
                "-i", original_file_name,  # 輸入文件
                "-ar", "16000",            # 採樣率設為16kHz
                "-ac", "1",                # 單聲道
                "-c:a", "pcm_s16le",       # 16位PCM編碼
                "-y",                      # 覆蓋輸出文件（如果存在）
                converted_file_name        # 輸出文件
            ], check=True, stderr=subprocess.PIPE)
            
            print(f"[{datetime.now()}] 音頻轉換成功，保存於: {converted_file_name}")
            
            # 使用轉換後的檔案進行轉錄
            print(f"[{datetime.now()}] 開始轉錄...")
            segments, info = model.transcribe(
                audio=converted_file_name,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=700),
                language="zh",
            )
            
            for segment in segments:
                input_text += segment.text
            
            print(f"[{datetime.now()}] 轉錄完成，文本長度: {len(input_text)}")

            url = "https://ddgsrvchat.aicreate360.com/custom_service"
                
            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoxLCJlbWFpbCI6ImFkbWluQGFpY3JlYXRlMzYwLmNvbSIsInJvbGUiOiJhZG1pbiIsImlhdCI6MTYwNjQ4NjU5Mn0.0j5z1E1l1z8Vr6ZtX9Dz4J6G1m6wD3jV5b8W7Xw5Y2c"
            }
            
            # 這裡保持原有的payload結構
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
            
            print(f"[{datetime.now()}] 呼叫聊天機器人")
            response = requests.post(url, headers=headers, json=payload)
            print(f"[{datetime.now()}] 聊天機器人回應: {response.text}")
            response_json = response.json()
            message = response_json['message']
            decoded_message = message
            return_json = {
                "input_text": converter.convert(input_text),
                "text": converter.convert(decoded_message)
            }
            return JSONResponse(return_json)
            
        except subprocess.CalledProcessError as e:
            print(f"[{datetime.now()}] FFmpeg轉換錯誤: {e}")
            print(f"[{datetime.now()}] FFmpeg錯誤輸出: {e.stderr.decode() if e.stderr else '無'}")
            raise Exception(f"音頻轉換失敗: {str(e)}")
            
    except Exception as e:
        # 錯誤處理與記錄
        import traceback
        print(f"[{datetime.now()}] 轉錄處理錯誤: {str(e)}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
    finally:
        # 確保所有臨時檔案被刪除
        for temp_file in [original_file_name, converted_file_name]:
            if os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                    print(f"[{datetime.now()}] 已刪除臨時檔案: {temp_file}")
                except Exception as e:
                    print(f"[{datetime.now()}] 刪除臨時檔案失敗 {temp_file}: {str(e)}")

@APP.post("/transcribe2")
async def transcribe(audio: UploadFile = File(...)):
    try:
        # 初始化Google Speech client
        client = speech.SpeechClient()
        print(f"[{datetime.now()}] 開始轉錄...")
        # 讀取上傳的音頻文件
        audio_content = await audio.read()
        
        # 配置識別參數
        audio = speech.RecognitionAudio(content=audio_content)

        # 讀取關鍵詞文件
        speech_phrases = load_speech_phrases(os.path.join(now_dir, "speech_phrases_maxgut_zh.txt"))

        # 只有當有關鍵詞時才添加 speech_contexts
        speech_contexts = []
        if speech_phrases:  # 如果有關鍵詞才添加
            print(f"[{datetime.now()}] 發現 {len(speech_phrases)} 個關鍵詞")
            print(f"[{datetime.now()}] 關鍵詞列表: {speech_phrases}")
            speech_contexts = [
                speech.SpeechContext(
                    phrases=speech_phrases,
                    boost=20.0
                )
            ]

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="zh-TW",
            enable_automatic_punctuation=True,
            speech_contexts=speech_contexts  # 可能是空列表
        )
        
        # 執行識別
        response = client.recognize(config=config, audio=audio)
        
        # 提取識別結果
        input_text = ""
        for result in response.results:
            input_text += result.alternatives[0].transcript
        
        # 強制將獨立的 OEM/ODM 大寫，允許中間有任意空白或標點
        input_text = re.sub(
            r'(?<![A-Za-z])O\W*E\W*M(?![A-Za-z])',
            'OEM',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])O\W*D\W*M(?![A-Za-z])',
            'ODM',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])M\W*B\W*T\W*I(?![A-Za-z])',
            'MBTI',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])A\W*I\W*K\W*K\W*A(?![A-Za-z])',
            'Aikka',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])M\W*O\W*Q(?![A-Za-z])',
            'MOQ',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])G\W*M\W*P(?![A-Za-z])',
            'GMP',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])I\W*S\W*O(?![A-Za-z])',
            'ISO',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])H\W*A\W*L\W*A(?![A-Za-z])',
            'Halal',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])E\W*D\W*M(?![A-Za-z])',
            'EDM',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])S\W*D\W*S(?![A-Za-z])',
            'SDS',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])M\W*P\W*M(?![A-Za-z])',
            'MPM',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])A\W*Q\W*U\W*A\W*X\W*Y\W*L(?![A-Za-z])',
            'AQUAXYL',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])M\W*a\W*t\W*r\W*i\W*x\W*Y\W*l(?![A-Za-z])',
            'Matrixyl',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])S\W*P\W*F(?![A-Za-z])',
            'SPF',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])P\W*A(?![A-Za-z])',
            'PA',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])C\W*O\W*A(?![A-Za-z])',
            'COA',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])C\W*M\W*Y\W*K(?![A-Za-z])',
            'CMYK',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])s\W*a\W*c\W*h\W*e\W*t(?![A-Za-z])',
            'sachet',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])B\W*3(?![A-Za-z])',
            'B3',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])B\W*1\W*2(?![A-Za-z])',
            'B12',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])E\W*G\W*F(?![A-Za-z])',
            'EGF',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])P\W*P(?![A-Za-z])',
            'PP',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])U\W*V\W*A(?![A-Za-z])',
            'UVA',
            input_text,
            flags=re.IGNORECASE
        )

        print(f"[{datetime.now()}] 轉錄完成，文本長度: {len(input_text)}")

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
        
        print(f"[{datetime.now()}] 呼叫聊天機器人")
        response = requests.post(url, headers=headers, json=payload)
        print(f"[{datetime.now()}] 聊天機器人回應: {response.text}")
        response_json = response.json()
        message = response_json['message']

        message = re.sub(
            r'產品品質',
            '產品的品質',
            message,
            flags=re.IGNORECASE
        )
        
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

@APP.post("/transcribe2_en")
async def transcribe(audio: UploadFile = File(...)):
    try:
        # 初始化Google Speech client
        client = speech.SpeechClient()
        print(f"[{datetime.now()}] 開始轉錄...")
        # 讀取上傳的音頻文件
        audio_content = await audio.read()
        
        # 配置識別參數
        audio = speech.RecognitionAudio(content=audio_content)
        # 讀取關鍵詞文件
        speech_phrases = load_speech_phrases(os.path.join(now_dir, "speech_phrases_maxgut_en.txt"))

        # 只有當有關鍵詞時才添加 speech_contexts
        speech_contexts = []
        if speech_phrases:  # 如果有關鍵詞才添加
            print(f"[{datetime.now()}] 發現 {len(speech_phrases)} 個關鍵詞")
            print(f"[{datetime.now()}] 關鍵詞列表: {speech_phrases}")
            speech_contexts = [
                speech.SpeechContext(
                    phrases=speech_phrases,
                    boost=20.0
                )
            ]

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",  # 改為英文
            enable_automatic_punctuation=True,
            speech_contexts=speech_contexts  # 可能是空列表
        )
        
        # 執行識別
        response = client.recognize(config=config, audio=audio)
        
        # 提取識別結果
        input_text = ""
        for result in response.results:
            input_text += result.alternatives[0].transcript
        
        # 強制將獨立的 OEM/ODM 大寫，允許中間有任意空白或標點
        input_text = re.sub(
            r'(?<![A-Za-z])O\W*E\W*M(?![A-Za-z])',
            'OEM',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])O\W*D\W*M(?![A-Za-z])',
            'ODM',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])M\W*B\W*T\W*I(?![A-Za-z])',
            'MBTI',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])A\W*I\W*K\W*K\W*A(?![A-Za-z])',
            'Aikka',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])M\W*O\W*Q(?![A-Za-z])',
            'MOQ',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])G\W*M\W*P(?![A-Za-z])',
            'GMP',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])I\W*S\W*O(?![A-Za-z])',
            'ISO',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])H\W*A\W*L\W*A(?![A-Za-z])',
            'Halal',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])E\W*D\W*M(?![A-Za-z])',
            'EDM',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])S\W*D\W*S(?![A-Za-z])',
            'SDS',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])M\W*P\W*M(?![A-Za-z])',
            'MPM',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])A\W*Q\W*U\W*A\W*X\W*Y\W*L(?![A-Za-z])',
            'AQUAXYL',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])M\W*a\W*t\W*r\W*i\W*x\W*Y\W*l(?![A-Za-z])',
            'Matrixyl',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])S\W*P\W*F(?![A-Za-z])',
            'SPF',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])P\W*A(?![A-Za-z])',
            'PA',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])C\W*O\W*A(?![A-Za-z])',
            'COA',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])C\W*M\W*Y\W*K(?![A-Za-z])',
            'CMYK',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])s\W*a\W*c\W*h\W*e\W*t(?![A-Za-z])',
            'sachet',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])B\W*3(?![A-Za-z])',
            'B3',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])B\W*1\W*2(?![A-Za-z])',
            'B12',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])E\W*G\W*F(?![A-Za-z])',
            'EGF',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])P\W*P(?![A-Za-z])',
            'PP',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])U\W*V\W*A(?![A-Za-z])',
            'UVA',
            input_text,
            flags=re.IGNORECASE
        )

        print(f"[{datetime.now()}] 轉錄完成，文本長度: {len(input_text)}")

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
        
        print(f"[{datetime.now()}] 呼叫聊天機器人")
        response = requests.post(url, headers=headers, json=payload)
        print(f"[{datetime.now()}] 聊天機器人回應: {response.text}")
        response_json = response.json()
        message = response_json['message']
        
        message = re.sub(
            r'產品品質',
            '產品的品質',
            message,
            flags=re.IGNORECASE
        )

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

@APP.post("/transcribe3")
async def transcribe(audio: UploadFile = File(...), userId: str = Form("U03cd17c02ef4a297c2c2a910e5b6219f")):
    try:
        # 打印接收到的 user_id
        print(f"[{datetime.now()}] 接收到的 user_id: {userId}")
        
        # 初始化Google Speech client
        client = speech.SpeechClient()
        print(f"[{datetime.now()}] 開始轉錄...")
        # 讀取上傳的音頻文件
        audio_content = await audio.read()
        
        # 配置識別參數
        audio = speech.RecognitionAudio(content=audio_content)
        # 讀取關鍵詞文件
        speech_phrases = load_speech_phrases(os.path.join(now_dir, "speech_phrases_aikka_zh.txt"))
        speech_phrases2 = load_speech_phrases(os.path.join(now_dir, "speech_phrases_aikka_zh2.txt"))

        # 只有當有關鍵詞時才添加 speech_contexts
        speech_contexts = []
        if speech_phrases:  # 如果有關鍵詞才添加
            print(f"[{datetime.now()}] 發現 {len(speech_phrases)} 個關鍵詞")
            print(f"[{datetime.now()}] 關鍵詞列表: {speech_phrases}")
            speech_contexts.append(
                speech.SpeechContext(
                    phrases=speech_phrases,
                    boost=10.0
                )
            )
        if speech_phrases2:  # 如果有第二組關鍵詞才添加
            print(f"[{datetime.now()}] 發現 {len(speech_phrases2)} 個第二組關鍵詞")
            print(f"[{datetime.now()}] 第二組關鍵詞列表: {speech_phrases2}")
            speech_contexts.append(
                speech.SpeechContext(
                    phrases=speech_phrases2,
                    boost=20.0
                )
            )


        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="zh-TW",
            enable_automatic_punctuation=True,
            speech_contexts=speech_contexts  # 可能是空列表
        )
        
        
        # 執行識別
        response = client.recognize(config=config, audio=audio)
        
        # 提取識別結果
        input_text = ""
        for result in response.results:
            input_text += result.alternatives[0].transcript
        
        # 強制將獨立的 OEM/ODM 大寫，允許中間有任意空白或標點
        input_text = re.sub(
            r'(?<![A-Za-z])M\W*B\W*T\W*I(?![A-Za-z])',
            'MBTI',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])A\W*I\W*K\W*K\W*A(?![A-Za-z])',
            'Aikka',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])愛卡(?![A-Za-z])',
            'Aikka',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])FAMPO卡(?![A-Za-z])',
            'Fanpokka',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])Ok，go(?![A-Za-z])',
            'AiTAGO',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])E\W*N\W*F\W*J(?![A-Za-z])',
            'ENFJ',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])E\W*N\W*F\W*P(?![A-Za-z])',
            'ENFP',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])E\W*N\W*T\W*J(?![A-Za-z])',
            'ENTJ',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])E\W*N\W*T\W*P(?![A-Za-z])',
            'ENTP',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])I\W*N\W*F\W*J(?![A-Za-z])',
            'INFJ',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(    
            r'(?<![A-Za-z])I\W*N\W*F\W*P(?![A-Za-z])',
            'INFP',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])I\W*N\W*T\W*J(?![A-Za-z])',
            'INTJ',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])I\W*N\W*T\W*P(?![A-Za-z])',
            'INTP',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])I\W*S\W*F\W*J(?![A-Za-z])',
            'ISFJ',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])I\W*S\W*F\W*P(?![A-Za-z])',
            'ISFP',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])I\W*S\W*T\W*J(?![A-Za-z])',
            'ISTJ',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])I\W*S\W*T\W*P(?![A-Za-z])',
            'ISTP',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])E\W*S\W*F\W*J(?![A-Za-z])',
            'ESFJ',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])E\W*S\W*F\W*P(?![A-Za-z])',
            'ESFP',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(    
            r'(?<![A-Za-z])E\W*S\W*T\W*J(?![A-Za-z])',
            'ESTJ',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])E\W*S\W*T\W*P(?![A-Za-z])',
            'ESTP',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'1 1 1',
            '1',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])C\W*R\W*M(?![A-Za-z])',
            'CRM',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])M\W*C\W*N(?![A-Za-z])',
            'MCN',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])O\W*M\W*O(?![A-Za-z])',
            'OMO',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])A\W*I\W*G\W*C(?![A-Za-z])',
            'AIGC',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])A\W*I\W*D\W*C(?![A-Za-z])',
            'AIGC',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])A\W*I\W*G 1 1(?![A-Za-z])',
            'AIGC',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])地\W*2(?![A-Za-z])',
            '第二',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])的\W*2(?![A-Za-z])',
            '第二',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])第\W*2(?![A-Za-z])',
            '第二',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])第\W*1(?![A-Za-z])',
            '第一',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])地\W*1(?![A-Za-z])',
            '第一',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])的\W*1(?![A-Za-z])',
            '第一',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])的1 1(?![A-Za-z])',
            '第一',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])A\W*I C\W*C(?![A-Za-z])',
            'AIGC',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])A\W*I技第(?![A-Za-z])',
            'AIGC',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])A\W*I P\W*C(?![A-Za-z])',
            'AIGC',
            input_text,
            flags=re.IGNORECASE
        )


        # 使用正則表達式方式去除末尾的中文句號
        input_text = re.sub(r'。$', '', input_text)

        print(f"[{datetime.now()}] 轉錄完成，文本長度: {len(input_text)}")

        # 調用聊天機器人API
        url = "http://35.201.212.192:5000/custom_service"
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
                "userId": userId
            },
            "replyToken": "d6e582b0f794446c980b50653439d9f3",
            "mode": "active"
        }
        
        print(f"[{datetime.now()}] 呼叫聊天機器人")
        response = requests.post(url, headers=headers, json=payload)
        print(f"[{datetime.now()}] 聊天機器人回應: {response.text}")
        response_json = response.json()
        message = response_json['message']

        print(f"[{datetime.now()}] 聊天機器人回應的消息: {message}")

        conversation_id = response_json['conversation_id']
        print(f"[{datetime.now()}] 會話 ID: {conversation_id}")
        external_user_id = response_json['external_user_id']
        print(f"[{datetime.now()}] 外部用戶 ID: {external_user_id}")
        result_url = response_json.get('metadata', {}).get('result_url', '')
        print(f"[{datetime.now()}] QR Code 結果 URL: {result_url}")
        
        if 'INFP' in message or 'INFJ' in message or 'INTP' in message or 'INTJ' in message or 'ISFP' in message or 'ISFJ' in message or 'ISTP' in message or 'ISTJ' in message or 'ENFP' in message or 'ENFJ' in message or 'ENTP' in message or 'ENTJ' in message or 'ESFP' in message or 'ESFJ' in message or 'ESTP' in message or 'ESTJ' in message:
            if result_url=='':
                # 在選擇前重新設定隨機種子
                random.seed(time.time())  # 使用當前時間作為種子

                MBTI_list = ['INFP', 'INFJ', 'INTP', 'INTJ', 'ISFP', 'ISFJ', 'ISTP', 'ISTJ', 'ENFP', 'ENFJ', 'ENTP', 'ENTJ', 'ESFP', 'ESFJ', 'ESTP', 'ESTJ']
                random_MBTI = random.choice(MBTI_list)
                print(f"[{datetime.now()}] 沒有提供 QR Code 結果 URL，使用隨機 MBTI: {random_MBTI}")
                result_url = f'https://liff.line.me/2007453830-aEP7r5ep?user_id=f449d1e1-2a8f-4154-80d5-b3f65d5fa2c8&mbti_type={random_MBTI}'

        message = re.sub(
            r'喔；',
            '喔～',
            message,
            flags=re.IGNORECASE
        )
        
        return_json = {
            "input_text": converter.convert(input_text),
            "text": converter.convert(message),
            "userId": external_user_id,
            "conversation_id": conversation_id,
            "result_url": result_url
        }
        return JSONResponse(return_json)
        
    except Exception as e:
        print(f"Error in transcribe: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@APP.post("/transcribe4")
async def transcribe(audio: UploadFile = File(...), userId: str = Form("U03cd17c02ef4a297c2c2a910e5b6219f")):
    try:
        # 打印接收到的 user_id
        print(f"[{datetime.now()}] 接收到的 user_id: {userId}")
        
        # 初始化Google Speech client
        client = speech.SpeechClient()
        print(f"[{datetime.now()}] 開始轉錄...")
        # 讀取上傳的音頻文件
        audio_content = await audio.read()
        
        # 配置識別參數
        audio = speech.RecognitionAudio(content=audio_content)
        # 讀取關鍵詞文件
        speech_phrases = load_speech_phrases(os.path.join(now_dir, "speech_phrases_aikka_zh.txt"))
        speech_phrases2 = load_speech_phrases(os.path.join(now_dir, "speech_phrases_aikka_zh2.txt"))

        # 只有當有關鍵詞時才添加 speech_contexts
        speech_contexts = []
        if speech_phrases:  # 如果有關鍵詞才添加
            print(f"[{datetime.now()}] 發現 {len(speech_phrases)} 個關鍵詞")
            print(f"[{datetime.now()}] 關鍵詞列表: {speech_phrases}")
            speech_contexts.append(
                speech.SpeechContext(
                    phrases=speech_phrases,
                    boost=10.0
                )
            )
        if speech_phrases2:  # 如果有第二組關鍵詞才添加
            print(f"[{datetime.now()}] 發現 {len(speech_phrases2)} 個第二組關鍵詞")
            print(f"[{datetime.now()}] 第二組關鍵詞列表: {speech_phrases2}")
            speech_contexts.append(
                speech.SpeechContext(
                    phrases=speech_phrases2,
                    boost=20.0
                )
            )


        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="zh-TW",
            enable_automatic_punctuation=True,
            speech_contexts=speech_contexts  # 可能是空列表
        )
        
        
        # 執行識別
        response = client.recognize(config=config, audio=audio)
        
        # 提取識別結果
        input_text = ""
        for result in response.results:
            input_text += result.alternatives[0].transcript
        
        # 強制將獨立的 OEM/ODM 大寫，允許中間有任意空白或標點
        input_text = re.sub(
            r'(?<![A-Za-z])A\W*I\W*K\W*K\W*A(?![A-Za-z])',
            'Aikka',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])愛卡(?![A-Za-z])',
            'Aikka',
            input_text,
            flags=re.IGNORECASE
        )
        input_text = re.sub(
            r'(?<![A-Za-z])五是(?![A-Za-z])',
            '五四',
            input_text,
            flags=re.IGNORECASE
        )
        
        # 使用正則表達式方式去除末尾的中文句號
        input_text = re.sub(r'。$', '', input_text)

        print(f"[{datetime.now()}] 轉錄完成，文本長度: {len(input_text)}")
        print(f"[{datetime.now()}] 轉錄結果: {input_text}")

        if len(input_text)==0 or input_text.isspace():
            print("無說話內容,修改輸入給聊天機器人的內容變成.")
            input_text="."

        # 調用聊天機器人API
        url = "http://llm.5gao.ai:1987/api/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer fastgpt-bQfldADmYtYOaOaZZFanv7oet7oZel1BQWfgiHuOSNIJm4TzVbi8DC"
        }
        
        payload = {
            "chatId": userId,
            "stream": False,
            "detail": False,
            "messages": [
                {
                    "content": converter.convert(input_text),
                    "role": "user"
                }
            ]
        }
        
        print(f"[{datetime.now()}] 呼叫聊天機器人")
        response = requests.post(url, headers=headers, json=payload)
        print(f"[{datetime.now()}] 聊天機器人回應: {response.text}")
        response_json = response.json()
        message = response_json['choices'][0]['message']['content']

        print(f"[{datetime.now()}] 聊天機器人回應的消息: {message}")

        conversation_id = response_json['id']
        print(f"[{datetime.now()}] 會話 ID: {conversation_id}")
        converted_input = converter.convert(input_text)
        converted_input = re.sub(
            r'(?<![A-Za-z])喫(?![A-Za-z])',
            '吃',
            converted_input,
            flags=re.IGNORECASE
        )
        converted_input = re.sub(
            r'(?<![A-Za-z])羣(?![A-Za-z])',
            '群',
            converted_input,
            flags=re.IGNORECASE
        )
        converted_input = re.sub(
            r'(?<![A-Za-z])嬢(?![A-Za-z])',
            '娘',
            converted_input,
            flags=re.IGNORECASE
        )
        print(f"[{datetime.now()}] 轉換後的輸入內容: {converted_input}")
        converted_message = converter.convert(message)
        converted_message = re.sub(
            r'(?<![A-Za-z])喫(?![A-Za-z])',
            '吃',
            converted_message,
            flags=re.IGNORECASE
        )
        converted_message = re.sub(
            r'(?<![A-Za-z])羣(?![A-Za-z])',
            '群',
            converted_message,
            flags=re.IGNORECASE
        )
        converted_message = re.sub(
            r'(?<![A-Za-z])嬢(?![A-Za-z])',
            '娘',
            converted_message,
            flags=re.IGNORECASE
        )
        print(f"[{datetime.now()}] 轉換後的回應內容: {converted_message}")
        
        
        return_json = {
            "input_text": converted_input,
            "text": converted_message,
            "userId": userId,
            "conversation_id": conversation_id
        }
        return JSONResponse(return_json)
        
    except Exception as e:
        print(f"Error in transcribe: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )




@APP.post("/ai_chat")
async def ai_chat(request_data: Dict[str, str]):
    try:
        # 檢查 text 欄位
        input_text = request_data.get("text")
        if not input_text:
            raise HTTPException(status_code=400, detail="必須提供 text 參數")
        
        # 轉換繁簡或其他需要的處理
        converted_input = converter.convert(input_text)

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
                "text": converted_input
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
        
        print(f"[{datetime.now()}] 呼叫聊天機器人")
        response = requests.post(url, headers=headers, json=payload)
        print(f"[{datetime.now()}] 聊天機器人回應: {response.text}")
        response_json = response.json()
        message = response_json['message']

        message = re.sub(
            r'產品品質',
            '產品的品質',
            message,
            flags=re.IGNORECASE
        )
        
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

@APP.post("/ai_chat_en")
async def ai_chat(request_data: Dict[str, str]):
    try:
        # 檢查 text 欄位
        input_text = request_data.get("text")
        if not input_text:
            raise HTTPException(status_code=400, detail="必須提供 text 參數")
        
        # 轉換繁簡或其他需要的處理
        converted_input = converter.convert(input_text)

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
                "text": converted_input
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
        
        print(f"[{datetime.now()}] 呼叫聊天機器人")
        response = requests.post(url, headers=headers, json=payload)
        print(f"[{datetime.now()}] 聊天機器人回應: {response.text}")
        response_json = response.json()
        message = response_json['message']

        message = re.sub(
            r'產品品質',
            '產品的品質',
            message,
            flags=re.IGNORECASE
        )
        
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

@APP.post("/ai_chat_aicreate360")
async def ai_chat(request_data: Dict[str, str]):
    try:
        # 檢查 text 欄位
        input_text = request_data.get("text")
        if not input_text:
            raise HTTPException(status_code=400, detail="必須提供 text 參數")
        user_id = request_data.get("userId", "U03cd17c02ef4a297c2c2a910e5b6219f")

        # 轉換繁簡或其他需要的處理
        converted_input = converter.convert(input_text)

        # 調用聊天機器人API
        url = "http://35.201.212.192:5000/custom_service"
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
                "text": converted_input
            },
            "webhookEventId": "01J2G7V0RXWR040AK1Q8PQD2R0",
            "deliveryContext": {
                "isRedelivery": False
            },
            "timestamp": 1720679498017,
            "source": {
                "type": "user",
                "userId": user_id
            },
            "replyToken": "d6e582b0f794446c980b50653439d9f3",
            "mode": "active"
        }
        
        print(f"[{datetime.now()}] 呼叫聊天機器人")
        response = requests.post(url, headers=headers, json=payload)
        print(f"[{datetime.now()}] 聊天機器人回應: {response.text}")
        response_json = response.json()
        message = response_json['message']
        print(f"[{datetime.now()}] 聊天機器人回應的消息: {message}")
        conversation_id = response_json['conversation_id']
        print(f"[{datetime.now()}] 會話 ID: {conversation_id}")
        external_user_id = response_json['external_user_id']
        print(f"[{datetime.now()}] 外部用戶 ID: {external_user_id}")
        result_url = response_json.get('metadata', {}).get('result_url', '')
        print(f"[{datetime.now()}] QR Code 結果 URL: {result_url}")
        
        if 'INFP' in message or 'INFJ' in message or 'INTP' in message or 'INTJ' in message or 'ISFP' in message or 'ISFJ' in message or 'ISTP' in message or 'ISTJ' in message or 'ENFP' in message or 'ENFJ' in message or 'ENTP' in message or 'ENTJ' in message or 'ESFP' in message or 'ESFJ' in message or 'ESTP' in message or 'ESTJ' in message:
            if result_url=='':
                # 在選擇前重新設定隨機種子
                random.seed(time.time())  # 使用當前時間作為種子

                MBTI_list = ['INFP', 'INFJ', 'INTP', 'INTJ', 'ISFP', 'ISFJ', 'ISTP', 'ISTJ', 'ENFP', 'ENFJ', 'ENTP', 'ENTJ', 'ESFP', 'ESFJ', 'ESTP', 'ESTJ']
                random_MBTI = random.choice(MBTI_list)
                print(f"[{datetime.now()}] 沒有提供 QR Code 結果 URL，使用隨機 MBTI: {random_MBTI}")
                result_url = f'https://liff.line.me/2007453830-aEP7r5ep?user_id=f449d1e1-2a8f-4154-80d5-b3f65d5fa2c8&mbti_type={random_MBTI}'
        elif input_text == 'SET3QDAY':
            if result_url=='':
                # 在選擇前重新設定隨機種子
                random.seed(time.time())  # 使用當前時間作為種子

                MBTI_list = ['INFP', 'INFJ', 'INTP', 'INTJ', 'ISFP', 'ISFJ', 'ISTP', 'ISTJ', 'ENFP', 'ENFJ', 'ENTP', 'ENTJ', 'ESFP', 'ESFJ', 'ESTP', 'ESTJ']
                random_MBTI = random.choice(MBTI_list)
                print(f"[{datetime.now()}] 沒有提供 QR Code 結果 URL，使用隨機 MBTI: {random_MBTI}")
                result_url = f'https://liff.line.me/2007453830-aEP7r5ep?user_id=f449d1e1-2a8f-4154-80d5-b3f65d5fa2c8&mbti_type={random_MBTI}'
        
        message = re.sub(
            r'喔；',
            '喔～',
            message,
            flags=re.IGNORECASE
        )
        
        return_json = {
            "input_text": converter.convert(input_text),
            "text": converter.convert(message),
            "userId": external_user_id,
            "conversation_id": conversation_id,
            "result_url": result_url
        }
        return JSONResponse(return_json)
        
    except Exception as e:
        print(f"Error in transcribe: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@APP.post("/ai_chat_set")
async def ai_chat(request_data: Dict[str, str]):
    try:
        # 檢查 text 欄位
        input_text = request_data.get("text")
        if not input_text:
            raise HTTPException(status_code=400, detail="必須提供 text 參數")
        user_id = request_data.get("userId", "U03cd17c02ef4a297c2c2a910e5b6219f")

        # 轉換繁簡或其他需要的處理
        converted_input = converter.convert(input_text)
        converted_input = re.sub(
            r'(?<![A-Za-z])喫(?![A-Za-z])',
            '吃',
            converted_input,
            flags=re.IGNORECASE
        )
        converted_input = re.sub(
            r'(?<![A-Za-z])羣(?![A-Za-z])',
            '群',
            converted_input,
            flags=re.IGNORECASE
        )
        converted_input = re.sub(
            r'(?<![A-Za-z])嬢(?![A-Za-z])',
            '娘',
            converted_input,
            flags=re.IGNORECASE
        )
        print(f"[{datetime.now()}] 轉換後的輸入內容: {converted_input}")
        


        # 調用聊天機器人API
        url = "http://llm.5gao.ai:1987/api/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer fastgpt-bQfldADmYtYOaOaZZFanv7oet7oZel1BQWfgiHuOSNIJm4TzVbi8DC"
        }
        
        payload = {
            "chatId": user_id,
            "stream": False,
            "detail": False,
            "messages": [
                {
                    "content": converted_input,
                    "role": "user"
                }
            ]
        }
        
        print(f"[{datetime.now()}] 呼叫聊天機器人")
        response = requests.post(url, headers=headers, json=payload)
        print(f"[{datetime.now()}] 聊天機器人回應: {response.text}")
        response_json = response.json()
        message = response_json['choices'][0]['message']['content']

        print(f"[{datetime.now()}] 聊天機器人回應的消息: {message}")

        conversation_id = response_json['id']
        print(f"[{datetime.now()}] 會話 ID: {conversation_id}")
        converted_message = converter.convert(message)
        converted_message = re.sub(
            r'(?<![A-Za-z])喫(?![A-Za-z])',
            '吃',
            converted_message,
            flags=re.IGNORECASE
        )
        converted_message = re.sub(
            r'(?<![A-Za-z])羣(?![A-Za-z])',
            '群',
            converted_message,
            flags=re.IGNORECASE
        )
        converted_message = re.sub(
            r'(?<![A-Za-z])嬢(?![A-Za-z])',
            '娘',
            converted_message,
            flags=re.IGNORECASE
        )
        print(f"[{datetime.now()}] 轉換後的回應內容: {converted_message}")
        
        
        return_json = {
            "input_text": converted_input,
            "text": converted_message,
            "userId": user_id,
            "conversation_id": conversation_id
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
