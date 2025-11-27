"""
# WebAPI文档

` python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml `

## 执行参数:
    `-a` - `绑定地址, 默认"127.0.0.1"`
    `-p` - `绑定端口, 默认9880`
    `-c` - `TTS配置文件路径, 默认"GPT_SoVITS/configs/tts_infer.yaml"`

## 调用:

### 推理

endpoint: `/tts`
GET:
```
http://127.0.0.1:9880/tts?text=先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。&text_lang=zh&ref_audio_path=archive_jingyuan_1.wav&prompt_lang=zh&prompt_text=我是「罗浮」云骑将军景元。不必拘谨，「将军」只是一时的身份，你称呼我景元便可&text_split_method=cut5&batch_size=1&media_type=wav&streaming_mode=true
```

POST:
```json
{
    "text": "",                   # str.(required) text to be synthesized
    "text_lang": "",              # str.(required) language of the text to be synthesized
    "ref_audio_path": "",         # str.(required) reference audio path.
    "prompt_text": "",            # str.(optional) prompt text for the reference audio
    "prompt_lang": "",            # str.(required) language of the prompt text for the reference audio
    "top_k": 5,                   # int.(optional) top k sampling
    "top_p": 1,                   # float.(optional) top p sampling
    "temperature": 1,             # float.(optional) temperature for sampling
    "text_split_method": "cut5",  # str.(optional) text split method, see text_segmentation_method.py for details.
    "batch_size": 1,              # int.(optional) batch size for inference
    "batch_threshold": 0.75,      # float.(optional) threshold for batch splitting.
    "split_bucket": true,         # bool.(optional) whether to split the batch into multiple buckets.
    "speed_factor":1.0,           # float.(optional) control the speed of the synthesized audio.
    "fragment_interval":0.3,      # float.(optional) to control the interval of the audio fragment.
    "seed": -1,                   # int.(optional) random seed for reproducibility.
    "media_type": "wav",          # str.(optional) media type of the output audio, support "wav", "raw", "ogg", "aac".
    "streaming_mode": false,      # bool.(optional) whether to return a streaming response.
    "parallel_infer": True,       # bool.(optional) whether to use parallel inference.
    "repetition_penalty": 1.35    # float.(optional) repetition penalty for T2S model.
}
```

RESP:
成功: 直接返回 wav 音频流， http code 200
失败: 返回包含错误信息的 json, http code 400

### 命令控制

endpoint: `/control`

command:
"restart": 重新运行
"exit": 结束运行

GET:
```
http://127.0.0.1:9880/control?command=restart
```
POST:
```json
{
    "command": "restart"
}
```

RESP: 无


### 切换GPT模型

endpoint: `/set_gpt_weights`

GET:
```
http://127.0.0.1:9880/set_gpt_weights?weights_path=GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt
```
RESP: 
成功: 返回"success", http code 200
失败: 返回包含错误信息的 json, http code 400


### 切换Sovits模型

endpoint: `/set_sovits_weights`

GET:
```
http://127.0.0.1:9880/set_sovits_weights?weights_path=GPT_SoVITS/pretrained_models/s2G488k.pth
```

RESP: 
成功: 返回"success", http code 200
失败: 返回包含错误信息的 json, http code 400
    
"""
import os
import sys
import traceback
from typing import Generator
import logging
import time
import json

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))

from starlette.middleware.cors import CORSMiddleware  #引入 CORS中间件模块

#设置允许访问的域名
origins = ["*"]  #"*"，即为所有。
import nltk
nltk.download('averaged_perceptron_tagger_eng')

import config as global_config

import argparse
import subprocess
import wave
import signal
import numpy as np
import soundfile as sf
from fastapi import FastAPI, Request, HTTPException, Response
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
import uvicorn
from io import BytesIO
from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import get_method_names as get_cut_method_names
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import requests
from faster_whisper import WhisperModel
# print(sys.path)
i18n = I18nAuto()
cut_method_names = get_cut_method_names()

parser = argparse.ArgumentParser(description="GPT-SoVITS api")
parser.add_argument("-c", "--tts_config", type=str, default="GPT_SoVITS/configs/tts_infer.yaml", help="tts_infer路径")
parser.add_argument("-a", "--bind_addr", type=str, default="0.0.0.0", help="default: 127.0.0.1")
parser.add_argument("-p", "--port", type=int, default="9880", help="default: 9880")
args = parser.parse_args()
config_path = args.tts_config
# device = args.device
port = args.port
host = args.bind_addr
argv = sys.argv
import opencc
converter = opencc.OpenCC('s2t')  # t2s 代表繁體到簡體
if config_path in [None, ""]:
    config_path = "GPT-SoVITS/configs/tts_infer.yaml"

tts_config = TTS_Config(config_path)
tts_pipeline = TTS(tts_config)
tts_pipeline1 = TTS(TTS_Config("GPT_SoVITS/configs/tts_infer1.yaml"))
tts_pipeline2 = TTS(TTS_Config("GPT_SoVITS/configs/tts_infer_clone.yaml"))


APP = FastAPI()



APP.mount("/srt", StaticFiles(directory="./srt"), name="srt")
APP.mount("/audio", StaticFiles(directory="./audio"), name="audio")

APP.add_middleware(
    CORSMiddleware, 
    allow_origins=origins,  #设置允许的origins来源
    allow_credentials=True,
    allow_methods=["*"],  # 设置允许跨域的http方法，比如 get、post、put等。
    allow_headers=["*"])  #允许跨域的headers，可以用来鉴别来源等作用。

class TTS_Request(BaseModel):
    text: str = None
    text_lang: str = None
    ref_audio_path: str = None
    prompt_lang: str = None
    prompt_text: str = ""
    top_k:int = 5
    top_p:float = 1
    temperature:float = 1
    text_split_method:str = "cut5"
    batch_size:int = 1
    batch_threshold:float = 0.75
    split_bucket:bool = True
    speed_factor:float = 1.0
    fragment_interval:float = 0.3
    seed:int = -1
    media_type:str = "wav"
    streaming_mode:bool = False
    parallel_infer:bool = True
    repetition_penalty:float = 1.35

### modify from https://github.com/RVC-Boss/GPT-SoVITS/pull/894/files
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
        '-f', 's16le',  # 输入16位有符号小端整数PCM
        '-ar', str(rate),  # 设置采样率
        '-ac', '1',  # 单声道
        '-i', 'pipe:0',  # 从管道读取输入
        '-c:a', 'aac',  # 音频编码器为AAC
        '-b:a', '192k',  # 比特率
        '-vn',  # 不包含视频
        '-f', 'adts',  # 输出AAC数据流格式
        'pipe:1'  # 将输出写入管道
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



# from https://huggingface.co/spaces/coqui/voice-chat-with-mistral/blob/main/app.py
def wave_header_chunk(frame_input=b"", channels=1, sample_width=2, sample_rate=32000):
    # This will create a wave header then append the frame input
    # It should be first on a streaming wav file
    # Other frames better should not have it (else you will hear some artifacts each chunk start)
    wav_buf = BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)

    wav_buf.seek(0)
    return wav_buf.read()


def handle_control(command:str):
    if command == "restart":
        os.execl(sys.executable, sys.executable, *argv)
    elif command == "exit":
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)


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
    if (text_lang in [None, ""]) :
        return JSONResponse(status_code=400, content={"message": "text_lang is required"})
    elif text_lang.lower() not in tts_config.languages:
        return JSONResponse(status_code=400, content={"message": "text_lang is not supported"})
    if (prompt_lang in [None, ""]) :
        return JSONResponse(status_code=400, content={"message": "prompt_lang is required"})
    elif prompt_lang.lower() not in tts_config.languages:
        return JSONResponse(status_code=400, content={"message": "prompt_lang is not supported"})
    if media_type not in ["wav", "raw", "ogg", "aac"]:
        return JSONResponse(status_code=400, content={"message": "media_type is not supported"})
    elif media_type == "ogg" and  not streaming_mode:
        return JSONResponse(status_code=400, content={"message": "ogg format is not supported in non-streaming mode"})
    
    if text_split_method not in cut_method_names:
        return JSONResponse(status_code=400, content={"message": f"text_split_method:{text_split_method} is not supported"})

    return None

async def tts_handle(req:dict):
    """
    Text to speech handler with processing speed measurement.
    
    Args:
        req (dict): 
            {
                "text": "",                   # str.(required) text to be synthesized
                "text_lang: "",               # str.(required) language of the text to be synthesized
                "ref_audio_path": "",         # str.(required) reference audio path
                "prompt_text": "",            # str.(optional) prompt text for the reference audio
                "prompt_lang": "",            # str.(required) language of the prompt text for the reference audio
                "top_k": 5,                   # int. top k sampling
                "top_p": 1,                   # float. top p sampling
                "temperature": 1,             # float. temperature for sampling
                "text_split_method": "cut5",  # str. text split method, see text_segmentation_method.py for details.
                "batch_size": 1,              # int. batch size for inference
                "batch_threshold": 0.75,      # float. threshold for batch splitting.
                "split_bucket: True,          # bool. whether to split the batch into multiple buckets.
                "speed_factor":1.0,           # float. control the speed of the synthesized audio.
                "fragment_interval":0.3,      # float. to control the interval of the audio fragment.
                "seed": -1,                   # int. random seed for reproducibility.
                "media_type": "wav",          # str. media type of the output audio, support "wav", "raw", "ogg", "aac".
                "streaming_mode": False,      # bool. whether to return a streaming response.
                "parallel_infer": True,       # bool.(optional) whether to use parallel inference.
                "repetition_penalty": 1.35    # float.(optional) repetition penalty for T2S model.          
            }
    returns:
        StreamingResponse: audio stream response.
    """
    
    streaming_mode = req.get("streaming_mode", False)
    media_type = req.get("media_type", "wav")

    check_res = check_params(req)
    if check_res is not None:
        return check_res

    if streaming_mode:
        req["return_fragment"] = True
    
    try:
        # 計算文本長度（字數）
        text = req.get("text", "")
        text_length = len(text)
        
        # 開始計時
        start_time = time.time()
        
        tts_generator=tts_pipeline.run(req)
        
        if streaming_mode:
            def streaming_generator(tts_generator:Generator, media_type:str):
                # 初始化統計資料
                start_time = time.time()
                total_chunks = 0
                text = req.get("text", "")
                text_length = len(text)
                total_audio_samples = 0  # 累計音頻樣本數
                
                if media_type == "wav":
                    yield wave_header_chunk()
                    media_type = "raw"
                    
                for sr, chunk in tts_generator:
                    total_chunks += 1
                    total_audio_samples += len(chunk)  # 累加音頻樣本數
                    audio_duration = total_audio_samples / sr  # 計算實際音頻長度(秒)
                    
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    chunks_per_second = total_chunks / elapsed_time if elapsed_time > 0 else 0
                    chars_per_audio_second = text_length / audio_duration if audio_duration > 0 else 0
                    
                    chars_per_process_second = text_length / elapsed_time if elapsed_time > 0 else 0
                    print(f"\r處理進度：已處理{total_chunks}個片段，總字數={text_length}字，"
                          f"已用時間={elapsed_time:.2f}秒，音頻長度={audio_duration:.2f}秒，"
                          f"當前生成速度={chars_per_process_second:.2f}字/秒，"
                          f"音頻每秒字數={chars_per_audio_second:.2f}字/秒", end="", flush=True)
                    
                    yield pack_audio(BytesIO(), chunk, sr, media_type).getvalue()
                
                # 完成時顯示最終統計
                print(f"\n處理完成！")
                print(f"最終統計：")
                print(f"- 總字數：{text_length}字")
                print(f"- 音頻長度：{audio_duration:.2f}秒")
                print(f"- 處理耗時：{elapsed_time:.2f}秒")
                print(f"- 當前生成速度：{chars_per_process_second:.2f}字/秒")
                print(f"- 音頻每秒字數：{chars_per_audio_second:.2f}字/秒")
            # _media_type = f"audio/{media_type}" if not (streaming_mode and media_type in ["wav", "raw"]) else f"audio/x-{media_type}"
            return StreamingResponse(streaming_generator(tts_generator, media_type, ), media_type=f"audio/{media_type}")
    
        else:
            sr, audio_data = next(tts_generator)
            # 計算處理時間和速度
            process_time = time.time() - start_time
            chars_per_second = text_length / process_time if process_time > 0 else 0
            print(f"處理統計：總字數={text_length}字，處理時間={process_time:.2f}秒，速度={chars_per_second:.2f}字/秒")
            audio_data = pack_audio(BytesIO(), audio_data, sr, media_type).getvalue()
            return Response(audio_data, media_type=f"audio/{media_type}")
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"tts failed", "Exception": str(e)})


async def tts_handle1(req:dict):
    """
    Text to speech handler.
    
    Args:
        req (dict): 
            {
                "text": "",                   # str.(required) text to be synthesized
                "text_lang: "",               # str.(required) language of the text to be synthesized
                "ref_audio_path": "",         # str.(required) reference audio path
                "prompt_text": "",            # str.(optional) prompt text for the reference audio
                "prompt_lang": "",            # str.(required) language of the prompt text for the reference audio
                "top_k": 5,                   # int. top k sampling
                "top_p": 1,                   # float. top p sampling
                "temperature": 1,             # float. temperature for sampling
                "text_split_method": "cut5",  # str. text split method, see text_segmentation_method.py for details.
                "batch_size": 1,              # int. batch size for inference
                "batch_threshold": 0.75,      # float. threshold for batch splitting.
                "split_bucket: True,          # bool. whether to split the batch into multiple buckets.
                "speed_factor":1.0,           # float. control the speed of the synthesized audio.
                "fragment_interval":0.3,      # float. to control the interval of the audio fragment.
                "seed": -1,                   # int. random seed for reproducibility.
                "media_type": "wav",          # str. media type of the output audio, support "wav", "raw", "ogg", "aac".
                "streaming_mode": False,      # bool. whether to return a streaming response.
                "parallel_infer": True,       # bool.(optional) whether to use parallel inference.
                "repetition_penalty": 1.35    # float.(optional) repetition penalty for T2S model.          
            }
    returns:
        StreamingResponse: audio stream response.
    """
    
    streaming_mode = req.get("streaming_mode", False)
    media_type = req.get("media_type", "wav")

    check_res = check_params(req)
    if check_res is not None:
        return check_res

    if streaming_mode:
        req["return_fragment"] = True
    
    try:
        tts_generator=tts_pipeline1.run(req)
        
        if streaming_mode:
            def streaming_generator(tts_generator:Generator, media_type:str):
                # 初始化統計資料
                start_time = time.time()
                total_chunks = 0
                text = req.get("text", "")
                text_length = len(text)
                total_audio_samples = 0  # 累計音頻樣本數
                
                if media_type == "wav":
                    yield wave_header_chunk()
                    media_type = "raw"
                    
                for sr, chunk in tts_generator:
                    total_chunks += 1
                    total_audio_samples += len(chunk)  # 累加音頻樣本數
                    audio_duration = total_audio_samples / sr  # 計算實際音頻長度(秒)
                    
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    chunks_per_second = total_chunks / elapsed_time if elapsed_time > 0 else 0
                    chars_per_audio_second = text_length / audio_duration if audio_duration > 0 else 0
                    
                    chars_per_process_second = text_length / elapsed_time if elapsed_time > 0 else 0
                    print(f"\r處理進度：已處理{total_chunks}個片段，總字數={text_length}字，"
                          f"已用時間={elapsed_time:.2f}秒，音頻長度={audio_duration:.2f}秒，"
                          f"當前生成速度={chars_per_process_second:.2f}字/秒，"
                          f"音頻每秒字數={chars_per_audio_second:.2f}字/秒", end="", flush=True)
                    
                    yield pack_audio(BytesIO(), chunk, sr, media_type).getvalue()
                
                # 完成時顯示最終統計
                print(f"\n處理完成！")
                print(f"最終統計：")
                print(f"- 總字數：{text_length}字")
                print(f"- 音頻長度：{audio_duration:.2f}秒")
                print(f"- 處理耗時：{elapsed_time:.2f}秒")
                print(f"- 當前生成速度：{chars_per_process_second:.2f}字/秒")
                print(f"- 音頻每秒字數：{chars_per_audio_second:.2f}字/秒")
            # _media_type = f"audio/{media_type}" if not (streaming_mode and media_type in ["wav", "raw"]) else f"audio/x-{media_type}"
            return StreamingResponse(streaming_generator(tts_generator, media_type, ), media_type=f"audio/{media_type}")
    
        else:
            sr, audio_data = next(tts_generator)
            # 計算處理時間和速度
            process_time = time.time() - start_time
            chars_per_second = text_length / process_time if process_time > 0 else 0
            print(f"處理統計：總字數={text_length}字，處理時間={process_time:.2f}秒，速度={chars_per_second:.2f}字/秒")
            audio_data = pack_audio(BytesIO(), audio_data, sr, media_type).getvalue()
            return Response(audio_data, media_type=f"audio/{media_type}")
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"tts failed", "Exception": str(e)})


async def tts_handle2(req:dict):
    """
    Text to speech handler.
    
    Args:
        req (dict): 
            {
                "text": "",                   # str.(required) text to be synthesized
                "text_lang: "",               # str.(required) language of the text to be synthesized
                "ref_audio_path": "",         # str.(required) reference audio path
                "prompt_text": "",            # str.(optional) prompt text for the reference audio
                "prompt_lang": "",            # str.(required) language of the prompt text for the reference audio
                "top_k": 5,                   # int. top k sampling
                "top_p": 1,                   # float. top p sampling
                "temperature": 1,             # float. temperature for sampling
                "text_split_method": "cut5",  # str. text split method, see text_segmentation_method.py for details.
                "batch_size": 1,              # int. batch size for inference
                "batch_threshold": 0.75,      # float. threshold for batch splitting.
                "split_bucket: True,          # bool. whether to split the batch into multiple buckets.
                "speed_factor":1.0,           # float. control the speed of the synthesized audio.
                "fragment_interval":0.3,      # float. to control the interval of the audio fragment.
                "seed": -1,                   # int. random seed for reproducibility.
                "media_type": "wav",          # str. media type of the output audio, support "wav", "raw", "ogg", "aac".
                "streaming_mode": False,      # bool. whether to return a streaming response.
                "parallel_infer": True,       # bool.(optional) whether to use parallel inference.
                "repetition_penalty": 1.35    # float.(optional) repetition penalty for T2S model.          
            }
    returns:
        StreamingResponse: audio stream response.
    """
    
    streaming_mode = req.get("streaming_mode", False)
    media_type = req.get("media_type", "wav")

    check_res = check_params(req)
    if check_res is not None:
        return check_res

    if streaming_mode:
        req["return_fragment"] = True
    
    try:
        tts_generator=tts_pipeline2.run(req)
        
        if streaming_mode:
            def streaming_generator(tts_generator:Generator, media_type:str):
                # 初始化統計資料
                start_time = time.time()
                total_chunks = 0
                text = req.get("text", "")
                text_length = len(text)
                total_audio_samples = 0  # 累計音頻樣本數
                
                if media_type == "wav":
                    yield wave_header_chunk()
                    media_type = "raw"
                    
                for sr, chunk in tts_generator:
                    total_chunks += 1
                    total_audio_samples += len(chunk)  # 累加音頻樣本數
                    audio_duration = total_audio_samples / sr  # 計算實際音頻長度(秒)
                    
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    chunks_per_second = total_chunks / elapsed_time if elapsed_time > 0 else 0
                    chars_per_audio_second = text_length / audio_duration if audio_duration > 0 else 0
                    
                    chars_per_process_second = text_length / elapsed_time if elapsed_time > 0 else 0
                    print(f"\r處理進度：已處理{total_chunks}個片段，總字數={text_length}字，"
                          f"已用時間={elapsed_time:.2f}秒，音頻長度={audio_duration:.2f}秒，"
                          f"當前生成速度={chars_per_process_second:.2f}字/秒，"
                          f"音頻每秒字數={chars_per_audio_second:.2f}字/秒", end="", flush=True)
                    
                    yield pack_audio(BytesIO(), chunk, sr, media_type).getvalue()
                
                # 完成時顯示最終統計
                print(f"\n處理完成！")
                print(f"最終統計：")
                print(f"- 總字數：{text_length}字")
                print(f"- 音頻長度：{audio_duration:.2f}秒")
                print(f"- 處理耗時：{elapsed_time:.2f}秒")
                print(f"- 當前生成速度：{chars_per_process_second:.2f}字/秒")
                print(f"- 音頻每秒字數：{chars_per_audio_second:.2f}字/秒")
            # _media_type = f"audio/{media_type}" if not (streaming_mode and media_type in ["wav", "raw"]) else f"audio/x-{media_type}"
            return StreamingResponse(streaming_generator(tts_generator, media_type, ), media_type=f"audio/{media_type}")
    
        else:
            sr, audio_data = next(tts_generator)
            # 計算處理時間和速度
            process_time = time.time() - start_time
            chars_per_second = text_length / process_time if process_time > 0 else 0
            print(f"處理統計：總字數={text_length}字，處理時間={process_time:.2f}秒，速度={chars_per_second:.2f}字/秒")
            audio_data = pack_audio(BytesIO(), audio_data, sr, media_type).getvalue()
            return Response(audio_data, media_type=f"audio/{media_type}")
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"tts failed", "Exception": str(e)})



async def tts_handle_srt(req:dict,request):
    """
    Text to speech handler.
    
    Args:
        req (dict): 
            {
                "text": "",                   # str.(required) text to be synthesized
                "text_lang: "",               # str.(required) language of the text to be synthesized
                "ref_audio_path": "",         # str.(required) reference audio path
                "prompt_text": "",            # str.(optional) prompt text for the reference audio
                "prompt_lang": "",            # str.(required) language of the prompt text for the reference audio
                "top_k": 5,                   # int. top k sampling
                "top_p": 1,                   # float. top p sampling
                "temperature": 1,             # float. temperature for sampling
                "text_split_method": "cut5",  # str. text split method, see text_segmentation_method.py for details.
                "batch_size": 1,              # int. batch size for inference
                "batch_threshold": 0.75,      # float. threshold for batch splitting.
                "split_bucket: True,          # bool. whether to split the batch into multiple buckets.
                "speed_factor":1.0,           # float. control the speed of the synthesized audio.
                "fragment_interval":0.3,      # float. to control the interval of the audio fragment.
                "seed": -1,                   # int. random seed for reproducibility.
                "media_type": "wav",          # str. media type of the output audio, support "wav", "raw", "ogg", "aac".
                "streaming_mode": False,      # bool. whether to return a streaming response.
                "parallel_infer": True,       # bool.(optional) whether to use parallel inference.
                "repetition_penalty": 1.35    # float.(optional) repetition penalty for T2S model.          
            }
    returns:
        StreamingResponse: audio stream response.
    """
    
    streaming_mode = req.get("streaming_mode", False)
    media_type = req.get("media_type", "wav")

    check_res = check_params(req)
    if check_res is not None:
        return check_res

    
    try:
        tts_generator=tts_pipeline.run(req)
        
        sr, audio_data = next(tts_generator)
        print(audio_data)
        #audio_data = pack_audio(BytesIO(), audio_data, sr, media_type).getvalue()
        #return Response(audio_data, media_type=f"audio/{media_type}")
        return JSONResponse({"code":"200", "srt":f"http://{request.url.hostname}:{request.url.port}/srt/tts-out.srt","audio":f"http://{request.url.hostname}:{request.url.port}/audio/audio.wav"})
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"tts failed", "Exception": str(e)})
    





@APP.get("/control")
async def control(command: str = None):
    if command is None:
        return JSONResponse(status_code=400, content={"message": "command is required"})
    handle_control(command)



@APP.get("/srt")
async def tts_get_endpoint_srt(request: Request,
                        text: str = None,
                        text_lang: str = None,
                        ref_audio_path: str = None,
                        prompt_lang: str = None,
                        prompt_text: str = "",
                        top_k:int = 5,
                        top_p:float = 1,
                        temperature:float = 1,
                        text_split_method:str = "cut5",
                        batch_size:int = 10,
                        batch_threshold:float = 0.75,
                        split_bucket:bool = True,
                        speed_factor:float = 1.0,
                        fragment_interval:float = 0.3,
                        seed:int = -1,
                        media_type:str = "wav",
                        streaming_mode:bool = False,
                        parallel_infer:bool = True,
                        repetition_penalty:float = 1.35
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
        "batch_size":int(batch_size),
        "batch_threshold":float(batch_threshold),
        "speed_factor":float(speed_factor),
        "split_bucket":split_bucket,
        "fragment_interval":fragment_interval,
        "seed":seed,
        "media_type":media_type,
        "streaming_mode":streaming_mode,
        "parallel_infer":parallel_infer,
        "repetition_penalty":float(repetition_penalty)
    }
    return await tts_handle_srt(req,request)

@APP.post("/srt")
async def tts_post_endpoint_srt(request: TTS_Request,req1: Request):
    req = request.dict()
    return await tts_handle_srt(req,req1)

@APP.get("/")
async def tts_get_endpoint(
                        text: str = None,
                        text_lang: str = "auto", # None
                        # ref_audio_path: str = "logs/sweet/5-wav32k/sweet.wav_0005979520_0006144000.wav",# None
                        # ref_audio_path: str = "/home/ubuntu/gpt-sovits/gpt-sovits/output/2025/02/12/f116a94fc997de347378f258eb366e32/ref_audios/科技人看世界1_9_1446720_1583360.wav",# None
                        ref_audio_path: str = "./output/2025/02/19/c819eb70a3aa0d8b79bcb11beb3752bb/ref_audios/Lula_voice_45_7794560_7971840.wav",# None
                        prompt_lang: str = "zh", # None
                        # prompt_text: str = "或是小溜球旅游的时候,不妨深入体会在地的文化热情", # ""
                        # prompt_text: str = "才有那么多的时间去做诗、演戏、运动、算数学", # ""
                        prompt_text: str = "到底在哪裡?帥哥在哪裡?我沒看到這個是真實的這是真實", # ""
                        top_k:int = 5,
                        top_p:float = 1,
                        temperature:float = 1,
                        text_split_method:str = "cut1",
                        batch_size:int = 20,
                        batch_threshold:float = 0.75,
                        split_bucket:bool = True,
                        speed_factor:float = 0.9,
                        fragment_interval:float = 0.3,
                        # seed:int = -1,
                        seed:int = 1413775942,
                        # seed:int = 3510518998,
                        
                        media_type:str = "wav",
                        streaming_mode:bool = True,
                        parallel_infer:bool = True,
                        repetition_penalty:float = 1.8
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
        "batch_size":int(batch_size),
        "batch_threshold":float(batch_threshold),
        "speed_factor":float(speed_factor),
        "split_bucket":split_bucket,
        "fragment_interval":fragment_interval,
        "seed":seed,
        "media_type":media_type,
        "streaming_mode":streaming_mode,
        "parallel_infer":parallel_infer,
        "repetition_penalty":float(repetition_penalty)
    }
    return await tts_handle(req)

@APP.get("/test")
async def tts_get_endpoint(
                        text: str = None,
                        text_lang: str = "auto", # None
                        # ref_audio_path: str = "logs/sweet/5-wav32k/sweet.wav_0005979520_0006144000.wav",# None
                        ref_audio_path: str = "./output/2024/09/18/4c22dbd4462eb8623bce49674dc0b684/ref_audios/sweet_11_2300160_2405120.wav",# None
                        prompt_lang: str = "zh", # None
                        # prompt_text: str = "或是小溜球旅游的时候,不妨深入体会在地的文化热情", # ""
                        prompt_text: str = "不妨深入體會在地的文化熱情", # ""
                        top_k:int = 5,
                        top_p:float = 1,
                        temperature:float = 1,
                        text_split_method:str = "cut1",
                        batch_size:int = 20,
                        batch_threshold:float = 0.75,
                        split_bucket:bool = True,
                        speed_factor:float = 1.0,
                        fragment_interval:float = 0.3,
                        # seed:int = -1,
                        seed:int = 717357706,
                        media_type:str = "wav",
                        streaming_mode:bool = True,
                        parallel_infer:bool = True,
                        repetition_penalty:float = 1.8
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
        "batch_size":int(batch_size),
        "batch_threshold":float(batch_threshold),
        "speed_factor":float(speed_factor),
        "split_bucket":split_bucket,
        "fragment_interval":fragment_interval,
        "seed":seed,
        "media_type":media_type,
        "streaming_mode":streaming_mode,
        "parallel_infer":parallel_infer,
        "repetition_penalty":float(repetition_penalty)
    }
    return await tts_handle1(req)
                
@APP.get("/clone")
async def tts_get_endpoint(
                        text: str = None,
                        text_lang: str = "auto", # None
                        # ref_audio_path: str = "logs/sweet/5-wav32k/sweet.wav_0005979520_0006144000.wav",# None
                        # ref_audio_path: str = "/home/ubuntu/gpt-sovits/gpt-sovits/output/2025/02/12/f116a94fc997de347378f258eb366e32/ref_audios/科技人看世界1_9_1446720_1583360.wav",# None
                        # ref_audio_path: str = "./output/2025/03/11/0bdadf87d368634eac3713ee7c45f0b8/ref_audios/teacher_46_6979520_7114240.wav",# None
                        ref_audio_path: str = "teacher.wav",# None
                        prompt_lang: str = "zh", # None
                        # prompt_text: str = "或是小溜球旅游的时候,不妨深入体会在地的文化热情", # ""
                        # prompt_text: str = "才有那么多的时间去做诗、演戏、运动、算数学", # ""
                        prompt_text: str = "這個 你可以發現所有論說 佛教的論說 沒有一個人能把佛性講這麼清楚的", # ""
                        top_k:int = 5,
                        top_p:float = 1,
                        temperature:float = 1,
                        text_split_method:str = "cut1",
                        batch_size:int = 20,
                        batch_threshold:float = 0.75,
                        split_bucket:bool = True,
                        speed_factor:float = 1,
                        fragment_interval:float = 0.3,
                        seed:int = -1,
                        # seed:int = 902450350,
                        # seed:int = 2133708477,
                        
                        media_type:str = "wav",
                        streaming_mode:bool = True,
                        parallel_infer:bool = True,
                        repetition_penalty:float = 1.8
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
        "batch_size":int(batch_size),
        "batch_threshold":float(batch_threshold),
        "speed_factor":float(speed_factor),
        "split_bucket":split_bucket,
        "fragment_interval":fragment_interval,
        "seed":seed,
        "media_type":media_type,
        "streaming_mode":streaming_mode,
        "parallel_infer":parallel_infer,
        "repetition_penalty":float(repetition_penalty)
    }
    return await tts_handle2(req)

@APP.post("/")
async def tts_post_endpoint(request: TTS_Request):
    req = request.dict()
    return await tts_handle(req)

@APP.post("/test")
async def tts_post_endpoint(request: TTS_Request):
    req = request.dict()
    return await tts_handle1(req)

@APP.post("/clone")
async def tts_post_endpoint(request: TTS_Request):
    req = request.dict()
    return await tts_handle2(req)

@APP.get("/set_refer_audio")
async def set_refer_aduio(refer_audio_path: str = None):
    try:
        tts_pipeline.set_ref_audio(refer_audio_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"set refer audio failed", "Exception": str(e)})
    return JSONResponse(status_code=200, content={"message": "success"})

import torch
# 檢查可用的 GPU 設備


model_path = "tools/asr/models/faster-whisper-large-v3-turbo"
device = "cuda"  # 或 "cuda" 如果你有 GPU
model = WhisperModel(model_path, device=device, compute_type="float32")

@APP.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    input_text = ""
    print("開始辨識")
    audio_file = await audio.read()
    audio_file = BytesIO(audio_file)  # 將音頻文件轉換為 BytesIO 對象
    segments, info = model.transcribe(
        audio=audio_file,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=700),
        language="zh",
    )
    for segment in segments:
        input_text += segment.text

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
    decoded_message = message
    return_json = {
        "input_text": converter.convert(input_text),
        "text": converter.convert(decoded_message)
    }
    return JSONResponse(return_json)
# @APP.post("/set_refer_audio")
# async def set_refer_aduio_post(audio_file: UploadFile = File(...)):
#     try:
#         # 检查文件类型，确保是音频文件
#         if not audio_file.content_type.startswith("audio/"):
#             return JSONResponse(status_code=400, content={"message": "file type is not supported"})
        
#         os.makedirs("uploaded_audio", exist_ok=True)
#         save_path = os.path.join("uploaded_audio", audio_file.filename)
#         # 保存音频文件到服务器上的一个目录
#         with open(save_path , "wb") as buffer:
#             buffer.write(await audio_file.read())
            
#         tts_pipeline.set_ref_audio(save_path)
#     except Exception as e:
#         return JSONResponse(status_code=400, content={"message": f"set refer audio failed", "Exception": str(e)})
#     return JSONResponse(status_code=200, content={"message": "success"})

@APP.get("/set_gpt_weights")
async def set_gpt_weights(weights_path: str = None):
    try:
        if weights_path in ["", None]:
            return JSONResponse(status_code=400, content={"message": "gpt weight path is required"})
        tts_pipeline.init_t2s_weights(weights_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"change gpt weight failed", "Exception": str(e)})

    return JSONResponse(status_code=200, content={"message": "success"})


@APP.get("/set_sovits_weights")
async def set_sovits_weights(weights_path: str = None):
    try:
        if weights_path in ["", None]:
            return JSONResponse(status_code=400, content={"message": "sovits weight path is required"})
        tts_pipeline.init_vits_weights(weights_path)
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"change sovits weight failed", "Exception": str(e)})
    return JSONResponse(status_code=200, content={"message": "success"})


@APP.get("/speakers")
def speakers_endpoint():
    return JSONResponse([{"name":"default","vid":1}], status_code=200)


@APP.get("/speakers_list")
def speakerlist_endpoint():
    return JSONResponse(["female_calm","female","male"], status_code=200)


@APP.post("/tts_to_audio/")
async def tts_to_audio(request: TTS_Request):
    req = request.dict()
    # "text": "",                   # str.(required) text to be synthesized
    # "text_lang": "",              # str.(required) language of the text to be synthesized
    # "ref_audio_path": "",         # str.(required) reference audio path.
    # "prompt_text": "",            # str.(optional) prompt text for the reference audio
    # "prompt_lang": "", 
    req["text_lang"] = global_config.llama_lang
    req["ref_audio_path"] = global_config.llama_audio
    req["prompt_text"] = global_config.llama_text
    req["prompt_lang"] = global_config.llama_prompt_lang
    req["batch_size"] = 10
    return await tts_handle(req)



if __name__ == "__main__":
    try:
        uvicorn.run(app="api_v2:APP", host=host, port=port, workers=1, log_level="info")
    except Exception as e:
        traceback.print_exc()
        os.kill(os.getpid(), signal.SIGTERM)
        exit(0)
logging.getLogger("multipart").setLevel(logging.WARNING)
logging.getLogger("faster_whisper").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)