import signal
import os
import sys
import traceback
from typing import Generator, Dict, List,Union
import logging
import time
import json
import threading
import yaml
import torch
import asyncio
from datetime import datetime
now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))
sys.path.append("%s/tools" % (now_dir))
sys.path.append("%s/tools/uvr5" % (now_dir))
paths_to_remove = ['/workspace/gpt-sovits', '/workspace/gpt-sovits/tools', '/workspace/gpt-sovits/GPT_SoVITS', '/workspace/gpt-sovits/tools/uvr5']
sys.path = [p for p in sys.path if p not in paths_to_remove]
print(sys.path)
from starlette.middleware.cors import CORSMiddleware
from sqlalchemy import Column, String
from sqlalchemy.orm import Session
from database import Base, get_db
from fastapi import FastAPI, Request, Response, Depends,UploadFile, File, Form
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
from pydantic import BaseModel
import requests
from faster_whisper import WhisperModel
nltk.download('averaged_perceptron_tagger_eng')

# 基礎設置
origins = ["*"]
i18n = I18nAuto()
cut_method_names = get_cut_method_names()
converter = opencc.OpenCC('s2t')

# 數據庫模型
class Model(Base):
    __tablename__ = "models"
    
    model_id = Column(String(200), primary_key=True)
    client_id = Column(String(200))
    output_dir = Column(String(5000))
    ref_audio = Column(String(5000))
    ref_audio_path = Column(String(5000))
    status = Column(String(200))
    gpt_model = Column(String(5000))
    sovits_model = Column(String(5000))
    upload_file = Column(String(5000))
    voice_name = Column(String(200))
    end_time = Column(String(200))

# 請求模型
class TTS_Request(BaseModel):
    model_id: str = None  # 改為可選字段
    text: str = None
    text_lang: str = "auto"
    ref_audio_path: str = None
    prompt_lang: str = "zh"
    prompt_text: str = ""
    top_k: int = 5
    top_p: float = 1
    temperature: float = 1
    text_split_method: str = "cut1"
    batch_size: int = 20
    batch_threshold: float = 0.75
    split_bucket: bool = True
    speed_factor: float = 1.0
    fragment_interval: float = 0.3
    seed: int = -1
    media_type: str = "wav"
    streaming_mode: bool = True
    parallel_infer: bool = True
    repetition_penalty: float = 1.8

# 初始化參數解析
parser = argparse.ArgumentParser(description="GPT-SoVITS api")
parser.add_argument("-c", "--tts_config", type=str, default="../gpt-sovits/GPT_SoVITS/configs/tts_infer.yaml", help="tts_infer路径")
parser.add_argument("-a", "--bind_addr", type=str, default="0.0.0.0", help="default: 127.0.0.1")
parser.add_argument("-p", "--port", type=int, default="9885", help="default: 9880")
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


# 聲音配置類
class VoiceConfig:
    def __init__(self, model_data: Model):
        self.model_id = model_data.model_id
        # 將所有相對路徑轉換為指向 gpt-sovits 目錄
        self.output_dir = os.path.join("../gpt-sovits", model_data.output_dir.lstrip('./'))
        self.ref_audio = model_data.ref_audio
        self.ref_audio_path = os.path.join("../gpt-sovits", model_data.ref_audio_path.lstrip('./'), self.ref_audio)
        
        # 讀取 prompt_text 時也要修改路徑
        ref_audio_name = model_data.ref_audio.split("/")[-1].split(".")[0]
        self.prompt_text_path = os.path.join(self.output_dir, "input_audio_text", f"{ref_audio_name}.txt")
        try:
            with open(self.prompt_text_path, "r") as f:
                self.prompt_text = f.read().strip()
        except:
            self.prompt_text = ""
        self.voice_name = model_data.voice_name
        # 直接傳入配置字典，不需要外層的custom
        config_dict = {
            "custom": {
                "bert_base_path": "../gpt-sovits/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
                "cnhuhbert_base_path": "../gpt-sovits/GPT_SoVITS/pretrained_models/chinese-hubert-base",
                "device": "cuda",
                "is_half": True,
                "t2s_weights_path": os.path.join("../gpt-sovits", model_data.gpt_model.lstrip('./')),
                "vits_weights_path": os.path.join("../gpt-sovits", model_data.sovits_model.lstrip('./'))
            }
        }
        
        # print(f"model_data.gpt_model: {os.path.join('../gpt-sovits', model_data.gpt_model.lstrip('./'))}")
        # print(f"model_data.sovits_model: {os.path.join('../gpt-sovits', model_data.sovits_model.lstrip('./'))}")
        
        if not os.path.exists(os.path.join("../gpt-sovits", model_data.gpt_model.lstrip('./'))):
            raise FileNotFoundError(f"GPT model not found: {os.path.join('../gpt-sovits', model_data.gpt_model.lstrip('./'))}")
        if not os.path.exists(os.path.join("../gpt-sovits", model_data.sovits_model.lstrip('./'))):
            raise FileNotFoundError(f"SoVITS model not found: {os.path.join('../gpt-sovits', model_data.sovits_model.lstrip('./'))}")

        # 初始化TTS Pipeline
        try:
            # 使用 no_save_file=True 來防止修改配置文件
            self.tts_pipeline = TTS(TTS_Config.from_dict(config_dict, no_save_file=True))
        except Exception as e:
            logging.error(f"Failed to initialize TTS pipeline: {e}")
            raise e
        
        # # 在記憶體中保存配置
        # self.config = {
        #     "custom": {
        #         "bert_base_path": "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large",
        #         "cnhuhbert_base_path": "GPT_SoVITS/pretrained_models/chinese-hubert-base",
        #         "device": "cuda",
        #         "is_half": True,
        #         "t2s_weights_path": model_data.gpt_model,
        #         "vits_weights_path": model_data.sovits_model
        #     }
        # }
        # # 創建專屬的 pipeline
        # self.tts_pipeline = TTS(TTS_Config.from_dict(self.config))
    def __del__(self):
        try:
            if hasattr(self, 'tts_pipeline'):
                if hasattr(self.tts_pipeline, 't2s_model'):
                    self.tts_pipeline.t2s_model = None
                if hasattr(self.tts_pipeline, 'vits_model'):
                    self.tts_pipeline.vits_model = None
                if hasattr(self.tts_pipeline, 'bert_model'):
                    self.tts_pipeline.bert_model = None
                if hasattr(self.tts_pipeline, 'cnhuhbert_model'):
                    self.tts_pipeline.cnhuhbert_model = None
                self.tts_pipeline = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        except:
            pass

# 在 voice_manager 變量定義之前
voice_operation_lock = asyncio.Lock()

class VoiceManager:
    def __init__(self, max_voices=3):
        self.max_voices = max_voices
        self.active_voices: Dict[str, VoiceConfig] = {}
        self.load_times: Dict[str, float] = {}
        # 移除 threading.Lock()

    def load_voice(self, model_data: Model) -> Union[bool, JSONResponse]:
        # 檢查是否已經載入
        if model_data.model_id in self.active_voices:
            return JSONResponse(
                status_code=200,
                content={"message": "此聲音模型已經載入"}
            )

        # 檢查並處理最大數量限制
        if len(self.active_voices) >= self.max_voices:
            try:
                oldest_model_id = min(self.load_times.items(), key=lambda x: x[1])[0]
                self.unload_voice(oldest_model_id)
            except Exception as e:
                return JSONResponse(
                    status_code=500,
                    content={"message": f"卸載舊模型時發生錯誤: {str(e)}"}
                )

        # 載入新聲音
        try:
            voice_config = VoiceConfig(model_data)
            self.active_voices[model_data.model_id] = voice_config
            self.load_times[model_data.model_id] = time.time()
            return True
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"message": f"載入模型失敗: {str(e)}"}
            )
    def unload_voice(self, model_id: str) -> bool:
        if model_id in self.active_voices:
            try:
                voice_config = self.active_voices[model_id]
                # 執行配置的清理
                if hasattr(voice_config, 'tts_pipeline'):
                    if hasattr(voice_config.tts_pipeline, 't2s_model'):
                        voice_config.tts_pipeline.t2s_model = None
                    if hasattr(voice_config.tts_pipeline, 'vits_model'):
                        voice_config.tts_pipeline.vits_model = None
                    if hasattr(voice_config.tts_pipeline, 'bert_model'):
                        voice_config.tts_pipeline.bert_model = None
                    if hasattr(voice_config.tts_pipeline, 'cnhuhbert_model'):
                        voice_config.tts_pipeline.cnhuhbert_model = None
                    voice_config.tts_pipeline = None
                # 移除引用
                del self.active_voices[model_id]
                del self.load_times[model_id]
                # 強制垃圾回收
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                return True
            except Exception as e:
                print(f"Error during voice unload: {e}")
        return False

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

# 初始化聲音管理器
voice_manager = VoiceManager(max_voices=3)



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

# API 端點
@APP.get("/voices")
def list_all_voices(db: Session = Depends(get_db)):
    models = db.query(Model).filter(Model.status == "done").all()
    return [{
        "model_id": m.model_id,
        "filename": m.upload_file.split("/")[-1],
        "is_loaded": m.model_id in voice_manager.active_voices,
        "voice_name": m.voice_name,
        "end_time": m.end_time,
        "sample_voice_path":m.output_dir+"/sample_voice_dir/test_voice.wav",
        "ref_audio_path": m.ref_audio_path
    } for m in models]

@APP.post("/load_voice/{model_id}")
async def load_voice(model_id: str, db: Session = Depends(get_db)):
    async with voice_operation_lock:
        try:
            model = db.query(Model).filter(Model.model_id == model_id).first()
            if not model:
                return JSONResponse(
                    status_code=404,
                    content={"message": "找不到指定的聲音模型"}
                )
            
            result = voice_manager.load_voice(model)
            if isinstance(result, JSONResponse):
                return result
            
            return JSONResponse(
                status_code=200,
                content={"message": "聲音模型載入成功", "voice_name": model.voice_name}
            )
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"message": f"載入聲音時發生錯誤: {str(e)}"}
            )

@APP.post("/unload_voice/{model_id}")
async def unload_voice(model_id: str):
    async with voice_operation_lock:
        try:
            if voice_manager.unload_voice(model_id):
                return JSONResponse(
                    status_code=200,
                    content={"message": "聲音已成功移除"}
                )
            return JSONResponse(
                status_code=404,
                content={"message": "找不到指定的聲音"}
            )
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"message": f"移除聲音時發生錯誤: {str(e)}"}
            )

@APP.get("/active_voices")
async def list_active_voices():
    """列出當前已載入的聲音"""
    return [{
        "model_id": vid,
        "ref_audio_path": config.ref_audio_path,
        "prompt_text": config.prompt_text,
        "voice_name" : config.voice_name
    } for vid, config in voice_manager.active_voices.items()]

async def handle_tts(req: dict, voice_config):
    streaming_mode = req.get("streaming_mode", False)
    media_type = req.get("media_type", "wav")

    check_res = check_params(req)
    if check_res is not None:
        return check_res

    if streaming_mode:
        req["return_fragment"] = True
    
    try:
        tts_generator = voice_config.tts_pipeline.run(req)
        
        
        if streaming_mode:
            def streaming_generator(tts_generator: Generator, media_type: str):
                start_time = time.time()
                total_chunks = 0
                text = req.get("text", "")
                text_length = len(text)
                total_audio_samples = 0
                
                if media_type == "wav":
                    yield wave_header_chunk()
                    media_type = "raw"
                    
                for sr, chunk in tts_generator:
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
                    
                    yield pack_audio(BytesIO(), chunk, sr, media_type).getvalue()
                print(f"\n處理完成！")
                print(f"最終統計：")
                print(f"- 總字數：{text_length}字")
                print(f"- 音頻長度：{audio_duration:.2f}秒")
                print(f"- 處理耗時：{elapsed_time:.2f}秒")
                print(f"- 當前生成速度={chars_per_process_second:.2f}字/秒")
                print(f"- 音頻每秒字數={chars_per_audio_second:.2f}字/秒")
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

@APP.get("/clone")
async def tts_get_endpoint(
    model_id: str = None,  # 改為可選參數
    text: str = None,
    text_lang: str = "auto",
    ref_audio_path: str = None,
    prompt_text: str = None,
    prompt_lang: str = "zh",
    top_k: int = 5,
    top_p: float = 1,
    temperature: float = 1,
    text_split_method: str = "cut1",
    batch_size: int = 20,
    batch_threshold: float = 0.75,
    split_bucket: bool = True,
    speed_factor: float = 1,
    fragment_interval: float = 0.3,
    media_type: str = "wav",
    streaming_mode: bool = True,
    parallel_infer: bool = True,
    repetition_penalty: float = 1.8
):
    async with voice_operation_lock:
        try:
            # 如果沒有指定 model_id 且有已載入的聲音，使用最新的
            if model_id is None:
                if not voice_manager.load_times:
                    return JSONResponse(
                        status_code=400,
                        content={"message": "沒有可用的聲音模型"}
                    )
                model_id = max(voice_manager.load_times.items(), key=lambda x: x[1])[0]
                
            voice_config = voice_manager.active_voices.get(model_id)
            if not voice_config:
                return JSONResponse(
                    status_code=400,
                    content={"message": "請先載入此聲音模型"}
                )
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"message": f"無法找到此聲音模型", "Exception": str(e)}
            )
        
        ref_audio_path = voice_config.ref_audio_path

        prompt_text = voice_config.prompt_text
        
        req = {
            "model_id": model_id,  # 添加 model_id 到請求中
            "text": text,
            "text_lang": text_lang.lower(),
            "ref_audio_path": ref_audio_path,
            "prompt_text": prompt_text,
            "prompt_lang": prompt_lang.lower(),
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "text_split_method": text_split_method,
            "batch_size": batch_size,
            "batch_threshold": batch_threshold,
            "speed_factor": speed_factor,
            "split_bucket": split_bucket,
            "fragment_interval": fragment_interval,
            "media_type": media_type,
            "streaming_mode": streaming_mode,
            "parallel_infer": parallel_infer,
            "repetition_penalty": repetition_penalty
        }
        
        return await handle_tts(req, voice_config)

@APP.post("/clone")
async def tts_post_endpoint(request: TTS_Request= None):
    async with voice_operation_lock:
        try:
            req = request.dict()
            # 如果沒有指定 model_id，使用最新的聲音
            if req["model_id"] is None:
                if not voice_manager.load_times:
                    return JSONResponse(
                        status_code=400,
                        content={"message": "沒有可用的聲音模型"}
                    )
                req["model_id"] = max(voice_manager.load_times.items(), key=lambda x: x[1])[0]
                
            voice_config = voice_manager.active_voices.get(req["model_id"])
            if not voice_config:
                return JSONResponse(
                    status_code=400,
                    content={"message": "請先載入此聲音模型"}
                )
        
            
            if not req.get("ref_audio_path"):
                req["ref_audio_path"] = voice_config.ref_audio_path
            if not req.get("prompt_text"):
                req["prompt_text"] = voice_config.prompt_text
            
            return await handle_tts(req, voice_config)
        except Exception as e:
            return JSONResponse(
                status_code=400,
                content={"message": f"無法找到此聲音模型", "Exception": str(e)}
            )


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
                    yield wave_header_chunk()
                    media_type = "raw"
                    
                for sr, chunk in tts_generator:
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
                    
                    yield pack_audio(BytesIO(), chunk, sr, media_type).getvalue()
                print(f"\n處理完成！")
                print(f"最終統計：")
                print(f"- 總字數：{text_length}字")
                print(f"- 音頻長度：{audio_duration:.2f}秒")
                print(f"- 處理耗時：{elapsed_time:.2f}秒")
                print(f"- 當前生成速度={chars_per_process_second:.2f}字/秒")
                print(f"- 音頻每秒字數={chars_per_audio_second:.2f}字/秒")
            
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
                    yield wave_header_chunk()
                    media_type = "raw"
                    
                for sr, chunk in tts_generator:
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
                    
                    yield pack_audio(BytesIO(), chunk, sr, media_type).getvalue()
                print(f"\n處理完成！")
                print(f"最終統計：")
                print(f"- 總字數：{text_length}字")
                print(f"- 音頻長度：{audio_duration:.2f}秒")
                print(f"- 處理耗時：{elapsed_time:.2f}秒")
                print(f"- 當前生成速度={chars_per_process_second:.2f}字/秒")
                print(f"- 音頻每秒字數={chars_per_audio_second:.2f}字/秒")
            
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
    ref_audio_path: str = "../gpt-sovits/output/2025/02/19/c819eb70a3aa0d8b79bcb11beb3752bb/ref_audios/Lula_voice_45_7794560_7971840.wav",
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
    ref_audio_path: str = "../gpt-sovits/output/2024/09/18/4c22dbd4462eb8623bce49674dc0b684/ref_audios/sweet_11_2300160_2405120.wav",
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
