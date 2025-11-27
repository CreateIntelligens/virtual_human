###############################################################################
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  email: lipku@foxmail.com
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################
import os
import time
import numpy as np
import soundfile as sf
import resampy
import asyncio
import edge_tts
import json
import opencc

from typing import Iterator

import requests

import queue
from queue import Queue
from io import BytesIO
from threading import Thread, Event
from enum import Enum
from datetime import datetime
import re

import wave

# 添加 MP3 解碼支援
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
    print(f'[{datetime.now()}] pydub available for MP3 decoding')
except ImportError:
    PYDUB_AVAILABLE = False
    print(f'[{datetime.now()}] pydub not available, MP3 decoding will be limited')

# 添加 WebSocket 支援
try:
    import websockets
    import ssl
    WEBSOCKETS_AVAILABLE = True
    print(f'[{datetime.now()}] websockets available for WebSocket TTS')
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    print(f'[{datetime.now()}] websockets not available, WebSocket TTS will be disabled')

# 添加調試音檔儲存支援
import shutil
from pathlib import Path

# 載入所有替換規則
with open(os.path.join(os.path.dirname(__file__), 'replacements.json'), 'r', encoding='utf-8') as f:
    REPLACE_RULES = json.load(f)


class State(Enum):
    RUNNING=0
    PAUSE=1

class BaseTTS:
    def __init__(self, opt, parent):
        self.opt=opt
        self.parent = parent

        self.fps = opt.fps # 20 ms per frame
        self.sample_rate = 16000
        self.chunk = self.sample_rate // self.fps # 320 samples per chunk (20ms * 16000 / 1000)
        self.input_stream = BytesIO()

        self.msgqueue = Queue()
        self.state = State.RUNNING

    def flush_talk(self):
        self.msgqueue.queue.clear()
        self.state = State.PAUSE

    def put_msg_txt(self,msg,eventpoint=None): 
        if len(msg)>0:
            self.msgqueue.put((msg,eventpoint))

    def render(self,quit_event):
        process_thread = Thread(target=self.process_tts, args=(quit_event,))
        process_thread.start()
    
    def process_tts(self,quit_event):        
        while not quit_event.is_set():
            try:
                msg = self.msgqueue.get(block=True, timeout=1)
                self.state=State.RUNNING
            except queue.Empty:
                continue
            self.txt_to_audio(msg)
        print(f'[{datetime.now()}] ttsreal thread stop')
    
    def txt_to_audio(self,msg):
        pass
    

###########################################################################################
class EdgeTTS(BaseTTS):
    def txt_to_audio(self,msg):
        voicename = "zh-CN-YunxiaNeural"
        text,textevent = msg
        t = time.time()
        asyncio.new_event_loop().run_until_complete(self.__main(voicename,text))
        print(f'[{datetime.now()}] -------edge tts time:{time.time()-t:.4f}s')
        if self.input_stream.getbuffer().nbytes<=0: #edgetts err
            print(f'[{datetime.now()}] edgetts err!!!!!')
            return
        
        self.input_stream.seek(0)
        stream = self.__create_bytes_stream(self.input_stream)
        streamlen = stream.shape[0]
        idx=0
        while streamlen >= self.chunk and self.state==State.RUNNING:
            eventpoint=None
            streamlen -= self.chunk
            if idx==0:
                eventpoint={'status':'start','text':text,'msgenvent':textevent}
            elif streamlen<self.chunk:
                eventpoint={'status':'end','text':text,'msgenvent':textevent}
            self.parent.put_audio_frame(stream[idx:idx+self.chunk],eventpoint)
            idx += self.chunk
        #if streamlen>0:  #skip last frame(not 20ms)
        #    self.queue.put(stream[idx:])
        self.input_stream.seek(0)
        self.input_stream.truncate() 

    def __create_bytes_stream(self,byte_stream):
        #byte_stream=BytesIO(buffer)
        stream, sample_rate = sf.read(byte_stream) # [T*sample_rate,] float64
        print(f'[INFO] [{datetime.now()}] tts audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            print(f'[WARN] [{datetime.now()}] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]
    
        if sample_rate != self.sample_rate and stream.shape[0]>0:
            print(f'[WARN] [{datetime.now()}] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream
    
    async def __main(self,voicename: str, text: str):
        try:
            communicate = edge_tts.Communicate(text, voicename)

            #with open(OUTPUT_FILE, "wb") as file:
            first = True
            async for chunk in communicate.stream():
                if first:
                    first = False
                if chunk["type"] == "audio" and self.state==State.RUNNING:
                    #self.push_audio(chunk["data"])
                    self.input_stream.write(chunk["data"])
                    #file.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    pass
        except Exception as e:
            print(f'[{datetime.now()}] {str(e)}')

###########################################################################################
class FishTTS(BaseTTS):
    def txt_to_audio(self,msg): 
        text,textevent = msg
        self.stream_tts(
            self.fish_speech(
                text,
                self.opt.REF_FILE,  
                self.opt.REF_TEXT,
                "zh", #en args.language,
                self.opt.TTS_SERVER, #"http://127.0.0.1:5000", #args.server_url,
            ),
            msg
        )

    def fish_speech(self, text, reffile, reftext,language, server_url) -> Iterator[bytes]:
        start = time.perf_counter()
        req={
            'text':text,
            'reference_id':reffile,
            'format':'wav',
            'streaming':True,
            'use_memory_cache':'on'
        }
        try:
            res = requests.post(
                f"{server_url}/v1/tts",
                json=req,
                stream=True,
                headers={
                    "content-type": "application/json",
                },
            )
            end = time.perf_counter()
            print(f'[{datetime.now()}] fish_speech Time to make POST: {end-start}s')

            if res.status_code != 200:
                print(f'[{datetime.now()}] Error:', res.text)
                return
                
            first = True
        
            for chunk in res.iter_content(chunk_size=17640): # 1764 44100*20ms*2
                #print('chunk len:',len(chunk))
                if first:
                    end = time.perf_counter()
                    print(f'[{datetime.now()}] fish_speech Time to first chunk: {end-start}s')
                    first = False
                if chunk and self.state==State.RUNNING:
                    yield chunk
            #print("gpt_sovits response.elapsed:", res.elapsed)
        except Exception as e:
            print(f'[{datetime.now()}] {str(e)}')

    def stream_tts(self,audio_stream,msg):
        text,textevent = msg
        first = True
        for chunk in audio_stream:
            if chunk is not None and len(chunk)>0:          
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = resampy.resample(x=stream, sr_orig=44100, sr_new=self.sample_rate)
                #byte_stream=BytesIO(buffer)
                #stream = self.__create_bytes_stream(byte_stream)
                streamlen = stream.shape[0]
                idx=0
                while streamlen >= self.chunk:
                    eventpoint=None
                    if first:
                        eventpoint={'status':'start','text':text,'msgenvent':textevent}
                        first = False
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk],eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk
        eventpoint={'status':'end','text':text,'msgenvent':textevent}
        self.parent.put_audio_frame(np.zeros(self.chunk,np.float32),eventpoint) 

###########################################################################################
class VoitsTTS(BaseTTS):
    def __init__(self, opt, parent):
        super().__init__(opt, parent)
        self.default_req = {
            "text": "text",
            "text_lang": "auto",
            "ref_audio_path":"./output/2025/02/19/c819eb70a3aa0d8b79bcb11beb3752bb/ref_audios/Lula_voice_45_7794560_7971840.wav",
            "prompt_text":"到底在哪裡?帥哥在哪裡?我沒看到這個是真實的這是真實",
            "prompt_lang": "zh",
            "top_k": 5,
            "top_p": 1,
            "temperature": 1,
            "text_split_method": "cut1",
            "batch_size":20,
            "batch_threshold":0.75,
            "speed_factor":0.9,
            "split_bucket":True,
            "fragment_interval":0.3,
            "seed":717357706,
            "media_type":"wav",
            "streaming_mode":True,
            "parallel_infer":True,
            "repetition_penalty":1.35
        }

    def txt_to_audio(self,msg): 
        text,textevent = msg
        self.stream_tts(
            self.gpt_sovits(
                text,
                self.opt.REF_FILE,  
                self.opt.REF_TEXT,
                "zh", #en args.language,
                self.opt.TTS_SERVER, #"http://127.0.0.1:5000", #args.server_url,
            ),
            msg
        )

    def gpt_sovits(self, text, reffile, reftext,language, server_url) -> Iterator[bytes]:
        start = time.perf_counter()
        req = self.default_req.copy()
        # 根據 replacements.json 來做一系列替換
        for rule in REPLACE_RULES:
            flag_val = 0
            for f in rule.get('flags', []):
                flag_val |= getattr(re, f)
            text = re.sub(rule['pattern'], rule['replacement'], text, flags=flag_val)

        print(f'[{datetime.now()}] gpt_sovits text: {text}')

        req["text"] = text  # 更新 text 參數
        
        try:
            print(f'[{datetime.now()}] Making request to: {server_url}')
            print(f'[{datetime.now()}] Request content: {json.dumps(req, indent=2)}')
            
            res = requests.post(
                f"{server_url}",
                json=req,
                stream=True,
            )
            end = time.perf_counter()
            print(f'[{datetime.now()}] gpt_sovits Time to make POST: {end-start}s')

            print(f'[{datetime.now()}] Response status: {res.status_code}')
            print(f'[{datetime.now()}] Response headers: {dict(res.headers)}')

            if res.status_code != 200:
                print(f'[{datetime.now()}] Error:', res.text)
                return
                
            first = True
        
            for chunk in res.iter_content(chunk_size=None): #12800 1280 32K*20ms*2
                print(f'[{datetime.now()}] chunk len:',len(chunk))
                if first:
                    end = time.perf_counter()
                    print(f'[{datetime.now()}] gpt_sovits Time to first chunk: {end-start}s')
                    first = False
                if chunk and self.state==State.RUNNING:
                    yield chunk
            #print("gpt_sovits response.elapsed:", res.elapsed)
        except Exception as e:
            print(f'[{datetime.now()}] Exception occurred: {str(e)}')

    def __create_bytes_stream(self,byte_stream):
        #byte_stream=BytesIO(buffer)
        stream, sample_rate = sf.read(byte_stream) # [T*sample_rate,] float64
        print(f'[INFO] [{datetime.now()}] tts audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            print(f'[WARN] [{datetime.now()}] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]
    
        if sample_rate != self.sample_rate and stream.shape[0]>0:
            print(f'[WARN] [{datetime.now()}] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream

    def stream_tts(self,audio_stream,msg):
        text,textevent = msg
        first = True
        for chunk in audio_stream:
            if chunk is not None and len(chunk)>0:          
                #stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                #stream = resampy.resample(x=stream, sr_orig=32000, sr_new=self.sample_rate)
                byte_stream=BytesIO(chunk)
                stream = self.__create_bytes_stream(byte_stream)
                streamlen = stream.shape[0]
                idx=0
                while streamlen >= self.chunk:
                    eventpoint=None
                    if first:
                        eventpoint={'status':'start','text':text,'msgenvent':textevent}
                        first = False
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk],eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk
        eventpoint={'status':'end','text':text,'msgenvent':textevent}
        self.parent.put_audio_frame(np.zeros(self.chunk,np.float32),eventpoint)



###########################################################################################
class CosyVoiceTTS(BaseTTS):
    def txt_to_audio(self,msg):
        text,textevent = msg 
        self.stream_tts(
            self.cosy_voice(
                text,
                self.opt.REF_FILE,  
                self.opt.REF_TEXT,
                "zh", #en args.language,
                self.opt.TTS_SERVER, #"http://127.0.0.1:5000", #args.server_url,
            ),
            msg
        )

    def cosy_voice(self, text, reffile, reftext,language, server_url) -> Iterator[bytes]:
        start = time.perf_counter()
        payload = {
            'tts_text': text,
            'prompt_text': reftext
        }
        try:
            files = [('prompt_wav', ('prompt_wav', open(reffile, 'rb'), 'application/octet-stream'))]
            res = requests.request("GET", f"{server_url}/inference_zero_shot", data=payload, files=files, stream=True)
            
            end = time.perf_counter()
            print(f'[{datetime.now()}] cosy_voice Time to make POST: {end-start}s')

            if res.status_code != 200:
                print(f'[{datetime.now()}] Error:', res.text)
                return
                
            first = True
        
            for chunk in res.iter_content(chunk_size=8820): # 882 22.05K*20ms*2
                if first:
                    end = time.perf_counter()
                    print(f'[{datetime.now()}] cosy_voice Time to first chunk: {end-start}s')
                    first = False
                if chunk and self.state==State.RUNNING:
                    yield chunk
        except Exception as e:
            print(f'[{datetime.now()}] {str(e)}')

    def stream_tts(self,audio_stream,msg):
        text,textevent = msg
        first = True
        for chunk in audio_stream:
            if chunk is not None and len(chunk)>0:          
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = resampy.resample(x=stream, sr_orig=22050, sr_new=self.sample_rate)
                #byte_stream=BytesIO(buffer)
                #stream = self.__create_bytes_stream(byte_stream)
                streamlen = stream.shape[0]
                idx=0
                while streamlen >= self.chunk:
                    eventpoint=None
                    if first:
                        eventpoint={'status':'start','text':text,'msgenvent':textevent}
                        first = False
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk],eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk
        eventpoint={'status':'end','text':text,'msgenvent':textevent}
        self.parent.put_audio_frame(np.zeros(self.chunk,np.float32),eventpoint) 

###########################################################################################
class XTTS(BaseTTS):
    def __init__(self, opt, parent):
        super().__init__(opt,parent)
        self.speaker = self.get_speaker(opt.REF_FILE, opt.TTS_SERVER)

    def txt_to_audio(self,msg):
        text,textevent = msg  
        self.stream_tts(
            self.xtts(
                text,
                self.speaker,
                "zh-cn", #en args.language,
                self.opt.TTS_SERVER, #"http://localhost:9000", #args.server_url,
                "20" #args.stream_chunk_size
            ),
            msg
        )

    def get_speaker(self,ref_audio,server_url):
        files = {"wav_file": ("reference.wav", open(ref_audio, "rb"))}
        response = requests.post(f"{server_url}/clone_speaker", files=files)
        return response.json()

    def xtts(self,text, speaker, language, server_url, stream_chunk_size) -> Iterator[bytes]:
        start = time.perf_counter()
        speaker["text"] = text
        speaker["language"] = language
        speaker["stream_chunk_size"] = stream_chunk_size  # you can reduce it to get faster response, but degrade quality
        try:
            res = requests.post(
                f"{server_url}/tts_stream",
                json=speaker,
                stream=True,
            )
            end = time.perf_counter()
            print(f'[{datetime.now()}] xtts Time to make POST: {end-start}s')

            if res.status_code != 200:
                print(f'[{datetime.now()}] Error:', res.text)
                return

            first = True
        
            for chunk in res.iter_content(chunk_size=9600): #24K*20ms*2
                if first:
                    end = time.perf_counter()
                    print(f'[{datetime.now()}] xtts Time to first chunk: {end-start}s')
                    first = False
                if chunk:
                    yield chunk
        except Exception as e:
            print(f'[{datetime.now()}] {str(e)}')
    
    def stream_tts(self,audio_stream,msg):
        text,textevent = msg
        first = True
        for chunk in audio_stream:
            if chunk is not None and len(chunk)>0:          
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = resampy.resample(x=stream, sr_orig=24000, sr_new=self.sample_rate)
                #byte_stream=BytesIO(buffer)
                #stream = self.__create_bytes_stream(byte_stream)
                streamlen = stream.shape[0]
                idx=0
                while streamlen >= self.chunk:
                    eventpoint=None
                    if first:
                        eventpoint={'status':'start','text':text,'msgenvent':textevent}
                        first = False
                    self.parent.put_audio_frame(stream[idx:idx+self.chunk],eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk
        eventpoint={'status':'end','text':text,'msgenvent':textevent}
        self.parent.put_audio_frame(np.zeros(self.chunk,np.float32),eventpoint)

###########################################################################################
class MiniMaxTTS(BaseTTS):
    def __init__(self, opt, parent):
        super().__init__(opt, parent)
        self.api_key = os.getenv("MINIMAX_API_KEY")
        self.group_id = os.getenv("MINIMAX_GROUP_ID")
        self.base_url = "https://api.minimaxi.chat/v1/t2a_v2"
        self.model = os.getenv("MINIMAX_MODEL", "speech-02-turbo")
        self.voice_id = os.getenv("MINIMAX_VOICE_ID", "moss_audio_9e3d9106-42a6-11f0-b6c4-9e15325fe584")
        
        # 音頻平滑處理
        self.overlap_size = 160  # 10ms overlap at 16kHz (16000 * 0.01)
        self.previous_tail = None
        self.audio_buffer = []
        
        # 緩衝模式設定
        self.buffered_mode = os.getenv("MINIMAX_BUFFERED_MODE", "false").lower() == "true"
        self.collected_chunks = []  # 用於緩衝模式收集音頻
        self.collected_audio = None  # 合併後的完整音頻
        
        # 調試音檔儲存功能
        self.debug_enabled = os.getenv("MINIMAX_DEBUG_AUDIO", "false").lower() == "true"
        self.debug_path = os.getenv("MINIMAX_DEBUG_PATH", "debug_audio")
        self.session_id = getattr(opt, 'sessionid', 0)
        self.chunk_counter = 0
        self.frame_counter = 0
        self.metadata = []
        
        if self.debug_enabled:
            self.setup_debug_directories()
            print(f'[{datetime.now()}] MiniMax debug mode enabled, saving to: {self.debug_path}')
        
        # 加載發音字典
        self.pronunciation_dict = self.load_pronunciation_dict()
        
        if not self.api_key or not self.group_id:
            print(f'[{datetime.now()}] Warning: MiniMax API credentials not found in environment variables')

    def setup_debug_directories(self):
        """設置調試目錄結構"""
        try:
            session_path = Path(self.debug_path) / f"session_{self.session_id}"
            
            # 創建所有必要的目錄
            (session_path / "raw_chunks").mkdir(parents=True, exist_ok=True)
            (session_path / "decoded_chunks").mkdir(parents=True, exist_ok=True)
            (session_path / "smoothed_chunks").mkdir(parents=True, exist_ok=True)
            (session_path / "final_frames").mkdir(parents=True, exist_ok=True)
            
            # 重置計數器和元數據
            self.chunk_counter = 0
            self.frame_counter = 0
            self.metadata = []
            
            print(f'[{datetime.now()}] Debug directories created at: {session_path}')
            
        except Exception as e:
            print(f'[{datetime.now()}] Failed to create debug directories: {e}')
            self.debug_enabled = False

    def save_debug_audio(self, audio_data, stage, chunk_id=None, frame_id=None, metadata=None):
        """儲存調試音檔"""
        if not self.debug_enabled:
            return
            
        try:
            session_path = Path(self.debug_path) / f"session_{self.session_id}"
            
            if stage == "raw_chunk":
                # 儲存原始 hex 解碼後的音頻
                filename = f"raw_chunk_{chunk_id:04d}.wav"
                filepath = session_path / "raw_chunks" / filename
                sf.write(filepath, audio_data, self.sample_rate)
                
            elif stage == "decoded_chunk":
                # 儲存格式解碼後的音頻
                filename = f"decoded_chunk_{chunk_id:04d}.wav"
                filepath = session_path / "decoded_chunks" / filename
                sf.write(filepath, audio_data, self.sample_rate)
                
            elif stage == "smoothed_chunk":
                # 儲存平滑處理後的音頻
                filename = f"smoothed_chunk_{chunk_id:04d}.wav"
                filepath = session_path / "smoothed_chunks" / filename
                sf.write(filepath, audio_data, self.sample_rate)
                
            elif stage == "complete_audio":
                # 儲存完整合併音頻（緩衝模式專用）
                filename = f"complete_audio_{chunk_id:04d}.wav"
                filepath = session_path / filename
                sf.write(filepath, audio_data, self.sample_rate)
                
            elif stage == "final_frame":
                # 儲存最終 20ms 幀
                filename = f"final_frame_{frame_id:06d}.wav"
                filepath = session_path / "final_frames" / filename
                sf.write(filepath, audio_data, self.sample_rate)
            
            # 記錄元數據
            if metadata:
                metadata['timestamp'] = datetime.now().isoformat()
                metadata['stage'] = stage
                metadata['filename'] = filename
                self.metadata.append(metadata)
                
        except Exception as e:
            print(f'[{datetime.now()}] Failed to save debug audio ({stage}): {e}')

    def save_debug_metadata(self):
        """儲存調試元數據"""
        if not self.debug_enabled or not self.metadata:
            return
            
        try:
            session_path = Path(self.debug_path) / f"session_{self.session_id}"
            metadata_file = session_path / "metadata.json"
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
                
            print(f'[{datetime.now()}] Debug metadata saved: {metadata_file}')
            
        except Exception as e:
            print(f'[{datetime.now()}] Failed to save debug metadata: {e}')

    def decode_mp3_optimized(self, audio_bytes):
        """優化的 MP3 解碼方法"""
        if not PYDUB_AVAILABLE:
            print(f'[{datetime.now()}] pydub not available, cannot decode MP3')
            return None, None
            
        try:
            start_time = time.perf_counter()
            
            # 使用 pydub 解碼 MP3，直接設定目標格式
            audio_segment = AudioSegment.from_file(
                BytesIO(audio_bytes), 
                format="mp3"
            ).set_frame_rate(self.sample_rate).set_channels(1)  # 直接轉換為 16kHz 單聲道
            
            # 快速轉換為 numpy array
            samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32) / 32767.0
            
            decode_time = time.perf_counter() - start_time
            print(f'[{datetime.now()}] MP3 decode time: {decode_time*1000:.2f}ms, shape: {samples.shape}')
            
            return samples, self.sample_rate
            
        except Exception as e:
            print(f'[{datetime.now()}] MP3 decode failed: {e}')
            return None, None

    def decode_audio_smart(self, hex_chunk, audio_bytes):
        """智能音頻解碼，根據格式選擇最佳方法"""
        # WAV 文件：最快，直接處理
        if hex_chunk.startswith('49443304'):
            try:
                audio_io = BytesIO(audio_bytes)
                stream, sr = sf.read(audio_io)
                print(f'[{datetime.now()}] Decoded as WAV, sr: {sr}, shape: {stream.shape}')
                return stream, sr
            except Exception as e:
                print(f'[{datetime.now()}] WAV decode failed: {e}')
                return None, None
        
        # MP3 格式：使用優化解碼
        elif hex_chunk.startswith(('fffb', 'fff3', 'fff2')):
            print(f'[{datetime.now()}] Detected MP3 format, using optimized decoder')
            return self.decode_mp3_optimized(audio_bytes)
        
        # PCM 格式：最快的回退選項
        else:
            try:
                stream = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767
                print(f'[{datetime.now()}] Decoded as 16-bit PCM, shape: {stream.shape}')
                return stream, 32000  # 假設 32kHz
            except Exception as e:
                print(f'[{datetime.now()}] PCM decode failed: {e}')
                return None, None

    def smooth_audio_transition(self, new_chunk):
        """音頻平滑處理，消除 chunk 連接處的不連續"""
        if self.previous_tail is None:
            # 第一個 chunk，直接返回
            if len(new_chunk) >= self.overlap_size:
                self.previous_tail = new_chunk[-self.overlap_size:].copy()
            print(f'[{datetime.now()}] First chunk, no smoothing needed')
            return new_chunk
        
        if len(new_chunk) < self.overlap_size:
            # chunk 太小，直接返回
            print(f'[{datetime.now()}] Chunk too small for smoothing: {len(new_chunk)}')
            return new_chunk
        
        try:
            # 交叉淡化處理
            fade_out = np.linspace(1.0, 0.0, self.overlap_size)  # 前一個 chunk 的尾部淡出
            fade_in = np.linspace(0.0, 1.0, self.overlap_size)   # 新 chunk 的開頭淡入
            
            # 混合重疊部分
            overlap_mix = (self.previous_tail * fade_out + 
                          new_chunk[:self.overlap_size] * fade_in)
            
            # 組合音頻：混合的重疊部分 + 新 chunk 的剩餘部分
            smoothed_chunk = np.concatenate([overlap_mix, new_chunk[self.overlap_size:]])
            
            # 保存新的尾部用於下次處理
            if len(new_chunk) >= self.overlap_size:
                self.previous_tail = new_chunk[-self.overlap_size:].copy()
            
            print(f'[{datetime.now()}] Applied audio smoothing, original: {len(new_chunk)}, smoothed: {len(smoothed_chunk)}')
            return smoothed_chunk
            
        except Exception as e:
            print(f'[{datetime.now()}] Audio smoothing failed: {e}')
            # 如果平滑處理失敗，返回原始 chunk
            if len(new_chunk) >= self.overlap_size:
                self.previous_tail = new_chunk[-self.overlap_size:].copy()
            return new_chunk


    def load_pronunciation_dict(self):
        """加載發音字典"""
        try:
            with open('pronunciation_dict.json', 'r', encoding='utf-8') as f:
                pronunciation_dict = json.load(f)
                print(f'[{datetime.now()}] Loaded pronunciation dictionary with {len(pronunciation_dict)} entries')
                return pronunciation_dict
        except FileNotFoundError:
            print(f'[{datetime.now()}] pronunciation_dict.json not found, using empty dict')
            return []
        except Exception as e:
            print(f'[{datetime.now()}] Error loading pronunciation_dict.json: {e}')
            return []

    def reset_audio_buffer(self):
        """重置音頻緩衝，用於新的對話開始"""
        self.previous_tail = None
        self.audio_buffer = []
        print(f'[{datetime.now()}] Audio buffer reset')

    def update_voice_id(self, new_voice_id):
        """動態更新語音 ID"""
        old_voice_id = self.voice_id
        self.voice_id = new_voice_id
        print(f'[{datetime.now()}] MiniMax TTS voice_id updated from {old_voice_id} to {new_voice_id}')

    def update_voice_config(self, voice_id, config):
        """動態更新完整的語音配置"""
        old_voice_id = self.voice_id
        self.voice_id = voice_id
        
        # 更新所有配置參數
        if 'model' in config:
            self.model = config['model']
        if 'speed' in config:
            self.speed = config.get('speed', 1)
        if 'vol' in config:
            self.vol = config.get('vol', 2)
        if 'pitch' in config:
            self.pitch = config.get('pitch', 0)
        if 'emotion' in config:
            self.emotion = config.get('emotion', 'happy')
            
        print(f'[{datetime.now()}] MiniMax TTS config updated: voice_id={voice_id}, speed={getattr(self, "speed", 1)}, vol={getattr(self, "vol", 2)}, pitch={getattr(self, "pitch", 0)}, emotion={getattr(self, "emotion", "happy")}')

    def txt_to_audio(self, msg):
        text, textevent = msg
        # 重置音頻緩衝，確保每次新對話都有乾淨的開始
        self.reset_audio_buffer()
        self.stream_tts(
            self.minimax_request(text),
            msg
        )

    def minimax_request(self, text) -> Iterator[bytes]:
        start = time.perf_counter()
        
        # 根據 replacements.json 來做一系列替換
        for rule in REPLACE_RULES:
            flag_val = 0
            for f in rule.get('flags', []):
                flag_val |= getattr(re, f)
            text = re.sub(rule['pattern'], rule['replacement'], text, flags=flag_val)

        print(f'[{datetime.now()}] minimax_tts text: {text}')

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "text": text,
            "stream": True,
            "pronunciation_dict": {
                "tone" : self.pronunciation_dict
            },
            "voice_setting": {
                "voice_id": self.voice_id,
                "speed": getattr(self, 'speed', 1),
                "vol": getattr(self, 'vol', 2),
                "pitch": getattr(self, 'pitch', 0),
                "emotion": getattr(self, 'emotion', 'happy')
            },
            "group_id": self.group_id
        }
        
        try:
            url = f"{self.base_url}"
            print(f'[{datetime.now()}] Making request to: {url}')
            print(f'[{datetime.now()}] Request payload: {json.dumps(payload, indent=2)}')
            
            res = requests.post(
                url,
                json=payload,
                headers=headers,
                stream=True
            )
            
            end = time.perf_counter()
            print(f'[{datetime.now()}] minimax_tts Time to make POST: {end-start}s')
            print(f'[{datetime.now()}] Response status: {res.status_code}')
            
            if res.status_code != 200:
                print(f'[{datetime.now()}] MiniMax API Error: {res.status_code} - {res.text}')
                return
                
            first = True
            # # 修改為:
            # print(f'[{datetime.now()}] Response status: {res.status_code}')
            # if res.headers.get('content-type') and 'json' in res.headers.get('content-type'):
            #     # 如果是 JSON 響應，直接打印完整內容
            #     print(f'[{datetime.now()}] Response content: {res.text}')
            # else:
            #     # 如果是流式響應，打印每一行
            #     print(f'[{datetime.now()}] Response content:')
            #     # 創建原始響應的副本以供檢查
            #     response_copy = res.content.decode('utf-8')
            #     print(response_copy)

            for line in res.iter_lines():
                if line and self.state == State.RUNNING:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data:'):
                        try:
                            data = json.loads(line_str[5:])  # Remove 'data:' prefix
                            if first:
                                end = time.perf_counter()
                                print(f'[{datetime.now()}] minimax_tts Time to first chunk: {end-start}s')
                                first = False
                            
                            if "data" in data and "audio" in data["data"]:
                                audio_hex = data["data"]["audio"]
                                if audio_hex:
                                    yield audio_hex
                        except json.JSONDecodeError as e:
                            print(f'[{datetime.now()}] JSON decode error: {e}')
                            continue
                            
        except Exception as e:
            print(f'[{datetime.now()}] MiniMax TTS Exception: {str(e)}')

    def stream_tts(self, audio_stream, msg):
        text, textevent = msg
        
        if self.buffered_mode:
            # 緩衝模式：收集所有 chunk，最後一次播放
            print(f'[{datetime.now()}] Using buffered mode for MiniMax TTS')
            self.stream_tts_buffered(audio_stream, msg)
        else:
            # 串流模式：即時播放每個 chunk（移除最後的完整 WAV）
            print(f'[{datetime.now()}] Using streaming mode for MiniMax TTS')
            self.stream_tts_streaming(audio_stream, msg)

    def find_complete_wav(self, chunk_data_list):
        """識別並返回完整版 WAV 音頻"""
        if not chunk_data_list:
            return None
        
        print(f'[{datetime.now()}] Analyzing {len(chunk_data_list)} chunks to find complete WAV')
        
        # 分析每個 chunk 的特徵
        chunk_analysis = []
        for i, (hex_chunk, audio_bytes, stream) in enumerate(chunk_data_list):
            analysis = {
                'index': i,
                'hex_length': len(hex_chunk),
                'bytes_length': len(audio_bytes),
                'audio_length': len(stream),
                'duration_ms': len(stream) / self.sample_rate * 1000,
                'is_wav_file': hex_chunk.startswith('49443304'),  # WAV 文件標識
                'stream': stream
            }
            chunk_analysis.append(analysis)
            
            print(f'[{datetime.now()}] Chunk {i}: {analysis["audio_length"]} samples, '
                  f'{analysis["duration_ms"]:.1f}ms, WAV: {analysis["is_wav_file"]}')
        
        # 策略 1：找到最大的音頻片段（通常是完整版）
        largest_chunk = max(chunk_analysis, key=lambda x: x['audio_length'])
        print(f'[{datetime.now()}] Largest chunk: index {largest_chunk["index"]}, '
              f'{largest_chunk["audio_length"]} samples, {largest_chunk["duration_ms"]:.1f}ms')
        
        # 策略 2：如果有明顯大於其他的 chunk，選擇它
        avg_length = sum(c['audio_length'] for c in chunk_analysis) / len(chunk_analysis)
        large_chunks = [c for c in chunk_analysis if c['audio_length'] > avg_length * 2]
        
        if large_chunks:
            # 選擇最大的那個
            selected_chunk = max(large_chunks, key=lambda x: x['audio_length'])
            print(f'[{datetime.now()}] Selected complete WAV: index {selected_chunk["index"]}, '
                  f'{selected_chunk["audio_length"]} samples, {selected_chunk["duration_ms"]:.1f}ms')
            return selected_chunk['stream']
        else:
            # 如果沒有明顯大的 chunk，選擇最大的
            print(f'[{datetime.now()}] No obviously large chunk found, using largest: '
                  f'index {largest_chunk["index"]}')
            return largest_chunk['stream']

    def stream_tts_buffered(self, audio_stream, msg):
        """緩衝模式：只播放完整版 WAV"""
        text, textevent = msg
        chunk_data_list = []
        
        print(f'[{datetime.now()}] Collecting audio chunks in buffered mode (complete WAV only)...')
        
        # 收集所有音頻 chunk 的原始數據和解碼結果
        for hex_chunk in audio_stream:
            if hex_chunk is not None and len(hex_chunk) > 0:
                try:
                    # 解碼音頻數據
                    audio_bytes = bytes.fromhex(hex_chunk)
                    stream, detected_sr = self.decode_audio_smart(hex_chunk, audio_bytes)
                    
                    if stream is not None:
                        # 重採樣到目標採樣率
                        if detected_sr != self.sample_rate:
                            stream = resampy.resample(x=stream, sr_orig=detected_sr, sr_new=self.sample_rate)
                        
                        # 確保數據類型和維度正確
                        stream = stream.astype(np.float32)
                        if stream.ndim > 1:
                            stream = stream[:, 0]
                        
                        # 收集原始數據和解碼結果
                        chunk_data_list.append((hex_chunk, audio_bytes, stream))
                        print(f'[{datetime.now()}] Collected chunk {len(chunk_data_list)}, '
                              f'shape: {stream.shape}, duration: {len(stream)/self.sample_rate:.2f}s')
                        
                except Exception as e:
                    print(f'[{datetime.now()}] Error collecting audio chunk: {e}')
                    continue
        
        # 找到完整版 WAV 並播放
        complete_wav = self.find_complete_wav(chunk_data_list)
        if complete_wav is not None:
            print(f'[{datetime.now()}] Playing complete WAV only, duration: {len(complete_wav)/self.sample_rate:.2f}s')
            self.play_complete_audio(complete_wav, text, textevent)
        else:
            print(f'[{datetime.now()}] No complete WAV found in buffered mode')
            # 發送結束事件
            eventpoint = {'status': 'end', 'text': text, 'msgenvent': textevent}
            self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)

    def play_complete_audio(self, complete_audio, text, textevent):
        """播放完整的合併音頻"""
        print(f'[{datetime.now()}] Playing complete audio, duration: {len(complete_audio)/self.sample_rate:.2f}s')
        
        # 儲存完整音頻（調試用）
        if self.debug_enabled:
            self.save_debug_audio(
                complete_audio, "complete_audio", 0,
                metadata={
                    'total_chunks': len(self.collected_chunks),
                    'total_duration_ms': len(complete_audio) / self.sample_rate * 1000,
                    'sample_rate': self.sample_rate,
                    'text': text
                }
            )
        
        # 分割為 20ms 幀並播放
        streamlen = complete_audio.shape[0]
        idx = 0
        first = True
        
        while streamlen >= self.chunk:
            eventpoint = None
            if first:
                eventpoint = {'status': 'start', 'text': text, 'msgenvent': textevent}
                first = False
            
            frame_data = complete_audio[idx:idx+self.chunk]
            
            # 儲存最終 20ms 幀（調試用）
            if self.debug_enabled:
                self.save_debug_audio(
                    frame_data, "final_frame", frame_id=self.frame_counter,
                    metadata={
                        'frame_in_complete': idx // self.chunk,
                        'frame_length': len(frame_data),
                        'is_start': eventpoint is not None and eventpoint.get('status') == 'start',
                        'text': text[:50] + '...' if len(text) > 50 else text
                    }
                )
                self.frame_counter += 1
            
            self.parent.put_audio_frame(frame_data, eventpoint)
            streamlen -= self.chunk
            idx += self.chunk
        
        # 發送結束事件
        eventpoint = {'status': 'end', 'text': text, 'msgenvent': textevent}
        self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)

    def stream_tts_streaming(self, audio_stream, msg):
        """串流模式：即時播放每個 chunk，跳過最後一個"""
        text, textevent = msg
        first = True
        cached_chunk = None  # 緩存機制，用於跳過最後一個 chunk
        
        print(f'[{datetime.now()}] Using streaming mode with last chunk skip')
        
        for hex_chunk in audio_stream:
            if hex_chunk is not None and len(hex_chunk) > 0:
                # 如果有緩存的 chunk，先處理它
                if cached_chunk is not None:
                    self.process_cached_chunk(cached_chunk, text, textevent, first)
                    first = False
                
                # 緩存當前 chunk（最後一個會被自動丟棄）
                cached_chunk = hex_chunk
        
        # 串流結束，最後一個 chunk 被自動丟棄
        print(f'[{datetime.now()}] Last chunk skipped in streaming mode')
        
        # 儲存調試元數據
        if self.debug_enabled:
            self.save_debug_metadata()
        
        # 發送結束事件
        eventpoint = {'status': 'end', 'text': text, 'msgenvent': textevent}
        self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)

    def process_cached_chunk(self, hex_chunk, text, textevent, first):
        """處理緩存的 chunk"""
        try:
            # 調試信息：檢查 hex 數據
            print(f'[{datetime.now()}] Processing cached chunk, length: {len(hex_chunk)}')
            print(f'[{datetime.now()}] First 40 chars: {hex_chunk[:40]}')
            
            # 檢查 hex 字符串長度是否為偶數
            if len(hex_chunk) % 2 != 0:
                print(f'[{datetime.now()}] Invalid hex length (not even): {len(hex_chunk)}')
                return
            
            # Hex 解碼為音頻數據
            audio_bytes = bytes.fromhex(hex_chunk)
            print(f'[{datetime.now()}] Audio bytes length: {len(audio_bytes)}')
            
            # 檢測數據類型：WAV 文件頭 vs 純音頻數據
            is_wav_file = hex_chunk.startswith('49443304')  # WAV 文件標識
            print(f'[{datetime.now()}] Is WAV file: {is_wav_file}')
            
            # 對於 WAV 文件，不檢查奇偶數長度
            if not is_wav_file and len(audio_bytes) % 2 != 0:
                print(f'[{datetime.now()}] Invalid audio bytes length for 16-bit PCM: {len(audio_bytes)}')
                return
            
            # 使用智能解碼方法
            stream, detected_sr = self.decode_audio_smart(hex_chunk, audio_bytes)
            
            if stream is None:
                print(f'[{datetime.now()}] Failed to decode audio chunk, skipping')
                return
            
            # 重採樣到目標採樣率（如果需要）
            if detected_sr != self.sample_rate:
                stream = resampy.resample(x=stream, sr_orig=detected_sr, sr_new=self.sample_rate)
                print(f'[{datetime.now()}] Resampled from {detected_sr}Hz to {self.sample_rate}Hz')
            
            # 確保數據類型為 float32
            stream = stream.astype(np.float32)
            if stream.ndim > 1:
                stream = stream[:, 0]
            
            # 儲存解碼後的音頻（調試用）
            if self.debug_enabled:
                self.save_debug_audio(
                    stream, "decoded_chunk", self.chunk_counter,
                    metadata={
                        'final_sr': self.sample_rate,
                        'shape': stream.shape,
                        'dtype': str(stream.dtype),
                        'duration_ms': len(stream) / self.sample_rate * 1000,
                        'text': text[:50] + '...' if len(text) > 50 else text
                    }
                )
            
            # 應用音頻平滑處理
            if stream.shape[0] > 0:
                smoothed_stream = self.smooth_audio_transition(stream)
                
                # 儲存平滑處理後的音頻（調試用）
                if self.debug_enabled:
                    self.save_debug_audio(
                        smoothed_stream, "smoothed_chunk", self.chunk_counter,
                        metadata={
                            'original_length': len(stream),
                            'smoothed_length': len(smoothed_stream),
                            'overlap_size': self.overlap_size,
                            'has_previous_tail': self.previous_tail is not None,
                            'text': text[:50] + '...' if len(text) > 50 else text
                        }
                    )
                
                streamlen = smoothed_stream.shape[0]
                idx = 0
                
                while streamlen >= self.chunk:
                    eventpoint = None
                    if first:
                        eventpoint = {'status': 'start', 'text': text, 'msgenvent': textevent}
                        first = False
                    
                    # 儲存最終 20ms 幀（調試用）
                    if self.debug_enabled:
                        frame_data = smoothed_stream[idx:idx+self.chunk]
                        self.save_debug_audio(
                            frame_data, "final_frame", frame_id=self.frame_counter,
                            metadata={
                                'chunk_id': self.chunk_counter,
                                'frame_in_chunk': idx // self.chunk,
                                'frame_length': len(frame_data),
                                'is_start': eventpoint is not None and eventpoint.get('status') == 'start',
                                'text': text[:50] + '...' if len(text) > 50 else text
                            }
                        )
                        self.frame_counter += 1
                    
                    self.parent.put_audio_frame(smoothed_stream[idx:idx+self.chunk], eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk
            else:
                print(f'[{datetime.now()}] Empty audio stream after processing')
            
            # 增加 chunk 計數器
            self.chunk_counter += 1
                    
        except Exception as e:
            print(f'[{datetime.now()}] Error processing cached chunk: {e}')
            import traceback
            traceback.print_exc()

###########################################################################################
class MiniMaxNonStreamTTS(BaseTTS):
    def __init__(self, opt, parent):
        super().__init__(opt, parent)
        self.api_key = os.getenv("MINIMAX_API_KEY")
        self.group_id = os.getenv("MINIMAX_GROUP_ID")
        self.base_url = "https://api.minimaxi.chat/v1/t2a_v2"
        self.model = os.getenv("MINIMAX_MODEL", "speech-02-turbo")
        self.voice_id = os.getenv("MINIMAX_VOICE_ID", "moss_audio_9e3d9106-42a6-11f0-b6c4-9e15325fe584")
        
        # 非串流模式專用配置
        self.audio_cache = os.getenv("MINIMAX_AUDIO_CACHE", "false").lower() == "true"
        self.retry_count = int(os.getenv("MINIMAX_RETRY_COUNT", "3"))
        self.timeout = int(os.getenv("MINIMAX_TIMEOUT", "30"))
        
        # 調試音檔儲存功能
        self.debug_enabled = os.getenv("MINIMAX_DEBUG_AUDIO", "false").lower() == "true"
        self.debug_path = os.getenv("MINIMAX_DEBUG_PATH", "debug_audio")
        self.session_id = getattr(opt, 'sessionid', 0)
        
        # 加載發音字典
        self.pronunciation_dict = self.load_pronunciation_dict()
        
        if self.debug_enabled:
            self.setup_debug_directories()
            print(f'[{datetime.now()}] MiniMax NonStream debug mode enabled, saving to: {self.debug_path}')
        
        if not self.api_key or not self.group_id:
            print(f'[{datetime.now()}] Warning: MiniMax API credentials not found in environment variables')

    def setup_debug_directories(self):
        """設置調試目錄結構"""
        try:
            session_path = Path(self.debug_path) / f"nonstream_session_{self.session_id}"
            session_path.mkdir(parents=True, exist_ok=True)
            print(f'[{datetime.now()}] NonStream debug directory created at: {session_path}')
        except Exception as e:
            print(f'[{datetime.now()}] Failed to create NonStream debug directories: {e}')
            self.debug_enabled = False

    def load_pronunciation_dict(self):
        """加載發音字典"""
        try:
            with open('pronunciation_dict.json', 'r', encoding='utf-8') as f:
                pronunciation_dict = json.load(f)
                print(f'[{datetime.now()}] NonStream loaded pronunciation dictionary with {len(pronunciation_dict)} entries')
                return pronunciation_dict
        except FileNotFoundError:
            print(f'[{datetime.now()}] pronunciation_dict.json not found, using empty dict')
            return []
        except Exception as e:
            print(f'[{datetime.now()}] Error loading pronunciation_dict.json: {e}')
            return []

    def update_voice_id(self, new_voice_id):
        """動態更新語音 ID"""
        old_voice_id = self.voice_id
        self.voice_id = new_voice_id
        print(f'[{datetime.now()}] MiniMax NonStream TTS voice_id updated from {old_voice_id} to {new_voice_id}')

    def update_voice_config(self, voice_id, config):
        """動態更新完整的語音配置"""
        old_voice_id = self.voice_id
        self.voice_id = voice_id
        
        # 更新所有配置參數
        if 'model' in config:
            self.model = config['model']
        if 'speed' in config:
            self.speed = config.get('speed', 1)
        if 'vol' in config:
            self.vol = config.get('vol', 2)
        if 'pitch' in config:
            self.pitch = config.get('pitch', 0)
        if 'emotion' in config:
            self.emotion = config.get('emotion', 'happy')
            
        print(f'[{datetime.now()}] MiniMax NonStream TTS config updated: voice_id={voice_id}, speed={getattr(self, "speed", 1)}, vol={getattr(self, "vol", 2)}, pitch={getattr(self, "pitch", 0)}, emotion={getattr(self, "emotion", "happy")}')

    def txt_to_audio(self, msg):
        """主要的文本轉音頻方法"""
        text, textevent = msg
        
        try:
            # 發送非串流請求獲取音檔 URL
            audio_url = self.minimax_non_stream_request(text)
            
            if audio_url:
                # 下載並處理音檔
                self.download_and_process_audio(audio_url, text, textevent)
            else:
                print(f'[{datetime.now()}] Failed to get audio URL from MiniMax NonStream API')
                # 發送結束事件
                eventpoint = {'status': 'end', 'text': text, 'msgenvent': textevent}
                self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)
                
        except Exception as e:
            print(f'[{datetime.now()}] MiniMax NonStream TTS error: {e}')
            import traceback
            traceback.print_exc()
            # 發送結束事件
            eventpoint = {'status': 'end', 'text': text, 'msgenvent': textevent}
            self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)

    def minimax_non_stream_request(self, text):
        """發送非串流請求，獲取音檔 URL"""
        start = time.perf_counter()
        
        # 根據 replacements.json 來做一系列替換
        for rule in REPLACE_RULES:
            flag_val = 0
            for f in rule.get('flags', []):
                flag_val |= getattr(re, f)
            text = re.sub(rule['pattern'], rule['replacement'], text, flags=flag_val)

        print(f'[{datetime.now()}] MiniMax NonStream TTS text: {text}')

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "text": text,
            "voice_setting": {
                "voice_id": self.voice_id,
                "speed": getattr(self, 'speed', 1),
                "vol": getattr(self, 'vol', 2),
                "pitch": getattr(self, 'pitch', 0),
                "emotion": getattr(self, 'emotion', 'happy')
            },
            "audio_setting": {
                "sample_rate": 16000, 
                "bitrate": 128000,     
                "format": "wav",   
                "channel": 1  
            },
            "stream": False,
            "output_format": "url",
            "pronunciation_dict": {
                "tone": self.pronunciation_dict
            },
            "group_id": self.group_id
        }
        
        for attempt in range(self.retry_count):
            try:
                url = f"{self.base_url}"
                print(f'[{datetime.now()}] Making NonStream request to: {url} (attempt {attempt + 1})')
                print(f'[{datetime.now()}] Request payload: {json.dumps(payload, indent=2)}')
                
                res = requests.post(
                    url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout
                )
                
                end = time.perf_counter()
                print(f'[{datetime.now()}] MiniMax NonStream Time to make POST: {end-start:.2f}s')
                print(f'[{datetime.now()}] Response status: {res.status_code}')
                
                if res.status_code == 200:
                    response_data = res.json()
                    print(f'[{datetime.now()}] Response data: {json.dumps(response_data, indent=2)}')
                    
                    if response_data.get("base_resp", {}).get("status_code") == 0:
                        audio_url = response_data.get("data", {}).get("audio")
                        extra_info = response_data.get("extra_info", {})
                        
                        print(f'[{datetime.now()}] Got audio URL: {audio_url}')
                        print(f'[{datetime.now()}] Audio info: {extra_info}')
                        
                        return audio_url
                    else:
                        print(f'[{datetime.now()}] API returned error: {response_data}')
                else:
                    print(f'[{datetime.now()}] HTTP error {res.status_code}: {res.text}')
                    
            except requests.exceptions.Timeout:
                print(f'[{datetime.now()}] Request timeout (attempt {attempt + 1})')
            except Exception as e:
                print(f'[{datetime.now()}] Request error (attempt {attempt + 1}): {e}')
            
            if attempt < self.retry_count - 1:
                print(f'[{datetime.now()}] Retrying in 1 second...')
                time.sleep(1)
        
        print(f'[{datetime.now()}] All retry attempts failed')
        return None

    def download_and_process_audio(self, audio_url, text, textevent):
        """下載音檔並轉換為 LiveTalking 串流格式"""
        try:
            print(f'[{datetime.now()}] Downloading audio from: {audio_url}')
            
            # 下載音檔
            for attempt in range(self.retry_count):
                try:
                    audio_response = requests.get(audio_url, timeout=self.timeout)
                    if audio_response.status_code == 200:
                        audio_data = audio_response.content
                        print(f'[{datetime.now()}] Downloaded audio: {len(audio_data)} bytes')
                        break
                    else:
                        print(f'[{datetime.now()}] Download failed with status {audio_response.status_code}')
                except Exception as e:
                    print(f'[{datetime.now()}] Download error (attempt {attempt + 1}): {e}')
                    if attempt < self.retry_count - 1:
                        time.sleep(1)
            else:
                print(f'[{datetime.now()}] Failed to download audio after {self.retry_count} attempts')
                return
            
            # 儲存原始音檔（調試用）
            if self.debug_enabled:
                debug_path = Path(self.debug_path) / f"nonstream_session_{self.session_id}"
                audio_file = debug_path / f"original_audio_{int(time.time())}.wav"
                with open(audio_file, 'wb') as f:
                    f.write(audio_data)
                print(f'[{datetime.now()}] Saved original audio to: {audio_file}')
            
            # 使用 soundfile 讀取音檔
            stream, sample_rate = sf.read(BytesIO(audio_data))
            print(f'[{datetime.now()}] Loaded audio: sample_rate={sample_rate}, shape={stream.shape}')
            
            # 確保數據類型為 float32
            stream = stream.astype(np.float32)
            
            # 確保是單聲道
            if stream.ndim > 1:
                print(f'[{datetime.now()}] Converting multi-channel to mono')
                stream = stream[:, 0]
            
            # 重採樣到目標採樣率（如果需要）
            if sample_rate != self.sample_rate:
                print(f'[{datetime.now()}] Resampling from {sample_rate}Hz to {self.sample_rate}Hz')
                stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)
            
            # 轉換為串流格式並播放
            self.convert_to_stream_format(stream, text, textevent)
            
        except Exception as e:
            print(f'[{datetime.now()}] Error processing audio: {e}')
            import traceback
            traceback.print_exc()

    def convert_to_stream_format(self, audio_stream, text, textevent):
        """將完整音檔轉換為 LiveTalking 串流格式"""
        print(f'[{datetime.now()}] Converting to stream format, duration: {len(audio_stream)/self.sample_rate:.2f}s')
        
        streamlen = audio_stream.shape[0]
        idx = 0
        first = True
        
        while streamlen >= self.chunk:
            eventpoint = None
            if first:
                eventpoint = {'status': 'start', 'text': text, 'msgenvent': textevent}
                first = False
            
            frame_data = audio_stream[idx:idx+self.chunk]
            
            # 儲存 20ms 幀（調試用）
            if self.debug_enabled:
                debug_path = Path(self.debug_path) / f"nonstream_session_{self.session_id}"
                frame_file = debug_path / f"frame_{idx//self.chunk:06d}.wav"
                sf.write(frame_file, frame_data, self.sample_rate)
            
            self.parent.put_audio_frame(frame_data, eventpoint)
            streamlen -= self.chunk
            idx += self.chunk
        
        # 發送結束事件
        eventpoint = {'status': 'end', 'text': text, 'msgenvent': textevent}
        self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)
        
        print(f'[{datetime.now()}] NonStream audio conversion completed')

###########################################################################################
class MiniMaxWebSocketTTS(BaseTTS):
    def __init__(self, opt, parent):
        super().__init__(opt, parent)
        
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets library not available. Please install: pip install websockets")
            
        self.api_key = os.getenv("MINIMAX_API_KEY")
        self.group_id = os.getenv("MINIMAX_GROUP_ID")
        self.ws_url = "wss://api.minimaxi.chat/ws/v1/t2a_v2"
        self.model = os.getenv("MINIMAX_MODEL", "speech-02-turbo")
        self.voice_id = os.getenv("MINIMAX_VOICE_ID", "moss_audio_9e3d9106-42a6-11f0-b6c4-9e15325fe584")
        
        # WebSocket 連接管理
        self.websocket = None
        self.session_id = None
        self.trace_id = None
        self.is_connected = False
        self.is_task_started = False
        
        # 音頻處理
        self.audio_queue = asyncio.Queue()
        self.loop = None
        
        # MP3 流累積緩衝
        self.mp3_buffer = b''
        self.min_buffer_size = 4096  # 最小緩衝大小
        self.max_buffer_size = 32768  # 最大緩衝大小
        
        # 調試音檔儲存功能
        self.debug_enabled = os.getenv("MINIMAX_DEBUG_AUDIO", "false").lower() == "true"
        self.debug_path = os.getenv("MINIMAX_DEBUG_PATH", "debug_audio")
        self.session_debug_id = getattr(opt, 'sessionid', 0)
        self.chunk_counter = 0
        self.frame_counter = 0
        self.metadata = []
        
        if self.debug_enabled:
            self.setup_debug_directories()
            print(f'[{datetime.now()}] MiniMax WebSocket debug mode enabled, saving to: {self.debug_path}')
        
        if not self.api_key or not self.group_id:
            print(f'[{datetime.now()}] Warning: MiniMax API credentials not found in environment variables')

    def setup_debug_directories(self):
        """設置調試目錄結構"""
        try:
            session_path = Path(self.debug_path) / f"ws_session_{self.session_debug_id}"
            
            # 創建所有必要的目錄
            (session_path / "raw_chunks").mkdir(parents=True, exist_ok=True)
            (session_path / "decoded_chunks").mkdir(parents=True, exist_ok=True)
            (session_path / "final_frames").mkdir(parents=True, exist_ok=True)
            
            # 重置計數器和元數據
            self.chunk_counter = 0
            self.frame_counter = 0
            self.metadata = []
            
            print(f'[{datetime.now()}] WebSocket debug directories created at: {session_path}')
            
        except Exception as e:
            print(f'[{datetime.now()}] Failed to create WebSocket debug directories: {e}')
            self.debug_enabled = False

    def save_debug_audio(self, audio_data, stage, chunk_id=None, frame_id=None, metadata=None):
        """儲存調試音檔"""
        if not self.debug_enabled:
            return
            
        try:
            session_path = Path(self.debug_path) / f"ws_session_{self.session_debug_id}"
            
            if stage == "raw_chunk":
                # 儲存原始 MP3 解碼後的音頻
                filename = f"ws_raw_chunk_{chunk_id:04d}.wav"
                filepath = session_path / "raw_chunks" / filename
                sf.write(filepath, audio_data, self.sample_rate)
                
            elif stage == "decoded_chunk":
                # 儲存格式解碼後的音頻
                filename = f"ws_decoded_chunk_{chunk_id:04d}.wav"
                filepath = session_path / "decoded_chunks" / filename
                sf.write(filepath, audio_data, self.sample_rate)
                
            elif stage == "final_frame":
                # 儲存最終 20ms 幀
                filename = f"ws_final_frame_{frame_id:06d}.wav"
                filepath = session_path / "final_frames" / filename
                sf.write(filepath, audio_data, self.sample_rate)
            
            # 記錄元數據
            if metadata:
                metadata['timestamp'] = datetime.now().isoformat()
                metadata['stage'] = stage
                metadata['filename'] = filename
                self.metadata.append(metadata)
                
        except Exception as e:
            print(f'[{datetime.now()}] Failed to save WebSocket debug audio ({stage}): {e}')

    def save_debug_metadata(self):
        """儲存調試元數據"""
        if not self.debug_enabled or not self.metadata:
            return
            
        try:
            session_path = Path(self.debug_path) / f"ws_session_{self.session_debug_id}"
            metadata_file = session_path / "ws_metadata.json"
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
                
            print(f'[{datetime.now()}] WebSocket debug metadata saved: {metadata_file}')
            
        except Exception as e:
            print(f'[{datetime.now()}] Failed to save WebSocket debug metadata: {e}')

    def update_voice_id(self, new_voice_id):
        """動態更新語音 ID"""
        old_voice_id = self.voice_id
        self.voice_id = new_voice_id
        print(f'[{datetime.now()}] MiniMax WebSocket TTS voice_id updated from {old_voice_id} to {new_voice_id}')

    async def connect_websocket(self):
        """建立 WebSocket 連接"""
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            
            # 創建 SSL 上下文
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            print(f'[{datetime.now()}] Connecting to WebSocket: {self.ws_url}')
            
            self.websocket = await websockets.connect(
                self.ws_url, 
                extra_headers=headers, 
                ssl=ssl_context
            )
            
            # 等待連接成功響應
            response = await self.websocket.recv()
            connected_data = json.loads(response)
            
            if connected_data.get("event") == "connected_success":
                self.session_id = connected_data.get("session_id")
                self.trace_id = connected_data.get("trace_id")
                self.is_connected = True
                print(f'[{datetime.now()}] WebSocket connected successfully, session_id: {self.session_id}')
                return True
            else:
                print(f'[{datetime.now()}] WebSocket connection failed: {connected_data}')
                return False
                
        except Exception as e:
            print(f'[{datetime.now()}] WebSocket connection error: {e}')
            return False

    async def start_task(self):
        """發送 task_start 事件"""
        if not self.is_connected:
            return False
            
        try:
            start_msg = {
                "event": "task_start",
                "model": self.model,
                "voice_setting": {
                    "voice_id": self.voice_id,
                    "speed": 1,
                    "vol": 1,
                    "pitch": 0,
                    "emotion": "happy"
                }
            }
            
            print(f'[{datetime.now()}] Sending task_start: {json.dumps(start_msg, indent=2)}')
            await self.websocket.send(json.dumps(start_msg))
            
            # 等待 task_started 響應
            response = await self.websocket.recv()
            response_data = json.loads(response)
            
            if response_data.get("event") == "task_started":
                self.is_task_started = True
                print(f'[{datetime.now()}] Task started successfully')
                return True
            else:
                print(f'[{datetime.now()}] Task start failed: {response_data}')
                return False
                
        except Exception as e:
            print(f'[{datetime.now()}] Task start error: {e}')
            return False

    async def send_text(self, text):
        """發送文本進行合成"""
        if not self.is_task_started:
            return
            
        try:
            # 根據 replacements.json 來做一系列替換
            for rule in REPLACE_RULES:
                flag_val = 0
                for f in rule.get('flags', []):
                    flag_val |= getattr(re, f)
                text = re.sub(rule['pattern'], rule['replacement'], text, flags=flag_val)

            print(f'[{datetime.now()}] WebSocket sending text: {text}')
            
            continue_msg = {
                "event": "task_continue",
                "text": text
            }
            
            await self.websocket.send(json.dumps(continue_msg))
            
        except Exception as e:
            print(f'[{datetime.now()}] Send text error: {e}')

    async def receive_audio_stream(self, text, textevent):
        """接收音頻串流"""
        first = True
        
        try:
            while True:
                response = await self.websocket.recv()
                response_data = json.loads(response)
                
                # 檢查錯誤
                if response_data.get("event") == "task_failed":
                    print(f'[{datetime.now()}] Task failed: {response_data}')
                    break
                
                # 處理音頻數據
                if "data" in response_data and "audio" in response_data["data"]:
                    audio_hex = response_data["data"]["audio"]
                    extra_info = response_data.get("extra_info", {})
                    
                    print(f'[{datetime.now()}] Received audio chunk, length: {len(audio_hex)}, extra_info: {extra_info}')
                    
                    if audio_hex:
                        await self.process_audio_chunk(audio_hex, text, textevent, first)
                        first = False
                        
                # 檢查是否完成
                if response_data.get("is_final"):
                    print(f'[{datetime.now()}] Audio stream completed')
                    # 處理剩餘的緩衝區數據
                    await self.flush_remaining_buffer(text, textevent)
                    break
                    
        except Exception as e:
            print(f'[{datetime.now()}] Receive audio stream error: {e}')

    async def process_audio_chunk(self, audio_hex, text, textevent, is_first):
        """處理音頻 chunk - 使用累積解碼法"""
        try:
            # 如果是空的 chunk，跳過處理
            if not audio_hex:
                print(f'[{datetime.now()}] Empty audio chunk, skipping')
                return
                
            # Hex 解碼為音頻數據
            audio_bytes = bytes.fromhex(audio_hex)
            
            # 累積 MP3 數據到緩衝區
            self.mp3_buffer += audio_bytes
            print(f'[{datetime.now()}] Added {len(audio_bytes)} bytes to buffer, total: {len(self.mp3_buffer)} bytes')
            
            # 嘗試解碼累積的 MP3 數據
            decoded_audio = self.try_decode_mp3_buffer()
            
            if decoded_audio is not None:
                # 成功解碼，處理音頻數據
                await self.process_decoded_audio(decoded_audio, text, textevent, is_first)
            else:
                # 解碼失敗，檢查緩衝區大小
                if len(self.mp3_buffer) > self.max_buffer_size:
                    print(f'[{datetime.now()}] Buffer too large ({len(self.mp3_buffer)} bytes), clearing')
                    self.mp3_buffer = b''
                else:
                    print(f'[{datetime.now()}] Waiting for more data to decode MP3')
            
        except Exception as e:
            print(f'[{datetime.now()}] Process audio chunk error: {e}')
            import traceback
            traceback.print_exc()

    def try_decode_mp3_buffer(self):
        """嘗試解碼 MP3 緩衝區"""
        if len(self.mp3_buffer) < self.min_buffer_size:
            return None
            
        if not PYDUB_AVAILABLE:
            print(f'[{datetime.now()}] pydub not available, cannot decode MP3')
            return None
            
        try:
            # 嘗試解碼整個緩衝區
            audio_segment = AudioSegment.from_file(
                BytesIO(self.mp3_buffer), 
                format="mp3"
            ).set_frame_rate(self.sample_rate).set_channels(1)
            
            # 轉換為 numpy array
            stream = np.array(audio_segment.get_array_of_samples(), dtype=np.float32) / 32767.0
            
            print(f'[{datetime.now()}] Successfully decoded MP3 buffer, shape: {stream.shape}')
            
            # 清空緩衝區
            self.mp3_buffer = b''
            
            return stream
            
        except Exception as e:
            # 解碼失敗，可能需要更多數據
            print(f'[{datetime.now()}] MP3 decode failed: {e}')
            return None

    async def process_decoded_audio(self, stream, text, textevent, is_first):
        """處理解碼後的音頻數據"""
        try:
            # 儲存解碼後的音頻（調試用）
            if self.debug_enabled:
                self.save_debug_audio(
                    stream, "decoded_chunk", self.chunk_counter,
                    metadata={
                        'sample_rate': self.sample_rate,
                        'shape': stream.shape,
                        'duration_ms': len(stream) / self.sample_rate * 1000,
                        'buffer_size': len(self.mp3_buffer),
                        'text': text[:50] + '...' if len(text) > 50 else text
                    }
                )
            
            # 分割為 20ms 幀並發送
            if stream.shape[0] > 0:
                streamlen = stream.shape[0]
                idx = 0
                
                while streamlen >= self.chunk:
                    eventpoint = None
                    if is_first and idx == 0:
                        eventpoint = {'status': 'start', 'text': text, 'msgenvent': textevent}
                        is_first = False
                    
                    frame_data = stream[idx:idx+self.chunk]
                    
                    # 儲存最終 20ms 幀（調試用）
                    if self.debug_enabled:
                        self.save_debug_audio(
                            frame_data, "final_frame", frame_id=self.frame_counter,
                            metadata={
                                'chunk_id': self.chunk_counter,
                                'frame_in_chunk': idx // self.chunk,
                                'frame_length': len(frame_data),
                                'is_start': eventpoint is not None and eventpoint.get('status') == 'start',
                                'text': text[:50] + '...' if len(text) > 50 else text
                            }
                        )
                        self.frame_counter += 1
                    
                    self.parent.put_audio_frame(frame_data, eventpoint)
                    streamlen -= self.chunk
                    idx += self.chunk
            
            # 增加 chunk 計數器
            self.chunk_counter += 1
            
        except Exception as e:
            print(f'[{datetime.now()}] Process decoded audio error: {e}')
            import traceback
            traceback.print_exc()

    async def flush_remaining_buffer(self, text, textevent):
        """處理剩餘的緩衝區數據"""
        if len(self.mp3_buffer) > 0:
            print(f'[{datetime.now()}] Flushing remaining buffer: {len(self.mp3_buffer)} bytes')
            
            # 嘗試強制解碼剩餘數據
            try:
                if not PYDUB_AVAILABLE:
                    print(f'[{datetime.now()}] pydub not available, cannot decode remaining MP3')
                    return
                    
                # 嘗試解碼剩餘的緩衝區，即使可能不完整
                audio_segment = AudioSegment.from_file(
                    BytesIO(self.mp3_buffer), 
                    format="mp3"
                ).set_frame_rate(self.sample_rate).set_channels(1)
                
                # 轉換為 numpy array
                stream = np.array(audio_segment.get_array_of_samples(), dtype=np.float32) / 32767.0
                
                print(f'[{datetime.now()}] Successfully decoded remaining buffer, shape: {stream.shape}')
                
                # 處理剩餘音頻
                await self.process_decoded_audio(stream, text, textevent, False)
                
                # 清空緩衝區
                self.mp3_buffer = b''
                
            except Exception as e:
                print(f'[{datetime.now()}] Failed to decode remaining buffer: {e}')
                # 清空緩衝區，避免下次使用時出現問題
                self.mp3_buffer = b''

    async def finish_task(self):
        """結束任務並關閉連接"""
        try:
            if self.is_task_started:
                finish_msg = {"event": "task_finish"}
                await self.websocket.send(json.dumps(finish_msg))
                
                # 等待 task_finished 響應
                response = await self.websocket.recv()
                response_data = json.loads(response)
                print(f'[{datetime.now()}] Task finish response: {response_data}')
            
            if self.websocket:
                await self.websocket.close()
                print(f'[{datetime.now()}] WebSocket connection closed')
                
        except Exception as e:
            print(f'[{datetime.now()}] Finish task error: {e}')
        finally:
            self.is_connected = False
            self.is_task_started = False
            self.websocket = None

    def txt_to_audio(self, msg):
        """主要的文本轉音頻方法"""
        text, textevent = msg
        
        # 在新的事件循環中運行 WebSocket 操作
        try:
            # 創建新的事件循環
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            
            # 運行 WebSocket TTS
            self.loop.run_until_complete(self._websocket_tts(text, textevent))
            
        except Exception as e:
            print(f'[{datetime.now()}] WebSocket TTS error: {e}')
            import traceback
            traceback.print_exc()
        finally:
            if self.loop:
                self.loop.close()

    async def _websocket_tts(self, text, textevent):
        """WebSocket TTS 的異步實現"""
        try:
            # 建立連接
            if not await self.connect_websocket():
                return
            
            # 開始任務
            if not await self.start_task():
                return
            
            # 發送文本
            await self.send_text(text)
            
            # 接收音頻串流
            await self.receive_audio_stream(text, textevent)
            
            # 儲存調試元數據
            if self.debug_enabled:
                self.save_debug_metadata()
            
            # 發送結束事件
            eventpoint = {'status': 'end', 'text': text, 'msgenvent': textevent}
            self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)
            
        finally:
            # 結束任務
            await self.finish_task()

###########################################################################################
class IndexTTS(BaseTTS):
    def __init__(self, opt, parent):
        super().__init__(opt, parent)
        self.server_url = os.getenv("INDEX_TTS_SERVER", "http://57.182.124.55:8001")
        self.voice_id = int(os.getenv("INDEX_VOICE_ID", "0"))
        self.voice_configs = self.load_voice_configs()
        
        # 簡繁轉換
        self.converter = opencc.OpenCC('t2s')  # 繁體轉簡體
        print(f'[{datetime.now()}] IndexTTS initialized with server: {self.server_url}, voice_id: {self.voice_id}')
        
    def load_voice_configs(self):
        """載入語音配置"""
        try:
            with open('index_voices.json', 'r', encoding='utf-8') as f:
                configs = json.load(f)
                print(f'[{datetime.now()}] Loaded {len(configs)} voice configurations from index_voices.json')
                return configs
        except Exception as e:
            print(f'[{datetime.now()}] Error loading index_voices.json: {e}')
            return {}
    
    def update_voice_id(self, new_voice_id):
        """動態更新語音 ID"""
        old_voice_id = self.voice_id
        self.voice_id = new_voice_id
        print(f'[{datetime.now()}] IndexTTS voice_id updated from {old_voice_id} to {new_voice_id}')
    
    def update_voice_config(self, voice_id, config=None):
        """動態更新語音配置"""
        old_voice_id = self.voice_id
        self.voice_id = voice_id
        
        # 重新載入語音配置以確保最新
        self.voice_configs = self.load_voice_configs()
        
        print(f'[{datetime.now()}] IndexTTS voice_id updated from {old_voice_id} to {voice_id}')
        print(f'[{datetime.now()}] Available voice configs reloaded: {list(self.voice_configs.keys())}')
    
    def txt_to_audio(self, msg):
        text, textevent = msg
        
        # 根據 replacements.json 來做一系列替換
        for rule in REPLACE_RULES:
            flag_val = 0
            for f in rule.get('flags', []):
                flag_val |= getattr(re, f)
            text = re.sub(rule['pattern'], rule['replacement'], text, flags=flag_val)
        # 繁體轉簡體
        simplified_text = self.converter.convert(text)
        print(f'[{datetime.now()}] IndexTTS Original: {text}')
        print(f'[{datetime.now()}] IndexTTS Simplified: {simplified_text}')
        
        # 獲取語音配置
        voice_key = f"voice_{self.voice_id}"
        if voice_key not in self.voice_configs:
            print(f'[{datetime.now()}] Voice ID {self.voice_id} not found in index_voices.json')
            # 發送結束事件
            eventpoint = {'status': 'end', 'text': text, 'msgenvent': textevent}
            self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)
            return
            
        voice_config = self.voice_configs[voice_key]
        print(f'[{datetime.now()}] Using voice config: {voice_config["name"]}')
        
        # 發送請求並處理音頻
        self.stream_tts(
            self.index_tts_request(simplified_text, voice_config),
            msg
        )
    
    def index_tts_request(self, text, voice_config):
        """發送 Index TTS 請求"""
        start = time.perf_counter()
        
        

        print(f'[{datetime.now()}] IndexTTS final text: {text}')
        
        payload = {
            "text": text,
            "audio_paths": voice_config["audio_paths"],
            "seed": voice_config.get("seed", 2)
        }
        
        try:
            url = f"{self.server_url}/tts_url"
            print(f'[{datetime.now()}] IndexTTS request to: {url}')
            print(f'[{datetime.now()}] IndexTTS payload: {json.dumps(payload, indent=2)}')
            
            response = requests.post(
                url,
                json=payload,
                timeout=30
            )
            
            end = time.perf_counter()
            print(f'[{datetime.now()}] IndexTTS Time to make POST: {end-start:.2f}s')
            print(f'[{datetime.now()}] IndexTTS Response status: {response.status_code}')
            
            if response.status_code == 200:
                print(f'[{datetime.now()}] IndexTTS Response size: {len(response.content)} bytes')
                # 直接返回 WAV 音頻數據
                yield response.content
            else:
                print(f'[{datetime.now()}] IndexTTS Error: {response.status_code} - {response.text}')
                
        except Exception as e:
            print(f'[{datetime.now()}] IndexTTS Exception: {str(e)}')
    
    def stream_tts(self, audio_stream, msg):
        """處理音頻流"""
        text, textevent = msg
        first = True
        
        for audio_data in audio_stream:
            if audio_data and len(audio_data) > 0:
                # 直接處理 16kHz WAV 數據
                stream = self.process_wav_data(audio_data)
                
                if stream is not None:
                    streamlen = stream.shape[0]
                    idx = 0
                    
                    while streamlen >= self.chunk:
                        eventpoint = None
                        if first:
                            eventpoint = {'status': 'start', 'text': text, 'msgenvent': textevent}
                            first = False
                        
                        self.parent.put_audio_frame(stream[idx:idx+self.chunk], eventpoint)
                        streamlen -= self.chunk
                        idx += self.chunk
        
        # 發送結束事件
        eventpoint = {'status': 'end', 'text': text, 'msgenvent': textevent}
        self.parent.put_audio_frame(np.zeros(self.chunk, np.float32), eventpoint)
    
    def process_wav_data(self, audio_data):
        """處理 WAV 音頻數據"""
        try:
            # 使用 soundfile 讀取 WAV
            stream, sample_rate = sf.read(BytesIO(audio_data))
            print(f'[{datetime.now()}] IndexTTS audio: sr={sample_rate}, shape={stream.shape}')
            
            # 確保是 float32 格式
            stream = stream.astype(np.float32)
            
            # 確保單聲道
            if stream.ndim > 1:
                print(f'[{datetime.now()}] IndexTTS converting to mono')
                stream = stream[:, 0]
            
            # 驗證採樣率（應該是 16kHz）
            if sample_rate != self.sample_rate:
                print(f'[{datetime.now()}] IndexTTS Warning: Expected 16kHz, got {sample_rate}Hz, resampling...')
                stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)
            else:
                print(f'[{datetime.now()}] IndexTTS Perfect! Audio is already 16kHz, no resampling needed')
            
            return stream
            
        except Exception as e:
            print(f'[{datetime.now()}] IndexTTS Error processing WAV data: {e}')
            return None
