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

from typing import Iterator

import requests

import queue
from queue import Queue
from io import BytesIO
from threading import Thread, Event
from enum import Enum
from datetime import datetime
import re

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
        if re.search(r"oem\s*/\s*odm", text, flags=re.IGNORECASE):
            text = re.sub(r"\s*/\s*", "或", text, flags=re.IGNORECASE)
        elif re.search(r"odm\s*/\s*oem", text, flags=re.IGNORECASE):
            text = re.sub(r"\s*/\s*", "或", text, flags=re.IGNORECASE)
        req["text"] = text # 更新text參數
        
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
                stream = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32767
                stream = resampy.resample(x=stream, sr_orig=32000, sr_new=self.sample_rate)
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
