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

import math
import torch
import numpy as np

#from .utils import *
import subprocess
import os
import time
import torch.nn.functional as F
import cv2
import glob
import pickle
import copy

import queue
from queue import Queue
from threading import Thread, Event
import torch.multiprocessing as mp

from musetalk.utils.utils import get_file_type,get_video_fps,datagen
#from musetalk.utils.preprocessing import get_landmark_and_bbox,read_imgs,coord_placeholder
from musetalk.utils.blending import get_image,get_image_prepare_material,get_image_blending
from musetalk.utils.utils import load_all_model,load_diffusion_model,load_audio_model
from musetalk.whisper.audio2feature import Audio2Feature

from museasr import MuseASR
import asyncio
from av import AudioFrame, VideoFrame
from basereal import BaseReal

from tqdm import tqdm
from datetime import datetime

def load_model():
    # load model weights
    audio_processor,vae, unet, pe = load_all_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timesteps = torch.tensor([0], device=device)
    pe = pe.half()
    vae.vae = vae.vae.half()
    #vae.vae.share_memory()
    unet.model = unet.model.half()
    #unet.model.share_memory()
    return vae, unet, pe, timesteps, audio_processor

def load_avatar(avatar_id):
    #self.video_path = '' #video_path
    #self.bbox_shift = opt.bbox_shift
    avatar_path = f"./data/avatars/{avatar_id}"
    full_imgs_path = f"{avatar_path}/full_imgs" 
    coords_path = f"{avatar_path}/coords.pkl"
    latents_out_path= f"{avatar_path}/latents.pt"
    video_out_path = f"{avatar_path}/vid_output/"
    mask_out_path =f"{avatar_path}/mask"
    mask_coords_path =f"{avatar_path}/mask_coords.pkl"
    avatar_info_path = f"{avatar_path}/avator_info.json"
    # self.avatar_info = {
    #     "avatar_id":self.avatar_id,
    #     "video_path":self.video_path,
    #     "bbox_shift":self.bbox_shift   
    # }

    input_latent_list_cycle = torch.load(latents_out_path)  #,weights_only=True
    with open(coords_path, 'rb') as f:
        coord_list_cycle = pickle.load(f)
    input_img_list = glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    frame_list_cycle = read_imgs(input_img_list)
    with open(mask_coords_path, 'rb') as f:
        mask_coords_list_cycle = pickle.load(f)
    input_mask_list = glob.glob(os.path.join(mask_out_path, '*.[jpJP][pnPN]*[gG]'))
    input_mask_list = sorted(input_mask_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    mask_list_cycle = read_imgs(input_mask_list)
    return frame_list_cycle,mask_list_cycle,coord_list_cycle,mask_coords_list_cycle,input_latent_list_cycle

@torch.no_grad()
def warm_up(batch_size,model):
    # 预热函数
    print(f'[{datetime.now()}] warmup model...')
    vae, unet, pe, timesteps, audio_processor = model
    #batch_size = 16
    #timesteps = torch.tensor([0], device=unet.device)
    whisper_batch = np.ones((batch_size, 50, 384), dtype=np.uint8)
    latent_batch = torch.ones(batch_size, 8, 32, 32).to(unet.device)

    audio_feature_batch = torch.from_numpy(whisper_batch)
    audio_feature_batch = audio_feature_batch.to(device=unet.device, dtype=unet.model.dtype)
    audio_feature_batch = pe(audio_feature_batch)
    latent_batch = latent_batch.to(dtype=unet.model.dtype)
    pred_latents = unet.model(latent_batch,
                              timesteps,
                              encoder_hidden_states=audio_feature_batch).sample
    vae.decode_latents(pred_latents)

def read_imgs(img_list):
    frames = []
    print(f'[{datetime.now()}] reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

def __mirror_index(size, index):
    #size = len(self.coord_list_cycle)
    turn = index // size
    res = index % size
    if turn % 2 == 0:
        return res
    else:
        return size - res - 1 

@torch.no_grad()
def inference(render_event,batch_size,input_latent_list_cycle,audio_feat_queue,audio_out_queue,res_frame_queue,
              vae, unet, pe,timesteps): #vae, unet, pe,timesteps
    
    # vae, unet, pe = load_diffusion_model()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # timesteps = torch.tensor([0], device=device)
    # pe = pe.half()
    # vae.vae = vae.vae.half()
    # unet.model = unet.model.half()
    
    length = len(input_latent_list_cycle)
    index = 0
    prev_was_silence = True  # 新增：追踪之前是否是靜音狀態
    count=0
    counttime=0
    print(f'[{datetime.now()}] start inference')
    while render_event.is_set():
        starttime=time.perf_counter()
        try:
            whisper_chunks = audio_feat_queue.get(block=True, timeout=1)
        except queue.Empty:
            continue
        is_all_silence=True
        audio_frames = []
        for _ in range(batch_size*2):
            frame,type,eventpoint = audio_out_queue.get()
            audio_frames.append((frame,type,eventpoint))
            if type==0:
                is_all_silence=False

        # 新增：檢測狀態變化
        if not is_all_silence and prev_was_silence:
            index = 0  # 從頭開始播放
            
        prev_was_silence = is_all_silence

        if is_all_silence:
            for i in range(batch_size):
                res_frame_queue.put((None,__mirror_index(length,index),audio_frames[i*2:i*2+2]))
                index = index + 1
        else:
            # print('infer=======')
            t=time.perf_counter()
            whisper_batch = np.stack(whisper_chunks)
            latent_batch = []
            for i in range(batch_size):
                idx = __mirror_index(length,index+i)
                latent = input_latent_list_cycle[idx]
                latent_batch.append(latent)
            latent_batch = torch.cat(latent_batch, dim=0)
            
            # for i, (whisper_batch,latent_batch) in enumerate(gen):
            audio_feature_batch = torch.from_numpy(whisper_batch)
            audio_feature_batch = audio_feature_batch.to(device=unet.device,
                                                            dtype=unet.model.dtype)
            audio_feature_batch = pe(audio_feature_batch)
            latent_batch = latent_batch.to(dtype=unet.model.dtype)
            # print('prepare time:',time.perf_counter()-t)
            # t=time.perf_counter()

            pred_latents = unet.model(latent_batch, 
                                        timesteps, 
                                        encoder_hidden_states=audio_feature_batch).sample
            # print('unet time:',time.perf_counter()-t)
            # t=time.perf_counter()
            recon = vae.decode_latents(pred_latents)
            # infer_inqueue.put((whisper_batch,latent_batch,sessionid))
            # recon,outsessionid = infer_outqueue.get()
            # if outsessionid != sessionid:
            #     print('outsessionid:',outsessionid,' mysessionid:',sessionid)

            # print('vae time:',time.perf_counter()-t)
            #print('diffusion len=',len(recon))
            counttime += (time.perf_counter() - t)
            count += batch_size
            #_totalframe += 1
            if count>=100:
                print(f'[{datetime.now()}] ------actual avg infer fps:{count/counttime:.4f}')
                count=0
                counttime=0
            for i,res_frame in enumerate(recon):
                #self.__pushmedia(res_frame,loop,audio_track,video_track)
                # res_frame_queue.put((res_frame,__mirror_index(length,index),audio_frames[i*2:i*2+2]))
                # index = index + 1frame_index = index + i
                frame_index = index + i
                # 檢查是否播放完畢
                # 不檢查 frame_index，持續送出 frame
                res_frame_queue.put((res_frame, frame_index, audio_frames[i*2:i*2+2]))
            index = index + batch_size
            #print('total batch time:',time.perf_counter()-starttime)            
    print(f'[{datetime.now()}] musereal inference processor stop')

class MuseReal(BaseReal):
    @torch.no_grad()
    def __init__(self, opt, model, avatar):
        super().__init__(opt)
        self.W = opt.W
        self.H = opt.H
        self.fps = opt.fps
        self.batch_size = opt.batch_size
        self.idx = 0
        
        # 初始化隊列
        self.res_frame_queue = mp.Queue(self.batch_size*2)

        # 載入默認虛擬人(從 .env 中的 AVATAR_ID)
        self.default_frames, self.default_masks, self.default_coords, \
        self.default_mask_coords, self.default_latents = avatar

        # 載入動作影片資源(avator_8)並保留其遮罩資訊
        action_frames, action_masks, action_coords, \
        action_mask_coords, action_latents = load_avatar('avator_8')
        self.action_frames = action_frames
        self.action_masks = action_masks
        self.action_coords = action_coords
        self.action_mask_coords = action_mask_coords
        
        # 初始設定使用默認資源
        self.frames = self.default_frames
        self.masks = self.default_masks
        self.coords = self.default_coords
        self.mask_coords = self.default_mask_coords
        self.input_latents = self.default_latents
        
        self.vae, self.unet, self.pe, self.timesteps, self.audio_processor = model
        self.asr = MuseASR(opt,self,self.audio_processor)
        self.asr.warm_up()
        
        self.render_event = mp.Event()
        self.render_thread = None
        
    def init_avatar(self, avatar):
        """初始化或重新初始化虛擬人相關資源"""
        # Set switch_completed to False at start of initialization
        self.switch_completed.clear()
        
        # 保存說話時使用的虛擬人資源
        self.default_avatar = avatar
        self.frame_list_cycle, self.mask_list_cycle, self.coord_list_cycle, \
        self.mask_coords_list_cycle, self.input_latent_list_cycle = avatar
        
        # 只保留說話虛擬人的配置
        self.idx = 0
        
        # Clear and recreate queue to avoid any stale frames
        if hasattr(self, 'res_frame_queue'):
            while not self.res_frame_queue.empty():
                try:
                    self.res_frame_queue.get_nowait()
                except:
                    pass
        self.res_frame_queue = mp.Queue(self.batch_size*2)
        
        # Signal that switch is complete
        self.switch_completed.set()

    def __del__(self):
        print(f'[{datetime.now()}] musereal({self.sessionid}) delete')
    

    def __mirror_index(self, index):
        size = len(self.coord_list_cycle)
        turn = index // size
        res = index % size
        if turn % 2 == 0:
            return res
        else:
            return size - res - 1  

    def __warm_up(self): 
        self.asr.run_step()
        whisper_chunks = self.asr.get_next_feat()
        whisper_batch = np.stack(whisper_chunks)
        latent_batch = []
        for i in range(self.batch_size):
            idx = self.__mirror_index(self.idx+i)
            latent = self.input_latent_list_cycle[idx]
            latent_batch.append(latent)
        latent_batch = torch.cat(latent_batch, dim=0)
        print(f'[{datetime.now()}] infer=======')
        # for i, (whisper_batch,latent_batch) in enumerate(gen):
        audio_feature_batch = torch.from_numpy(whisper_batch)
        audio_feature_batch = audio_feature_batch.to(device=self.unet.device,
                                                        dtype=self.unet.model.dtype)
        audio_feature_batch = self.pe(audio_feature_batch)
        latent_batch = latent_batch.to(dtype=self.unet.model.dtype)

        pred_latents = self.unet.model(latent_batch, 
                                    self.timesteps, 
                                    encoder_hidden_states=audio_feature_batch).sample
        recon = self.vae.decode_latents(pred_latents)
      
    
    def process_frames(self,quit_event,loop=None,audio_track=None,video_track=None):
        last_speaking = False
        action_length = len(self.action_frames)
        default_length = len(self.default_frames)
        action_index = 0
        default_index = 0
        in_action = False
        is_audio_ended = False
        
        while not quit_event.is_set():
            try:
                res_frame,idx,audio_frames = self.res_frame_queue.get(block=True, timeout=1)
                has_audio = any(frame[1] == 0 for frame in audio_frames)

                if last_speaking and not has_audio:
                    is_audio_ended = True
                
                if has_audio and not last_speaking:
                    action_index = 0
                    in_action = True
                    is_audio_ended = False

                if in_action:
                    if action_index < action_length:
                        # 使用動作影片的資源進行口型合成
                        ori_frame = self.action_frames[action_index]
                        bbox = self.action_coords[action_index]
                        mask = self.action_masks[action_index]
                        mask_crop_box = self.action_mask_coords[action_index]

                        if has_audio and res_frame is not None:
                            x1, y1, x2, y2 = bbox
                            try:
                                res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
                                combine_frame = get_image_blending(ori_frame,res_frame,bbox,mask,mask_crop_box)
                            except Exception as e:
                                print(f'[{datetime.now()}] Error resizing frame: {e}')
                                combine_frame = ori_frame
                        else:
                            combine_frame = ori_frame
                        
                        action_index += 1
                        if action_index >= action_length:
                            in_action = False  # 動作播完後切換到默認
                    else:
                        in_action = False
                else:
                    # 使用默認影片資源進行口型合成
                    ori_frame = self.default_frames[default_index]
                    bbox = self.default_coords[default_index]
                    mask = self.default_masks[default_index]
                    mask_crop_box = self.default_mask_coords[default_index]
                    
                    if has_audio and res_frame is not None:
                        x1, y1, x2, y2 = bbox
                        try:
                            res_frame = cv2.resize(res_frame.astype(np.uint8),(x2-x1,y2-y1))
                            combine_frame = get_image_blending(ori_frame,res_frame,bbox,mask,mask_crop_box)
                        except Exception as e:
                            print(f'[{datetime.now()}] Error resizing frame: {e}')
                            combine_frame = ori_frame
                    else:
                        combine_frame = ori_frame
                    
                    default_index = (default_index + 1) % default_length
                
                last_speaking = has_audio

                # 輸出處理
                image = combine_frame
                new_frame = VideoFrame.from_ndarray(image, format="bgr24")
                asyncio.run_coroutine_threadsafe(video_track._queue.put((new_frame,None)), loop)
                self.record_video_data(image)

                for audio_frame in audio_frames:
                    frame,type,eventpoint = audio_frame
                    frame = (frame * 32767).astype(np.int16)
                    new_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
                    new_frame.planes[0].update(frame.tobytes())
                    new_frame.sample_rate=16000
                    asyncio.run_coroutine_threadsafe(audio_track._queue.put((new_frame,eventpoint)), loop)
                    self.record_audio_data(frame)
                    
            except Exception as e:
                print(f'[{datetime.now()}] Error processing frame: {e}')
                continue
            
            
    def render(self,quit_event,loop=None,audio_track=None,video_track=None):
        self.tts.render(quit_event)
        self.init_customindex()
        process_thread = Thread(target=self.process_frames, args=(quit_event,loop,audio_track,video_track))
        process_thread.start()

        self.render_event.set()
        self.render_thread = Thread(target=inference, args=(
            self.render_event,
            self.batch_size,
            self.default_latents,
            self.asr.feat_queue,
            self.asr.output_queue,
            self.res_frame_queue,
            self.vae, self.unet, self.pe,self.timesteps
        ))
        self.render_thread.start()
        count=0
        totaltime=0
        _starttime=time.perf_counter()
        #_totalframe=0
        while not quit_event.is_set():
            t = time.perf_counter()
            self.asr.run_step()
            
            if video_track._queue.qsize()>=1.5*self.opt.batch_size:
                print(f'[{datetime.now()}] sleep qsize=',video_track._queue.qsize())
                time.sleep(0.04*video_track._queue.qsize()*0.8)
                
        self.render_event.clear()
        if self.render_thread:
            self.render_thread.join(timeout=1)
        print(f'[{datetime.now()}] musereal thread stop')

    def reinit_render(self, avatar):
        """重新初始化渲染相關資源"""
        print(f'[{datetime.now()}] Reinitializing avatar...')
        
        # Set pause flag before stopping render
        self.pause_processing.set()
        
        # 停止當前渲染
        self.render_event.clear()
        if self.render_thread:
            self.render_thread.join(timeout=1)
            
        try:
            # 重新初始化avatar相關資源
            self.init_avatar(avatar)
            
            # 重新啟動渲染
            self.render_event.set()
            self.render_thread = Thread(target=inference, args=(
                self.render_event,
                self.batch_size,
                self.input_latent_list_cycle,
                self.asr.feat_queue,
                self.asr.output_queue,
                self.res_frame_queue,
                self.vae, self.unet, self.pe, self.timesteps
            ))
            self.render_thread.start()
            
        finally:
            # Ensure processing is resumed even if initialization fails
            self.pause_processing.clear()
