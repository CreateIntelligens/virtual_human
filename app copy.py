# server.py - Part 1: Imports and Basic Configuration
from flask import Flask, render_template,send_from_directory,request, jsonify
from flask_sockets import Sockets
from logger import CustomLogger
import base64
import time
import json
import os
import re
import numpy as np
from threading import Thread,Event
import torch.multiprocessing as mp

from aiohttp import web
import aiohttp
import aiohttp_cors
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.rtcrtpsender import RTCRtpSender
from webrtc import HumanPlayer

import argparse
import random
import shutil
import asyncio
import torch
from typing import Optional, Dict

# 初始化日誌系統
logger = CustomLogger('livetalking', 'app')

app = Flask(__name__)
nerfreals = {}
nerfreal_ready = {}  # 追踪每個 session 的 nerfreal 實例是否準備好
opt = None
model = None
avatar = None

# 全局變數追踪當前狀態
current_avatar_id = None  # 追踪當前虛擬人
current_voice_type = None  # 追踪當前聲音

# 全局配置
CONFIGS = {
    'avatars': {}  # 虛擬人配置將在初始化時填充
}

def set_nerfreal_ready(sessionid: int):
    """標記 nerfreal 實例為準備就緒"""
    if sessionid not in nerfreal_ready:
        nerfreal_ready[sessionid] = asyncio.Event()
    nerfreal_ready[sessionid].set()

async def wait_nerfreal_ready(sessionid: int, timeout: float = 5.0) -> bool:
    """等待 nerfreal 實例準備就緒"""
    if sessionid not in nerfreal_ready:
        nerfreal_ready[sessionid] = asyncio.Event()
    try:
        await asyncio.wait_for(nerfreal_ready[sessionid].wait(), timeout)
        return True
    except asyncio.TimeoutError:
        return False
# server.py - Part 2: Helper Functions

def set_current_avatar(avatar_id):
    global current_avatar_id
    current_avatar_id = avatar_id
    logger.info(f"Current avatar set to: {avatar_id}")

def set_current_voice(voice_type):
    global current_voice_type
    current_voice_type = voice_type
    logger.info(f"Current voice type set to: {voice_type}")

def load_voice_configs():
    """從 avatar_voice.json 讀取語音配置"""
    try:
        with open('avatar_voice.json', 'r', encoding='utf-8') as f:
            voice_data = json.load(f)
            voice_configs = {}
            for voice_key, voice_info in voice_data.items():
                voice_configs[voice_info['id']] = voice_info['config']
            return voice_configs
    except Exception as e:
        logger.error(f"Could not load avatar_voice.json: {e}")
        return {}

def get_available_voices():
    """獲取可用的語音配置"""
    try:
        with open('avatar_voice.json', 'r', encoding='utf-8') as f:
            voices = json.load(f)
            # Sort voices by ID to ensure consistent order
            sorted_voices = dict(sorted(voices.items(), key=lambda x: x[1]['id']))
            return sorted_voices
    except Exception as e:
        logger.error(f"Could not load avatar_voice.json: {e}")
        return {}

def get_available_avatars(avatar_id, avatar_list=None):
    """動態返回可用的虛擬人配置"""
    avatar_configs = {}
    
    try:
        with open('avatar_name.json', 'r', encoding='utf-8') as f:
            avatar_names = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load avatar_name.json: {e}")  # 保留原有print
        logger.warning(f"Could not load avatar_name.json: {e}")
        avatar_names = {}
    
    if not os.path.exists(os.path.join('data/avatars', avatar_id, 'full_imgs')):
        error_msg = f"Initial avatar {avatar_id} not found"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if avatar_list:
        avatars = avatar_list.split(',')
        if avatar_id not in avatars:
            avatars.insert(0, avatar_id)
            logger.info(f"Added initial avatar {avatar_id} to list")
            
        for aid in avatars:
            if os.path.exists(os.path.join('data/avatars', aid, 'full_imgs')):
                avatar_configs[aid] = {
                    'id': aid,
                    'display_name': avatar_names.get(aid, {}).get('name', f'Avatar {aid}')
                }
                logger.info(f"Added avatar {aid} to configuration")
            else:
                print(f"Warning: Avatar {aid} not found, skipping...")  # 保留原有print
                logger.warning(f"Avatar {aid} not found, skipping...")
    else:
        avatar_configs[avatar_id] = {
            'id': avatar_id,
            'display_name': avatar_names.get(avatar_id, {}).get('name', f'Avatar {avatar_id}')
        }
        logger.info(f"Using single avatar: {avatar_id}")
    
    return avatar_configs

def randN(N):
    '''生成长度为 N的随机数 '''
    min = pow(10, N - 1)
    max = pow(10, N)
    return random.randint(min, max - 1)
# server.py - Part 3: WebRTC and LLM Core Functions

def build_nerfreal(sessionid):
    global current_voice_type
    opt.sessionid = sessionid
    logger.info(f"Building nerfreal for session {sessionid}")
    
    # 創建 nerfreal 實例
    if opt.model == 'wav2lip':
        from lipreal import LipReal
        nerfreal = LipReal(opt,model,avatar)
        logger.info(f"Created LipReal instance for session {sessionid}")
    elif opt.model == 'musetalk':
        from musereal import MuseReal
        nerfreal = MuseReal(opt,model,avatar)
        logger.info(f"Created MuseReal instance for session {sessionid}")
    elif opt.model == 'ernerf':
        from nerfreal import NeRFReal
        nerfreal = NeRFReal(opt,model,avatar)
        logger.info(f"Created NeRFReal instance for session {sessionid}")
    elif opt.model == 'ultralight':
        from lightreal import LightReal
        nerfreal = LightReal(opt,model,avatar)
        logger.info(f"Created LightReal instance for session {sessionid}")
    
    # 獲取當前語音設定
    actual_voice_type = current_voice_type if current_voice_type is not None else 1
    voice_configs = load_voice_configs()
    selected_config = voice_configs.get(actual_voice_type, {})
    
    # 設置 TTS 服務器：0 是 Lula (default)，1 是 Sweet (test)
    new_server = 'http://52.69.235.212:9885/test' if actual_voice_type == 1 else 'http://52.69.235.212:9885/'
    nerfreal.opt.TTS_SERVER = new_server
    
    # 確保設置正確的 TTS 請求配置
    if hasattr(nerfreal.tts, 'default_req'):
        nerfreal.tts.default_req = selected_config.copy()
        print(f"Setting TTS config for voice type {actual_voice_type}")  # 保留原有print
        logger.info(f"Set TTS config for voice type {actual_voice_type}", sessionid)
    
    # 標記 nerfreal 實例為準備就緒
    set_nerfreal_ready(sessionid)
    
    return nerfreal

def llm_response(message, nerfreal):
    start = time.perf_counter()
    logger.info(f"Starting LLM response for message: {message[:50]}...", nerfreal.opt.sessionid)
    
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    end = time.perf_counter()
    print(f"llm Time init: {end-start}s")  # 保留原有print
    logger.info(f"llm Time init: {end-start}s", nerfreal.opt.sessionid)
    
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[{'role': 'system', 'content': 'You are a helpful assistant.'},
                  {'role': 'user', 'content': message}],
        stream=True,
        stream_options={"include_usage": True}
    )
    result=""
    first = True
    
    for chunk in completion:
        if len(chunk.choices)>0:
            if first:
                end = time.perf_counter()
                print(f"llm Time to first chunk: {end-start}s")  # 保留原有print
                logger.info(f"llm Time to first chunk: {end-start}s", nerfreal.opt.sessionid)
                first = False
            
            msg = chunk.choices[0].delta.content
            lastpos=0
            for i, char in enumerate(msg):
                if char in ",.!;:，。！？：；" :
                    result = result+msg[lastpos:i+1]
                    lastpos = i+1
                    if len(result)>10:
                        print(result)  # 保留原有print
                        logger.info(f"Speaking: {result}", nerfreal.opt.sessionid)
                        nerfreal.put_msg_txt(result)
                        result=""
            result = result+msg[lastpos:]
    end = time.perf_counter()
    print(f"llm Time to last chunk: {end-start}s")  # 保留原有print
    logger.info(f"llm Time to last chunk: {end-start}s", nerfreal.opt.sessionid)
    
    if result:  # 處理最後剩餘的文本
        logger.info(f"Speaking final chunk: {result}", nerfreal.opt.sessionid)
        nerfreal.put_msg_txt(result)

#####webrtc###############################
pcs = set()
# server.py - Part 4: Request Handlers

async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    logger.info("Received WebRTC offer")

    if len(nerfreals) >= opt.max_session:
        print('reach max session')  # 保留原有print
        logger.warning("Maximum session limit reached")
        return -1
        
    sessionid = randN(6)
    print('sessionid=',sessionid)  # 保留原有print
    logger.info(f"Created new session with ID: {sessionid}")
    
    nerfreals[sessionid] = None
    nerfreal = await asyncio.get_event_loop().run_in_executor(None, build_nerfreal,sessionid)
    nerfreals[sessionid] = nerfreal
    logger.info(f"Initialized nerfreal for session {sessionid}")
    
    pc = RTCPeerConnection()
    pcs.add(pc)
    logger.info(f"Created new RTCPeerConnection for session {sessionid}")

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("Connection state is %s" % pc.connectionState)  # 保留原有print
        logger.info(f"Connection state changed to: {pc.connectionState}", sessionid)
        
        if pc.connectionState == "failed":
            logger.error("WebRTC connection failed", sessionid)
            await pc.close()
            pcs.discard(pc)
            del nerfreals[sessionid]
            
        if pc.connectionState == "closed":
            print("Connection closed, cleaning up...")  # 保留原有print
            logger.info("Connection closed, cleaning up...", sessionid)
            pcs.discard(pc)
            del nerfreals[sessionid]
            # 重新執行程序
            import sys, os
            restart_cmd = f"{sys.executable} {' '.join(sys.argv)}"
            logger.info(f"Restarting with command: {restart_cmd}", sessionid)
            os.execv(sys.executable, [sys.executable] + sys.argv)

    player = HumanPlayer(nerfreals[sessionid])
    audio_sender = pc.addTrack(player.audio)
    video_sender = pc.addTrack(player.video)
    logger.info("Added audio and video tracks", sessionid)

    capabilities = RTCRtpSender.getCapabilities("video")
    preferences = list(filter(lambda x: x.name == "H264", capabilities.codecs))
    preferences += list(filter(lambda x: x.name == "VP8", capabilities.codecs))
    preferences += list(filter(lambda x: x.name == "rtx", capabilities.codecs))
    transceiver = pc.getTransceivers()[1]
    transceiver.setCodecPreferences(preferences)
    logger.info("Set video codec preferences", sessionid)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    logger.info("Created and set local description", sessionid)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type, "sessionid":sessionid}
        ),
    )

async def human(request):
    params = await request.json()
    sessionid = params.get('sessionid', 0)
    logger.info(f"Received human request for session {sessionid}", sessionid)

    if params.get('interrupt'):
        logger.info("Interrupting current talk", sessionid)
        nerfreals[sessionid].flush_talk()

    if params['type'] == 'echo':
        logger.info(f"Echo mode: {params['text']}", sessionid)
        nerfreals[sessionid].put_msg_txt(params['text'])
    elif params['type'] == 'chat':
        logger.info(f"Chat mode: {params['text']}", sessionid)
        try:
            res = await asyncio.get_event_loop().run_in_executor(
                None, 
                llm_response, 
                params['text'],
                nerfreals[sessionid]
            )
            logger.info("Chat response completed successfully", sessionid)
        except Exception as e:
            logger.error(f"Chat response failed: {str(e)}", sessionid)
            raise

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"code": 0, "data": "ok"}
        ),
    )

async def humanaudio(request):
    try:
        form = await request.post()
        sessionid = int(form.get('sessionid', 0))
        fileobj = form["file"]
        filename = fileobj.filename
        filebytes = fileobj.file.read()
        
        logger.info(f"Received audio file: {filename} for session {sessionid}", sessionid)
        nerfreals[sessionid].put_audio_file(filebytes)
        logger.info("Audio file processed successfully", sessionid)

        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": 0, "msg": "ok"}
            ),
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error processing audio file: {error_msg}", sessionid)
        return web.Response(
            content_type="application/json",
            text=json.dumps(
                {"code": -1, "msg": "err", "data": error_msg}
            ),
        )

async def set_audiotype(request):
    params = await request.json()
    sessionid = params.get('sessionid', 0)
    logger.info(f"Setting audio type for session {sessionid}", sessionid)
    
    nerfreals[sessionid].set_curr_state(params['audiotype'], params['reinit'])
    logger.info(f"Audio type set to {params['audiotype']}, reinit: {params['reinit']}", sessionid)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"code": 0, "data": "ok"}
        ),
    )

async def record(request):
    params = await request.json()
    sessionid = params.get('sessionid', 0)
    
    if params['type'] == 'start_record':
        logger.info("Starting recording", sessionid)
        nerfreals[sessionid].start_recording()
    elif params['type'] == 'end_record':
        logger.info("Stopping recording", sessionid)
        nerfreals[sessionid].stop_recording()
        
    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"code": 0, "data": "ok"}
        ),
    )

async def is_speaking(request):
    params = await request.json()
    sessionid = params.get('sessionid', 0)
    
    # 等待 nerfreal 實例準備就緒，最多等待 5 秒
    if not await wait_nerfreal_ready(sessionid, timeout=5.0):
        logger.warning(f"Nerfreal instance not ready for session {sessionid}", sessionid)
        return web.Response(
            content_type="application/json",
            text=json.dumps({
                "code": 0,
                "data": False  # 如果實例未準備好，返回非說話狀態
            })
        )
    
    speaking_status = nerfreals[sessionid].is_speaking()
    # logger.debug(f"Speaking status checked: {speaking_status}", sessionid)
    
    return web.Response(
        content_type="application/json",
        text=json.dumps({
            "code": 0,
            "data": speaking_status
        }),
    )

async def health_check(request):
    """健康檢查端點，用於 Docker 容器的健康監控"""
    try:
        # 最多嘗試 3 次檢查 SRS 服務
        max_retries = 3
        srs_status = "unknown"
        
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get('http://srs:1985/api/v1/versions', timeout=2) as resp:
                        if resp.status == 200:
                            srs_status = "up"
                            break
                        else:
                            srs_status = "down"
            except:
                if attempt < max_retries - 1:
                    await asyncio.sleep(5)  # 等待 1 秒後重試
                    continue
                srs_status = "down"
                
        status = {
            "app": "healthy",
            "srs": srs_status,
            "sessions": len(nerfreals),
            "connections": len(pcs)
        }
        
        # 只有當 SRS 可用時返回 200
        return web.Response(
            status=200 if srs_status == "up" else 500,
            text=json.dumps(status),
            content_type='application/json'
        )
    except Exception as e:
        error_status = {
            "app": "error",
            "error": str(e)
        }
        logger.error(f"Health check failed: {e}")
        return web.Response(
            status=500,
            text=json.dumps(error_status),
            content_type='application/json'
        )


# server.py - Part 5: Avatar and TTS Management

async def switch_tts_endpoint(request):
    params = await request.json()
    sessionid = params.get('sessionid', 0)
    config_type = params.get('config_type', 0)
    
    logger.info(f"Switching TTS endpoint for session {sessionid} to config type {config_type}", sessionid)
    
    # 等待 nerfreal 實例準備就緒
    if not await wait_nerfreal_ready(sessionid, timeout=5.0):
        logger.error(f"Nerfreal instance not ready for session {sessionid}", sessionid)
        return web.Response(
            content_type="application/json",
            text=json.dumps({
                "code": -1,
                "message": "Nerfreal instance not ready"
            })
        )
    
    nerfreal = nerfreals[sessionid]
    voice_configs = load_voice_configs()
    selected_config = voice_configs.get(config_type, {})
    
    # 設置 TTS 服務器：0 是 Lula (default)，1 是 Sweet (test)
    new_server = 'http://52.69.235.212:9885/test' if config_type == 1 else 'http://52.69.235.212:9885/'
    nerfreal.opt.TTS_SERVER = new_server
    logger.info(f"TTS server set to: {new_server}", sessionid)
    
    if hasattr(nerfreal.tts, 'default_req'):
        nerfreal.tts.default_req = selected_config.copy()
        logger.info("Updated TTS configuration", sessionid)
    
    nerfreal.flush_talk()
    nerfreal.tts.msgqueue.queue.clear()
    logger.info("Cleared message queue", sessionid)
    
    # 更新當前聲音
    set_current_voice(config_type)
    
    # 取得語音訊息
    voices = get_available_voices()
    voice_info = next((v for v in voices.values() if v['id'] == config_type), None)
    voice_name = voice_info['name'] if voice_info else f"Voice {config_type}"
    
    return web.Response(
        content_type="application/json",
        text=json.dumps({
            "code": 0, 
            "data": "ok",
            "config_type": config_type,
            "endpoint": nerfreal.opt.TTS_SERVER,
            "voice_name": voice_name,
            "req": selected_config
        }),
    )

async def switch_avatar(request):
    params = await request.json()
    sessionid = params.get('sessionid', 0)
    avatar_id = params.get('avatar_id', 'avator_2')
    
    logger.info(f"Switching avatar for session {sessionid} to {avatar_id}", sessionid)
    
    # 等待 nerfreal 實例準備就緒
    if not await wait_nerfreal_ready(sessionid, timeout=5.0):
        logger.error(f"Nerfreal instance not ready for session {sessionid}", sessionid)
        return web.Response(
            content_type="application/json",
            text=json.dumps({
                "code": -1,
                "message": "Nerfreal instance not ready"
            })
        )
    
    if avatar_id not in CONFIGS['avatars']:
        logger.error(f"Invalid avatar_id: {avatar_id}", sessionid)
        return web.Response(
            content_type="application/json",
            text=json.dumps({
                "code": -1,
                "message": "Invalid avatar_id"
            })
        )
    
    nerfreal = nerfreals[sessionid]
    
    try:
        avatar_path = os.path.join('data/avatars', avatar_id, 'full_imgs')
        if not os.path.exists(avatar_path):
            error_msg = f"Avatar resources not found in {avatar_path}"
            logger.error(error_msg, sessionid)
            raise Exception(error_msg)

        print(f"Loading avatar: {avatar_id} from {avatar_path}")  # 保留原有print
        logger.info(f"Loading avatar: {avatar_id} from {avatar_path}", sessionid)
        
        new_avatar = await asyncio.get_event_loop().run_in_executor(None, load_avatar, avatar_id)
        await asyncio.get_event_loop().run_in_executor(None, nerfreal.reinit_render, new_avatar)
        logger.info("Avatar loaded and renderer reinitialized", sessionid)
        
        # 更新當前虛擬人
        set_current_avatar(avatar_id)
        
        return web.Response(
            content_type="application/json",
            text=json.dumps({
                "code": 0,
                "data": "ok",
                "avatar_id": avatar_id,
                "display_name": CONFIGS['avatars'][avatar_id]['display_name']
            })
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Failed to switch avatar: {error_msg}", sessionid)
        return web.Response(
            content_type="application/json",
            text=json.dumps({
                "code": -1,
                "message": f"Failed to switch avatar: {error_msg}"
            })
        )

async def get_avatars(request):
    """获取可用的虚拟人列表"""
    logger.info("Getting available avatars list")
    return web.Response(
        content_type="application/json",
        text=json.dumps({
            "code": 0,
            "data": CONFIGS['avatars']
        })
    )

async def get_voices(request):
    """获取可用的声音列表"""
    logger.info("Getting available voices list")
    voices = get_available_voices()
    return web.Response(
        content_type="application/json",
        text=json.dumps({
            "code": 0,
            "data": voices
        })
    )

async def get_current_config(request):
    """获取当前的虚拟人和声音配置"""
    global current_avatar_id, current_voice_type
    params = await request.json()
    sessionid = params.get('sessionid', 0)
    
    # 如果還沒有切換過，使用初始值
    actual_avatar_id = current_avatar_id if current_avatar_id else opt.avatar_id
    actual_voice_type = current_voice_type if current_voice_type is not None else 1
    
    logger.info(f"Current config - Avatar: {actual_avatar_id}, Voice: {actual_voice_type}", sessionid)
    
    return web.Response(
        content_type="application/json",
        text=json.dumps({
            "code": 0,
            "data": {
                "current_avatar_id": actual_avatar_id,
                "current_voice_type": actual_voice_type
            }
        })
    )
# server.py - Part 6: Server Setup and Main

import signal

def handle_sigterm(signum, frame):
    """處理 SIGTERM 信號，實現優雅退出"""
    logger.info("Received SIGTERM signal, initiating graceful shutdown...")
    # 清理資源
    for pc in pcs:
        try:
            asyncio.run(pc.close())
        except:
            pass
    pcs.clear()
    # 清理其他資源
    for session_id in list(nerfreals.keys()):
        try:
            del nerfreals[session_id]
        except:
            pass
    logger.info("Cleanup completed, exiting...")
    sys.exit(0)

async def on_shutdown(app):
    logger.info("Shutting down application")
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()
    logger.info("All peer connections closed")

async def check_srs_availability(session):
    """檢查 SRS 服務是否可用"""
    try:
        async with session.get("http://srs:1985/api/v1/versions") as resp:
            return resp.status == 200
    except:
        return False

async def post(url, data, max_retries=3, retry_delay=2):
    """發送請求到 SRS 服務，帶有重試機制"""
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                # 檢查 SRS 服務可用性
                if not await check_srs_availability(session):
                    logger.warning(f"SRS not ready, attempt {attempt + 1}/{max_retries}")
                    if attempt == max_retries - 1:
                        raise Exception("SRS service not available")
                    await asyncio.sleep(retry_delay)
                    continue

                # 執行實際的 post 請求
                async with session.post(url, data=data) as response:
                    result = await response.text()
                    if not result:  # 檢查空響應
                        raise Exception("Empty response from SRS")
                    return result

        except Exception as e:
            error_msg = str(e)
            logger.error(f'Request error (attempt {attempt + 1}/{max_retries}): {error_msg}')
            print(f'Error: {e}')  # 保留原有print
            
            if attempt == max_retries - 1:
                raise
            
            await asyncio.sleep(retry_delay)
            logger.info(f"Retrying connection... ({attempt + 1}/{max_retries})")

async def run(push_url, sessionid):
    nerfreal = await asyncio.get_event_loop().run_in_executor(None, build_nerfreal, sessionid)
    nerfreals[sessionid] = nerfreal

    pc = RTCPeerConnection()
    pcs.add(pc)
    logger.info(f"Created RTCPeerConnection for push URL session {sessionid}")

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print("Connection state is %s" % pc.connectionState)  # 保留原有print
        logger.info(f"Connection state changed to: {pc.connectionState}", sessionid)
        
        if pc.connectionState == "failed":
            logger.error("WebRTC connection failed", sessionid)
            await pc.close()
            pcs.discard(pc)
            del nerfreals[sessionid]
            
        if pc.connectionState == "closed":
            print("Connection closed, cleaning up...")  # 保留原有print
            logger.info("Connection closed, cleaning up...", sessionid)
            pcs.discard(pc)
            del nerfreals[sessionid]
            # 重新執行程序
            import sys, os
            restart_cmd = f"{sys.executable} {' '.join(sys.argv)}"
            logger.info(f"Restarting with command: {restart_cmd}", sessionid)
            os.execv(sys.executable, [sys.executable] + sys.argv)

    player = HumanPlayer(nerfreals[sessionid])
    audio_sender = pc.addTrack(player.audio)
    video_sender = pc.addTrack(player.video)
    logger.info("Added audio and video tracks", sessionid)

    await pc.setLocalDescription(await pc.createOffer())
    
    try:
        answer = await post(push_url, pc.localDescription.sdp)
        if answer:  # 確保收到有效回應
            await pc.setRemoteDescription(RTCSessionDescription(sdp=answer, type='answer'))
            logger.info("WebRTC connection established", sessionid)
        else:
            raise Exception("Failed to get valid SDP answer from SRS")
    except Exception as e:
        logger.error(f"Failed to establish WebRTC connection: {e}", sessionid)
        await pc.close()
        pcs.discard(pc)
        del nerfreals[sessionid]
        raise


if __name__ == '__main__':
    # 註冊信號處理程序
    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)
    
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    
    # Training options
    parser.add_argument('--pose', type=str, default="data/data_kf.json", help="transforms.json, pose source")
    parser.add_argument('--au', type=str, default="data/au.csv", help="eye blink area")
    parser.add_argument('--torso_imgs', type=str, default="", help="torso images path")
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --exp_eye")
    parser.add_argument('--data_range', type=int, nargs='*', default=[0, -1], help="data range to use")
    parser.add_argument('--workspace', type=str, default='data/video')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ckpt', type=str, default='data/pretrained/ngp_kf.pth')
    parser.add_argument('--num_rays', type=int, default=4096 * 16)
    parser.add_argument('--cuda_ray', action='store_true')
    parser.add_argument('--max_steps', type=int, default=16)
    parser.add_argument('--num_steps', type=int, default=16)
    parser.add_argument('--upsample_steps', type=int, default=0)
    parser.add_argument('--update_extra_interval', type=int, default=16)
    parser.add_argument('--max_ray_batch', type=int, default=4096)
    parser.add_argument('--fp16', action='store_true')
    
    # Network backbone options
    parser.add_argument('--bg_img', type=str, default='white', help="background image")
    parser.add_argument('--fbg', action='store_true', help="frame-wise bg")
    parser.add_argument('--exp_eye', action='store_true', help="explicitly control the eyes")
    parser.add_argument('--fix_eye', type=float, default=-1, help="fixed eye area, negative to disable, set to 0-0.3 for a reasonable eye")
    parser.add_argument('--smooth_eye', action='store_true', help="smooth the eye area sequence")
    parser.add_argument('--torso_shrink', type=float, default=0.8, help="shrink bg coords to allow more flexibility in deform")

    # Dataset options
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', type=int, default=0, help="0 means load data from disk on-the-fly, 1 means preload to CPU, 2 means GPU.")
    parser.add_argument('--bound', type=float, default=1, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=4, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--dt_gamma', type=float, default=1/256, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.05, help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied (sigma)")
    parser.add_argument('--density_thresh_torso', type=float, default=0.01, help="threshold for density grid to be occupied (alpha)")
    parser.add_argument('--patch_size', type=int, default=1, help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")

    # Lips and torso options
    parser.add_argument('--init_lips', action='store_true', help="init lips region")
    parser.add_argument('--finetune_lips', action='store_true', help="use LPIPS and landmarks to fine tune lips region")
    parser.add_argument('--smooth_lips', action='store_true', help="smooth the enc_a in a exponential decay way...")
    parser.add_argument('--torso', action='store_true', help="fix head and train torso")
    parser.add_argument('--head_ckpt', type=str, default='', help="head model")

    # GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=450, help="GUI width")
    parser.add_argument('--H', type=int, default=450, help="GUI height")
    parser.add_argument('--radius', type=float, default=3.35, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=21.24, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=1, help="GUI rendering max sample per pixel")

    # Loss and features
    parser.add_argument('--warmup_step', type=int, default=10000, help="warm up steps")
    parser.add_argument('--amb_aud_loss', type=int, default=1, help="use ambient aud loss")
    parser.add_argument('--amb_eye_loss', type=int, default=1, help="use ambient eye loss")
    parser.add_argument('--unc_loss', type=int, default=1, help="use uncertainty loss")
    parser.add_argument('--lambda_amb', type=float, default=1e-4, help="lambda for ambient loss")
    parser.add_argument('--amb_dim', type=int, default=2, help="ambient dimension")
    parser.add_argument('--part', action='store_true', help="use partial training data (1/10)")
    parser.add_argument('--part2', action='store_true', help="use partial training data (first 15s)")

    # Camera options
    parser.add_argument('--train_camera', action='store_true', help="optimize camera pose")
    parser.add_argument('--smooth_path', action='store_true', help="brute-force smooth camera pose trajectory with a window size")
    parser.add_argument('--smooth_path_window', type=int, default=7, help="smoothing window size")

    # Audio attention options
    parser.add_argument('--att', type=int, default=2, help="audio attention mode (0 = turn off, 1 = left-direction, 2 = bi-direction)")
    parser.add_argument('--aud', type=str, default='', help="audio source (empty will load the default, else should be a path to a npy file)")
    parser.add_argument('--emb', action='store_true', help="use audio class + embedding instead of logits")
    parser.add_argument('--ind_dim', type=int, default=4, help="individual code dim, 0 to turn off")
    parser.add_argument('--ind_num', type=int, default=10000, help="number of individual codes, should be larger than training dataset size")
    parser.add_argument('--ind_dim_torso', type=int, default=8, help="individual code dim, 0 to turn off")

    # ASR options
    parser.add_argument('--asr', action='store_true', help="load asr for real-time app")
    parser.add_argument('--asr_wav', type=str, default='', help="load the wav and use as input")
    parser.add_argument('--asr_play', action='store_true', help="play out the audio")
    parser.add_argument('--asr_model', type=str, default='cpierse/wav2vec2-large-xlsr-53-esperanto')
    parser.add_argument('--asr_save_feats', action='store_true')

    # Fullbody options
    parser.add_argument('--fullbody', action='store_true', help="fullbody human")
    parser.add_argument('--fullbody_img', type=str, default='data/fullbody/img')
    parser.add_argument('--fullbody_width', type=int, default=580)
    parser.add_argument('--fullbody_height', type=int, default=1080)
    parser.add_argument('--fullbody_offset_x', type=int, default=0)
    parser.add_argument('--fullbody_offset_y', type=int, default=0)
    parser.add_argument('--bbox_shift', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)

    # Configuration and virtual human options
    parser.add_argument('--customvideo_config', type=str, default='')
    parser.add_argument('--avatar_id', type=str, default='avator_1')
    parser.add_argument('--avatar_list', type=str, default=None, help="Comma-separated list of avatar_ids")
    
    # TTS options
    parser.add_argument('--tts', type=str, default='edgetts')
    parser.add_argument('--REF_FILE', type=str, default=None)
    parser.add_argument('--REF_TEXT', type=str, default=None)
    parser.add_argument('--TTS_SERVER', type=str, default='http://127.0.0.1:9880')

    # Model and transport options
    parser.add_argument('--model', type=str, default='ernerf')
    parser.add_argument('--transport', type=str, default='rtcpush')
    parser.add_argument('--push_url', type=str, default='http://srs:1985/rtc/v1/whip/?app=live&stream=livestream')  # 修改為使用服務名稱
    parser.add_argument('--max_session', type=int, default=1)
    parser.add_argument('--listenport', type=int, default=8010)

    # Audio FPS and window options
    parser.add_argument('--fps', type=int, default=50)
    parser.add_argument('-l', type=int, default=10)
    parser.add_argument('-m', type=int, default=8)
    parser.add_argument('-r', type=int, default=10)
    
    # Parse arguments
    opt = parser.parse_args()
    # 使用環境變量覆蓋命令行參數
    if os.environ.get('PUSH_URL'):
        opt.push_url = os.environ['PUSH_URL']
        logger.info(f"Using PUSH_URL from environment: {opt.push_url}")
    
    opt.customopt = []
    if opt.customvideo_config != '':
        with open(opt.customvideo_config, 'r') as file:
            opt.customopt = json.load(file)
    
    logger.info("Application starting up...")
    logger.info(f"Model: {opt.model}")
    logger.info(f"Avatar: {opt.avatar_id}")
    
    # 初始化配置
    CONFIGS['avatars'] = get_available_avatars(opt.avatar_id, opt.avatar_list)
    logger.info("Avatars configuration initialized")
    
    if opt.model == 'ernerf':       
        from nerfreal import NeRFReal,load_model,load_avatar
        model = load_model(opt)
        avatar = load_avatar(opt)
        logger.info("Loaded NeRFReal model and avatar")
    elif opt.model == 'musetalk':
        from musereal import MuseReal,load_model,load_avatar,warm_up
        print(opt)  # 保留原有print
        model = load_model()
        avatar = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size,model)
        logger.info("Loaded MuseReal model and avatar")
    elif opt.model == 'wav2lip':
        from lipreal import LipReal,load_model,load_avatar,warm_up
        print(opt)  # 保留原有print
        model = load_model("./models/wav2lip.pth")
        avatar = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size,model,384)
        logger.info("Loaded Wav2Lip model and avatar")
    elif opt.model == 'ultralight':
        from lightreal import LightReal,load_model,load_avatar,warm_up
        print(opt)  # 保留原有print
        model = load_model(opt)
        avatar = load_avatar(opt.avatar_id)
        warm_up(opt.batch_size,avatar,160)
        logger.info("Loaded Ultralight model and avatar")

    if opt.transport == 'rtmp':
        thread_quit = Event()
        nerfreals[0] = build_nerfreal(0)
        rendthrd = Thread(target=nerfreals[0].render,args=(thread_quit,))
        rendthrd.start()
        logger.info("Started RTMP render thread")

    #############################################################################
    appasync = web.Application()
    appasync.on_shutdown.append(on_shutdown)
    
    # 設置路由
    appasync.router.add_post("/offer", offer)
    appasync.router.add_post("/human", human)
    appasync.router.add_post("/humanaudio", humanaudio)
    appasync.router.add_post("/set_audiotype", set_audiotype)
    appasync.router.add_post("/record", record)
    appasync.router.add_post("/is_speaking", is_speaking)
    appasync.router.add_post("/switch_tts_endpoint", switch_tts_endpoint)
    appasync.router.add_post("/switch_avatar", switch_avatar)
    appasync.router.add_get("/get_avatars", get_avatars)
    appasync.router.add_get("/get_voices", get_voices)
    appasync.router.add_post("/get_current_config", get_current_config)
    appasync.router.add_get("/health", health_check)
    appasync.router.add_static('/',path='web')
    logger.info("Routes configured")

    # 設置CORS
    cors = aiohttp_cors.setup(appasync, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods=["POST", "GET", "OPTIONS"]
        )
    })
    
    for route in list(appasync.router.routes()):
        cors.add(route)
    logger.info("CORS configured")

    pagename = 'webrtcapi.html'
    if opt.transport == 'rtmp':
        pagename = 'echoapi.html'
    elif opt.transport == 'rtcpush':
        pagename = 'rtcpushapi.html'
    print('start http server; http://<serverip>:'+str(opt.listenport)+'/'+pagename)  # 保留原有print
    logger.info(f"Starting HTTP server on port {opt.listenport}")

    def run_server(runner):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, '0.0.0.0', opt.listenport)
        loop.run_until_complete(site.start())
        logger.info("HTTP server started")
        
        if opt.transport == 'rtcpush':
            for k in range(opt.max_session):
                push_url = opt.push_url
                if k != 0:
                    push_url = opt.push_url+str(k)
                logger.info(f"Initializing push URL session {k}")
                loop.run_until_complete(run(push_url,k))
        
        loop.run_forever()
    
    try:
        run_server(web.AppRunner(appasync))
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, initiating shutdown...")
        handle_sigterm(None, None)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise
