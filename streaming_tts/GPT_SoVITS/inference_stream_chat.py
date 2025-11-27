import os, sys
import re, logging

current_dir = os.getcwd()
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "GPT_SoVITS"))

logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)
logging.getLogger("PIL").setLevel(logging.ERROR)
logging.getLogger("multipart").setLevel(logging.WARNING)
import librosa
import soundfile as sf
import requests, json
import torch
import tempfile, io, wave
import argparse
from pydub import AudioSegment
import numpy as np
from TTS_infer_pack.TTS import TTS, TTS_Config
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto()

if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
is_half = eval(os.environ.get("is_half", "True")) and not torch.backends.mps.is_available()

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # 确保直接启动推理UI时也能够设置。
from faster_whisper import WhisperModel
import threading
import time
model_path = 'tools/asr/models/faster-whisper-large-v3-turbo'
import traceback
import gradio as gr

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
try:
    model = WhisperModel(model_path, device=device, compute_type="float32")
except:
    print(traceback.format_exc())
dict_language = {
    i18n("中文"): "all_zh",  # 全部按中文识别
    i18n("英文"): "en",  # 全部按英文识别#######不变
    i18n("日文"): "all_ja",  # 全部按日文识别
    i18n("中英混合"): "zh",  # 按中英混合识别####不变
    i18n("日英混合"): "ja",  # 按日英混合识别####不变
    i18n("多语种混合"): "auto",  # 多语种启动切分识别语种
}

cut_method = {
    i18n("不切"): "cut0",
    i18n("凑四句一切"): "cut1",
    i18n("凑50字一切"): "cut2",
    i18n("按中文句号。切"): "cut3",
    i18n("按英文句号.切"): "cut4",
    i18n("按标点符号切"): "cut5",
}

tts_config = TTS_Config("GPT_SoVITS/configs/tts_infer.yaml")
tts_config.device = device
tts_config.is_half = is_half
tts_pipline = TTS(tts_config)
gpt_path = tts_config.t2s_weights_path
sovits_path = tts_config.vits_weights_path

parser = argparse.ArgumentParser(description="GPT-SoVITS Streaming")
parser.add_argument(
    "-s",
    "--sovits_path",
    type=str,
    default=sovits_path,
    help="SoVITS模型路径",
)
parser.add_argument(
    "-g",
    "--gpt_path",
    type=str,
    default=gpt_path,
    help="GPT模型路径",
)
parser.add_argument(
    "-rw",
    "--ref_wav",
    type=str,
    default="./example/archive_ruanmei_8.wav",
    help="参考音频路径",
)
parser.add_argument(
    "-rt",
    "--prompt_text",
    type=str,
    default="我听不惯现代乐，听戏却极易入迷，琴弦拨动，时间便流往过去。",
    help="参考音频文本",
)
parser.add_argument(
    "-rl",
    "--prompt_language",
    type=str,
    default=i18n("中文"),
    help="参考音频语种",
)

args = parser.parse_args()

SoVITS_weight_root = "SoVITS_weights"
GPT_weight_root = "GPT_weights"
os.makedirs(SoVITS_weight_root, exist_ok=True)
os.makedirs(GPT_weight_root, exist_ok=True)

sovits_path = args.sovits_path
gpt_path = args.gpt_path


def get_weights_names():
    SoVITS_names = [sovits_path]
    for name in os.listdir(SoVITS_weight_root):
        if name.endswith(".pth"):
            SoVITS_names.append("%s/%s" % (SoVITS_weight_root, name))
    GPT_names = [gpt_path]
    for name in os.listdir(GPT_weight_root):
        if name.endswith(".ckpt"):
            GPT_names.append("%s/%s" % (GPT_weight_root, name))
    return SoVITS_names, GPT_names


def custom_sort_key(s):
    # 使用正则表达式提取字符串中的数字部分和非数字部分
    parts = re.split("(\d+)", s)
    # 将数字部分转换为整数，非数字部分保持不变
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts


# 初始化引导音频列表


def replace_chinese(text):
    pattern = r'([\u4e00-\u9fa5]{5}).*'
    result = re.sub(pattern, r'\1...', text)
    return result

def init_wav_list(sovits_path):

    wav_path = "./output/slicer_opt"
    match = re.search(r'([a-zA-Z0-9\-_]+)_e\d+_s\d+\.pth',sovits_path)
    if match:
        result = match.group(1)
        wav_path = f"./logs/{result}/5-wav32k/"
    else:
        return [],{}

    res_wavs = {}

    res_text = ["请选择参考音频"]

    # 读取文本
    text = ""
    try:

        with open(rf'./logs/{result}/2-name2text.txt', 'r',encoding='utf-8') as f:
            text = f.read()
    except Exception as e:

        return [],{}


    # 遍历目录
    for file_path in os.listdir(wav_path)[:200]:
        # 检查当前file_path是否为文件
        if os.path.isfile(os.path.join(wav_path, file_path)):
            # 将文件名添加到列表中
            match = re.search(rf'{file_path}\t(.+?)\t(.+?)\t(.+?)\n', text)
            if match:
        
                # 提取匹配到的内容
                extracted_text = match.group(3)
                # print(extracted_text)
                
                # 传入音频文件路径，获取音频数据和采样率
                audio_data, sample_rate = librosa.load(f'./logs/{result}/5-wav32k/{file_path}')
                # 使用librosa.get_duration函数计算音频文件的长度
                duration = librosa.get_duration(y=audio_data, sr=sample_rate)
                duration = int(duration)
                key = f"{replace_chinese(extracted_text)}_{duration}秒"

                if duration > 2 and duration < 11:
                    res_text.append(key)
                    res_wavs[key] = (f'./logs/{result}/5-wav32k/{file_path}',extracted_text)


            else:
                print("No match found")

    # print(res_text)
    # print(res_wavs)

    return res_text,res_wavs

# 切换参考音频

def change_wav(audio_name):

    first_key = list(reference_dict.keys())[0]

    try:
        value = reference_dict[audio_name]
        return value[0],value[1]
    except Exception as e:
        return reference_dict[first_key][0],reference_dict[first_key][1]

def change_choices():
    SoVITS_names, GPT_names = get_weights_names()
    return {
        "choices": sorted(SoVITS_names, key=custom_sort_key),
        "__type__": "update",
    }, {
        "choices": sorted(GPT_names, key=custom_sort_key),
        "__type__": "update",
    }


def change_sovits_weights(sovits_path_in):


    sovits_path_in = sovits_path_in.replace("SoVITS_weights/","")

    global reference_wavs,reference_dict
    reference_wavs,reference_dict = init_wav_list(sovits_path_in)

    return gr.update(choices=reference_wavs)


SoVITS_names, GPT_names = get_weights_names()

reference_wavs,reference_dict = init_wav_list(sovits_path)




# from https://huggingface.co/spaces/coqui/voice-chat-with-mistral/blob/main/app.py
def wave_header_chunk(frame_input=b"", channels=1, sample_width=2, sample_rate=32000):
    # This will create a wave header then append the frame input
    # It should be first on a streaming wav file
    # Other frames better should not have it (else you will hear some artifacts each chunk start)
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)

    wav_buf.seek(0)
    return wav_buf.read()


def stream_inference(
    input_audio,
    # text,
    text_lang,
    ref_audio_path,
    prompt_text,
    prompt_lang,
    top_k,
    top_p,
    temperature,
    text_split_method,
    batch_size,
    speed_factor,
    ref_text_free,
    split_bucket,
    byte_stream=True,
):
    def check_audio(audio):
        if audio is None:
            return None
        if not os.path.exists(audio):
            raise ValueError("音頻文件不存在")
        return audio
    
    
    try:
        audio = check_audio(input_audio)
        if audio is None:
            print("請先錄製音頻")
            return None
        input_text = ''
        print("開始辨識")
        with open(audio, "rb") as audio_file:
        # with open("temp.wav", "rb") as audio_file:
            segments, info = model.transcribe(
                audio=audio_file,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=700),
                language="zh",
            )
            for segment in segments:
                input_text += segment.text

    except Exception as e:
        print(e)
        return None
    
    try:
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
                "text": input_text
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
        print("Input audio transcription:", input_text)

        print("Decoded message:", decoded_message)
    except Exception as e:
        print(traceback.format_exc())
        print("An error occurred during transcription:", str(e))
        return None
    except json.JSONDecodeError:
        print("Received non-JSON response:", response.text)
        return None
    
    inputs = {
        "text": decoded_message,
        "text_lang": dict_language[text_lang],
        "ref_audio_path": ref_audio_path,
        "prompt_text": prompt_text if not ref_text_free else "",
        "prompt_lang": dict_language[prompt_lang],
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": cut_method[text_split_method],
        "batch_size": int(batch_size),
        "speed_factor": float(speed_factor),
        "split_bucket": split_bucket,
        "return_fragment": True,  # IMPORTANT!
    }

    chunks = tts_pipline.run(inputs)
    if byte_stream:
        yield wave_header_chunk()
        for _sr, data in chunks:
            yield data.tobytes()
    else:
        # Send chunk files
        i = 0
        format = "wav"
        for _sr, data in chunks:
            i += 1
            file = f"{tempfile.gettempdir()}/{i}.{format}"
            segment = AudioSegment(data, frame_rate=32000, sample_width=2, channels=1)
            segment.export(file, format=format)
            yield file

def save_audio(audio):
    if audio is None:
        return None
    # 讀取音頻文件並轉換為 WAV 格式
    with open(audio, "rb") as audio_file:
        audio_data = audio_file.read()

    # 假設音頻數據是 numpy 數組
    audio_array = np.frombuffer(audio_data, dtype=np.int16)

    # 設置 WAV 文件參數
    sample_rate = 48000  # 假設采樣率為 48000 Hz
    num_channels = 1  # 假設單聲道
    sampwidth = 2  # 假設 16 位音頻

    # 創建 WAV 文件
    wav_filename = "temp.wav"
    with wave.open(wav_filename, "wb") as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(sampwidth)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_array.tobytes())


with gr.Blocks(title="GPT-SoVITS Streaming Demo") as app:
    gr.Markdown(value=i18n("歡迎使用智能客服聊天機器人"))
    input_audio=gr.Audio(value=None,type="filepath",sources=["microphone"])

    # text = gr.Textbox(label=i18n("需要合成的文本"), value="", lines=9, interactive=True)

    inference_button = gr.Button(i18n("請回答我"), variant="primary")
    # gr.Markdown(value=i18n("* 结果输出(等待第2句推理结束后会自动播放)"))
    with gr.Row():
        audio_file = gr.Audio(
            value=None,
            # label=i18n("输出的语音"),
            streaming=True,
            autoplay=True,  # disable auto play for Windows, due to https://developer.chrome.com/blog/autoplay#webaudio
            interactive=False,
            # show_label=True,
        )

    text_language = gr.Dropdown(choices=["多语种混合"], value=i18n("多语种混合"), label="文本语言", visible=False)
    inp_ref = gr.Audio(label="请上传 3~10 秒内参考音频，超过会报警！", type="filepath", value="Keira.wav", visible=False)
    prompt_text = gr.Textbox(label="提示文本", value="光动嘴不如亲自做给你看,等我一下喔", visible=False)
    prompt_language = gr.Dropdown(choices=["多语种混合"], value=i18n("多语种混合"), label="提示语言", visible=False)
    top_k = gr.Slider(minimum=1, maximum=10, step=1, value=5, label="Top K", visible=False)
    top_p = gr.Slider(minimum=0, maximum=1, step=0.1, value=1, label="Top P", visible=False)
    temperature = gr.Slider(minimum=0.1, maximum=2, step=0.1, value=1, label="Temperature", visible=False)
    how_to_cut = gr.Dropdown(choices=["凑四句一切"], value=i18n("凑四句一切"), label="文本切割方式", visible=False)
    batch_size = gr.Slider(minimum=1, maximum=100, step=1, value=20, label="Batch Size", visible=False)
    speed_factor = gr.Slider(minimum=0.1, maximum=2, step=0.1, value=1, label="Speed Factor", visible=False)
    ref_text_free = gr.Checkbox(value=False, label="参考文本自由", visible=False)
    split_bucket = gr.Checkbox(value=True, label="分割桶", visible=False)
    # inp_ref="Keira.wav"
    # prompt_text="光动嘴不如亲自做给你看,等我一下喔"
    # text_language=i18n("多语种混合")
    # prompt_language=i18n("多语种混合")
    # top_k=5
    # top_p=1
    # temperature=1
    # how_to_cut=i18n("凑四句一切")
    # batch_size=20
    # speed_factor=1
    # ref_text_free=False
    # split_bucket=True
    # input_audio.change(save_audio,input_audio,input_audio)
    inference_button.click(
        stream_inference,
        [   
            input_audio,
            # text,
            text_language,
            inp_ref,
            prompt_text,
            prompt_language,
            top_k,
            top_p,
            temperature,
            how_to_cut,
            batch_size,
            speed_factor,
            ref_text_free,
            split_bucket,
        ],
        [audio_file],
    )
    # ).then(lambda: gr.update(interactive=True), None, [input], queue=False)



app.queue().launch(
    server_name="127.0.0.1",
    inbrowser=True,
    share=True,
    server_port=8086,
    quiet=True,
)


# ui = gr.Interface(
#     fn=stream_inference,
#     inputs=[
#         gr.Audio(type="filepath", label="输入音频"),
#         gr.Dropdown(choices=[i18n("多语种混合")], value=i18n("多语种混合"), label="文本语言", visible=False),
#         gr.Audio(value="Keira.wav", type="filepath", label="参考音频（3~10秒）", visible=False),
#         gr.Textbox(value="光动嘴不如亲自做给你看,等我一下喔", label="提示文本", visible=False),
#         gr.Dropdown(choices=[i18n("多语种混合")], value=i18n("多语种混合"), label="提示语言", visible=False),
#         gr.Slider(minimum=1, maximum=10, step=1, value=5, label="Top K", visible=False),
#         gr.Slider(minimum=0, maximum=1, step=0.1, value=1, label="Top P", visible=False),
#         gr.Slider(minimum=0.1, maximum=2, step=0.1, value=1, label="Temperature", visible=False),
#         gr.Dropdown(choices=[i18n("凑四句一切")], value=i18n("凑四句一切"), label="文本切割方式", visible=False),
#         gr.Slider(minimum=1, maximum=100, step=1, value=20, label="Batch Size", visible=False),
#         gr.Slider(minimum=0.1, maximum=2, step=0.1, value=1, label="Speed Factor", visible=False),
#         gr.Checkbox(value=False, label="参考文本自由", visible=False),
#         gr.Checkbox(value=True, label="分割桶", visible=False)
#     ],
#     outputs=gr.Audio(label="输出音频", autoplay=True, streaming=True),
#     title="GPT-SoVITS Streaming Demo",
#     description="歡迎使用智能客服聊天機器人",
#     # allow_flagging='never'
# )

# ui.launch(
#     server_name="127.0.0.1",
#     inbrowser=True,
#     share=True,
#     server_port=8086,
#     quiet=True
# )