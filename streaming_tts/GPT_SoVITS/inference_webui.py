'''
按中英混合识别
按日英混合识别
多语种启动切分识别语种
全部按中文识别
全部按英文识别
全部按日文识别
'''
import os,sys,shutil
now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS"%(now_dir))


from resemble_enhance.enhancer.inference import denoise, enhance

import os, re, logging
logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)
import pdb
import torch
import torchaudio
import gc
import librosa
import soundfile as sf

import numpy as np

import time


infer_ttswebui = os.environ.get("infer_ttswebui", 9872)
infer_ttswebui = int(infer_ttswebui)
is_share = os.environ.get("is_share", "False")
is_share = eval(is_share)
if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
# is_half = eval(os.environ.get("is_half", "True")) and not torch.backends.mps.is_available()

is_half = True

import gradio as gr
from TTS_infer_pack.TTS import TTS, TTS_Config
from TTS_infer_pack.text_segmentation_method import cut1, cut2, cut3, cut4, cut5
from TTS_infer_pack.text_segmentation_method import get_method
from tools.i18n.i18n import I18nAuto

import random

i18n = I18nAuto()

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # 确保直接启动推理UI时也能够设置。

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    is_half = False

import subprocess



def clear_gpu_cash():
    # del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _up_loud(video_file,loud):

    avg_volume = float(loud)

    if video_file is None:
        return None, None
    
    print(video_file)

    sf.write('./output_loud.wav', video_file[1], video_file[0], 'PCM_24')

    # 使用EBU R128标准校正音频
    cmd = ['ffmpeg','-y','-i', './output_loud.wav', '-af', f'loudnorm=I={avg_volume}:TP=-2:LRA=11','-c:v', 'copy', '-c:a', 'aac', "./output_loud.aac"]

    subprocess.call(cmd)

    return './output_loud.aac'

    # return "./output.aac"



def _fn(path, solver="Midpoint", nfe=64, tau=0.5,chunk_seconds=10,chunks_overlap=0.5, denoising=True):
    if path is None:
        return None, None
    print(path)
    sf.write('./output.wav', path[1], path[0], 'PCM_24')

    solver = solver.lower()
    nfe = int(nfe)
    lambd = 0.9 if denoising else 0.1

    dwav, sr = torchaudio.load('./output.wav')
    dwav = dwav.mean(dim=0)

    wav2, new_sr = enhance(dwav = dwav, sr = sr, device = device, nfe=nfe,chunk_seconds=chunk_seconds,chunks_overlap=chunks_overlap, solver=solver, lambd=lambd, tau=tau)

    wav2 = wav2.cpu().numpy()

    clear_gpu_cash()
    return (new_sr, wav2)


# def set_seed(seed):
#     seed = seed if seed != -1 else random.randrange(1 << 32)
#     torch.manual_seed(seed)
#     print("Seed:", seed)
#     return seed

# def reuse_seed(seed):
#     return seed

dict_language = {
    i18n("中文"): "all_zh",#全部按中文识别
    i18n("英文"): "en",#全部按英文识别#######不变
    i18n("日文"): "all_ja",#全部按日文识别
    i18n("中英混合"): "zh",#按中英混合识别####不变
    i18n("日英混合"): "ja",#按日英混合识别####不变
    i18n("多语种混合"): "auto",#多语种启动切分识别语种
}

cut_method = {
    i18n("不切"):"cut0",
    i18n("凑四句一切"): "cut1",
    i18n("凑50字一切"): "cut2",
    i18n("按中文句号。切"): "cut3",
    i18n("按英文句号.切"): "cut4",
    i18n("按标点符号切"): "cut5",
}

tts_config = TTS_Config("GPT_SoVITS/configs/tts_infer.yaml")
tts_config.device = device
tts_config.is_half = is_half
tts_pipeline = TTS(tts_config)
gpt_path = tts_config.t2s_weights_path
sovits_path = tts_config.vits_weights_path



def inference(text, text_lang, 
              ref_audio_path, prompt_text, 
              prompt_lang, top_k, 
              top_p, temperature, 
              text_split_method, batch_size, 
              speed_factor, ref_text_free,
              split_bucket,fragment_interval,seed, keep_random, parallel_infer,
              repetition_penalty
              ):
    seed = -1 if keep_random else seed
    actual_seed = seed if seed not in [-1, "", None] else random.randrange(1 << 32)
    start_time = time.time()  # 记录开始时间
    inputs={
        "text": text,
        "text_lang": dict_language[text_lang],
        "ref_audio_path": ref_audio_path,
        "prompt_text": prompt_text if not ref_text_free else "",
        "prompt_lang": dict_language[prompt_lang],
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": cut_method[text_split_method],
        "batch_size":int(batch_size),
        "speed_factor":float(speed_factor),
        "split_bucket":split_bucket,
        "return_fragment":False,
        "fragment_interval":fragment_interval,
        "seed":actual_seed,
        "parallel_infer": parallel_infer,
        "repetition_penalty": repetition_penalty,
    }

    # yield next(tts_pipline.run(inputs))
    for item in tts_pipeline.run(inputs):
        yield item, actual_seed

    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算经过的时间
    print(f"推理 executed in {elapsed_time:.6f} seconds")  # 输出执行时间，保留6位小数


def custom_sort_key(s):
    # 使用正则表达式提取字符串中的数字部分和非数字部分
    parts = re.split('(\d+)', s)
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
    return {"choices": sorted(SoVITS_names, key=custom_sort_key), "__type__": "update"}, {"choices": sorted(GPT_names, key=custom_sort_key), "__type__": "update"}





pretrained_sovits_name = "GPT_SoVITS/pretrained_models/s2G488k.pth"
pretrained_gpt_name = "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
SoVITS_weight_root = "SoVITS_weights"
GPT_weight_root = "GPT_weights"
os.makedirs(SoVITS_weight_root, exist_ok=True)
os.makedirs(GPT_weight_root, exist_ok=True)


def get_weights_names():
    SoVITS_names = [pretrained_sovits_name]
    for name in os.listdir(SoVITS_weight_root):
        if name.endswith(".pth"): SoVITS_names.append("%s/%s" % (SoVITS_weight_root, name))
    GPT_names = [pretrained_gpt_name]
    for name in os.listdir(GPT_weight_root):
        if name.endswith(".ckpt"): GPT_names.append("%s/%s" % (GPT_weight_root, name))
    return SoVITS_names, GPT_names


SoVITS_names, GPT_names = get_weights_names()


reference_wavs,reference_dict = init_wav_list(sovits_path)


#region 输出音频历史记录相关
output_history =[]
history_max_num = 10
change_num = 10




max_textboxes = len(reference_wavs)




def sync_output_history_to_checkbox_audio():
    checkbox_result = []
    audio_result = []
    for item in output_history:
        try:
            label = item['label']
        except Exception as e:
            label = ""
        if len(label)>15:
            label=label[:15]+'...'
        checkbox_result.append(gr.update(label=label,value=False))
        try:
            audio_result.append(gr.update(value=item['value']))
        except Exception as e:
            audio_result.append(gr.update(value = None))
    for _ in range(len(audio_result),history_max_num):
        checkbox_result.append(gr.update(label="",value=False))
        audio_result.append(gr.update(value = None))
    return [*checkbox_result,*audio_result]

def add_to_history(audio,input_text):
    if(audio is None or audio[1] is not None):
        if len(output_history) == history_max_num:
            output_history.pop()
        output_history.insert(0,{'value':audio,'label':input_text})

    return [*sync_output_history_to_checkbox_audio()]


# 连续抽卡

def inference_take(text, text_lang, 
              ref_audio_path, prompt_text, 
              prompt_lang, top_k, 
              top_p, temperature, 
              text_split_method, batch_size, 
              speed_factor, ref_text_free,
              split_bucket,fragment_interval,seed, keep_random, parallel_infer,
              repetition_penalty
              ):

    # seed = -1 if keep_random else seed
    actual_seed = -1

    inputs={
        "text": text,
        "text_lang": dict_language[text_lang],
        "ref_audio_path": ref_audio_path,
        "prompt_text": prompt_text if not ref_text_free else "",
        "prompt_lang": dict_language[prompt_lang],
        "top_k": top_k,
        "top_p": top_p,
        "temperature": temperature,
        "text_split_method": cut_method[text_split_method],
        "batch_size":int(batch_size),
        "speed_factor":float(speed_factor),
        "split_bucket":split_bucket,
        "return_fragment":False,
        "fragment_interval":fragment_interval,
        "seed":actual_seed,
        "parallel_infer": parallel_infer,
        "repetition_penalty": repetition_penalty,
    }

    print(change_num)

    for x in range(change_num):

        audio = next(tts_pipeline.run(inputs))

        # output_history.insert(0,audio)

        # output_history.insert(0,{'value':audio,'label':text})

        add_to_history(audio,text)

    return [*sync_output_history_to_checkbox_audio()]



def clear_history():
    output_history = []
    checkbox_result = []
    audio_result = []
    for _ in range(history_max_num):
        checkbox_result.append(gr.update(label="",value=False))
        audio_result.append(gr.update(value = None))
    return [*checkbox_result,*audio_result]

def shown_audio_num_change(audio_num):
    audio_num = int(audio_num)
    audio_result = []
    checkbox_result = []

    global change_num

    change_num = audio_num

    for _ in range(audio_num):
        audio_result.append(gr.update(visible=True))
        checkbox_result.append(gr.update(visible=True))
    for _ in range(audio_num,history_max_num):
        audio_result.append(gr.update(visible=False))
        checkbox_result.append(gr.update(visible=False))
    return [*checkbox_result,*audio_result]

def delete_selected_history(*selected_list):
    for i in reversed(range(len(output_history))):
        if(selected_list[i]):
            output_history.pop(i)
    return [*sync_output_history_to_checkbox_audio()]

def merge_selected_history(*selected_list):
    m_list = []
    labels = ""
    for i in reversed(range(len(output_history))):
        if(selected_list[i]):
            labels+=" "+output_history[i]["label"]
            m_list.append(output_history[i]["value"][1])
    if(m_list):
        combined = np.hstack(m_list)
        delete_selected_history(*selected_list)       
        return add_to_history((32000, combined),labels)
    return [*sync_output_history_to_checkbox_audio()]




def create_audio():


    sovits_path = tts_config.vits_weights_path

    sovits_path = sovits_path.replace("SoVITS_weights/","")

    match = re.search(r'([a-zA-Z0-9\-_]+)_e\d+_s\d+\.pth',sovits_path)
    if match:
        result = match.group(1)

        audio_list = []

        for file_path in os.listdir(f"./{result}_audio/"):

            audio_list.append(file_path)

        if len(audio_list) == 0:
            return false

        dir_path = f'./logs/{result}'

        # 如果目录已经存在，则删除它
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)

        # 创建新的目录
        os.makedirs(dir_path)

        # 指定源目录的路径
        source_directory = f'./{result}_audio/'

        # 指定目标目录的路径
        destination_directory = f'./logs/{result}/5-wav32k'

        # 使用shutil.copytree()函数拷贝源目录到目标目录
        shutil.copytree(source_directory, destination_directory)

        txt = ""


        for x in audio_list:
            text = x.replace(".wav","")
            txt += f'{x}\t123\t123\t{text}\n'


        with open(rf'./logs/{result}/2-name2text.txt', 'w',encoding='utf-8') as f:
            f.write(txt)

        sovits_path = sovits_path.replace("SoVITS_weights/","")

        print(sovits_path)

        global reference_wavs,reference_dict,max_textboxes
        reference_wavs,reference_dict = init_wav_list(sovits_path)
        max_textboxes = len(reference_wavs)
        return gr.update(choices=reference_wavs)


        





def change_sovits_weights(sovits_path_in):


    sovits_path_in = sovits_path_in.replace("SoVITS_weights/","")

    print(sovits_path_in)

    global reference_wavs,reference_dict,max_textboxes
    reference_wavs,reference_dict = init_wav_list(sovits_path_in)
    max_textboxes = len(reference_wavs)

    


    

    return gr.update(choices=reference_wavs)


def refresh_au():

    # global textboxes

    textboxes = []



    for x_int in range(30):

        if x_int != 0:

            try:
                textboxes.append(gr.update(label=reference_dict[reference_wavs[x_int]][1],value=reference_dict[reference_wavs[x_int]][0]))
            except Exception as e:
                textboxes.append(gr.update(label="无参考音频"))

    return textboxes



def main():

    with gr.Blocks(title="GPT-SoVITS WebUI") as app:
        gr.Markdown(
            value=i18n("本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责. <br>如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录<b>LICENSE</b>.")
        )
        
        with gr.Column():
            # with gr.Group():
            gr.Markdown(value=i18n("模型切换"))
            with gr.Row():
                GPT_dropdown = gr.Dropdown(label=i18n("GPT模型列表"), choices=sorted(GPT_names, key=custom_sort_key), value=gpt_path, interactive=True)
                SoVITS_dropdown = gr.Dropdown(label=i18n("SoVITS模型列表"), choices=sorted(SoVITS_names, key=custom_sort_key), value=sovits_path, interactive=True)
                wavs_dropdown = gr.Dropdown(label=i18n("参考音频列表"), choices=reference_wavs,value="请选择参考音频",interactive=True)
                refresh_button = gr.Button(i18n("刷新模型路径"), variant="primary")
                create_button = gr.Button("生成参考音频", variant="primary")
                refresh_button.click(fn=change_choices, inputs=[], outputs=[SoVITS_dropdown, GPT_dropdown])
                create_button.click(fn=create_audio, inputs=[], outputs=[wavs_dropdown])
                
                # SoVITS_dropdown.change(tts_pipline.init_vits_weights, [SoVITS_dropdown], [])

                # SoVITS_dropdown.change(change_sovits_weights,[SoVITS_dropdown],[wavs_dropdown]).then(

                #     tts_pipeline.init_vits_weights, [SoVITS_dropdown], []

                # )

                GPT_dropdown.change(tts_pipeline.init_t2s_weights, [GPT_dropdown], [])

        textboxes = []
        with gr.Accordion("参考音频审阅",open=False):
            for x_int in range(30):
                with gr.Row():
                    if x_int != 0:
                        # t = gr.Markdown(value=reference_wavs[r_num])
                        try:
                            a = gr.Audio(label=reference_dict[reference_wavs[x_int]][1],show_download_button=True,value=reference_dict[reference_wavs[x_int]][0])
                            textboxes.append(a)
                        except Exception as e:
                            # print(str(e))
                            textboxes.append(gr.Audio(label="无参考音频"))
                        # gr.Audio(label="",show_download_button=True)

                        # textboxes.append(gr.Audio(label="",show_download_button=True))
                        # button = gr.Button("点击使用")
                        # button.click(on_button_click,[reference_dict[reference_wavs[r_num]][0],reference_dict[reference_wavs[r_num]][1]],[])
        # refresh_audios = gr.Button("刷新参考音频", variant="primary")

        SoVITS_dropdown.change(change_sovits_weights,[SoVITS_dropdown],[wavs_dropdown]).then(

                    tts_pipeline.init_vits_weights, [SoVITS_dropdown], []

                ).then(

                    refresh_au, [], [*textboxes]

                )

        # refresh_audios.click(refresh_au,[],[*textboxes])
        
        with gr.Row():
            with gr.Column():
                gr.Markdown(value=i18n("*请上传并填写参考信息"))
                inp_ref = gr.Audio(label=i18n("请上传3~10秒内参考音频，超过会报错！"), type="filepath")
                prompt_text = gr.Textbox(label=i18n("参考音频的文本"), value="", lines=2)
                with gr.Row():
                    prompt_language = gr.Dropdown(
                        label=i18n("参考音频的语种"), choices=[i18n("中文"), i18n("英文"), i18n("日文"), i18n("中英混合"), i18n("日英混合"), i18n("多语种混合")], value=i18n("中文")
                    )
                    with gr.Column():
                        ref_text_free = gr.Checkbox(label=i18n("开启无参考文本模式。不填参考文本亦相当于开启。"), value=False, interactive=True, show_label=True)
                        gr.Markdown(i18n("使用无参考文本模式时建议使用微调的GPT，听不清参考音频说的啥(不晓得写啥)可以开，开启后无视填写的参考文本。"))
            wavs_dropdown.change(change_wav,[wavs_dropdown],[inp_ref,prompt_text])
        
            with gr.Column():
                gr.Markdown(value=i18n("*请填写需要合成的目标文本和语种模式"))
                text = gr.Textbox(label=i18n("需要合成的文本"), value="", lines=16, max_lines=16)
                text_language = gr.Dropdown(
                    label=i18n("需要合成的语种"), choices=[i18n("中文"), i18n("英文"), i18n("日文"), i18n("中英混合"), i18n("日英混合"), i18n("多语种混合")], value=i18n("中文")
                )

            
        with gr.Group():
            gr.Markdown(value=i18n("推理设置"))
            with gr.Row():

                with gr.Column():
                    batch_size = gr.Slider(minimum=1,maximum=200,step=1,label=i18n("batch_size"),value=20,interactive=True)
                    fragment_interval = gr.Slider(minimum=0.01,maximum=1,step=0.01,label=i18n("分段间隔(秒)"),value=0.3,interactive=True)
                    speed_factor = gr.Slider(minimum=0.25,maximum=4,step=0.05,label="speed_factor",value=1.0,interactive=True)
                    top_k = gr.Slider(minimum=1,maximum=100,step=1,label=i18n("top_k"),value=5,interactive=True)
                    top_p = gr.Slider(minimum=0,maximum=1,step=0.05,label=i18n("top_p"),value=1,interactive=True)
                    temperature = gr.Slider(minimum=0,maximum=1,step=0.05,label=i18n("temperature"),value=1,interactive=True)
                    repetition_penalty = gr.Slider(minimum=0,maximum=2,step=0.05,label=i18n("重复惩罚"),value=1.35,interactive=True)
                    loudness = gr.Slider(minimum=-23,maximum=-1,step=0.05,label=i18n("音频响度"),value=-12.0,interactive=True)
                with gr.Column():
                    # seed = gr.Number(label=i18n("种子"), value=-1, precision=0)
                    # last_seed = gr.State(value=-1)
                    # reuse_button = gr.Button(value=i18n("种子复用"))
                    how_to_cut = gr.Radio(
                        label=i18n("怎么切"),
                        choices=[i18n("不切"), i18n("凑四句一切"), i18n("凑50字一切"), i18n("按中文句号。切"), i18n("按英文句号.切"), i18n("按标点符号切"), ],
                        value=i18n("按标点符号切"),
                        interactive=True,
                    )
                    parallel_infer = gr.Checkbox(label=i18n("并行推理(速度更快，但可能增大复读概率)"), value=True, interactive=True, show_label=True)
                    split_bucket = gr.Checkbox(label=i18n("数据分桶(并行推理时会降低一点计算量)"), value=True, interactive=True, show_label=True)
                    seed = gr.Number(label=i18n("随机种子"),value=-1)
                    keep_random = gr.Checkbox(label=i18n("保持随机"), value=True, interactive=True, show_label=True)
                # with gr.Column():
                    output = gr.Audio(label=i18n("输出的语音"),show_download_button=True)
                    with gr.Row():
                    
                        inference_button = gr.Button(i18n("合成语音"), variant="primary")

                        stop_infer = gr.Button(i18n("终止合成"), variant="primary")
                        up_button = gr.Button(i18n("音频降噪增强"), variant="primary")

                        take_button = gr.Button(i18n("连续抽卡"), variant="primary")

                        up_loud = gr.Button("调整响度")

            # inference_button.click(set_seed, seed, last_seed).then(
            #     inference,
            #     [
            #         text,text_language, inp_ref, 
            #         prompt_text, prompt_language, 
            #         top_k, top_p, temperature, 
            #         how_to_cut, batch_size, 
            #         speed_factor, ref_text_free,
            #         split_bucket,fragment_interval,seed, keep_random, parallel_infer,repetition_penalty
            #     ],
            #     [output,seed],
            # )   


            
            # inference_button.click(
            #     inference,
            #     [
            #         text,text_language, inp_ref, 
            #         prompt_text, prompt_language, 
            #         top_k, top_p, temperature, 
            #         how_to_cut, batch_size, 
            #         speed_factor, ref_text_free,
            #         split_bucket,fragment_interval
            #     ],
            #     [output],
            # )


        #     inference_button_s.click(
        #     inference_stream,
        #     [
        #         text,text_language, inp_ref, 
        #         prompt_text, prompt_language, 
        #         top_k, top_p, temperature, 
        #         how_to_cut, batch_size, 
        #         speed_factor, ref_text_free,
        #         split_bucket,fragment_interval,
        #         seed, keep_random, parallel_infer,
        #         repetition_penalty
        #      ],
        #     [output],
        # ).then(lambda: gr.update(interactive=True), None, [text], queue=False)


            inference_button.click(
            inference,
            [
                text,text_language, inp_ref, 
                prompt_text, prompt_language, 
                top_k, top_p, temperature, 
                how_to_cut, batch_size, 
                speed_factor, ref_text_free,
                split_bucket,fragment_interval,
                seed, keep_random, parallel_infer,
                repetition_penalty
             ],
            [output, seed],
        )
            stop_infer.click(tts_pipeline.stop, [], [])
            up_button.click(_fn, [output], [output])

            up_loud.click(_up_loud, [output,loudness], [output])

        with gr.Group():
            gr.Markdown(value=i18n("文本切分工具。太长的文本合成出来效果不一定好，所以太长建议先切。合成会根据文本的换行分开合成再拼起来。"))
            with gr.Row():
                text_inp = gr.Textbox(label=i18n("需要合成的切分前文本"), value="", lines=4)
                with gr.Column():
                    _how_to_cut = gr.Radio(
                                label=i18n("怎么切"),
                                choices=[i18n("不切"), i18n("凑四句一切"), i18n("凑50字一切"), i18n("按中文句号。切"), i18n("按英文句号.切"), i18n("按标点符号切"), ],
                                value=i18n("凑四句一切"),
                                interactive=True,
                            )
                    cut_text= gr.Button(i18n("切分"), variant="primary")
                
                def to_cut(text_inp, how_to_cut):
                    if len(text_inp.strip()) == 0 or text_inp==[]:
                        return ""
                    method = get_method(cut_method[how_to_cut])
                    return method(text_inp)
            
                text_opt = gr.Textbox(label=i18n("切分后文本"), value="", lines=4)
                cut_text.click(to_cut, [text_inp, _how_to_cut], [text_opt])
            gr.Markdown(value=i18n("后续将支持转音素、手工修改音素、语音合成分步执行。"))

            history_audio = []
            history_checkbox = []
            with gr.Accordion("生成历史"):
                with gr.Row():
                    shown_audio_num = gr.Slider(1,20,history_max_num,step=1,interactive=True,label="记录显示数量")
                    add_history_button = gr.Button("添加当前音频记录",variant="primary")
                    merge_audio_button = gr.Button("合并选中音频",variant="primary")
                    delete_select_history_button = gr.Button("删除选择的记录")
                    clear_history_button = gr.Button("清空记录")
                index=0
                while(index<history_max_num):
                    index += 1
                    with gr.Row():
                        for _ in range(1):
                            with gr.Row():
                                with gr.Group():
                                    history_checkbox.append(gr.Checkbox(interactive=True,show_label=False,label=""))
                                    history_audio.append(gr.Audio(label="",show_download_button=True))

                shown_audio_num.change(shown_audio_num_change,[shown_audio_num],[*history_checkbox,*history_audio])
                add_history_button.click(add_to_history,[output,text],[*history_checkbox,*history_audio])
                merge_audio_button.click(merge_selected_history,[*history_checkbox],[*history_checkbox,*history_audio])
                delete_select_history_button.click(delete_selected_history,[*history_checkbox],[*history_checkbox,*history_audio])
                clear_history_button.click(clear_history,outputs=[*history_checkbox,*history_audio])

                take_button.click(
                inference_take,
                [
                    text,text_language, inp_ref, 
                prompt_text, prompt_language, 
                top_k, top_p, temperature, 
                how_to_cut, batch_size, 
                speed_factor, ref_text_free,
                split_bucket,fragment_interval,
                seed, keep_random, parallel_infer,
                repetition_penalty
                ],
                [*history_checkbox,*history_audio],
            )


    app.queue().launch(
        server_name="0.0.0.0",
        inbrowser=True,
        share=is_share,
        server_port=infer_ttswebui,
        quiet=True,
    )


if __name__ == '__main__':
    main()
