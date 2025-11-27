# -*- coding: utf-8 -*-

import sys, os
import warnings
warnings.filterwarnings("ignore")
import torch
import argparse
import pdb, traceback, numpy as np
from scipy.io import wavfile
import librosa
from feature_extractor import cnhubert
from time import time as ttime
import shutil
from tools.my_utils import load_audio

def my_save(fea, path, i_part):#####fix issue: torch.save doesn't support chinese path
    dir = os.path.dirname(path)
    name = os.path.basename(path)
    tmp_path = "%s%s.pth"%(ttime(),i_part)
    torch.save(fea,tmp_path)
    shutil.move(tmp_path,"%s/%s"%(dir,name))

def name2go(wav_name, wav_path, model, hubert_dir, wav32dir, device, is_half):
    hubert_path = "%s/%s.pt"%(hubert_dir,wav_name)
    if os.path.exists(hubert_path):
        return
        
    tmp_audio = load_audio(wav_path, 32000)
    tmp_max = np.abs(tmp_audio).max()
    if tmp_max > 2.2:
        print("%s-filtered,%s" % (wav_name, tmp_max))
        return
        
    maxx = 0.95
    alpha = 0.5
    tmp_audio32 = (tmp_audio / tmp_max * (maxx * alpha*32768)) + ((1 - alpha)*32768) * tmp_audio
    tmp_audio32b = (tmp_audio / tmp_max * (maxx * alpha*1145.14)) + ((1 - alpha)*1145.14) * tmp_audio
    tmp_audio = librosa.resample(
        tmp_audio32b, orig_sr=32000, target_sr=16000
    )
    tensor_wav16 = torch.from_numpy(tmp_audio)
    if is_half:
        tensor_wav16 = tensor_wav16.half().to(device)
    else:
        tensor_wav16 = tensor_wav16.to(device)
        
    ssl = model.model(tensor_wav16.unsqueeze(0))["last_hidden_state"].transpose(1,2).cpu()
    if np.isnan(ssl.detach().numpy()).sum()!= 0:
        print("nan filtered:%s"%wav_name)
        return False
        
    wavfile.write(
        "%s/%s"%(wav32dir,wav_name),
        32000,
        tmp_audio32.astype("int16"),
    )
    my_save(ssl, hubert_path, 0)
    return True

def process_audio(i_part, all_parts, args, is_half):
    hubert_dir = "%s/4-cnhubert"%(args.opt_dir)
    wav32dir = "%s/5-wav32k"%(args.opt_dir)
    os.makedirs(args.opt_dir, exist_ok=True)
    os.makedirs(hubert_dir, exist_ok=True)
    os.makedirs(wav32dir, exist_ok=True)

    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    cnhubert.cnhubert_base_path = args.cnhubert_base_dir
    model = cnhubert.get_model()
    if is_half:
        model = model.half().to(device)
    else:
        model = model.to(device)

    with open(args.inp_text, "r", encoding="utf8") as f:
        lines = f.read().strip("\n").split("\n")

    nan_fails = []
    for line in lines[int(i_part)::int(all_parts)]:
        try:
            wav_name, spk_name, language, text = line.split("|")
            if (args.inp_wav_dir != "" and args.inp_wav_dir != None):
                wav_name = os.path.basename(wav_name)
                wav_path = "%s/%s"%(args.inp_wav_dir, wav_name)
            else:
                wav_path = wav_name
                wav_name = os.path.basename(wav_name)
                
            success = name2go(wav_name, wav_path, model, hubert_dir, wav32dir, device, is_half)
            if success == False:
                nan_fails.append((wav_name, wav_path))
        except:
            print(line, traceback.format_exc())

    if len(nan_fails) > 0 and is_half:
        model = model.float()
        for wav_name, wav_path in nan_fails:
            try:
                name2go(wav_name, wav_path, model, hubert_dir, wav32dir, device, False)
            except:
                print(wav_name, traceback.format_exc())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get Hubert features and wav32k")
    parser.add_argument("-t", "--inp_text", required=True, help="Path to the input text file")
    parser.add_argument("-w", "--inp_wav_dir", required=True, help="Path to the input wav directory")
    parser.add_argument("-o", "--opt_dir", required=False, default="./output/test/fine_tune_dataset", help="Output directory")
    parser.add_argument("-b", "--cnhubert_base_dir", required=False, default="GPT_SoVITS/pretrained_models/chinese-hubert-base", help="Path to the cnhubert base directory")
    args = parser.parse_args()

    if "_CUDA_VISIBLE_DEVICES" in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("_CUDA_VISIBLE_DEVICES")

    is_half = torch.cuda.is_available()
    ngpu = torch.cuda.device_count()
    if ngpu == 0:
        all_parts = 1
    else:
        all_parts = ngpu

    for i_part in range(all_parts):
        process_audio(i_part, all_parts, args, is_half)
