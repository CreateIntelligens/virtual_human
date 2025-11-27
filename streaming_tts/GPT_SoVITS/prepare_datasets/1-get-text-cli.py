# -*- coding: utf-8 -*-

import os,sys
import warnings
warnings.filterwarnings("ignore")
import torch
import argparse
import traceback
import numpy as np
from text.cleaner import clean_text
from transformers import AutoModelForMaskedLM, AutoTokenizer
from time import time as ttime
import shutil

def my_save(fea, path, i_part):#####fix issue: torch.save doesn't support chinese path
    dir = os.path.dirname(path)
    name = os.path.basename(path)
    tmp_path = "%s%s.pth"%(ttime(),i_part)
    torch.save(fea,tmp_path)
    shutil.move(tmp_path,"%s/%s"%(dir,name))

def main(i_part, all_parts, args, is_half):
    txt_path = "%s/2-name2text-%s.txt" % (args.opt_dir, i_part)
    if os.path.exists(txt_path) == False:
        bert_dir = "%s/3-bert" % (args.opt_dir)
        os.makedirs(args.opt_dir, exist_ok=True)
        os.makedirs(bert_dir, exist_ok=True)
        
        if torch.cuda.is_available():
            device = "cuda:0"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        if os.path.exists(args.bert_pretrained_dir):...
        else:
            raise FileNotFoundError(args.bert_pretrained_dir)

        tokenizer = AutoTokenizer.from_pretrained(args.bert_pretrained_dir)
        bert_model = AutoModelForMaskedLM.from_pretrained(args.bert_pretrained_dir)
        if is_half:
            bert_model = bert_model.half().to(device)
        else:
            bert_model = bert_model.to(device)

        def get_bert_feature(text, word2ph):
            with torch.no_grad():
                inputs = tokenizer(text, return_tensors="pt")
                for i in inputs:
                    inputs[i] = inputs[i].to(device)
                res = bert_model(**inputs, output_hidden_states=True)
                res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]

            assert len(word2ph) == len(text)
            phone_level_feature = []
            for i in range(len(word2ph)):
                repeat_feature = res[i].repeat(word2ph[i], 1)
                phone_level_feature.append(repeat_feature)

            phone_level_feature = torch.cat(phone_level_feature, dim=0)
            return phone_level_feature.T

        def process(data, res):
            for name, text, lan in data:
                try:
                    name = os.path.basename(name)
                    phones, word2ph, norm_text = clean_text(
                        text.replace("%", "-").replace("￥", ","), lan
                    )
                    path_bert = "%s/%s.pt" % (bert_dir, name)
                    if os.path.exists(path_bert) == False and lan == "zh":
                        bert_feature = get_bert_feature(norm_text, word2ph)
                        assert bert_feature.shape[-1] == len(phones)
                        my_save(bert_feature, path_bert, i_part)
                    phones = " ".join(phones)
                    res.append([name, phones, word2ph, norm_text])
                except:
                    print(name, text, traceback.format_exc())

        todo = []
        res = []
        with open(args.inp_text, "r", encoding="utf8") as f:
            lines = f.read().strip("\n").split("\n")

        language_v1_to_language_v2 = {
            "ZH": "zh",
            "zh": "zh",
            "JP": "ja",
            "jp": "ja",
            "JA": "ja",
            "ja": "ja",
            "EN": "en",
            "en": "en",
            "En": "en",
        }
        for line in lines[int(i_part)::int(all_parts)]:
            try:
                wav_name, spk_name, language, text = line.split("|")
                todo.append(
                    [wav_name, text, language_v1_to_language_v2.get(language, language)]
                )
            except:
                print(line, traceback.format_exc())

        process(todo, res)
        opt = []
        for name, phones, word2ph, norm_text in res:
            opt.append("%s\t%s\t%s\t%s" % (name, phones, word2ph, norm_text))
        with open(txt_path, "w", encoding="utf8") as f:
            f.write("\n".join(opt) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get text features")
    parser.add_argument("-t", "--inp_text", required=True, help="Path to the input text file")
    parser.add_argument("-w", "--inp_wav_dir", required=True, help="Path to the wav file directory")
    parser.add_argument("-o", "--opt_dir", required=False, default="./output/test/fine_tune_dataset", help="Output directory")
    parser.add_argument("-b", "--bert_pretrained_dir", required=False, default="GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large", help="Path to the pretrained bert model")
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
        main(i_part, all_parts, args, is_half)

    # Combine all part files
    opt = []
    for i_part in range(all_parts):
        txt_path = "%s/2-name2text-%s.txt" % (args.opt_dir, i_part)
        with open(txt_path, "r", encoding="utf8") as f:
            opt += f.read().strip("\n").split("\n")
        os.remove(txt_path)
    path_text = "%s/2-name2text.txt" % args.opt_dir
    with open(path_text, "w", encoding="utf8") as f:
        f.write("\n".join(opt) + "\n")
