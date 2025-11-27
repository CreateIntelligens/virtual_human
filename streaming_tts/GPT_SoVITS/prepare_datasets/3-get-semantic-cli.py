# -*- coding: utf-8 -*-

import os, sys
import warnings
warnings.filterwarnings("ignore")
import torch
import argparse
import math, traceback
import multiprocessing
from random import shuffle
import torch.multiprocessing as mp
from glob import glob
from tqdm import tqdm
import logging, librosa, utils
from module.models import SynthesizerTrn
logging.getLogger("numba").setLevel(logging.WARNING)

def main(i_part, all_parts, args, is_half):
    if os.path.exists(args.pretrained_s2G):...
    else:
        raise FileNotFoundError(args.pretrained_s2G)

    hubert_dir = "%s/4-cnhubert" % (args.opt_dir)
    semantic_path = "%s/6-name2semantic-%s.tsv" % (args.opt_dir, i_part)
    if os.path.exists(semantic_path) == False:
        os.makedirs(args.opt_dir, exist_ok=True)

        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        hps = utils.get_hparams_from_file(args.s2config_path)
        vq_model = SynthesizerTrn(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            **hps.model
        )
        if is_half:
            vq_model = vq_model.half().to(device)
        else:
            vq_model = vq_model.to(device)
        vq_model.eval()
        print(
            vq_model.load_state_dict(
                torch.load(args.pretrained_s2G, map_location="cpu")["weight"], strict=False
            )
        )

        def name2go(wav_name, lines):
            hubert_path = "%s/%s.pt" % (hubert_dir, wav_name)
            if os.path.exists(hubert_path) == False:
                return
            ssl_content = torch.load(hubert_path, map_location="cpu")
            if is_half:
                ssl_content = ssl_content.half().to(device)
            else:
                ssl_content = ssl_content.to(device)
            codes = vq_model.extract_latent(ssl_content)
            semantic = " ".join([str(i) for i in codes[0, 0, :].tolist()])
            lines.append("%s\t%s" % (wav_name, semantic))

        with open(args.inp_text, "r", encoding="utf8") as f:
            lines = f.read().strip("\n").split("\n")

        lines1 = []
        for line in lines[int(i_part)::int(all_parts)]:
            try:
                wav_name, spk_name, language, text = line.split("|")
                wav_name = os.path.basename(wav_name)
                name2go(wav_name, lines1)
            except:
                print(line, traceback.format_exc())
        with open(semantic_path, "w", encoding="utf8") as f:
            f.write("\n".join(lines1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get semantic features")
    parser.add_argument("-t", "--inp_text", required=True, help="Path to the input text file")
    parser.add_argument("-o", "--opt_dir", required=False, default="./output/test/fine_tune_dataset", help="Output directory")
    parser.add_argument("-g", "--pretrained_s2G", required=False, default="GPT_SoVITS/pretrained_models/s2G488k.pth", help="Path to the pretrained s2G model")
    parser.add_argument("-p", "--s2config_path", required=False, default="GPT_SoVITS/configs/s2.json", help="Path to the s2 config file")
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
    opt = ["item_name\tsemantic_audio"]
    path_semantic = "%s/6-name2semantic.tsv" % args.opt_dir
    for i_part in range(all_parts):
        semantic_path = "%s/6-name2semantic-%s.tsv" % (args.opt_dir, i_part)
        with open(semantic_path, "r", encoding="utf8") as f:
            opt += f.read().strip("\n").split("\n")
        os.remove(semantic_path)
    with open(path_semantic, "w", encoding="utf8") as f:
        f.write("\n".join(opt) + "\n")
