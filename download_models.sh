#!/bin/bash

# 模型自動下載腳本
# 使用方法: bash download_models.sh

set -e

echo "🚀 開始下載虛擬人系統模型..."

# 創建必要目錄
echo "📁 創建目錄結構..."
mkdir -p models/{dwpose,face-parse-bisent,musetalk,sd-vae-ft-mse,whisper}
mkdir -p streaming_tts/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large
mkdir -p streaming_tts/tools/asr/models/faster-whisper-large-v3-turbo
mkdir -p streaming_tts/tools/uvr5/uvr5_weights

# 下載 Wav2Lip 模型
echo "🎭 下載 Wav2Lip 模型..."
cd models
if [ ! -f "wav2lip.pth" ]; then
    echo "  下載 wav2lip.pth..."
    wget -O wav2lip.pth "https://github.com/Rudrabha/Wav2Lip/releases/download/v1.0/wav2lip.pth"
else
    echo "  wav2lip.pth 已存在，跳過下載"
fi

if [ ! -f "wav2lip_gan.pth" ]; then
    echo "  下載 wav2lip_gan.pth..."
    wget -O wav2lip_gan.pth "https://github.com/Rudrabha/Wav2Lip/releases/download/v1.0/wav2lip_gan.pth"
else
    echo "  wav2lip_gan.pth 已存在，跳過下載"
fi

# 下載 DWPose 模型
echo "🤸 下載 DWPose 模型..."
cd dwpose
if [ ! -f "dw-ll_ucoco_384.pth" ]; then
    echo "  下載 dw-ll_ucoco_384.pth..."
    wget -O dw-ll_ucoco_384.pth "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.pth"
else
    echo "  dw-ll_ucoco_384.pth 已存在，跳過下載"
fi

# 下載人臉解析模型
echo "👤 下載人臉解析模型..."
cd ../face-parse-bisent
if [ ! -f "79999_iter.pth" ]; then
    echo "  下載 79999_iter.pth..."
    wget -O 79999_iter.pth "https://github.com/zllrunning/face-parsing.PyTorch/releases/download/v1.0/79999_iter.pth"
else
    echo "  79999_iter.pth 已存在，跳過下載"
fi

if [ ! -f "resnet18-5c106cde.pth" ]; then
    echo "  下載 resnet18-5c106cde.pth..."
    wget -O resnet18-5c106cde.pth "https://download.pytorch.org/models/resnet18-5c106cde.pth"
else
    echo "  resnet18-5c106cde.pth 已存在，跳過下載"
fi

# 下載 MuseTalk 模型
echo "🎵 下載 MuseTalk 模型..."
cd ../musetalk
if [ ! -f "musetalk.json" ]; then
    echo "  下載 musetalk.json..."
    wget -O musetalk.json "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk.json"
else
    echo "  musetalk.json 已存在，跳過下載"
fi

if [ ! -f "pytorch_model.bin" ]; then
    echo "  下載 pytorch_model.bin..."
    wget -O pytorch_model.bin "https://huggingface.co/TMElyralab/MuseTalk/resolve/main/pytorch_model.bin"
else
    echo "  pytorch_model.bin 已存在，跳過下載"
fi

# 下載 VAE 模型
echo "🔄 下載 VAE 模型..."
cd ../sd-vae-ft-mse
if [ ! -f "config.json" ]; then
    echo "  下載 config.json..."
    wget -O config.json "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json"
else
    echo "  config.json 已存在，跳過下載"
fi

if [ ! -f "diffusion_pytorch_model.bin" ]; then
    echo "  下載 diffusion_pytorch_model.bin..."
    wget -O diffusion_pytorch_model.bin "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin"
else
    echo "  diffusion_pytorch_model.bin 已存在，跳過下載"
fi

if [ ! -f "diffusion_pytorch_model.safetensors" ]; then
    echo "  下載 diffusion_pytorch_model.safetensors..."
    wget -O diffusion_pytorch_model.safetensors "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.safetensors"
else
    echo "  diffusion_pytorch_model.safetensors 已存在，跳過下載"
fi

# 下載 Whisper 模型
echo "🎤 下載 Whisper 模型..."
cd ../whisper
if [ ! -f "tiny.pt" ]; then
    echo "  下載 tiny.pt..."
    wget -O tiny.pt "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e/tiny.pt"
else
    echo "  tiny.pt 已存在，跳過下載"
fi

# 下載 GPT-SoVITS 模型
echo "🗣️ 下載 GPT-SoVITS 模型..."
cd ../../streaming_tts/GPT_SoVITS/pretrained_models
if [ ! -f "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt" ]; then
    echo "  下載 s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt..."
    wget -O "s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt" "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/s1bert25hz-2kh-longer-epoch%3D68e-step%3D50232.ckpt"
else
    echo "  s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt 已存在，跳過下載"
fi

if [ ! -f "s2G488k.pth" ]; then
    echo "  下載 s2G488k.pth..."
    wget -O s2G488k.pth "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/s2G488k.pth"
else
    echo "  s2G488k.pth 已存在，跳過下載"
fi

if [ ! -f "s2D488k.pth" ]; then
    echo "  下載 s2D488k.pth..."
    wget -O s2D488k.pth "https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/s2D488k.pth"
else
    echo "  s2D488k.pth 已存在，跳過下載"
fi

# 下載中文 BERT 模型
echo "🔤 下載中文 BERT 模型..."
cd chinese-roberta-wwm-ext-large
if [ ! -f "config.json" ]; then
    echo "  下載 config.json..."
    wget -O config.json "https://huggingface.co/hfl/chinese-roberta-wwm-ext-large/resolve/main/config.json"
else
    echo "  config.json 已存在，跳過下載"
fi

if [ ! -f "pytorch_model.bin" ]; then
    echo "  下載 pytorch_model.bin..."
    wget -O pytorch_model.bin "https://huggingface.co/hfl/chinese-roberta-wwm-ext-large/resolve/main/pytorch_model.bin"
else
    echo "  pytorch_model.bin 已存在，跳過下載"
fi

if [ ! -f "tokenizer.json" ]; then
    echo "  下載 tokenizer.json..."
    wget -O tokenizer.json "https://huggingface.co/hfl/chinese-roberta-wwm-ext-large/resolve/main/tokenizer.json"
else
    echo "  tokenizer.json 已存在，跳過下載"
fi

if [ ! -f "vocab.txt" ]; then
    echo "  下載 vocab.txt..."
    wget -O vocab.txt "https://huggingface.co/hfl/chinese-roberta-wwm-ext-large/resolve/main/vocab.txt"
else
    echo "  vocab.txt 已存在，跳過下載"
fi

# 下載 UVR5 模型
echo "🎵 下載 UVR5 音頻分離模型..."
cd ../../../tools/uvr5/uvr5_weights
if [ ! -f "HP2-人声vocals+非人声instrumentals.pth" ]; then
    echo "  下載 HP2-人声vocals+非人声instrumentals.pth..."
    wget -O "HP2-人声vocals+非人声instrumentals.pth" "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/HP2-人声vocals+非人声instrumentals.pth"
else
    echo "  HP2-人声vocals+非人声instrumentals.pth 已存在，跳過下載"
fi

if [ ! -f "HP5-主旋律人声vocals+其他instrumentals.pth" ]; then
    echo "  下載 HP5-主旋律人声vocals+其他instrumentals.pth..."
    wget -O "HP5-主旋律人声vocals+其他instrumentals.pth" "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/HP5-主旋律人声vocals+其他instrumentals.pth"
else
    echo "  HP5-主旋律人声vocals+其他instrumentals.pth 已存在，跳過下載"
fi

echo "✅ 所有模型下載完成！"
echo "📊 檢查下載的文件..."

# 返回專案根目錄
cd ../../../../

# 檢查文件大小
echo "📁 模型文件大小："
if [ -d "models" ]; then
    echo "  models/: $(du -sh models/ 2>/dev/null | cut -f1)"
fi
if [ -d "streaming_tts/GPT_SoVITS" ]; then
    echo "  streaming_tts/GPT_SoVITS/: $(du -sh streaming_tts/GPT_SoVITS/ 2>/dev/null | cut -f1)"
fi
if [ -d "streaming_tts/tools" ]; then
    echo "  streaming_tts/tools/: $(du -sh streaming_tts/tools/ 2>/dev/null | cut -f1)"
fi

echo ""
echo "🎉 模型下載和設置完成！"
echo "💡 提示："
echo "   - 某些模型可能需要在首次運行時自動下載"
echo "   - 如果下載失敗，請檢查網路連接並重新運行此腳本"
echo "   - 建議將模型文件存放在 SSD 上以提高載入速度"
echo ""
echo "📝 下一步："
echo "   1. 配置 IP 地址（參考 README.md）"
echo "   2. 啟動 Docker 服務：docker-compose up -d"
echo "   3. 訪問 http://<YOUR_IP>:8010/rtcpushapi_google_360_test.html"
