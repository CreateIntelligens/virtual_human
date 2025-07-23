# 📦 模型下載指南

本文檔詳細說明如何下載和配置虛擬人系統所需的各種模型文件。

## 📋 模型概覽

### 必需模型
- **虛擬人渲染模型**: MuseTalk, Wav2Lip, ER-NeRF, UltraLight
- **TTS 語音模型**: GPT-SoVITS 預訓練模型
- **ASR 語音識別模型**: Whisper, Faster-Whisper
- **輔助模型**: 人臉檢測、姿態估計、音頻分離等

### 存儲需求
- **總空間**: 約 30-50GB
- **虛擬人模型**: 10-20GB
- **TTS 模型**: 5-10GB
- **ASR 模型**: 3-5GB
- **輔助模型**: 5-10GB

## 🎭 虛擬人渲染模型

### 參考來源
主要參考 [LiveTalking](https://github.com/lipku/LiveTalking) 專案

### 1. MuseTalk 模型

```bash
# 創建模型目錄
mkdir -p models/musetalk

# 下載 MuseTalk 主模型
cd models/musetalk
wget https://huggingface.co/TMElyralab/MuseTalk/resolve/main/musetalk.json
wget https://huggingface.co/TMElyralab/MuseTalk/resolve/main/pytorch_model.bin

# 下載 VAE 模型
mkdir -p ../sd-vae-ft-mse
cd ../sd-vae-ft-mse
wget https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json
wget https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin
wget https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.safetensors
```

### 2. Wav2Lip 模型

```bash
# 創建目錄
mkdir -p models

# 下載 Wav2Lip 模型
cd models
wget https://github.com/Rudrabha/Wav2Lip/releases/download/v1.0/wav2lip.pth
wget https://github.com/Rudrabha/Wav2Lip/releases/download/v1.0/wav2lip_gan.pth

# 或從 Google Drive 下載
# wav2lip.pth: https://drive.google.com/file/d/1fQtBSYEyuai9MjBOF8j7zZ9nEy_CLkAb/view
# wav2lip_gan.pth: https://drive.google.com/file/d/1fEHHy-DbkLL-ue4k0sOcFrpd9R6XCWuZ/view
```

### 3. 人臉相關模型

```bash
# DWPose 模型
mkdir -p models/dwpose
cd models/dwpose
wget https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.pth

# 人臉解析模型
mkdir -p ../face-parse-bisent
cd ../face-parse-bisent
wget https://github.com/zllrunning/face-parsing.PyTorch/releases/download/v1.0/79999_iter.pth
wget https://download.pytorch.org/models/resnet18-5c106cde.pth
```

### 4. Whisper 模型

```bash
# Whisper 模型
mkdir -p models/whisper
cd models/whisper
wget https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e/tiny.pt

# 或下載其他大小的模型
# base.pt: https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt
# small.pt: https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19a8717b3f3bbc1a1e6c1e/small.pt
```

## 🗣️ TTS 語音模型

### 參考來源
主要參考 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 專案

### 1. GPT-SoVITS 預訓練模型

```bash
# 創建 GPT-SoVITS 目錄
mkdir -p streaming_tts/GPT_SoVITS/pretrained_models

# 下載 GPT 預訓練模型
cd streaming_tts/GPT_SoVITS/pretrained_models
wget https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/s1bert25hz-2kh-longer-epoch%3D68e-step%3D50232.ckpt
wget https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/s2G488k.pth
wget https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/s2D488k.pth

# 下載中文 BERT 模型
mkdir -p chinese-roberta-wwm-ext-large
cd chinese-roberta-wwm-ext-large
wget https://huggingface.co/hfl/chinese-roberta-wwm-ext-large/resolve/main/config.json
wget https://huggingface.co/hfl/chinese-roberta-wwm-ext-large/resolve/main/pytorch_model.bin
wget https://huggingface.co/hfl/chinese-roberta-wwm-ext-large/resolve/main/tokenizer.json
wget https://huggingface.co/hfl/chinese-roberta-wwm-ext-large/resolve/main/vocab.txt
```

### 2. ASR 模型

```bash
# 創建 ASR 模型目錄
mkdir -p streaming_tts/tools/asr/models

# 下載 Faster-Whisper 模型
cd streaming_tts/tools/asr/models
# 這些模型會在首次運行時自動下載，或手動下載：

# Faster-Whisper Large-v3-turbo
mkdir -p faster-whisper-large-v3-turbo
cd faster-whisper-large-v3-turbo
wget https://huggingface.co/Systran/faster-whisper-large-v3-turbo/resolve/main/config.json
wget https://huggingface.co/Systran/faster-whisper-large-v3-turbo/resolve/main/model.bin
wget https://huggingface.co/Systran/faster-whisper-large-v3-turbo/resolve/main/tokenizer.json
wget https://huggingface.co/Systran/faster-whisper-large-v3-turbo/resolve/main/vocabulary.txt
```

### 3. UVR5 音頻分離模型

```bash
# 創建 UVR5 模型目錄
mkdir -p streaming_tts/tools/uvr5/uvr5_weights

# 下載 UVR5 模型
cd streaming_tts/tools/uvr5/uvr5_weights
wget https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/HP2-人声vocals+非人声instrumentals.pth
wget https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/HP5-主旋律人声vocals+其他instrumentals.pth
```

## 📁 目錄結構

下載完成後，你的模型目錄結構應該如下：

```
models/
├── put models here.txt
├── wav2lip.pth
├── wav2lip_gan.pth
├── dwpose/
│   └── dw-ll_ucoco_384.pth
├── face-parse-bisent/
│   ├── 79999_iter.pth
│   └── resnet18-5c106cde.pth
├── musetalk/
│   ├── musetalk.json
│   └── pytorch_model.bin
├── sd-vae-ft-mse/
│   ├── config.json
│   ├── diffusion_pytorch_model.bin
│   └── diffusion_pytorch_model.safetensors
└── whisper/
    └── tiny.pt

streaming_tts/
├── GPT_SoVITS/
│   └── pretrained_models/
│       ├── s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt
│       ├── s2G488k.pth
│       ├── s2D488k.pth
│       └── chinese-roberta-wwm-ext-large/
│           ├── config.json
│           ├── pytorch_model.bin
│           ├── tokenizer.json
│           └── vocab.txt
├── tools/
│   ├── asr/
│   │   └── models/
│   │       └── faster-whisper-large-v3-turbo/
│   │           ├── config.json
│   │           ├── model.bin
│   │           ├── tokenizer.json
│   │           └── vocabulary.txt
│   └── uvr5/
│       └── uvr5_weights/
│           ├── HP2-人声vocals+非人声instrumentals.pth
│           └── HP5-主旋律人声vocals+其他instrumentals.pth
```

## 🚀 自動下載腳本

為了方便使用，請參考專案根目錄的 `download_models.sh` 腳本：

```bash
# 使用自動下載腳本
chmod +x download_models.sh
./download_models.sh
```

## ⚠️ 注意事項

### 1. 網路連接
- 模型文件較大，建議使用穩定的網路連接
- 如果下載中斷，可以重新運行腳本（會跳過已下載的文件）

### 2. 存儲空間
- 確保有足夠的磁碟空間（至少 50GB）
- 建議使用 SSD 以提高模型載入速度

### 3. 下載來源
- 某些模型可能需要從 Hugging Face 下載，可能需要設置代理
- 如果官方連結失效，請查看對應專案的最新下載連結

### 4. 模型版本
- 建議使用文檔中指定的模型版本以確保相容性
- 如需使用其他版本，可能需要調整配置文件

## 🔧 故障排除

### 下載失敗
```bash
# 如果 wget 失敗，可以嘗試使用 curl
curl -L -o model_file.pth "https://example.com/model.pth"

# 或使用 aria2c 進行多線程下載
aria2c -x 16 -s 16 "https://example.com/model.pth"
```

### 模型文件損壞
```bash
# 檢查文件完整性
md5sum model_file.pth

# 重新下載損壞的文件
rm model_file.pth
wget "https://example.com/model.pth"
```

### 空間不足
```bash
# 檢查磁碟空間
df -h

# 清理不必要的文件
docker system prune -a
```

## 📚 模型說明

### MuseTalk
- **用途**: 高品質面部表情和唇形同步
- **特點**: 支援多語言、表情自然
- **大小**: 約 2-3GB

### Wav2Lip
- **用途**: 唇形同步
- **特點**: 快速、輕量級
- **大小**: 約 200MB

### GPT-SoVITS
- **用途**: 高品質語音合成
- **特點**: 支援多語音、情感表達
- **大小**: 約 5-8GB

### Whisper
- **用途**: 語音識別
- **特點**: 多語言支援、高準確率
- **大小**: 約 1-3GB（依版本而定）

## 🔄 模型更新

定期檢查模型更新：

```bash
# 檢查 LiveTalking 專案更新
git -C LiveTalking pull

# 檢查 GPT-SoVITS 專案更新
git -C GPT-SoVITS pull

# 重新運行下載腳本獲取最新模型
./download_models.sh
```

## 💡 優化建議

1. **使用 SSD**: 將模型存放在 SSD 上以提高載入速度
2. **記憶體管理**: 確保有足夠的 RAM 載入模型
3. **GPU 記憶體**: 根據 GPU 記憶體調整批次大小
4. **網路代理**: 如果下載速度慢，考慮使用代理或鏡像站點

---

📝 **提示**: 首次運行系統時，某些模型可能會自動下載。請確保網路連接穩定。
