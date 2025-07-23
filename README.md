# 🤖 AI Virtual Human Live Talking System

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![CUDA](https://img.shields.io/badge/CUDA-Required-orange.svg)](https://developer.nvidia.com/cuda-downloads)

一個基於 AI 的即時虛擬人對話系統，支援多種渲染技術、語音合成和 WebRTC 串流。

## ✨ 主要特色

- 🎭 **多種渲染技術**: 支援 MuseTalk、Wav2Lip、ER-NeRF、UltraLight 等模型
- 🗣️ **高品質語音合成**: 基於 GPT-SoVITS 的多語音配置
- 🎥 **即時串流**: WebRTC 技術實現低延遲視頻串流
- 🎯 **多會話支援**: 同時處理多個用戶對話
- 🔄 **動態切換**: 即時更換虛擬人和語音類型
- 🌐 **Web 界面**: 直觀的瀏覽器操作界面
- 🐳 **容器化部署**: Docker 一鍵部署

## 🏗️ 系統架構

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Frontend  │    │   Main App      │    │   TTS Service   │
│   (WebRTC)      │◄──►│   (Flask)       │◄──►│  (GPT-SoVITS)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         └──────────────►│   SRS Server    │◄─────────────┘
                        │   (WebRTC)      │
                        └─────────────────┘
```

## 🚀 快速開始

### 系統需求

- **操作系統**: Linux (推薦 Ubuntu 20.04+)
- **GPU**: NVIDIA GPU with CUDA 11.3+ (推薦 RTX 3080 或更高)
- **內存**: 16GB+ RAM
- **存儲**: 50GB+ 可用空間
- **軟體**: Docker, Docker Compose, Git

### 部署方式選擇

本系統提供兩種部署方式：

#### 🐳 **Docker 部署 (推薦)**
- ✅ 一鍵部署，環境隔離
- ✅ 自動處理依賴和配置
- ✅ 支援 GPU 加速
- ✅ 包含完整的服務編排

#### 🔧 **手動部署**
- 適合開發和自定義需求
- 需要手動安裝依賴
- 更靈活的配置選項

---

## 🐳 Docker 部署 (推薦)

### 前置需求

```bash
# 安裝 Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# 安裝 Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 安裝 NVIDIA Docker (GPU 支援)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### 快速部署

```bash
# 1. 克隆專案
git clone <your-repository-url>
cd virtual_human

# 2. 配置 IP 地址 (將 192.168.1.100 替換為你的實際 IP)
./configure_ip.sh 192.168.1.100

# 3. 設置環境變數
export CANDIDATE=192.168.1.100

# 4. 啟動所有服務
docker-compose up -d

# 5. 檢查服務狀態
docker-compose ps

# 6. 查看日誌
docker-compose logs -f
```

### 訪問系統

服務啟動後，可以通過以下地址訪問：

- **主界面**: `http://192.168.1.100:8010/rtcpushapi_google_360_test.html`
- **SRS 管理**: `http://192.168.1.100:8080`
- **健康檢查**: `http://192.168.1.100:8010/health`

### Docker 服務管理

```bash
# 啟動服務
docker-compose up -d

# 停止服務
docker-compose down

# 重啟服務
docker-compose restart

# 查看服務狀態
docker-compose ps

# 查看日誌
docker-compose logs -f app
docker-compose logs -f srs

# 進入容器
docker-compose exec app bash

# 更新服務
docker-compose pull
docker-compose up -d
```

---

## 🔧 手動部署

### 1. 克隆專案

```bash
git clone <your-repository-url>
cd virtual_human
```

### 2. 準備虛擬人模型

> ⚠️ **重要**: 虛擬人模型需要自行訓練或準備

#### 訓練虛擬人模型

使用 MuseTalk 訓練你的虛擬人：

```bash
# 準備訓練視頻（建議 1-5 分鐘，清晰的正面人臉視頻）
# 將視頻放置在 data/video/ 目錄下

# 使用 MuseTalk 訓練
cd musetalk
python simple_musetalk.py --avatar_id avator_1 --file ../data/video/your_video.mp4

# 訓練完成後，模型會自動保存到 data/avatars/avator_1/
```

#### 虛擬人模型結構

每個虛擬人模型應包含以下文件：

```
data/avatars/avator_1/
├── full_imgs/          # 完整圖像序列
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
├── coords.pkl          # 座標信息
├── latents.pt         # 潛在空間表示
├── mask/              # 遮罩圖像
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
├── mask_coords.pkl    # 遮罩座標
└── avator_info.json   # 虛擬人信息
```

#### 配置虛擬人

在 `.env` 文件中配置虛擬人：

```bash
# 主要使用的虛擬人（必須存在於 data/avatars/ 中）
AVATAR_ID=avator_1

# 可選的虛擬人列表（用戶可在界面中切換）
AVATAR_LIST=avator_1,avator_2,avator_3

# 如果只有一個虛擬人
AVATAR_ID=avator_1
AVATAR_LIST=avator_1  # 界面中只會顯示一個選項
```

### 3. 下載基礎模型文件

> ⚠️ **重要**: 基礎模型文件未包含在此倉庫中，需要單獨下載

#### 虛擬人渲染模型

參考 [LiveTalking](https://github.com/lipku/LiveTalking) 下載以下模型：

```bash
# 創建模型目錄
mkdir -p models

# 下載 MuseTalk 模型
cd models
wget <musetalk-model-url> -O musetalk.pth

# 下載 Wav2Lip 模型
wget <wav2lip-model-url> -O wav2lip.pth
wget <wav2lip-gan-model-url> -O wav2lip_gan.pth

# 下載其他必要模型
# 詳細下載連結請參考 LiveTalking 專案說明
```

#### TTS 語音模型

##### 1. 安裝 FFmpeg

> ⚠️ **重要**: FFmpeg 是音頻處理的必要組件

```bash
# 方法 1: 下載靜態構建版本 (推薦)
cd streaming_tts/ffmpeg
wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
tar -xf ffmpeg-release-amd64-static.tar.xz
mv ffmpeg-*-amd64-static/* .
rm -rf ffmpeg-*-amd64-static*

# 方法 2: 使用系統包管理器
sudo apt install ffmpeg  # Ubuntu/Debian
# 然後創建符號連結
mkdir -p streaming_tts/ffmpeg/bin
ln -s /usr/bin/ffmpeg streaming_tts/ffmpeg/bin/ffmpeg
ln -s /usr/bin/ffprobe streaming_tts/ffmpeg/bin/ffprobe

# 驗證安裝
streaming_tts/ffmpeg/bin/ffmpeg -version
```

##### 2. 下載基礎模型

參考 [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) 下載基礎模型：

```bash
# 下載 GPT-SoVITS 預訓練模型到 pretrained_models/
cd streaming_tts/GPT_SoVITS/pretrained_models
# 下載以下模型：
# - chinese-roberta-wwm-ext-large
# - chinese-hubert-base  
# - s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt
# - s2G488k.pth

# 下載 ASR 模型
cd ../../tools/asr/models
# 下載 Whisper 或其他 ASR 模型

# 下載 UVR5 模型
cd ../uvr5/uvr5_weights
# 下載音頻分離模型
```

##### 2. 訓練自定義語音模型

使用 GPT-SoVITS 訓練你的語音模型：

```bash
# 準備語音數據（建議 10-30 分鐘的清晰語音）
# 將音頻文件放置在適當目錄

# 使用 GPT-SoVITS WebUI 訓練
cd streaming_tts
python GPT_SoVITS/inference_webui.py

# 或使用命令行訓練
# 詳細步驟請參考 GPT-SoVITS 官方文檔
```

##### 3. 配置訓練好的模型

訓練完成後，將模型放置到正確位置：

```bash
# 模型文件結構
streaming_tts/output/
├── YYYY/MM/DD/YOUR_MODEL_ID/
│   ├── gpt_models/
│   │   └── YOUR_MODEL_ID-e10.ckpt
│   └── sovits_models/
│       └── YOUR_MODEL_ID_e8_s72.pth
```

修改對應的 YAML 配置文件：

```bash
# 根據語音 ID 修改對應的配置文件
# voice_0 對應 tts_infer.yaml
# voice_1 對應 tts_infer1.yaml  
# voice_2 對應 tts_infer2.yaml

# 修改 custom 部分的模型路徑
vim streaming_tts/GPT_SoVITS/configs/tts_infer.yaml
```

YAML 配置範例：
```yaml
custom:
  t2s_weights_path: ./output/2025/06/02/your_model_id/gpt_models/your_model_id-e10.ckpt
  vits_weights_path: ./output/2025/06/02/your_model_id/sovits_models/your_model_id_e8_s72.pth
```

### 3. 配置 IP 地址

> 📝 **重要**: 將所有配置文件中的 IP 地址改為你的實際 IP

#### 修改主要配置文件

```bash
# 1. 修改 avatar_voice.json 中的 TTS 服務器地址
sed -i 's/<OLD_IP>/<YOUR_IP>/g' avatar_voice.json

# 2. 修改 streaming_tts/api_v3_live.py 中的 API 地址
sed -i 's/<OLD_DOMAIN>/<YOUR_IP>/g' streaming_tts/api_v3_live.py

# 3. 修改前端文件中的 API 地址
sed -i 's/<OLD_DOMAIN>/<YOUR_IP>/g' web/rtcpushapi_google_360_test.html

# 4. 修改 environment.md 中的示例 IP
sed -i 's/<OLD_IP>/<YOUR_IP>/g' environment.md
```

### 4. Docker 部署

```bash
# 設置環境變數
export CANDIDATE='<YOUR_IP>'

# 啟動服務
docker-compose up -d

# 查看服務狀態
docker-compose ps

# 查看日誌
docker-compose logs -f
```

### 5. 訪問系統

打開瀏覽器訪問：
- **主界面**: `http://<YOUR_IP>:8010/rtcpushapi_google_360_test.html`
- **SRS 管理**: `http://<YOUR_IP>:8080`
- **健康檢查**: `http://<YOUR_IP>:8010/health`

## 📖 詳細安裝 (手動部署)

### 環境準備

```bash
# 安裝 Python 虛擬環境
apt install python3.10-venv
python3.10 -m venv venv
source venv/bin/activate

# 安裝 PyTorch (根據你的 CUDA 版本調整)
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# 安裝依賴
pip install -r requirements.txt
pip install ffmpeg
pip install --no-cache-dir -U openmim 
mim install mmengine 
mim install "mmcv==2.0.1" 
mim install "mmdet==3.2.0" 
mim install "mmpose==1.3.2"
```

### 啟動 SRS 媒體服務器

```bash
export CANDIDATE='<YOUR_IP>'
docker run --rm --env CANDIDATE=$CANDIDATE \
  -p 1935:1935 -p 8080:8080 -p 1985:1985 -p 8000:8000/udp \
  registry.cn-hangzhou.aliyuncs.com/ossrs/srs:5 \
  objs/srs -c conf/rtc.conf
```

### 啟動主應用

```bash
# 單一虛擬人模式
python app.py --transport rtcpush --model musetalk --avatar_id avator_1 --tts gpt-sovits --TTS_SERVER http://<YOUR_IP>:9880 --max_session 5

# 多虛擬人模式
python app.py --transport rtcpush --model musetalk --avatar_id avator_1 --tts gpt-sovits --TTS_SERVER http://<YOUR_IP>:9880 --max_session 5 --avatar_list "avator_2,avator_10,avator_3"
```

### 啟動 TTS 服務

```bash
cd streaming_tts
python api_v3_live.py -p 9885 -a 0.0.0.0
```

## 🎮 使用說明

### Web 界面操作

1. **選擇虛擬人**: 使用下拉選單切換不同的虛擬人角色
2. **選擇語音**: 切換不同的語音類型 (Hayley, 小安, Cindy)
3. **語音對話**: 
   - 點擊「開始對話」或按 `R` 鍵開始錄音
   - 點擊「停止對話」或按 `T` 鍵停止錄音
   - 按空白鍵快速開始/停止錄音
4. **文字輸入**: 在底部輸入框直接輸入文字進行對話
5. **快速介紹**: 點擊「快速介紹」按鈕讓虛擬人自我介紹

### 快捷鍵

- `R`: 開始錄音
- `T`: 停止錄音  
- `Space`: 開始/停止錄音切換
- `Enter`: 發送文字訊息

### API 接口

#### 虛擬人對話
```bash
curl -X POST http://<YOUR_IP>:8010/human \
  -H "Content-Type: application/json" \
  -d '{
    "text": "你好",
    "type": "echo",
    "sessionid": 0
  }'
```

#### 切換虛擬人
```bash
curl -X POST http://<YOUR_IP>:8010/switch_avatar \
  -H "Content-Type: application/json" \
  -d '{
    "sessionid": 0,
    "avatar_id": "avator_2"
  }'
```

#### 切換語音
```bash
curl -X POST http://<YOUR_IP>:8010/switch_tts_endpoint \
  -H "Content-Type: application/json" \
  -d '{
    "sessionid": 0,
    "config_type": 1
  }'
```

## 🔧 配置說明

### 虛擬人配置 (avatar_name.json)

```json
{
    "avator_1": {
        "id": "avator_1",
        "name": "虛擬人1"
    }
}
```

### 語音配置 (avatar_voice.json)

```json
{
    "voice_0": {
        "id": 0,
        "name": "Hayley",
        "tts_server": "http://<YOUR_IP>:9885/",
        "config": {
            "ref_audio_path": "path/to/reference.wav",
            "prompt_text": "參考文本",
            "speed_factor": 1.0
        }
    }
}
```

### 環境變數

| 變數名 | 說明 | 預設值 |
|--------|------|--------|
| `TRANSPORT` | 傳輸模式 | `rtcpush` |
| `MODEL` | 渲染模型 | `musetalk` |
| `AVATAR_ID` | 預設虛擬人 | `avator_1` |
| `TTS_SERVER` | TTS 服務器地址 | `http://192.168.1.100:9885` |
| `MAX_SESSION` | 最大會話數 | `5` |

## 🐳 Docker 配置

### docker-compose.yaml 說明

系統包含三個主要服務：
- **srs**: WebRTC 媒體服務器
- **app**: 主應用服務 (需要 GPU)
- **TTS 服務**: 在 app 容器內運行

### GPU 支援

確保 Docker 支援 NVIDIA GPU：

```bash
# 安裝 nvidia-docker2
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## 🔍 故障排除

### 常見問題

#### 1. GPU 記憶體不足
```bash
# 減少批次大小
python app.py --batch_size 8

# 或使用 CPU 模式（性能較差）
export CUDA_VISIBLE_DEVICES=""
```

#### 2. WebRTC 連接失敗
```bash
# 檢查防火牆設定
sudo ufw allow 1985
sudo ufw allow 8000/udp

# 檢查 SRS 服務狀態
docker logs <srs-container-id>
```

#### 3. TTS 服務無法啟動
```bash
# 檢查模型文件是否存在
ls -la streaming_tts/GPT_SoVITS/

# 檢查端口是否被佔用
netstat -tulpn | grep 9885
```

#### 4. 虛擬人無法載入
```bash
# 檢查虛擬人資源
ls -la data/avatars/avator_1/

# 確認必要文件存在
# - full_imgs/
# - coords.pkl
# - latents.pt
# - mask/
# - mask_coords.pkl
```

### Docker 相關問題

#### 5. Docker 服務無法啟動
```bash
# 檢查 Docker 服務狀態
sudo systemctl status docker

# 檢查 NVIDIA Docker 支援
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# 檢查 docker-compose 配置
docker-compose config
```

#### 6. 容器內 GPU 無法使用
```bash
# 驗證 GPU 支援
docker-compose exec app nvidia-smi

# 檢查 CUDA 環境
docker-compose exec app python -c "import torch; print(torch.cuda.is_available())"
```

#### 7. 端口衝突
```bash
# 檢查端口使用情況
sudo netstat -tulpn | grep -E "(8010|9885|1985|8080)"

# 修改 docker-compose.yaml 中的端口映射
# 例如：將 8010:8010 改為 8011:8010
```

#### 8. 容器記憶體不足
```bash
# 檢查容器資源使用
docker stats

# 增加 Docker 記憶體限制
# 在 docker-compose.yaml 中添加：
# deploy:
#   resources:
#     limits:
#       memory: 8G
```

### 日誌查看

```bash
# Docker 日誌
docker-compose logs -f app
docker-compose logs -f srs

# 查看特定時間的日誌
docker-compose logs --since="1h" app

# 應用日誌
tail -f logs/app.log

# TTS 服務日誌
tail -f logs/tts.log

# 實時監控容器狀態
watch docker-compose ps
```

### Docker 環境重置

```bash
# 完全重置環境
docker-compose down -v
docker system prune -a
docker volume prune

# 重新構建並啟動
docker-compose build --no-cache
docker-compose up -d
```

### 性能優化

1. **GPU 記憶體優化**:
   - 調整 `batch_size` 參數
   - 使用混合精度訓練

2. **網路優化**:
   - 確保足夠的頻寬
   - 優化 WebRTC 設定

3. **存儲優化**:
   - 使用 SSD 存儲模型文件
   - 定期清理日誌文件

## 📚 進階功能

### 自定義虛擬人

1. 準備虛擬人視頻素材
2. 使用 MuseTalk 處理工具：
```bash
python simple_musetalk.py --avatar_id 2 --file path/to/video.mp4
```

### 自定義語音

1. 準備語音訓練數據
2. 使用 GPT-SoVITS 訓練：
```bash
# 參考 GPT-SoVITS 專案文檔進行語音訓練
```

### API 擴展

系統提供完整的 RESTful API，支援：
- 虛擬人管理
- 語音配置
- 會話控制
- 健康監控

## 🤝 貢獻指南

1. Fork 此專案
2. 創建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 開啟 Pull Request

## 📄 授權條款

此專案採用 Apache 2.0 授權條款 - 詳見 [LICENSE](LICENSE) 文件

## 🙏 致謝

- [LiveTalking](https://github.com/lipku/LiveTalking) - 虛擬人渲染技術
- [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) - 語音合成技術
- [SRS](https://github.com/ossrs/srs) - WebRTC 媒體服務器

## 📞 支援

如果你遇到問題或有建議，請：
1. 查看 [故障排除](#-故障排除) 章節
2. 搜索現有的 [Issues](../../issues)
3. 創建新的 Issue 描述你的問題

---

⭐ 如果這個專案對你有幫助，請給我們一個星星！
