```bash
apt install python3.10-venv
python3.10 -m venv venv
source venv/bin/activate.fish
#如果cuda版本不为11.3(运行nvidia-smi确认版本)，根据<https://pytorch.org/get-started/previous-versions/>安装对应版本的pytorch 
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install ffmpeg
pip install --no-cache-dir -U openmim 
mim install mmengine 
mim install "mmcv==2.0.1" 
mim install "mmdet==3.2.0" 
mim install "mmpose==1.3.2"
``` 

```bash
export CANDIDATE='192.168.1.100'
docker run --rm --env CANDIDATE=$CANDIDATE \
  -p 1935:1935 -p 8080:8080 -p 1985:1985 -p 8000:8000/udp \
  registry.cn-hangzhou.aliyuncs.com/ossrs/srs:5 \
  objs/srs -c conf/rtc.conf
``` 

```bash
python app.py --transport rtcpush --model musetalk --avatar_id avator_2 --tts gpt-sovits --TTS_SERVER http://192.168.1.100:9880 --max_session 5
``` 

```bash
python app.py --transport rtcpush --model musetalk --avatar_id avator_1 --tts gpt-sovits --TTS_SERVER http://192.168.1.100:9880 --max_session 5 --avatar_list "avator_2,avator_10,avator_3"
``` 

```bash
python simple_musetalk.py --avatar_id 2  --file ../data/video/Professional_Mode1.mp4
```
