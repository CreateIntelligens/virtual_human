
```bash
sudo apt update && sudo apt install -y libcudnn8 libcudnn8-dev
sudo apt install ffmpeg
sudo apt install libsox-dev
apt install python3.10-venv
python3.10 -m venv venv
source venv/bin/activate.fish
#如果cuda版本不为11.3(运行nvidia-smi确认版本)，根据<https://pytorch.org/get-started/previous-versions/>安装对应版本的pytorch 
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install ffmpeg

``` 

```bash
python api_v2.py -p 9884 -a 0.0.0.0
``` 
