FROM python:3.10-slim

WORKDIR /app

# 設置環境變量
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive
ENV CXXFLAGS="-std=c++17"
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
ENV CUDA_HOME=/usr/local/cuda

# 添加 NVIDIA repository
RUN apt-get update && apt-get install -y --no-install-recommends gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    apt-get update

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    netcat-traditional \
    build-essential \  
    wget \
    jq \
    # cmake \  
    ninja-build \  
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    python3-dev \
    pkg-config \
    git \
    libssl-dev \
    libffi-dev \
    # 添加 CUDNN 相關套件
    libcudnn8 \
    libcudnn8-dev \
    && rm -rf /var/lib/apt/lists/*



# 安裝較新版本的 CMake (選擇性步驟，若需要更新版本)
RUN wget https://github.com/Kitware/CMake/releases/download/v3.24.1/cmake-3.24.1-Linux-x86_64.sh \
    -q -O /tmp/cmake-install.sh \
    && chmod u+x /tmp/cmake-install.sh \
    && mkdir /opt/cmake-3.24.1 \
    && /tmp/cmake-install.sh --skip-license --prefix=/opt/cmake-3.24.1 \
    && rm /tmp/cmake-install.sh \
    && ln -s /opt/cmake-3.24.1/bin/* /usr/local/bin

# 創建必要目錄
RUN mkdir -p /app/logs && \
    chmod -R 777 /app/logs

# 複製必要文件
COPY requirements.txt .
COPY LangSegment-0.3.5-py3-none-any.whl .
COPY app.py .
COPY . .



# 賦予腳本執行權限
RUN chmod +x start.sh

# 安裝 Python 依賴
RUN pip install --no-cache-dir torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt && \
    # pip install --no-cache-dir ffmpeg && \
    pip install --no-cache-dir -U openmim && \
    mim install mmengine && \
    mim install "mmdet==3.2.0" && \
    mim install "mmpose==1.3.2" && \
    mim install mmcv && \
    pip install --no-cache-dir LangSegment-0.3.5-py3-none-any.whl && \
    pip install numpy==1.24.4

# 安裝 pyopenjtalk
RUN wget "https://files.pythonhosted.org/packages/source/p/pyopenjtalk/pyopenjtalk-0.4.0.tar.gz" && \
    tar -xzf "pyopenjtalk-0.4.0.tar.gz" && \
    rm "pyopenjtalk-0.4.0.tar.gz" && \
    CMAKE_FILE="pyopenjtalk-0.4.0/lib/open_jtalk/src/CMakeLists.txt" && \
    sed -i -E 's/cmake_minimum_required\(VERSION[^\)]*\)/cmake_minimum_required(VERSION 3.5...3.31)/' "$CMAKE_FILE" && \
    tar -czf "pyopenjtalk-0.4.0.tar.gz" "pyopenjtalk-0.4.0" && \
    pip install "pyopenjtalk-0.4.0.tar.gz" && \
    rm -rf "pyopenjtalk-0.4.0.tar.gz" "pyopenjtalk-0.4.0"

# 設置啟動命令
CMD ["./start.sh"]