FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Install base dependencies
RUN apt update && apt install -y \
    git build-essential zlib1g-dev libncurses5-dev \
    libgdbm-dev libnss3-dev libssl-dev libreadline-dev \
    libffi-dev wget curl libbz2-dev liblzma-dev \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.13 from source
RUN cd /opt && \
    wget https://www.python.org/ftp/python/3.13.0/Python-3.13.0.tgz && \
    tar -xf Python-3.13.0.tgz && \
    cd Python-3.13.0 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    ln -s /usr/local/bin/python3.13 /usr/local/bin/python && \
    ln -s /usr/local/bin/pip3.13 /usr/local/bin/pip

# Show versions just for sanity check
RUN python --version && pip --version

WORKDIR /wood-chipper-ai

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["gunicorn", "--workers=1", "--threads=1", "--bind=0.0.0.0:8000", "src.server:app"]
