FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.13 \
    python3-pip \
    python3.13-venv \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /wood-chipper-ai

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["gunicorn", "--workers=1", "--threads=1", "--bind=0.0.0.0:8000", "src.server:app"]
