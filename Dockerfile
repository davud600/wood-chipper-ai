FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends redis-server nginx

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x ./start.sh

RUN chmod +x ./config_nginx.sh

RUN ./config_nginx.sh

EXPOSE 8000

CMD ["sh", "-c", "./start.sh"]
