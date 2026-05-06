FROM docker.io/nvidia/cuda:12.8.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies only (source is bind-mounted)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

CMD ["python3", "src/ui/openwebui/server.py"]
