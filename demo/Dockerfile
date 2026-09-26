# Live-demo backend (CPU). Build from the repository root:
#   docker build -f demo/Dockerfile -t trumen-demo .
#   docker run -p 7860:7860 trumen-demo
# Works as a Hugging Face Docker Space, on Railway, Render or any container host ($PORT honoured).
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY demo/requirements.txt demo/requirements.txt
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r demo/requirements.txt
COPY src src
COPY configs configs
COPY weights/yolo11n.pt weights/yolo11n.pt
COPY demo demo
ENV TRAFFIC_FORCE_CPU=1 YOLO_OFFLINE=1 PORT=7860
RUN useradd -m app && chown -R app /app
USER app
EXPOSE 7860
CMD ["sh", "-c", "uvicorn demo.app:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
