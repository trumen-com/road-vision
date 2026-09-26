# Alternative to requirements.txt, same two commands inside the container:
#   docker build -t truminds .
#   docker run --gpus all --network none -v /data/test:/data/test truminds \
#       python run_submission.py --videos /data/test --out /data/test/predictions.json
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV YOLO_OFFLINE=1 PYTHONUNBUFFERED=1
# fail the build, not the evaluation, if weights are missing
RUN test -f weights/yolo11m.pt && test -f weights/yolo11n.pt
CMD ["python", "run_submission.py", "--videos", "/data/test", "--out", "/data/test/predictions.json"]
