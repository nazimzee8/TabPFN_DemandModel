# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder
RUN pip install --no-cache-dir --upgrade pip
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim
COPY --from=builder /install /usr/local
WORKDIR /app
COPY model.py .
COPY train.py .
COPY evaluate.py .
CMD ["bash", "-c", "python train.py && python evaluate.py --model_path best.pt --data_dir /data --results_dir results/"]
