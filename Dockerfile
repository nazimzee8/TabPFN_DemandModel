# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder
RUN pip install --no-cache-dir --upgrade pip

# Heavy, stable: ~2.4 GB torch wheel — cache busts only when torch/numpy version changes
COPY requirements-torch.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements-torch.txt

# Light, mutable: pyarrow, snowpark — changes here do NOT bust the torch layer above
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim
COPY --from=builder /install /usr/local
WORKDIR /app
LABEL model.architecture="deepset-sab-v2" \
      model.pooling="pna|sum|mean|max|learned|attn|multipool" \
      model.attention="set-transformer-sab" \
      model.normalization="per-context-feat-target"
COPY model.py .
COPY train.py .
COPY evaluate.py .
CMD ["bash", "-c", "python train.py && python evaluate.py --model_path best.pt --data_dir /data --results_dir results/"]
