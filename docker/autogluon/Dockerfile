# docker/autogluon/Dockerfile
# Build: docker build --platform linux/amd64 -f docker/autogluon/Dockerfile -t tabpfn-autogluon-ray:1.0.0 .
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ make libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY docker/autogluon/requirements.txt /app/requirements.txt
# Key packages: ray==2.10.0, autogluon.tabular==1.3.0, snowflake-snowpark-python>=1.10, lightgbm, catboost, xgboost
RUN pip install --no-cache-dir -r /app/requirements.txt
# Smoke check: verify critical imports succeed before accepting the image layer
RUN python -c "import ray, autogluon.tabular, snowflake.snowpark, pandas, numpy, pyarrow; print('smoke check ok')"

COPY scripts/ /app/scripts/
COPY src/ /app/src/

ENV PYTHONUNBUFFERED=1
ENV HOME=/tmp
ENV PYTHONPATH=/app/scripts:/app/src

ENTRYPOINT ["python"]
