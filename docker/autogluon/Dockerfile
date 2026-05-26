# docker/autogluon/Dockerfile
# Build: docker build --platform linux/amd64 -f docker/autogluon/Dockerfile -t tabpfn-autogluon-ray:1.0.0 .
FROM python:3.11-slim AS deps

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ make libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

WORKDIR /app

RUN python -m venv "${VIRTUAL_ENV}"
COPY docker/autogluon/requirements.txt /app/requirements.txt
# Key packages: ray==2.10.0, autogluon.tabular==1.3.0, snowflake-snowpark-python>=1.10, lightgbm, catboost, xgboost
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt
# Smoke check: verify critical imports succeed before accepting the image layer
RUN python -c "import ray, autogluon.tabular, snowflake.snowpark, pandas, numpy, pyarrow; print('smoke check ok')"

FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

ENV PYTHONUNBUFFERED=1
ENV HOME=/tmp
ENV PYTHONPATH=/app/scripts:/app/src

WORKDIR /app

COPY --from=deps /opt/venv /opt/venv
RUN mkdir -p /app/scripts /app/src

COPY scripts/autogluon_import_timing_probe.py /app/scripts/autogluon_import_timing_probe.py
COPY scripts/autogluon_ray.py /app/scripts/autogluon_ray.py
COPY scripts/autogluon_worker_access_probe.py /app/scripts/autogluon_worker_access_probe.py
COPY scripts/ray_capacity_probe.py /app/scripts/ray_capacity_probe.py
COPY scripts/spcs_ray_head.py /app/scripts/spcs_ray_head.py
COPY scripts/spcs_ray_worker.py /app/scripts/spcs_ray_worker.py
COPY scripts/spcs_snowpark_session_probe.py /app/scripts/spcs_snowpark_session_probe.py

COPY src/autogluon_models.py /app/src/autogluon_models.py
COPY src/baseline_models.py /app/src/baseline_models.py
COPY src/deepset_inference.py /app/src/deepset_inference.py
COPY src/evaluate_synthetic_regression.py /app/src/evaluate_synthetic_regression.py
COPY src/model.py /app/src/model.py
COPY src/sanity_checks.py /app/src/sanity_checks.py

ENTRYPOINT ["python"]
