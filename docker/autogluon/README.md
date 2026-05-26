# AutoGluon SPCS Custom Image

## Overview

This Docker image provides a self-contained environment for running AutoGluon and Ray distributed
evaluation inside Snowpark Container Services (SPCS). It eliminates per-job pip install overhead
by preinstalling AutoGluon, Ray, and all required dependencies at image build time.

The image is used with the `SYNREG_AUTOGLUON_EXECUTION_BACKEND=spcs_job` backend. Instead of
Snowflake MLJob runtime environments with `pip_requirements`, the orchestrator submits SPCS job
services that use this image directly. No `runtime_environment` or `pip_requirements` fields are
required or used for SPCS procedures.

PYTHONPATH includes `/app/scripts` and `/app/src`, so all evaluation helpers in `scripts/` and
`src/` are importable without additional path manipulation.

## Build

```bash
docker build --platform linux/amd64 -f docker/autogluon/Dockerfile -t tabpfn-autogluon-ray:1.0.0 .
```

Run from the repository root (the build context must include `scripts/` and `src/`).

## Health Check

```bash
python -c "import ray, autogluon.tabular; print('ok')"
```

## Push to Snowflake Image Repository

### 1. Create the image repository (run once in Snowflake)

```sql
-- See sql/create_autogluon_spcs_image_repository.sql
CREATE IMAGE REPOSITORY IF NOT EXISTS AUTOGLUON_IMAGE_REPOSITORY;
SHOW IMAGE REPOSITORIES;  -- note the repository_url column
```

### 2. Authenticate and push

```bash
# Log in to the Snowflake image registry
docker login <account>.registry.snowflakecomputing.com

# Tag the local image with the full repository URL
docker tag tabpfn-autogluon-ray:1.0.0 \
  <repository_url>/tabpfn-autogluon-ray:1.0.0

# Push to the Snowflake image repository
docker push <repository_url>/tabpfn-autogluon-ray:1.0.0
```

Replace `<account>` with your Snowflake account identifier and `<repository_url>` with the URL
shown in `SHOW IMAGE REPOSITORIES` (typically
`<account>.registry.snowflakecomputing.com/<db>/<schema>/AUTOGLUON_IMAGE_REPOSITORY`).

### 3. Set the image reference in environment

```bash
export SYNREG_AUTOGLUON_SPCS_IMAGE=<repository_url>/tabpfn-autogluon-ray:1.0.0
export SYNREG_AUTOGLUON_EXECUTION_BACKEND=spcs_job
```

## Notes

- PYTHONPATH is set to `/app/scripts:/app/src` inside the image.
- The image ENTRYPOINT is `python`; SPCS job service `args` specify the script path.
- Self-managed Ray: the orchestrator starts a Ray head container and worker containers;
  the driver container connects with `SYNREG_RAY_ADDRESS_MODE=explicit` and `RAY_HEAD_ADDRESS`.
- For SPCS session access (Snowpark), containers use the Snowflake-injected OAuth token
  at `/snowflake/session/token` with `SNOWFLAKE_ACCOUNT` and `SNOWFLAKE_HOST`.
