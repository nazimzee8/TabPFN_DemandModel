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

For GitHub Actions pushes, set these repository variables after creating a new Snowflake
account or image repository:

```text
SNOWFLAKE_REGISTRY_HOST=<account>.registry.snowflakecomputing.com
SNOWFLAKE_IMAGE_REPOSITORY=<account>.registry.snowflakecomputing.com/<db>/<schema>/AUTOGLUON_IMAGE_REPOSITORY
SNOWFLAKE_IMAGE_VERSION=1.0.0
```

Set `SNOWFLAKE_REGISTRY_PASSWORD` as a repository secret containing the Snowflake PAT
`token_secret`. The workflow authenticates Snowflake CLI as `GITHUB_ACTIONS_IMAGE_PUSHER`,
then uses `snow spcs image-registry login` to log Docker in with a short-lived registry
session token.

The workflow can also be run manually with `workflow_dispatch` inputs for the registry host,
image repository, and image version. This avoids committing free-trial account-specific
repository URLs.

### 3. Set the image reference in environment

```bash
export SYNREG_AUTOGLUON_SPCS_IMAGE=<repository_url>/tabpfn-autogluon-ray:1.0.0
export SYNREG_AUTOGLUON_EXECUTION_BACKEND=spcs_job
```

## Notes

- PYTHONPATH is set to `/app/scripts:/app/src` inside the image.
- The image ENTRYPOINT is `python`; SPCS job spec `args` specify the script path.
- **Coordinator topology (default for distributed mode):** one coordinator container per shard
  merges the Ray head and AutoGluon driver. For a 6-shard × 4-worker deployment this is
  **30 SPCS containers** (6 coordinators + 24 workers).
  - `scripts/spcs_ray_coordinator.py` starts `ray start --head --num-cpus=0` as a subprocess,
    polls localhost until reachable, then runs the configured driver script with
    `SYNREG_RAY_ADDRESS_MODE=explicit` and `RAY_HEAD_ADDRESS=localhost:<port>`.
  - The driver script defaults to `autogluon_ray.py`. Set `SPCS_RAY_DRIVER_SCRIPT` env var
    to run a different script (e.g. `ray_capacity_probe.py` for capacity probes).
- **SPCS DNS:** service names have underscores replaced by dashes in DNS. The orchestrator
  auto-resolves the DNS domain via `SYSTEM$GET_SERVICE_DNS_DOMAIN`; set `SPCS_RAY_HEAD_DNS_SUFFIX`
  to override.
- **Default SPCS resource profiles:**
  | Role | CPU request | CPU limit | Memory request | Memory limit |
  |------|-------------|-----------|----------------|--------------|
  | Coordinator | 1 | 2 | 4Gi | 8Gi |
  | Worker | 1 | 1 | 8Gi | 16Gi |
  | Single-node AutoGluon | 4 | 4 | 16Gi | 16Gi |
  | Probe / import-timing | 0.5 | 0.5 | 2Gi | 2Gi |
  Override via `SYNREG_SPCS_RAY_COORDINATOR_*`, `SYNREG_SPCS_RAY_WORKER_*`,
  `SYNREG_SPCS_SINGLE_NODE_*`, or `SYNREG_SPCS_PROBE_*` with `_CPU`, `_MEMORY`,
  `_CPU_REQUEST`, `_CPU_LIMIT`, `_MEMORY_REQUEST`, or `_MEMORY_LIMIT` suffixes.
- **Worker data access (default):** `driver_presigned_url` — the driver builds a time-limited
  HTTPS presigned URL for each dataset and passes it in the compact work-item dict.
  Workers download using `urllib.request` without creating a Snowpark session.
  Set `SYNREG_WORKER_DATA_ACCESS_MODE=scoped_file_url` to use scoped Snowflake URLs instead.
- **No `ray.put()` for datasets.** Workers receive only compact metadata dicts (~8 KB each).
  Datasets are loaded inside the worker task from the presigned or scoped URL.
- **Snowpark session in coordinator:** SPCS job services automatically receive the OAuth token
  at `/snowflake/session/token`, `SNOWFLAKE_ACCOUNT`, and `SNOWFLAKE_HOST`. The coordinator/driver
  creates a Snowpark session from these values using `authenticator='oauth'`. Workers do not
  create Snowpark sessions in the default `driver_presigned_url` mode. No `snowflakeService`
  YAML block is needed or supported in the spec.
- **`MAX_IN_FLIGHT` must not exceed `WORKERS_PER_SHARD`.** If `SYNREG_AUTOGLUON_MAX_IN_FLIGHT`
  is set higher than `SYNREG_AUTOGLUON_WORKERS_PER_SHARD`, `autogluon_ray.py` fails fast.
