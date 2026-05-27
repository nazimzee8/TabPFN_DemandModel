"""spcs_snowpark_session_probe.py

Validates that a Snowpark session can be created inside an SPCS container and that
the session context (database, schema, role) is correct.

SPCS exposes a Snowflake OAuth token at /snowflake/session/token plus
SNOWFLAKE_ACCOUNT and SNOWFLAKE_HOST environment variables. The probe builds an
explicit OAuth Snowpark session from those injected values.

Usage:
  python /app/scripts/spcs_snowpark_session_probe.py

Environment variables (optional):
  SYNREG_SPCS_SESSION_PROBE_QUERY  SQL query to run after connecting (default: SELECT CURRENT_DATABASE(), CURRENT_SCHEMA(), CURRENT_ROLE())
"""

from __future__ import annotations

import json
import os
import sys
import time


def _log(event: str, **fields) -> None:
    print(json.dumps({"event": event, **fields}, default=str), flush=True)


_OAUTH_TOKEN_PATH = "/snowflake/session/token"
_t0 = time.perf_counter()

_log("spcs_session_probe_starting", token_path=_OAUTH_TOKEN_PATH)

# Check for token file
if not os.path.exists(_OAUTH_TOKEN_PATH):
    _log(
        "spcs_session_probe_token_missing",
        token_path=_OAUTH_TOKEN_PATH,
        remediation=(
            "The Snowflake OAuth token is not present at the expected path. "
            "SPCS job services receive this token automatically — verify the container "
            "is running as a Snowflake job service (not a standalone container) and that "
            "the compute pool and service are configured correctly. "
            "See Snowflake SPCS docs: Using a Snowflake connector in a Snowpark Container."
        ),
    )
    sys.exit(1)

_log("spcs_session_probe_token_found", token_path=_OAUTH_TOKEN_PATH)

try:
    from snowflake.snowpark import Session
except ImportError as exc:
    _log("spcs_session_probe_import_failed", error=str(exc))
    sys.exit(1)

_log("spcs_session_probe_creating_session")
try:
    with open(_OAUTH_TOKEN_PATH, "r") as f:
        token = f.read().strip()
    configs = {
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "host": os.getenv("SNOWFLAKE_HOST"),
        "authenticator": "oauth",
        "token": token,
    }
    for config_key, env_key in {
        "database": "SNOWFLAKE_DATABASE",
        "schema": "SNOWFLAKE_SCHEMA",
        "role": "SNOWFLAKE_ROLE",
        "warehouse": "SNOWFLAKE_WAREHOUSE",
    }.items():
        value = os.getenv(env_key)
        if value:
            configs[config_key] = value
    missing = [k for k in ("account", "host", "token") if not configs.get(k)]
    if missing:
        raise RuntimeError(
            "Missing required SPCS Snowflake connection values: "
            + ", ".join(missing)
        )
    session = Session.builder.configs(configs).create()
    _log("spcs_session_probe_session_created", elapsed_s=round(time.perf_counter() - _t0, 3))
except Exception as exc:
    _log(
        "spcs_session_probe_session_failed",
        error_type=type(exc).__name__,
        error=str(exc),
        elapsed_s=round(time.perf_counter() - _t0, 3),
        remediation=(
            "SPCS OAuth session creation failed. Ensure the job service is running inside "
            "Snowpark Container Services and the service role has the necessary privileges."
        ),
    )
    sys.exit(1)

query = os.getenv(
    "SYNREG_SPCS_SESSION_PROBE_QUERY",
    "SELECT CURRENT_DATABASE(), CURRENT_SCHEMA(), CURRENT_ROLE()",
)
_log("spcs_session_probe_running_query", query=query)
try:
    rows = session.sql(query).collect()
    result = [dict(zip(row._fields, row)) if hasattr(row, "_fields") else {"row": str(row)} for row in rows]
    _log(
        "spcs_session_probe_query_ok",
        result=result,
        elapsed_s=round(time.perf_counter() - _t0, 3),
    )
except Exception as exc:
    _log(
        "spcs_session_probe_query_failed",
        query=query,
        error_type=type(exc).__name__,
        error=str(exc),
    )
    sys.exit(1)

_log("spcs_session_probe_complete", elapsed_s=round(time.perf_counter() - _t0, 3))
