"""preload.py — Teardown and Reinstall Script

Generates preload.sql: a self-contained SnowSQL script that tears down
idle-cost objects (compute pools, stages, warehouse) and reinstalls the
TABPFN_DB environment from scratch.

Usage:
    python preload.py [--output PATH] [--warehouse NAME] [--dry-run]

Then run the generated file:
    snowsql -c <connection> -f preload.sql

NOTE: Stored procedures are not created by this script. Create/manage them
manually inside Snowflake after running this script.
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR    = Path(__file__).parent.resolve()   # C:/Documents/TabPFN_DemandModel
DATABASE    = "TABPFN_DB"
SCHEMA      = "TABPFN_SCHEMA"
SRC_DIR     = BASE_DIR / "src"
SCRIPTS_DIR = BASE_DIR / "scripts"
DATA_TRAIN  = BASE_DIR / "data" / "train"
DATA_VAL    = BASE_DIR / "data" / "val"
DATA_TEST   = BASE_DIR / "data" / "test"

# Forward-slash paths for SnowSQL PUT commands (correct on Windows too)
def _put_path(p: Path) -> str:
    return "file://" + p.as_posix()


# ---------------------------------------------------------------------------
# SQL section builders
# ---------------------------------------------------------------------------

def _section_header(title: str) -> str:
    bar = "-" * 66
    return f"\n-- {bar}\n-- {title}\n-- {bar}\n"


def _section0_meta() -> str:
    return (
        "!set exit_on_error=true\n"
        "!set variable_substitution=false\n"
    )


def _section1_context(warehouse: str) -> str:
    return (
        _section_header("Section 1 — Context") +
        f"USE DATABASE {DATABASE};\n"
        f"USE SCHEMA {SCHEMA};\n"
    )


def _section_teardown(warehouse: str) -> str:
    return (
        _section_header("Section: Teardown — drop idle-cost objects and recreate warehouse") +
        "-- Compute pools\n"
        "DROP COMPUTE POOL IF EXISTS DEEPSET_GPU_POOL;\n"
        "DROP COMPUTE POOL IF EXISTS DEEPSET_CPU_POOL;\n"
        "DROP COMPUTE POOL IF EXISTS AUTOGLUON_CPU_POOL;\n"
        "-- Stages\n"
        "DROP STAGE IF EXISTS META_REGRESSION_DATASET_STAGE;\n"
        "DROP STAGE IF EXISTS MODEL_STAGE;\n"
        "DROP STAGE IF EXISTS EVALUATION_RESULTS_STAGE;\n"
        "DROP STAGE IF EXISTS MLJOB_PAYLOAD_STAGE;\n"
        "DROP STAGE IF EXISTS EPOCH_STAGE;\n"
        f"-- Warehouse: drop then recreate fresh (auto-suspend=60s, initially suspended)\n"
        f"DROP WAREHOUSE IF EXISTS {warehouse};\n"
        f"CREATE WAREHOUSE IF NOT EXISTS {warehouse}\n"
        f"  WAREHOUSE_SIZE = X-SMALL\n"
        f"  AUTO_SUSPEND = 60\n"
        f"  AUTO_RESUME = TRUE\n"
        f"  INITIALLY_SUSPENDED = TRUE;\n"
        f"USE WAREHOUSE {warehouse};\n"
    )


def _section2_stages() -> str:
    stages = [
        "META_REGRESSION_DATASET_STAGE",
        "MODEL_STAGE",
        "EVALUATION_RESULTS_STAGE",
        "MLJOB_PAYLOAD_STAGE",
        "EPOCH_STAGE",
    ]
    lines = [_section_header("Section 2 — Stages (idempotent)")]
    for s in stages:
        lines.append(
            f"CREATE STAGE IF NOT EXISTS {s}"
            f"  ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');\n"
        )
    return "".join(lines)


def _section3_index_table() -> str:
    return (
        _section_header("Section 3 — META_REGRESSION_DATASET_INDEX table") +
        "DROP TABLE IF EXISTS META_REGRESSION_DATASET_INDEX;\n"
        "CREATE TRANSIENT TABLE META_REGRESSION_DATASET_INDEX (\n"
        "  split        STRING NOT NULL,\n"
        "  task_id      STRING NOT NULL,\n"
        "  stage_path   STRING NOT NULL,\n"
        "  n            NUMBER,\n"
        "  p            NUMBER,\n"
        "  n_train      NUMBER,\n"
        "  n_test       NUMBER,\n"
        "  prior_regime STRING,\n"
        "  hpo_bucket   NUMBER\n"
        ")\n"
        "DATA_RETENTION_TIME_IN_DAYS = 0\n"
        "CLUSTER BY (split, hpo_bucket, prior_regime, p, n_train);\n"
    )


def _section4_clean_scripts() -> str:
    return (
        _section_header("Section 4 — Clean @MODEL_STAGE/scripts/") +
        "-- No-op after a fresh DROP STAGE; kept as a safety net.\n"
        "REMOVE @MODEL_STAGE/scripts/ PATTERN='.*[.]py';\n"
    )


def _section5_upload_scripts() -> str:
    src_put     = _put_path(SRC_DIR / "*.py")
    scripts_put = _put_path(SCRIPTS_DIR / "*.py")
    return (
        _section_header("Section 5 — Upload Python scripts") +
        f"PUT {src_put}\n"
        f"    @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;\n"
        f"PUT {scripts_put}\n"
        f"    @MODEL_STAGE/scripts/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;\n"
    )


def _section6_upload_data() -> str:
    train_put = _put_path(DATA_TRAIN / "*.parquet")
    val_put   = _put_path(DATA_VAL   / "*.parquet")
    test_put  = _put_path(DATA_TEST  / "*.parquet")
    return (
        _section_header("Section 6 — Upload synthetic datasets") +
        "REMOVE @META_REGRESSION_DATASET_STAGE/train/;\n"
        "REMOVE @META_REGRESSION_DATASET_STAGE/val/;\n"
        "REMOVE @META_REGRESSION_DATASET_STAGE/test/;\n"
        f"PUT {train_put}\n"
        f"    @META_REGRESSION_DATASET_STAGE/train/ AUTO_COMPRESS=FALSE OVERWRITE=TRUE;\n"
        f"PUT {val_put}\n"
        f"    @META_REGRESSION_DATASET_STAGE/val/   AUTO_COMPRESS=FALSE OVERWRITE=TRUE;\n"
        f"PUT {test_put}\n"
        f"    @META_REGRESSION_DATASET_STAGE/test/  AUTO_COMPRESS=FALSE OVERWRITE=TRUE;\n"
    )


def _section10_verify() -> str:
    return (
        _section_header("Section 10 — Verification") +
        "LIST @MODEL_STAGE/scripts/ PATTERN='.*[.]py';\n"
        "LIST @META_REGRESSION_DATASET_STAGE/train/ PATTERN='.*[.]parquet';\n"
        "LIST @META_REGRESSION_DATASET_STAGE/val/   PATTERN='.*[.]parquet';\n"
        "LIST @META_REGRESSION_DATASET_STAGE/test/  PATTERN='.*[.]parquet';\n"
        "-- NOTE: The table below will return 0 rows until you run\n"
        "--   CALL build_meta_dataset_index();\n"
        "-- manually in Snowflake after this script completes.\n"
        "SELECT split, COUNT(*) AS task_count\n"
        "FROM META_REGRESSION_DATASET_INDEX\n"
        "GROUP BY split\n"
        "ORDER BY split;\n"
        "-- Expected after build_meta_dataset_index(): train=800, val=100, test=100\n"
    )


# ---------------------------------------------------------------------------
# Top-level generator
# ---------------------------------------------------------------------------

def generate_sql(warehouse: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    header = (
        f"-- preload.sql — generated by preload.py at {ts}\n"
        f"-- Tears down idle-cost objects and reinstalls the TABPFN_DB environment.\n"
        f"--\n"
        f"-- Drops compute pools, stages, and the virtual warehouse, then recreates\n"
        f"-- all of them fresh. Stored procedures are NOT created by this script;\n"
        f"-- create/manage them manually inside Snowflake.\n"
        f"--\n"
        f"-- Run with: snowsql -c <connection> -f preload.sql\n"
        f"--\n\n"
    )

    parts = [
        header,
        _section0_meta(),
        _section1_context(warehouse),
        _section_teardown(warehouse),
        _section2_stages(),
        _section3_index_table(),
        _section4_clean_scripts(),
        _section5_upload_scripts(),
        _section6_upload_data(),
        _section10_verify(),
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# File count summary
# ---------------------------------------------------------------------------

def _count(directory: Path, pattern: str) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for _ in directory.glob(pattern))


def print_summary() -> None:
    src_count     = _count(SRC_DIR,     "*.py")
    scripts_count = _count(SCRIPTS_DIR, "*.py")
    train_count   = _count(DATA_TRAIN,  "*.parquet")
    val_count     = _count(DATA_VAL,    "*.parquet")
    test_count    = _count(DATA_TEST,   "*.parquet")

    print("Upload counts:")
    print(f"  src/*.py        : {src_count}")
    print(f"  scripts/*.py    : {scripts_count}")
    print(f"  data/train/     : {train_count} parquet files")
    print(f"  data/val/       : {val_count} parquet files")
    print(f"  data/test/      : {test_count} parquet files")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate preload.sql to tear down and reinstall the TABPFN_DB environment.",
    )
    parser.add_argument(
        "--output",
        default="preload.sql",
        help="Path for generated SQL file (default: preload.sql next to this script)",
    )
    parser.add_argument(
        "--warehouse",
        default="COMPUTE_WH",
        help="Snowflake virtual warehouse name (default: COMPUTE_WH)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print generated SQL to stdout instead of writing a file",
    )
    args = parser.parse_args()

    sql = generate_sql(args.warehouse)

    if args.dry_run:
        print(sql)
    else:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = BASE_DIR / output_path
        output_path.write_text(sql, encoding="utf-8")
        print(f"Generated: {output_path}")
        print()
        print_summary()
        print()
        print("Next steps:")
        print(f"  snowsql -c <connection> -f {output_path}")
        print("  Then create stored procedures manually in Snowflake.")
        print("  Then run: CALL build_meta_dataset_index();")


if __name__ == "__main__":
    main()
