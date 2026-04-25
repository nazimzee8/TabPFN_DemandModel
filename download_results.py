"""
Download model checkpoint and evaluation results from Snowflake stage.

Usage:
    pip install snowflake-snowpark-python   # already in requirements.txt
    set SNOWFLAKE_ACCOUNT=<account-identifier>
    set SNOWFLAKE_USER=<user>
    set SNOWFLAKE_PASSWORD=<password>
    python download_results.py
"""

import os
from pathlib import Path
from snowflake.snowpark import Session

STAGE_CHECKPOINT = "@MODEL_STAGE/checkpoints/"
STAGE_RESULTS    = "@MODEL_STAGE/results/"
LOCAL_DIR        = Path("models")

def main():
    connection_params = {
        "account":   os.environ["SNOWFLAKE_ACCOUNT"],
        "user":      os.environ["SNOWFLAKE_USER"],
        "password":  os.environ["SNOWFLAKE_PASSWORD"],
        "database":  "TABPFN_DB",
        "schema":    "TABPFN_SCHEMA",
        "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
    }

    LOCAL_DIR.mkdir(exist_ok=True)

    with Session.builder.configs(connection_params).create() as session:
        print("Connected to Snowflake.")

        print("Stage contents:")
        for row in session.sql("LIST @MODEL_STAGE").collect():
            print(f"  {row[0]}")
        print()

        print(f"Downloading {STAGE_CHECKPOINT} ...")
        session.file.get(STAGE_CHECKPOINT, str(LOCAL_DIR))

        print(f"Downloading {STAGE_RESULTS} ...")
        session.file.get(STAGE_RESULTS, str(LOCAL_DIR))

    print(f"\nDone. Files saved to ./{LOCAL_DIR}/")
    for f in LOCAL_DIR.iterdir():
        print(f"  {f}")

if __name__ == "__main__":
    main()
