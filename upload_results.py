"""
Upload pre-trained model checkpoint and evaluation results to Snowflake stage.

Usage:
    set SNOWFLAKE_ACCOUNT=<account-identifier>
    set SNOWFLAKE_USER=<user>
    set SNOWFLAKE_PASSWORD=<password>
    python upload_results.py
"""

import os
from pathlib import Path
from snowflake.snowpark import Session

LOCAL_CHECKPOINT = Path("best.pt")
LOCAL_RESULTS    = Path("results/test_report.csv")
STAGE_CHECKPOINT = "@MODEL_STAGE/checkpoints/"
STAGE_RESULTS    = "@MODEL_STAGE/results/"


def main():
    connection_params = {
        "account":   os.environ["SNOWFLAKE_ACCOUNT"],
        "user":      os.environ["SNOWFLAKE_USER"],
        "password":  os.environ["SNOWFLAKE_PASSWORD"],
        "database":  "TABPFN_DB",
        "schema":    "TABPFN_SCHEMA",
        "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
    }

    with Session.builder.configs(connection_params).create() as session:
        print("Connected to Snowflake.")

        print(f"Uploading {LOCAL_CHECKPOINT} -> {STAGE_CHECKPOINT} ...")
        session.file.put(str(LOCAL_CHECKPOINT), STAGE_CHECKPOINT, overwrite=True)

        print(f"Uploading {LOCAL_RESULTS} -> {STAGE_RESULTS} ...")
        session.file.put(str(LOCAL_RESULTS), STAGE_RESULTS, overwrite=True)

    print("\nVerifying stage contents:")
    with Session.builder.configs(connection_params).create() as session:
        for row in session.sql("LIST @MODEL_STAGE").collect():
            print(f"  {row[0]}")


if __name__ == "__main__":
    main()
