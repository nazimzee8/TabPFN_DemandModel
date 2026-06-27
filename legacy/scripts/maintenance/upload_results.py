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
LOCAL_RESULTS_DIR = Path("results")
STAGE_CHECKPOINT = "@MODEL_STAGE/checkpoints/"
STAGE_RESULTS    = "@EVALUATION_RESULTS_STAGE/"


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

        for csv_path in LOCAL_RESULTS_DIR.rglob("*.csv"):
            rel_parent = csv_path.parent.relative_to(LOCAL_RESULTS_DIR)
            target_stage = STAGE_RESULTS
            if str(rel_parent) != ".":
                target_stage = f"{STAGE_RESULTS}{str(rel_parent).replace(os.sep, '/')}/"
            print(f"Uploading {csv_path} -> {target_stage} ...")
            session.file.put(str(csv_path), target_stage, overwrite=True)

    print("\nVerifying stage contents:")
    with Session.builder.configs(connection_params).create() as session:
        for stage in ("@MODEL_STAGE/checkpoints/", "@EVALUATION_RESULTS_STAGE/"):
            print(f"{stage}:")
            for row in session.sql(f"LIST {stage}").collect():
                print(f"  {row[0]}")


if __name__ == "__main__":
    main()
