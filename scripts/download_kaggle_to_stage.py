"""
Download Kaggle benchmark data inside Snowflake and upload .npz files to a stage.

This script is intended to run as a Snowflake MLJob submitted from
run_training_job.run_kaggle_download(). Kaggle credentials are read from a
Snowflake PASSWORD secret and exposed only to the in-process Kaggle client through
KAGGLE_USERNAME / KAGGLE_KEY.
"""
import os
import tempfile

from kaggle.api.kaggle_api_extended import KaggleApi
from snowflake.snowpark.context import get_active_session

from download_kaggle_benchmark import COMPETITIONS, MAX_SAMPLES, convert_competition


DEFAULT_STAGE = "@META_REGRESSION_DATASET_STAGE/kaggle/"


def _parse_slugs(raw_slugs):
    if not raw_slugs:
        return None
    return {slug.strip() for slug in raw_slugs.split(",") if slug.strip()}


def _require_kaggle_env():
    missing = [
        name for name in ("KAGGLE_USERNAME", "KAGGLE_KEY")
        if not os.environ.get(name)
    ]
    if missing:
        raise RuntimeError(
            "Missing Kaggle credential environment variable(s): "
            + ", ".join(missing)
            + ". Pass them to the MLJob through spec_overrides container secrets."
        )
    os.environ.setdefault("KAGGLE_CONFIG_DIR", "/tmp")
    print("Loaded Kaggle credentials from MLJob secret environment.")


def main():
    stage_path = os.environ.get("KAGGLE_STAGE", DEFAULT_STAGE)
    max_samples = int(os.environ.get("KAGGLE_MAX_SAMPLES", str(MAX_SAMPLES)))
    selected_slugs = _parse_slugs(os.environ.get("KAGGLE_SLUGS"))

    competitions = COMPETITIONS
    if selected_slugs:
        competitions = [c for c in COMPETITIONS if c["slug"] in selected_slugs]
        missing = selected_slugs.difference({c["slug"] for c in competitions})
        if missing:
            raise ValueError(f"Unknown Kaggle slug(s): {', '.join(sorted(missing))}")
    if not competitions:
        raise ValueError("No Kaggle competitions selected.")

    _require_kaggle_env()

    api = KaggleApi()
    api.authenticate()
    session = get_active_session()

    with tempfile.TemporaryDirectory(dir="/tmp") as out_dir:
        uploaded = []
        for comp in competitions:
            out_path = convert_competition(
                api,
                comp,
                out_dir,
                max_samples=max_samples,
            )
            session.file.put(out_path, stage_path, overwrite=True, auto_compress=False)
            uploaded.append(os.path.basename(out_path))
            print(f"Uploaded {out_path} to {stage_path}")

    print("Kaggle stage upload complete:")
    for fname in uploaded:
        print(f"  {fname}")


if __name__ == "__main__":
    main()
