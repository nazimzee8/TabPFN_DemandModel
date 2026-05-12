"""
migrate_checkpoint.py — convert a legacy TabPFN checkpoint to checkpoint_format_version=2.

A legacy checkpoint stores cfg as a pickled ModelConfig object, which fails
torch.load(..., weights_only=True) in PyTorch 2.6+. This script converts it to
a plain-dict cfg so the checkpoint loads cleanly with weights_only=True.

Usage:
  # Local path (backs up original as <path>.bak):
  python scripts/migrate_checkpoint.py --path /tmp/best.pt

  # Snowflake stage (GET, migrate, PUT back; backup kept as best.pt.bak on stage):
  python scripts/migrate_checkpoint.py --stage-name MODEL_STAGE --name best.pt
  python scripts/migrate_checkpoint.py --stage-name MODEL_STAGE --name pretrain.pt
"""

import argparse
import dataclasses
import os
import shutil
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import torch  # noqa: E402
from model import ModelConfig  # noqa: E402
from evaluate import load_checkpoint_compat, model_config_to_dict  # noqa: E402


def migrate_checkpoint(src_path, dst_path):
    """
    Load a checkpoint from src_path (using weights_only=False — trusted migration
    context only), convert any ModelConfig cfg to a plain dict, and write a
    checkpoint_format_version=2 file to dst_path.

    Idempotent: a v2 checkpoint with a dict cfg is written back unchanged.

    Verifies the output loads cleanly with weights_only=True before returning.
    """
    # Load using the unsafe path — this is intentional: we are migrating a trusted
    # internally generated file that may contain a pickled ModelConfig.
    # Do not use this script on untrusted checkpoints.
    print(f"Loading checkpoint for migration: {src_path}", flush=True)
    try:
        ckpt = torch.load(src_path, map_location="cpu", weights_only=False)
    except TypeError:
        # Very old PyTorch without weights_only parameter
        ckpt = torch.load(src_path, map_location="cpu")

    if not isinstance(ckpt, dict):
        raise TypeError(
            f"Unsupported checkpoint object type: {type(ckpt)!r}. "
            "Expected a dict."
        )

    # Determine state_dict and cfg from the various legacy formats.
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        cfg_payload = ckpt.get("cfg")
        metadata = ckpt.get("metadata", {})
    else:
        # Bare state dict — no "state_dict" key, just weight tensors.
        state_dict = ckpt
        cfg_payload = None
        metadata = {}

    # Convert cfg to a plain dict.
    if cfg_payload is None:
        # No cfg stored; use a minimal empty dict — load_model will apply defaults.
        cfg_dict = {}
        print("No cfg found in checkpoint; writing empty cfg dict.", flush=True)
    elif isinstance(cfg_payload, dict):
        cfg_dict = dict(cfg_payload)
        print("cfg is already a plain dict (v2 format or compatible).", flush=True)
    elif isinstance(cfg_payload, ModelConfig):
        cfg_dict = dataclasses.asdict(cfg_payload)
        print(
            f"Converted pickled ModelConfig to plain dict: {cfg_dict}", flush=True
        )
    elif dataclasses.is_dataclass(cfg_payload) and not isinstance(cfg_payload, type):
        cfg_dict = dataclasses.asdict(cfg_payload)
        print(
            f"Converted dataclass cfg to plain dict: {cfg_dict}", flush=True
        )
    else:
        cfg_dict = model_config_to_dict(cfg_payload)
        print(
            f"Converted cfg via model_config_to_dict: {cfg_dict}", flush=True
        )

    # Validate: all metadata values must be primitive-safe.
    _validate_metadata(metadata)

    # Build the v2 checkpoint.
    v2_ckpt = {
        "checkpoint_format_version": 2,
        "cfg": cfg_dict,
        "state_dict": state_dict,
        "metadata": metadata,
    }

    print(f"Writing v2 checkpoint to: {dst_path}", flush=True)
    torch.save(v2_ckpt, dst_path)

    # Verify the output loads cleanly with weights_only=True.
    try:
        verification = torch.load(dst_path, map_location="cpu", weights_only=True)
    except TypeError:
        # Very old PyTorch without weights_only — skip strict verification.
        print(
            "[WARNING] PyTorch does not support weights_only=True; "
            "skipping strict verification.",
            flush=True,
        )
        return

    assert verification.get("checkpoint_format_version") == 2, (
        "Verification failed: checkpoint_format_version != 2"
    )
    assert isinstance(verification.get("cfg"), dict), (
        "Verification failed: cfg is not a plain dict after migration"
    )
    print("Verification passed: checkpoint loads cleanly with weights_only=True.", flush=True)


def _validate_metadata(metadata):
    """Raise ValueError if metadata contains non-primitive values."""
    if not isinstance(metadata, dict):
        raise ValueError(f"metadata must be a dict, got {type(metadata)!r}")
    _check_primitives(metadata, path="metadata")


def _check_primitives(obj, path):
    primitive_types = (str, int, float, bool, type(None))
    if isinstance(obj, primitive_types):
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            _check_primitives(v, f"{path}.{k}")
        return
    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _check_primitives(v, f"{path}[{i}]")
        return
    raise ValueError(
        f"Non-primitive value at {path}: {type(obj)!r}. "
        "Checkpoint metadata must contain only str, int, float, bool, None, "
        "or nested lists/dicts of those types."
    )


def migrate_local(path):
    """
    Back up path → path.bak, then migrate in place.
    Prints the verification result.
    """
    backup_path = path + ".bak"
    print(f"Backing up {path} → {backup_path}", flush=True)
    shutil.copy2(path, backup_path)
    print(f"Backup written to {backup_path}", flush=True)

    migrate_checkpoint(path, path)
    print(f"Migration complete: {path}", flush=True)
    print(
        f"To roll back: cp {backup_path!r} {path!r}",
        flush=True,
    )


def migrate_stage(stage_name, checkpoint_name):
    """
    GET the checkpoint from a Snowflake stage, migrate it locally, then PUT it
    back. The original is also PUT as <name>.bak before overwriting.

    stage_name: bare stage name, e.g. 'MODEL_STAGE' (no '@' prefix).
    checkpoint_name: filename under <stage>/checkpoints/, e.g. 'best.pt'.
    """
    try:
        from snowflake.snowpark import Session
        session = Session.builder.getOrCreate()
    except Exception as exc:
        raise RuntimeError(
            f"Could not establish a Snowflake session: {exc}. "
            "Run this script from an environment with Snowflake credentials configured."
        ) from exc

    stage_prefix = f"@{stage_name}/checkpoints"
    stage_src = f"{stage_prefix}/{checkpoint_name}"
    local_tmp = f"/tmp/{checkpoint_name}"
    backup_name = checkpoint_name + ".bak"
    stage_bak = f"{stage_prefix}/{backup_name}"

    # GET the checkpoint to local /tmp.
    print(f"Downloading {stage_src} → {local_tmp}", flush=True)
    session.file.get(stage_src, "/tmp/")
    if not os.path.exists(local_tmp):
        raise FileNotFoundError(
            f"GET succeeded but {local_tmp} not found. "
            "Check the stage path and file name."
        )

    # PUT backup to stage before modifying anything.
    print(f"Uploading backup {local_tmp} → {stage_bak}", flush=True)
    session.file.put(local_tmp, stage_prefix, target_path=backup_name, auto_compress=False, overwrite=True)
    print(f"Backup written to {stage_bak}", flush=True)

    # Migrate in place locally.
    migrated_tmp = local_tmp + ".migrated"
    migrate_checkpoint(local_tmp, migrated_tmp)

    # PUT the migrated file back to the stage under the original name.
    print(f"Uploading migrated checkpoint → {stage_src}", flush=True)
    session.file.put(migrated_tmp, stage_prefix, target_path=checkpoint_name, auto_compress=False, overwrite=True)
    print(f"Migration complete. {stage_src} now contains a v2 checkpoint.", flush=True)
    print(f"Original preserved at {stage_bak}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Convert a legacy TabPFN checkpoint to checkpoint_format_version=2."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--path",
        metavar="PATH",
        help="Local path to the checkpoint file (migrated in place; .bak backup created).",
    )
    group.add_argument(
        "--stage-name",
        metavar="STAGE_NAME",
        help="Snowflake stage name (without '@'), e.g. MODEL_STAGE.",
    )
    parser.add_argument(
        "--name",
        metavar="CHECKPOINT_NAME",
        help="Checkpoint filename under <stage>/checkpoints/ (required with --stage-name).",
    )
    args = parser.parse_args()

    if args.path:
        if not os.path.exists(args.path):
            parser.error(f"File not found: {args.path}")
        migrate_local(args.path)
    else:
        if not args.name:
            parser.error("--name is required when --stage-name is provided.")
        migrate_stage(args.stage_name, args.name)


if __name__ == "__main__":
    main()
