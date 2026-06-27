"""
model_ddp_memory_probe.py

Deployment-safety probe: measures peak CUDA memory per DDP worker for representative
MODEL3 shapes before pretrain / HPO / final training.

Runs via PyTorchDistributor on DEEPSET_GPU_POOL using the same topology as training
(TRAIN_NUM_NODES nodes x 4 workers/GPUs per node = world_size 40 by default).

Because MODEL3 meta-training uses back-propagation through the full forward graph, the
probe performs a forward-and-backward pass by default (MODEL_PROBE_RUN_BACKWARD=true).
This gives a faithful peak-memory measurement that covers gradient tensors and activation
storage retained during backprop.

Launched via the run_model_ddp_memory_probe() stored procedure handler in
run_model_training_job.py.

Usage (via stored procedure):
    CALL run_model_ddp_memory_probe(
        'inductive_forecasting', 'market_exchangeable_icl',
        200,   -- N_CONTEXT
        128,   -- P_FEATURES
        128,   -- M_QUERY
        128,   -- D_PHI
        1,     -- N_BLOCKS
        TRUE   -- RUN_BACKWARD (always TRUE for training-regime validation)
    );

See results:
    LIST @MODEL_STAGE/diagnostics/;
    SELECT $1 FROM @MODEL_STAGE/diagnostics/model_ddp_memory_probe.json
      (FILE_FORMAT => (TYPE = JSON));
"""

import json
import math
import os
import socket
import tempfile
import time
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

try:
    from snowflake.ml.modeling.distributors.pytorch import (
        PyTorchDistributor,
        PyTorchScalingConfig,
        WorkerResourceConfig,
        get_context,
    )
except ImportError:
    PyTorchDistributor = None
    PyTorchScalingConfig = None
    WorkerResourceConfig = None
    get_context = None


# ---------------------------------------------------------------------------
# Module-level env var constants (read at import time by main(); overridden
# per-worker by reading os.environ directly inside probe_fn()).
# ---------------------------------------------------------------------------
MODEL_DESIGN_PATTERN = os.environ.get("MODEL_DESIGN_PATTERN", "inductive_forecasting")
MODEL_FAMILY  = os.environ.get("MODEL_FAMILY",  "market_exchangeable_icl")

MODEL_PROBE_N_CONTEXT   = int(os.environ.get("MODEL_PROBE_N_CONTEXT",  "200"))
MODEL_PROBE_P_FEATURES  = int(os.environ.get("MODEL_PROBE_P_FEATURES", "128"))
MODEL_PROBE_M_QUERY     = int(os.environ.get("MODEL_PROBE_M_QUERY",    "128"))
MODEL_PROBE_D_PHI       = int(os.environ.get("MODEL_PROBE_D_PHI",      "128"))
MODEL_PROBE_N_BLOCKS    = int(os.environ.get("MODEL_PROBE_N_BLOCKS",   "1"))
MODEL_PROBE_RUN_BACKWARD = (
    os.environ.get("MODEL_PROBE_RUN_BACKWARD", "true").lower() == "true"
)
MODEL_PROBE_DTYPE = os.environ.get("MODEL_PROBE_DTYPE", "float32")
MODEL_PROBE_MAX_GPU_MEMORY_FRACTION = float(
    os.environ.get("MODEL_PROBE_MAX_GPU_MEMORY_FRACTION", "0.9")
)
MODEL_PROBE_OUTPUT_STAGE = os.environ.get(
    "MODEL_PROBE_OUTPUT_STAGE", "@MODEL_STAGE/diagnostics/"
)
MODEL_PROBE_STRICT_MEMORY_GUARD = (
    os.environ.get("MODEL_PROBE_STRICT_MEMORY_GUARD", "true").lower() == "true"
)
MODEL_PROBE_MEMORY_SAFETY_FACTOR = float(
    os.environ.get("MODEL_PROBE_MEMORY_SAFETY_FACTOR", "1.5")
)

TRAIN_NUM_NODES           = int(os.environ.get("TRAIN_NUM_NODES", "10"))
EXPECTED_TRAIN_WORLD_SIZE = int(os.environ.get("EXPECTED_TRAIN_WORLD_SIZE", "40"))
MODEL_STAGE               = "@MODEL_STAGE"


# ---------------------------------------------------------------------------
# Supported configurations
# ---------------------------------------------------------------------------
_SUPPORTED_DESIGN_PATTERNS = frozenset({"inductive_forecasting"})
_SUPPORTED_FAMILIES            = frozenset({"market_exchangeable_icl"})
_SUPPORTED_DTYPES              = frozenset({"float32", "bfloat16"})
_DTYPE_BYTES                   = {"float32": 4, "bfloat16": 2}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_probe_config(
    model_design_pattern: str,
    model_family: str,
    n_context: int,
    p_features: int,
    m_query: int,
    d_phi: int,
    n_blocks: int,
    dtype: str,
) -> None:
    """Validate probe configuration. Raises ValueError for unsupported combinations.

    Called before the PyTorchDistributor is launched so bad configs fail fast
    without consuming GPU pool resources.
    """
    if model_design_pattern == "transductive_completion":
        raise ValueError(
            "model_ddp_memory_probe does not yet support "
            "model_design_pattern='transductive_completion'. "
            "A transductive-completion-specific probe will be added in a future release. "
            "Use model_design_pattern='inductive_forecasting' to probe ICL memory."
        )
    if model_design_pattern not in _SUPPORTED_DESIGN_PATTERNS:
        raise ValueError(
            f"Unsupported model_design_pattern={model_design_pattern!r}. "
            f"Supported: {sorted(_SUPPORTED_DESIGN_PATTERNS)}"
        )
    if model_family not in _SUPPORTED_FAMILIES:
        raise ValueError(
            f"Unsupported model_family={model_family!r}. "
            f"Supported: {sorted(_SUPPORTED_FAMILIES)}"
        )
    for name, val in [
        ("n_context",  n_context),
        ("p_features", p_features),
        ("m_query",    m_query),
        ("d_phi",      d_phi),
        ("n_blocks",   n_blocks),
    ]:
        if val <= 0:
            raise ValueError(
                f"Shape parameter {name}={val!r} is invalid; must be a positive integer."
            )
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"Unsupported dtype={dtype!r}. Supported: {sorted(_SUPPORTED_DTYPES)}"
        )


# ---------------------------------------------------------------------------
# Static memory estimator
# ---------------------------------------------------------------------------

def estimate_h_tensor_bytes(
    n_context: int,
    p_features: int,
    m_query: int,
    d_phi: int,
    dtype: str = "float32",
) -> int:
    """Estimate bytes for the MODEL3 ICL H tensor: shape (m_query, n_context, p_features, d_phi).

    The H tensor is the dominant memory allocation in DeepSetICLModel.
    Each element is stored as one value of the given dtype.
    """
    return m_query * n_context * p_features * d_phi * _DTYPE_BYTES[dtype]


def estimate_reserved_bytes(
    h_tensor_bytes: int,
    run_backward: bool,
    safety_factor: float = 1.5,
) -> tuple:
    """Estimate total reserved GPU memory.

    Conservative activation multipliers:
      - activation_factor = 8   for forward-only
      - activation_factor = 20  for forward+backward

    The higher backward factor accounts for:
      - Gradient tensors (same size as model parameters and activations)
      - Activation storage retained for backprop through ExchangeableMatrixBlocks
      - DDP gradient buffers
      - Framework and PyTorch caching allocator overhead

    Args:
        h_tensor_bytes: Bytes for the H tensor (primary cost).
        run_backward:   Whether a backward pass is included.
        safety_factor:  Multiplier for framework + caching overhead.

    Returns:
        (estimated_reserved_bytes, activation_factor)
    """
    activation_factor = 20 if run_backward else 8
    estimated = int(math.ceil(h_tensor_bytes * activation_factor * safety_factor))
    return estimated, activation_factor


# ---------------------------------------------------------------------------
# JSON upload helper (rank 0 only)
# ---------------------------------------------------------------------------

def _upload_probe_result(
    model_design_pattern: str,
    model_family: str,
    n_context: int,
    p_features: int,
    m_query: int,
    d_phi: int,
    n_blocks: int,
    dtype_str: str,
    run_backward: bool,
    static_estimate: dict,
    all_results: list,
    train_num_nodes: int,
    expected_world_size: int,
    safety_factor: float,
    output_stage: str,
) -> None:
    """Assemble aggregated JSON diagnostics and upload to the Snowflake stage."""
    # Compute summary from valid rank results
    valid_ranks = [
        r for r in all_results
        if r is not None and "peak_memory_reserved_bytes" in r
    ]
    if valid_ranks:
        max_peak_reserved  = max(r["peak_memory_reserved_bytes"]      for r in valid_ranks)
        max_peak_allocated = max(r["peak_memory_allocated_bytes"]      for r in valid_ranks)
        total_mems = [
            r["cuda_total_memory_bytes"] for r in valid_ranks
            if r.get("cuda_total_memory_bytes", 0) > 0
        ]
        max_reserved_frac  = max_peak_reserved / total_mems[0] if total_mems else None
        min_free_before    = min(r["cuda_free_memory_bytes_before"]    for r in valid_ranks)
    else:
        max_peak_reserved = max_peak_allocated = max_reserved_frac = min_free_before = None

    any_error = any(
        r is not None and r.get("status") == "error" for r in all_results
    )
    skipped = any(
        r is not None and r.get("status") == "skipped_static_memory_guard"
        for r in all_results
    )

    if skipped:
        overall_status = "skipped_static_memory_guard"
    elif any_error:
        overall_status = "error"
    else:
        overall_status = "ok"

    payload = {
        "status":                overall_status,
        "probe_type":            "model_ddp_memory_probe",
        "model_design_pattern": model_design_pattern,
        "model_family":          model_family,
        "shape": {
            "n_context":    n_context,
            "p_features":   p_features,
            "m_query":      m_query,
            "d_phi":        d_phi,
            "n_blocks":     n_blocks,
            "dtype":        dtype_str,
            "run_backward": run_backward,
        },
        "static_estimate": static_estimate,
        "world": {
            "train_num_nodes":     train_num_nodes,
            "expected_world_size": expected_world_size,
            "actual_world_size":   len([r for r in all_results if r is not None]),
        },
        "ranks":   [r for r in all_results if r is not None],
        "summary": {
            "max_peak_reserved_bytes":       max_peak_reserved,
            "max_peak_allocated_bytes":      max_peak_allocated,
            "max_reserved_fraction":         max_reserved_frac,
            "min_free_memory_before_bytes":  min_free_before,
        },
    }

    local_path = os.path.join(tempfile.gettempdir(), "model_ddp_memory_probe.json")
    with open(local_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print(
        f"[model_ddp_memory_probe] rank=0 diagnostics written to {local_path}",
        flush=True,
    )
    print(
        "[model_ddp_memory_probe] PROBE JSON:\n"
        + json.dumps(payload, indent=2, sort_keys=True),
        flush=True,
    )

    # Upload to Snowflake stage
    try:
        from snowflake.snowpark import Session as _Session
        _session = _Session.builder.getOrCreate()
        _session.file.put(
            local_path, output_stage,
            overwrite=True, auto_compress=False,
        )
        print(
            f"[model_ddp_memory_probe] uploaded {local_path} to {output_stage}",
            flush=True,
        )
    except Exception as upload_exc:
        print(
            f"[model_ddp_memory_probe] upload to {output_stage} failed: {upload_exc}. "
            "JSON payload was printed above for manual retrieval.",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Per-worker probe function (invoked by PyTorchDistributor on every GPU worker)
# ---------------------------------------------------------------------------

def probe_fn():
    """Invoked by PyTorchDistributor on each DDP worker.

    1. Reads probe config from env vars (injected by the launcher).
    2. Performs static H-tensor memory estimate and checks against hard limits.
    3. Allocates synthetic tensors; instantiates and wraps model in DDP.
    4. Runs forward pass, computes MSE loss.
    5. Optionally runs backward pass (required for training-regime validation).
    6. Measures peak CUDA allocated/reserved memory.
    7. Aggregates all rank results to rank 0 via dist.all_gather_object.
    8. Rank 0 uploads JSON diagnostics.
    """
    print("[model_ddp_memory_probe] probe_fn entered", flush=True)
    import torch.distributed as dist
    from model import ModelConfig, _instantiate_model

    ctx        = get_context()
    local_rank = ctx.get_local_rank() if hasattr(ctx, "get_local_rank") else ctx.local_rank
    rank       = ctx.get_rank()       if hasattr(ctx, "get_rank")       else ctx.rank
    world_size = ctx.get_world_size() if hasattr(ctx, "get_world_size") else ctx.world_size
    is_main    = (rank == 0)
    device_str = f"cuda:{local_rank}"
    device     = torch.device(device_str)
    hostname   = socket.gethostname()

    # Read shape / probe config from env (propagated by the launcher)
    n_context    = int(os.environ.get("MODEL_PROBE_N_CONTEXT",  "200"))
    p_features   = int(os.environ.get("MODEL_PROBE_P_FEATURES", "128"))
    m_query      = int(os.environ.get("MODEL_PROBE_M_QUERY",    "128"))
    d_phi        = int(os.environ.get("MODEL_PROBE_D_PHI",      "128"))
    n_blocks     = int(os.environ.get("MODEL_PROBE_N_BLOCKS",   "1"))
    run_backward = os.environ.get("MODEL_PROBE_RUN_BACKWARD", "true").lower() == "true"
    dtype_str    = os.environ.get("MODEL_PROBE_DTYPE", "float32")
    max_fraction = float(os.environ.get("MODEL_PROBE_MAX_GPU_MEMORY_FRACTION", "0.9"))
    safety_factor = float(os.environ.get("MODEL_PROBE_MEMORY_SAFETY_FACTOR", "1.5"))
    strict_guard = os.environ.get("MODEL_PROBE_STRICT_MEMORY_GUARD", "true").lower() == "true"
    max_tensor_bytes_env = os.environ.get("MODEL_PROBE_MAX_TENSOR_BYTES", "")
    max_tensor_bytes = int(max_tensor_bytes_env) if max_tensor_bytes_env else None
    output_stage = os.environ.get("MODEL_PROBE_OUTPUT_STAGE", "@MODEL_STAGE/diagnostics/")

    model_design_pattern = os.environ.get("MODEL_DESIGN_PATTERN", "inductive_forecasting")
    model_family  = os.environ.get("MODEL_FAMILY",  "market_exchangeable_icl")
    train_num_nodes       = int(os.environ.get("TRAIN_NUM_NODES", "10"))
    expected_ws           = int(os.environ.get("EXPECTED_TRAIN_WORLD_SIZE", "40"))
    strict_ws             = os.environ.get("STRICT_WORLD_SIZE_CHECK", "true").lower() == "true"

    print(
        f"[model_ddp_memory_probe] rank={rank} local_rank={local_rank} "
        f"world_size={world_size} device={device_str} host={hostname} "
        f"TRAIN_NUM_NODES={train_num_nodes} EXPECTED_TRAIN_WORLD_SIZE={expected_ws} "
        f"shape=(n={n_context}, p={p_features}, m={m_query}, d_phi={d_phi}, "
        f"n_blocks={n_blocks}) run_backward={run_backward} dtype={dtype_str}",
        flush=True,
    )

    # World-size topology check (mirrors train_fn logic)
    if is_main:
        print(
            f"[model_ddp_memory_probe] topology: actual_world_size={world_size} "
            f"expected_world_size={expected_ws if expected_ws > 0 else '(not set)'} "
            f"STRICT_WORLD_SIZE_CHECK={strict_ws}",
            flush=True,
        )
    if expected_ws > 0 and strict_ws and world_size != expected_ws:
        raise RuntimeError(
            f"[model_ddp_memory_probe] World-size mismatch: "
            f"EXPECTED_TRAIN_WORLD_SIZE={expected_ws} but actual world_size={world_size}. "
            f"TRAIN_NUM_NODES={train_num_nodes}, num_workers_per_node=4. "
            "Verify target_instances in submit_from_stage() equals num_nodes."
        )

    # Base result dict (always populated, used in error path too)
    rank_result = {
        "rank":        rank,
        "local_rank":  local_rank,
        "world_size":  world_size,
        "hostname":    hostname,
        "device":      device_str,
    }

    # Static memory estimate (computed before any allocation)
    h_bytes = estimate_h_tensor_bytes(n_context, p_features, m_query, d_phi, dtype_str)
    est_reserved, act_factor = estimate_reserved_bytes(h_bytes, run_backward, safety_factor)
    static_estimate = {
        "h_tensor_bytes":           h_bytes,
        "estimated_reserved_bytes": est_reserved,
        "activation_factor":        act_factor,
        "safety_factor":            safety_factor,
    }

    try:
        # ---- CUDA device info (before any allocation) ----
        cuda_device_name = (
            torch.cuda.get_device_name(device) if torch.cuda.is_available() else "N/A"
        )
        cuda_total_bytes = (
            torch.cuda.get_device_properties(device).total_memory
            if torch.cuda.is_available() else 0
        )
        cuda_free_before = cuda_total_bytes - torch.cuda.memory_allocated(device)

        rank_result["cuda_device_name"]               = cuda_device_name
        rank_result["cuda_total_memory_bytes"]        = cuda_total_bytes
        rank_result["cuda_free_memory_bytes_before"]  = cuda_free_before

        print(
            f"[model_ddp_memory_probe] rank={rank} "
            f"cuda_device={cuda_device_name} "
            f"cuda_total={cuda_total_bytes/1e9:.2f}GB "
            f"cuda_free_before={cuda_free_before/1e9:.2f}GB "
            f"h_tensor_estimate={h_bytes/1e9:.3f}GB "
            f"est_reserved={est_reserved/1e9:.3f}GB "
            f"(activation_factor={act_factor}, safety_factor={safety_factor})",
            flush=True,
        )

        # ---- Static memory guard ----
        budget_bytes = (
            int(cuda_total_bytes * max_fraction) if cuda_total_bytes > 0 else None
        )
        candidates = [b for b in (max_tensor_bytes, budget_bytes) if b is not None]
        effective_limit = min(candidates) if candidates else None

        if effective_limit is not None and est_reserved > effective_limit:
            msg = (
                f"[model_ddp_memory_probe] rank={rank} static memory guard triggered: "
                f"estimated_reserved_bytes={est_reserved:,} > limit={effective_limit:,} "
                f"(cuda_total={cuda_total_bytes:,}, max_fraction={max_fraction}, "
                f"max_tensor_bytes_override={max_tensor_bytes}). "
                f"H-tensor estimate: m={m_query}*n={n_context}*p={p_features}*d={d_phi}"
                f"*{_DTYPE_BYTES[dtype_str]}B = {h_bytes:,}B; "
                f"activation_factor={act_factor}; safety_factor={safety_factor}. "
                "Reduce shape parameters or increase GPU memory limit."
            )
            print(msg, flush=True)

            guard_result = {
                "status":                  "skipped_static_memory_guard",
                "rank":                    rank,
                "local_rank":              local_rank,
                "world_size":              world_size,
                "hostname":                hostname,
                "device":                  device_str,
                "cuda_device_name":        cuda_device_name,
                "cuda_total_memory_bytes": cuda_total_bytes,
                "static_estimate":         static_estimate,
                "effective_limit_bytes":   effective_limit,
                "message":                 msg,
            }

            all_guard = [None] * world_size
            if dist.is_available() and dist.is_initialized():
                dist.all_gather_object(all_guard, guard_result)
            else:
                all_guard = [guard_result]

            if is_main:
                _upload_probe_result(
                    model_design_pattern, model_family,
                    n_context, p_features, m_query, d_phi, n_blocks,
                    dtype_str, run_backward, static_estimate, all_guard,
                    train_num_nodes, expected_ws, safety_factor, output_stage,
                )
            if strict_guard:
                raise RuntimeError(msg)
            return

        # ---- Allocate synthetic tensors ----
        torch_dtype = torch.float32 if dtype_str == "float32" else torch.bfloat16
        torch.manual_seed(rank + 42)
        X_train = torch.randn(n_context, p_features, dtype=torch_dtype, device=device)
        y_train = torch.randn(n_context,             dtype=torch_dtype, device=device)
        x_test  = torch.randn(m_query,   p_features, dtype=torch_dtype, device=device)
        target  = torch.randn(m_query,               dtype=torch_dtype, device=device)

        # ---- Instantiate model ----
        cfg = ModelConfig(
            model_arch_version="model3",
            model_design_pattern="inductive_forecasting",
            model_family=model_family,
            d_phi=d_phi,
            n_sab_feat=n_blocks,
            n_heads=4,
            norm_feat=True,
            norm_target=True,
            dropout=0.0,     # deterministic probe (no dropout noise)
        )
        model = _instantiate_model(cfg)
        model = model.to(device).to(torch_dtype)
        param_count = sum(p.numel() for p in model.parameters())

        # ---- DDP wrap (mirrors train_fn) ----
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        model_ddp = DistributedDataParallel(model, device_ids=[local_rank])

        # ---- Reset peak stats immediately before the measured pass ----
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        t_start = time.perf_counter()

        # ---- Forward pass ----
        model_ddp.train()
        y_hat = model_ddp(X_train, y_train, x_test)
        forward_ok    = True
        output_shape  = list(y_hat.shape)

        # ---- Loss + backward (training-regime validation) ----
        backward_ok = False
        if run_backward:
            loss = F.mse_loss(y_hat, target)
            loss.backward()
            backward_ok = True

        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t_start

        # ---- Peak memory stats ----
        peak_allocated = torch.cuda.max_memory_allocated(device)
        peak_reserved  = torch.cuda.max_memory_reserved(device)
        curr_allocated = torch.cuda.memory_allocated(device)
        curr_reserved  = torch.cuda.memory_reserved(device)

        rank_result.update({
            "peak_memory_allocated_bytes":    peak_allocated,
            "peak_memory_reserved_bytes":     peak_reserved,
            "current_memory_allocated_bytes": curr_allocated,
            "current_memory_reserved_bytes":  curr_reserved,
            "parameter_count":                param_count,
            "forward_ok":                     forward_ok,
            "backward_ok":                    backward_ok,
            "output_shape":                   output_shape,
            "elapsed_seconds":                elapsed,
        })

        print(
            f"[model_ddp_memory_probe] rank={rank} "
            f"peak_allocated={peak_allocated/1e9:.3f}GB "
            f"peak_reserved={peak_reserved/1e9:.3f}GB "
            f"params={param_count:,} "
            f"forward_ok={forward_ok} backward_ok={backward_ok} "
            f"elapsed={elapsed:.2f}s",
            flush=True,
        )

    except Exception as exc:
        tb_str = traceback.format_exc()
        rank_result["status"]      = "error"
        rank_result["error"]       = str(exc)
        rank_result["traceback"]   = tb_str
        rank_result["forward_ok"]  = False
        rank_result["backward_ok"] = False
        print(
            f"[model_ddp_memory_probe] rank={rank} ERROR: {exc}\n{tb_str}",
            flush=True,
        )

    # ---- Aggregate across all ranks to rank 0 ----
    all_results = [None] * world_size
    try:
        if dist.is_available() and dist.is_initialized():
            dist.all_gather_object(all_results, rank_result)
        else:
            all_results = [rank_result]
    except Exception as gather_exc:
        print(
            f"[model_ddp_memory_probe] rank={rank} all_gather_object failed: "
            f"{gather_exc}; uploading rank-0 result only.",
            flush=True,
        )
        all_results = [rank_result]

    # ---- Rank 0: assemble summary and upload ----
    if is_main:
        _upload_probe_result(
            model_design_pattern, model_family,
            n_context, p_features, m_query, d_phi, n_blocks,
            dtype_str, run_backward, static_estimate, all_results,
            train_num_nodes, expected_ws, safety_factor, output_stage,
        )


# ---------------------------------------------------------------------------
# Entry point — submitted by submit_from_stage; launches probe_fn via
# PyTorchDistributor on DEEPSET_GPU_POOL
# ---------------------------------------------------------------------------

def main():
    print("[model_ddp_memory_probe] main() started", flush=True)
    print(
        f"[model_ddp_memory_probe] config: "
        f"model_design_pattern={MODEL_DESIGN_PATTERN!r} "
        f"model_family={MODEL_FAMILY!r} "
        f"n_context={MODEL_PROBE_N_CONTEXT} "
        f"p_features={MODEL_PROBE_P_FEATURES} "
        f"m_query={MODEL_PROBE_M_QUERY} "
        f"d_phi={MODEL_PROBE_D_PHI} "
        f"n_blocks={MODEL_PROBE_N_BLOCKS} "
        f"run_backward={MODEL_PROBE_RUN_BACKWARD} "
        f"dtype={MODEL_PROBE_DTYPE} "
        f"TRAIN_NUM_NODES={TRAIN_NUM_NODES}",
        flush=True,
    )

    # Validate before consuming GPU pool resources
    validate_probe_config(
        MODEL_DESIGN_PATTERN, MODEL_FAMILY,
        MODEL_PROBE_N_CONTEXT, MODEL_PROBE_P_FEATURES, MODEL_PROBE_M_QUERY,
        MODEL_PROBE_D_PHI, MODEL_PROBE_N_BLOCKS, MODEL_PROBE_DTYPE,
    )

    if PyTorchDistributor is None:
        raise RuntimeError(
            "snowflake.ml is required. Run inside the Snowflake ML runtime or "
            "install snowflake-ml-python."
        )

    distributor = PyTorchDistributor(
        train_func=probe_fn,
        scaling_config=PyTorchScalingConfig(
            num_nodes=TRAIN_NUM_NODES,
            num_workers_per_node=4,   # 4 A10G GPUs per GPU_NV_M node
            resource_requirements_per_worker=WorkerResourceConfig(
                num_cpus=4,
                num_gpus=1,
            ),
        ),
    )
    print(
        f"[model_ddp_memory_probe] launching PyTorchDistributor: "
        f"TRAIN_NUM_NODES={TRAIN_NUM_NODES} workers_per_node=4 "
        f"world_size={TRAIN_NUM_NODES * 4}",
        flush=True,
    )
    try:
        result = distributor.run(
            artifact_stage_location="TABPFN_DB.TABPFN_SCHEMA.MODEL_STAGE"
        )
        print(
            "[model_ddp_memory_probe] PyTorchDistributor.run completed",
            flush=True,
        )
    except Exception as exc:
        print(
            f"[model_ddp_memory_probe] PyTorchDistributor.run failed: {exc!r}",
            flush=True,
        )
        raise
    print("[model_ddp_memory_probe] result:", result, flush=True)


if __name__ == "__main__":
    main()
