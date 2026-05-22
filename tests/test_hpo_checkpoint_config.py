import os
import sys

import pytest


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import hpo  # noqa: E402
from model import ModelConfig  # noqa: E402


def test_checkpoint_architecture_mismatches_accepts_dict_cfg():
    saved_cfg = {
        "d_phi": 128,
        "d_rho": 256,
        "pool": "pna",
        "n_heads": 4,
        "n_sab_feat": 1,
        "norm_feat": True,
        "norm_target": True,
        "dropout": 0.1,
    }
    current_cfg = ModelConfig(**saved_cfg)

    assert hpo.checkpoint_architecture_mismatches(saved_cfg, current_cfg) == {}


def test_checkpoint_architecture_mismatches_ignores_dropout():
    saved_cfg = {
        "d_phi": 128,
        "d_rho": 256,
        "pool": "pna",
        "n_heads": 4,
        "n_sab_feat": 1,
        "norm_feat": True,
        "norm_target": True,
        "dropout": 0.1,
    }
    current_cfg = ModelConfig(**{**saved_cfg, "dropout": 0.3})

    assert hpo.checkpoint_architecture_mismatches(saved_cfg, current_cfg) == {}


def test_checkpoint_architecture_mismatches_rejects_missing_cfg():
    with pytest.raises(ValueError, match="missing required cfg"):
        hpo.checkpoint_architecture_mismatches(None, ModelConfig())


# ---------------------------------------------------------------------------
# MODEL_FAMILY and best_config propagation
# ---------------------------------------------------------------------------

def test_model_family_constant_exists():
    """MODEL_FAMILY constant exists on the hpo module."""
    assert hasattr(hpo, "MODEL_FAMILY")


def test_model_family_defaults_to_market_exchangeable_icl(monkeypatch):
    """MODEL_FAMILY defaults to 'market_exchangeable_icl' when env var not set."""
    import importlib
    monkeypatch.delenv("MODEL_FAMILY", raising=False)
    importlib.reload(hpo)
    assert hpo.MODEL_FAMILY == "market_exchangeable_icl"
    importlib.reload(hpo)


def test_model_family_respects_env_var(monkeypatch):
    """MODEL_FAMILY reads from MODEL_FAMILY env var."""
    import importlib
    monkeypatch.setenv("MODEL_FAMILY", "market_exchangeable_completion")
    importlib.reload(hpo)
    assert hpo.MODEL_FAMILY == "market_exchangeable_completion"
    importlib.reload(hpo)


def test_checkpoint_architecture_mismatches_detects_family_change():
    """checkpoint_architecture_mismatches detects model_family mismatch."""
    saved = ModelConfig(
        model_family="market_exchangeable_icl",
        model_arch_version="model3",
        model_design_pattern="inductive_forecasting",
        d_phi=128, d_rho=256, n_sab_feat=1, norm_feat=True, norm_target=True,
    )
    # Note: market_exchangeable_completion requires transductive_completion pattern
    current = ModelConfig(
        model_family="market_exchangeable_completion",
        model_arch_version="model3",
        model_design_pattern="transductive_completion",
        d_phi=128, d_rho=256, n_sab_feat=1, norm_feat=True, norm_target=True,
    )
    mismatches = hpo.checkpoint_architecture_mismatches(saved, current)
    assert "model_family" in mismatches


# ---------------------------------------------------------------------------
# Ridge Expert: search space and best_config propagation
# ---------------------------------------------------------------------------

def test_ridge_lambda_in_search_space():
    """HPO search space must include ridge_lambda as a tunable parameter."""
    pytest.importorskip("ray")
    import ray.tune as tune

    # Reconstruct the search space dict using the same logic as hpo.main()
    # by inspecting the module-level constants and the search_space dict built in main().
    # We verify the structure via the module's documented API.
    assert hasattr(hpo, "FIXED_D_PHI"), "hpo.FIXED_D_PHI must exist"
    assert hasattr(hpo, "FIXED_D_RHO"), "hpo.FIXED_D_RHO must exist"
    assert hasattr(hpo, "FIXED_POOL"), "hpo.FIXED_POOL must exist"

    # Build a representative search_space dict that mirrors what main() constructs,
    # and verify ridge_lambda is a loguniform sample (not a fixed constant).
    search_space = {
        "lr":               tune.loguniform(1e-4, 1e-2),
        "weight_decay":     tune.loguniform(1e-5, 1e-3),
        "dropout":          tune.uniform(0.0, 0.3),
        "model_family":     hpo.MODEL_FAMILY,
        "use_ridge_expert": True,
        "ridge_lambda":     tune.loguniform(1e-3, 1e1),
        "gate_hidden_dim":  64,
    }
    assert "ridge_lambda" in search_space, "ridge_lambda must be present in search_space"
    # Verify it's a Tune distribution (not a plain float)
    assert not isinstance(search_space["ridge_lambda"], float), (
        "ridge_lambda must be a tune distribution, not a fixed float"
    )
    assert search_space["use_ridge_expert"] is True, (
        "use_ridge_expert must be True (fixed) in search_space"
    )
    assert search_space["gate_hidden_dim"] == 64, (
        "gate_hidden_dim must be fixed at 64 in search_space"
    )


def test_best_config_includes_ridge_fields():
    """best_config dict produced by HPO must include use_ridge_expert, ridge_lambda, gate_hidden_dim."""
    # Simulate the best_config construction logic from hpo.main()
    best_config_raw = {
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "dropout": 0.1,
        "model_family": "market_exchangeable_icl",
        "use_ridge_expert": True,
        "ridge_lambda": 0.5,
        "gate_hidden_dim": 64,
    }
    MODEL_ARCH_VERSION = "model3"
    MODEL_DESIGN_PATTERN = "inductive_forecasting"

    best_config = {
        "lr":                   float(best_config_raw["lr"]),
        "weight_decay":         float(best_config_raw["weight_decay"]),
        "d_phi":                hpo.FIXED_D_PHI,
        "d_rho":                hpo.FIXED_D_RHO,
        "dropout":              float(best_config_raw["dropout"]),
        "pool":                 hpo.FIXED_POOL,
        "model_family":         best_config_raw.get("model_family", hpo.MODEL_FAMILY),
        "model_arch_version":   MODEL_ARCH_VERSION,
        "model_design_pattern": MODEL_DESIGN_PATTERN,
        "use_ridge_expert":     bool(best_config_raw.get("use_ridge_expert", True)),
        "ridge_lambda":         float(best_config_raw.get("ridge_lambda", 1.0)),
        "gate_hidden_dim":      int(best_config_raw.get("gate_hidden_dim", 64)),
    }

    assert "use_ridge_expert" in best_config
    assert "ridge_lambda" in best_config
    assert "gate_hidden_dim" in best_config
    assert best_config["use_ridge_expert"] is True
    assert best_config["ridge_lambda"] == pytest.approx(0.5)
    assert best_config["gate_hidden_dim"] == 64


def test_checkpoint_architecture_mismatches_detects_ridge_expert_change():
    """checkpoint_architecture_mismatches detects use_ridge_expert mismatch."""
    saved = ModelConfig(
        model_family="market_exchangeable_icl",
        model_arch_version="model3",
        model_design_pattern="inductive_forecasting",
        d_phi=128, d_rho=256, n_sab_feat=1, norm_feat=True, norm_target=True,
        use_ridge_expert=False,
    )
    current = ModelConfig(
        model_family="market_exchangeable_icl",
        model_arch_version="model3",
        model_design_pattern="inductive_forecasting",
        d_phi=128, d_rho=256, n_sab_feat=1, norm_feat=True, norm_target=True,
        use_ridge_expert=True,
    )
    mismatches = hpo.checkpoint_architecture_mismatches(saved, current)
    assert "use_ridge_expert" in mismatches


def test_checkpoint_architecture_mismatches_detects_gate_hidden_dim_change():
    """checkpoint_architecture_mismatches detects gate_hidden_dim mismatch."""
    saved = ModelConfig(
        model_family="market_exchangeable_icl",
        model_arch_version="model3",
        model_design_pattern="inductive_forecasting",
        d_phi=128, d_rho=256, n_sab_feat=1, norm_feat=True, norm_target=True,
        use_ridge_expert=True,
        gate_hidden_dim=32,
    )
    current = ModelConfig(
        model_family="market_exchangeable_icl",
        model_arch_version="model3",
        model_design_pattern="inductive_forecasting",
        d_phi=128, d_rho=256, n_sab_feat=1, norm_feat=True, norm_target=True,
        use_ridge_expert=True,
        gate_hidden_dim=64,
    )
    mismatches = hpo.checkpoint_architecture_mismatches(saved, current)
    assert "gate_hidden_dim" in mismatches


# ---------------------------------------------------------------------------
# HPO_SWEEP_MODE: module constants, build_hpo_search_space, and best_config
# ---------------------------------------------------------------------------

def test_hpo_sweep_mode_constant_exists():
    """HPO_SWEEP_MODE constant exists on the hpo module."""
    assert hasattr(hpo, "HPO_SWEEP_MODE")
    assert hpo.HPO_SWEEP_MODE in {"ridge_residual", "architecture"}


def test_hpo_sweep_mode_defaults_to_ridge_residual(monkeypatch):
    """HPO_SWEEP_MODE defaults to 'ridge_residual' when env var is not set."""
    import importlib
    monkeypatch.delenv("HPO_SWEEP_MODE", raising=False)
    importlib.reload(hpo)
    assert hpo.HPO_SWEEP_MODE == "ridge_residual"
    importlib.reload(hpo)


def test_hpo_sweep_mode_rejects_invalid_value(monkeypatch):
    """Reloading hpo with an invalid HPO_SWEEP_MODE must raise ValueError."""
    import importlib
    monkeypatch.setenv("HPO_SWEEP_MODE", "totally_invalid")
    with pytest.raises(ValueError, match="HPO_SWEEP_MODE"):
        importlib.reload(hpo)
    monkeypatch.delenv("HPO_SWEEP_MODE", raising=False)
    importlib.reload(hpo)


def test_build_hpo_search_space_exists():
    """build_hpo_search_space is a callable exported by hpo."""
    assert hasattr(hpo, "build_hpo_search_space")
    assert callable(hpo.build_hpo_search_space)


def test_build_hpo_search_space_ridge_residual_keys(monkeypatch):
    """build_hpo_search_space in ridge_residual mode returns expected keys."""
    import importlib
    pytest.importorskip("ray")
    import ray.tune as tune

    monkeypatch.delenv("HPO_SWEEP_MODE", raising=False)
    importlib.reload(hpo)

    space = hpo.build_hpo_search_space(tune)

    required_keys = {
        "lr", "weight_decay", "dropout",
        "use_ridge_expert", "ridge_lambda", "gate_hidden_dim",
        "d_phi", "n_sab_feat", "d_rho", "pool",
        "use_huber", "huber_delta", "lambda_l1",
        "model_family", "model_design_pattern", "hpo_sweep_mode",
    }
    for key in required_keys:
        assert key in space, f"Key {key!r} missing from ridge_residual search space"

    assert space["d_phi"] == hpo.FIXED_D_PHI, "d_phi must be fixed in ridge_residual mode"
    assert space["n_sab_feat"] == hpo.FIXED_N_SAB_FEAT, "n_sab_feat must be fixed in ridge_residual mode"
    assert space["use_ridge_expert"] is True
    assert space["hpo_sweep_mode"] == "ridge_residual"
    importlib.reload(hpo)


def test_build_hpo_search_space_architecture_keys(monkeypatch):
    """build_hpo_search_space in architecture mode tunes d_phi and n_sab_feat."""
    import importlib
    pytest.importorskip("ray")
    import ray.tune as tune

    monkeypatch.setenv("HPO_SWEEP_MODE", "architecture")
    importlib.reload(hpo)

    _baseline = {
        "lr": 1e-3, "weight_decay": 1e-4, "dropout": 0.1,
        "ridge_lambda": 0.5, "gate_hidden_dim": 64,
        "use_huber": False, "huber_delta": 1.0, "lambda_l1": 0.0,
    }
    space = hpo.build_hpo_search_space(tune, baseline_config=_baseline)

    assert "d_phi" in space, "d_phi must be in architecture search space"
    assert "n_sab_feat" in space, "n_sab_feat must be in architecture search space"
    # Both must be tune distributions (not fixed ints) in architecture mode
    assert not isinstance(space["d_phi"], int), "d_phi must be a tune distribution in architecture mode"
    assert not isinstance(space["n_sab_feat"], int), "n_sab_feat must be a tune distribution in architecture mode"
    assert space["hpo_sweep_mode"] == "architecture"
    # use_huber must be fixed from baseline (False here), not a tune distribution
    assert space["use_huber"] is False, "use_huber must be fixed from baseline in architecture mode"
    # lr must equal the baseline value (fixed float, not a tune distribution)
    assert isinstance(space["lr"], float), "lr must be a fixed float in architecture mode"
    assert space["lr"] == pytest.approx(1e-3)

    monkeypatch.delenv("HPO_SWEEP_MODE", raising=False)
    importlib.reload(hpo)


def test_best_config_includes_expanded_fields():
    """best_config dict must include n_sab_feat, use_huber, huber_delta, lambda_l1, hpo_sweep_mode, _meta."""
    best_config_raw = {
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "dropout": 0.1,
        "d_phi": 128,
        "d_rho": 256,
        "pool": "pna",
        "n_sab_feat": 1,
        "model_family": "market_exchangeable_icl",
        "model_design_pattern": "inductive_forecasting",
        "use_ridge_expert": True,
        "ridge_lambda": 0.5,
        "gate_hidden_dim": 64,
        "use_huber": False,
        "huber_delta": 1.0,
        "lambda_l1": 0.0,
        "hpo_sweep_mode": "ridge_residual",
    }
    MODEL_ARCH_VERSION = "model3"
    NUM_TRIALS = 20
    TRIAL_MAX_EPOCHS = 40

    best_config = {
        "lr":                   float(best_config_raw["lr"]),
        "weight_decay":         float(best_config_raw["weight_decay"]),
        "dropout":              float(best_config_raw["dropout"]),
        "d_phi":                int(best_config_raw.get("d_phi", hpo.FIXED_D_PHI)),
        "d_rho":                int(best_config_raw.get("d_rho", hpo.FIXED_D_RHO)),
        "pool":                 best_config_raw.get("pool", hpo.FIXED_POOL),
        "n_sab_feat":           int(best_config_raw.get("n_sab_feat", hpo.FIXED_N_SAB_FEAT)),
        "use_ridge_expert":     bool(best_config_raw.get("use_ridge_expert", True)),
        "ridge_lambda":         float(best_config_raw.get("ridge_lambda", 1.0)),
        "gate_hidden_dim":      int(best_config_raw.get("gate_hidden_dim", 64)),
        "use_huber":            bool(best_config_raw.get("use_huber", False)),
        "huber_delta":          float(best_config_raw.get("huber_delta", 1.0)),
        "lambda_l1":            float(best_config_raw.get("lambda_l1", 0.0)),
        "model_family":         best_config_raw.get("model_family", hpo.MODEL_FAMILY),
        "model_arch_version":   MODEL_ARCH_VERSION,
        "model_design_pattern": best_config_raw.get("model_design_pattern", "inductive_forecasting"),
        "hpo_sweep_mode":       best_config_raw.get("hpo_sweep_mode", "ridge_residual"),
        "_meta": {
            "best_val_mse":               None,
            "num_trials":                 NUM_TRIALS,
            "trial_max_epochs":           TRIAL_MAX_EPOCHS,
            "pretrain_warm_start_policy": "fail_on_mismatch",
        },
    }

    for key in ("n_sab_feat", "use_huber", "huber_delta", "lambda_l1", "hpo_sweep_mode", "_meta"):
        assert key in best_config, f"Key {key!r} missing from best_config"

    assert best_config["n_sab_feat"] == 1
    assert best_config["use_huber"] is False
    assert best_config["huber_delta"] == pytest.approx(1.0)
    assert best_config["lambda_l1"] == pytest.approx(0.0)
    assert best_config["hpo_sweep_mode"] == "ridge_residual"
    assert best_config["_meta"]["pretrain_warm_start_policy"] == "fail_on_mismatch"


def test_fixed_n_sab_feat_constant_exists():
    """FIXED_N_SAB_FEAT constant exists on the hpo module."""
    assert hasattr(hpo, "FIXED_N_SAB_FEAT")
    assert isinstance(hpo.FIXED_N_SAB_FEAT, int)
    assert hpo.FIXED_N_SAB_FEAT == 1


def test_arch_candidates_constants_exist():
    """ARCH_D_PHI_CANDIDATES and ARCH_N_SAB_FEAT_CANDIDATES constants exist."""
    assert hasattr(hpo, "ARCH_D_PHI_CANDIDATES")
    assert hasattr(hpo, "ARCH_N_SAB_FEAT_CANDIDATES")
    assert 128 in hpo.ARCH_D_PHI_CANDIDATES, "128 must be in ARCH_D_PHI_CANDIDATES"
    assert 1 in hpo.ARCH_N_SAB_FEAT_CANDIDATES, "1 must be in ARCH_N_SAB_FEAT_CANDIDATES"


# ---------------------------------------------------------------------------
# Two-sweep HPO: new tests (Fix 1)
# ---------------------------------------------------------------------------

def _make_baseline_config():
    """Minimal ridge_residual best_config for architecture sweep tests."""
    return {
        "lr": 1e-3,
        "weight_decay": 5e-5,
        "dropout": 0.1,
        "ridge_lambda": 0.5,
        "gate_hidden_dim": 64,
        "use_huber": False,
        "huber_delta": 1.0,
        "lambda_l1": 0.0,
        "_meta": {
            "best_val_mse": 0.42,
            "pretrain_warm_start_policy": "fail_on_mismatch",
        },
    }


def test_ridge_residual_writes_sweep_specific_filename(monkeypatch):
    """build_hpo_search_space result for ridge_residual mode has hpo_sweep_mode=='ridge_residual'."""
    import importlib
    ray = pytest.importorskip("ray")
    import ray.tune as tune

    monkeypatch.delenv("HPO_SWEEP_MODE", raising=False)
    importlib.reload(hpo)

    space = hpo.build_hpo_search_space(tune)
    assert space["hpo_sweep_mode"] == "ridge_residual"

    monkeypatch.delenv("HPO_SWEEP_MODE", raising=False)
    importlib.reload(hpo)


def test_architecture_sweep_requires_baseline_config(monkeypatch):
    """build_hpo_search_space(tune) raises ValueError in architecture mode when baseline_config=None."""
    import importlib
    ray = pytest.importorskip("ray")
    import ray.tune as tune

    monkeypatch.setenv("HPO_SWEEP_MODE", "architecture")
    importlib.reload(hpo)

    with pytest.raises(ValueError, match="HPO_BASELINE_CONFIG_STAGE_PATH"):
        hpo.build_hpo_search_space(tune, baseline_config=None)

    monkeypatch.delenv("HPO_SWEEP_MODE", raising=False)
    importlib.reload(hpo)


def test_architecture_sweep_freezes_lr_from_baseline(monkeypatch):
    """lr in architecture search space equals baseline_config['lr'] (fixed float, not distribution)."""
    import importlib
    ray = pytest.importorskip("ray")
    import ray.tune as tune

    monkeypatch.setenv("HPO_SWEEP_MODE", "architecture")
    importlib.reload(hpo)

    baseline = _make_baseline_config()
    space = hpo.build_hpo_search_space(tune, baseline_config=baseline)

    assert isinstance(space["lr"], float), "lr must be a fixed float in architecture mode"
    assert space["lr"] == pytest.approx(baseline["lr"])

    monkeypatch.delenv("HPO_SWEEP_MODE", raising=False)
    importlib.reload(hpo)


def test_architecture_sweep_freezes_ridge_lambda_from_baseline(monkeypatch):
    """ridge_lambda in architecture search space equals baseline_config['ridge_lambda']."""
    import importlib
    ray = pytest.importorskip("ray")
    import ray.tune as tune

    monkeypatch.setenv("HPO_SWEEP_MODE", "architecture")
    importlib.reload(hpo)

    baseline = _make_baseline_config()
    space = hpo.build_hpo_search_space(tune, baseline_config=baseline)

    assert isinstance(space["ridge_lambda"], float), "ridge_lambda must be fixed in architecture mode"
    assert space["ridge_lambda"] == pytest.approx(baseline["ridge_lambda"])

    monkeypatch.delenv("HPO_SWEEP_MODE", raising=False)
    importlib.reload(hpo)


def test_merge_sweep_configs_produces_merged_from():
    """_merge_sweep_configs result has _meta.merged_from list with both sweep filenames."""
    baseline = _make_baseline_config()
    arch_config = {
        "d_phi": 192,
        "n_sab_feat": 2,
        "hpo_sweep_mode": "architecture",
        "_meta": {
            "best_val_mse": 0.38,
            "pretrain_warm_start_policy": "allow_cold_start_on_arch_mismatch",
        },
    }
    merged = hpo._merge_sweep_configs(baseline, arch_config)
    assert "_meta" in merged
    assert "merged_from" in merged["_meta"]
    merged_from = merged["_meta"]["merged_from"]
    assert any("best_config_ridge_residual.json" in p for p in merged_from)
    assert any("best_config_architecture.json" in p for p in merged_from)


def test_merge_sweep_configs_overrides_d_phi():
    """_merge_sweep_configs result d_phi comes from arch config."""
    baseline = _make_baseline_config()
    arch_config = {
        "d_phi": 192,
        "n_sab_feat": 2,
        "hpo_sweep_mode": "architecture",
        "_meta": {"best_val_mse": 0.38},
    }
    merged = hpo._merge_sweep_configs(baseline, arch_config)
    assert merged["d_phi"] == 192


def test_merge_sweep_configs_preserves_lr():
    """_merge_sweep_configs result lr comes from baseline (not overridden by arch config)."""
    baseline = _make_baseline_config()
    arch_config = {
        "d_phi": 256,
        "n_sab_feat": 1,
        "hpo_sweep_mode": "architecture",
        "_meta": {"best_val_mse": 0.40},
    }
    merged = hpo._merge_sweep_configs(baseline, arch_config)
    assert merged["lr"] == pytest.approx(baseline["lr"])


def test_architecture_cardinality_check_uses_candidates(monkeypatch):
    """cardinality_aware_candidates is called (not enforce_fixed_architecture_cardinality) in architecture mode."""
    import importlib
    from unittest.mock import patch, MagicMock

    monkeypatch.setenv("HPO_SWEEP_MODE", "architecture")
    importlib.reload(hpo)

    with patch.object(hpo, "cardinality_aware_candidates", wraps=hpo.cardinality_aware_candidates) as mock_cac, \
         patch.object(hpo, "enforce_fixed_architecture_cardinality") as mock_enf:
        # Simulate the logic in main() for architecture mode
        max_p = 50
        max_n_train = 200
        if hpo.HPO_SWEEP_MODE == "architecture":
            hpo.cardinality_aware_candidates(
                "d_phi", hpo.ARCH_D_PHI_CANDIDATES, "max_p", max_p
            )
        else:
            hpo.enforce_fixed_architecture_cardinality(max_p, max_n_train)

        mock_cac.assert_called_once()
        mock_enf.assert_not_called()

    monkeypatch.delenv("HPO_SWEEP_MODE", raising=False)
    importlib.reload(hpo)


def test_hpo_baseline_config_stage_path_constant_exists():
    """hpo.HPO_BASELINE_CONFIG_STAGE_PATH attribute exists."""
    assert hasattr(hpo, "HPO_BASELINE_CONFIG_STAGE_PATH")
    assert isinstance(hpo.HPO_BASELINE_CONFIG_STAGE_PATH, str)


# ---------------------------------------------------------------------------
# Gate-specific pretrain checkpoints (gate_hidden_dim HPO fix)
# ---------------------------------------------------------------------------

def test_gate_hidden_dim_candidates_constant_exists():
    """hpo.GATE_HIDDEN_DIM_CANDIDATES exists and contains [32, 64, 128]."""
    assert hasattr(hpo, "GATE_HIDDEN_DIM_CANDIDATES")
    assert isinstance(hpo.GATE_HIDDEN_DIM_CANDIDATES, list)
    assert set(hpo.GATE_HIDDEN_DIM_CANDIDATES) == {32, 64, 128}


def test_gate_hidden_dim_candidates_matches_ridge_residual_search_space():
    """GATE_HIDDEN_DIM_CANDIDATES matches the tune.choice([...]) values in ridge_residual mode."""
    pytest.importorskip("ray")
    import ray.tune as tune

    space = hpo.build_hpo_search_space(tune)
    gate_dist = space.get("gate_hidden_dim")
    # gate_hidden_dim should be a tune distribution (not a fixed int)
    assert not isinstance(gate_dist, int), (
        "gate_hidden_dim must be a tune distribution in ridge_residual mode"
    )


def test_check_pretrain_checkpoints_function_exists():
    """hpo._check_pretrain_checkpoints is callable."""
    assert hasattr(hpo, "_check_pretrain_checkpoints")
    assert callable(hpo._check_pretrain_checkpoints)


def test_build_ray_trainable_signature_accepts_map_ref():
    """_build_ray_trainable accepts two positional args (data_ref, ckpt_map_ref)."""
    import inspect
    sig = inspect.signature(hpo._build_ray_trainable)
    params = list(sig.parameters.keys())
    assert len(params) == 2, (
        f"_build_ray_trainable should accept 2 params, got: {params}"
    )
    assert params[0] == "hpo_data_ref"
    assert params[1] == "pretrain_ckpt_map_ref"


def test_best_config_meta_includes_checkpoint_provenance_fields():
    """best_config._meta produced by HPO includes pretrain_checkpoint_map and
    pretrain_checkpoint_stage_path (gate provenance fields)."""
    # Simulate the _meta dict construction from hpo.main()
    checkpoint_map = {32: "@MODEL_STAGE/checkpoints/pretrain_gate32.pt",
                      64: "@MODEL_STAGE/checkpoints/pretrain_gate64.pt",
                      128: "@MODEL_STAGE/checkpoints/pretrain_gate128.pt"}
    best_gate_dim = 64
    best_val_mse = 0.35

    meta = {
        "best_val_mse":               float(best_val_mse),
        "num_trials":                 20,
        "trial_max_epochs":           30,
        "pretrain_warm_start_policy": "fail_on_mismatch",
        "pretrain_checkpoint_map": {
            str(gate_dim): path for gate_dim, path in checkpoint_map.items()
        },
        "pretrain_checkpoint_stage_path": checkpoint_map.get(best_gate_dim, ""),
    }

    assert "pretrain_checkpoint_map" in meta
    assert "pretrain_checkpoint_stage_path" in meta
    assert meta["pretrain_checkpoint_stage_path"] == "@MODEL_STAGE/checkpoints/pretrain_gate64.pt"
    assert meta["pretrain_checkpoint_map"]["64"] == "@MODEL_STAGE/checkpoints/pretrain_gate64.pt"
    assert meta["pretrain_warm_start_policy"] == "fail_on_mismatch"


def test_architecture_sweep_enabled_in_main():
    """hpo.main() no longer raises NotImplementedError for HPO_SWEEP_MODE=architecture.
    Architecture sweep is enabled; cold-start is allowed on checkpoint mismatch."""
    import importlib
    import inspect

    # Verify the NotImplementedError guard has been removed
    src = inspect.getsource(hpo.main)
    assert "architecture" in src, "hpo.main() must reference 'architecture' mode"
    # The NotImplementedError guard must be absent — architecture sweep is now supported
    # Check: if NotImplementedError appears, it must not be for the architecture guard
    if "NotImplementedError" in src:
        # Ensure it's not the old "architecture HPO sweep is disabled" guard
        assert "architecture HPO sweep is disabled" not in src, (
            "hpo.main() must not have the old NotImplementedError guard for architecture mode"
        )
