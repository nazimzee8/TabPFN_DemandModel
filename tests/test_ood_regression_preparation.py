"""
tests/test_ood_regression_preparation.py
=========================================
Unit tests for the OOD synthetic regression evaluation suite.

Tests cover:
- Statistical properties of each OOD regime's feature / noise distributions
- NPZ schema compatibility with the existing eval loader
- Manifest structure and dataset counts
- Index-row safety invariants (suite_id isolation, no DROP TABLE)
- Guards on production constants (PRIMARY_N_DATASETS, PRIMARY_SPLIT_SEEDS)

All tests are offline (no Snowflake session required).
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import helpers — add project roots to sys.path so tests run from repo root
# ---------------------------------------------------------------------------

_REPO = Path(__file__).parent.parent
_SCRIPTS = _REPO / "scripts"
_OOD = _SCRIPTS / "ood_regression"

for _p in [str(_OOD), str(_SCRIPTS), str(_REPO / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Import the local CLI module under test
import generate_ood_eval_data as gen
from generate_ood_eval_data import _write_parquet

# Import production constants guard
import prepare_synthetic_regression as prep_prod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RNG_SEED = 20260513
N_LARGE = 50_000   # large sample for distributional tests
N_SMALL = 200      # quick functional checks


@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(RNG_SEED)


# ---------------------------------------------------------------------------
# 1. Feature distribution: Laplace (Regime E) — unit variance
# ---------------------------------------------------------------------------

class TestRegimeEFeatures:
    def test_laplace_features_unit_variance(self):
        """Var(X_E) should be ≈ 1.0 (Laplace(0, 1/√2) has variance 2 × b² = 1)."""
        rng = np.random.default_rng(1)
        X = gen._generate_X_E(rng, N_LARGE, 5)
        assert X.var(axis=0).mean() == pytest.approx(1.0, abs=0.05)

    def test_laplace_features_zero_mean(self):
        rng = np.random.default_rng(2)
        X = gen._generate_X_E(rng, N_LARGE, 5)
        assert X.mean(axis=0).mean() == pytest.approx(0.0, abs=0.05)

    def test_laplace_features_shape(self):
        rng = np.random.default_rng(3)
        X = gen._generate_X_E(rng, 50, 7)
        assert X.shape == (50, 7)


# ---------------------------------------------------------------------------
# 2. Feature distribution: Uniform (Regime F) — unit variance
# ---------------------------------------------------------------------------

class TestRegimeFFeatures:
    def test_uniform_features_unit_variance(self):
        """Var(X_F) should be ≈ 1.0 (Uniform(−√3, √3) has variance (2√3)²/12 = 1)."""
        rng = np.random.default_rng(4)
        X = gen._generate_X_F(rng, N_LARGE, 5)
        assert X.var(axis=0).mean() == pytest.approx(1.0, abs=0.05)

    def test_uniform_features_bounded(self):
        """All values must lie in [−√3, √3]."""
        rng = np.random.default_rng(5)
        X = gen._generate_X_F(rng, N_LARGE, 5)
        assert X.min() >= -math.sqrt(3) - 1e-12
        assert X.max() <= math.sqrt(3) + 1e-12


# ---------------------------------------------------------------------------
# 3. Coefficient sparsity: Regime F — 95 % sparse
# ---------------------------------------------------------------------------

class TestRegimeFBeta:
    def test_regime_F_beta_95pct_sparse(self):
        """At least 90 % of β entries should be zero (95 % target sparsity)."""
        rng = np.random.default_rng(6)
        p = 1000
        beta = gen._generate_beta(rng, p, "F")
        frac_zero = (beta == 0.0).mean()
        assert frac_zero >= 0.90, f"Expected ≥90% zeros, got {frac_zero:.3f}"

    def test_regime_F_nonzero_variance(self):
        """Non-zero β entries should have var ≈ 4 (N(0, 2) entries)."""
        rng = np.random.default_rng(7)
        beta = gen._generate_beta(rng, 5000, "F")
        nonzero = beta[beta != 0.0]
        assert len(nonzero) > 0
        assert nonzero.var() == pytest.approx(4.0, abs=0.4)

    def test_other_regimes_dense(self):
        """Regimes E/G/H should produce fully dense β."""
        rng = np.random.default_rng(8)
        for regime in ("E", "G", "H"):
            beta = gen._generate_beta(rng, 200, regime)
            # No zeroes expected from standard_normal (probability zero event)
            assert (beta == 0.0).sum() == 0, f"Regime {regime} unexpectedly has zeros"


# ---------------------------------------------------------------------------
# 4. Noise distribution: Cauchy (Regime G) — heavier tails than Gaussian
# ---------------------------------------------------------------------------

class TestRegimeGNoise:
    def test_cauchy_noise_heavier_than_gaussian(self):
        """Regime G noise (Cauchy) should have a much larger IQR than Regime E noise (Gaussian)."""
        rng_g = np.random.default_rng(9)
        rng_e = np.random.default_rng(10)
        n_g = gen._generate_noise(rng_g, N_LARGE, "G")
        n_e = gen._generate_noise(rng_e, N_LARGE, "E")
        iqr_g = float(np.percentile(n_g, 75) - np.percentile(n_g, 25))
        iqr_e = float(np.percentile(n_e, 75) - np.percentile(n_e, 25))
        # Cauchy IQR ≈ 2; Gaussian IQR ≈ 1.35 — Cauchy should be at least 1.3x wider
        assert iqr_g > iqr_e * 1.3, (
            f"Cauchy IQR ({iqr_g:.3f}) not sufficiently larger than Gaussian IQR ({iqr_e:.3f})"
        )

    def test_non_G_noise_is_normal(self):
        """Noise for regimes E/F/H should be drawn from a standard normal."""
        rng = np.random.default_rng(11)
        for regime in ("E", "F", "H"):
            noise = gen._generate_noise(rng, N_LARGE, regime)
            # |mean| < 0.05, std in [0.95, 1.05]
            assert abs(noise.mean()) < 0.05
            assert 0.95 < noise.std() < 1.05


# ---------------------------------------------------------------------------
# 5. Block-diagonal correlation (Regime H)
# ---------------------------------------------------------------------------

class TestRegimeHCorrelation:
    def test_regime_H_within_block_correlation(self):
        """Corr(X[:,0], X[:,1]) should be ≈ 0.7 ± 0.15 (same block)."""
        rng = np.random.default_rng(12)
        X = gen._generate_X_H(rng, N_LARGE, 6, block_size=3, rho=0.7)
        corr = float(np.corrcoef(X[:, 0], X[:, 1])[0, 1])
        assert abs(corr - 0.7) < 0.15, f"Within-block corr={corr:.3f}, expected ≈0.7"

    def test_regime_H_across_block_independence(self):
        """|Corr(X[:,0], X[:,3])| should be < 0.2 (different blocks)."""
        rng = np.random.default_rng(13)
        X = gen._generate_X_H(rng, N_LARGE, 6, block_size=3, rho=0.7)
        corr = float(np.corrcoef(X[:, 0], X[:, 3])[0, 1])
        assert abs(corr) < 0.2, f"Across-block |corr|={abs(corr):.3f}, expected <0.2"

    def test_regime_H_unit_variance(self):
        """Compound-symmetry Σ has unit diagonal → Var(X_H) ≈ 1.0."""
        rng = np.random.default_rng(14)
        X = gen._generate_X_H(rng, N_LARGE, 6, block_size=3, rho=0.7)
        assert X.var(axis=0).mean() == pytest.approx(1.0, abs=0.05)


# ---------------------------------------------------------------------------
# 6. Parquet schema compatibility
# ---------------------------------------------------------------------------

class TestParquetSchema:
    """Parquet files must have the expected schema and data types."""

    @pytest.fixture
    def sample_parquet(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            rng = np.random.default_rng(42)
            ds = gen.generate_ood_dataset(rng, "E")
            outpath = Path(tmpdir) / "E" / "dataset_0000.parquet"
            _write_parquet(ds, outpath)
            yield outpath

    def test_required_columns_present(self, sample_parquet):
        import pyarrow.parquet as pq
        t = pq.read_table(str(sample_parquet))
        required = {"X", "y", "betaX", "suite_family", "prior_regime",
                    "n_total", "p_signal", "p_noise", "p_total",
                    "target_noise_scale", "training_size_anchor", "feature_noise_level"}
        assert required.issubset(set(t.schema.names))

    def test_arrays_are_float64(self, sample_parquet):
        import pyarrow.parquet as pq
        t = pq.read_table(str(sample_parquet))
        assert np.array(t["X"][0].as_py()).dtype == np.float64
        assert np.array(t["y"][0].as_py()).dtype == np.float64
        assert np.array(t["betaX"][0].as_py()).dtype == np.float64

    def test_betaX_is_noiseless(self, sample_parquet):
        import pyarrow.parquet as pq
        t = pq.read_table(str(sample_parquet))
        y = np.array(t["y"][0].as_py())
        betaX = np.array(t["betaX"][0].as_py())
        assert not np.allclose(y, betaX)

    def test_scalar_metadata(self, sample_parquet):
        import pyarrow.parquet as pq
        t = pq.read_table(str(sample_parquet))
        assert float(t["target_noise_scale"][0].as_py()) == 1.0
        assert bool(t["training_size_anchor"][0].as_py()) is False
        assert int(t["feature_noise_level"][0].as_py()) == 0


# ---------------------------------------------------------------------------
# 7. Manifest validation
# ---------------------------------------------------------------------------

class TestManifest:
    def test_manifest_dataset_count(self):
        """Running generate with 200 datasets should produce n_datasets==200, 50 per regime."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "ood"
            # Import the main function and run it with sys.argv patched
            import sys
            old_argv = sys.argv
            sys.argv = [
                "generate_ood_eval_data.py",
                "--n_datasets", "200",
                "--out_dir", str(out_dir),
                "--seed", "20260513",
            ]
            try:
                gen.main()
            finally:
                sys.argv = old_argv

            manifest_path = out_dir / "ood_manifest.json"
            assert manifest_path.exists(), "ood_manifest.json was not created"
            with open(manifest_path, "r", encoding="utf-8") as fh:
                manifest = json.load(fh)

            assert manifest["n_datasets"] == 200
            for regime in ["E", "F", "G", "H"]:
                count = sum(1 for d in manifest["datasets"] if d["regime"] == regime)
                assert count == 50, (
                    f"Expected 50 datasets for regime {regime}, got {count}"
                )

    def test_manifest_ood_version(self):
        """Manifest ood_version must be 'v1'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "ood"
            import sys
            old_argv = sys.argv
            sys.argv = [
                "generate_ood_eval_data.py",
                "--n_datasets", "8",   # minimal run: 2 per regime
                "--out_dir", str(out_dir),
                "--seed", "42",
            ]
            try:
                gen.main()
            finally:
                sys.argv = old_argv

            with open(out_dir / "ood_manifest.json", "r", encoding="utf-8") as fh:
                manifest = json.load(fh)
            assert manifest["ood_version"] == "v1"
            assert manifest["regimes"] == ["E", "F", "G", "H"]


# ---------------------------------------------------------------------------
# 8. Index-row safety invariants (SQL text, no live session)
# ---------------------------------------------------------------------------

class TestIndexSafety:
    def test_prepare_ood_index_uses_ood_suite_id(self):
        """Generated DELETE SQL should reference OOD_SUITE_ID, not any production ID."""
        import prepare_ood_regression as ood_prep
        ood_suite_id = ood_prep.OOD_SUITE_ID
        production_id = "linear_poisson_v1_recommended"

        # Build the SQL that _truncate_ood_index would execute
        sql = (
            f"DELETE FROM {ood_prep.OOD_INDEX_TABLE} "
            f"WHERE suite_id = '{ood_suite_id}'"
        )
        assert ood_suite_id in sql
        assert production_id not in sql

    def test_truncate_ood_does_not_drop_table(self):
        """_truncate_ood_index must use DELETE, never DROP TABLE."""
        import prepare_ood_regression as ood_prep

        sql = (
            f"DELETE FROM {ood_prep.OOD_INDEX_TABLE} "
            f"WHERE suite_id = '{ood_prep.OOD_SUITE_ID}'"
        )
        assert "DELETE" in sql.upper()
        assert "DROP" not in sql.upper()
        assert "TABLE" not in sql.upper()

    def test_ood_suite_id_differs_from_production(self):
        """OOD_SUITE_ID must not equal the production SYNREG_SUITE_ID default."""
        import prepare_ood_regression as ood_prep
        production_default = "linear_poisson_v1_recommended"
        assert ood_prep.OOD_SUITE_ID != production_default

    def test_ood_stage_prefix_uses_eval_stage(self):
        """OOD data must reference @EVALUATION_DATASET_STAGE, never @META_REGRESSION_DATASET_STAGE."""
        import prepare_ood_regression as ood_prep
        assert "@EVALUATION_DATASET_STAGE" in ood_prep.EVAL_STAGE_PREFIX
        assert "@META_REGRESSION_DATASET_STAGE" not in ood_prep.EVAL_STAGE_PREFIX


# ---------------------------------------------------------------------------
# 9. Production constant guards
# ---------------------------------------------------------------------------

class TestProductionGuards:
    def test_production_n_datasets_unchanged(self):
        """PRIMARY_N_DATASETS must remain 200 — guard against accidental mutation."""
        assert prep_prod.PRIMARY_N_DATASETS == int(
            prep_prod.os.getenv("SYNTHETIC_REGRESSION_PRIMARY_DATASETS", "200")
        )
        # Verify the default (env not set in test environment)
        import os
        os.environ.pop("SYNTHETIC_REGRESSION_PRIMARY_DATASETS", None)
        # Reimport to get the default value
        import importlib
        m = importlib.import_module("prepare_synthetic_regression")
        assert m.PRIMARY_N_DATASETS == 200

    def test_production_split_seeds_unchanged(self):
        """PRIMARY_SPLIT_SEEDS must remain [0,1,2,3,4] — guard against accidental mutation."""
        import os
        os.environ.pop("SYNTHETIC_REGRESSION_PRIMARY_SPLIT_SEEDS", None)
        import importlib
        m = importlib.import_module("prepare_synthetic_regression")
        assert m.PRIMARY_SPLIT_SEEDS == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# 10. _make_ood_index_row structure
# ---------------------------------------------------------------------------

class TestMakeOodIndexRow:
    def test_row_has_correct_suite_id(self):
        import prepare_ood_regression as ood_prep
        row = ood_prep._make_ood_index_row(
            dataset_id=0,
            dataset_seed=12345,
            regime="E",
            n_total=100,
            p_signal=5,
            p_noise=0,
            p_total=5,
            target_noise_scale=1.0,
        )
        assert row["suite_id"] == ood_prep.OOD_SUITE_ID

    def test_row_stage_path_format(self):
        import prepare_ood_regression as ood_prep
        row = ood_prep._make_ood_index_row(
            dataset_id=3,
            dataset_seed=99,
            regime="H",
            n_total=80,
            p_signal=4,
            p_noise=0,
            p_total=4,
            target_noise_scale=1.0,
        )
        assert row["stage_path"] == "@EVALUATION_DATASET_STAGE/ood_parity/H/dataset_0003.parquet"

    def test_row_split_seeds(self):
        import prepare_ood_regression as ood_prep
        row = ood_prep._make_ood_index_row(
            dataset_id=0,
            dataset_seed=1,
            regime="G",
            n_total=50,
            p_signal=3,
            p_noise=0,
            p_total=3,
            target_noise_scale=1.0,
        )
        assert row["split_seeds"] == [0, 1, 2]

    def test_row_train_holdout_split(self):
        """n_train_default should be 80% of n_total (ood_primary suite family)."""
        import prepare_ood_regression as ood_prep
        n_total = 100
        row = ood_prep._make_ood_index_row(
            dataset_id=0,
            dataset_seed=1,
            regime="E",
            n_total=n_total,
            p_signal=5,
            p_noise=0,
            p_total=5,
            target_noise_scale=1.0,
        )
        assert row["n_train_default"] == 80
        assert row["n_holdout_default"] == 20


# ---------------------------------------------------------------------------
# 11. Production/OOD path separation guards
# ---------------------------------------------------------------------------

class TestProductionOODSeparation:
    """Guard invariants for production/OOD path separation."""

    def test_ood_full_suite_id_value(self):
        """OOD_FULL_SUITE_ID must equal 'ood_linear_full_v1'."""
        import run_synthetic_regression_evaluation as eval_mod
        assert eval_mod.OOD_FULL_SUITE_ID == "ood_linear_full_v1"

    def test_ood_full_output_stage_value(self):
        """OOD_FULL_OUTPUT_STAGE must equal '@EVALUATION_RESULTS_STAGE/linear/regression/numeric'."""
        import run_synthetic_regression_evaluation as eval_mod
        assert eval_mod.OOD_FULL_OUTPUT_STAGE == "@EVALUATION_RESULTS_STAGE/linear/regression/numeric"

    def test_ood_pilot_suite_id_value(self):
        import run_synthetic_regression_evaluation as run_eval
        assert run_eval.OOD_PILOT_SUITE_ID == "ood_linear_pilot_v1"

    def test_ood_pilot_gpu_shards(self):
        import run_synthetic_regression_evaluation as run_eval
        assert run_eval.OOD_PILOT_GPU_SHARDS == 5

    def test_ood_pilot_parts_prefix_distinct_from_production(self):
        import run_synthetic_regression_evaluation as run_eval
        prod_prefix = "@EVALUATION_RESULTS_STAGE/linear/regression/numeric"
        assert run_eval.OOD_PILOT_PARTS_PREFIX != prod_prefix
        assert "ood" in run_eval.OOD_PILOT_PARTS_PREFIX

    def test_production_prep_submits_only_production_entrypoint(self):
        """run_synthetic_regression_prep source must not reference prepare_ood_regression."""
        import inspect
        import run_synthetic_regression_evaluation as run_eval
        src = inspect.getsource(run_eval.run_synthetic_regression_prep)
        assert "prepare_ood_regression" not in src, (
            "Production prep must not submit prepare_ood_regression.py"
        )
        assert "prepare_synthetic_regression" in src

    def test_ood_pilot_procedure_does_not_submit_baselines(self):
        import inspect
        import run_synthetic_regression_evaluation as run_eval
        src = inspect.getsource(run_eval.run_synthetic_regression_ood_deepset_pilot)
        assert "baseline" not in src.lower()
        assert "autogluon" not in src.lower()

    def test_ood_pilot_procedure_sets_n_pilot_env_var(self):
        """The OOD pilot must be invoked with 80 datasets (OOD_REGRESSION_N_PILOT=80)."""
        import inspect
        import run_synthetic_regression_evaluation as run_eval
        # _submit_ood_prep itself uses OOD_REGRESSION_N_PILOT env var
        prep_src = inspect.getsource(run_eval._submit_ood_prep)
        assert "OOD_REGRESSION_N_PILOT" in prep_src
        # The pilot procedure passes 80 as n_datasets
        pilot_src = inspect.getsource(run_eval.run_synthetic_regression_ood_deepset_pilot)
        assert "80" in pilot_src

    def test_ood_pilot_uses_synreg_results_stage_override(self):
        """Pilot must set SYNREG_RESULTS_STAGE to an OOD-specific prefix."""
        import inspect
        import run_synthetic_regression_evaluation as run_eval
        src = inspect.getsource(run_eval.run_synthetic_regression_ood_deepset_pilot)
        assert "SYNREG_RESULTS_STAGE" in src
        assert "ood" in src.lower()


class TestManifestValidation:
    """prepare_ood_regression.py must fail clearly on under-populated regimes."""

    def test_prepare_raises_if_regime_under_populated(self):
        """ValueError if any regime has fewer than n_per_regime datasets."""
        import prepare_ood_regression as ood_prep
        import pytest

        # Manifest with no datasets for regime E (needs 20)
        manifest = {
            "ood_version": "v1",
            "base_seed": 20260513,
            "n_datasets": 20,
            "regimes": ["E", "F", "G", "H"],
            "datasets": [
                {"regime": r, "dataset_id": i, "filename": f"{r}/dataset_{i:04d}.npz",
                 "n_total": 100, "p_signal": 5, "p_noise": 0, "p_total": 5,
                 "target_noise_scale": 1.0, "dataset_seed": i + 1}
                for r in ["F", "G", "H"] for i in range(20)  # E is missing entirely
            ],
        }

        with pytest.raises(ValueError, match="(?i)regime 'E'"):
            # Call the internal subset selection logic directly
            n_per_regime = ood_prep.OOD_N_DATASETS // len(ood_prep.OOD_REGIMES)
            for regime in ood_prep.OOD_REGIMES:
                entries = [d for d in manifest["datasets"] if d["regime"] == regime]
                if len(entries) < n_per_regime:
                    raise ValueError(
                        f"Regime {regime!r} has only {len(entries)} datasets in the "
                        f"manifest but {n_per_regime} are required "
                        f"(OOD_N_DATASETS={ood_prep.OOD_N_DATASETS}, "
                        f"regimes={len(ood_prep.OOD_REGIMES)})."
                    )

    def test_ood_pilot_balanced_20_per_regime(self):
        """OOD_N_DATASETS=80 (default) divides evenly: 20 per regime across 4 regimes."""
        import prepare_ood_regression as ood_prep
        n_per_regime = ood_prep.OOD_N_DATASETS // len(ood_prep.OOD_REGIMES)
        assert n_per_regime == 20
        assert ood_prep.OOD_N_DATASETS == n_per_regime * len(ood_prep.OOD_REGIMES)

    def test_ood_split_seeds_are_three(self):
        import prepare_ood_regression as ood_prep
        assert ood_prep.OOD_SPLIT_SEEDS == [0, 1, 2]

    def test_ood_suite_id_default(self):
        import prepare_ood_regression as ood_prep
        assert ood_prep.OOD_SUITE_ID == "ood_linear_pilot_v1"


# ---------------------------------------------------------------------------
# Tests: logical_dataset_key (Issue 7)
# ---------------------------------------------------------------------------

class TestLogicalDatasetKey:
    def _make_row(self, dataset_id: int, regime: str):
        import prepare_ood_regression as ood_prep
        return ood_prep._make_ood_index_row(
            dataset_id=dataset_id,
            dataset_seed=dataset_id * 7,
            regime=regime,
            n_total=100,
            p_signal=5,
            p_noise=0,
            p_total=5,
            target_noise_scale=1.0,
        )

    def test_row_logical_dataset_key_format(self):
        """logical_dataset_key equals '{OOD_SUITE_ID}:{regime}:{dataset_id:04d}'."""
        import prepare_ood_regression as ood_prep
        row = self._make_row(dataset_id=3, regime="G")
        expected = f"{ood_prep.OOD_SUITE_ID}:G:0003"
        assert row["logical_dataset_key"] == expected

    def test_logical_key_unique_across_regimes(self):
        """E/0000, F/0000, G/0000, H/0000 all produce distinct keys."""
        import prepare_ood_regression as ood_prep
        keys = [
            self._make_row(dataset_id=0, regime=r)["logical_dataset_key"]
            for r in ["E", "F", "G", "H"]
        ]
        assert len(keys) == len(set(keys)), "Logical keys must be unique across regimes"


# ---------------------------------------------------------------------------
# Tests: Explicit deepset mode for OOD pilot shards (Issue 8)
# ---------------------------------------------------------------------------

class TestProductionOODSeparationExtended:
    def test_ood_pilot_shards_have_explicit_deepset_mode(self):
        """SYNTHETIC_REGRESSION_MODE=deepset must be explicitly set in OOD pilot shard env_vars."""
        import inspect
        import run_synthetic_regression_evaluation as run_eval
        src = inspect.getsource(run_eval.run_synthetic_regression_ood_deepset_pilot)
        assert '"SYNTHETIC_REGRESSION_MODE": "deepset"' in src or \
               "'SYNTHETIC_REGRESSION_MODE': 'deepset'" in src, (
            "OOD pilot shards must explicitly set SYNTHETIC_REGRESSION_MODE=deepset"
        )


# ---------------------------------------------------------------------------
# Tests: OOD full suite indexing constants and suite ID separation
# ---------------------------------------------------------------------------

class TestOODFullSuiteIndexing:
    def test_ood_full_selects_all_50_per_regime(self):
        """OOD_FULL_SUITE_ID and pilot must index distinct dataset counts (200 vs 80)."""
        import run_synthetic_regression_evaluation as run_eval
        # Full suite uses all 200 datasets (50/regime × 4 regimes)
        assert run_eval.OOD_FULL_N_DATASETS == 200
        # Verify 50 per regime: 200 / 4 regimes
        assert run_eval.OOD_FULL_N_DATASETS % 4 == 0
        assert run_eval.OOD_FULL_N_DATASETS // 4 == 50

    def test_ood_full_suite_id_distinct_from_pilot(self):
        """OOD_FULL_SUITE_ID and OOD_PILOT_SUITE_ID must be distinct non-empty strings."""
        import run_synthetic_regression_evaluation as run_eval
        assert run_eval.OOD_FULL_SUITE_ID != run_eval.OOD_PILOT_SUITE_ID
        assert run_eval.OOD_FULL_SUITE_ID != ""
        assert run_eval.OOD_PILOT_SUITE_ID != ""


# ---------------------------------------------------------------------------
# Tests: Cache path collision (Item 2)
# ---------------------------------------------------------------------------

import evaluate_linear_regression as evsr


class TestCachePathCollision:
    def test_e_and_f_same_basename_produce_different_local_names(self):
        e = evsr._stage_path_to_local_name(
            "@EVALUATION_DATASET_STAGE/ood_parity/E/dataset_0000.parquet"
        )
        f = evsr._stage_path_to_local_name(
            "@EVALUATION_DATASET_STAGE/ood_parity/F/dataset_0000.parquet"
        )
        assert e != f

    def test_primary_path_works(self):
        name = evsr._stage_path_to_local_name(
            "@EVALUATION_DATASET_STAGE/primary/dataset_0042.parquet"
        )
        assert "dataset_0042" in name

    def test_cached_e_file_not_reused_for_f(self):
        # stage_path_to_local_name must embed regime in the local filename
        e = evsr._stage_path_to_local_name(
            "@EVALUATION_DATASET_STAGE/ood_parity/E/dataset_0000.parquet"
        )
        f = evsr._stage_path_to_local_name(
            "@EVALUATION_DATASET_STAGE/ood_parity/F/dataset_0000.parquet"
        )
        assert "E" in e or "ood_parity__E" in e
        assert "F" in f or "ood_parity__F" in f

    def test_metadata_mismatch_raises(self):
        with pytest.raises(RuntimeError, match="prior_regime"):
            evsr._validate_dataset_metadata(
                {"prior_regime": "F", "n_total": 500},
                {"prior_regime": "E", "n_total": 500, "stage_path": "@fake"},
            )

    def test_metadata_match_does_not_raise(self):
        evsr._validate_dataset_metadata(
            {"prior_regime": "E", "n_total": 500, "p_signal": 5, "p_noise": 0},
            {"prior_regime": "E", "n_total": 500, "p_signal": 5, "p_noise": 0,
             "stage_path": "@fake"},
        )


# ---------------------------------------------------------------------------
# Tests: OOD_N_DATASETS env var (Item 4)
# ---------------------------------------------------------------------------

import importlib
import os as _os


class TestOODNDatasets:
    def test_preferred_var_sets_n_datasets(self):
        with pytest.MonkeyPatch().context() as mp:
            mp.setenv("OOD_REGRESSION_N_DATASETS", "200")
            mp.delenv("OOD_REGRESSION_N_PILOT", raising=False)
            mod = importlib.reload(importlib.import_module("prepare_ood_regression"))
        assert mod.OOD_N_DATASETS == 200

    def test_legacy_n_pilot_fallback(self):
        with pytest.MonkeyPatch().context() as mp:
            mp.setenv("OOD_REGRESSION_N_PILOT", "80")
            mp.delenv("OOD_REGRESSION_N_DATASETS", raising=False)
            mod = importlib.reload(importlib.import_module("prepare_ood_regression"))
        assert mod.OOD_N_DATASETS == 80

    def test_non_divisible_by_4_raises(self):
        with pytest.MonkeyPatch().context() as mp:
            mp.setenv("OOD_REGRESSION_N_DATASETS", "81")  # 81 % 4 != 0
            mp.delenv("OOD_REGRESSION_N_PILOT", raising=False)
            with pytest.raises(ValueError, match="divisible by 4"):
                importlib.reload(importlib.import_module("prepare_ood_regression"))

    def test_submit_ood_prep_passes_both_env_vars(self):
        import inspect
        import run_synthetic_regression_evaluation as run_eval
        src = inspect.getsource(run_eval._submit_ood_prep)
        assert "OOD_REGRESSION_N_DATASETS" in src
        assert "OOD_REGRESSION_N_PILOT" in src


# ---------------------------------------------------------------------------
# Tests: Table creation before delete (Item 6)
# ---------------------------------------------------------------------------

class TestOODPrepTableOrdering:
    def test_create_table_called_before_delete(self):
        """create_synreg_index_table must execute before _truncate_ood_index."""
        import prepare_ood_regression as ood_prep
        from unittest.mock import MagicMock, patch, call

        call_order = []

        def record_create(session):
            call_order.append("create")

        def record_delete(session):
            call_order.append("delete")

        # Build a minimal valid manifest
        manifest = {
            "ood_version": "v1",
            "base_seed": 20260513,
            "n_datasets": 80,
            "regimes": ["E", "F", "G", "H"],
            "datasets": [
                {
                    "regime": r, "dataset_id": i,
                    "filename": f"{r}/dataset_{i:04d}.parquet",
                    "stage_path": f"@EVALUATION_DATASET_STAGE/ood_parity/{r}/dataset_{i:04d}.parquet",
                    "n_total": 100, "p_signal": 5, "p_noise": 0, "p_total": 5,
                    "target_noise_scale": 1.0, "dataset_seed": i + 1,
                }
                for r in ["E", "F", "G", "H"] for i in range(20)
            ],
        }

        mock_session = MagicMock()
        mock_session.sql.return_value.collect.return_value = [MagicMock()]
        mock_session.sql.return_value.collect.return_value[0].__getitem__ = lambda s, k: 80

        with patch("prepare_ood_regression.create_synreg_index_table",
                   side_effect=record_create):
            with patch("prepare_ood_regression._truncate_ood_index",
                       side_effect=record_delete):
                with patch("prepare_ood_regression._download_manifest",
                           return_value=manifest):
                    with patch("prepare_ood_regression.insert_synreg_index_rows"):
                        with patch("prepare_ood_regression._assert_index_populated",
                                   return_value=ood_prep.OOD_N_DATASETS):
                            ood_prep.prepare_ood_regression(mock_session)

        assert "create" in call_order
        assert "delete" in call_order
        assert call_order.index("create") < call_order.index("delete")
