# src/constants.py
"""Shared constants for the TabPFN DemandModel pipeline."""

# MODEL3 legacy limits (preserved for backward compatibility)
MODEL3_MAX_P       = 128
MODEL3_MAX_N_TRAIN = 256

# MODEL4 limits for all 12 regimes
# p_signal max=64 + p_noise max=120 → p_total up to 184; round up to 256
# n max=1024 → n_train up to ~819 at 80% split; use 1024
MODEL4_MAX_P       = 256
MODEL4_MAX_N_TRAIN = 1024

# HPO
HPO_BUCKETS      = 40
HPO_SPLIT_LIMITS = {"train": 200, "val": 40}

# All 12 regime names — must match dgp_helpers.py REGIME_WEIGHTS exactly
EXPECTED_REGIMES = frozenset({
    "A_iid_dense",
    "B_iid_sparse",
    "C_heavy_tail_noise",
    "D_correlated_ar",
    "E_high_dim_dense",
    "F_high_dim_sparse",
    "G_noise_features",
    "H_block_correlated",
    "I_equicorrelated",
    "J_low_n_high_p",
    "K_feature_noise",
    "L_market_linear",
})

# Suite IDs
SUITE_ID_ALL_REGIMES = "linear_stat_aware_all_regimes_v1"
ALL_SUITE_IDS = frozenset({
    "linear_poisson_v1_recommended",
    "ood_linear_full_v1",
    "linear_all_v1",
    SUITE_ID_ALL_REGIMES,
})

# Canonical stage-path template
# Usage: CANONICAL_STAGE_PATH_TEMPLATE.format(
#     stage_root="@EVALUATION_DATASET_STAGE",
#     suite_id=suite_id, family=family, filename=filename)
CANONICAL_STAGE_PATH_TEMPLATE = (
    "{stage_root}/synthetic_regression_prepared/{suite_id}/{family}/{filename}"
)

# Feature noise column names
FEATURE_NOISE_COL        = "feature_noise_level"        # canonical float
FEATURE_NOISE_LEGACY_COL = "feature_noise_level_float"  # dgp_helpers.py alias

# DeepSet runtime defaults (for memory estimation at generation time)
DEFAULT_CONTEXT_SIZE = 200
DEFAULT_CONTEXT_ENSEMBLES = 5
DEFAULT_TEST_BATCH_SIZE = 128
DEFAULT_FEATURE_CAP = 128          # equals MODEL3_MAX_P; do not remove MODEL3_MAX_P
DEFAULT_HIDDEN_ESTIMATE = 64       # conservative hidden-channel proxy
DEFAULT_GPU_GUARD_BYTES = 268_435_456  # 256 MB

# Controlled suite design
ALLOCATION_MODES = frozenset({"weighted", "weighted_quota", "balanced", "balanced_cartesian", "curriculum"})
CURRICULUM_POLICIES = frozenset({"none", "core_first", "balanced_tiers", "stress_eval_only"})
DIFFICULTY_TIERS = frozenset({"core", "robust", "stress"})
MEMORY_RISK_BUCKETS = frozenset({"low", "medium", "high", "exceeds_default_guard"})
DEFAULT_DIFFICULTY_MIX: dict[str, float] = {"core": 0.65, "robust": 0.30, "stress": 0.05}
MAX_MEMORY_RISK_CHOICES: tuple[str, ...] = ("low", "medium", "high", "exceeds_default_guard")

# Canonical classification stage-path template (separate from regression)
CANONICAL_CLASSIFICATION_STAGE_PATH_TEMPLATE = (
    "{stage_root}/synthetic_classification_prepared/{suite_id}/{family}/{filename}"
)

# Coverage axes
REGRESSION_COVERAGE_AXES: list[str] = [
    "prior_regime", "suite_family", "sample_complexity_bucket",
    "difficulty_tier", "memory_risk_bucket", "covariance_type",
]
CLASSIFICATION_COVERAGE_AXES: list[str] = [
    "classification_regime", "suite_family", "num_classes", "class_imbalance_type",
    "margin_level", "sample_complexity_bucket", "difficulty_tier", "memory_risk_bucket",
]

# Expected classification regimes (mirrors EXPECTED_REGIMES for regression)
EXPECTED_CLASSIFICATION_REGIMES = frozenset({
    "A_iid_dense_logistic", "B_iid_sparse_logistic", "C_label_noise_margin",
    "D_correlated_ar_logistic", "E_high_dim_dense_softmax", "F_high_dim_sparse_softmax",
    "G_noise_features_classification", "H_block_correlated_classification",
    "I_equicorrelated_classification", "J_low_n_high_p_classification",
    "K_feature_noise_classification", "L_market_sign_classification",
})

# ---------------------------------------------------------------------------
# Canonical linear regression family (replaces legacy "synthetic_regression_combined")
# ---------------------------------------------------------------------------

LINEAR_REGRESSION_TRAINING_FAMILY = "synthetic_linear_regression"

# ---------------------------------------------------------------------------
# Mixed-categorical training families
# ---------------------------------------------------------------------------

MIXED_CAT_REGRESSION_TRAINING_FAMILY    = "synthetic_linear_regression_mixed_categorical"
MIXED_CAT_CLASSIFICATION_TRAINING_FAMILY = "synthetic_linear_classification_mixed_categorical"

# Entity embedding special token IDs (must never equal a real category ID in DGP output)
ENTITY_EMBED_PAD_ID      = 0   # padding token — never a real category
ENTITY_EMBED_MISSING_ID  = 1   # category was missing (null) in the dataset
ENTITY_EMBED_UNKNOWN_ID  = 2   # query category not seen in context vocabulary
ENTITY_EMBED_FIRST_REAL_ID = 3   # DGP re-indexes real categories starting here

# Maximum category cardinality the model is designed for (DGP must not exceed this)
MIXED_CAT_MAX_CARDINALITY = 50           # real categories; vocab size = 50 + 3 special = 53
MIXED_CAT_VOCAB_SIZE = MIXED_CAT_MAX_CARDINALITY + ENTITY_EMBED_FIRST_REAL_ID   # 53

# Schema versions for mixed-categorical parquet
MIXED_REG_DGP_SCHEMA_VERSION  = "linear_regression_mixed_categorical_v1"
MIXED_CLS_DGP_SCHEMA_VERSION  = "linear_classification_mixed_categorical_v1"

# Checkpoint format versions for mixed families
CHECKPOINT_VERSION_MIXED_REGRESSION     = 6
CHECKPOINT_VERSION_MIXED_CLASSIFICATION = 7

# ---------------------------------------------------------------------------
# Nonlinear task family names (DGP task_family values — all 4 nonlinear families)
# ---------------------------------------------------------------------------

NONLINEAR_CLASSIFICATION_TRAINING_FAMILY        = "synthetic_nonlinear_classification"
NONLINEAR_MIXED_REGRESSION_TRAINING_FAMILY      = "synthetic_nonlinear_regression_mixed_categorical"
NONLINEAR_MIXED_CLASSIFICATION_TRAINING_FAMILY  = "synthetic_nonlinear_classification_mixed_categorical"

# Schema versions for nonlinear mixed-categorical training parquet
NONLINEAR_MIXED_REG_DGP_SCHEMA_VERSION  = "nonlinear_regression_mixed_categorical_v1"
NONLINEAR_MIXED_CLS_DGP_SCHEMA_VERSION  = "nonlinear_classification_mixed_categorical_v1"

# ---------------------------------------------------------------------------
# Training stages and index tables — canonical names used by builders,
# handlers, and task_routing.py.  Import these constants everywhere; never
# repeat the literal strings.
#
# Stage layout (both linear and nonlinear):
#   @<STAGE>/numeric/{train,val,test}/<tid>.parquet  (numeric-only families)
#   @<STAGE>/mixed/{train,val,test}/<tid>.parquet    (mixed-categorical families)
# ---------------------------------------------------------------------------

# ---- Linear regression training data ----
META_REGRESSION_DATASET_STAGE           = "@META_REGRESSION_DATASET_STAGE"
META_REGRESSION_DATASET_INDEX           = "META_REGRESSION_DATASET_INDEX"
META_MIXED_REGRESSION_DATASET_INDEX     = "META_MIXED_REGRESSION_DATASET_INDEX"
META_REGRESSION_DATASET_EXPECTED_TOTAL_ENV = "META_REGRESSION_DATASET_EXPECTED_TOTAL"
META_MIXED_REGRESSION_DATASET_EXPECTED_TOTAL_ENV = "META_MIXED_REGRESSION_DATASET_EXPECTED_TOTAL"

# ---- Linear classification training data ----
META_CLASSIFICATION_DATASET_STAGE       = "@META_CLASSIFICATION_DATASET_STAGE"
META_CLASSIFICATION_DATASET_INDEX       = "META_CLASSIFICATION_DATASET_INDEX"
META_MIXED_CATEGORICAL_DATASET_INDEX    = "META_MIXED_CATEGORICAL_DATASET_INDEX"
META_CLASSIFICATION_DATASET_EXPECTED_TOTAL_ENV = "META_CLASSIFICATION_DATASET_EXPECTED_TOTAL"
META_MIXED_CATEGORICAL_DATASET_EXPECTED_TOTAL_ENV = "META_MIXED_CATEGORICAL_DATASET_EXPECTED_TOTAL"

# ---- Nonlinear regression training data ----
META_NONLINEAR_REGRESSION_DATASET_STAGE          = "@META_NONLINEAR_REGRESSION_DATASET_STAGE"
META_NONLINEAR_REGRESSION_DATASET_INDEX          = "META_NONLINEAR_REGRESSION_DATASET_INDEX"
META_NONLINEAR_MIXED_REGRESSION_DATASET_INDEX    = "META_NONLINEAR_MIXED_REGRESSION_DATASET_INDEX"
META_NONLINEAR_REGRESSION_DATASET_EXPECTED_TOTAL_ENV = "META_NONLINEAR_REGRESSION_DATASET_EXPECTED_TOTAL"
META_NONLINEAR_MIXED_REGRESSION_DATASET_EXPECTED_TOTAL_ENV = "META_NONLINEAR_MIXED_REGRESSION_DATASET_EXPECTED_TOTAL"

# ---- Nonlinear classification training data ----
META_NONLINEAR_CLASSIFICATION_DATASET_STAGE        = "@META_NONLINEAR_CLASSIFICATION_DATASET_STAGE"
META_NONLINEAR_CLASSIFICATION_DATASET_INDEX        = "META_NONLINEAR_CLASSIFICATION_DATASET_INDEX"
META_NONLINEAR_MIXED_CATEGORICAL_DATASET_INDEX     = "META_NONLINEAR_MIXED_CATEGORICAL_DATASET_INDEX"
META_NONLINEAR_CLASSIFICATION_DATASET_EXPECTED_TOTAL_ENV = "META_NONLINEAR_CLASSIFICATION_DATASET_EXPECTED_TOTAL"
META_NONLINEAR_MIXED_CATEGORICAL_DATASET_EXPECTED_TOTAL_ENV = "META_NONLINEAR_MIXED_CATEGORICAL_DATASET_EXPECTED_TOTAL"

# ---------------------------------------------------------------------------
# Evaluation index tables — canonical names consumed by prep and evaluate
# scripts.  The prefix identifies the suite type (LINEAR / NONLINEAR).
# ---------------------------------------------------------------------------

# Linear evaluation index tables
LINEAR_REGRESSION_DATASET_INDEX           = "LINEAR_REGRESSION_DATASET_INDEX"
LINEAR_CLASSIFICATION_DATASET_INDEX       = "LINEAR_CLASSIFICATION_DATASET_INDEX"
LINEAR_MIXED_REGRESSION_DATASET_INDEX     = "LINEAR_MIXED_REGRESSION_DATASET_INDEX"
LINEAR_MIXED_CLASSIFICATION_DATASET_INDEX = "LINEAR_MIXED_CLASSIFICATION_DATASET_INDEX"

# Nonlinear evaluation index tables
NONLINEAR_REGRESSION_DATASET_INDEX           = "NONLINEAR_REGRESSION_DATASET_INDEX"
NONLINEAR_CLASSIFICATION_DATASET_INDEX       = "NONLINEAR_CLASSIFICATION_DATASET_INDEX"
NONLINEAR_MIXED_REGRESSION_DATASET_INDEX     = "NONLINEAR_MIXED_REGRESSION_DATASET_INDEX"
NONLINEAR_MIXED_CLASSIFICATION_DATASET_INDEX = "NONLINEAR_MIXED_CLASSIFICATION_DATASET_INDEX"
