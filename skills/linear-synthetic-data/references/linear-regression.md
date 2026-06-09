# Linear Regression Data

## Contents

- Core DGP
- Profiles and regimes
- Coefficients, features, dimensions, and noise
- Generation and Parquet contract
- DeepSet semantics
- Validation and guardrails

## Core DGP

Treat the implementation in `src/generate_dgp.py` and `src/dgp_helpers.py` as
the source of truth for current DeepSet training data.

Model every task as:

```text
X_clean ~ feature distribution
betaX = X_clean @ beta
y = betaX + epsilon
X_observed = X_clean + feature_noise, when enabled
```

The target-generating function remains linear in the clean features. Gaussian,
heavy-tailed, sparse, correlated, noisy, or high-dimensional conditions do not
make the target nonlinear.

Create each Parquet file as one task:

- Use the first 80 percent of rows as context.
- Use the remaining 20 percent as query.
- Store observed features in `X_train` and `X_test`.
- Store noisy responses in `y_train` and `y_test`.
- Store noiseless targets in `betaX_train` and `betaX_test`.
- Store optional coefficient or teacher arrays only when enabled.

Use `betaX_test` when the active objective requires clean function recovery.
Follow the model and training configuration when selecting ground-truth
coefficients, OLS teachers, or ridge teachers.

Keep `scripts/generate_synthetic_regression.py` scoped to evaluation data. Do
not use it as the authority for DeepSet training-task composition.

## Profiles and Regimes

Choose from the implemented profiles:

- `legacy`: reproduce the original A-D mixture.
- `linear_stat_aware`: use the broad default A-L training mixture.
- `linear_stress`: emphasize difficult statistical conditions.
- `market_linear`: emphasize signed market-style effects.

The default `linear_stat_aware` probabilities are:

| Regime | Weight | Capability |
| --- | ---: | --- |
| `A_iid_dense` | 0.15 | Baseline dense recovery |
| `B_iid_sparse` | 0.15 | Sparse signal recovery |
| `C_heavy_tail_noise` | 0.10 | Student-t residual robustness |
| `D_correlated_ar` | 0.10 | AR(1) multicollinearity |
| `E_high_dim_dense` | 0.05 | Dense larger feature spaces |
| `F_high_dim_sparse` | 0.05 | Sparse larger feature spaces |
| `G_noise_features` | 0.10 | Irrelevant-column resistance |
| `H_block_correlated` | 0.05 | Grouped multicollinearity |
| `I_equicorrelated` | 0.05 | Global correlation stress |
| `J_low_n_high_p` | 0.05 | Underdetermined sparse inference |
| `K_feature_noise` | 0.05 | Errors-in-variables robustness |
| `L_market_linear` | 0.10 | Structured coefficient signs |

Interpret weights as probabilities, not quotas. Remove
`J_low_n_high_p` and renormalize when `--allow_underdetermined` is absent.

Use these implemented regime definitions:

- `A_iid_dense`: independent Gaussian features, dense Gaussian coefficients,
  and Gaussian target noise.
- `B_iid_sparse`: independent Gaussian features, approximately 70 percent zero
  coefficients, and Gaussian target noise.
- `C_heavy_tail_noise`: independent Gaussian features, dense coefficients, and
  Student-t target noise with 3 degrees of freedom.
- `D_correlated_ar`: AR(1)-correlated columns, normally `rho=0.6`, with dense
  coefficients.
- `E_high_dim_dense`: at least 32 signal features, dense coefficients, and
  normally `n >= 2p`.
- `F_high_dim_sparse`: at least 32 signal features, support no larger than
  roughly `p/4`, and normally `n >= 2p`.
- `G_noise_features`: signal columns plus 8-120 irrelevant columns whose
  coefficients are zero.
- `H_block_correlated`: strongly correlated feature blocks with sampled block
  sizes such as 4, 8, or 16.
- `I_equicorrelated`: a common pairwise correlation, typically selected from
  0.1 through 0.9.
- `J_low_n_high_p`: more features than observations with sparse coefficients;
  opt in explicitly.
- `K_feature_noise`: compute `betaX` from `X_clean`, but expose measurement-noisy
  features.
- `L_market_linear`: use negative own-price effects, alternating cross-price
  signs, unconstrained attributes, and optional irrelevant features.

## Coefficients, Features, Dimensions, and Noise

Support dense, sparse, decaying, and market-sign coefficients. Set appended
noise-feature coefficients to exactly zero. Interpret `sparsity_ratio` as the
fraction of zero signal coefficients.

Support independent Gaussian, AR(1), block-correlated, and equicorrelated
Gaussian designs. Do not describe correlated Gaussian features as independent.

Use the current default grids:

```text
n_grid:                 32, 64, 128, 256, 512, 1024
p_signal_grid:          4, 8, 16, 32, 64
p_noise_grid:           0, 8, 24, 56, 120
active_s_grid:          2, 4, 8, 16, 32
rho_grid:               0.0, 0.3, 0.6, 0.9
target_noise_grid:      0.25, 0.5, 1.0, 2.0
feature_noise_grid:     0.0, 0.05, 0.10, 0.25
```

Keep these concepts distinct:

- `p_signal`: columns that may have nonzero coefficients.
- `p_noise`: appended irrelevant columns with zero coefficients.
- `p_total`: `p_signal + p_noise`.
- `target_noise_scale`: residual response noise.
- `feature_noise_level`: measurement error applied to clean features.

Use `feature_noise_level_float` for exact measurement-noise analysis. The
legacy integer field may be rounded.

## Generation and Parquet Contract

Generate the broad default suite deterministically:

```powershell
python src/generate_dgp.py `
  --out_dir data `
  --n_datasets 1000 `
  --profile linear_stat_aware `
  --base_seed 42
```

Expect 800 training, 100 validation, and 100 test task files from 1,000 tasks.
Use `--store_beta` for exact coefficients and `--store_teacher_preds` for
serialized teacher predictions.

Require the implemented training contract:

```text
X_train, y_train, X_test, y_test
betaX_train, betaX_test
n, p, n_train, n_test
p_signal, p_noise, p_total
prior_regime
active_s, sparsity_ratio
covariance_type, rho
target_noise_scale
feature_noise_level, feature_noise_level_float
teacher_available, oracle_ridge_lambda
condition_number, matrix_rank, effective_rank
snr
```

Treat coefficient, support, and full teacher arrays as optional extensions.

The current writer does not persist `profile`, `base_seed`,
`target_noise_type`, `has_noise_features`, `has_feature_noise`, `p_over_n`, or
`n_over_p`. Record command provenance and checksums in a suite manifest. Do not
interpret stored `snr` as a conventional ratio; the current value is the
variance of the linear signal.

## DeepSet Semantics

Treat each file as one exchangeable context/query episode:

1. Encode context rows from `X_train` and `y_train`.
2. Aggregate context rows as an order-invariant set.
3. Condition predictions on query rows from `X_test`.
4. Compare predictions with the configured clean or noisy target.
5. Never mix rows from separate task files into one context.

Fit normalization, imputation, and feature preprocessing on context rows only.
Keep outer train/validation/test membership at the task-file level.

## Validation and Guardrails

Validate every generated suite:

1. Check required columns and declared dimensions.
2. Reject non-finite features, targets, coefficients, teachers, or diagnostics.
3. Verify `p_total = p_signal + p_noise`.
4. Verify irrelevant-feature coefficients are zero.
5. Verify `betaX = X_clean @ beta` when latent arrays permit the check.
6. Verify sparse support and guard against accidentally sparse dense tasks.
7. Check covariance behavior against the declared structure.
8. Regenerate a small suite from the same command and seed.
9. Audit realized coverage by split, regime, dimensions, sparsity,
   correlation, target noise, and feature noise.
10. Inspect condition number, effective rank, `p/n`, and signal-strength tails.
11. Verify split counts and file membership against the manifest.
12. Smoke-test representative files through the actual regression loader.

Do not call these datasets nonlinear. Do not conflate irrelevant features,
feature measurement noise, and target noise. Do not enable underdetermined
tasks implicitly, infer exact counts from weights, fit preprocessing on query
rows, or overwrite a suite without recording provenance and coverage.

