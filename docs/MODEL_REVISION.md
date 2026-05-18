Plan: MarketAwareDeepSetModel — Fix Early Feature Compression & Query Collapse

 Context

 The current DeepSetModel pools the feature dimension at step 5 of the forward pass, before
 sample-level evidence is aggregated at step 8. This destroys feature identity before the model
 has learned "what does feature j tell us about labels across training samples," preventing it
 from computing anything analogous to Σ_i X_ij·y_i or Σ_i X_ij·X_ik. The result is
 query collapse: predictions barely vary across different x_test inputs because all
 query-differentiating information is pooled away too early.

 The fix is a new MarketAwareDeepSetModel class that reverses the pooling order: sample
 evidence is aggregated per feature first, then features interact. A modular RidgeExpert
 provides an explicit inductive bias without replacing the neural path. Legacy DeepSetModel
 is fully preserved.

 ---
 Four Subagent Design Conclusions (Required by Prompt)

 Principal ML Engineer: The explicit inductive bias must be modular. A differentiable ridge
 expert provides a useful prior for synthetic linear data, but the neural residual path must
 remain active and query-conditioned. The gate final = ridge + gate * neural_residual keeps
 ridge as the stable baseline while the network learns corrections. This pattern is extensible:
 future experts (sparse cross-price, low-rank market) plug in at the same level.

 Model Architecture Scientist: The fix is the reversal of pooling order. Reshape to
 (m*p, n, token_dim) first so that n samples are the sequence dimension within each
 (query, feature) slot. After pooling over n, the resulting (m, p, d_feature) tensor
 preserves feature identity for the cross-feature SAB at step 4.

 Evaluation Reliability Engineer: Pre-evaluation sanity checks must run on a fresh
 (random-weight) model for structural correctness, and separately on any trained checkpoint
 for query sensitivity. Permutation invariance and feature permutation consistency are
 pass/fail checks. Query sensitivity ratio is a diagnostic metric, not a hard gate for
 random-weight models.

 Market Mental Model Architect: The (m, p, d_feature) feature evidence tensor at step 4
 is the natural hook for future market extensions. Cross-price effects = cross-feature attention.
 Sparse substitution = masked feature SAB. The gate mechanism is already an expert-mixing
 interface. No architectural choices foreclose these extensions.

 ---
 Files Changed

 File: src/model.py
 Type: Modify
 Summary: Add 8 new ModelConfig fields; add RidgeExpert; add MarketAwareDeepSetModel; add _instantiate_model()
 ────────────────────────────────────────
 File: src/evaluate.py
 Type: Modify
 Summary: Import _instantiate_model; replace hardcoded DeepSetModel(cfg) in load_model()
 ────────────────────────────────────────
 File: src/train.py
 Type: Modify
 Summary: Add DEEPSET_MODEL_FAMILY constant; import/use _instantiate_model; add model_family to mismatch check; v3
   checkpoint for market_aware
 ────────────────────────────────────────
 File: src/hpo.py
 Type: Modify
 Summary: Add model_family to mismatch check fields; use _instantiate_model in Ray worker
 ────────────────────────────────────────
 File: src/sanity_checks.py
 Type: Create
 Summary: 6 structural/correctness checks + JSON/CSV output
 ────────────────────────────────────────
 File: tests/test_market_aware_deepset_shapes.py
 Type: Create
 Summary: Forward shape tests + model routing
 ────────────────────────────────────────
 File: tests/test_market_aware_deepset_permutation.py
 Type: Create
 Summary: Row + column permutation invariance
 ────────────────────────────────────────
 File: tests/test_ridge_expert.py
 Type: Create
 Summary: Ridge expert correctness, primal/dual consistency
 ────────────────────────────────────────
 File: tests/test_query_sanity_checks.py
 Type: Create
 Summary: Sanity script outputs JSON/CSV correctly
 ────────────────────────────────────────
 File: MODEL.md
 Type: Modify
 Summary: New §13: architecture, token design, memory guard, v3 format
 ────────────────────────────────────────
 File: MEMORY_TabPFN.md
 Type: Modify
 Summary: Add 2 guardrail sections: query collapse + HPO deferral

 ---
 Change 1 — src/model.py

 1a. ModelConfig extension

 Add 8 new fields after dropout, with defaults chosen for backward compatibility
 (all new fields have safe defaults that do not change DeepSetModel behavior):

 # New fields — MarketAwareDeepSetModel only
 model_family:             str   = "deepset"   # "deepset" | "market_aware"
 d_sample:                 int   = 64          # phi_sample output dim (must be divisible by n_heads when sample attn
 used)
 n_sab_sample_per_feature: int   = 0           # SAB layers over n within each (q,j) slot — keep 0 (memory: see §Memory
  Guard)
 sample_pool:              str   = "attn"      # pooling over n: "attn" | "pna" | "mean"
 use_ridge_expert:         bool  = False       # enable RidgeExpert + gate
 ridge_lambda:             float = 1.0         # ridge regularisation λ (must be > 0)
 residual_scale_init:      float = 0.1         # init value of learned residual_scale scalar
 gate_hidden_dim:          int   = 64          # gate_head MLP hidden dim

 Append to __post_init__:

 if self.model_family not in ("deepset", "market_aware"):
     raise ValueError(f"Invalid model_family: {self.model_family!r}")
 if self.sample_pool not in POOL_SCALE:
     raise ValueError(f"Invalid sample_pool: {self.sample_pool!r}")
 if self.model_family == "market_aware" and self.n_sab_sample_per_feature > 0:
     if self.d_sample % self.n_heads != 0:
         raise ValueError(
             f"d_sample={self.d_sample} must be divisible by n_heads={self.n_heads}"
         )

 1b. RidgeExpert class (no nn.Module; no parameters)

 Place immediately above MarketAwareDeepSetModel.

 class RidgeExpert:
     """Closed-form ridge regression. Stateless; no nn.Module parameters.
     Primal form (n >= p): β = (XᵀX + λI)⁻¹ Xᵀy
     Dual form  (n < p):   α = (XXᵀ + λI)⁻¹ y,  β = Xᵀα
     """
     def predict(self, X_norm, y_norm, x_test_norm, lam: float):
         # X_norm: (n,p), y_norm: (n,), x_test_norm: (m,p), lam: float → (m,)
         n, p = X_norm.shape
         kw = dict(device=X_norm.device, dtype=X_norm.dtype)
         if n >= p:
             A    = X_norm.T @ X_norm + lam * torch.eye(p, **kw)
             beta = torch.linalg.solve(A, X_norm.T @ y_norm)   # (p,)
         else:
             K    = X_norm @ X_norm.T + lam * torch.eye(n, **kw)
             alpha = torch.linalg.solve(K, y_norm)              # (n,)
             beta  = X_norm.T @ alpha                           # (p,)
         return x_test_norm @ beta                              # (m,)

 1c. MarketAwareDeepSetModel class

 Constructor modules

 # TOKEN_DIM = 6 (y, X_ij, xq_j, X_ij*xq_j, X_ij*y, |X_ij-xq_j|)
 self.phi_sample = build_mlp(6, cfg.d_sample, cfg.d_sample, cfg.dropout)

 # Optional SAB over n (n_sab_sample_per_feature > 0)
 if cfg.n_sab_sample_per_feature > 0:
     self.sab_sample = nn.Sequential(*[
         SAB(cfg.d_sample, cfg.n_heads, cfg.dropout)
         for _ in range(cfg.n_sab_sample_per_feature)])

 # Pool over n
 self.sample_pool_layer = SetPool(cfg.d_sample, cfg.sample_pool, cfg.n_heads, cfg.dropout)
 d_feat = POOL_SCALE[cfg.sample_pool] * cfg.d_sample      # stored as self._d_feat

 # Feature SAB over p
 if cfg.n_sab_feat > 0:
     self.sab_feat = nn.Sequential(*[
         SAB(d_feat, cfg.n_heads, cfg.dropout)
         for _ in range(cfg.n_sab_feat)])
 else:
     self.lambda_feat = nn.Parameter(torch.tensor(1.0))
     self.gamma_feat  = nn.Parameter(torch.tensor(0.1))

 # Prediction head
 self.beta_head        = build_mlp(d_feat, 1, d_feat // 2, cfg.dropout)
 self.residual_scale   = nn.Parameter(torch.tensor(cfg.residual_scale_init))
 self.feat_summary_pool= SetPool(d_feat, "pna", cfg.n_heads, cfg.dropout)
 self.residual_head    = build_mlp(POOL_SCALE["pna"] * d_feat, 1,
                                    POOL_SCALE["pna"] * d_feat // 2, cfg.dropout)
 self.gate_head        = build_mlp(d_feat, 1, cfg.gate_hidden_dim, cfg.dropout)

 # Ridge expert
 if cfg.use_ridge_expert:
     self._ridge = RidgeExpert()

 Forward pass (exact tensor shapes)

 Inputs: X_train (n,p), y_train (n,), x_test (p,) or (m,p)
 single = x_test.ndim == 1  →  unsqueeze to (1,p); remember to squeeze output

 ─── Step 1: Normalize (identical to DeepSetModel) ───
 X_norm: (n,p),  x_test_norm: (m,p),  y_norm: (n,)
 y_mean, y_std: scalars (if norm_target)

 ─── Step 2: Build 6-feature tokens ───
 y_e   = y_norm.view(1,1,n).expand(m,p,n)            # (m,p,n)
 Xi_e  = X_norm.T.unsqueeze(0).expand(m,p,n)         # X_norm.T is (p,n) → (m,p,n)
 xq_e  = x_test_norm.unsqueeze(2).expand(m,p,n)      # (m,p,n)
 tokens= torch.stack([y_e, Xi_e, xq_e,
                      Xi_e*xq_e, Xi_e*y_e,
                      (Xi_e-xq_e).abs()], dim=3)      # (m,p,n,6)
 flat  = tokens.view(m*p, n, 6)                       # (m*p, n, 6)

 ─── Step 3: Sample evidence per feature ───
 h  = phi_sample(flat)                                # (m*p, n, d_sample)
 [if n_sab_sample_per_feature > 0: h = sab_sample(h)]
 ev = sample_pool_layer(h)                            # (m*p, d_feat)
 ev = ev.view(m, p, d_feat)                           # (m, p, d_feat)  ← KEY: feature identity preserved

 ─── Step 4: Cross-feature interaction ───
 if n_sab_feat > 0:   ctx = sab_feat(ev)              # (m, p, d_feat)
 else:                ctx = λ*ev + γ*ev.mean(1,True)  # (m, p, d_feat)

 ─── Step 5: Neural prediction ───
 beta_like = beta_head(ctx).squeeze(-1)               # (m, p)
 lin_pred  = (beta_like * x_test_norm).sum(dim=1)     # (m,)
 summary   = feat_summary_pool(ctx)                   # (m, 4*d_feat)
 resid     = residual_head(summary).squeeze(-1)       # (m,)
 neural    = lin_pred + residual_scale * resid        # (m,)

 ─── Step 6: Ridge expert (optional) ───
 if use_ridge_expert:
     ridge = _ridge.predict(X_norm, y_norm, x_test_norm, ridge_lambda)  # (m,)

 ─── Step 7: Gate + combine ───
 ctx_mean = ctx.mean(dim=1)                           # (m, d_feat)
 gate     = sigmoid(gate_head(ctx_mean).squeeze(-1))  # (m,)
 if use_ridge_expert:
     pred_norm = ridge + gate * neural
 else:
     pred_norm = neural

 ─── Step 8: Denormalize (identical to DeepSetModel) ───
 y_hat = pred_norm * y_std + y_mean  (if norm_target)
 return y_hat.squeeze(0) if single else y_hat

 1d. _instantiate_model() module-level function

 Add at the bottom of model.py, after both model classes:

 def _instantiate_model(cfg: ModelConfig) -> torch.nn.Module:
     """Route to DeepSetModel or MarketAwareDeepSetModel based on cfg.model_family."""
     family = getattr(cfg, "model_family", "deepset")
     if family == "deepset":
         return DeepSetModel(cfg=cfg)
     if family == "market_aware":
         return MarketAwareDeepSetModel(cfg=cfg)
     raise ValueError(f"Unknown model_family: {family!r}")

 ---
 Change 2 — src/evaluate.py

 2a. Import

 # Before:
 from model import DeepSetModel, ModelConfig, ...
 # After:
 from model import DeepSetModel, MarketAwareDeepSetModel, ModelConfig, _instantiate_model, ...

 2b. load_model() — replace hardcoded class

 # Before:
 model = DeepSetModel(cfg=cfg)
 # After:
 model = _instantiate_model(cfg)

 No other changes to evaluate.py. model_config_from_payload() already uses
 dataclasses.fields(ModelConfig) dynamically, so new fields are automatically allowed.

 ---
 Change 3 — src/train.py

 3a. Import + constant

 from model import DeepSetModel, ModelConfig, _instantiate_model
 DEEPSET_MODEL_FAMILY = os.environ.get("DEEPSET_MODEL_FAMILY", "deepset")

 3b. _checkpoint_architecture_mismatches() — add model_family

 fields = (
     "d_phi", "d_rho", "pool", "n_heads",
     "n_sab_feat", "n_sab_samp",
     "norm_feat", "norm_target",
     "model_family",   # NEW
 )

 3c. train_fn() — model instantiation

 # Before:
 cfg   = ModelConfig(d_phi=d_phi, d_rho=d_rho, pool=pool, ...)
 model = DeepSetModel(cfg=cfg).to(device)

 # After:
 model_family = hyper_params.get("model_family", DEEPSET_MODEL_FAMILY)
 cfg   = ModelConfig(d_phi=d_phi, d_rho=d_rho, pool=pool, ...,
                     model_family=model_family)
 model = _instantiate_model(cfg).to(device)

 3d. Checkpoint save — version 3 for market_aware

 format_version = 3 if cfg.model_family == "market_aware" else 2
 torch.save({
     "checkpoint_format_version": format_version,
     "cfg": _dc.asdict(ckpt.cfg),
     "state_dict": ckpt.state_dict(),
     "metadata": {
         "source": "train.py",
         "checkpoint_name": os.path.basename(checkpoint_output_name),
         "pytorch_version": torch.__version__,
         "model_family": cfg.model_family,
     },
 }, checkpoint_output_name)

 ---
 Change 4 — src/hpo.py

 Two minimal changes only:

 4a. checkpoint_architecture_mismatches() — add model_family

 Same change as train.py §3b — add "model_family" to the fields tuple.

 4b. _build_ray_trainable Ray worker — use _instantiate_model

 # Existing import inside worker closure:
 from model import DeepSetModel, ModelConfig, _instantiate_model

 # cfg construction:
 cfg = ModelConfig(..., model_family=config.get("model_family", "deepset"))

 # Before:
 model = DeepSetModel(cfg=cfg).to(device)
 # After:
 model = _instantiate_model(cfg).to(device)

 Do NOT change HPO search space.

 ---
 Change 5 — src/sanity_checks.py (new file)

 Six checks. Each returns a dict with at least {"passed": bool, ...metrics}.

 ┌─────────────────────────────────────┬────────────────────────────────────────────┬──────────────────────┐
 │                Check                │              What it verifies              │    Pass criterion    │
 ├─────────────────────────────────────┼────────────────────────────────────────────┼──────────────────────┤
 │ check_permutation_invariance        │ row-shuffle + col-permute                  │ max_abs_delta ≤ 1e-5 │
 ├─────────────────────────────────────┼────────────────────────────────────────────┼──────────────────────┤
 │ check_ridge_primal_dual_consistency │ primal vs dual form on same problem        │ max_abs_delta ≤ 1e-4 │
 ├─────────────────────────────────────┼────────────────────────────────────────────┼──────────────────────┤
 │ check_gate_range                    │ gate ∈ (0,1) for 10 random contexts        │ all values in [0,1]  │
 ├─────────────────────────────────────┼────────────────────────────────────────────┼──────────────────────┤
 │ check_ridge_expert_output_shape     │ use_ridge_expert=True forward completes    │ output shape == (m,) │
 ├─────────────────────────────────────┼────────────────────────────────────────────┼──────────────────────┤
 │ check_no_test_label_in_signature    │ RidgeExpert.predict param inspection       │ no y_test param      │
 ├─────────────────────────────────────┼────────────────────────────────────────────┼──────────────────────┤
 │ check_forward_smoke_small           │ tiny context (n=20, p=5, m=8) forward pass │ no exception         │
 └─────────────────────────────────────┴────────────────────────────────────────────┴──────────────────────┘

 Output files:
 - artifacts/sanity/sanity_checks.json — full results dict with all_passed key
 - artifacts/sanity/sanity_checks.csv — one row per check: check_name, passed, metric_name, metric_value

 CLI: python src/sanity_checks.py [--out_dir artifacts/sanity] [--checkpoint PATH]

 If --checkpoint is not given, a freshly-initialized MarketAwareDeepSetModel with defaults
 (use_ridge_expert=True) is used (sufficient for structural checks).

 ---
 Change 6 — Test files (4 new files)

 tests/test_market_aware_deepset_shapes.py

 Minimum tests:
 - test_forward_shape_single_query — x_test=(p,) → scalar output
 - test_forward_shape_batch_query — x_test=(m,p) → shape (m,)
 - test_forward_works_without_ridge — use_ridge_expert=False
 - test_forward_works_with_ridge — use_ridge_expert=True
 - test_modelconfig_rejects_bad_model_family — ModelConfig(model_family="bad") raises
 - test_instantiate_model_routes_deepset — _instantiate_model returns DeepSetModel
 - test_instantiate_model_routes_market_aware — returns MarketAwareDeepSetModel
 - test_instantiate_model_unknown_family_raises — raises ValueError

 tests/test_market_aware_deepset_permutation.py

 Minimum tests:
 - test_row_permutation_invariance — shuffle X_train rows → max_abs_delta ≤ 1e-5
 - test_column_permutation_consistency — permute features consistently → max_abs_delta ≤ 1e-5
 - test_row_perm_invariance_with_ridge_expert — same as above but use_ridge_expert=True

 tests/test_ridge_expert.py

 Minimum tests:
 - test_primal_path_shape — n=50, p=5 → output (m,)
 - test_dual_path_shape — n=5, p=50 → output (m,)
 - test_primal_dual_consistency — same problem → max_abs_delta ≤ 1e-4
 - test_no_y_test_in_signature — inspect RidgeExpert.predict params
 - test_known_solution_identity_X — X = I_n (padded), lambda≈0, verify y_hat ≈ x_test @ β
 - test_device_consistency — output tensor on same device as input

 tests/test_query_sanity_checks.py

 Minimum tests:
 - test_check_permutation_invariance_passes — fresh model, passes
 - test_check_ridge_primal_dual_passes — passes
 - test_run_all_checks_returns_six_keys — 6 keys in result dict
 - test_save_results_writes_json — save_results(results, tmp_path) → sanity_checks.json exists
 - test_save_results_writes_csv — sanity_checks.csv exists with correct header

 ---
 Change 7 — Documentation

 MODEL.md — add §13

 Add section "13. MarketAwareDeepSetModel" after existing last section.

 Subsections:
 - 13.1 Motivation: why early feature pooling causes query collapse
 - 13.2 Architecture tensor flow: the 8-step summary from §1c above (condensed)
 - 13.3 Token construction: 6-feature token with intuition per component
 - 13.4 RidgeExpert: primal/dual selection, λ constraint
 - 13.5 Memory guard: n_sab_sample_per_feature=0 default; the 2.6 GB attention matrix
 warning at (m=128, p=128, n=200)
 - 13.6 Checkpoint v3 format: parallel table to existing v2 entry
 - 13.7 New ModelConfig fields: table of 8 new fields with types and defaults
 - 13.8 Extensibility: how cross-price, treatment effects, regime-aware priors attach at step 4

 MEMORY_TabPFN.md — add 2 guardrail sections

 Section: "Guardrail: Query Collapse and Early Feature Compression"

 Per prompt §12 verbatim (the full markdown block specified there).

 Section: "Guardrail: Defer HPO Expansion Until Architecture Sanity Passes"

 Per prompt §2 verbatim (the full markdown block specified there).

 ---
 Memory Guard (Critical Operational Note)

 n_sab_sample_per_feature > 0 at evaluation time with m=128, p=128, n=200 creates
 an attention matrix (m*p, n, n) = (16384, 200, 200) = 655M fp32 elements = 2.6 GB.
 This will OOM on A10G GPUs.

 Default: n_sab_sample_per_feature=0, sample_pool="attn" (single-seed cross-attention,
 O(n) memory). The token already encodes x_test[q,j], so AttentionPool produces
 query-differentiated evidence without O(n²) cost.

 Do NOT change this default until a chunking implementation is in place.

 ---
 Verification

 # 1. Syntax check both modified source files
 python -m py_compile src/model.py
 python -m py_compile src/evaluate.py
 python -m py_compile src/train.py
 python -m py_compile src/hpo.py
 python -m py_compile src/sanity_checks.py

 # 2. Run new targeted tests
 pytest tests/test_market_aware_deepset_shapes.py -v
 pytest tests/test_market_aware_deepset_permutation.py -v
 pytest tests/test_ridge_expert.py -v
 pytest tests/test_query_sanity_checks.py -v

 # 3. Run full test suite (no regressions)
 pytest tests -q

 # 4. Run structural sanity checks against a fresh model
 python src/sanity_checks.py --out_dir artifacts/sanity
 # Expected: all_passed=True, artifacts/sanity/sanity_checks.json written

 ---
 Implementation Order

 1. src/model.py — ModelConfig fields, RidgeExpert, MarketAwareDeepSetModel, _instantiate_model
 2. src/evaluate.py — import _instantiate_model, update load_model
 3. src/train.py — constant, import, mismatch check, train_fn, checkpoint save
 4. src/hpo.py — mismatch check fields, Ray worker instantiation
 5. src/sanity_checks.py — new file
 6. tests/test_market_aware_deepset_shapes.py — new file
 7. tests/test_market_aware_deepset_permutation.py — new file
 8. tests/test_ridge_expert.py — new file
 9. tests/test_query_sanity_checks.py — new file
 10. MODEL.md — §13
 11. MEMORY_TabPFN.md — 2 guardrail sections