# Changelog

All notable changes are recorded here. The format loosely follows
[Keep a Changelog](https://keepachangelog.com/).

## Stability policy

While the version is `0.x` the public API may change **without** a deprecation
cycle. From `1.0` onward, public-API changes will follow scikit-learn's
deprecation pattern (`utils.deprecated`, `FutureWarning`, a two-release window),
and default-value changes will be documented here.

## Unreleased

### Added

- `KOPLS`, a kernel OPLS regressor and supervised transformer implementing
  Rantalainen et al. (2007). It keeps the predictive / Y-orthogonal split of
  `OPLS` but performs it in the feature space induced by a kernel, so non-linear
  X/y relationships can be modelled. Kernels follow the `KernelRidge` parameter
  set (`kernel`, `gamma`, `degree`, `coef0`, `kernel_params`, including
  `"precomputed"`), the kernel is always centered in feature space, and multiple
  targets are supported natively. With `kernel="linear"` and `scale="none"` it
  reproduces `OPLS` exactly. Diagnostics mirror `OPLS`: `r2x_` / `r2x_ortho_` /
  `r2y_` with per-component arrays, `score_distance` and `q_residuals`. There are
  no input-space loadings in feature space, so `KOPLS` exposes no `coef_` and no
  VIP scores.

## 0.1.0 — 2026-09-07

First public release: the `OPLS` regressor, the `OPLSDA` binary classifier and
the `O2PLS` two-block estimator, all scikit-learn compatible, with VIP scores,
permutation testing and diagnostic plots.

Items under *Changed* and *Removed* describe how the API settled during
development. No earlier version was published, so nothing here breaks a released
interface.

### Added

- `OPLS`, an OPLS regressor and supervised transformer: OSC-style orthogonal
  filtering followed by `PLSRegression` on the cleaned `X`.
- `OPLSDA`, a binary classifier composing `OPLS` against a -1/+1 dummy response.
- `O2PLS`, a dense two-block estimator with X/Y preprocessing, sequential X- and
  Y-orthogonal filtering, final joint-subspace re-estimation, and bidirectional
  `predict` / `predict_x`. The v1 implementation is dense only and exposes
  `coef_filtered_` for scaled, X-filtered inputs rather than a raw-space `coef_`
  alias.
- `OPLS.coef_raw_` / `OPLS.intercept_raw_`: linear coefficients on the original
  raw input feature space, collapsing scaling, the orthogonal filter and the
  predictive PLS into one map, so `X @ coef_raw_.T + intercept_raw_` reproduces
  `predict(X)`. No bare sklearn `coef_` alias is exposed (it would be the
  raw-space coefficient, not the engine's filtered-space one).
- `OPLS.filter_transform(X)` returns the preprocessed, orthogonal-filtered `X`
  actually passed to the predictive PLS engine (so
  `pls_.predict(filter_transform(X))` matches `predict(X)`); useful for
  diagnostics and downstream modelling.
- `OPLS.get_feature_names_out` so `set_output(transform="pandas")` yields named
  predictive-score columns (`opls_pred0, …`).
- Lazy `vip_` / `ortho_vip_` properties on `OPLS` (and on `OPLSDA`, delegating to
  the inner OPLS), following scikit-learn's `feature_importances_` convention —
  computed on access, not eagerly in `fit`. Feature selection is supported via
  `SelectFromModel(OPLS(), importance_getter="vip_", threshold=1.0)` (the
  VIP > 1 rule), composable in a `Pipeline` / `GridSearchCV`.
- `OPLSScoresDisplay` and `SPlotDisplay` plotting classes following
  scikit-learn's Display convention (`from_estimator(...)`, `plot(ax=...)`,
  `ax_` / `figure_`).
- `n_jobs` on `validation.permutation_test` (runs the independent permutations
  in parallel; reproducible regardless of `n_jobs`). Cross-validated
  `n_orthogonal` selection inherits `n_jobs` from `GridSearchCV`.
- `_orthogonal.orthogonal_filter`, a block-agnostic OSC-style deflation
  primitive shared by `opls_filter` and `O2PLS`.
- Richer `__sklearn_tags__` (`target_tags.required`, `input_tags.sparse=False`,
  `non_deterministic=False`) with tests asserting the resolved tags.
- `ConvergenceWarning` when the orthogonal filter truncates early.
- Input validation (`check_array`, `check_consistent_length`) and an
  `n_permutations` guard in `permutation_test`; `check_array` in the plotting
  helpers.
- Full numpydoc docstrings on all public methods and functions.
- Zensical documentation site (`zensical.toml`, mkdocstrings, numpy docstring
  style) with a `zensical build` CI gate, a GitHub Pages (Actions) deploy
  workflow, and `docs/citing.md`.
- Packaging metadata for PyPI: SPDX `license` expression with the bundled
  `LICENSE` file, `authors`/`maintainers`, `keywords` and trove `classifiers`.
- `.github/workflows/release.yml`: tag-driven build, PyPI publication through
  Trusted Publishing (OIDC, no stored token), signed build attestations and an
  automatic GitHub release. It refuses to publish when the tag does not match
  `__version__`.
- GitHub Actions CI (lint, format, type-check, tests, pre-commit) on Linux,
  macOS and Windows across Python 3.12, 3.13 and 3.14, plus a `package` job that
  builds the sdist and wheel, runs `twine check --strict`, and imports the
  package from each installed artifact in a clean environment.
- Explicit Ruff rule selection (`E,W,F,I,N,UP,D`, numpy docstring convention),
  `pytest-cov` and `[tool.coverage]` configuration.
- `CONTRIBUTING.md`, a pull-request template, and `RELEASING.md`.

### Changed

- Renamed the second parameter of `O2PLS.fit` from `Y` to `y`, matching the
  scikit-learn estimator contract (sklearn's own `cross_decomposition`
  deprecated `Y` in favour of `y`). Positional calls are unaffected; keyword
  calls must use `fit(X, y=...)`. Y-block-specific helpers (`predict_x`,
  `transform_y`, `filter_transform_y`, …) keep their uppercase `Y` block
  argument, as they are outside the sklearn contract.
- Renamed the fitted attribute `rmsee_` to `rmse_` (uncorrected training root
  mean squared error). The old name implied a degrees-of-freedom-corrected
  calibration error, which it never computed; no alias is kept.
- `predictive_weight(X, Y)` now uses the leading left singular vector of `XᵀY`,
  generalising to multivariate `Y`. For single-column `Y` the direction is
  unchanged (up to sign) and single-`y` OPLS output is bit-for-bit identical.
- Cross-validated selection of `n_orthogonal` is done with scikit-learn's
  `GridSearchCV` directly — there is no bespoke selection API.
  `OPLS.n_orthogonal` is a plain `int`. Use
  `GridSearchCV(OPLS(...), {"n_orthogonal": [...]}).fit(X, y)` and read
  `best_params_["n_orthogonal"]`, `best_estimator_` and
  `cv_results_["mean_test_score"]`. For a parsimony bias, pass a `refit`
  callable (recipe in the README / quickstart). For `OPLSDA`, use
  `GridSearchCV(OPLSDA(), {"n_orthogonal": [...]}, scoring="roc_auc")`, which
  gives stratified folds for classification.
- `matplotlib` is an optional dependency in the `plot` extra
  (`pip install "scikit-opls[plot]"`). Only `scikit_opls.plotting` needs it and
  it is imported lazily.
- `OPLSDA` uses its fitted `LabelEncoder` as the single class-label source.
- The supported Python floor is 3.12 (`requires-python = ">=3.12"`); lint and
  type checks target 3.12 while development happens on 3.13
  (`.python-version`).
- Numerical tests use `sklearn.utils._testing.assert_allclose`.
- Pinned the pre-commit `ruff` rev to the dev-group `ruff` version.

### Removed

- `O2PLS.score`; it duplicated the inherited `RegressorMixin.score` (R² of
  `predict(X)` against `y`) with identical behaviour. `score` remains available
  via the mixin.
- `OPLSDA`'s `probability` parameter and its in-sample Platt calibration
  (`predict_proba`, `raw_score`). `OPLSDA` is now a clean score classifier:
  `decision_function` returns the raw signed OPLS regression output and
  `predict` its sign. For probabilities, wrap in
  `CalibratedClassifierCV(OPLSDA(...))` (cross-fitted, better calibrated). This
  also removes the `predict`/`predict_proba` boundary inconsistency the
  in-sample calibrator caused.
- The `"auto"` option and the `cv` parameter on `OPLS` and `OPLSDA`, the
  `OPLSCV` estimator, and the `selection.select_orthogonal` factory. Use the
  `GridSearchCV` recipe under *Changed*. `OPLSDACV` will not be added.
- The public `scikit_opls.inspection` module and its `vip(model)` /
  `orthogonal_vip(model)` functions; the stateless math moved to a private
  `_inspection` module and is reached through the `vip_` properties.

### Fixed

- Orthogonal filtering no longer extracts components past the point where a
  block's rank is exhausted. Both `OPLS`'s filter and O2PLS's block-specific
  extraction judged convergence against the *current* deflated block, whose sum
  of squares shrinks with every deflation, so rounding noise stayed significant
  relative to itself. Convergence is now measured against the original block and
  the component count is bounded by `min(n_samples, n_features)`. Fitted
  `n_orthogonal_`, `n_x_orthogonal_` and `n_y_orthogonal_` on rank-deficient data
  may be lower than before, and no longer vary with the BLAS implementation.
- O2PLS orthogonal extraction is now invariant to a global rescaling of the
  blocks. Resolvability was measured against `max(block_ssq, 1.0)`, an absolute
  floor in the units of the data, so identical blocks yielded different component
  counts depending only on their scale.

### Documentation

- Completed the numpydoc `Attributes` sections of `OPLS`, `OPLSDA` and `O2PLS`
  (per-component diagnostics, training Q residuals, `b_t_`/`b_u_`,
  `n_features_in_`/`feature_names_in_`, etc.), documented that
  `O2PLS.x_filtered_`/`y_filtered_`/`x_residuals_`/`y_residuals_` make the fitted
  estimator scale with training-data size, and noted that the string `scale`
  parameter differs from `PLSRegression`'s boolean `scale`.
- `OPLS.score` docstring documenting the inherited `RegressorMixin` R² score.
- `CITATION.cff` gained `type`, split author names, ORCID, `version`, `license`,
  `repository-code`, keywords and the method references.
- `RELEASING.md` describes the automated tag-driven release and names
  `src/scikit_opls/version.py` as the single source of truth for the version.
