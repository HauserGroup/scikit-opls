# Changelog

All notable changes are recorded here. The format loosely follows
[Keep a Changelog](https://keepachangelog.com/).

## Stability policy

While the version is `0.x` the public API may change **without** a deprecation
cycle. From `1.0` onward, public-API changes will follow scikit-learn's
deprecation pattern (`utils.deprecated`, `FutureWarning`, a two-release window),
and default-value changes will be documented here.

## Unreleased

### Changed (breaking, pre-1.0)

- Cross-validated selection of `n_orthogonal` moved out of `OPLS` into a new
  `OPLSCV` meta-estimator (precedent: `RidgeCV`/`LassoCV`). `OPLS.n_orthogonal`
  is now a plain `int`; the `"auto"` option and the `cv` parameter are removed
  from both `OPLS` and `OPLSDA`. Use `OPLSCV(...).fit(X, y)` and read
  `n_orthogonal_` / `q2_path_`.
- VIP is no longer computed eagerly in `fit`. `model.vip_` / `model.ortho_vip_`
  are removed; use `scikit_opls.inspection.vip(model)` /
  `scikit_opls.inspection.orthogonal_vip(model)` (these unwrap DA/CV wrappers).
- `vip.py` and `metrics.py` are folded into a new `inspection.py`.
- `predictive_weight(X, Y)` now uses the leading left singular vector of `XᵀY`,
  generalising to multivariate `Y`. For single-column `Y` the direction is
  unchanged (up to sign) and single-`y` OPLS output is bit-for-bit identical.

### Added

- `OPLSCV` (and `OPLSCV` in the `parametrize_with_checks` compliance suite).
- `inspection.vip` / `inspection.orthogonal_vip` model-level helpers.
- `_orthogonal.orthogonal_filter`, a block-agnostic NIPALS deflation primitive
  shared by `opls_filter` (and a future `O2PLS`).

- Full numpydoc docstrings on all public methods and functions.
- `OPLS.score` docstring documenting the inherited `RegressorMixin` R² score.
- Richer `__sklearn_tags__` (`target_tags.required`, `input_tags.sparse=False`,
  `non_deterministic=False`) with tests asserting the resolved tags.
- `ConvergenceWarning` when the orthogonal filter truncates early.
- Input validation (`check_array`, `check_consistent_length`) and an
  `n_permutations` guard in `permutation_test`; `check_array` in the plotting
  helpers.
- Explicit Ruff rule selection (`E,W,F,I,N,UP,D`, numpy docstring convention).
- `pytest-cov` and `[tool.coverage]` configuration.
- GitHub Actions CI (lint, format, type-check, tests, pre-commit) on Linux,
  macOS and Windows.
- `CONTRIBUTING.md`, a pull-request template, and `RELEASING.md`.

### Changed

- `OPLSDA` discovers classes with `unique_labels`.
- Numerical tests use `sklearn.utils._testing.assert_allclose`.
- Pinned the pre-commit `ruff` rev to the dev-group `ruff` version.

## 0.1.0

- Initial release: `OPLS` regressor and `OPLSDA` classifier with a
  scikit-learn-compatible API, VIP scores, permutation testing and diagnostic
  plots.
