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
