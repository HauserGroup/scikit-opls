# scikit-opls

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/HauserGroup/scikit-opls/actions/workflows/ci.yml/badge.svg)](https://github.com/HauserGroup/scikit-opls/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://hausergroup.github.io/scikit-opls/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Orcid: Jakob](https://img.shields.io/badge/Jakob-bar?style=flat&logo=orcid&labelColor=white&color=grey)](https://orcid.org/0000-0002-2841-7284)

Orthogonal Projections to Latent Structures (**OPLS** / **OPLS-DA**) with a
scikit-learn interface.

OPLS (Trygg & Wold, 2002) splits the variation in `X` into a *predictive* part
correlated with the response and *orthogonal* parts that are not. `scikit-opls`
removes the orthogonal variation with a NIPALS filter and then fits
[`sklearn.cross_decomposition.PLSRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html)
on the cleaned `X` as the predictive engine. With `n_orthogonal=0` the model
reduces *exactly* to `PLSRegression`.

This project is inspired by the R
[`ropls`](https://www.rdocumentation.org/packages/ropls/versions/1.4.2) package
and uses the orthogonal-scores PLS algorithm of
[`pls::oscorespls.fit`](https://cran.r-project.org/package=pls) as its engine.

## Install

```bash
uv sync
```

## Usage

### OPLS regression

```python
import numpy as np
from scikit_opls import OPLS

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 50))
y = X[:, 0] * 2.0 + rng.normal(scale=0.1, size=100)

model = OPLS(n_components=1, n_orthogonal=2, scale="standard").fit(X, y)

model.predict(X)              # predictions
model.transform(X)            # predictive scores
model.transform_orthogonal(X) # orthogonal scores
model.r2x_, model.r2y_        # fit summaries
model.vip_                    # variable importance (predictive), lazy property
```

Let cross-validated Q2 choose the number of orthogonal components with
scikit-learn's `GridSearchCV` — no bespoke estimator needed (`scoring=None` gives
out-of-fold R2, which equals Q2 for `OPLS`):

```python
from sklearn.model_selection import GridSearchCV
from scikit_opls import OPLS

search = GridSearchCV(
    OPLS(n_components=1), {"n_orthogonal": list(range(10))}, cv=7
).fit(X, y)
search.best_params_["n_orthogonal"]       # chosen count
search.best_estimator_                    # final OPLS refit on all data
search.cv_results_["mean_test_score"]     # out-of-fold R2/Q2 path
```

For OPLS-DA, wrap `OPLSDA()` the same way; an `int` `cv` becomes stratified
automatically and `scoring="roc_auc"` is usually preferable.

To bias toward fewer orthogonal components — prefer the smallest count whose mean
score is within a tolerance of the best — pass a `refit` callable:

```python
import numpy as np

def parsimonious_refit(cv_results, tol=0.01):
    scores = np.asarray(cv_results["mean_test_score"], dtype=float)
    counts = np.asarray(cv_results["param_n_orthogonal"], dtype=int)
    within = np.flatnonzero(scores >= np.nanmax(scores) - tol)
    return int(within[np.argmin(counts[within])])

GridSearchCV(
    OPLS(n_components=1), {"n_orthogonal": list(range(10))},
    cv=7, refit=parsimonious_refit,
).fit(X, y)
```

### OPLS-DA (binary classification)

```python
from scikit_opls import OPLSDA

y = np.where(X[:, 0] > 0, "case", "ctrl")
clf = OPLSDA(n_components=1, n_orthogonal=2).fit(X, y)

clf.predict(X)            # class labels
clf.predict_proba(X)      # Platt-scaled probabilities
clf.decision_function(X)  # signed confidence
clf.opls_.transform(X)    # predictive scores of the underlying OPLS model
```

### Diagnostics

Plotting needs the optional `plot` extra (`pip install scikit-opls[plot]`); it
follows scikit-learn's Display convention.

```python
from scikit_opls.plotting import OPLSScoresDisplay, SPlotDisplay
from scikit_opls.validation import permutation_test

OPLSScoresDisplay.from_estimator(model, X, y)  # t_pred vs t_ortho
SPlotDisplay.from_estimator(model, X)          # covariance vs correlation
permutation_test(OPLS(n_orthogonal=2), X, y)   # model significance
```

The older `scores_plot(model, X, y)` / `s_plot(model, X)` functions remain as
thin wrappers.

### Example datasets

Two small scripts under `examples/` show usage with CSV data hosted as GitHub
release assets. The examples read these URLs directly with `pandas.read_csv`, so
the datasets do not need to be stored in the local checkout:

- `https://github.com/HauserGroup/scikit-opls/releases/download/data/colorectal_cancer_nmr.csv`
- `https://github.com/HauserGroup/scikit-opls/releases/download/data/palmerpenguins.csv`

```bash
uv run python examples/colorectal_cancer_nmr_oplsda.py
uv run python examples/palmerpenguins_opls_regression.py
```

## Parameters

| Parameter      | Meaning                                                           |
| -------------- | ----------------------------------------------------------------- |
| `n_components` | Predictive components (classic OPLS uses 1).                      |
| `n_orthogonal` | Orthogonal components to remove (`int`; tune via `GridSearchCV`). |
| `scale`        | `"none"`, `"center"`, `"pareto"`, `"standard"`.                   |

Wrap `OPLS` in `GridSearchCV` over `n_orthogonal` for cross-validated selection
(see the snippet above); `cv`, `scoring` and `n_jobs` come from `GridSearchCV`.

## Development

```bash
uv sync --dev              # install the project and dev tools
uv run pre-commit install  # enable the git hooks (run once)

uv run pytest --cov        # tests + coverage (incl. sklearn check_estimator)
uv run ruff check          # lint
uv run ruff format --check # format check
uv run pyright src         # type-check
uv run pre-commit run --all-files  # run every hook
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contributor workflow.

## References

- Trygg, J. & Wold, S. (2002). *Orthogonal projections to latent structures
  (O-PLS).* Journal of Chemometrics, 16(3), 119–128.
- Galindo-Prieto, B., Eriksson, L. & Trygg, J. (2014). *Variable influence on
  projection (VIP) for OPLS models.* Journal of Chemometrics, 28(8), 623–632.
