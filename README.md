# scikit-opls

[![PyPI](https://img.shields.io/pypi/v/scikit-opls.svg)](https://pypi.org/project/scikit-opls/)
[![Python versions](https://img.shields.io/pypi/pyversions/scikit-opls.svg)](https://pypi.org/project/scikit-opls/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/HauserGroup/scikit-opls/blob/main/LICENSE)
[![CI](https://github.com/HauserGroup/scikit-opls/actions/workflows/ci.yml/badge.svg)](https://github.com/HauserGroup/scikit-opls/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://hausergroup.github.io/scikit-opls/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Orcid: Jakob](https://img.shields.io/badge/Jakob-bar?style=flat&logo=orcid&labelColor=white&color=grey)](https://orcid.org/0000-0002-2841-7284)

Orthogonal Projections to Latent Structures (**OPLS** / **OPLS-DA**) with a
scikit-learn interface.

OPLS (Trygg & Wold, 2002) splits the variation in `X` into a *predictive* part
correlated with the response and *orthogonal* parts that are not. `scikit-opls`
removes the orthogonal variation with an OSC-style orthogonal filter and then fits
[`sklearn.cross_decomposition.PLSRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html)
on the cleaned `X` as the predictive engine. With `n_orthogonal=0` the model
reduces *exactly* to `PLSRegression`.

## Install

Requires Python 3.12+.

```bash
pip install scikit-opls
```

Plotting is an opt-in extra, since `matplotlib` is only imported by
`scikit_opls.plotting`:

```bash
pip install "scikit-opls[plot]"
```

With uv: `uv add scikit-opls`. To work on the package itself, see
[Development](#development).

## Usage

### OPLS regression

```python
import numpy as np
from scikit_opls import OPLS

rng = np.random.default_rng(0)
X = rng.normal(size=(100, 50))
y = X[:, 0] * 2.0 + rng.normal(scale=0.1, size=100)

model = OPLS(n_components=1, n_orthogonal=2, scale="standard").fit(X, y)

model.predict(X)  # predictions
model.transform(X)  # predictive scores
model.transform_orthogonal(X)  # orthogonal scores
model.filter_transform(X)  # preprocessed, orthogonal-filtered X fed to the engine
model.r2x_, model.r2y_  # fit summaries
model.vip_  # variable importance (predictive), lazy property
```

The whole fitted pipeline (scaling → orthogonal filter → predictive PLS) is linear,
so it collapses to coefficients on the raw input space:

```python
y_hat = (X @ model.coef_raw_.T + model.intercept_raw_).ravel()  # == model.predict(X)
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
search.best_params_["n_orthogonal"]  # chosen count
search.best_estimator_  # final OPLS refit on all data
search.cv_results_["mean_test_score"]  # out-of-fold R2/Q2 path
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
    OPLS(n_components=1),
    {"n_orthogonal": list(range(10))},
    cv=7,
    refit=parsimonious_refit,
).fit(X, y)
```

### OPLS-DA (binary classification)

```python
from scikit_opls import OPLSDA

y = np.where(X[:, 0] > 0, "case", "ctrl")
clf = OPLSDA(n_components=1, n_orthogonal=2).fit(X, y)

clf.predict(X)  # class labels
clf.decision_function(X)  # raw signed OPLS regression output
clf.opls_.transform(X)  # predictive scores of the underlying OPLS model

# Probabilities: wrap in a cross-fitted calibrator when each class has enough
# samples for the chosen calibration CV split.
from sklearn.calibration import CalibratedClassifierCV

CalibratedClassifierCV(clf, cv=5).fit(X, y).predict_proba(X)
```

### O2PLS (two-block)

`O2PLS` models two blocks jointly, separating the covariation they share from
the structured variation specific to each, and predicts in both directions.

```python
from scikit_opls import O2PLS

T = rng.normal(size=(100, 2))
Xb = T @ rng.normal(size=(2, 20)) + 0.1 * rng.normal(size=(100, 20))
Yb = T @ rng.normal(size=(2, 5)) + 0.1 * rng.normal(size=(100, 5))

o2 = O2PLS(n_components=2, n_x_orthogonal=1, n_y_orthogonal=1).fit(Xb, Yb)

o2.predict(Xb)  # Y from X
o2.predict_x(Yb)  # X from Y
o2.transform(Xb)  # joint X scores
o2.transform_orthogonal_x(Xb)  # X-specific orthogonal scores
o2.r2x_, o2.r2y_  # joint fit summaries
```

### K-OPLS (non-linear regression)

`KOPLS` performs the OPLS split in the feature space induced by a kernel, so
non-linear X/y relationships can be modelled while the predictive and
Y-orthogonal scores stay interpretable. `kernel="linear"` with `scale="none"`
reproduces `OPLS` exactly.

```python
from scikit_opls import KOPLS

Xk = rng.normal(size=(120, 6))
yk = np.sin(2 * Xk[:, 0]) + Xk[:, 1] ** 2

k = KOPLS(n_orthogonal=1, kernel="rbf", gamma=0.2).fit(Xk, yk)

k.predict(Xk)  # predicted y
k.transform(Xk)  # predictive scores
k.transform_orthogonal(Xk)  # Y-orthogonal scores
k.q_residuals(Xk)  # feature-space residual per sample
```

Kernels follow the `KernelRidge` parameter set, `"precomputed"` included (pass
`scale="none"`, the `(n_test, n_train)` block at `predict`). The kernel is always
centered in feature space. There are no input-space loadings, so `KOPLS` exposes
no `coef_` and no VIP scores.

### Diagnostics

Plotting needs the optional `plot` extra (`pip install "scikit-opls[plot]"`); it
follows scikit-learn's Display convention.

```python
from scikit_opls.plotting import OPLSScoresDisplay, SPlotDisplay
from scikit_opls.validation import permutation_test

# Draw score plot (t_pred vs t_ortho). Supports component selection for multi-component PLS
OPLSScoresDisplay.from_estimator(
    model, X, y, predictive_component=0, orthogonal_component=0
)

# Draw S-plot (covariance vs correlation) for a specific predictive component
SPlotDisplay.from_estimator(model, X, component=0)

# Permutation significance testing
permutation_test(OPLS(n_orthogonal=2), X, y)
```

> [!NOTE]
> **Pipeline support in plotting:** Diagnostic plotting displays support `OPLS`,
> `OPLSDA`, and pipelines ending in one. For tuned models, pass
> `search.best_estimator_` explicitly. When passing a pipeline, pass raw `X` as
> expected by the pipeline. When passing the final OPLS step directly, pass the
> already transformed matrix. For pipeline S-plots, points are in the transformed
> feature space received by the final OPLS step.

### Example datasets

Two runnable scripts live under `examples/`, and CI executes both on every push:

```bash
uv run python examples/palmerpenguins_opls_regression.py  # OPLS regression
uv run python examples/o2pls_synthetic.py                 # two-block O2PLS
```

The penguins example reads its CSV straight from a GitHub release asset with
`pandas.read_csv`, so no dataset is stored in the checkout:

- `https://github.com/HauserGroup/scikit-opls/releases/download/data/palmerpenguins.csv`

## Parameters

| Parameter                  | Meaning                                                           |
| -------------------------- | ----------------------------------------------------------------- |
| `n_components`             | Predictive components (classic OPLS uses 1).                      |
| `n_orthogonal`             | Orthogonal components to remove (`int`; tune via `GridSearchCV`). |
| `scale`                    | `"none"`, `"center"`, `"pareto"`, `"standard"`.                   |
| `kernel`                   | `KOPLS` only: kernel name or callable, `"precomputed"` included.  |
| `gamma`, `degree`, `coef0` | `KOPLS` only: kernel coefficients.                                |

`OPLSDA` takes the same parameters. `O2PLS` replaces `n_orthogonal` with
`n_x_orthogonal` and `n_y_orthogonal`, one per block.

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

See [CONTRIBUTING.md](https://github.com/HauserGroup/scikit-opls/blob/main/CONTRIBUTING.md) for the full contributor workflow.

## References

This project is inspired by the R
[`ropls`](https://www.rdocumentation.org/packages/ropls/versions/1.4.2) package
and uses the orthogonal-scores PLS algorithm of
[`pls::oscorespls.fit`](https://cran.r-project.org/package=pls) as its engine.

- Trygg, J. & Wold, S. (2002). *Orthogonal projections to latent structures
  (O-PLS).* Journal of Chemometrics, 16(3), 119–128.
  [doi:10.1002/cem.695](https://doi.org/10.1002/cem.695)
- Trygg, J. & Wold, S. (2003). *O2-PLS, a two-block (X–Y) latent variable
  regression (LVR) method with an integral OSC filter.* Journal of
  Chemometrics, 17(1), 53–64.
  [doi:10.1002/cem.775](https://doi.org/10.1002/cem.775)
- Wold, S., Antti, H., Lindgren, F. & Öhman, J. (1998). *Orthogonal signal
  correction of near-infrared spectra.* Chemometrics and Intelligent Laboratory
  Systems, 44(1–2), 175–185.
  [doi:10.1016/S0169-7439(98)00109-9](<https://doi.org/10.1016/S0169-7439(98)00109-9>)
- Bylesjö, M., Rantalainen, M., Cloarec, O., Nicholson, J. K., Holmes, E. &
  Trygg, J. (2006). *OPLS discriminant analysis: combining the strengths of
  PLS-DA and SIMCA classification.* Journal of Chemometrics, 20(8–10), 341–351.
  [doi:10.1002/cem.1006](https://doi.org/10.1002/cem.1006)
- Galindo-Prieto, B., Eriksson, L. & Trygg, J. (2014). *Variable influence on
  projection (VIP) for OPLS models.* Journal of Chemometrics, 28(8), 623–632.
  [doi:10.1002/cem.2627](https://doi.org/10.1002/cem.2627)

## Citing

If `scikit-opls` supports published work, please cite the software and the
methods above. Citation metadata lives in [CITATION.cff](https://github.com/HauserGroup/scikit-opls/blob/main/CITATION.cff); GitHub
renders a formatted citation from it under **Cite this repository**. See the
[Citing page](https://hausergroup.github.io/scikit-opls/citing/) for BibTeX.

## License

[MIT](https://github.com/HauserGroup/scikit-opls/blob/main/LICENSE).
