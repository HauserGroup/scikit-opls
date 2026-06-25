# scikit-opls

Orthogonal Projections to Latent Structures (**OPLS** / **OPLS-DA**) with a
scikit-learn interface.

OPLS (Trygg & Wold, 2002) splits the variation in `X` into a *predictive* part
correlated with the response and *orthogonal* parts that are not. `scikit-opls`
removes the orthogonal variation with a NIPALS filter and then fits
[`sklearn.cross_decomposition.PLSRegression`](https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html)
on the cleaned `X` as the predictive engine. With `n_orthogonal=0` the model
reduces *exactly* to `PLSRegression`.

This mirrors the behaviour of the R packages
[`ropls::opls`](https://bioconductor.org/packages/ropls/) (OPLS / OPLS-DA) and
uses the orthogonal-scores PLS algorithm of
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

from scikit_opls.inspection import vip
vip(model)                    # variable importance (predictive), computed on demand
```

Let cross-validated Q2 choose the number of orthogonal components
(like `ropls` `orthoI = NA`) with the `OPLSCV` meta-estimator:

```python
from scikit_opls import OPLSCV

cv = OPLSCV(n_components=1, cv=7).fit(X, y)
cv.n_orthogonal_              # chosen number of orthogonal components
cv.q2_path_                   # out-of-fold Q2 at k = 0, 1, …
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

```python
from scikit_opls.plotting import scores_plot, s_plot
from scikit_opls.validation import permutation_test

scores_plot(model, X, y)                       # t_pred vs t_ortho
s_plot(model, X)                               # covariance vs correlation
permutation_test(OPLS(n_orthogonal=2), X, y)   # model significance
```

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

| Parameter      | Meaning                                                            |
| -------------- | ----------------------------------------------------------------- |
| `n_components` | Predictive components (classic OPLS uses 1).                      |
| `n_orthogonal` | Orthogonal components to remove (`int`; use `OPLSCV` to select).  |
| `scale`        | `"none"`, `"center"`, `"pareto"`, `"standard"` (`ropls` `scaleC`).|

`OPLSCV` adds `cv`, `max_orthogonal` and `q2_tol` for cross-validated selection.

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
- Thévenot, E.A. et al. (2015). *ropls* R package.
