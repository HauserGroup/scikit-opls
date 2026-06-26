# Quickstart

## Regression

```python
import numpy as np
from scikit_opls import OPLS

rng = np.random.default_rng(0)
X = rng.normal(size=(80, 30))
y = X[:, 0] + 0.1 * rng.normal(size=80)

model = OPLS(n_components=1, n_orthogonal=2).fit(X, y)
model.predict(X)              # predicted y
model.transform(X)            # predictive scores
model.transform_orthogonal(X) # orthogonal scores
model.r2y_, model.rmsee_      # training-fit summaries
```

## Choosing `n_orthogonal` by cross-validation

Use scikit-learn's `GridSearchCV` directly — `OPLS` has no path structure, so a
dedicated `…CV` class buys nothing. `scoring=None` gives out-of-fold R2, which
equals Q2 for `OPLS`.

```python
from sklearn.model_selection import GridSearchCV
from scikit_opls import OPLS

search = GridSearchCV(
    OPLS(n_components=1), {"n_orthogonal": list(range(10))}, cv=7
).fit(X, y)
search.best_params_["n_orthogonal"]       # selected count
search.cv_results_["mean_test_score"]     # out-of-fold R2/Q2 path
search.best_estimator_.predict(X)         # final model refit on all data
```

### Parsimonious selection

To prefer the fewest orthogonal components whose score is within a tolerance of
the best, pass a `refit` callable:

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

## Classification (OPLS-DA)

```python
from sklearn.model_selection import GridSearchCV
from scikit_opls import OPLSDA

y_lab = np.where(X[:, 0] > 0, "case", "ctrl")
clf = OPLSDA(n_components=1, n_orthogonal=2).fit(X, y_lab)
clf.predict(X)            # class labels
clf.predict_proba(X)      # Platt-scaled probabilities
clf.decision_function(X)  # signed confidence

# Cross-validated OPLS-DA selection: an int cv is stratified automatically.
GridSearchCV(
    OPLSDA(), {"n_orthogonal": list(range(10))}, cv=5, scoring="roc_auc"
).fit(X, y_lab)
```

## Inspection, plotting and validation

```python
from scikit_opls.inspection import vip
from scikit_opls.plotting import OPLSScoresDisplay, SPlotDisplay
from scikit_opls.validation import permutation_test

vip(model)                                     # predictive VIP per feature
OPLSScoresDisplay.from_estimator(model, X, y)  # t_pred vs t_ortho
SPlotDisplay.from_estimator(model, X)          # covariance vs correlation
permutation_test(OPLS(n_orthogonal=2), X, y)   # model significance
```
