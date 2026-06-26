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
clf.decision_function(X)  # raw signed OPLS regression output

# Probabilities via cross-fitted calibration when each class has enough samples
# for the chosen calibration CV split:
from sklearn.calibration import CalibratedClassifierCV
calibrated_clf = CalibratedClassifierCV(clf, cv=5).fit(X, y_lab)
calibrated_clf.predict_proba(X)

# Cross-validated OPLS-DA selection: an int cv is stratified automatically.
GridSearchCV(
    OPLSDA(), {"n_orthogonal": list(range(10))}, cv=5, scoring="roc_auc"
).fit(X, y_lab)
```

## Inspection, plotting and validation

```python
from scikit_opls.plotting import OPLSScoresDisplay, SPlotDisplay
from scikit_opls.validation import permutation_test

model.vip_                                     # predictive VIP per feature (lazy)

# Draw score plot (t_pred vs t_ortho). Supports component selection for multi-component PLS
OPLSScoresDisplay.from_estimator(
    model, X, y, predictive_component=0, orthogonal_component=0
)

# Draw S-plot (covariance vs correlation) for a specific predictive component
SPlotDisplay.from_estimator(model, X, component=0)

# Permutation significance testing
permutation_test(OPLS(n_orthogonal=2), X, y)
```

!!! note "Pipeline support in plotting"
Diagnostic plotting displays support `OPLS`, `OPLSDA`, pipelines ending in one,
and fitted search meta-estimators exposing `best_estimator_` around either shape.
When passing a pipeline, pass raw `X` as expected by the pipeline. When passing the
final OPLS step directly, pass the already transformed matrix. For pipeline
S-plots, points are in the transformed feature space received by the final OPLS
step.
