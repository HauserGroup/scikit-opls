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

```python
from scikit_opls import OPLSCV

cv_model = OPLSCV(n_components=1, cv=7).fit(X, y)
cv_model.n_orthogonal_   # selected number of orthogonal components
cv_model.q2_path_        # out-of-fold Q2 at each step
cv_model.predict(X)      # delegates to the refit final model
```

## Classification (OPLS-DA)

```python
from scikit_opls import OPLSDA

y_lab = np.where(X[:, 0] > 0, "case", "ctrl")
clf = OPLSDA(n_components=1, n_orthogonal=2).fit(X, y_lab)
clf.predict(X)            # class labels
clf.predict_proba(X)      # Platt-scaled probabilities
clf.decision_function(X)  # signed confidence
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
