"""Cross-validated selection of OPLS orthogonal components.

``OPLSCV`` is to :class:`~scikit_opls.OPLS` what ``LassoCV``/``RidgeCV`` are to
``Lasso``/``Ridge``: the same model, with ``n_orthogonal`` chosen by cross-validated
Q2 rather than fixed. Selection is forward (the ``ropls`` heuristic): add orthogonal
components while out-of-fold Q2 improves by more than ``q2_tol``, capped at
``max_orthogonal``. The final model is refit on all data and composed in (mirroring
how :class:`~scikit_opls.OPLSDA` wraps ``OPLS``).
"""

# See _opls.py: scikit-learn's validate_data is under-typed; suppress the
# resulting static-checker false positives.
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportReturnType=false

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.metrics import r2_score
from sklearn.model_selection import check_cv, cross_val_predict
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data

from ._opls import OPLS
from ._preprocessing import VALID_SCALING


class OPLSCV(RegressorMixin, TransformerMixin, BaseEstimator):
    """OPLS with the number of orthogonal components chosen by cross-validated Q2.

    Parameters
    ----------
    n_components : int, default=1
        Number of predictive components passed to the inner :class:`~scikit_opls.OPLS`.
    scale : {"none", "center", "pareto", "standard"}, default="standard"
        Column preprocessing applied to ``X``.
    cv : int, cross-validation generator or iterable, default=7
        Cross-validation used to score each candidate ``n_orthogonal``.
    max_orthogonal : int, default=9
        Cap on the number of orthogonal components (``ropls`` orthoI = NA cap).
    q2_tol : float, default=0.01
        Minimum out-of-fold Q2 improvement required to keep an extra component.
    copy : bool, default=True
        Whether the input arrays are copied during validation.

    Attributes
    ----------
    n_orthogonal_ : int
        Chosen number of orthogonal components.
    q2_path_ : ndarray
        Out-of-fold Q2 at ``k = 0, 1, …`` up to the stopping point.
    opls_ : OPLS
        Final model refit on all data with ``n_orthogonal=n_orthogonal_``.
    """

    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "scale": [StrOptions(set(VALID_SCALING))],
        "cv": ["cv_object"],
        "max_orthogonal": [Interval(Integral, 0, None, closed="left")],
        "q2_tol": [Interval(Real, 0, None, closed="left")],
        "copy": ["boolean"],
    }

    def __init__(
        self,
        n_components: int = 1,
        scale: str = "standard",
        cv: object = 7,
        max_orthogonal: int = 9,
        q2_tol: float = 0.01,
        copy: bool = True,
    ) -> None:
        self.n_components = n_components
        self.scale = scale
        self.cv = cv
        self.max_orthogonal = max_orthogonal
        self.q2_tol = q2_tol
        self.copy = copy

    def fit(self, X: ArrayLike, y: ArrayLike) -> OPLSCV:
        """Select ``n_orthogonal`` by cross-validated Q2, then refit on all data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training predictors.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : OPLSCV
            The fitted estimator.
        """
        self._validate_params()
        X, y = validate_data(
            self, X, y, dtype=np.float64, ensure_min_samples=2, copy=self.copy
        )
        cv = check_cv(self.cv)
        cap = max(min(self.max_orthogonal, X.shape[1] - 1, X.shape[0] - 2), 0)

        path = [self._cv_q2(X, y, 0, cv)]
        best_k, prev = 0, path[0]
        for k in range(1, cap + 1):
            q2 = self._cv_q2(X, y, k, cv)
            path.append(q2)
            if q2 - prev <= self.q2_tol:
                break
            best_k, prev = k, q2

        self.n_orthogonal_ = best_k
        self.q2_path_ = np.asarray(path)
        self.opls_ = OPLS(
            n_components=self.n_components,
            n_orthogonal=best_k,
            scale=self.scale,
            copy=self.copy,
        ).fit(X, y)
        return self

    def _cv_q2(
        self, X: NDArray[np.float64], y: NDArray[np.float64], k: int, cv
    ) -> float:
        """Out-of-fold Q2 for an inner ``OPLS`` configured with ``n_orthogonal=k``."""
        est = OPLS(
            n_components=self.n_components,
            n_orthogonal=k,
            scale=self.scale,
            copy=self.copy,
        )
        return float(r2_score(y, cross_val_predict(est, X, y, cv=cv)))

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predict ``y`` with the selected final model."""
        check_is_fitted(self)
        return self.opls_.predict(X)

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Project samples onto the predictive components of the final model."""
        check_is_fitted(self)
        return self.opls_.transform(X)

    def transform_orthogonal(self, X: ArrayLike) -> NDArray[np.float64]:
        """Project samples onto the orthogonal components of the final model."""
        check_is_fitted(self)
        return self.opls_.transform_orthogonal(X)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.regressor_tags.poor_score = True
        tags.target_tags.required = True
        tags.input_tags.sparse = False
        tags.non_deterministic = False
        return tags
