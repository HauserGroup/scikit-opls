"""Orthogonal PLS (OPLS) regressor with a scikit-learn interface.

OPLS (Trygg & Wold, 2002) splits the variation in ``X`` into a *predictive* part
correlated with ``y`` and *orthogonal* parts uncorrelated with ``y``. This estimator
removes the orthogonal variation with a NIPALS filter (:mod:`scikit_opls._orthogonal`)
and then fits :class:`sklearn.cross_decomposition.PLSRegression` on the cleaned ``X`` as
the predictive engine. With ``n_orthogonal=0`` it reduces exactly to ``PLSRegression``.
"""

# scikit-learn's validate_data / check_cv / clone use sentinel-string parameter
# defaults that lead static type checkers to flag every downstream array op as a
# false positive. Suppress those categories here; the test suite and
# ``check_estimator`` are the real correctness gate.
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportReturnType=false

from __future__ import annotations

from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import check_cv, cross_val_predict
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import check_is_fitted, validate_data

from ._orthogonal import apply_orthogonal_filter, opls_filter
from ._preprocessing import VALID_SCALING, apply_scaling, compute_scaling
from .metrics import explained_x_variance
from .vip import orthogonal_vip, predictive_vip

# Selection rule shared with ropls: keep adding orthogonal components while the
# cross-validated Q2 improves by more than this margin. ropls caps automatic
# orthogonal components at 9; we use the same cap.
_Q2_IMPROVEMENT_TOL = 0.01
_MAX_AUTO_ORTHO = 9


class OPLS(RegressorMixin, TransformerMixin, BaseEstimator):
    """Orthogonal Projections to Latent Structures regression.

    Parameters
    ----------
    n_components : int, default=1
        Number of predictive components. True OPLS uses 1, which is required
        whenever ``n_orthogonal != 0``. Values > 1 are only allowed with
        ``n_orthogonal=0`` (plain multi-component PLS, a non-``ropls`` mode).
    n_orthogonal : int or "auto", default=1
        Number of orthogonal (y-uncorrelated) components to remove from ``X``.
        ``"auto"`` selects the count by cross-validated Q2 (like ``ropls`` with
        ``orthoI = NA``), capped at 9 as in ``ropls``.
    scale : {"none", "center", "pareto", "standard"}, default="standard"
        Column preprocessing applied to ``X`` (matches ``ropls`` ``scaleC``).
    cv : int, cross-validation generator or iterable, default=7
        Cross-validation used for ``n_orthogonal="auto"``.
    copy : bool, default=True
        Whether the input arrays are copied during validation.

    Attributes
    ----------
    n_orthogonal_ : int
        Number of orthogonal components actually used.
    x_ortho_weights_, x_ortho_loadings_, x_ortho_scores_ : ndarray
        Orthogonal weight/loading/score matrices.
    x_weights_, x_loadings_, x_scores_, y_loadings_, coef_ : ndarray
        Predictive model parameters taken from the underlying PLS engine (in the
        preprocessed, orthogonal-filtered space).
    pls_ : PLSRegression
        The fitted predictive engine.
    x_mean_, x_std_ : ndarray
        Centering/scaling vectors applied to ``X``.
    r2x_, r2x_ortho_, r2y_, rmsee_ : float
        Training-set fit summaries. For cross-validated Q2 use
        :func:`sklearn.model_selection.cross_val_score`.
    """

    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "n_orthogonal": [
            Interval(Integral, 0, None, closed="left"),
            StrOptions({"auto"}),
        ],
        "scale": [StrOptions(set(VALID_SCALING))],
        "cv": ["cv_object"],
        "copy": ["boolean"],
    }

    def __init__(
        self,
        n_components: int = 1,
        n_orthogonal: int | str = 1,
        scale: str = "standard",
        cv: object = 7,
        copy: bool = True,
    ) -> None:
        self.n_components = n_components
        self.n_orthogonal = n_orthogonal
        self.scale = scale
        self.cv = cv
        self.copy = copy

    def fit(self, X: ArrayLike, y: ArrayLike) -> OPLS:
        self._validate_params()
        # multi_output=False ravels a column-vector y (with a DataConversionWarning)
        # and rejects multi-column y: OPLS is univariate.
        X, y = validate_data(
            self, X, y, dtype=np.float64, ensure_min_samples=2, copy=self.copy
        )

        # True-OPLS contract: one predictive component when any orthogonal
        # filtering is requested (ropls predI=1). n_orthogonal=0 keeps the
        # unrestricted PLSRegression-equivalence mode.
        if self.n_components != 1 and self.n_orthogonal != 0:
            raise ValueError(
                f"OPLS uses one predictive component when orthogonal filtering is "
                f"requested; got n_components={self.n_components} with "
                f"n_orthogonal={self.n_orthogonal!r}. Set n_components=1, or use "
                "n_orthogonal=0 for plain (multi-component) PLS."
            )
        if self.n_components > min(X.shape):
            raise ValueError(
                f"n_components={self.n_components} exceeds the maximum of "
                f"min(n_samples, n_features)={min(X.shape)}."
            )

        n_ortho = self._resolve_n_orthogonal(X, y)
        self.x_mean_, self.x_std_ = compute_scaling(X, self.scale)
        Xs = apply_scaling(X, self.x_mean_, self.x_std_)

        if n_ortho > 0:
            ofit = opls_filter(Xs, y - y.mean(), n_ortho)
            X_filtered = ofit.x_filtered
            self.x_ortho_weights_ = ofit.x_ortho_weights
            self.x_ortho_loadings_ = ofit.x_ortho_loadings
            self.x_ortho_scores_ = ofit.x_ortho_scores
            self.n_orthogonal_ = ofit.n_components
        else:
            X_filtered = Xs
            self.x_ortho_weights_ = np.zeros((X.shape[1], 0))
            self.x_ortho_loadings_ = np.zeros((X.shape[1], 0))
            self.x_ortho_scores_ = np.zeros((X.shape[0], 0))
            self.n_orthogonal_ = 0

        self.pls_ = PLSRegression(n_components=self.n_components, scale=False)
        self.pls_.fit(X_filtered, y)

        # Surface the predictive model parameters from the engine.
        self.x_weights_ = self.pls_.x_weights_
        self.x_loadings_ = self.pls_.x_loadings_
        self.x_scores_ = self.pls_.x_scores_
        self.y_loadings_ = self.pls_.y_loadings_
        self.coef_ = self.pls_.coef_

        y_fit = self.pls_.predict(X_filtered)
        self.r2x_ = explained_x_variance(Xs, self.x_scores_, self.x_loadings_)
        self.r2x_ortho_ = explained_x_variance(
            Xs, self.x_ortho_scores_, self.x_ortho_loadings_
        )
        self.r2y_ = float(r2_score(y, y_fit))
        self.rmsee_ = float(root_mean_squared_error(y, y_fit))
        self.vip_ = predictive_vip(self.x_weights_, self.x_scores_, self.y_loadings_)
        self.ortho_vip_ = orthogonal_vip(
            self.x_ortho_weights_, self.x_ortho_scores_, self.x_ortho_loadings_
        )
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predict ``y`` for new samples."""
        check_is_fitted(self)
        X_filtered, _ = self._filter(X)
        return self.pls_.predict(X_filtered).ravel()

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Project samples onto the predictive components."""
        check_is_fitted(self)
        X_filtered, _ = self._filter(X)
        return self.pls_.transform(X_filtered)

    def transform_orthogonal(self, X: ArrayLike) -> NDArray[np.float64]:
        """Project samples onto the orthogonal components."""
        check_is_fitted(self)
        return self._filter(X)[1]

    def _filter(self, X: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Preprocess and orthogonal-filter new ``X`` exactly as at fit time.

        Returns the filtered ``X`` and the orthogonal scores.
        """
        X = validate_data(self, X, reset=False, dtype=np.float64)
        Xs = apply_scaling(X, self.x_mean_, self.x_std_)
        return apply_orthogonal_filter(
            Xs, self.x_ortho_weights_, self.x_ortho_loadings_
        )

    def _resolve_n_orthogonal(
        self, X: NDArray[np.float64], y: NDArray[np.float64]
    ) -> int:
        if self.n_orthogonal == "auto":
            return self._auto_select_orthogonal(X, y)
        return int(self.n_orthogonal)

    def _auto_select_orthogonal(
        self, X: NDArray[np.float64], y: NDArray[np.float64]
    ) -> int:
        """Add orthogonal components while cross-validated Q2 keeps improving."""
        cv = check_cv(self.cv)
        cap = max(min(_MAX_AUTO_ORTHO, X.shape[1] - 1, X.shape[0] - 2), 0)

        best_k = 0
        prev_q2 = self._cv_q2(X, y, 0, cv)
        for k in range(1, cap + 1):
            q2 = self._cv_q2(X, y, k, cv)
            if q2 - prev_q2 <= _Q2_IMPROVEMENT_TOL:
                break
            best_k, prev_q2 = k, q2
        return best_k

    def _cv_q2(
        self, X: NDArray[np.float64], y: NDArray[np.float64], k: int, cv
    ) -> float:
        """Out-of-fold Q2 for a clone configured with ``n_orthogonal=k``."""
        model = clone(self).set_params(n_orthogonal=k, cv=cv)
        return float(r2_score(y, cross_val_predict(model, X, y, cv=cv)))

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.regressor_tags.poor_score = True
        return tags
