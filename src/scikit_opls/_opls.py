"""Orthogonal PLS (OPLS) regressor with a scikit-learn interface.

OPLS (Trygg & Wold, 2002) splits the variation in ``X`` into a *predictive* part
correlated with ``y`` and *orthogonal* parts uncorrelated with ``y``. This estimator
removes the orthogonal variation with a NIPALS filter (:mod:`scikit_opls._orthogonal`)
and then fits :class:`sklearn.cross_decomposition.PLSRegression` on the cleaned ``X`` as
the predictive engine. With ``n_orthogonal=0`` it reduces exactly to ``PLSRegression``.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.exceptions import DataConversionWarning
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.utils.validation import check_is_fitted

from ._orthogonal import apply_orthogonal_filter, opls_filter
from ._preprocessing import apply_scaling, check_scaling, compute_scaling
from ._validation import clone_estimator, resolve_cv, validate_fit, validate_predict
from .metrics import explained_x_variance, q2_y, r2_y, rmsee
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
        Cross-validation used for Q2 and for ``n_orthogonal="auto"``.
    copy : bool, default=True
        Whether to copy ``X`` (and ``y``) or perform in-place preprocessing.

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
        Training fit summaries. ``q2_`` is set by :meth:`fit` only when computable.
    """

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

    # ------------------------------------------------------------------ fitting
    def fit(self, X: ArrayLike, y: ArrayLike) -> OPLS:
        check_scaling(self.scale)
        self._check_n_components()
        X, y = validate_fit(self, X, y, copy=self.copy)
        if y.ndim == 2 and y.shape[-1] == 1:
            warnings.warn(
                "A column-vector y was passed when a 1d array was expected. "
                "Please change the shape of y to (n_samples,).",
                DataConversionWarning,
                stacklevel=2,
            )
            y = y.ravel()
        self._y_ndim = y.ndim
        Y2 = y.reshape(len(y), -1)  # always 2-D: (n,) -> (n, 1)

        # Univariate-response contract: OPLS is defined for a single response.
        if Y2.shape[1] > 1:
            raise ValueError(
                "OPLS requires a single response column; got y with "
                f"{Y2.shape[1]} columns. Multi-output OPLS is not supported."
            )
        # True-OPLS predictive-component contract: a single predictive component
        # when any orthogonal filtering is requested (ropls predI=1 for OPLS).
        # n_orthogonal=0 keeps the unrestricted PLSRegression-equivalence mode.
        if self.n_components != 1 and self.n_orthogonal != 0:
            raise ValueError(
                f"OPLS uses one predictive component when orthogonal filtering is "
                f"requested; got n_components={self.n_components} with "
                f"n_orthogonal={self.n_orthogonal!r}. Set n_components=1, or use "
                "n_orthogonal=0 for plain (multi-component) PLS."
            )

        n_ortho = self._resolve_n_orthogonal(X, Y2)
        max_components = min(X.shape[0], X.shape[1])
        if self.n_components > max_components:
            raise ValueError(
                f"n_components={self.n_components} exceeds the maximum of "
                f"min(n_samples, n_features)={max_components}."
            )

        self.x_mean_, self.x_std_ = compute_scaling(X, self.scale)
        Xs = apply_scaling(X, self.x_mean_, self.x_std_)

        if n_ortho > 0:
            yc = Y2[:, 0] - Y2[:, 0].mean()
            ofit = opls_filter(Xs, yc, n_ortho)
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
        self.pls_.fit(X_filtered, Y2)

        # Surface the predictive model parameters from the engine.
        self.x_weights_ = self.pls_.x_weights_
        self.x_loadings_ = self.pls_.x_loadings_
        self.x_scores_ = self.pls_.x_scores_
        self.y_loadings_ = self.pls_.y_loadings_
        self.coef_ = self.pls_.coef_

        self._x_scaled_ref = Xs
        self.r2x_ = explained_x_variance(Xs, self.x_scores_, self.x_loadings_)
        self.r2x_ortho_ = explained_x_variance(
            Xs, self.x_ortho_scores_, self.x_ortho_loadings_
        )
        y_fit = self.pls_.predict(X_filtered)
        self.r2y_ = r2_y(Y2, y_fit)
        self.rmsee_ = rmsee(Y2, y_fit)

        self.vip_ = predictive_vip(self.x_weights_, self.x_scores_, self.y_loadings_)
        self.ortho_vip_ = orthogonal_vip(
            self.x_ortho_weights_, self.x_ortho_scores_, self.x_ortho_loadings_
        )
        return self

    # ---------------------------------------------------------------- predict/transform
    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predict ``y`` for new samples."""
        check_is_fitted(self)
        Xs = self._filter_new(X)
        pred = self.pls_.predict(Xs)
        if self._y_ndim == 1:
            pred = np.asarray(pred).ravel()
        return pred

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Project samples onto the predictive components."""
        check_is_fitted(self)
        Xs = self._filter_new(X)
        return np.asarray(self.pls_.transform(Xs), dtype=np.float64)

    def transform_orthogonal(self, X: ArrayLike) -> NDArray[np.float64]:
        """Project samples onto the orthogonal components."""
        check_is_fitted(self)
        Xv = validate_predict(self, X)
        Xs = apply_scaling(Xv, self.x_mean_, self.x_std_)
        _, scores = apply_orthogonal_filter(
            Xs, self.x_ortho_weights_, self.x_ortho_loadings_
        )
        return scores

    def _filter_new(self, X: ArrayLike) -> NDArray[np.float64]:
        """Preprocess + orthogonal-filter new ``X`` exactly as at fit time."""
        Xv = validate_predict(self, X)
        Xs = apply_scaling(Xv, self.x_mean_, self.x_std_)
        X_filtered, _ = apply_orthogonal_filter(
            Xs, self.x_ortho_weights_, self.x_ortho_loadings_
        )
        return X_filtered

    # ------------------------------------------------------------ validation
    def _check_n_components(self) -> None:
        nc = self.n_components
        if isinstance(nc, (bool, np.bool_)) or not isinstance(nc, (int, np.integer)):
            raise ValueError(f"n_components must be a positive int, got {nc!r}")
        if nc < 1:
            raise ValueError(f"n_components must be >= 1, got {nc}")

    # ------------------------------------------------------------ n_orthogonal
    def _resolve_n_orthogonal(
        self, X: NDArray[np.float64], Y2: NDArray[np.float64]
    ) -> int:
        if isinstance(self.n_orthogonal, str):
            if self.n_orthogonal != "auto":
                raise ValueError(
                    f"n_orthogonal must be a non-negative int or 'auto', "
                    f"got {self.n_orthogonal!r}"
                )
            return self._auto_select_orthogonal(X, Y2)
        if isinstance(self.n_orthogonal, (bool, np.bool_)) or not isinstance(
            self.n_orthogonal, (int, np.integer)
        ):
            raise ValueError(
                f"n_orthogonal must be a non-negative int or 'auto', "
                f"got {self.n_orthogonal!r}"
            )
        if self.n_orthogonal < 0:
            raise ValueError(f"n_orthogonal must be >= 0, got {self.n_orthogonal}")
        return int(self.n_orthogonal)

    def _auto_select_orthogonal(
        self, X: NDArray[np.float64], Y2: NDArray[np.float64]
    ) -> int:
        """Add orthogonal components while cross-validated Q2 keeps improving."""
        if Y2.shape[1] > 1:
            raise ValueError("n_orthogonal='auto' requires a single response.")
        y = Y2[:, 0]
        cv = resolve_cv(self.cv)
        cap = min(_MAX_AUTO_ORTHO, X.shape[1] - 1, X.shape[0] - 2)
        cap = max(cap, 0)

        best_k = 0
        prev_q2 = self._cv_q2(X, y, 0, cv)
        for k in range(1, cap + 1):
            q2 = self._cv_q2(X, y, k, cv)
            if q2 - prev_q2 > _Q2_IMPROVEMENT_TOL:
                best_k = k
                prev_q2 = q2
            else:
                break
        return best_k

    def _cv_q2(
        self, X: NDArray[np.float64], y: NDArray[np.float64], k: int, cv: Any
    ) -> float:
        """Out-of-fold Q2 for a clone configured with ``n_orthogonal=k``."""
        model = clone_estimator(self).set_params(n_orthogonal=k, cv=cv)
        y_cv = np.asarray(cross_val_predict(model, X, y, cv=cv))
        return q2_y(y, y_cv)

    # --------------------------------------------------------------------- misc
    def score_q2(self, X: ArrayLike, y: ArrayLike) -> float:
        """Cross-validated Q2 of this configuration on ``(X, y)``."""
        Xa = np.asarray(X, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64).ravel()
        cv = resolve_cv(self.cv)
        y_cv = np.asarray(cross_val_predict(clone_estimator(self), Xa, ya, cv=cv))
        return q2_y(ya, y_cv)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.regressor_tags.poor_score = True
        return tags
