"""O2PLS regressor with a scikit-learn interface."""

# scikit-learn's validation helpers use dynamic sentinel defaults that static
# type checkers do not model well. Runtime sklearn checks and tests are the gate.
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
# pyright: reportReturnType=false, reportAbstractUsage=false
# reportIncompatibleMethodOverride: sparse input is rejected by tags/validation.
# pyright: reportIncompatibleMethodOverride=false

from __future__ import annotations

from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.metrics import r2_score
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import (
    _check_feature_names_in,
    check_array,
    check_is_fitted,
    validate_data,
)

from scikit_opls._o2pls_core import (
    _TOL,
    _replay_orthogonal_filter,
    o2pls_fit,
)
from scikit_opls._preprocessing import VALID_SCALING, apply_scaling, compute_scaling
from scikit_opls._utils import _has_nonzero_variation


class O2PLS(RegressorMixin, TransformerMixin, BaseEstimator):
    """Two-block Orthogonal Projections to Latent Structures regression.

    O2PLS decomposes two preprocessed blocks into joint X-Y covariation,
    X-specific orthogonal structure, Y-specific orthogonal structure, and residual
    variation. Unlike :class:`sklearn.cross_decomposition.PLSRegression`, this
    implementation uses the Trygg-Wold orthonormal joint-loading convention:
    ``x_joint_loadings_`` equals ``x_joint_weights_`` and ``y_joint_loadings_``
    equals ``y_joint_weights_`` for the final joint part.

    Parameters
    ----------
    n_components : int, default=1
        Number of final joint O2PLS components.
    n_x_orthogonal : int, default=0
        Number of X-specific orthogonal components to remove.
    n_y_orthogonal : int, default=0
        Number of Y-specific orthogonal components to remove. For univariate
        ``Y``, v1 requires this to be zero because there is no multivariate
        Y-feature subspace for a stable Y-specific direction.
    scale : {"none", "center", "pareto", "standard"}, default="standard"
        Column preprocessing applied to both X and Y blocks.
    copy : bool, default=True
        Whether input arrays are copied during validation. Filtering still
        allocates working arrays.

    Attributes
    ----------
    x_joint_weights_, y_joint_weights_ : ndarray
        Final orthonormal joint weights fitted on the filtered blocks.
    x_joint_loadings_, y_joint_loadings_ : ndarray
        Copies of the final joint weights under the O2PLS orthonormal-loading
        convention.
    x_joint_scores_, y_joint_scores_ : ndarray
        Final joint scores on the filtered training blocks.
    x_orthogonal_weights_, x_orthogonal_scores_, x_orthogonal_loadings_ : ndarray
        Sequential X-specific orthogonal components.
    y_orthogonal_weights_, y_orthogonal_scores_, y_orthogonal_loadings_ : ndarray
        Sequential Y-specific orthogonal components.
    coef_filtered_ : ndarray of shape (n_features_in_, n_targets_)
        Coefficient matrix mapping scaled, X-orthogonally-filtered X to scaled
        predicted Y. This orientation is intentionally ``(n_features, n_targets)``
        and no raw-space ``coef_`` alias is exposed in v1.
    x_mean_, x_std_, y_mean_, y_std_ : ndarray
        Centering/scaling vectors for each block.
    r2x_, r2y_, r2x_ortho_, r2y_ortho_ : float
        Training-set diagnostic sum-of-squares ratios on preprocessed blocks.
        These are not guaranteed additive variance partitions.
    """

    n_features_in_: int
    feature_names_in_: NDArray[np.str_]
    n_targets_: int
    x_mean_: NDArray[np.float64]
    x_std_: NDArray[np.float64]
    y_mean_: NDArray[np.float64]
    y_std_: NDArray[np.float64]
    x_joint_weights_: NDArray[np.float64]
    y_joint_weights_: NDArray[np.float64]
    x_joint_scores_: NDArray[np.float64]
    y_joint_scores_: NDArray[np.float64]
    x_joint_loadings_: NDArray[np.float64]
    y_joint_loadings_: NDArray[np.float64]
    x_orthogonal_weights_: NDArray[np.float64]
    x_orthogonal_scores_: NDArray[np.float64]
    x_orthogonal_loadings_: NDArray[np.float64]
    y_orthogonal_weights_: NDArray[np.float64]
    y_orthogonal_scores_: NDArray[np.float64]
    y_orthogonal_loadings_: NDArray[np.float64]
    b_t_: NDArray[np.float64]
    b_u_: NDArray[np.float64]
    coef_filtered_: NDArray[np.float64]
    x_filtered_: NDArray[np.float64]
    y_filtered_: NDArray[np.float64]
    x_residuals_: NDArray[np.float64]
    y_residuals_: NDArray[np.float64]
    r2x_: float
    r2y_: float
    r2x_ortho_: float
    r2y_ortho_: float
    n_components_: int
    n_x_orthogonal_: int
    n_y_orthogonal_: int
    n_features_out_: int
    _y_ndim: int

    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "n_x_orthogonal": [Interval(Integral, 0, None, closed="left")],
        "n_y_orthogonal": [Interval(Integral, 0, None, closed="left")],
        "scale": [StrOptions(set(VALID_SCALING))],
        "copy": ["boolean"],
    }

    def __init__(
        self,
        n_components: int = 1,
        n_x_orthogonal: int = 0,
        n_y_orthogonal: int = 0,
        scale: str = "standard",
        copy: bool = True,
    ) -> None:
        self.n_components = n_components
        self.n_x_orthogonal = n_x_orthogonal
        self.n_y_orthogonal = n_y_orthogonal
        self.scale = scale
        self.copy = copy

    def fit(self, X: ArrayLike, Y: ArrayLike) -> O2PLS:
        """Fit the O2PLS model."""
        if isinstance(self.n_components, bool):
            raise ValueError("n_components must be an integer, not bool.")
        if isinstance(self.n_x_orthogonal, bool):
            raise ValueError("n_x_orthogonal must be an integer, not bool.")
        if isinstance(self.n_y_orthogonal, bool):
            raise ValueError("n_y_orthogonal must be an integer, not bool.")
        self._validate_params()

        self._y_ndim = np.asarray(Y).ndim
        X, Y_valid = validate_data(
            self,
            X,
            Y,
            dtype=np.float64,
            ensure_min_samples=2,
            copy=self.copy,
            multi_output=True,
        )
        Y2 = self._as_2d_target(Y_valid)
        self.n_targets_ = Y2.shape[1]
        if self.n_targets_ == 1:
            if self.n_components != 1:
                raise ValueError("O2PLS v1 requires n_components=1 for univariate Y.")
            if self.n_y_orthogonal != 0:
                raise ValueError("O2PLS v1 requires n_y_orthogonal=0 for univariate Y.")

        self.x_mean_, self.x_std_ = compute_scaling(X, self.scale)
        self.y_mean_, self.y_std_ = compute_scaling(Y2, self.scale)
        Xs = apply_scaling(X, self.x_mean_, self.x_std_)
        Ys = apply_scaling(Y2, self.y_mean_, self.y_std_)
        if not _has_nonzero_variation(Xs, axis=0):
            raise ValueError("X has no non-zero variation after preprocessing.")
        if not _has_nonzero_variation(Ys, axis=0):
            raise ValueError("Y has no non-zero variation after preprocessing.")

        fit = o2pls_fit(
            Xs,
            Ys,
            self.n_components,
            self.n_x_orthogonal,
            self.n_y_orthogonal,
            tol=_TOL,
        )

        self.x_joint_weights_ = fit.x_joint_weights
        self.y_joint_weights_ = fit.y_joint_weights
        self.x_joint_scores_ = fit.x_joint_scores
        self.y_joint_scores_ = fit.y_joint_scores
        self.x_joint_loadings_ = fit.x_joint_loadings
        self.y_joint_loadings_ = fit.y_joint_loadings
        self.x_orthogonal_weights_ = fit.x_orthogonal_weights
        self.x_orthogonal_scores_ = fit.x_orthogonal_scores
        self.x_orthogonal_loadings_ = fit.x_orthogonal_loadings
        self.y_orthogonal_weights_ = fit.y_orthogonal_weights
        self.y_orthogonal_scores_ = fit.y_orthogonal_scores
        self.y_orthogonal_loadings_ = fit.y_orthogonal_loadings
        self.b_t_ = fit.b_t
        self.b_u_ = fit.b_u
        self.x_filtered_ = fit.x_filtered
        self.y_filtered_ = fit.y_filtered
        self.x_residuals_ = fit.x_residuals
        self.y_residuals_ = fit.y_residuals
        self.r2x_ = fit.r2x
        self.r2y_ = fit.r2y
        self.r2x_ortho_ = fit.r2x_ortho
        self.r2y_ortho_ = fit.r2y_ortho
        self.n_components_ = fit.n_components
        self.n_x_orthogonal_ = fit.n_x_orthogonal
        self.n_y_orthogonal_ = fit.n_y_orthogonal
        self.singular_values_initial_ = fit.singular_values_initial
        self.singular_values_final_ = fit.singular_values_final
        self.coef_filtered_ = (
            self.x_joint_weights_ @ self.b_t_ @ self.y_joint_loadings_.T
        )
        self.n_features_out_ = self.n_components_
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predict Y from X, reconstructing only the joint Y structure."""
        check_is_fitted(self)
        X_filtered = self.filter_transform_x(X)
        y_scaled = X_filtered @ self.coef_filtered_
        y_pred = self._unscale_y(y_scaled)
        return self._restore_y_shape(y_pred)

    def predict_x(self, Y: ArrayLike) -> NDArray[np.float64]:
        """Predict X from Y, reconstructing only the joint X structure."""
        check_is_fitted(self)
        Y_filtered = self.filter_transform_y(Y)
        U = Y_filtered @ self.y_joint_weights_
        T_pred = U @ self.b_u_
        x_scaled = T_pred @ self.x_joint_loadings_.T
        return x_scaled * self.x_std_ + self.x_mean_

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return X-side joint scores after replaying the fitted X-orthogonal filter."""
        check_is_fitted(self)
        return self.filter_transform_x(X) @ self.x_joint_weights_

    def transform_y(self, Y: ArrayLike) -> NDArray[np.float64]:
        """Return Y-side joint scores after replaying the fitted Y-orthogonal filter."""
        check_is_fitted(self)
        return self.filter_transform_y(Y) @ self.y_joint_weights_

    def transform_pair(
        self, X: ArrayLike, Y: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(transform(X), transform_y(Y))``."""
        return self.transform(X), self.transform_y(Y)

    def transform_orthogonal_x(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return sequential X-specific orthogonal scores."""
        check_is_fitted(self)
        return self._filter_x(X)[1]

    def transform_orthogonal_y(self, Y: ArrayLike) -> NDArray[np.float64]:
        """Return sequential Y-specific orthogonal scores."""
        check_is_fitted(self)
        return self._filter_y(Y)[1]

    def filter_transform_x(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return preprocessed X after the fitted X-orthogonal filter."""
        check_is_fitted(self)
        return self._filter_x(X)[0]

    def filter_transform_y(self, Y: ArrayLike) -> NDArray[np.float64]:
        """Return preprocessed Y after the fitted Y-orthogonal filter."""
        check_is_fitted(self)
        return self._filter_y(Y)[0]

    def get_feature_names_out(self, input_features=None) -> NDArray[np.object_]:
        """Output names for :meth:`transform` joint-score columns."""
        check_is_fitted(self, "n_features_out_")
        _check_feature_names_in(self, input_features)
        return np.asarray(
            [f"o2pls_joint{i}" for i in range(self.n_features_out_)], dtype=object
        )

    def score(self, X: ArrayLike, y: ArrayLike, sample_weight=None) -> float:
        """Coefficient of determination R² of ``predict(X)`` against ``y``."""
        return float(r2_score(y, self.predict(X), sample_weight=sample_weight))

    @staticmethod
    def _as_2d_target(Y: ArrayLike) -> NDArray[np.float64]:
        """Convert a validated target block to 2D."""
        arr = np.asarray(Y, dtype=np.float64)
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        if arr.ndim == 2:
            return arr
        raise ValueError(f"Y must be 1D or 2D, got shape {arr.shape}.")

    def _validate_y_block(self, Y: ArrayLike) -> NDArray[np.float64]:
        """Validate a Y-like block for Y-side methods."""
        arr = check_array(
            Y,
            dtype=np.float64,
            ensure_2d=False,
            ensure_min_samples=1,
            copy=self.copy,
        )
        Y2 = self._as_2d_target(arr)
        if Y2.shape[1] != self.n_targets_:
            raise ValueError(
                f"Y has {Y2.shape[1]} target columns, but O2PLS was fitted with "
                f"{self.n_targets_}."
            )
        return Y2

    def _filter_x(
        self, X: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Scale and replay the fitted X-orthogonal filter."""
        X_valid = validate_data(self, X, reset=False, dtype=np.float64)
        Xs = apply_scaling(X_valid, self.x_mean_, self.x_std_)
        return _replay_orthogonal_filter(
            Xs, self.x_orthogonal_weights_, self.x_orthogonal_loadings_
        )

    def _filter_y(
        self, Y: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Scale and replay the fitted Y-orthogonal filter."""
        Y_valid = self._validate_y_block(Y)
        Ys = apply_scaling(Y_valid, self.y_mean_, self.y_std_)
        return _replay_orthogonal_filter(
            Ys, self.y_orthogonal_weights_, self.y_orthogonal_loadings_
        )

    def _unscale_y(self, Ys: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map a scaled Y block back to raw units."""
        return Ys * self.y_std_ + self.y_mean_

    def _restore_y_shape(self, Y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return predictions with the fitted target dimensionality convention."""
        if self._y_ndim == 1 and Y.shape[1] == 1:
            return Y.ravel()
        return Y

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.regressor_tags.poor_score = True
        tags.target_tags.required = True
        tags.target_tags.multi_output = True
        tags.target_tags.single_output = True
        tags.input_tags.sparse = False
        tags.non_deterministic = False
        return tags
