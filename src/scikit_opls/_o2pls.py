"""O2PLS regressor with a scikit-learn interface."""

# reportAbstractUsage covers Interval/StrOptions: their base _Constraint declares
# __str__ abstract and the subclass override is not visible to static type checkers.
# reportIncompatibleMethodOverride: sparse input is rejected by tags/validation.
# pyright: reportAbstractUsage=false, reportIncompatibleMethodOverride=false

from __future__ import annotations

import warnings
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import (
    _check_feature_names_in,
    check_is_fitted,
)

from scikit_opls._o2pls_core import (
    _TOL,
    _replay_orthogonal_filter,
    o2pls_fit,
)
from scikit_opls._preprocessing import VALID_SCALING, apply_scaling, compute_scaling
from scikit_opls._utils import (
    _check_array_float64,
    _extract_feature_names,
    _has_nonzero_variation,
    _reject_bool_param,
    _validate_x,
    _validate_xy,
)


class O2PLS(RegressorMixin, TransformerMixin, BaseEstimator):
    """Two-block Orthogonal Projections to Latent Structures regression.

    O2PLS decomposes two preprocessed blocks into joint X-Y covariation,
    X-specific orthogonal structure, Y-specific orthogonal structure, and residual
    variation. Unlike [`PLSRegression`][sklearn.cross_decomposition.PLSRegression], this
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
        Column preprocessing applied to both X and Y blocks. Note: unlike the
        boolean ``scale`` parameter of
        [`PLSRegression`][sklearn.cross_decomposition.PLSRegression], this is a string
        mode; passing ``True``/``False`` raises an error.
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
    b_t_ : ndarray of shape (n_components_, n_components_)
        Regression matrix mapping joint X scores to joint Y scores
        (used by [`predict`][scikit_opls.O2PLS.predict]).
    b_u_ : ndarray of shape (n_components_, n_components_)
        Regression matrix mapping joint Y scores to joint X scores
        (used by [`predict_x`][scikit_opls.O2PLS.predict_x]).
    coef_filtered_ : ndarray of shape (n_features_in_, n_targets_)
        Coefficient matrix mapping scaled, X-orthogonally-filtered X to scaled
        predicted Y. This orientation is intentionally ``(n_features, n_targets)``
        and no raw-space ``coef_`` alias is exposed in v1.
    x_filtered_, y_filtered_ : ndarray
        Preprocessed training blocks after orthogonal filtering, shapes
        ``(n_samples, n_features)`` and ``(n_samples, n_targets_)``. Note:
        together with the residual blocks these training-set diagnostics make
        the fitted (and pickled) estimator scale with the training data size.
    x_residuals_, y_residuals_ : ndarray
        Training residual blocks after removing joint and orthogonal structure,
        same shapes as ``x_filtered_``/``y_filtered_``.
    x_mean_, x_std_, y_mean_, y_std_ : ndarray
        Centering/scaling vectors for each block.
    r2x_, r2y_, r2x_ortho_, r2y_ortho_ : float
        Training-set diagnostic sum-of-squares ratios on preprocessed blocks.
        These are not guaranteed additive variance partitions.
    singular_values_initial_, singular_values_final_ : ndarray
        Singular values of the X'Y covariance block before and after orthogonal
        filtering.
    n_components_, n_x_orthogonal_, n_y_orthogonal_ : int
        Numbers of joint / X-orthogonal / Y-orthogonal components actually
        fitted; may be lower than requested after truncation.
    n_targets_ : int
        Number of target columns seen during [`fit`][scikit_opls.O2PLS.fit].
    n_features_out_ : int
        Number of joint-score columns returned by
        [`transform`][scikit_opls.O2PLS.transform].
    n_features_in_ : int
        Number of features seen during [`fit`][scikit_opls.O2PLS.fit].
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during [`fit`][scikit_opls.O2PLS.fit]. Defined only when
        ``X`` has
        feature names that are all strings.
    y_feature_names_in_ : ndarray of shape (n_targets_,)
        Names of target columns seen during [`fit`][scikit_opls.O2PLS.fit]. Defined
        only when 2D ``y`` has column names that are all strings.

    See Also
    --------
    OPLS : Single-block variant removing only X-orthogonal structure.
    OPLSDA : Binary OPLS discriminant analysis.
    sklearn.cross_decomposition.PLSCanonical : Symmetric two-block decomposition
        without an integral orthogonal-signal-correction filter.

    Notes
    -----
    Requested orthogonal components may be truncated with a
    [`ConvergenceWarning`][sklearn.exceptions.ConvergenceWarning] when the preliminary
    joint
    subspace leaves no numerically resolvable block-specific residual variation.
    This is most common when ``n_components`` approaches the rank or feature
    dimension of one block.

    References
    ----------
    .. [1] Trygg, J. & Wold, S. (2003). O2-PLS, a two-block (X-Y) latent
           variable regression (LVR) method with an integral OSC filter.
           Journal of Chemometrics, 17(1), 53-64.
           https://doi.org/10.1002/cem.775
    .. [2] Trygg, J. & Wold, S. (2002). Orthogonal projections to latent
           structures (O-PLS). Journal of Chemometrics, 16(3), 119-128.
           https://doi.org/10.1002/cem.695

    Examples
    --------
    >>> import numpy as np
    >>> from scikit_opls import O2PLS
    >>> rng = np.random.default_rng(0)
    >>> T = rng.normal(size=(30, 2))
    >>> X = T @ rng.normal(size=(2, 6)) + 0.1 * rng.normal(size=(30, 6))
    >>> Y = T @ rng.normal(size=(2, 4)) + 0.1 * rng.normal(size=(30, 4))
    >>> model = O2PLS(n_components=2, n_x_orthogonal=1).fit(X, Y)
    >>> model.transform(X).shape
    (30, 2)
    >>> model.predict(X).shape
    (30, 4)
    """

    n_features_in_: int
    feature_names_in_: NDArray[np.object_]
    y_feature_names_in_: NDArray[np.object_]
    n_targets_: int
    x_mean_: NDArray[np.float64]
    x_std_: NDArray[np.float64]
    y_mean_: NDArray[np.float64]
    y_std_: NDArray[np.float64]
    _x_input_mean: NDArray[np.float64]
    _y_input_mean: NDArray[np.float64]
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
    singular_values_initial_: NDArray[np.float64]
    singular_values_final_: NDArray[np.float64]
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

    def fit(self, X: ArrayLike, y: ArrayLike) -> O2PLS:
        """Fit the O2PLS model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of predictors.
        y : array-like of shape (n_samples, n_targets)
            Target vectors, where `n_samples` is the number of samples and
            `n_targets` is the number of response variables.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        _reject_bool_param("n_components", self.n_components)
        _reject_bool_param("n_x_orthogonal", self.n_x_orthogonal)
        _reject_bool_param("n_y_orthogonal", self.n_y_orthogonal)
        self._validate_params()

        # Preserve whether the user supplied 1D or 2D Y so predict() can mirror the
        # fitted output convention after all internal work has used a 2D block.
        self._y_ndim = np.asarray(y).ndim
        y_names = _extract_feature_names(y)
        X_arr, Y_valid = _validate_xy(
            self,
            X,
            y,
            ensure_min_samples=2,
            copy=self.copy,
            reset=True,
            multi_output=True,
        )
        if y_names is not None:
            self.y_feature_names_in_ = y_names
        elif hasattr(self, "y_feature_names_in_"):
            del self.y_feature_names_in_

        Y2 = self._as_2d_target(Y_valid)
        self.n_targets_ = Y2.shape[1]
        # A univariate Y block has no separate Y-feature subspace, so keep v1 to the
        # single stable joint component and no Y-orthogonal extraction.
        if self.n_targets_ == 1:
            if self.n_components != 1:
                raise ValueError("O2PLS v1 requires n_components=1 for univariate Y.")
            if self.n_y_orthogonal != 0:
                raise ValueError("O2PLS v1 requires n_y_orthogonal=0 for univariate Y.")

        self.x_mean_, self.x_std_ = compute_scaling(X_arr, self.scale)
        self.y_mean_, self.y_std_ = compute_scaling(Y2, self.scale)
        Xs = apply_scaling(X_arr, self.x_mean_, self.x_std_)
        Ys = apply_scaling(Y2, self.y_mean_, self.y_std_)
        # Constant blocks make both joint and orthogonal directions undefined.
        if not _has_nonzero_variation(Xs, axis=0):
            raise ValueError("X has no non-zero variation after preprocessing.")
        if not _has_nonzero_variation(Ys, axis=0):
            raise ValueError("Y has no non-zero variation after preprocessing.")

        # O2PLS models covariation around the training means (matching OPLS and
        # PLSRegression). Every scaling mode except ``none`` already centers X and Y;
        # for ``none``, center internally while preserving the public identity
        # scaling vectors x_mean_/y_mean_ and filter_transform_{x,y} contract.
        if self.scale == "none":
            self._x_input_mean = Xs.mean(axis=0)
            self._y_input_mean = Ys.mean(axis=0)
            X_fit_input = Xs - self._x_input_mean
            Y_fit_input = Ys - self._y_input_mean
        else:
            self._x_input_mean = np.zeros(Xs.shape[1], dtype=np.float64)
            self._y_input_mean = np.zeros(Ys.shape[1], dtype=np.float64)
            X_fit_input = Xs
            Y_fit_input = Ys

        fit = o2pls_fit(
            X_fit_input,
            Y_fit_input,
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
        self.x_filtered_ = (
            fit.x_filtered + self._x_input_mean
            if self.scale == "none"
            else fit.x_filtered
        )
        self.y_filtered_ = (
            fit.y_filtered + self._y_input_mean
            if self.scale == "none"
            else fit.y_filtered
        )
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
        # In scaled/filtered coordinates, prediction is score projection
        # X @ W, score mapping B_T, then reconstruction through Y loadings.
        self.coef_filtered_ = (
            self.x_joint_weights_ @ self.b_t_ @ self.y_joint_loadings_.T
        )
        self.n_features_out_ = self.n_components_
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predict Y from X, reconstructing only the joint Y structure.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples, n_targets)
            Predicted values reconstructed from the joint X structure.
        """
        check_is_fitted(self)
        # Only X-specific structure is filtered when predicting Y from X; the Y-side
        # orthogonal model is used by predict_x() and transform_y().
        X_filtered = self.filter_transform_x(X)
        X_in = X_filtered - self._x_input_mean if self.scale == "none" else X_filtered
        y_scaled = X_in @ self.coef_filtered_
        if self.scale == "none":
            y_scaled = y_scaled + self._y_input_mean
        y_pred = self._unscale_y(y_scaled)
        return self._restore_y_shape(y_pred)

    def predict_x(self, Y: ArrayLike) -> NDArray[np.float64]:
        """Predict X from Y, reconstructing only the joint X structure.

        Parameters
        ----------
        Y : array-like of shape (n_samples, n_targets)
            Target samples.

        Returns
        -------
        x_pred : ndarray of shape (n_samples, n_features)
            Predicted X values reconstructed from the joint Y structure.
        """
        check_is_fitted(self)
        Y_filtered = self.filter_transform_y(Y)
        Y_in = Y_filtered - self._y_input_mean if self.scale == "none" else Y_filtered
        # Project filtered Y into joint score space, map to X scores, then rebuild X.
        U = Y_in @ self.y_joint_weights_
        T_pred = U @ self.b_u_
        x_scaled = T_pred @ self.x_joint_loadings_.T
        if self.scale == "none":
            x_scaled = x_scaled + self._x_input_mean
        return x_scaled * self.x_std_ + self.x_mean_

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return X-side joint scores after replaying the fitted X-orthogonal filter.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to transform.

        Returns
        -------
        X_scores : ndarray of shape (n_samples, n_components)
            Joint X scores on the filtered blocks.
        """
        check_is_fitted(self)
        X_filtered = self.filter_transform_x(X)
        X_in = X_filtered - self._x_input_mean if self.scale == "none" else X_filtered
        return X_in @ self.x_joint_weights_

    def transform_y(self, Y: ArrayLike) -> NDArray[np.float64]:
        """Return Y-side joint scores after replaying the fitted Y-orthogonal filter.

        Parameters
        ----------
        Y : array-like of shape (n_samples, n_targets)
            Targets to transform.

        Returns
        -------
        Y_scores : ndarray of shape (n_samples, n_components)
            Joint Y scores on the filtered blocks.
        """
        check_is_fitted(self)
        Y_filtered = self.filter_transform_y(Y)
        Y_in = Y_filtered - self._y_input_mean if self.scale == "none" else Y_filtered
        return Y_in @ self.y_joint_weights_

    def transform_pair(
        self, X: ArrayLike, Y: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(transform(X), transform_y(Y))``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to transform.
        Y : array-like of shape (n_samples, n_targets)
            Targets to transform.

        Returns
        -------
        X_scores : ndarray of shape (n_samples, n_components)
            Joint X scores on the filtered blocks.
        Y_scores : ndarray of shape (n_samples, n_components)
            Joint Y scores on the filtered blocks.
        """
        return self.transform(X), self.transform_y(Y)

    def transform_orthogonal_x(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return sequential X-specific orthogonal scores.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to transform.

        Returns
        -------
        X_orth_scores : ndarray of shape (n_samples, n_x_orthogonal)
            Orthogonal X scores.
        """
        check_is_fitted(self)
        return self._filter_x(X)[1]

    def transform_orthogonal_y(self, Y: ArrayLike) -> NDArray[np.float64]:
        """Return sequential Y-specific orthogonal scores.

        Parameters
        ----------
        Y : array-like of shape (n_samples, n_targets)
            Targets to transform.

        Returns
        -------
        Y_orth_scores : ndarray of shape (n_samples, n_y_orthogonal)
            Orthogonal Y scores.
        """
        check_is_fitted(self)
        return self._filter_y(Y)[1]

    def filter_transform_x(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return preprocessed X after the fitted X-orthogonal filter.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to filter.

        Returns
        -------
        X_filtered : ndarray of shape (n_samples, n_features)
            Filtered X block.
        """
        check_is_fitted(self)
        return self._filter_x(X)[0]

    def filter_transform_y(self, Y: ArrayLike) -> NDArray[np.float64]:
        """Return preprocessed Y after the fitted Y-orthogonal filter.

        Parameters
        ----------
        Y : array-like of shape (n_samples, n_targets)
            Targets to filter.

        Returns
        -------
        Y_filtered : ndarray of shape (n_samples, n_targets)
            Filtered Y block.
        """
        check_is_fitted(self)
        return self._filter_y(Y)[0]

    def get_feature_names_out(self, input_features=None) -> NDArray[np.object_]:
        """Output names for the joint-score columns.

        These are the columns returned by
        [`transform`][scikit_opls.O2PLS.transform].

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.

        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        check_is_fitted(self, "n_features_out_")
        _check_feature_names_in(self, input_features)
        return np.asarray(
            [f"o2pls_joint{i}" for i in range(self.n_features_out_)], dtype=object
        )

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
        y_cols = getattr(Y, "columns", None)
        fitted_y_names = getattr(self, "y_feature_names_in_", None)
        if fitted_y_names is not None:
            if y_cols is None:
                warnings.warn(
                    "Y does not have valid feature names, but O2PLS was fitted "
                    "with Y feature names.",
                    UserWarning,
                    stacklevel=3,
                )
            else:
                given = list(y_cols)
                expected = fitted_y_names.tolist()
                if given != expected:
                    raise ValueError(
                        "The feature names of Y should match those that were "
                        f"passed during fit. Expected {expected}, got {given}."
                    )
        elif y_cols is not None and _extract_feature_names(Y) is not None:
            warnings.warn(
                "Y has feature names, but O2PLS was fitted without Y feature names.",
                UserWarning,
                stacklevel=3,
            )

        # check_array is used directly because validate_data is tied to X feature
        # counts; Y-side public methods validate target-column counts separately.
        arr = _check_array_float64(
            Y,
            ensure_2d=False,
            ensure_min_samples=1,
            copy=False,
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
        X_valid = _validate_x(self, X, copy=False, reset=False)
        Xs = apply_scaling(X_valid, self.x_mean_, self.x_std_)
        X_in = Xs - self._x_input_mean if self.scale == "none" else Xs
        # Replay uses stored weights/loadings in fit order so new data is filtered
        # exactly like the training block.
        X_filt, t_ortho = _replay_orthogonal_filter(
            X_in, self.x_orthogonal_weights_, self.x_orthogonal_loadings_
        )
        if self.scale == "none":
            X_filt = X_filt + self._x_input_mean
        return X_filt, t_ortho

    def _filter_y(
        self, Y: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Scale and replay the fitted Y-orthogonal filter."""
        Y_valid = self._validate_y_block(Y)
        Ys = apply_scaling(Y_valid, self.y_mean_, self.y_std_)
        Y_in = Ys - self._y_input_mean if self.scale == "none" else Ys
        # This may be a no-op with zero columns, but keeping the same path preserves
        # shapes and score outputs for all fitted configurations.
        Y_filt, u_ortho = _replay_orthogonal_filter(
            Y_in, self.y_orthogonal_weights_, self.y_orthogonal_loadings_
        )
        if self.scale == "none":
            Y_filt = Y_filt + self._y_input_mean
        return Y_filt, u_ortho

    def _unscale_y(self, Ys: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map a scaled Y block back to raw units."""
        return Ys * self.y_std_ + self.y_mean_

    def _restore_y_shape(self, Y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return predictions with the fitted target dimensionality convention."""
        # sklearn regressors normally return 1D predictions when fitted on a 1D y.
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
