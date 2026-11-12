"""Orthogonal PLS (OPLS) regressor with a scikit-learn interface.

OPLS removes X-orthogonal variation with an OSC-style filter
(``scikit_opls._orthogonal``), then fits
[`PLSRegression`][sklearn.cross_decomposition.PLSRegression] on the filtered X block.
With ``n_orthogonal=0``, this reduces to ordinary PLS after this package's
selected X preprocessing.
"""

# reportAbstractUsage covers Interval/StrOptions: their base _Constraint declares
# __str__ abstract and the subclass override is not visible to static type checkers.
# pyright: reportAbstractUsage=false

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import (
    _check_feature_names_in,
    check_is_fitted,
)

from scikit_opls._inspection import (
    component_explained_x_variance,
    component_r2y_from_scores,
    explained_x_variance,
    orthogonal_vip,
    predictive_vip,
)
from scikit_opls._orthogonal import apply_orthogonal_filter, opls_filter
from scikit_opls._preprocessing import VALID_SCALING, apply_scaling, compute_scaling
from scikit_opls._utils import (
    _has_nonzero_variation,
    _reject_bool_param,
    _validate_x,
    _validate_xy,
)


@dataclass(frozen=True)
class _OPLSProjection:
    """All fitted OPLS model-space arrays for one validated raw X block."""

    Xs: NDArray[np.float64]
    X_filtered: NDArray[np.float64]
    t_pred: NDArray[np.float64]
    t_ortho: NDArray[np.float64]


def _apply_orthogonal_filter_left(
    vec: NDArray[np.float64],
    x_ortho_weights: NDArray[np.float64],
    x_ortho_loadings: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Apply the sequential orthogonal filter ``F`` from the left (``vec @ F``)."""
    out = np.asarray(vec, dtype=np.float64).copy()
    W = np.asarray(x_ortho_weights, dtype=np.float64)
    P = np.asarray(x_ortho_loadings, dtype=np.float64)
    for i in range(W.shape[1]):
        out = out - float(out @ W[:, i]) * P[:, i]
    return out


def _apply_orthogonal_filter_right(
    x_ortho_weights: NDArray[np.float64],
    x_ortho_loadings: NDArray[np.float64],
    mat: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Apply the sequential orthogonal filter ``F`` from the right (``F @ mat``)."""
    out = np.asarray(mat, dtype=np.float64).copy()
    W = np.asarray(x_ortho_weights, dtype=np.float64)
    P = np.asarray(x_ortho_loadings, dtype=np.float64)
    # Compose filters in reverse order so out == F @ mat without forming (p, p) F.
    for i in range(W.shape[1] - 1, -1, -1):
        out = out - np.outer(W[:, i], P[:, i] @ out)
    return out


def _compose_raw_coefficients(
    coef_filtered: NDArray[np.float64],
    intercept_filtered: float | NDArray[np.float64],
    x_mean: NDArray[np.float64],
    x_std: NDArray[np.float64],
    filter_offset: NDArray[np.float64],
    x_ortho_weights: NDArray[np.float64],
    x_ortho_loadings: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Collapse filtered/scaled-space PLS coefficients into raw-X coefficients.

    ``intercept_filtered`` must be the predictive engine's prediction at zero
    filtered input (``pls_.predict(0)``), not necessarily its public ``intercept_``.
    """
    coef_arr = np.asarray(coef_filtered, dtype=np.float64)
    if coef_arr.ndim == 1:
        coef_arr = coef_arr.reshape(1, -1)
    # sklearn PLSRegression exposes coef_ as (n_targets, n_features); work with the
    # transpose B as (n_features, n_targets).
    b_filtered = coef_arr.T

    # First collapse the sequential orthogonal filter into scaled feature space
    # in O(k_ortho * p) time without materializing a (p, p) matrix.
    b_scaled = _apply_orthogonal_filter_right(
        x_ortho_weights, x_ortho_loadings, b_filtered
    )

    inv_scale = 1.0 / np.asarray(x_std, dtype=np.float64)
    # Then fold in the original column scaling to obtain raw-X coefficients.
    b_raw = inv_scale[:, None] * b_scaled

    offset_scaled = np.asarray(x_mean, dtype=np.float64) * inv_scale
    intercept_raw = (
        np.asarray(intercept_filtered, dtype=np.float64)
        + np.asarray(filter_offset, dtype=np.float64) @ b_filtered
        - offset_scaled @ b_scaled
    )
    return b_raw.T, intercept_raw


class OPLS(RegressorMixin, TransformerMixin, BaseEstimator):
    """Orthogonal Projections to Latent Structures regression.

    Parameters
    ----------
    n_components : int, default=1
        Number of predictive PLS components fitted on the orthogonally filtered
        X block.
    n_orthogonal : int, default=1
        Number of X-orthogonal components removed before fitting the predictive
        PLS model. To choose this by cross-validated Q2, wrap ``OPLS`` in
        [`GridSearchCV`][sklearn.model_selection.GridSearchCV] over ``n_orthogonal``.
    scale : {"none", "center", "pareto", "standard"}, default="standard"
        Column preprocessing applied to ``X``. Note: unlike the boolean
        ``scale`` parameter of
        [`PLSRegression`][sklearn.cross_decomposition.PLSRegression], this is a string
        mode; passing ``True``/``False`` raises an error.
    copy : bool, default=True
        Whether the input arrays are copied during validation. Note that
        ``copy=False`` is passed to sklearn input validation; OPLS filtering
        still allocates working arrays.

    Attributes
    ----------
    n_orthogonal_ : int
        Number of orthogonal components actually used.
    x_ortho_weights_, x_ortho_loadings_, x_ortho_scores_ : ndarray
        Orthogonal weight/loading/score matrices.
    x_weights_, x_loadings_, x_scores_, y_loadings_ : ndarray
        Predictive model parameters taken from the underlying PLS engine (in the
        preprocessed, orthogonal-filtered space).
    coef_filtered_ : ndarray
        Coefficient matrix taken from the underlying PLS engine. Note: these
        coefficients act on the preprocessed, orthogonal-filtered space, and
        cannot be directly multiplied with raw input ``X``. Use ``predict(X)``
        for raw-input predictions.
    coef_raw_ : ndarray of shape (1, n_features)
        Linear coefficients on the original *raw* input feature space, collapsing the
        scaling, orthogonal filter and predictive PLS into one linear map.
        ``predict(X) == (X @ coef_raw_.T + intercept_raw_).ravel()`` up to
        floating-point tolerance. (No sklearn ``coef_`` alias is exposed.)
    intercept_ : float or ndarray
        Intercept of the underlying PLS model for predictions from the preprocessed,
        orthogonal-filtered X block to the original y scale.
    intercept_raw_ : float or ndarray
        Intercept paired with ``coef_raw_`` for prediction from raw input ``X``.
    pls_ : PLSRegression
        The fitted predictive engine.
    x_mean_, x_std_ : ndarray
        Centering/scaling vectors applied to ``X``.
    r2x_, r2x_ortho_, r2y_, rmse_ : float
        Training-set fit summaries. ``rmse_`` is the uncorrected training root mean
        squared error (no degrees-of-freedom correction). ``r2x_`` is computed from
        the predictive PLS
        scores/loadings on the filtered ``X`` block, relative to the preprocessed
        original ``X``. ``r2x_ortho_`` is computed from the removed orthogonal
        scores/loadings. These are diagnostic summaries, not a guaranteed exact
        additive partition; do not assume ``r2x_ + r2x_ortho_`` equals total
        explained ``X`` variance. For cross-validated Q2 use
        [`cross_val_score`][sklearn.model_selection.cross_val_score].
    r2x_components_ : ndarray of shape (n_components,)
        Per-component explained ``X`` sum-of-squares ratio of the predictive
        components, relative to the preprocessed ``X``.
    r2x_ortho_components_ : ndarray of shape (n_orthogonal_,)
        Per-component explained ``X`` sum-of-squares ratio of the removed
        orthogonal components.
    r2y_components_ : ndarray of shape (n_components,)
        Per-component ``y`` sum-of-squares ratio explained through
        ``t_i @ q_i.T``.
    q_residuals_train_ : ndarray of shape (n_samples,)
        Training Q residuals in the full (predictive + orthogonal)
        reconstruction space; equals ``q_residuals(X_train, space="full")``.
    q_residuals_predictive_train_ : ndarray of shape (n_samples,)
        Training Q residuals in the predictive-only reconstruction space.
    x_residual_ss_ : float
        Sum of ``q_residuals_train_``.
    y_residual_ss_ : float
        Training residual sum of squares of ``y`` against the fitted
        predictions.
    n_features_in_ : int
        Number of features seen during [`fit`][scikit_opls.OPLS.fit].
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during [`fit`][scikit_opls.OPLS.fit]. Defined only when
        ``X`` has
        feature names that are all strings.
    vip_, ortho_vip_ : ndarray of shape (n_features,)
        Lazy predictive / orthogonal Variable Importance in Projection scores,
        computed on first access (sklearn ``feature_importances_`` convention).
        For non-empty blocks with positive explained variance, each satisfies
        ``sum(vip**2) == n_features``. Empty or degenerate blocks return zeros.
        For ``n_components > 1``, predictive VIP aggregates across predictive
        PLS components.
        Use with [`SelectFromModel`][sklearn.feature_selection.SelectFromModel] via
        ``importance_getter="vip_"``.

    See Also
    --------
    OPLSDA : Binary OPLS discriminant analysis built on this regressor.
    O2PLS : Two-block variant that also models Y-specific orthogonal structure.
    sklearn.cross_decomposition.PLSRegression : Predictive engine fitted after
        orthogonal filtering; ``OPLS(n_orthogonal=0, scale="none")`` reduces to
        ``PLSRegression(scale=False)`` exactly.

    Notes
    -----
    Classic OPLS uses ``n_components=1``; ``n_orthogonal=0`` reduces to ordinary
    ``PLSRegression``, and ``n_components>1`` is orthogonal-filtered multi-component
    PLS (interpret score plots / S-plots component-wise).

    Constant and near-constant columns are retained rather than removed, preserving
    alignment with the input feature matrix, feature names, VIP arrays and
    ``coef_filtered_``. To drop them, prepend
    [`VarianceThreshold`][sklearn.feature_selection.VarianceThreshold] in a
    [`Pipeline`][sklearn.pipeline.Pipeline].

    References
    ----------
    .. [1] Trygg, J. & Wold, S. (2002). Orthogonal projections to latent
           structures (O-PLS). Journal of Chemometrics, 16(3), 119-128.
           https://doi.org/10.1002/cem.695
    .. [2] Wold, S., Antti, H., Lindgren, F. & Ohman, J. (1998). Orthogonal
           signal correction of near-infrared spectra. Chemometrics and
           Intelligent Laboratory Systems, 44(1-2), 175-185.
           https://doi.org/10.1016/S0169-7439(98)00109-9
    .. [3] Galindo-Prieto, B., Eriksson, L. & Trygg, J. (2014). Variable
           influence on projection (VIP) for OPLS models. Journal of
           Chemometrics, 28(8), 623-632. https://doi.org/10.1002/cem.2627

    Examples
    --------
    >>> import numpy as np
    >>> from scikit_opls import OPLS
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(20, 5))
    >>> y = X[:, 0] - X[:, 1] + rng.normal(scale=0.1, size=20)
    >>> model = OPLS(n_components=1, n_orthogonal=1).fit(X, y)
    >>> model.transform(X).shape
    (20, 1)
    >>> model.predict(X).shape
    (20,)
    """

    r2x_components_: NDArray[np.float64]
    r2x_ortho_components_: NDArray[np.float64]
    r2y_components_: NDArray[np.float64]
    x_residual_ss_: float
    y_residual_ss_: float
    q_residuals_train_: NDArray[np.float64]
    q_residuals_predictive_train_: NDArray[np.float64]

    n_features_in_: int
    feature_names_in_: NDArray[np.str_]
    n_orthogonal_: int
    x_mean_: NDArray[np.float64]
    x_std_: NDArray[np.float64]
    _filter_input_mean: NDArray[np.float64]
    _pls_x_mean: NDArray[np.float64]
    _score_dist_cache: dict[
        str, tuple[NDArray[np.float64], float | NDArray[np.float64]]
    ]
    x_ortho_weights_: NDArray[np.float64]
    x_ortho_loadings_: NDArray[np.float64]
    x_ortho_scores_: NDArray[np.float64]
    x_weights_: NDArray[np.float64]
    x_loadings_: NDArray[np.float64]
    x_scores_: NDArray[np.float64]
    y_loadings_: NDArray[np.float64]
    coef_filtered_: NDArray[np.float64]
    coef_raw_: NDArray[np.float64]
    intercept_: float | NDArray[np.float64]
    intercept_raw_: float | NDArray[np.float64]
    pls_: PLSRegression
    r2x_: float
    r2x_ortho_: float
    r2y_: float
    rmse_: float
    _n_features_out: int

    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "n_orthogonal": [Interval(Integral, 0, None, closed="left")],
        "scale": [StrOptions(set(VALID_SCALING))],
        "copy": ["boolean"],
    }

    def __init__(
        self,
        n_components: int = 1,
        n_orthogonal: int = 1,
        scale: str = "standard",
        copy: bool = True,
    ) -> None:
        self.n_components = n_components
        self.n_orthogonal = n_orthogonal
        self.scale = scale
        self.copy = copy

    def fit(self, X: ArrayLike, y: ArrayLike) -> OPLS:
        """Fit the OPLS model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training predictors.
        y : array-like of shape (n_samples,)
            Target values. Required: ``OPLS`` is a supervised transformer.

        Returns
        -------
        self : OPLS
            The fitted estimator.
        """
        self._clear_fit_caches()
        X_arr, y_arr = self._validate_fit_data(X, y)
        Xs, X_filtered = self._fit_orthogonal_filter(X_arr, y_arr)
        self._fit_predictive_engine(X_filtered, y_arr)
        self._set_raw_coefficients(X_filtered)
        self._set_fit_diagnostics(Xs, X_filtered, y_arr)
        return self

    def _clear_fit_caches(self) -> None:
        for attr in ("_vip_", "_ortho_vip_", "_score_dist_cache"):
            self.__dict__.pop(attr, None)

    def _validate_fit_data(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        _reject_bool_param("n_components", self.n_components)
        _reject_bool_param("n_orthogonal", self.n_orthogonal)
        self._validate_params()
        X_arr, y_raw = _validate_xy(
            self,
            X,
            y,
            ensure_min_samples=2,
            copy=self.copy,
            reset=True,
            multi_output=False,
        )
        y_arr = np.asarray(y_raw, dtype=np.float64)
        if not _has_nonzero_variation(y_arr):
            raise ValueError("OPLS requires a non-constant target y.")

        if self.n_components > min(X_arr.shape):
            raise ValueError(
                f"n_components={self.n_components} exceeds the maximum of "
                f"min(n_samples, n_features)={min(X_arr.shape)}."
            )
        return X_arr, y_arr

    def _fit_orthogonal_filter(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        self.x_mean_, self.x_std_ = compute_scaling(X, self.scale)
        Xs = apply_scaling(X, self.x_mean_, self.x_std_)
        if not _has_nonzero_variation(Xs, axis=0):
            raise ValueError("X has no non-zero variation after preprocessing.")
        # OPLS extracts variation around the training mean. Every scaling mode
        # except ``none`` already centers X; for ``none``, keep the public
        # preprocessing contract (identity) while centering internally so
        # arbitrary feature offsets never alter the orthogonal model or R2X.
        if self.scale == "none":
            self._filter_input_mean = Xs.mean(axis=0)
            X_filter_input = Xs - self._filter_input_mean
        else:
            self._filter_input_mean = np.zeros(Xs.shape[1], dtype=np.float64)
            X_filter_input = Xs
        ofit = opls_filter(X_filter_input, y - y.mean(), self.n_orthogonal)
        self.x_ortho_weights_ = ofit.x_ortho_weights
        self.x_ortho_loadings_ = ofit.x_ortho_loadings
        self.x_ortho_scores_ = ofit.x_ortho_scores
        self.n_orthogonal_ = ofit.n_components
        if not _has_nonzero_variation(ofit.x_filtered, axis=0):
            raise ValueError(
                "X has no remaining variation after orthogonal filtering; "
                "reduce n_orthogonal."
            )
        # Restore the preprocessing-space mean for ``none``. PLSRegression centers
        # this block internally, and filter_transform() remains an identity when no
        # orthogonal components are requested.
        X_filtered = (
            ofit.x_filtered + self._filter_input_mean
            if self.scale == "none"
            else ofit.x_filtered
        )
        return Xs, X_filtered

    def _fit_predictive_engine(
        self,
        X_filtered: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> None:
        if not _has_nonzero_variation(X_filtered, axis=0):
            raise ValueError(
                "X has no remaining variation after orthogonal filtering; "
                "reduce n_orthogonal."
            )
        # Non-zero centered variation above already guarantees numerical rank >= 1;
        # only incur a full matrix_rank SVD when multiple predictive components are
        # requested. PLSRegression always centers X even when scale=False.
        if self.n_components > 1:
            rank_filtered = int(
                np.linalg.matrix_rank(
                    X_filtered - X_filtered.mean(axis=0, keepdims=True)
                )
            )
            if self.n_components > rank_filtered:
                raise ValueError(
                    f"n_components={self.n_components} exceeds the numerical rank of "
                    f"X after orthogonal filtering ({rank_filtered}). "
                    "Reduce n_components or n_orthogonal."
                )
        self.pls_ = PLSRegression(n_components=self.n_components, scale=False)
        self.pls_.fit(X_filtered, y)
        self._n_features_out = self.n_components
        self.x_weights_ = np.asarray(self.pls_.x_weights_, dtype=np.float64)
        self.x_loadings_ = np.asarray(self.pls_.x_loadings_, dtype=np.float64)
        self.x_scores_ = np.asarray(self.pls_.x_scores_, dtype=np.float64)
        self.y_loadings_ = np.asarray(self.pls_.y_loadings_, dtype=np.float64)
        self.coef_filtered_ = np.asarray(self.pls_.coef_, dtype=np.float64)
        self.intercept_ = np.asarray(self.pls_.intercept_, dtype=np.float64)
        # PLSRegression(scale=False) centers X_filtered by X_filtered.mean(axis=0);
        # compute and store that mean directly without relying on private _x_mean.
        self._pls_x_mean = X_filtered.mean(axis=0)

    def _set_raw_coefficients(self, X_filtered: NDArray[np.float64]) -> None:
        engine_offset = np.asarray(
            self.pls_.predict(np.zeros((1, X_filtered.shape[1]), dtype=np.float64)),
            dtype=np.float64,
        ).ravel()
        if self.scale == "none" and self.n_orthogonal_ > 0:
            filter_offset = self._filter_input_mean - _apply_orthogonal_filter_left(
                self._filter_input_mean,
                self.x_ortho_weights_,
                self.x_ortho_loadings_,
            )
        else:
            filter_offset = np.zeros(X_filtered.shape[1], dtype=np.float64)
        self.coef_raw_, self.intercept_raw_ = _compose_raw_coefficients(
            self.coef_filtered_,
            engine_offset,
            self.x_mean_,
            self.x_std_,
            filter_offset,
            self.x_ortho_weights_,
            self.x_ortho_loadings_,
        )

    def _set_fit_diagnostics(
        self,
        Xs: NDArray[np.float64],
        X_filtered: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> None:
        y_fit = np.asarray(self.pls_.predict(X_filtered), dtype=np.float64)
        X_centered = Xs - self._filter_input_mean if self.scale == "none" else Xs
        self.r2x_ = explained_x_variance(X_centered, self.x_scores_, self.x_loadings_)
        self.r2x_ortho_ = explained_x_variance(
            X_centered, self.x_ortho_scores_, self.x_ortho_loadings_
        )
        self.r2y_ = float(r2_score(y, y_fit))
        self.rmse_ = float(root_mean_squared_error(y, y_fit))

        self.r2x_components_ = component_explained_x_variance(
            X_centered,
            self.x_scores_,
            self.x_loadings_,
        )
        self.r2x_ortho_components_ = component_explained_x_variance(
            X_centered,
            self.x_ortho_scores_,
            self.x_ortho_loadings_,
        )
        self.r2y_components_ = component_r2y_from_scores(
            y,
            self.x_scores_,
            self.y_loadings_,
        )

        proj = _OPLSProjection(
            Xs=Xs,
            X_filtered=X_filtered,
            t_pred=self.x_scores_,
            t_ortho=self.x_ortho_scores_,
        )
        self.q_residuals_train_ = self._q_residuals_from_projection(proj, space="full")
        self.x_residual_ss_ = float(np.sum(self.q_residuals_train_))

        self.q_residuals_predictive_train_ = self._q_residuals_from_projection(
            proj, space="predictive"
        )

        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim == 1 and y_fit.ndim == 2:
            y_arr = y_arr.reshape(-1, 1)
        elif y_arr.ndim == 2 and y_fit.ndim == 1:
            y_fit = y_fit.reshape(-1, 1)
        self.y_residual_ss_ = float(np.sum((y_arr - y_fit) ** 2))
        self._score_dist_cache = {}

    def _filter_validated(
        self, X_valid: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Scale and replay the fitted orthogonal filter without PLS projection."""
        Xs = apply_scaling(X_valid, self.x_mean_, self.x_std_)
        X_filter_input = Xs - self._filter_input_mean if self.scale == "none" else Xs
        X_filtered, t_ortho = apply_orthogonal_filter(
            X_filter_input,
            self.x_ortho_weights_,
            self.x_ortho_loadings_,
        )
        if self.scale == "none":
            X_filtered = X_filtered + self._filter_input_mean
        return Xs, X_filtered, t_ortho

    def _project_validated(self, X_valid: NDArray[np.float64]) -> _OPLSProjection:
        """Project already validated raw X into fitted OPLS model spaces."""
        Xs, X_filtered, t_ortho = self._filter_validated(X_valid)
        t_pred = np.asarray(self.pls_.transform(X_filtered), dtype=np.float64)
        return _OPLSProjection(
            Xs=Xs,
            X_filtered=X_filtered,
            t_pred=t_pred,
            t_ortho=t_ortho,
        )

    def _predictive_x_hat(self, t_pred: NDArray[np.float64]) -> NDArray[np.float64]:
        """Reconstruct scaled X from predictive scores."""
        return self._pls_x_mean + t_pred @ self.x_loadings_.T

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predict ``y`` for new samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted target values.
        """
        X_valid = self._validate_X_predict(X)
        return np.asarray(
            X_valid @ self.coef_raw_.T + self.intercept_raw_, dtype=np.float64
        ).ravel()

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Project samples onto the predictive components.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to project (orthogonal-filtered first, as at fit time).

        Returns
        -------
        x_scores : ndarray of shape (n_samples, n_components)
            Predictive scores.
        """
        X_valid = self._validate_X_predict(X)
        return self._project_validated(X_valid).t_pred

    def transform_orthogonal(self, X: ArrayLike) -> NDArray[np.float64]:
        """Project samples onto the orthogonal components.

        This is a non-standard method (outside the ``set_output`` contract).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to project.

        Returns
        -------
        x_ortho_scores : ndarray of shape (n_samples, n_orthogonal_)
            Orthogonal scores.
        """
        X_valid = self._validate_X_predict(X)
        return self._filter_validated(X_valid)[2]

    def filter_transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Return ``X`` after preprocessing and orthogonal filtering.

        This is the matrix actually passed to the predictive PLS engine, so
        ``self.pls_.predict(self.filter_transform(X))`` matches ``self.predict(X)``
        (up to output shape). The result is in the preprocessed, orthogonal-filtered
        space, **not** on the raw input scale. With ``n_orthogonal=0`` it is just the
        preprocessed ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to preprocess and filter.

        Returns
        -------
        X_filtered : ndarray of shape (n_samples, n_features)
            Preprocessed ``X`` with the fitted orthogonal variation removed.
        """
        X_valid = self._validate_X_predict(X)
        return self._filter_validated(X_valid)[1]

    @property
    def vip_(self) -> NDArray[np.float64]:
        """Predictive VIP per feature; ndarray (n_features,).

        Standard PLS Variable Importance in Projection computed from the predictive
        model fitted on the orthogonally filtered ``X``, normalised so
        ``sum(vip_**2) == n_features``. Computed lazily on first access from the
        fitted weights. Defined in the style of Galindo-Prieto et al. (2014); not
        intended to reproduce ropls VIP values exactly.
        """
        check_is_fitted(self)
        if not hasattr(self, "_vip_"):
            # Cache on first access; fit() clears this attribute before refitting.
            self._vip_ = predictive_vip(
                self.x_weights_, self.x_scores_, self.y_loadings_
            )
        return self._vip_

    @property
    def ortho_vip_(self) -> NDArray[np.float64]:
        """Orthogonal VIP per feature; ndarray (n_features,).

        Computed lazily on first access from the fitted weights.
        """
        check_is_fitted(self)
        if not hasattr(self, "_ortho_vip_"):
            # Cache on first access; fit() clears this attribute before refitting.
            self._ortho_vip_ = orthogonal_vip(
                self.x_ortho_weights_, self.x_ortho_scores_, self.x_ortho_loadings_
            )
        return self._ortho_vip_

    def get_feature_names_out(self, input_features=None) -> NDArray[np.object_]:
        """Output feature names for the predictive scores.

        [`transform`][scikit_opls.OPLS.transform] reduces ``X`` to
        ``n_components`` predictive scores, so the
        output columns are components, not input features. They are named
        ``opls_pred0, opls_pred1, …`` (the ``ClassNamePrefixFeaturesOutMixin``
        convention), independent of the input feature names. ``transform_orthogonal``
        is outside the ``set_output`` contract and has no names here.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input feature names; only validated for length against
            ``n_features_in_`` (the output names do not depend on them).

        Returns
        -------
        feature_names_out : ndarray of str of shape (n_components,)
            Names of the predictive-score columns.
        """
        check_is_fitted(self, "_n_features_out")
        _check_feature_names_in(self, input_features)
        return np.asarray(
            [f"opls_pred{i}" for i in range(self._n_features_out)], dtype=object
        )

    def _validate_X_predict(self, X: ArrayLike) -> NDArray[np.float64]:  # noqa: N802
        """Validate prediction/projection input against fitted OPLS metadata."""
        check_is_fitted(self)
        return _validate_x(self, X, copy=False, reset=False)

    def _compute_score_dist_stats(
        self, reference_scores: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], float | NDArray[np.float64]]:
        """Compute training score center and inverse dispersion for Mahalanobis dist."""
        T_ref = np.asarray(reference_scores, dtype=np.float64)
        if T_ref.ndim == 1:
            T_ref = T_ref.reshape(-1, 1)
        center = T_ref.mean(axis=0, keepdims=True)
        T_ref_centered = T_ref - center
        if T_ref.shape[1] == 1:
            var = float(np.var(T_ref_centered[:, 0], ddof=1))
            return center, max(var, float(np.finfo(np.float64).eps))
        cov = np.cov(T_ref_centered, rowvar=False)
        inv_cov = np.asarray(np.linalg.pinv(cov), dtype=np.float64)
        return center, inv_cov

    @staticmethod
    def _apply_score_dist_stats(
        scores: NDArray[np.float64],
        center: NDArray[np.float64],
        dispersion: float | NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Evaluate Mahalanobis-like squared distance from cached training stats."""
        T = np.asarray(scores, dtype=np.float64)
        if T.ndim == 1:
            T = T.reshape(-1, 1)
        T_centered = T - center
        if isinstance(dispersion, float):
            return (T_centered[:, 0] ** 2) / dispersion
        return np.sum((T_centered @ dispersion) * T_centered, axis=1)

    def score_distance(
        self,
        X: ArrayLike,
        *,
        kind: str = "predictive",
    ) -> NDArray[np.float64]:
        """Return Hotelling-like score distances for samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples in the same raw feature space used for fitting. Pass raw X.
            Do not manually center or scale before calling diagnostics. The estimator
            applies its fitted preprocessing internally.
        kind : {"predictive", "orthogonal", "all"}, default="predictive"
            Which latent score space to use.

        Returns
        -------
        distance : ndarray of shape (n_samples,)
            Squared Mahalanobis-like distance in the selected fitted score space.
        """
        X_valid = self._validate_X_predict(X)
        return self._score_distance_validated(X_valid, kind=kind)

    def _score_distance_validated(
        self,
        X_valid: NDArray[np.float64],
        kind: str = "predictive",
    ) -> NDArray[np.float64]:
        """Compute score distance internally, assuming X is already validated."""
        if kind not in {"predictive", "orthogonal", "all"}:
            raise ValueError("kind must be one of {'predictive', 'orthogonal', 'all'}.")
        if kind == "orthogonal":
            if self.n_orthogonal_ == 0:
                return np.zeros(X_valid.shape[0], dtype=np.float64)
            scores = self._filter_validated(X_valid)[2]
            reference = self.x_ortho_scores_
        else:
            proj = self._project_validated(X_valid)
            if kind == "predictive" or self.n_orthogonal_ == 0:
                scores = proj.t_pred
                reference = self.x_scores_
                kind = "predictive"
            else:
                scores = np.hstack([proj.t_pred, proj.t_ortho])
                reference = np.hstack([self.x_scores_, self.x_ortho_scores_])

        cache = getattr(self, "_score_dist_cache", None)
        if cache is None:
            cache = {}
            self._score_dist_cache = cache
        stats = cache.get(kind)
        if stats is None:
            stats = self._compute_score_dist_stats(reference)
            cache[kind] = stats
        return self._apply_score_dist_stats(scores, stats[0], stats[1])

    def q_residuals(
        self,
        X: ArrayLike,
        *,
        space: str = "full",
    ) -> NDArray[np.float64]:
        """Return Q residuals, i.e. squared X residual norm per sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples in raw feature space. Pass raw X. Do not manually center or scale
            before calling diagnostics. The estimator applies its fitted preprocessing
            internally.
        space : {"full", "predictive"}, default="full"
            Which model reconstruction space to use:

            - ``"full"`` reconstructs scaled X from predictive + orthogonal structure.
            - ``"predictive"`` reconstructs scaled X from predictive PLS structure
              only, treating orthogonal variation as part of the residual.

        Returns
        -------
        q : ndarray of shape (n_samples,)
            Squared residual norm per sample.
        """
        X_valid = self._validate_X_predict(X)
        return self._q_residuals_validated(X_valid, space=space)

    def _q_residuals_validated(
        self,
        X_valid: NDArray[np.float64],
        space: str = "full",
    ) -> NDArray[np.float64]:
        """Compute Q residuals internally, assuming X is already validated."""
        if space not in {"full", "predictive"}:
            raise ValueError("space must be one of {'full', 'predictive'}.")
        return self._q_residuals_from_projection(
            self._project_validated(X_valid), space=space
        )

    def _q_residuals_from_projection(
        self,
        proj: _OPLSProjection,
        *,
        space: str,
    ) -> NDArray[np.float64]:
        """Compute Q residuals from an existing OPLS projection."""
        X_pred_hat = self._predictive_x_hat(proj.t_pred)
        if space == "predictive":
            X_hat = X_pred_hat
        elif space == "full":
            if self.n_orthogonal_ > 0:
                X_ortho_hat = proj.t_ortho @ self.x_ortho_loadings_.T
                X_hat = X_ortho_hat + X_pred_hat
            else:
                X_hat = X_pred_hat
        else:
            raise ValueError("space must be one of {'full', 'predictive'}.")
        resid = proj.Xs - X_hat
        return np.sum(resid**2, axis=1)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.regressor_tags.poor_score = True
        # OPLS is a supervised transformer: fit requires y.
        tags.target_tags.required = True
        # The OSC-style orthogonal filter densifies; sparse input is unsupported.
        tags.input_tags.sparse = False
        # Deterministic for a fixed (non-shuffled) cv.
        tags.non_deterministic = False
        return tags
