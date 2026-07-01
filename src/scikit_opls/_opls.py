"""Orthogonal PLS (OPLS) regressor with a scikit-learn interface.

OPLS (Trygg & Wold, 2002) splits the variation in ``X`` into a *predictive* part
correlated with ``y`` and *orthogonal* parts uncorrelated with ``y``. This
estimator removes the orthogonal variation with an OSC-style orthogonal filter
(:mod:`scikit_opls._orthogonal`) and then fits
:class:`sklearn.cross_decomposition.PLSRegression` on the cleaned ``X`` as the
predictive engine. With ``n_orthogonal=0``, this becomes ordinary
``PLSRegression`` after the package's selected X preprocessing.
"""

# scikit-learn's validate_data uses sentinel-string parameter defaults that lead
# static type checkers to flag every downstream array op as a false positive.
# Suppress those categories here; the test suite and ``check_estimator`` are the
# real correctness gate. reportAbstractUsage covers Interval/StrOptions: their
# base _Constraint declares __str__ abstract and the subclass override is not
# visible to the type checker, so instantiating them looks abstract (it is not).
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportReturnType=false
# pyright: reportAbstractUsage=false
# reportIncompatibleMethodOverride: our score() narrows X to ArrayLike (no sparse)
# vs RegressorMixin's MatrixLike; OPLS rejects sparse anyway (input_tags.sparse=False).
# pyright: reportIncompatibleMethodOverride=false

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
    validate_data,
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
from scikit_opls._utils import _has_nonzero_variation, _reject_bool_param


@dataclass(frozen=True)
class _OPLSProjection:
    """All fitted OPLS model-space arrays for one validated raw X block."""

    Xs: NDArray[np.float64]
    X_filtered: NDArray[np.float64]
    t_pred: NDArray[np.float64]
    t_ortho: NDArray[np.float64]


def _orthogonal_filter_matrix(
    x_ortho_weights: NDArray[np.float64],
    x_ortho_loadings: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Right-side linear operator ``F`` such that ``X_filtered == X_scaled @ F``.

    The replayed orthogonal filter applies ``X <- X - (X w_i) p_iᵀ`` for each
    component, i.e. right multiplication by ``I - outer(w_i, p_i)``. Composing them
    in order yields the single matrix equivalent of :func:`apply_orthogonal_filter`.
    """
    W = np.asarray(x_ortho_weights, dtype=np.float64)
    P = np.asarray(x_ortho_loadings, dtype=np.float64)
    n_features = W.shape[0]
    eye = np.eye(n_features, dtype=np.float64)
    F = eye.copy()
    for i in range(W.shape[1]):
        # Compose filters in the same order they were fitted and replayed.
        F = F @ (eye - np.outer(W[:, i], P[:, i]))
    return F


def _compose_raw_coefficients(
    coef_filtered: NDArray[np.float64],
    intercept_filtered: float | NDArray[np.float64],
    x_mean: NDArray[np.float64],
    x_std: NDArray[np.float64],
    x_ortho_weights: NDArray[np.float64],
    x_ortho_loadings: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Collapse filtered/scaled-space PLS coefficients into raw-X coefficients.

    The fitted prediction is linear: ``X -> (X - mean) / std -> @ F -> @ Bᶠ + b``,
    where ``b`` is the predictive engine's prediction offset (``pls_.predict(0)``,
    not ``pls_.intercept_``). With ``B_scaled = F @ Bᶠ`` and ``inv_scale = 1 / std``
    this reduces to ``y = X @ B_raw + (b - (mean * inv_scale) @ B_scaled)`` where
    ``B_raw = inv_scale[:, None] * B_scaled``.
    """
    coef_arr = np.asarray(coef_filtered, dtype=np.float64)
    if coef_arr.ndim == 1:
        coef_arr = coef_arr.reshape(1, -1)
    # sklearn PLSRegression exposes coef_ as (n_targets, n_features); work with the
    # transpose B as (n_features, n_targets).
    b_filtered = coef_arr.T

    f_matrix = _orthogonal_filter_matrix(x_ortho_weights, x_ortho_loadings)
    # First collapse the sequential orthogonal filter into scaled feature space.
    b_scaled = f_matrix @ b_filtered

    inv_scale = 1.0 / np.asarray(x_std, dtype=np.float64)
    # Then fold in the original column scaling to obtain raw-X coefficients.
    b_raw = inv_scale[:, None] * b_scaled

    offset_scaled = np.asarray(x_mean, dtype=np.float64) * inv_scale
    intercept_raw = np.asarray(intercept_filtered, dtype=np.float64) - (
        offset_scaled @ b_scaled
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
        :class:`~sklearn.model_selection.GridSearchCV` over ``n_orthogonal``.
    scale : {"none", "center", "pareto", "standard"}, default="standard"
        Column preprocessing applied to ``X``.
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
        :func:`sklearn.model_selection.cross_val_score`.
    vip_, ortho_vip_ : ndarray of shape (n_features,)
        Lazy predictive / orthogonal Variable Importance in Projection scores,
        computed on first access (sklearn ``feature_importances_`` convention).
        For non-empty blocks with positive explained variance, each satisfies
        ``sum(vip**2) == n_features``. Empty or degenerate blocks return zeros.
        For ``n_components > 1``, predictive VIP aggregates across predictive
        PLS components.
        Use with :class:`~sklearn.feature_selection.SelectFromModel` via
        ``importance_getter="vip_"``.

    Notes
    -----
    Classic OPLS uses ``n_components=1``; ``n_orthogonal=0`` reduces to ordinary
    ``PLSRegression``, and ``n_components>1`` is orthogonal-filtered multi-component
    PLS (interpret score plots / S-plots component-wise).

    Constant and near-constant columns are retained rather than removed, preserving
    alignment with the input feature matrix, feature names, VIP arrays and
    ``coef_filtered_``. To drop them, prepend
    :class:`~sklearn.feature_selection.VarianceThreshold` in a
    :class:`~sklearn.pipeline.Pipeline`.
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
        X, y = self._validate_fit_data(X, y)
        Xs, X_filtered = self._fit_orthogonal_filter(X, y)
        self._fit_predictive_engine(X_filtered, y)
        self._set_raw_coefficients(X_filtered)
        self._set_fit_diagnostics(X, Xs, X_filtered, y)
        return self

    def _clear_fit_caches(self) -> None:
        for attr in ("_vip_", "_ortho_vip_"):
            self.__dict__.pop(attr, None)

    def _validate_fit_data(
        self,
        X: ArrayLike,
        y: ArrayLike,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        _reject_bool_param("n_components", self.n_components)
        _reject_bool_param("n_orthogonal", self.n_orthogonal)
        self._validate_params()
        X, y = validate_data(
            self,
            X,
            y,
            dtype=np.float64,
            ensure_min_samples=2,
            copy=self.copy,
        )
        if not _has_nonzero_variation(y):
            raise ValueError("OPLS requires a non-constant target y.")

        if self.n_components > min(X.shape):
            raise ValueError(
                f"n_components={self.n_components} exceeds the maximum of "
                f"min(n_samples, n_features)={min(X.shape)}."
            )
        return X, y

    def _fit_orthogonal_filter(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        self.x_mean_, self.x_std_ = compute_scaling(X, self.scale)
        Xs = apply_scaling(X, self.x_mean_, self.x_std_)
        if not _has_nonzero_variation(Xs, axis=0):
            raise ValueError("X has no non-zero variation after preprocessing.")
        ofit = opls_filter(Xs, y - y.mean(), self.n_orthogonal)
        self.x_ortho_weights_ = ofit.x_ortho_weights
        self.x_ortho_loadings_ = ofit.x_ortho_loadings
        self.x_ortho_scores_ = ofit.x_ortho_scores
        self.n_orthogonal_ = ofit.n_components
        return Xs, ofit.x_filtered

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
        rank_filtered = np.linalg.matrix_rank(X_filtered)
        if self.n_components > rank_filtered:
            raise ValueError(
                f"n_components={self.n_components} exceeds the numerical rank of "
                f"X after orthogonal filtering ({rank_filtered}). "
                "Reduce n_components or n_orthogonal."
            )
        self.pls_ = PLSRegression(n_components=self.n_components, scale=False)
        self.pls_.fit(X_filtered, y)
        self._n_features_out = self.n_components
        self.x_weights_ = self.pls_.x_weights_
        self.x_loadings_ = self.pls_.x_loadings_
        self.x_scores_ = self.pls_.x_scores_
        self.y_loadings_ = self.pls_.y_loadings_
        self.coef_filtered_ = self.pls_.coef_
        self.intercept_ = self.pls_.intercept_
        self._pls_x_mean_ = np.asarray(self.pls_._x_mean, dtype=np.float64)

    def _set_raw_coefficients(self, X_filtered: NDArray[np.float64]) -> None:
        engine_offset = self.pls_.predict(
            np.zeros((1, X_filtered.shape[1]), dtype=np.float64)
        ).ravel()
        self.coef_raw_, self.intercept_raw_ = _compose_raw_coefficients(
            self.coef_filtered_,
            engine_offset,
            self.x_mean_,
            self.x_std_,
            self.x_ortho_weights_,
            self.x_ortho_loadings_,
        )

    def _set_fit_diagnostics(
        self,
        X: NDArray[np.float64],
        Xs: NDArray[np.float64],
        X_filtered: NDArray[np.float64],
        y: NDArray[np.float64],
    ) -> None:
        y_fit = self.pls_.predict(X_filtered)
        self.r2x_ = explained_x_variance(Xs, self.x_scores_, self.x_loadings_)
        self.r2x_ortho_ = explained_x_variance(
            Xs, self.x_ortho_scores_, self.x_ortho_loadings_
        )
        self.r2y_ = float(r2_score(y, y_fit))
        self.rmse_ = float(root_mean_squared_error(y, y_fit))

        self.r2x_components_ = component_explained_x_variance(
            Xs,
            self.x_scores_,
            self.x_loadings_,
        )
        self.r2x_ortho_components_ = component_explained_x_variance(
            Xs,
            self.x_ortho_scores_,
            self.x_ortho_loadings_,
        )
        self.r2y_components_ = component_r2y_from_scores(
            y,
            self.x_scores_,
            self.y_loadings_,
        )

        proj = self._project_validated(X)
        self.q_residuals_train_ = self._q_residuals_from_projection(proj, space="full")
        self.x_residual_ss_ = float(np.sum(self.q_residuals_train_))

        self.q_residuals_predictive_train_ = self._q_residuals_from_projection(
            proj, space="predictive"
        )

        y_fit_arr = np.asarray(y_fit, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        self.y_residual_ss_ = float(np.sum((y_arr - y_fit_arr) ** 2))

    def _project_validated(self, X_valid: NDArray[np.float64]) -> _OPLSProjection:
        """Project already validated raw X into fitted OPLS model spaces."""
        Xs = apply_scaling(X_valid, self.x_mean_, self.x_std_)
        X_filtered, t_ortho = apply_orthogonal_filter(
            Xs,
            self.x_ortho_weights_,
            self.x_ortho_loadings_,
        )
        t_pred = self.pls_.transform(X_filtered)
        return _OPLSProjection(
            Xs=Xs,
            X_filtered=X_filtered,
            t_pred=t_pred,
            t_ortho=t_ortho,
        )

    def _predictive_x_hat(self, t_pred: NDArray[np.float64]) -> NDArray[np.float64]:
        """Reconstruct scaled X from predictive scores."""
        return self._pls_x_mean_ + t_pred @ self.x_loadings_.T

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
        proj = self._project_validated(X_valid)
        return self.pls_.predict(proj.X_filtered).ravel()

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
        return self._project_validated(X_valid).t_ortho

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
        return self._project_validated(X_valid).X_filtered

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
        """Output feature names for :meth:`transform` (the predictive scores).

        ``transform`` reduces ``X`` to ``n_components`` predictive scores, so the
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

    def score(self, X: ArrayLike, y: ArrayLike, sample_weight=None) -> float:
        """Coefficient of determination R² of the prediction.

        Inherited from :class:`~sklearn.base.RegressorMixin` (``OPLS`` is also a
        :class:`~sklearn.base.TransformerMixin`; the regression ``score`` applies,
        not a transformer score).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True target values for ``X``.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            R² of ``self.predict(X)`` against ``y``.
        """
        return super().score(X, y, sample_weight)

    def _validate_X_predict(self, X: ArrayLike) -> NDArray[np.float64]:  # noqa: N802
        """Validate prediction/projection input against fitted OPLS metadata."""
        check_is_fitted(self)
        return validate_data(
            self,
            X,
            dtype=np.float64,
            copy=self.copy,
            reset=False,
        )

    def _score_distance_from_scores(
        self,
        scores: NDArray[np.float64],
        reference_scores: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Mahalanobis-like distance in latent score space."""
        T = np.asarray(scores, dtype=np.float64)
        T_ref = np.asarray(reference_scores, dtype=np.float64)
        if T.ndim == 1:
            T = T.reshape(-1, 1)
        if T_ref.ndim == 1:
            T_ref = T_ref.reshape(-1, 1)
        center = T_ref.mean(axis=0, keepdims=True)
        T_centered = T - center
        T_ref_centered = T_ref - center
        if T_ref.shape[1] == 1:
            var = float(np.var(T_ref_centered[:, 0], ddof=1))
            var = max(var, np.finfo(np.float64).eps)
            return (T_centered[:, 0] ** 2) / var
        cov = np.cov(T_ref_centered, rowvar=False)
        inv_cov = np.linalg.pinv(cov)
        return np.sum((T_centered @ inv_cov) * T_centered, axis=1)

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
        proj = self._project_validated(X_valid)
        if kind == "predictive":
            scores = proj.t_pred
            reference = self.x_scores_
        elif kind == "orthogonal":
            if self.n_orthogonal_ == 0:
                return np.zeros(X_valid.shape[0], dtype=np.float64)
            scores = proj.t_ortho
            reference = self.x_ortho_scores_
        elif kind == "all":
            if self.n_orthogonal_ == 0:
                scores = proj.t_pred
                reference = self.x_scores_
            else:
                scores = np.hstack([proj.t_pred, proj.t_ortho])
                reference = np.hstack([self.x_scores_, self.x_ortho_scores_])
        else:
            raise ValueError("kind must be one of {'predictive', 'orthogonal', 'all'}.")
        return self._score_distance_from_scores(scores, reference)

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

    def _filter(self, X: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Preprocess and orthogonal-filter new ``X`` exactly as at fit time.

        Returns the filtered ``X`` and the orthogonal scores.
        """
        X_valid = self._validate_X_predict(X)
        proj = self._project_validated(X_valid)
        return proj.X_filtered, proj.t_ortho

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
