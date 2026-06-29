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

from scikit_opls._inspection import explained_x_variance, orthogonal_vip, predictive_vip
from scikit_opls._orthogonal import apply_orthogonal_filter, opls_filter
from scikit_opls._preprocessing import VALID_SCALING, apply_scaling, compute_scaling
from scikit_opls._utils import _has_nonzero_variation


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
        # Refits must invalidate lazily cached VIP arrays because fitted weights may
        # change even when the same Python estimator instance is reused.
        for _attr in ("_vip_", "_ortho_vip_"):
            self.__dict__.pop(_attr, None)
        if isinstance(self.n_components, bool):
            raise ValueError("n_components must be an integer, not bool.")
        if isinstance(self.n_orthogonal, bool):
            raise ValueError("n_orthogonal must be an integer, not bool.")
        self._validate_params()
        # multi_output=False ravels a column-vector y (with a DataConversionWarning)
        # and rejects multi-column y: OPLS is univariate.
        # Note: The internal OSC primitive supports multivariate blocks, but the public
        # OPLS estimator currently exposes only univariate OPLS regression.
        X, y = validate_data(
            self, X, y, dtype=np.float64, ensure_min_samples=2, copy=self.copy
        )
        if not _has_nonzero_variation(y):
            raise ValueError("OPLS requires a non-constant target y.")

        if self.n_components > min(X.shape):
            raise ValueError(
                f"n_components={self.n_components} exceeds the maximum of "
                f"min(n_samples, n_features)={min(X.shape)}."
            )

        self.x_mean_, self.x_std_ = compute_scaling(X, self.scale)
        Xs = apply_scaling(X, self.x_mean_, self.x_std_)
        # The predictive direction and downstream PLS model both require resolvable
        # variation after preprocessing.
        if not _has_nonzero_variation(Xs, axis=0):
            raise ValueError("X has no non-zero variation after preprocessing.")

        # opls_filter handles n_orthogonal=0 (pass-through) and truncation itself.
        # y is centered only for direction extraction; the predictive PLS engine
        # below still fits against the original y scale.
        ofit = opls_filter(Xs, y - y.mean(), self.n_orthogonal)
        X_filtered = ofit.x_filtered
        if not _has_nonzero_variation(X_filtered, axis=0):
            raise ValueError(
                "X has no remaining variation after orthogonal filtering; "
                "reduce n_orthogonal."
            )

        # Validate the numerical rank of the actual matrix passed to PLSRegression.
        rank_filtered = np.linalg.matrix_rank(X_filtered)
        if self.n_components > rank_filtered:
            raise ValueError(
                f"n_components={self.n_components} exceeds the numerical rank of "
                f"X after orthogonal filtering ({rank_filtered}). "
                "Reduce n_components or n_orthogonal."
            )

        self.x_ortho_weights_ = ofit.x_ortho_weights
        self.x_ortho_loadings_ = ofit.x_ortho_loadings
        self.x_ortho_scores_ = ofit.x_ortho_scores
        self.n_orthogonal_ = ofit.n_components

        # The sklearn PLS engine sees only the preprocessed, orthogonally filtered X.
        self.pls_ = PLSRegression(n_components=self.n_components, scale=False)
        self.pls_.fit(X_filtered, y)
        # transform() returns the predictive scores: one output column per component.
        self._n_features_out = self.n_components

        # Surface the predictive model parameters from the engine.
        self.x_weights_ = self.pls_.x_weights_
        self.x_loadings_ = self.pls_.x_loadings_
        self.x_scores_ = self.pls_.x_scores_
        self.y_loadings_ = self.pls_.y_loadings_
        self.coef_filtered_ = self.pls_.coef_
        self.intercept_ = self.pls_.intercept_
        # The engine's prediction offset is predict(0), not intercept_: sklearn's
        # PLSRegression centers the filtered X internally, so predict(Z) ==
        # Z @ coef_.T + predict(0) but intercept_ omits that centering term (it only
        # coincides with predict(0) when the filtered X is already centered).
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

        # Diagnostics are computed on the training data in the same coordinate
        # systems exposed by the fitted attributes.
        y_fit = self.pls_.predict(X_filtered)
        self.r2x_ = explained_x_variance(Xs, self.x_scores_, self.x_loadings_)
        self.r2x_ortho_ = explained_x_variance(
            Xs, self.x_ortho_scores_, self.x_ortho_loadings_
        )
        self.r2y_ = float(r2_score(y, y_fit))
        self.rmse_ = float(root_mean_squared_error(y, y_fit))
        return self

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
        check_is_fitted(self)
        # Reuse the same preprocessing/filtering path as transform() so prediction
        # matches the fitted PLS coordinate system.
        X_filtered, _ = self._filter(X)
        return self.pls_.predict(X_filtered).ravel()

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
        check_is_fitted(self)
        X_filtered, _ = self._filter(X)
        return self.pls_.transform(X_filtered)

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
        check_is_fitted(self)
        return self._filter(X)[1]

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
        check_is_fitted(self)
        return self._filter(X)[0]

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

    def _filter_validated(
        self, X: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Filter an already validated dense array without checking names again."""
        Xs = apply_scaling(X, self.x_mean_, self.x_std_)
        # apply_orthogonal_filter returns both the filtered matrix for prediction
        # and the replayed orthogonal scores for transform_orthogonal().
        return apply_orthogonal_filter(
            Xs, self.x_ortho_weights_, self.x_ortho_loadings_
        )

    def _filter(self, X: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Preprocess and orthogonal-filter new ``X`` exactly as at fit time.

        Returns the filtered ``X`` and the orthogonal scores.
        """
        X_valid = self._validate_X_predict(X)
        return self._filter_validated(X_valid)

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
