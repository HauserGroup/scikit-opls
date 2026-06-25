"""Orthogonal PLS (OPLS) regressor with a scikit-learn interface.

OPLS (Trygg & Wold, 2002) splits the variation in ``X`` into a *predictive* part
correlated with ``y`` and *orthogonal* parts uncorrelated with ``y``. This estimator
removes the orthogonal variation with a NIPALS filter (:mod:`scikit_opls._orthogonal`)
and then fits :class:`sklearn.cross_decomposition.PLSRegression` on the cleaned ``X`` as
the predictive engine. With ``n_orthogonal=0`` it reduces exactly to ``PLSRegression``.
"""

# scikit-learn's validate_data uses sentinel-string parameter defaults that lead
# static type checkers to flag every downstream array op as a false positive.
# Suppress those categories here; the test suite and ``check_estimator`` are the
# real correctness gate.
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportReturnType=false

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

from ._orthogonal import apply_orthogonal_filter, opls_filter
from ._preprocessing import VALID_SCALING, apply_scaling, compute_scaling
from .inspection import explained_x_variance


class OPLS(RegressorMixin, TransformerMixin, BaseEstimator):
    """Orthogonal Projections to Latent Structures regression.

    Parameters
    ----------
    n_components : int, default=1
        Number of predictive components. True OPLS uses 1, which is required
        whenever ``n_orthogonal != 0``. Values > 1 are only allowed with
        ``n_orthogonal=0`` (plain multi-component PLS, a non-``ropls`` mode).
    n_orthogonal : int, default=1
        Number of orthogonal (y-uncorrelated) components to remove from ``X``.
        To choose this by cross-validated Q2, use
        :func:`~scikit_opls.selection.select_orthogonal`.
    scale : {"none", "center", "pareto", "standard"}, default="standard"
        Column preprocessing applied to ``X`` (matches ``ropls`` ``scaleC``).
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

        self.x_mean_, self.x_std_ = compute_scaling(X, self.scale)
        Xs = apply_scaling(X, self.x_mean_, self.x_std_)

        # opls_filter handles n_orthogonal=0 (pass-through) and truncation itself.
        ofit = opls_filter(Xs, y - y.mean(), self.n_orthogonal)
        X_filtered = ofit.x_filtered
        self.x_ortho_weights_ = ofit.x_ortho_weights
        self.x_ortho_loadings_ = ofit.x_ortho_loadings
        self.x_ortho_scores_ = ofit.x_ortho_scores
        self.n_orthogonal_ = ofit.n_components

        self.pls_ = PLSRegression(n_components=self.n_components, scale=False)
        self.pls_.fit(X_filtered, y)
        # transform() returns the predictive scores: one output column per component.
        self._n_features_out = self.n_components

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
        # VIP is no longer computed eagerly — see scikit_opls.inspection.vip().
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

    def _filter(self, X: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Preprocess and orthogonal-filter new ``X`` exactly as at fit time.

        Returns the filtered ``X`` and the orthogonal scores.
        """
        X = validate_data(self, X, reset=False, dtype=np.float64)
        Xs = apply_scaling(X, self.x_mean_, self.x_std_)
        return apply_orthogonal_filter(
            Xs, self.x_ortho_weights_, self.x_ortho_loadings_
        )

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.regressor_tags.poor_score = True
        # OPLS is a supervised transformer: fit requires y.
        tags.target_tags.required = True
        # The NIPALS orthogonal filter densifies; sparse input is unsupported.
        tags.input_tags.sparse = False
        # Deterministic for a fixed (non-shuffled) cv.
        tags.non_deterministic = False
        return tags
