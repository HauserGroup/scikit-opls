"""OPLS-DA: Orthogonal PLS Discriminant Analysis (binary classification).

OPLS-DA fits an OPLS regression against a dummy-coded class label, then classifies
by the sign of the fitted regression output. OPLS-DA is commonly used in
metabolomics. The estimator wraps an internal :class:`~scikit_opls.OPLS`
(composition, so the regressor and classifier mixins never collide) and adds class
encoding. ``decision_function`` exposes the raw signed OPLS regression output, so
calibrated probabilities are available — cross-fitted, not in-sample — via
:class:`~sklearn.calibration.CalibratedClassifierCV` when each class has enough
samples for the chosen calibration CV split.
"""

# See _opls.py: scikit-learn's validate_data is under-typed; suppress the
# resulting static-checker false positives. reportAbstractUsage covers
# Interval/StrOptions (abstract __str__ not visibly overridden).
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false
# pyright: reportAbstractUsage=false

from __future__ import annotations

from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_is_fitted, validate_data

from scikit_opls._opls import OPLS
from scikit_opls._preprocessing import VALID_SCALING


class OPLSDA(ClassifierMixin, BaseEstimator):
    """Binary OPLS Discriminant Analysis.

    Parameters mirror :class:`~scikit_opls.OPLS`. ``decision_function`` returns the
    raw signed OPLS regression output (positive favours ``classes_[1]``) and
    ``predict`` returns class labels from its sign. For class probabilities, wrap in
    :class:`~sklearn.calibration.CalibratedClassifierCV` (cross-fitted, robust)
    when each class has enough samples for the chosen calibration CV split.

    Attributes
    ----------
    classes_ : ndarray
        The two class labels seen during fit.
    opls_ : OPLS
        The fitted underlying OPLS regressor against a -1/+1 dummy response.
    vip_, ortho_vip_ : ndarray of shape (n_features,)
        Predictive / orthogonal Variable Importance in Projection scores computed
        by the inner :attr:`opls_`. Use with
        :class:`~sklearn.feature_selection.SelectFromModel` via
        ``importance_getter="vip_"``.
    """

    classes_: NDArray
    n_features_in_: int
    feature_names_in_: NDArray[np.str_]
    opls_: OPLS
    n_orthogonal_: int
    _label_encoder: LabelEncoder

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

    def fit(self, X: ArrayLike, y: ArrayLike) -> OPLSDA:
        """Fit the binary OPLS-DA classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training predictors.
        y : array-like of shape (n_samples,)
            Binary class labels.

        Returns
        -------
        self : OPLSDA
            The fitted estimator.
        """
        if isinstance(self.n_components, bool):
            raise ValueError("n_components must be an integer, not bool.")
        if isinstance(self.n_orthogonal, bool):
            raise ValueError("n_orthogonal must be an integer, not bool.")

        self._validate_params()

        # validate_data(..., reset=True) records n_features_in_ and, for DataFrame
        # inputs with string columns, feature_names_in_ on the OUTER estimator.
        # The returned X is an ndarray, which is what we intentionally pass to the
        # inner OPLS so OPLSDA owns the public input validation contract.
        X_valid, y_valid = validate_data(
            self,
            X,
            y,
            dtype=np.float64,
            ensure_min_samples=2,
            copy=self.copy,
            reset=True,
        )
        check_classification_targets(y_valid)

        self._label_encoder = LabelEncoder().fit(y_valid)
        self.classes_ = self._label_encoder.classes_

        if self.classes_.shape[0] != 2:
            y_type = type_of_target(y_valid, input_name="y")
            raise ValueError(
                "Only binary classification is supported. "
                f"The type of the target is {y_type}."
            )

        y_encoded = self._label_encoder.transform(y_valid)

        counts = np.bincount(y_encoded)
        if np.any(counts < 2):
            raise ValueError("OPLSDA requires at least two samples per class.")

        # Fit regression against a symmetric coding so zero is the natural decision
        # boundary: negative -> classes_[0], positive -> classes_[1].
        y_dummy = np.where(y_encoded == 1, 1.0, -1.0)

        self.opls_ = OPLS(
            n_components=self.n_components,
            n_orthogonal=self.n_orthogonal,
            scale=self.scale,
            copy=self.copy,
        ).fit(X_valid, y_dummy)

        # The inner OPLS was deliberately fitted on a validated ndarray. Mirror the
        # public feature metadata onto it so plotting/helper code that uses
        # self.opls_ directly still gets consistent feature-name checks.
        self.opls_.n_features_in_ = self.n_features_in_
        if hasattr(self, "feature_names_in_"):
            self.opls_.feature_names_in_ = self.feature_names_in_.copy()
        elif hasattr(self.opls_, "feature_names_in_"):
            del self.opls_.feature_names_in_

        self.n_orthogonal_ = self.opls_.n_orthogonal_
        return self

    def _validate_X_predict(self, X: ArrayLike) -> NDArray[np.float64]:  # noqa: N802
        """Validate prediction input against the outer OPLSDA fit contract."""
        check_is_fitted(self)

        # validate_data(..., reset=False) checks n_features_in_ and feature_names_in_
        # against OPLSDA, then returns a nameless ndarray. Passing that ndarray to
        # the inner OPLS avoids the spurious "fitted without feature names" warning.
        X_valid = validate_data(
            self,
            X,
            dtype=np.float64,
            copy=self.copy,
            reset=False,
        )
        return np.asarray(X_valid, dtype=np.float64)

    def _validate_x_predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Backward-compatible alias for the canonical validation helper."""
        return self._validate_X_predict(X)

    def decision_function(self, X: ArrayLike) -> NDArray[np.float64]:
        """Raw signed OPLS regression output; positive favours ``classes_[1]``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to score.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Signed confidence; ``> 0`` predicts ``classes_[1]``. Scores equal to
            zero are assigned to ``classes_[0]`` by :meth:`predict`.
        """
        X_valid = self._validate_X_predict(X)
        X_filtered, _ = self.opls_._filter_validated(X_valid)
        return np.asarray(self.opls_.pls_.predict(X_filtered), dtype=np.float64).ravel()

    def predict(self, X: ArrayLike) -> NDArray:
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to classify.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted labels drawn from ``classes_``.
        """
        indices = (self.decision_function(X) > 0.0).astype(int)
        return self.classes_[indices]

    @property
    def vip_(self) -> NDArray[np.float64]:
        """Predictive VIP per feature, delegated to the inner OPLS."""
        check_is_fitted(self)
        return self.opls_.vip_

    @property
    def ortho_vip_(self) -> NDArray[np.float64]:
        """Orthogonal VIP per feature, delegated to the inner OPLS."""
        check_is_fitted(self)
        return self.opls_.ortho_vip_

    def score_distance(
        self,
        X: ArrayLike,
        *,
        kind: str = "predictive",
    ) -> NDArray[np.float64]:
        """Return Hotelling-like score distances from the inner OPLS model."""
        X_valid = self._validate_X_predict(X)
        return self.opls_._score_distance_validated(X_valid, kind=kind)

    def q_residuals(
        self,
        X: ArrayLike,
        *,
        space: str = "full",
    ) -> NDArray[np.float64]:
        """Return Q residuals from the inner OPLS model."""
        X_valid = self._validate_X_predict(X)
        return self.opls_._q_residuals_validated(X_valid, space=space)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        tags.target_tags.required = True
        tags.input_tags.sparse = False
        tags.non_deterministic = False
        return tags
