"""OPLS-DA: Orthogonal PLS Discriminant Analysis (binary classification).

OPLS-DA fits an OPLS regression against a dummy-coded class label, then classifies
by the sign of the predictive score. This is the dominant use of OPLS in
metabolomics. The estimator wraps an internal :class:`~scikit_opls.OPLS`
(composition, so the regressor and classifier mixins never collide) and adds class
encoding plus Platt-scaled probabilities.
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.metaestimators import available_if
from sklearn.utils.multiclass import (
    check_classification_targets,
    type_of_target,
    unique_labels,
)
from sklearn.utils.validation import check_is_fitted, validate_data

from scikit_opls._opls import OPLS
from scikit_opls._preprocessing import VALID_SCALING, _has_nonzero_variation


class OPLSDA(ClassifierMixin, BaseEstimator):
    """Binary OPLS Discriminant Analysis.

    Parameters mirror :class:`~scikit_opls.OPLS`. ``predict`` returns class labels;
    ``decision_function`` returns the Platt-calibrated logistic decision score;
    ``raw_score`` returns the raw (uncalibrated) predictive score;
    ``predict_proba`` returns in-sample Platt-scaled probabilities.

    Attributes
    ----------
    classes_ : ndarray
        The two class labels seen during fit.
    opls_ : OPLS
        The fitted underlying OPLS regressor (against a -1/+1 dummy response).
    vip_, ortho_vip_ : ndarray of shape (n_features,)
        Lazy predictive / orthogonal Variable Importance in Projection scores,
        delegating to the inner :attr:`opls_`. Use with
        :class:`~sklearn.feature_selection.SelectFromModel` via
        ``importance_getter="vip_"``.
    """

    classes_: NDArray
    n_features_in_: int
    feature_names_in_: NDArray[np.str_]
    opls_: OPLS
    n_orthogonal_: int
    _label_encoder: LabelEncoder
    _platt: LogisticRegression

    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "n_orthogonal": [Interval(Integral, 0, None, closed="left")],
        "scale": [StrOptions(set(VALID_SCALING))],
        "copy": ["boolean"],
        "probability": ["boolean"],
    }

    def __init__(
        self,
        n_components: int = 1,
        n_orthogonal: int = 1,
        scale: str = "standard",
        copy: bool = True,
        probability: bool = False,
    ) -> None:
        self.n_components = n_components
        self.n_orthogonal = n_orthogonal
        self.scale = scale
        self.copy = copy
        self.probability = probability

    def fit(self, X: ArrayLike, y: ArrayLike) -> OPLSDA:
        """Fit the binary OPLS-DA classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training predictors.
        y : array-like of shape (n_samples,)
            Binary class labels (exactly two classes).

        Returns
        -------
        self : OPLSDA
            The fitted estimator.
        """
        self._validate_params()
        # validate_data ravels a column-vector y and rejects multi-column y.
        X, y = validate_data(
            self, X, y, dtype=np.float64, ensure_min_samples=2, copy=self.copy
        )
        check_classification_targets(y)
        y_type = type_of_target(y, input_name="y")

        # unique_labels is the guide's recommended idiom for class discovery;
        # LabelEncoder then maps labels to the -1/+1 dummy response.
        self.classes_ = unique_labels(y)
        self._label_encoder = LabelEncoder().fit(y)
        if self.classes_.shape[0] != 2:
            raise ValueError(
                "Only binary classification is supported. "
                f"The type of the target is {y_type}."
            )

        y_encoded = self._label_encoder.transform(y)

        # Guard: At least 2 samples per class, and at least 5 samples overall
        counts = np.bincount(y_encoded)
        if np.any(counts < 2):
            raise ValueError("OPLSDA requires at least two samples per class.")
        if X.shape[0] < 5:
            raise ValueError("OPLSDA requires at least 5 samples overall.")

        y_dummy = np.where(y_encoded == 1, 1.0, -1.0)

        self.opls_ = OPLS(
            n_components=self.n_components,
            n_orthogonal=self.n_orthogonal,
            scale=self.scale,
            copy=self.copy,
        ).fit(X, y_dummy)
        self.n_orthogonal_ = self.opls_.n_orthogonal_

        if self.probability:
            raw = self.decision_function(X).reshape(-1, 1)
            if not _has_nonzero_variation(raw):
                raise ValueError(
                    "OPLSDA produced constant raw scores; "
                    "classification/calibration is undefined."
                )
            self._platt = LogisticRegression().fit(raw, y_encoded)
        return self

    def _raw_scores(self, X: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(self.opls_.predict(X), dtype=np.float64).reshape(-1, 1)

    def raw_score(self, X: ArrayLike) -> NDArray[np.float64]:
        """OPLS predictive regression score (uncalibrated).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to score.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            The raw OPLS predictive score.
        """
        return self.decision_function(X)

    def decision_function(self, X: ArrayLike) -> NDArray[np.float64]:
        """Signed uncalibrated confidence score; positive favours ``classes_[1]``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to score.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Signed uncalibrated confidence. ``> 0`` predicts ``classes_[1]``.
        """
        check_is_fitted(self)
        return np.asarray(self.opls_.predict(X), dtype=np.float64).ravel()

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

    def _check_probability(self) -> bool:
        if not getattr(self, "probability", False):
            raise AttributeError(
                "predict_proba is only available when probability=True."
            )
        return True

    @available_if(_check_probability)
    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """In-sample Platt-scaled class probabilities.

        .. warning::
            These probabilities are calibrated in-sample on the same training data
            used to fit the OPLS model, which can lead to overconfidence (especially
            on small datasets, common in metabolomics). For robust cross-fitted
            calibration, consider wrapping your estimator in
            :class:`~sklearn.calibration.CalibratedClassifierCV`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to score.

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            Class probabilities, column ``j`` for ``classes_[j]``.
        """
        check_is_fitted(self)
        return np.asarray(
            self._platt.predict_proba(self._raw_scores(X)), dtype=np.float64
        )

    @property
    def vip_(self) -> NDArray[np.float64]:
        """Predictive VIP per feature (delegates to the inner OPLS)."""
        check_is_fitted(self)
        return self.opls_.vip_

    @property
    def ortho_vip_(self) -> NDArray[np.float64]:
        """Orthogonal VIP per feature (delegates to the inner OPLS)."""
        check_is_fitted(self)
        return self.opls_.ortho_vip_

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        # Binary classifier: y is required and sparse X is unsupported.
        tags.target_tags.required = True
        tags.input_tags.sparse = False
        tags.non_deterministic = False
        return tags
