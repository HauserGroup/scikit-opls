"""OPLS-DA: Orthogonal PLS Discriminant Analysis (binary classification).

OPLS-DA fits an OPLS regression against a dummy-coded class label, then classifies
by the sign of the predictive score. This is the dominant use of OPLS in
metabolomics. The estimator wraps an internal :class:`~scikit_opls.OPLS`
(composition, so the regressor and classifier mixins never collide) and adds class
encoding plus Platt-scaled probabilities.
"""

# See _opls.py: scikit-learn's validate_data is under-typed; suppress the
# resulting static-checker false positives.
# pyright: reportArgumentType=false, reportAttributeAccessIssue=false

from __future__ import annotations

from numbers import Integral
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.multiclass import (
    check_classification_targets,
    type_of_target,
    unique_labels,
)
from sklearn.utils.validation import check_is_fitted, validate_data

from ._opls import OPLS
from ._preprocessing import VALID_SCALING


class OPLSDA(ClassifierMixin, BaseEstimator):
    """Binary OPLS Discriminant Analysis.

    Parameters mirror :class:`~scikit_opls.OPLS`. ``predict`` returns class labels;
    ``decision_function`` returns the raw predictive score; ``predict_proba`` returns
    Platt-scaled probabilities.

    Attributes
    ----------
    classes_ : ndarray
        The two class labels seen during fit.
    opls_ : OPLS
        The fitted underlying OPLS regressor (against a -1/+1 dummy response).
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
        y_dummy = np.where(y_encoded == 1, 1.0, -1.0)

        self.opls_ = OPLS(
            n_components=self.n_components,
            n_orthogonal=self.n_orthogonal,
            scale=self.scale,
            copy=self.copy,
        ).fit(X, y_dummy)
        self.n_orthogonal_ = self.opls_.n_orthogonal_

        # Platt scaling: calibrate the raw OPLS score into probabilities. Using the
        # calibrator for predict/decision_function/predict_proba keeps them mutually
        # consistent (argmax(proba) == decision_function > 0 == predict).
        self._platt = LogisticRegression().fit(self._raw_scores(X), y_encoded)
        return self

    def _raw_scores(self, X: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(self.opls_.predict(X), dtype=np.float64).reshape(-1, 1)

    def decision_function(self, X: ArrayLike) -> NDArray[np.float64]:
        """Signed confidence score; positive favours ``classes_[1]``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to score.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Signed confidence. ``> 0`` predicts ``classes_[1]``.
        """
        check_is_fitted(self)
        scores = self._platt.decision_function(self._raw_scores(X))
        return np.asarray(scores, dtype=np.float64).ravel()

    def predict(self, X: ArrayLike) -> NDArray[Any]:
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

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Platt-scaled class probabilities.

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

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        # Binary classifier: y is required and sparse X is unsupported.
        tags.target_tags.required = True
        tags.input_tags.sparse = False
        tags.non_deterministic = False
        return tags
