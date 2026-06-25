"""OPLS-DA: Orthogonal PLS Discriminant Analysis (binary classification).

OPLS-DA fits an OPLS regression against a dummy-coded class label, then classifies
by the sign of the predictive score. This is the dominant use of OPLS in
metabolomics. The estimator wraps an internal :class:`~scikit_opls.OPLS`
(composition, so the regressor and classifier mixins never collide) and adds class
encoding plus Platt-scaled probabilities.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import DataConversionWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_is_fitted

from ._opls import OPLS
from ._preprocessing import check_scaling
from ._validation import validate_fit_labels


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

    def fit(self, X: ArrayLike, y: ArrayLike) -> OPLSDA:
        check_scaling(self.scale)
        X, y = validate_fit_labels(self, X, y, copy=self.copy)
        if y.ndim == 2 and y.shape[-1] == 1:
            warnings.warn(
                "A column-vector y was passed when a 1d array was expected. "
                "Please change the shape of y to (n_samples,).",
                DataConversionWarning,
                stacklevel=2,
            )
            y = y.ravel()
        check_classification_targets(y)
        y_type = type_of_target(y, input_name="y")

        self._label_encoder = LabelEncoder().fit(y)
        self.classes_ = np.asarray(self._label_encoder.classes_)
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
            cv=self.cv,
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
        """Signed confidence; positive favours ``classes_[1]``."""
        check_is_fitted(self)
        scores = self._platt.decision_function(self._raw_scores(X))
        return np.asarray(scores, dtype=np.float64).ravel()

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        """Predict class labels."""
        indices = (self.decision_function(X) > 0.0).astype(int)
        return self.classes_[indices]

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Platt-scaled class probabilities, shape ``(n_samples, 2)``."""
        check_is_fitted(self)
        return np.asarray(
            self._platt.predict_proba(self._raw_scores(X)), dtype=np.float64
        )

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags
