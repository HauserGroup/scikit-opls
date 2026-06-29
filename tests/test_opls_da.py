"""Tests for the binary OPLS-DA classifier."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import DataConversionWarning, NotFittedError
from sklearn.utils._testing import assert_allclose

from scikit_opls import OPLSDA

from ._data import make_classification_data as _classification_data


def test_separates_classes():
    X, y = _classification_data()
    model = OPLSDA(n_components=1, n_orthogonal=2).fit(X, y)
    assert model.score(X, y) > 0.95
    np.testing.assert_array_equal(np.sort(model.classes_), np.array(["case", "ctrl"]))


def test_predict_returns_known_labels():
    X, y = _classification_data()
    model = OPLSDA(n_orthogonal=1).fit(X, y)
    preds = model.predict(X)
    assert set(np.unique(preds)).issubset(set(model.classes_))


def test_calibrated_classifier_cv_provides_proba():
    """Probabilities come from cross-fitted CalibratedClassifierCV, not OPLSDA."""
    from sklearn.calibration import CalibratedClassifierCV

    X, y = _classification_data()
    clf = CalibratedClassifierCV(OPLSDA(n_orthogonal=1), cv=3).fit(X, y)
    proba = clf.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    assert_allclose(proba.sum(axis=1), 1.0, atol=1e-8)


def test_decision_function_sign_matches_predict():
    X, y = _classification_data()
    model = OPLSDA(n_orthogonal=1).fit(X, y)
    scores = model.decision_function(X)
    expected = model.classes_[(scores > 0).astype(int)]
    np.testing.assert_array_equal(model.predict(X), expected)


@pytest.mark.parametrize("method", ["predict", "decision_function"])
def test_unfitted_methods_raise(method):
    X, _ = _classification_data()

    with pytest.raises(NotFittedError):
        getattr(OPLSDA(), method)(X)


@pytest.mark.parametrize("attr", ["vip_", "ortho_vip_"])
def test_unfitted_vip_properties_raise(attr):
    with pytest.raises(NotFittedError):
        getattr(OPLSDA(), attr)


@pytest.mark.parametrize(
    ("param", "kwargs"),
    [
        ("n_components", {"n_components": True}),
        ("n_components", {"n_components": False}),
        ("n_orthogonal", {"n_orthogonal": True}),
        ("n_orthogonal", {"n_orthogonal": False}),
    ],
)
def test_bool_integer_parameters_raise_clear_value_error(param, kwargs):
    X, y = _classification_data()

    with pytest.raises(ValueError, match=param):
        OPLSDA(**kwargs).fit(X, y)


def test_decision_function_zero_predicts_first_class():
    class ZeroScoreOPLSDA(OPLSDA):
        def decision_function(self, X):
            return np.array([-1.0, 0.0, 1.0])

    model = ZeroScoreOPLSDA()
    model.classes_ = np.array(["case", "ctrl"])

    np.testing.assert_array_equal(
        model.predict(np.zeros((3, 1))), np.array(["case", "case", "ctrl"])
    )


def test_scores_available_via_underlying_opls():
    X, y = _classification_data()
    model = OPLSDA(n_components=1, n_orthogonal=2).fit(X, y)
    assert model.opls_.transform(X).shape == (X.shape[0], 1)
    assert model.opls_.transform_orthogonal(X).shape == (
        X.shape[0],
        model.n_orthogonal_,
    )


def test_n_features_in_set():
    X, y = _classification_data()

    model = OPLSDA(n_orthogonal=1).fit(X, y)

    assert model.n_features_in_ == X.shape[1]


def test_feature_names_in_with_dataframe():
    pd = pytest.importorskip("pandas")
    X, y = _classification_data(n_features=6)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

    model = OPLSDA(n_orthogonal=1).fit(df, y)

    assert list(model.feature_names_in_) == list(df.columns)


@pytest.mark.parametrize("method", ["predict", "decision_function"])
def test_methods_reject_wrong_number_of_features(method):
    X, y = _classification_data(n_features=6)
    model = OPLSDA(n_orthogonal=1).fit(X, y)

    with pytest.raises(ValueError, match="features"):
        getattr(model, method)(X[:, :5])


def test_oplsda_rejects_sparse_input():
    sparse = pytest.importorskip("scipy.sparse")
    X, y = _classification_data()

    with pytest.raises(TypeError, match="Sparse data"):
        OPLSDA().fit(sparse.csr_matrix(X), y)


def test_column_vector_y_warns_and_ravels():
    X, y = _classification_data()

    with pytest.warns(DataConversionWarning):
        model = OPLSDA(n_orthogonal=1).fit(X, y.reshape(-1, 1))

    assert model.predict(X).shape == y.shape


def test_non_binary_raises():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 5))
    y = np.array([0, 1, 2] * 10)
    with pytest.raises(ValueError, match="binary"):
        OPLSDA().fit(X, y)


@pytest.mark.parametrize("scale", ["standard", "pareto"])
def test_clone_and_params(scale):
    model = OPLSDA(n_components=1, n_orthogonal=3, scale=scale)
    cloned = clone(model)
    assert isinstance(cloned, OPLSDA)
    assert cloned.get_params() == model.get_params()


def test_opls_da_sample_guards():
    rng = np.random.default_rng(0)
    X_imb = rng.normal(size=(10, 30))
    y_imbalanced = np.array(["ctrl"] * 9 + ["case"] * 1)
    with pytest.raises(ValueError, match="at least two samples per class"):
        OPLSDA().fit(X_imb, y_imbalanced)


def test_four_samples_with_two_per_class_is_allowed():
    X = np.array(
        [
            [-2.0, 0.0, 1.0],
            [-1.0, 1.0, 0.0],
            [1.0, 0.0, -1.0],
            [2.0, -1.0, 0.0],
        ]
    )
    y = np.array(["ctrl", "ctrl", "case", "case"])

    model = OPLSDA(n_orthogonal=0).fit(X, y)

    assert model.predict(X).shape == y.shape


def test_decision_function_is_raw_opls_regression_output():
    X, y = _classification_data()
    model = OPLSDA(n_orthogonal=1).fit(X, y)
    df = model.decision_function(X)
    assert df.shape == (X.shape[0],)
    np.testing.assert_allclose(df, model.opls_.predict(X).ravel())
    np.testing.assert_array_equal(
        model.predict(X), model.classes_[(df > 0).astype(int)]
    )


def test_no_probability_param():
    # probability mode removed: not a constructor parameter, no predict_proba.
    assert "probability" not in OPLSDA().get_params()
    model = OPLSDA(n_orthogonal=1).fit(*_classification_data())
    assert not hasattr(model, "predict_proba")
