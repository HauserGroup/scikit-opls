"""Estimator-contract tests: parameter validation and fitted-state guarantees."""

from __future__ import annotations

from importlib import resources

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils import get_tags

from scikit_opls import O2PLS, OPLS, OPLSDA

from ._data import make_regression_data as _regression_data


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, False, "x"])
def test_n_components_invalid_raises(bad):
    X, y = _regression_data()
    with pytest.raises(ValueError, match="n_components"):
        OPLS(n_components=bad).fit(X, y)


@pytest.mark.parametrize("bad", [0, -1, 1.5, True, False, "x"])
def test_o2pls_n_components_invalid_raises(bad):
    X, y = _regression_data()
    Y = np.column_stack([y, y + 0.1 * np.random.default_rng(0).normal(size=y.shape)])
    with pytest.raises(ValueError, match="n_components"):
        O2PLS(n_components=bad).fit(X, Y)


@pytest.mark.parametrize("param", ["n_x_orthogonal", "n_y_orthogonal"])
@pytest.mark.parametrize("bad", [-1, 1.5, True, False, "x"])
def test_o2pls_orthogonal_counts_invalid_raises(param, bad):
    X, y = _regression_data()
    Y = np.column_stack([y, y + 0.1 * np.random.default_rng(0).normal(size=y.shape)])
    with pytest.raises(ValueError, match=param):
        O2PLS(**{param: bad}).fit(X, Y)


def test_n_components_too_large_raises():
    X, y = _regression_data(n_samples=20, n_features=5)
    with pytest.raises(ValueError, match="exceeds"):
        OPLS(n_components=6, n_orthogonal=0).fit(X, y)


@pytest.mark.parametrize("method", ["predict", "transform", "transform_orthogonal"])
def test_not_fitted_raises(method):
    X, _ = _regression_data()
    with pytest.raises(NotFittedError):
        getattr(OPLS(), method)(X)


@pytest.mark.parametrize(
    ("method", "arg_name"),
    [
        ("predict", "X"),
        ("predict_x", "Y"),
        ("transform", "X"),
        ("transform_y", "Y"),
        ("transform_orthogonal_x", "X"),
        ("transform_orthogonal_y", "Y"),
        ("filter_transform_x", "X"),
        ("filter_transform_y", "Y"),
    ],
)
def test_o2pls_not_fitted_raises(method, arg_name):
    X, y = _regression_data()
    Y = y.reshape(-1, 1)
    argument = X if arg_name == "X" else Y
    with pytest.raises(NotFittedError):
        getattr(O2PLS(), method)(argument)


def test_o2pls_transform_pair_not_fitted_raises():
    X, y = _regression_data()
    Y = y.reshape(-1, 1)
    with pytest.raises(NotFittedError):
        O2PLS().transform_pair(X, Y)


def test_o2pls_score_not_fitted_raises():
    X, y = _regression_data()
    with pytest.raises(NotFittedError):
        O2PLS().score(X, y)


def test_o2pls_get_feature_names_out_not_fitted_raises():
    with pytest.raises(NotFittedError):
        O2PLS().get_feature_names_out()


def test_n_features_in_set():
    X, y = _regression_data()
    model = OPLS(n_orthogonal=1).fit(X, y)
    assert model.n_features_in_ == X.shape[1]


def test_feature_names_in_with_dataframe():
    pd = pytest.importorskip("pandas")
    X, y = _regression_data(n_features=6)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    model = OPLS(n_orthogonal=1).fit(df, y)
    assert list(model.feature_names_in_) == list(df.columns)


def test_more_orthogonal_than_rank_truncates():
    X, y = _regression_data(n_features=5)
    model = OPLS(n_components=1, n_orthogonal=50).fit(X, y)
    assert model.n_orthogonal_ <= 5


@pytest.mark.parametrize("est", [O2PLS(), OPLS(), OPLSDA()])
def test_tags_match_intent(est):
    """Resolved tags should pin the declared capabilities (guards refactors)."""
    tags = get_tags(est)
    assert tags.target_tags.required is True
    assert tags.input_tags.sparse is False
    assert tags.non_deterministic is False


def test_opls_regressor_tag_poor_score():
    tags = get_tags(OPLS())
    assert tags.regressor_tags is not None
    assert tags.regressor_tags.poor_score is True


def test_o2pls_regressor_tags():
    tags = get_tags(O2PLS())
    assert tags.regressor_tags is not None
    assert tags.regressor_tags.poor_score is True
    assert tags.target_tags.multi_output is True
    assert tags.target_tags.single_output is True


def test_opls_da_not_multiclass():
    tags = get_tags(OPLSDA())
    assert tags.classifier_tags is not None
    assert tags.classifier_tags.multi_class is False


def test_py_typed_marker_present():
    assert resources.files("scikit_opls").joinpath("py.typed").is_file()
