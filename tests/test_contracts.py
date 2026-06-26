"""Estimator-contract tests: parameter validation and fitted-state guarantees."""

from __future__ import annotations

import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils import get_tags

from scikit_opls import OPLS, OPLSDA

from .test_opls import _regression_data


@pytest.mark.parametrize("bad", [0, -1, 1.5, "x"])
def test_n_components_invalid_raises(bad):
    X, y = _regression_data()
    with pytest.raises(ValueError, match="n_components"):
        OPLS(n_components=bad).fit(X, y)


def test_n_components_too_large_raises():
    X, y = _regression_data(n_samples=20, n_features=5)
    with pytest.raises(ValueError, match="exceeds"):
        OPLS(n_components=6, n_orthogonal=0).fit(X, y)


@pytest.mark.parametrize("method", ["predict", "transform", "transform_orthogonal"])
def test_not_fitted_raises(method):
    X, _ = _regression_data()
    with pytest.raises(NotFittedError):
        getattr(OPLS(), method)(X)


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


@pytest.mark.parametrize("est", [OPLS(), OPLSDA()])
def test_tags_match_intent(est):
    """Resolved tags should pin the declared capabilities (guards refactors)."""
    tags = get_tags(est)
    assert tags.target_tags.required is True
    assert tags.input_tags.sparse is False
    assert tags.non_deterministic is False


def test_opls_regressor_tag_poor_score():
    assert get_tags(OPLS()).regressor_tags.poor_score is True


def test_opls_da_not_multiclass():
    assert get_tags(OPLSDA()).classifier_tags.multi_class is False
