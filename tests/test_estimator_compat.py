"""Estimator-contract tests: parameter validation and fitted-state guarantees."""

from __future__ import annotations

from importlib import resources

import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from scikit_opls import O2PLS, OPLS, OPLSDA

from ._data import make_regression_data as _regression_data


def test_opls_basic_param_validation_smoke():
    """Verify that estimators call scikit-learn's _validate_params."""
    X, y = _regression_data()
    with pytest.raises(ValueError):
        OPLS(n_components=0).fit(X, y)


def test_oplsda_rejects_invalid_n_orthogonal():
    """Smoke test for OPLSDA parameter validation."""
    X, y = _regression_data()
    y_bin = np.where(y > y.mean(), 1, 0)
    with pytest.raises(ValueError):
        OPLSDA(n_orthogonal=-1).fit(X, y_bin)


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
    tags = est.__sklearn_tags__()
    assert tags.target_tags.required is True
    assert tags.input_tags.sparse is False
    assert tags.non_deterministic is False


def test_opls_regressor_tag_poor_score():
    tags = OPLS().__sklearn_tags__()
    assert tags.regressor_tags is not None
    assert tags.regressor_tags.poor_score is True


def test_o2pls_regressor_tags():
    tags = O2PLS().__sklearn_tags__()
    assert tags.regressor_tags is not None
    assert tags.regressor_tags.poor_score is True
    assert tags.target_tags.multi_output is True
    assert tags.target_tags.single_output is True


def test_opls_da_not_multiclass():
    tags = OPLSDA().__sklearn_tags__()
    assert tags.classifier_tags is not None
    assert tags.classifier_tags.multi_class is False


def test_py_typed_marker_present():
    assert resources.files("scikit_opls").joinpath("py.typed").is_file()


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
@parametrize_with_checks(
    [
        O2PLS(),
        O2PLS(scale="pareto"),
        OPLS(),
        OPLS(n_orthogonal=0),
        OPLS(scale="pareto"),
        OPLSDA(),
    ]
)
def test_sklearn_compatible_estimator(estimator, check):
    # The tiny synthetic matrices check_estimator generates frequently have no
    # y-orthogonal structure, so the orthogonal filter legitimately truncates and
    # warns. That signal matters to real users (kept in _orthogonal) but is just
    # noise here, where the contract — not the data — is under test.
    check(estimator)
