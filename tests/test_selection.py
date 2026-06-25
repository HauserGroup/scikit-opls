"""Tests for GridSearchCV-based selection of n_orthogonal."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from scikit_opls import OPLS, OPLSDA, select_orthogonal

from .test_opls import _regression_data
from .test_opls_da import _classification_data


def test_select_orthogonal_returns_grid_search_and_refits_opls():
    X, y = _regression_data(n_ortho=2, amp=8.0, seed=1)

    search = select_orthogonal(OPLS(n_components=1), cv=5).fit(X, y)

    assert isinstance(search, GridSearchCV)
    assert isinstance(search.best_params_["n_orthogonal"], int)
    assert 0 <= search.best_params_["n_orthogonal"] <= 9
    assert isinstance(search.best_estimator_, OPLS)
    assert search.best_estimator_.n_orthogonal == search.best_params_["n_orthogonal"]
    assert "mean_test_score" in search.cv_results_


def test_select_orthogonal_can_choose_zero_for_noise_with_parsimony():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 20))
    y = rng.normal(size=60)

    search = select_orthogonal(OPLS(), cv=5, tol=1.0).fit(X, y)

    assert search.best_params_["n_orthogonal"] == 0


def test_parsimonious_refit_selects_fewest_count_within_tolerance():
    search = select_orthogonal(OPLS(), tol=0.02)

    selected = search.refit(
        {
            "mean_test_score": np.array([0.70, 0.79, 0.80, 0.795]),
            "param_n_orthogonal": np.array([0, 1, 2, 3]),
        }
    )

    assert selected == 1


def test_select_orthogonal_max_orthogonal_caps_grid():
    search = select_orthogonal(OPLS(), max_orthogonal=3)

    assert search.param_grid == {"n_orthogonal": [0, 1, 2, 3]}


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_orthogonal": -1}, "max_orthogonal"),
        ({"tol": -0.01}, "tol"),
    ],
)
def test_select_orthogonal_rejects_invalid_search_params(kwargs, match):
    with pytest.raises(ValueError, match=match):
        select_orthogonal(OPLS(), **kwargs)


def test_select_orthogonal_oplsda_with_auc_refits_classifier():
    X, y = _classification_data()

    search = select_orthogonal(OPLSDA(), scoring="roc_auc", cv=3).fit(X, y)

    assert isinstance(search.best_estimator_, OPLSDA)
    assert 0 <= search.best_params_["n_orthogonal"] <= 9


def test_select_orthogonal_is_cloneable():
    search = select_orthogonal(
        OPLS(n_components=1, scale="pareto"), cv=4, max_orthogonal=5, tol=0.02
    )

    cloned = clone(search)
    assert cloned.param_grid == search.param_grid
    assert cloned.cv == search.cv
    assert cloned.n_jobs == search.n_jobs
    assert isinstance(cloned.estimator, OPLS)
    assert cloned.estimator.get_params() == search.estimator.get_params()


def test_select_orthogonal_works_in_pipeline_and_n_jobs():
    X, y = _regression_data(n_ortho=2, amp=8.0, seed=1)
    pipe = Pipeline(
        [
            (
                "opls",
                select_orthogonal(
                    OPLS(n_components=1), cv=3, max_orthogonal=3, n_jobs=2
                ),
            )
        ]
    )

    pipe.fit(X, y)

    pred = pipe.predict(X)
    assert pred.shape == (X.shape[0],)
