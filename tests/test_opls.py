"""Tests for the OPLS regressor, including exact equivalence to PLSRegression."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils._testing import assert_allclose

import scikit_opls
from scikit_opls import OPLS


def _regression_data(n_samples=80, n_features=30, n_ortho=2, amp=6.0, seed=0):
    """y-correlated signal plus large y-orthogonal structured variation."""
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n_samples)
    y -= y.mean()
    p_pred = rng.normal(size=n_features)
    X = np.outer(y, p_pred)
    for _ in range(n_ortho):
        t_o = rng.normal(size=n_samples)
        t_o -= t_o.mean()
        t_o -= (t_o @ y) / (y @ y) * y  # exactly orthogonal to y
        p_o = amp * (0.5 * p_pred + rng.normal(size=n_features))
        X += np.outer(t_o, p_o)
    X += 0.05 * rng.normal(size=(n_samples, n_features))
    return X, y


def test_version():
    assert scikit_opls.__version__ == "0.1.0"


@pytest.mark.parametrize("n_components", [1, 2, 3])
def test_reduces_to_plsregression(n_components):
    """OPLS(n_orthogonal=0, scale='none') == PLSRegression(scale=False) exactly."""
    X, y = _regression_data()
    opls = OPLS(n_components=n_components, n_orthogonal=0, scale="none").fit(X, y)
    pls = PLSRegression(n_components=n_components, scale=False).fit(X, y)

    assert_allclose(opls.predict(X), pls.predict(X).ravel(), atol=1e-9)
    assert_allclose(opls.coef_, pls.coef_, atol=1e-9)
    assert_allclose(np.abs(opls.transform(X)), np.abs(pls.transform(X)), atol=1e-9)


def test_predict_shape_matches_y_ndim():
    X, y = _regression_data()
    pred_1d = OPLS(n_orthogonal=1).fit(X, y).predict(X)
    assert pred_1d.shape == (X.shape[0],)


def test_column_vector_y_warns_and_ravels():
    """A column-vector y is treated as 1d (sklearn convention) with a warning."""
    from sklearn.exceptions import DataConversionWarning

    X, y = _regression_data()
    with pytest.warns(DataConversionWarning):
        pred = OPLS(n_orthogonal=1).fit(X, y.reshape(-1, 1)).predict(X)
    assert pred.shape == (X.shape[0],)


@pytest.mark.parametrize("n_orthogonal", [0, 1])
def test_multi_output_rejected(n_orthogonal):
    """OPLS is univariate; multi-column Y is rejected regardless of n_orthogonal."""
    X, y = _regression_data()
    Y = np.column_stack([y, y * 0.5])
    with pytest.raises(ValueError, match="1d array"):
        OPLS(n_orthogonal=n_orthogonal).fit(X, Y)


def test_transform_shapes():
    X, y = _regression_data()
    model = OPLS(n_components=1, n_orthogonal=2).fit(X, y)
    assert model.transform(X).shape == (X.shape[0], 1)
    assert model.transform_orthogonal(X).shape == (X.shape[0], model.n_orthogonal_)


def test_predictive_scores_orthogonal_to_orthogonal_scores():
    """OPLS invariant: predictive ⟂ orthogonal scores (t_pred.T @ t_ortho ≈ 0)."""
    X, y = _regression_data()
    model = OPLS(n_components=1, n_orthogonal=2).fit(X, y)
    gram = model.x_scores_.T @ model.x_ortho_scores_
    assert_allclose(gram, 0.0, atol=1e-8)


def test_orthogonal_filter_improves_fit_focus():
    """Removing orthogonal variation yields a usable single-component model."""
    X, y = _regression_data()
    model = OPLS(n_components=1, n_orthogonal=2).fit(X, y)
    assert model.r2y_ > 0.9
    assert model.r2x_ortho_ > 0.0


def test_round_trip_prediction():
    X, y = _regression_data(seed=3)
    model = OPLS(n_components=1, n_orthogonal=2).fit(X, y)
    assert model.score(X, y) > 0.9  # R2 via RegressorMixin


def test_invalid_scale_raises():
    X, y = _regression_data()
    with pytest.raises(ValueError, match="scale"):
        OPLS(scale="bogus").fit(X, y)


@pytest.mark.parametrize("bad", [-1, 1.5, "nope"])
def test_invalid_n_orthogonal_raises(bad):
    X, y = _regression_data()
    with pytest.raises(ValueError, match="n_orthogonal"):
        OPLS(n_orthogonal=bad).fit(X, y)


def test_clone_and_params():
    model = OPLS(n_components=2, n_orthogonal=3, scale="pareto")
    cloned = clone(model)
    assert cloned.get_params() == model.get_params()


@pytest.mark.parametrize("n_components", [1, 2])
def test_feature_names_out_are_components(n_components):
    X, y = _regression_data()
    model = OPLS(n_components=n_components, n_orthogonal=0).fit(X, y)
    names = model.get_feature_names_out()
    expected = [f"opls_pred{i}" for i in range(n_components)]
    assert list(names) == expected
    assert model.transform(X).shape[1] == len(names)


def test_set_output_pandas_columns_named():
    pd = pytest.importorskip("pandas")
    X, y = _regression_data(n_features=6)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    model = OPLS(n_components=1, n_orthogonal=2).set_output(transform="pandas")
    out = model.fit(df, y).transform(df)
    assert isinstance(out, pd.DataFrame)
    assert list(out.columns) == ["opls_pred0"]


def test_html_repr_doc_links():
    from sklearn.utils import estimator_html_repr

    from scikit_opls import OPLS, OPLSDA

    opls = OPLS()
    opls_da = OPLSDA()

    assert opls._get_doc_link() == "https://hausergroup.github.io/scikit-opls/api/opls/"
    assert (
        opls_da._get_doc_link()
        == "https://hausergroup.github.io/scikit-opls/api/opls_da/"
    )

    # Verify they are correctly embedded in estimator HTML representation
    assert "https://hausergroup.github.io/scikit-opls/api/opls/" in estimator_html_repr(
        opls
    )
    assert (
        "https://hausergroup.github.io/scikit-opls/api/opls_da/"
        in estimator_html_repr(opls_da)
    )


def test_opls_n_orthogonal_zero_parity():
    # Parity with PLSRegression on pre-centered data
    X, y = _regression_data(seed=42)
    # pre-center X and y
    X_centered = X - X.mean(axis=0)
    y_centered = y - y.mean()

    opls = OPLS(n_components=1, n_orthogonal=0, scale="none").fit(
        X_centered, y_centered
    )
    pls = PLSRegression(n_components=1, scale=False).fit(X_centered, y_centered)
    assert_allclose(opls.predict(X_centered), pls.predict(X_centered).ravel())


def test_opls_zero_variance_columns():
    # Ensure constant columns don't cause division by zero
    X, y = _regression_data(seed=42)
    X[:, 0] = 5.0  # Constant column
    model = OPLS(n_orthogonal=1).fit(X, y)
    assert not np.isnan(model.predict(X)).any()


def test_opls_too_many_components():
    X, y = _regression_data(n_samples=10, n_features=5)
    with pytest.raises(ValueError, match="exceeds the maximum"):
        OPLS(n_components=6, n_orthogonal=0).fit(X, y)


def test_opls_constant_y_raises():
    X, y = _regression_data(seed=42)
    y_const = np.ones_like(y) * 5.0
    with pytest.raises(ValueError, match="non-constant target y"):
        OPLS().fit(X, y_const)


def test_opls_constant_x_raises():
    X, y = _regression_data(seed=42)
    X_const = np.ones_like(X) * 5.0
    with pytest.raises(ValueError, match="no non-zero variation"):
        OPLS(n_orthogonal=0).fit(X_const, y)


def test_opls_no_variation_after_filtering_raises(monkeypatch):
    from scikit_opls._orthogonal import OrthogonalComponents

    X, y = _regression_data(seed=42)
    # Mock opls_filter to return zero filtered X to trigger the guard
    dummy_components = OrthogonalComponents(
        x_ortho_weights=np.zeros((X.shape[1], 0)),
        x_ortho_scores=np.zeros((X.shape[0], 0)),
        x_ortho_loadings=np.zeros((X.shape[1], 0)),
        x_filtered=np.zeros_like(X),
        x_predictive_weight=np.zeros(X.shape[1]),
        n_components=0,
    )
    monkeypatch.setattr(
        "scikit_opls._opls.opls_filter", lambda *args, **kwargs: dummy_components
    )
    with pytest.raises(
        ValueError, match="no remaining variation after orthogonal filtering"
    ):
        OPLS().fit(X, y)


def test_opls_exposes_intercept():
    X, y = _regression_data(seed=42)
    model = OPLS(n_orthogonal=1).fit(X, y)
    assert hasattr(model, "intercept_")
    assert isinstance(model.intercept_, (float, np.ndarray))


def test_opls_constant_nonzero_x_scale_none_raises():
    X = np.ones((10, 5)) * 5.0
    y = np.arange(10.0)
    with pytest.raises(ValueError, match="no non-zero variation"):
        OPLS(scale="none", n_orthogonal=0).fit(X, y)


def test_opls_large_offset_small_variation_does_not_raise():
    rng = np.random.default_rng(0)
    X = 1e12 + rng.normal(size=(30, 5))
    y = rng.normal(size=30)
    # This should succeed since variance is normal, despite a huge offset
    OPLS(scale="center", n_orthogonal=0).fit(X, y)
