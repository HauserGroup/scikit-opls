# pyright: reportArgumentType=false, reportAttributeAccessIssue=false, reportReturnType=false
# pyright: reportAbstractUsage=false
"""Kernel-based Orthogonal Projections to Latent Structures (K-OPLS)."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import KernelCenterer
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.validation import (
    _check_feature_names_in,
    check_is_fitted,
    validate_data,
)

from scikit_opls._inspection import component_r2y_from_scores
from scikit_opls._kopls_core import KOPLSComponents, kopls_fit, kopls_transform
from scikit_opls._preprocessing import VALID_SCALING, apply_scaling, compute_scaling
from scikit_opls._utils import _has_nonzero_variation, _reject_bool_param

_VALID_KERNELS = (
    "linear",
    "poly",
    "polynomial",
    "rbf",
    "sigmoid",
    "laplacian",
    "cosine",
    "precomputed",
)
# Row block size used when reading a kernel diagonal, so that k(x, x) never
# requires materialising the full test-test kernel.
_DIAGONAL_CHUNK = 256


@dataclass(frozen=True)
class _KOPLSProjection:
    """All fitted K-OPLS model-space arrays for one validated raw X block."""

    k_test_train: NDArray[np.float64]
    t_pred: NDArray[np.float64]
    t_ortho: NDArray[np.float64]


class KOPLS(RegressorMixin, TransformerMixin, BaseEstimator):
    """Kernel-based Orthogonal Projections to Latent Structures regression.

    K-OPLS is the dual (kernel) reformulation of OPLS. It keeps the split between
    Y-predictive and Y-orthogonal variation, but performs it in the feature space
    induced by a kernel function, so non-linear X/y relationships can be modelled.
    With ``kernel="linear"`` and ``scale="none"`` it reproduces
    [`OPLS`][scikit_opls.OPLS].

    Because the model lives in feature space, no loadings in the original variable
    space exist: there is no ``coef_``, no ``x_loadings_`` and no VIP. Use the
    predictive and Y-orthogonal scores for interpretation.

    Parameters
    ----------
    n_components : int, default=1
        Number of predictive components ``A``. The predictive components come from
        an eigendecomposition of ``Y.T @ K @ Y``, so ``n_components`` cannot exceed
        the number of targets; a univariate ``y`` admits exactly one.
    n_orthogonal : int, default=1
        Number of Y-orthogonal components removed in feature space.
        To choose this by cross-validated Q2, wrap ``KOPLS`` in
        [`GridSearchCV`][sklearn.model_selection.GridSearchCV] over ``n_orthogonal``.
    kernel : str or callable, default="linear"
        Kernel passed to
        [`pairwise_kernels`][sklearn.metrics.pairwise.pairwise_kernels]: one of
        ``"linear"``, ``"poly"``/``"polynomial"``, ``"rbf"``, ``"sigmoid"``,
        ``"laplacian"``, ``"cosine"``, ``"precomputed"``, or a callable taking two
        arrays and returning a kernel matrix. With ``"precomputed"``, ``fit``
        expects the ``(n_samples, n_samples)`` training Gram matrix and ``predict``
        expects the ``(n_test, n_train)`` block.
    gamma : float or None, default=None
        Kernel coefficient for ``"rbf"``, ``"poly"``, ``"sigmoid"`` and
        ``"laplacian"``. ``None`` means ``1 / n_features``. The Gaussian kernel of
        Rantalainen et al. is parametrised by ``sigma``; the mapping is
        ``gamma = 1 / (2 * sigma**2)``.
    degree : float, default=3
        Degree of the polynomial kernel. The polynomial kernel of Rantalainen et al.,
        ``(<x, y> + 1) ** p``, corresponds to ``gamma=1, coef0=1, degree=p``.
    coef0 : float, default=1
        Independent term of the polynomial and sigmoid kernels.
    kernel_params : dict or None, default=None
        Extra keyword arguments for a callable ``kernel``.
    scale : {"none", "center", "pareto", "standard"}, default="standard"
        Column preprocessing applied to ``X`` *before* the kernel is evaluated. It
        matters for distance-based kernels. Must be ``"none"`` when
        ``kernel="precomputed"``, since there is no X block to scale.
    copy : bool, default=True
        Whether the input arrays are copied during validation. Note that
        ``copy=False`` is passed to sklearn input validation; K-OPLS still
        allocates working kernel matrices.

    Attributes
    ----------
    n_orthogonal_ : int
        Number of Y-orthogonal components actually used.
    X_fit_ : ndarray
        Training data used to build test kernels: the preprocessed ``X`` block, or
        the training Gram matrix when ``kernel="precomputed"``.
    kernel_centerer_ : KernelCenterer
        Fitted feature-space centerer. Kernel centering is always applied; without
        it the predictive/Y-orthogonal split is not meaningful.
    x_scores_ : ndarray of shape (n_samples, n_components)
        Predictive scores ``Tp`` after all orthogonal deflations.
    x_ortho_scores_ : ndarray of shape (n_samples, n_orthogonal_)
        Unit-norm Y-orthogonal scores ``To``.
    y_scores_, y_loadings_ : ndarray
        Predictive Y-scores ``Up`` and Y-loadings ``Cp``.
    eigenvalues_ : ndarray of shape (n_components,)
        Eigenvalues of ``Y.T @ K @ Y`` paired with ``y_loadings_``.
    ortho_loadings_, ortho_eigenvalues_, ortho_norms_ : ndarray
        Per-orthogonal-component loading vectors, eigenvalues, and the training
        score norms used to normalise out-of-sample orthogonal scores.
    coef_t_ : ndarray of shape (n_components, n_components)
        Regression coefficients ``Bt`` from predictive scores to ``Up``. Predictions
        are ``Tp @ coef_t_ @ y_loadings_.T + y_mean_``. There is no coefficient
        vector on the input features: the model is not linear in ``X``.
    y_mean_ : ndarray of shape (n_targets,)
        Training target mean, added back to predictions. ``Y`` is centered but not
        scaled, matching [`OPLS`][scikit_opls.OPLS].
    x_mean_, x_std_ : ndarray
        Centering/scaling vectors applied to ``X`` before the kernel. Zeros and ones
        when ``kernel="precomputed"``.
    r2x_, r2x_ortho_, r2y_, rmse_ : float
        Training-set fit summaries. ``r2x_ortho_`` is the share of the kernel trace
        removed by the orthogonal deflations; ``r2x_`` is the share captured by the
        predictive scores in the deflated kernel. Like
        [`OPLS`][scikit_opls.OPLS] these are diagnostic summaries, not a guaranteed
        additive partition. ``rmse_`` is the uncorrected training root mean squared
        error. For cross-validated Q2 use
        [`cross_val_score`][sklearn.model_selection.cross_val_score].
    r2x_components_, r2x_ortho_components_, r2y_components_ : ndarray
        Per-component versions of the summaries above.
    q_residuals_train_ : ndarray of shape (n_samples,)
        Training Q residuals in the full (predictive + orthogonal) feature-space
        reconstruction; equals ``q_residuals(X_train, space="full")``.
    q_residuals_predictive_train_ : ndarray of shape (n_samples,)
        Training Q residuals in the predictive-only reconstruction space.
    x_residual_ss_, y_residual_ss_ : float
        Sum of ``q_residuals_train_``, and the training residual sum of squares
        of ``y`` against the fitted predictions.
    n_features_in_ : int
        Number of features seen during [`fit`][scikit_opls.KOPLS.fit]. With
        ``kernel="precomputed"`` this is the number of training samples.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during [`fit`][scikit_opls.KOPLS.fit]. Defined only
        when ``X`` has feature names that are all strings.

    See Also
    --------
    OPLS : Linear OPLS regression; ``KOPLS(kernel="linear", scale="none")``
        reproduces it.
    O2PLS : Two-block variant that also models Y-specific orthogonal structure.
    sklearn.kernel_ridge.KernelRidge : Kernel regression without the
        predictive/orthogonal split.

    Notes
    -----
    Prediction replays the fitted deflations against the test-train kernel block,
    which requires keeping both deflation sequences from training. Memory is
    therefore ``O(n_orthogonal * n_samples**2)``, and fitting is ``O(n_samples**3)``
    -- K-OPLS is intended for the wide, small-``n`` data typical of omics, not for
    large sample counts.

    Y is centered but not scaled. With multiple targets on very different scales,
    the eigendecomposition of ``Y.T @ K @ Y`` is dominated by the large-variance
    targets; scale ``Y`` yourself if that is not what you want.

    References
    ----------
    .. [1] Rantalainen, M., Bylesjo, M., Cloarec, O., Nicholson, J. K., Holmes, E.
           & Trygg, J. (2007). Kernel-based orthogonal projections to latent
           structures (K-OPLS). Journal of Chemometrics, 21(7-9), 376-385.
           https://doi.org/10.1002/cem.1071
    .. [2] Bylesjo, M., Rantalainen, M., Nicholson, J. K., Holmes, E. & Trygg, J.
           (2008). K-OPLS package: Kernel-based orthogonal projections to latent
           structures for prediction and interpretation in feature space.
           BMC Bioinformatics, 9, 106. https://doi.org/10.1186/1471-2105-9-106

    Examples
    --------
    >>> import numpy as np
    >>> from scikit_opls import KOPLS
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(20, 5))
    >>> y = np.sin(X[:, 0]) + X[:, 1] ** 2
    >>> model = KOPLS(n_orthogonal=1, kernel="rbf", gamma=0.1).fit(X, y)
    >>> model.transform(X).shape
    (20, 1)
    >>> model.predict(X).shape
    (20,)
    """

    n_features_in_: int
    feature_names_in_: NDArray[np.str_]
    n_orthogonal_: int
    X_fit_: NDArray[np.float64]
    kernel_centerer_: KernelCenterer
    x_mean_: NDArray[np.float64]
    x_std_: NDArray[np.float64]
    y_mean_: NDArray[np.float64]
    x_scores_: NDArray[np.float64]
    x_ortho_scores_: NDArray[np.float64]
    y_scores_: NDArray[np.float64]
    y_loadings_: NDArray[np.float64]
    eigenvalues_: NDArray[np.float64]
    ortho_loadings_: NDArray[np.float64]
    ortho_eigenvalues_: NDArray[np.float64]
    ortho_norms_: NDArray[np.float64]
    coef_t_: NDArray[np.float64]
    r2x_: float
    r2x_ortho_: float
    r2y_: float
    rmse_: float
    r2x_components_: NDArray[np.float64]
    r2x_ortho_components_: NDArray[np.float64]
    r2y_components_: NDArray[np.float64]
    q_residuals_train_: NDArray[np.float64]
    q_residuals_predictive_train_: NDArray[np.float64]
    x_residual_ss_: float
    y_residual_ss_: float
    _components: KOPLSComponents
    _k_fit: NDArray[np.float64]
    _dual_pred: NDArray[np.float64]
    _dual_full: NDArray[np.float64]
    _y_ndim: int
    _n_features_out: int

    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "n_orthogonal": [Interval(Integral, 0, None, closed="left")],
        "kernel": [StrOptions(set(_VALID_KERNELS)), callable],
        "gamma": [Interval(Real, 0, None, closed="neither"), None],
        "degree": [Interval(Real, 0, None, closed="left")],
        "coef0": [Interval(Real, None, None, closed="neither")],
        "kernel_params": [dict, None],
        "scale": [StrOptions(set(VALID_SCALING))],
        "copy": ["boolean"],
    }

    def __init__(
        self,
        n_components: int = 1,
        n_orthogonal: int = 1,
        kernel: str = "linear",
        gamma: float | None = None,
        degree: float = 3,
        coef0: float = 1,
        kernel_params: dict | None = None,
        scale: str = "standard",
        copy: bool = True,
    ) -> None:
        self.n_components = n_components
        self.n_orthogonal = n_orthogonal
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.scale = scale
        self.copy = copy

    def fit(self, X: ArrayLike, y: ArrayLike) -> KOPLS:
        """Fit the K-OPLS model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, or the ``(n_samples, n_samples)`` training kernel matrix
            when ``kernel="precomputed"``.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        Returns
        -------
        self : KOPLS
            Fitted estimator.
        """
        X, Y = self._validate_fit_data(X, y)
        K = self._fit_kernel(X)
        self.y_mean_ = Y.mean(axis=0)
        components = kopls_fit(
            K, Y - self.y_mean_, self.n_components, self.n_orthogonal
        )
        self._set_fitted_components(components, K)
        self._set_fit_diagnostics(K, Y)
        return self

    def _validate_fit_data(
        self, X: ArrayLike, y: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Validate ``(X, y)`` and return the X block and a 2D target matrix."""
        _reject_bool_param("n_components", self.n_components)
        _reject_bool_param("n_orthogonal", self.n_orthogonal)
        self._validate_params()
        if self.kernel == "precomputed" and self.scale != "none":
            raise ValueError(
                "scale must be 'none' when kernel='precomputed'; there is no X "
                f"block to scale, got scale={self.scale!r}."
            )
        X, y = validate_data(
            self,
            X,
            y,
            dtype=np.float64,
            ensure_min_samples=2,
            multi_output=True,
            copy=self.copy,
        )
        if self.kernel == "precomputed" and X.shape[0] != X.shape[1]:
            raise ValueError(
                f"With kernel='precomputed', X must be a square training kernel "
                f"matrix, got shape {X.shape}."
            )
        if not _has_nonzero_variation(y):
            raise ValueError("KOPLS requires a non-constant target y.")
        y_arr = np.asarray(y, dtype=np.float64)
        self._y_ndim = y_arr.ndim
        Y = y_arr.reshape(-1, 1) if y_arr.ndim == 1 else y_arr
        if self.n_components > Y.shape[1]:
            raise ValueError(
                f"n_components={self.n_components} exceeds the number of targets "
                f"({Y.shape[1]}). K-OPLS derives predictive components from an "
                f"eigendecomposition of Y.T @ K @ Y, so a univariate target admits "
                f"exactly one predictive component."
            )
        return X, Y

    def _fit_kernel(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Preprocess ``X``, build the training kernel and center it."""
        if self.kernel == "precomputed":
            self.x_mean_ = np.zeros(X.shape[1], dtype=np.float64)
            self.x_std_ = np.ones(X.shape[1], dtype=np.float64)
            self.X_fit_ = X
        else:
            self.x_mean_, self.x_std_ = compute_scaling(X, self.scale)
            self.X_fit_ = apply_scaling(X, self.x_mean_, self.x_std_)
        K_raw = self._pairwise_kernel(self.X_fit_)
        self.kernel_centerer_ = KernelCenterer().fit(K_raw)
        return np.asarray(self.kernel_centerer_.transform(K_raw), dtype=np.float64)

    def _pairwise_kernel(
        self, X: NDArray[np.float64], Y: NDArray[np.float64] | None = None
    ) -> NDArray[np.float64]:
        """Evaluate the configured kernel, passing ``X`` through when precomputed."""
        if self.kernel == "precomputed":
            return X
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma, "degree": self.degree, "coef0": self.coef0}
        return np.asarray(
            pairwise_kernels(X, Y, metric=self.kernel, filter_params=True, **params),
            dtype=np.float64,
        )

    def _kernel_diagonal(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return ``k(x, x)`` per row without materialising the full kernel.

        Kernels that are constant on the diagonal are read off directly; anything
        else is evaluated in row blocks, so the cost stays linear in ``n_samples``.
        """
        if self.kernel in ("rbf", "laplacian", "cosine"):
            return np.ones(X.shape[0], dtype=np.float64)
        diagonal = np.empty(X.shape[0], dtype=np.float64)
        for start in range(0, X.shape[0], _DIAGONAL_CHUNK):
            block = X[start : start + _DIAGONAL_CHUNK]
            diagonal[start : start + _DIAGONAL_CHUNK] = np.diag(
                self._pairwise_kernel(block)
            )
        return diagonal

    def _set_fitted_components(
        self, components: KOPLSComponents, K: NDArray[np.float64]
    ) -> None:
        """Copy fitted core quantities onto the estimator."""
        self._components = components
        self._k_fit = K
        self.n_orthogonal_ = components.n_orthogonal
        self.x_scores_ = components.x_scores
        self.x_ortho_scores_ = components.x_ortho_scores
        self.y_scores_ = components.y_scores
        self.y_loadings_ = components.y_loadings
        self.eigenvalues_ = components.eigenvalues
        self.ortho_loadings_ = components.ortho_loadings
        self.ortho_eigenvalues_ = components.ortho_eigenvalues
        self.ortho_norms_ = components.ortho_norms
        self.coef_t_ = components.coef_t
        self._n_features_out = self.n_components
        # Feature-space directions are Phi.T @ B for these dual coefficients: the
        # predictive weights are Phi.T @ Up @ Lambda**-0.5, and the i-th orthogonal
        # loading is Phi.T @ to_i (each to_i is already orthogonal to its
        # predecessors, so the deflations leave it unchanged). Q residuals project
        # onto the span of those directions.
        self._dual_pred = components.y_scores * np.sqrt(components.eigenvalues) ** -1.0
        self._dual_full = np.hstack([components.x_ortho_scores, self._dual_pred])

    def _set_fit_diagnostics(
        self, K: NDArray[np.float64], Y: NDArray[np.float64]
    ) -> None:
        """Compute training-set fit summaries."""
        total = float(np.trace(K))
        total = total if total > 0.0 else 1.0
        k_final = self._components.k_ortho_final
        self.r2x_ortho_ = 1.0 - float(np.trace(k_final)) / total

        scores = self.x_scores_
        score_ss = np.sum(scores**2, axis=0)
        safe_ss = np.where(score_ss > 0.0, score_ss, 1.0)
        captured = np.einsum("ij,jk,ki->i", scores.T, k_final, scores) / safe_ss
        self.r2x_components_ = np.where(score_ss > 0.0, captured / total, 0.0)
        self.r2x_ = float(np.sum(self.r2x_components_))

        traces = [float(np.trace(k)) for k in self._components.k_ortho_seq]
        traces.append(float(np.trace(k_final)))
        self.r2x_ortho_components_ = np.array(
            [(traces[i] - traces[i + 1]) / total for i in range(self.n_orthogonal_)],
            dtype=np.float64,
        )

        y_fit = self._predict_from_scores(scores)
        self.r2y_ = float(r2_score(Y, y_fit))
        self.rmse_ = float(root_mean_squared_error(Y, y_fit))
        self.r2y_components_ = component_r2y_from_scores(
            Y - self.y_mean_, scores, self.y_loadings_ @ self.coef_t_.T
        )
        self.y_residual_ss_ = float(np.sum((Y - y_fit) ** 2))

        diagonal = np.diag(K).copy()
        self.q_residuals_train_ = self._q_residuals_from_kernel(
            K, diagonal, space="full"
        )
        self.q_residuals_predictive_train_ = self._q_residuals_from_kernel(
            K, diagonal, space="predictive"
        )
        self.x_residual_ss_ = float(np.sum(self.q_residuals_train_))

    def _predict_from_scores(self, t_pred: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map predictive scores to the original target scale."""
        return t_pred @ self.coef_t_ @ self.y_loadings_.T + self.y_mean_

    def _validate_X_predict(self, X: ArrayLike) -> NDArray[np.float64]:  # noqa: N802
        """Validate prediction/projection input against fitted K-OPLS metadata."""
        check_is_fitted(self)
        return validate_data(
            self,
            X,
            dtype=np.float64,
            copy=self.copy,
            reset=False,
        )

    def _project_validated(self, X_valid: NDArray[np.float64]) -> _KOPLSProjection:
        """Project already validated raw X into fitted K-OPLS model spaces."""
        if self.kernel == "precomputed":
            Xs = X_valid
        else:
            Xs = apply_scaling(X_valid, self.x_mean_, self.x_std_)
        K_raw = self._pairwise_kernel(Xs, self.X_fit_)
        K = np.asarray(self.kernel_centerer_.transform(K_raw), dtype=np.float64)
        t_pred, t_ortho = kopls_transform(K, self._components)
        return _KOPLSProjection(k_test_train=K, t_pred=t_pred, t_ortho=t_ortho)

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predict targets for new samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict, or the ``(n_test, n_train)`` kernel block when
            ``kernel="precomputed"``.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Predicted target values, shaped like the ``y`` seen during fit.
        """
        X_valid = self._validate_X_predict(X)
        y_pred = self._predict_from_scores(self._project_validated(X_valid).t_pred)
        return y_pred.ravel() if self._y_ndim == 1 else y_pred

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Project samples onto the predictive components.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to project.

        Returns
        -------
        x_scores : ndarray of shape (n_samples, n_components)
            Predictive scores.
        """
        X_valid = self._validate_X_predict(X)
        return self._project_validated(X_valid).t_pred

    def transform_orthogonal(self, X: ArrayLike) -> NDArray[np.float64]:
        """Project samples onto the Y-orthogonal components.

        This is a non-standard method (outside the ``set_output`` contract).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to project.

        Returns
        -------
        x_ortho_scores : ndarray of shape (n_samples, n_orthogonal_)
            Y-orthogonal scores.
        """
        X_valid = self._validate_X_predict(X)
        return self._project_validated(X_valid).t_ortho

    def get_feature_names_out(self, input_features=None) -> NDArray[np.object_]:
        """Output feature names for the predictive scores.

        [`transform`][scikit_opls.KOPLS.transform] reduces ``X`` to
        ``n_components`` predictive scores, so the output columns are components,
        not input features. They are named ``kopls_pred0, kopls_pred1, ...``,
        independent of the input feature names. ``transform_orthogonal`` is outside
        the ``set_output`` contract and has no names here.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input feature names; only validated for length against
            ``n_features_in_`` (the output names do not depend on them).

        Returns
        -------
        feature_names_out : ndarray of str objects
            Output feature names.
        """
        check_is_fitted(self, "_n_features_out")
        _check_feature_names_in(self, input_features)
        return np.asarray(
            [f"kopls_pred{i}" for i in range(self._n_features_out)], dtype=object
        )

    def _score_distance_from_scores(
        self,
        scores: NDArray[np.float64],
        reference_scores: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Mahalanobis-like distance in latent score space."""
        T = np.asarray(scores, dtype=np.float64)
        T_ref = np.asarray(reference_scores, dtype=np.float64)
        center = T_ref.mean(axis=0, keepdims=True)
        T_centered = T - center
        T_ref_centered = T_ref - center
        if T_ref.shape[1] == 1:
            var = float(np.var(T_ref_centered[:, 0], ddof=1))
            var = max(var, np.finfo(np.float64).eps)
            return (T_centered[:, 0] ** 2) / var
        cov = np.cov(T_ref_centered, rowvar=False)
        inv_cov = np.linalg.pinv(cov)
        return np.sum((T_centered @ inv_cov) * T_centered, axis=1)

    def score_distance(
        self,
        X: ArrayLike,
        *,
        kind: str = "predictive",
    ) -> NDArray[np.float64]:
        """Return Hotelling-like score distances for samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples in the same raw feature space used for fitting. Pass raw X.
            Do not manually center or scale before calling diagnostics. The
            estimator applies its fitted preprocessing internally.
        kind : {"predictive", "orthogonal", "all"}, default="predictive"
            Which latent score space to use.

        Returns
        -------
        distance : ndarray of shape (n_samples,)
            Squared Mahalanobis-like distance in the selected fitted score space.
        """
        X_valid = self._validate_X_predict(X)
        proj = self._project_validated(X_valid)
        if kind == "predictive":
            scores, reference = proj.t_pred, self.x_scores_
        elif kind == "orthogonal":
            if self.n_orthogonal_ == 0:
                return np.zeros(X_valid.shape[0], dtype=np.float64)
            scores, reference = proj.t_ortho, self.x_ortho_scores_
        elif kind == "all":
            if self.n_orthogonal_ == 0:
                scores, reference = proj.t_pred, self.x_scores_
            else:
                scores = np.hstack([proj.t_pred, proj.t_ortho])
                reference = np.hstack([self.x_scores_, self.x_ortho_scores_])
        else:
            raise ValueError("kind must be one of {'predictive', 'orthogonal', 'all'}.")
        return self._score_distance_from_scores(scores, reference)

    def q_residuals(
        self,
        X: ArrayLike,
        *,
        space: str = "full",
    ) -> NDArray[np.float64]:
        """Return Q residuals, i.e. squared feature-space residual norm per sample.

        The residual is measured against the orthogonal projection of the centered
        feature map onto the span of the fitted model directions, which is the
        kernel counterpart of the reconstruction residual reported by
        [`OPLS`][scikit_opls.OPLS].

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples in raw feature space. Pass raw X. Do not manually center or
            scale before calling diagnostics. The estimator applies its fitted
            preprocessing internally.
        space : {"full", "predictive"}, default="full"
            Which model space to project onto:

            - ``"full"`` uses the predictive and Y-orthogonal directions.
            - ``"predictive"`` uses the predictive directions only, treating
              Y-orthogonal variation as part of the residual.

        Returns
        -------
        q : ndarray of shape (n_samples,)
            Squared residual norm per sample.

        Raises
        ------
        ValueError
            If ``kernel="precomputed"``. The residual needs the test-test kernel
            diagonal ``k(x, x)``, which a test-train block does not carry. The
            fitted ``q_residuals_train_`` is still available in that case.
        """
        if self.kernel == "precomputed":
            raise ValueError(
                "q_residuals is unavailable with kernel='precomputed': it needs "
                "the test-test kernel diagonal k(x, x), which the test-train "
                "block does not provide. Use the fitted q_residuals_train_ "
                "attribute, or fit with a computed kernel."
            )
        X_valid = self._validate_X_predict(X)
        proj = self._project_validated(X_valid)
        Xs = apply_scaling(X_valid, self.x_mean_, self.x_std_)
        raw_diagonal = self._kernel_diagonal(Xs)
        # Centered diagonal, from the KernelCenterer identity applied at x == z.
        centerer = self.kernel_centerer_
        diagonal = (
            raw_diagonal
            - 2.0 * self._pairwise_kernel(Xs, self.X_fit_).mean(axis=1)
            + float(centerer.K_fit_all_)
        )
        return self._q_residuals_from_kernel(proj.k_test_train, diagonal, space=space)

    def _q_residuals_from_kernel(
        self,
        k_test_train: NDArray[np.float64],
        diagonal: NDArray[np.float64],
        *,
        space: str,
    ) -> NDArray[np.float64]:
        """Squared residual norms after projecting onto the fitted model span."""
        if space == "predictive":
            dual = self._dual_pred
        elif space == "full":
            dual = self._dual_full
        else:
            raise ValueError("space must be one of {'full', 'predictive'}.")
        if dual.shape[1] == 0:
            return diagonal
        gram = dual.T @ self._k_fit @ dual
        cross = k_test_train @ dual
        captured = np.sum((cross @ np.linalg.pinv(gram)) * cross, axis=1)
        return np.maximum(diagonal - captured, 0.0)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.regressor_tags.poor_score = True
        # K-OPLS is a supervised transformer: fit requires y.
        tags.target_tags.required = True
        # Y enters as a matrix in the algorithm, so multiple targets are native.
        tags.target_tags.multi_output = True
        # Kernel evaluation and deflation densify; sparse input is unsupported.
        tags.input_tags.sparse = False
        tags.input_tags.pairwise = self.kernel == "precomputed"
        tags.non_deterministic = False
        return tags
