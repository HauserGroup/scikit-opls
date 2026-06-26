# scikit-opls

Orthogonal Projections to Latent Structures (OPLS / OPLS-DA) with a
scikit-learn interface.

OPLS (Trygg & Wold, 2002) splits the variation in `X` into a *predictive* part
correlated with `y` and *orthogonal* parts uncorrelated with `y`, removes the
orthogonal variation with a NIPALS filter, then fits a standard PLS engine on the
cleaned `X`. With `n_orthogonal=0` it reduces exactly to `PLSRegression`.

## Highlights

- [`OPLS`](api/opls.md) — regressor and supervised transformer.
- Cross-validated `n_orthogonal` selection via scikit-learn's `GridSearchCV`
  (see [Quickstart](quickstart.md)).
- [`OPLSDA`](api/opls_da.md) — binary classifier composing `OPLS`.
- [Inspection](api/inspection.md) — on-demand VIP scores and variance metrics.
- [Plotting](api/plotting.md) — score and S-plot Displays.
- [Validation](api/validation.md) — permutation significance testing.

All estimators pass scikit-learn's `check_estimator` compliance suite, support
`clone` / `get_params` / `set_params`, and work inside `Pipeline` and
`GridSearchCV`.

See [Installation](installation.md) and [Quickstart](quickstart.md) to get going.

<div class="admonition tip">
  <p class="admonition-title">Getting Started</p>
  <p>Check out the <a href="quickstart.md">Quickstart</a> guide to see examples of regression, cross-validation, and classification with <code>scikit-opls</code>.</p>
</div>
