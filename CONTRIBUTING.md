# Contributing to scikit-opls

Thanks for your interest in improving scikit-opls. This project follows the
[scikit-learn Developer's Guide](https://scikit-learn.org/stable/developers/index.html)
where practical.

## Development setup

The project uses [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/HauserGroup/scikit-opls
cd scikit-opls
uv sync --dev              # install the package (editable) and dev tools
uv run pre-commit install  # enable the git hooks (run once)
```

## Checks

Run these before opening a pull request (CI runs the same set):

```bash
uv run pytest --cov        # tests + coverage
uv run ruff check          # lint
uv run ruff format --check # format check
uv run pyright src         # type-check
uv run pre-commit run --all-files
```

## Standards

- **Estimator compliance.** `tests/test_sklearn_compat.py` runs
  `parametrize_with_checks` over the estimators. This is a hard gate; resolve any
  newly triggered checks rather than silencing them. New public params/methods
  must keep the suite green.
- **Docstrings.** All public methods and functions use the
  [numpydoc standard](https://numpydoc.readthedocs.io) (`Parameters`, `Returns`,
  and `Notes`/`References` where useful). Init params go under **Parameters**,
  learned attributes under **Attributes**.
- **`__init__` purity.** Store every keyword argument unchanged; no logic or
  validation in `__init__`.
- **Naming/imports.** PEP8 names (`n_samples`, not `nsamples`). Intra-package
  imports are relative (`from ._opls import OPLS`); Ruff's isort enforces order.
- **Tests.** New code comes with tests. Use
  `sklearn.utils._testing.assert_allclose` for numeric comparisons; pass `atol`
  when comparing against zero.

## Reporting bugs

Please include a
[minimal reproducer](https://scikit-learn.org/stable/developers/minimal_reproducer.html):
the smallest self-contained snippet, exact versions, and the full traceback.

## Pull requests

Fill in the PR template checklist. Keep PRs focused. While the package is `0.x`
the public API may change without a deprecation cycle (see
[CHANGES.md](CHANGES.md)).
