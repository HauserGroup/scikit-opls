# Releasing scikit-opls

Build backend is [hatchling](https://hatch.pypa.io/); the source of truth for the
version is `__version__` in `src/scikit_opls/__init__.py` (mirrored in
`pyproject.toml`).

## Steps

1. Ensure `main` is green in CI.
1. Bump the version in `pyproject.toml` and `src/scikit_opls/__init__.py`
   (keep them identical). Follow semver.
1. Move the `Unreleased` section of `CHANGES.md` under the new version with the
   date.
1. Review minimum dependency versions in `pyproject.toml`; bump the floor per the
   scikit-learn "bump minimum versions" discipline if appropriate.
1. Commit, tag, and push:
   ```bash
   git commit -am "release: vX.Y.Z"
   git tag vX.Y.Z
   git push && git push --tags
   ```
1. Build and check the artifacts:
   ```bash
   uv build                       # sdist + wheel into dist/
   uvx twine check dist/*
   ```
1. Publish:
   ```bash
   uv publish                     # or: uvx twine upload dist/*
   ```
1. Create the GitHub release from the tag, pasting the changelog section.

## Versioning

- `0.x`: public API may change without a deprecation cycle.
- `>=1.0`: deprecation cycle required (`utils.deprecated`, `FutureWarning`,
  two releases). Document default-value changes in `CHANGES.md`.
