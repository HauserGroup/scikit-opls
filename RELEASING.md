# Releasing scikit-opls

The build backend is [hatchling](https://hatch.pypa.io/). The single source of
truth for the version is `__version__` in `src/scikit_opls/version.py`;
`pyproject.toml` declares `dynamic = ["version"]` and Hatchling reads that file
at build time, so there is nothing to keep in sync by hand.

Publishing is automated: pushing a `vX.Y.Z` tag runs
[`.github/workflows/release.yml`](.github/workflows/release.yml), which builds
the artifacts, publishes them to PyPI through Trusted Publishing and creates the
GitHub release.

## One-time setup

Register the Trusted Publisher on PyPI (no API token is stored anywhere) at
<https://pypi.org/manage/project/scikit-opls/settings/publishing/>:

| Field       | Value         |
| ----------- | ------------- |
| Owner       | `HauserGroup` |
| Repository  | `scikit-opls` |
| Workflow    | `release.yml` |
| Environment | `pypi`        |

Then create the matching `pypi` environment under repository
Settings → Environments, and restrict it to tags if you want a second gate.

For a first release of a name that does not exist on PyPI yet, register a
*pending* publisher instead, under Your projects → Publishing.

## Steps

1. Ensure `main` is green in CI.
1. Bump `__version__` in `src/scikit_opls/version.py`. Follow semver.
1. Move the `Unreleased` section of `CHANGES.md` under the new version with the
   release date.
1. Update `version` and `date-released` in `CITATION.cff`.
1. Review minimum dependency versions in `pyproject.toml`; bump the floor per
   the scikit-learn "bump minimum versions" discipline if appropriate.
1. Commit, tag, and push:
   ```bash
   git commit -am "release: vX.Y.Z" && git tag vX.Y.Z && git push && git push --tags
   ```
1. Watch the Release workflow. It refuses to publish if the tag does not match
   `__version__`, so a mismatch fails before anything reaches PyPI.

## Publishing by hand

Only needed if the workflow is unavailable. This uses a personal API token
rather than Trusted Publishing:

```bash
uv build && uvx twine check --strict dist/*
uv publish
```

## Versioning

- `0.x`: public API may change without a deprecation cycle.
- `>=1.0`: deprecation cycle required (`utils.deprecated`, `FutureWarning`,
  two releases). Document default-value changes in `CHANGES.md`.
