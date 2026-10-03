# Contributing to PhilTorch

Thank you for helping improve PhilTorch. Contributions should stay focused, be easy to review, and preserve the numerical behavior of the affected filters unless a change is explicitly intended and documented.

## Before starting

- Check the [issue tracker](https://github.com/yoyolicoris/philtorch/issues) and [project roadmap](https://github.com/users/yoyolicoris/projects/5) for related work.
- For a bug, open a bug report with a minimal reproduction.
- For a new API or a substantial behavior change, open a feature request before implementation so the scope can be agreed first.
- Keep unrelated refactors, formatting changes, and documentation rewrites out of the same pull request.

## Development setup

PhilTorch uses [Pixi](https://pixi.sh/) to describe its development environment.

```bash
git clone https://github.com/yoyolicoris/philtorch.git
cd philtorch
git switch dev
pixi install
```

Create a branch from the latest `dev` branch:

```bash
git pull --ff-only origin dev
git switch -c <type>/<short-description>
```

Use a descriptive prefix such as `fix/`, `feat/`, `docs/`, `test/`, or `ci/`.

## Testing and formatting

Run the test suite through the Pixi environment:

```bash
pixi run python -m pytest
```

For a focused change, run the smallest relevant test file while iterating, then run the full suite before requesting review when practical. If a test cannot be run on your platform, state that clearly in the pull request.

Python lint and formatting are enforced by Ruff in CI, which checks the whole repository. The Pixi environment provides the same pinned Ruff version, so run the same commands through it:

```bash
pixi run ruff check .
pixi run ruff format --check .
```

Apply formatting only to files in the scope of your change.

## Pull requests

Open pull requests against `dev`, not `main`. A pull request should:

- explain the problem and the chosen solution;
- link the relevant issue when one exists;
- include tests for behavior changes or explain why tests are not applicable;
- document user-visible API or behavior changes;
- avoid generated files, unrelated cleanup, and dependency changes unless required by the stated scope;
- pass the applicable CI checks.

Maintainers may ask for a larger proposal to be split into smaller pull requests. This keeps review precise and makes regressions easier to isolate.

## Releasing

Releases are cut from `dev`. The package version comes from the latest `v*` tag reachable from the commit being built (setuptools_scm), so the release tag must be on a commit that `dev` contains.

1. Pick the `dev` commit to release and check that its CI, including the TestPyPI upload, has passed.
2. Fast-forward `main` to that commit:

   ```bash
   git fetch origin
   git push origin <sha>:main
   ```

   This needs a role that can bypass the `release` ruleset. If Git refuses because the push is not a fast-forward, `main` has a commit `dev` lacks: merge `main` into `dev` with a merge commit first.
3. Publish a GitHub Release with a new three-part `vX.Y.Z` tag on the same commit, e.g. `v0.6.0` rather than `v0.6`:

   ```bash
   gh release create vX.Y.Z --target "$(git rev-parse <sha>)" --generate-notes
   ```

   `--target` needs the full 40-character SHA: GitHub rejects an abbreviated one with "Release.target_commitish is invalid".

   Publishing the release triggers the PyPI upload in `build-wheels.yml`, which accepts tags starting with `v0.` or `v1.`.

   It also starts the CUDA wheels, which take under an hour:
   - `build-cuda-wheels.yml` builds every CUDA wheel in `cuda_wheel_matrix.json`, tests one per torch/CUDA build on a Modal GPU, and attaches them all to the release. It attaches none unless every build and test passes.
   - `deploy-cuda-index.yml` then rebuilds the CUDA wheel index on GitHub Pages from every release's assets.

   If a CUDA job fails, re-run it from the Actions tab. The upload skips wheels already on the release with the same contents and refuses different ones, because a published release's assets are never replaced. A failure that needs a code change needs a new patch release.

   To rebuild the index without a release, for example after deleting one, run "Deploy CUDA wheel index" from the Actions tab.

   setuptools_scm versions `dev` builds by bumping the tag's last component. After `v0.6.0`, `dev` builds as `0.6.1.devN`, and versions keep increasing whichever release comes next. A two-part tag breaks that: after `v0.5`, `dev` built as `0.6.devN`, and once `v0.5.1` came out, the newer `0.5.2.devN` builds sorted below those older TestPyPI uploads.

Do not merge a pull request into `main`. GitHub creates a new commit for every merge method, even rebase, so the tag would land on a commit outside `dev`'s history. Builds from `dev` would then fall back to the previous version, as happened with `v0.5`.
