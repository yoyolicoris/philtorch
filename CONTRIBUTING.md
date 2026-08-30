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

Python formatting is enforced by Black in CI. If Black is available in your development environment, check changed Python files with:

```bash
black --check philtorch tests
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
