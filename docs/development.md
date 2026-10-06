# Development

The [contribution guide](https://github.com/yoyolicoris/philtorch/blob/dev/CONTRIBUTING.md) covers the development setup, tests, docstring conventions, pull requests and releases. Report bugs and request features on the [issue tracker](https://github.com/yoyolicoris/philtorch/issues).

## Building these docs

The docs import the real package, so PhilTorch must be built with its compiled extension. With [pixi](https://pixi.sh), whose default environment builds PhilTorch and includes the docs tools, run

```bash
pixi run docs
```

and open `docs/_build/html/index.html`. `pixi run docs-linkcheck` checks the external links.

Without pixi, from a checkout, install a CPU build of PyTorch, then PhilTorch in editable mode and the docs dependencies:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install "setuptools>=77.0.3" "setuptools_scm>=8" wheel ninja
python -m pip install --no-build-isolation --editable .
python -m pip install -r docs/requirements.txt
sphinx-build -W --keep-going -b html docs docs/_build/html
```

The build treats warnings as errors, including references that do not resolve. CI runs the same command on pull requests that touch the docs or the package, and checks the external links once a week.
