# CPU Wheel Build — Action Plan

> **Goal**: Ship pre-built CPU-only wheels for philtorch to PyPI for all
> supported Python versions, PyTorch minor versions, and OS targets (Linux,
> macOS arm64, Windows), without touching CUDA or the Stable ABI.
>
> **Out of scope for this plan**: CUDA wheels, PyTorch Stable ABI migration,
> macOS x86_64, Linux aarch64.

---

## Context and constraints

| Item | Current state |
|---|---|
| Build system | `setup.py` + `pyproject.toml` (setuptools) |
| Existing publish workflows | `python-publish.yml` (→ TestPyPI on `dev` push, sdist only) and `python-publish-stable.yml` (→ PyPI on release/main, sdist only) |
| C++ sources | `philtorch/csrc/*.cpp` + `.mm` (macOS MPS) + `.cu` (CUDA, skipped on CPU) |
| Op registration | `TORCH_LIBRARY_IMPL` (non-stable ABI — acceptable, one wheel per torch minor version) |
| Python versions supported | 3.10, 3.11, 3.12 (declared in `pyproject.toml`); add 3.13 if torch supports it |
| Torch versions to target | 2.6.x, 2.7.x (latest two stable minor releases; extend as new versions ship) |
| OS targets | `ubuntu-latest` (Linux x86\_64), `macos-latest` (macOS arm64, Apple M-series), `windows-latest` (Windows x86\_64) |
| Wheel count | 3 Python × 2 torch × 3 OS = **18 wheels** per release |
| PyPI strategy | Single package name `philtorch`; wheels tagged with platform + cpython version; torch compatibility documented in README |

---

## Step 1 — Audit and fix `setup.py` for cross-platform wheel builds

**File**: `setup.py`

### 1.1 — Remove implicit OpenMP requirement on macOS CI

Currently `get_macos_openmp_config()` calls `brew --prefix llvm` and
`brew --prefix libomp` and raises a hard error if they are missing.
GitHub-hosted `macos-latest` runners are Apple Silicon and have Homebrew
available, but `llvm` and `libomp` must be explicitly installed in CI.

**Action**: add a CI install step (see Step 4), and optionally add a
`PHILTORCH_DISABLE_OPENMP=1` env-var escape hatch in `setup.py` so builds
can opt out of OpenMP entirely when needed:

```python
def get_extensions():
    use_openmp = (
        torch.backends.openmp.is_available()
        and os.environ.get("PHILTORCH_DISABLE_OPENMP", "0") != "1"
    )
    ...
```

### 1.2 — Confirm CUDA is correctly skipped on CPU-only runners

`setup.py` line 70 already checks `torch.cuda.is_available() and CUDA_HOME is not None`.
On all GitHub-hosted runners (ubuntu/macos/windows), both will be `False`/`None`,
so CUDA and `.cu` sources are already skipped. **No change needed here.**

### 1.3 — Fix Windows OpenMP flag

`setup.py` uses `-fopenmp` (GCC/Clang flag). MSVC uses `/openmp` instead.
Add a platform guard:

```python
if use_openmp:
    if sys.platform == "win32":
        extra_compile_args["cxx"] = ["/openmp"]
        # no extra link arg needed for MSVC — OpenMP is linked automatically
    else:
        extra_compile_args["cxx"] = ["-fopenmp"]
        extra_link_args.append("-fopenmp")
    if sys.platform == "darwin":
        ...  # existing macOS homebrew logic, unchanged
```

### 1.4 — Verify `manylinux` compatibility of Linux builds

Linux wheels must be tagged `manylinux` (not `linux_x86_64`) to be
installable on most systems. This requires building inside a `manylinux`
Docker container. Use `cibuildwheel` for Linux only (it handles this
automatically). See Step 3.

---

## Step 2 — Update `pyproject.toml`

**File**: `pyproject.toml`

### 2.1 — Add Python 3.13 to classifiers (if torch 2.6+ supports it)

Verify torch 2.6 and 2.7 ship cp313 wheels on PyPI, then add:

```toml
"Programming Language :: Python :: 3.13",
```

to the `classifiers` list and update `requires-python = ">=3.10"` (no
change needed there).

### 2.2 — Remove `torch` and `numpy` from `build-system.requires`

Currently:
```toml
[build-system]
requires = [
    "setuptools >= 77.0.3",
    "setuptools-git-versioning>=2.0,<3",
    "wheel",
    "torch",
    "numpy",
]
```

`torch` in `build-system.requires` is problematic for wheel builds because
pip will install the *latest* torch into the build environment regardless of
the torch version the wheel is targeting. The CI workflow will install the
correct torch version explicitly *before* calling the build, so remove
`torch` and `numpy` from here:

```toml
[build-system]
requires = [
    "setuptools >= 77.0.3",
    "setuptools-git-versioning>=2.0,<3",
    "wheel",
]
build-backend = "setuptools.build_meta:__legacy__"
```

`setup.py` imports `torch` at module level, so torch must still be installed
before running the build — the workflow handles this (Step 4). Using
`build-system.requires` for torch is the wrong place for version-specific
dependencies.

### 2.3 — Pin `numpy` as a runtime optional dependency if needed

`numpy` was in build-system requires; check whether it is actually needed at
build time (it is likely only needed at test time via `scipy`). If so, remove
it entirely from build requires. If a C extension header from numpy is
included in the C++ sources, add it to the correct place instead.

---

## Step 3 — Create the wheel build workflow

**File**: `.github/workflows/build-wheels.yml` (new file)

This workflow runs on:
- Every push of a version tag (`v*`) — for releases
- Manual trigger via `workflow_dispatch` — for testing the matrix

### 3.1 — Workflow triggers

```yaml
on:
  push:
    tags:
      - 'v*'
  workflow_dispatch:
    inputs:
      torch_version:
        description: 'Torch version to build against (e.g. 2.7.0)'
        required: false
        default: ''
```

### 3.2 — Linux job (via `cibuildwheel`)

Use `cibuildwheel` to produce `manylinux_2_28` tagged wheels. `cibuildwheel`
runs the build inside a Docker container automatically.

```yaml
build-linux:
  runs-on: ubuntu-latest
  strategy:
    fail-fast: false
    matrix:
      torch: ["2.6.0", "2.7.0"]
      python: ["310", "311", "312", "313"]
  steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0

    - uses: actions/setup-python@v5
      with:
        python-version: "3.x"

    - name: Install cibuildwheel
      run: pip install cibuildwheel==2.23.3

    - name: Build wheels
      run: cibuildwheel --output-dir wheelhouse
      env:
        CIBW_BUILD: "cp${{ matrix.python }}-manylinux_x86_64"
        CIBW_MANYLINUX_X86_64_IMAGE: manylinux_2_28
        CIBW_BEFORE_BUILD: >
          pip install torch==${{ matrix.torch }}+cpu
          --index-url https://download.pytorch.org/whl/cpu &&
          pip install setuptools>=77.0.3 setuptools-git-versioning>=2.0 wheel
        CIBW_ENVIRONMENT: >
          PHILTORCH_DISABLE_OPENMP=0
        CIBW_TEST_SKIP: "*"

    - uses: actions/upload-artifact@v4
      with:
        name: wheels-linux-cp${{ matrix.python }}-torch${{ matrix.torch }}
        path: wheelhouse/*.whl
```

> **Note on `torch+cpu` index**: PyTorch CPU-only wheels are published at
> `https://download.pytorch.org/whl/cpu`. The package name is
> `torch==2.7.0+cpu`. Use `--index-url` (not `--extra-index-url`) to avoid
> accidentally pulling the CUDA variant.

### 3.3 — macOS arm64 job

```yaml
build-macos:
  runs-on: macos-latest   # macos-latest is arm64 (M-series) as of 2024
  strategy:
    fail-fast: false
    matrix:
      torch: ["2.6.0", "2.7.0"]
      python: ["3.10", "3.11", "3.12", "3.13"]
  steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0

    - uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python }}

    - name: Install Homebrew dependencies for OpenMP
      run: brew install llvm libomp

    - name: Install torch and build dependencies
      run: |
        pip install torch==${{ matrix.torch }} --index-url https://download.pytorch.org/whl/cpu
        pip install setuptools>=77.0.3 setuptools-git-versioning>=2.0 wheel build

    - name: Build wheel
      run: python -m build --wheel --no-isolation
      env:
        MACOSX_DEPLOYMENT_TARGET: "11.0"

    - uses: actions/upload-artifact@v4
      with:
        name: wheels-macos-${{ matrix.python }}-torch${{ matrix.torch }}
        path: dist/*.whl
```

> **Note on macOS OpenMP**: `brew install llvm libomp` is required before
> the build. `setup.py` calls `brew --prefix llvm` and `brew --prefix libomp`
> at build time to locate the compiler and headers. The GitHub `macos-latest`
> runner has Homebrew pre-installed, so `brew install` works.
>
> **Note on `MACOSX_DEPLOYMENT_TARGET`**: set to `11.0` (macOS Big Sur) which
> is the minimum for Apple Silicon. This controls the platform tag in the wheel
> filename (e.g., `macosx_11_0_arm64`).

### 3.4 — Windows job

```yaml
build-windows:
  runs-on: windows-latest
  strategy:
    fail-fast: false
    matrix:
      torch: ["2.6.0", "2.7.0"]
      python: ["3.10", "3.11", "3.12", "3.13"]
  steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0

    - uses: actions/setup-python@v5
      with:
        python-version: ${{ matrix.python }}

    - name: Install torch and build dependencies
      run: |
        pip install torch==${{ matrix.torch }}+cpu --index-url https://download.pytorch.org/whl/cpu
        pip install setuptools>=77.0.3 setuptools-git-versioning>=2.0 wheel build

    - name: Build wheel
      run: python -m build --wheel --no-isolation

    - uses: actions/upload-artifact@v4
      with:
        name: wheels-windows-${{ matrix.python }}-torch${{ matrix.torch }}
        path: dist/*.whl
```

> **Note on Windows OpenMP**: after the `setup.py` fix in Step 1.3, MSVC will
> use `/openmp` automatically. No additional toolchain setup is needed on
> `windows-latest` (MSVC is pre-installed).
>
> **Note on `.mm` files**: `setup.py` adds `.mm` (Objective-C++) sources only
> on `sys.platform == "darwin"`. Windows will skip them automatically.

### 3.5 — Publish job

```yaml
publish:
  runs-on: ubuntu-latest
  needs: [build-linux, build-macos, build-windows]
  if: startsWith(github.ref, 'refs/tags/v')
  permissions:
    id-token: write
  environment:
    name: pypi
  steps:
    - uses: actions/download-artifact@v4
      with:
        pattern: wheels-*
        path: dist/
        merge-multiple: true

    - uses: pypa/gh-action-pypi-publish@release/v1
      with:
        packages-dir: dist/
```

> Upload all wheels from all matrix jobs in one publish step.
> Requires the `pypi` trusted publisher environment to be configured on PyPI
> (already done for the stable publish workflow).

---

## Step 4 — Update existing publish workflows

**Files**: `python-publish.yml`, `python-publish-stable.yml`

Both currently build only an sdist. After the wheel workflow is added:

- Keep building the sdist in these workflows so source installs remain
  available for users who want CUDA or unsupported platforms.
- The sdist and wheels can coexist on PyPI — pip will prefer the wheel
  when available.
- No other change needed to these files unless you want to consolidate
  them (optional, out of scope for this step).

---

## Step 5 — Verify torch+cpu index URLs for each version

Before writing the matrix torch versions in stone, verify the exact package
names published at `https://download.pytorch.org/whl/cpu`:

```bash
pip index versions torch --index-url https://download.pytorch.org/whl/cpu
```

Confirm:
- `torch==2.6.0+cpu` exists
- `torch==2.7.0+cpu` exists
- Python 3.13 wheels exist for both versions (`cp313` tag)

Adjust the matrix accordingly. If `2.7.0` is not yet released when
implementing, use the latest available minor (e.g., `2.6.0` only, then
add `2.7.0` when it ships).

---

## Step 6 — Test the workflow manually before wiring to tags

Use the `workflow_dispatch` trigger to test each OS job independently before
cutting a release tag. Suggested test order:

1. Linux only — fastest, most likely to succeed first
2. macOS arm64 — verify OpenMP brew install works
3. Windows — verify MSVC OpenMP flag and `.mm` exclusion

After each job, download the artifact and run a quick smoke test locally:

```bash
pip install philtorch-*.whl --force-reinstall
python -c "import philtorch; print(philtorch.__version__)"
```

---

## Step 7 — Update `pyproject.toml` classifiers and README

After confirming the wheel build works:

- Add platform/wheel availability badges to `README.md`
- Document torch version compatibility (e.g., "pre-built wheels available for
  torch 2.6 and 2.7; for other versions install from source")
- Update `pyproject.toml` classifiers to include any newly supported Python
  versions

---

## Step 8 — (Optional) Pin `cibuildwheel` version

Pin the `cibuildwheel` version in the workflow to avoid unexpected breakage
from upstream updates:

```yaml
pip install cibuildwheel==2.23.3
```

Check the [cibuildwheel releases](https://github.com/pypa/cibuildwheel/releases)
for the latest stable version at implementation time and pin to that.

---

## Checklist for the implementing agent

- [ ] **Step 1.1** — Add `PHILTORCH_DISABLE_OPENMP` env-var escape hatch to `setup.py`
- [ ] **Step 1.3** — Add Windows OpenMP flag (`/openmp`) guard to `setup.py`
- [ ] **Step 2.2** — Remove `torch` and `numpy` from `build-system.requires` in `pyproject.toml`
- [ ] **Step 2.1** — Add Python 3.13 classifier to `pyproject.toml` (after verifying torch support)
- [ ] **Step 3** — Create `.github/workflows/build-wheels.yml` with Linux, macOS, Windows, and publish jobs
- [ ] **Step 4** — Confirm existing publish workflows still build sdist (no change expected)
- [ ] **Step 5** — Verify torch+cpu index URLs for 2.6.0 and 2.7.0 before finalising matrix
- [ ] **Step 6** — Trigger workflow manually and validate each OS artifact
- [ ] **Step 7** — Update `README.md` and `pyproject.toml` classifiers
- [ ] **Step 8** — Pin `cibuildwheel` version

---

## Files modified/created by this plan

| File | Action |
|---|---|
| `setup.py` | Modify — OpenMP escape hatch + Windows `/openmp` flag |
| `pyproject.toml` | Modify — remove torch/numpy from build-system.requires, add py313 classifier |
| `.github/workflows/build-wheels.yml` | **Create** — new wheel build + publish workflow |
| `README.md` | Modify — document wheel availability and torch version compatibility |

---

## Deferred to future plan: CUDA wheels

CUDA wheel builds are explicitly out of scope. When ready, the recommended
approach is to use an online GPU-enabled CI service (Namespace.so or Cirun.io)
as an additional job in the same `build-wheels.yml` workflow, building Linux
CUDA wheels only, against a matrix of CUDA versions (cu121, cu124) ×
torch minor versions.
