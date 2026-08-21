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
| Op registration | `TORCH_LIBRARY_IMPL` (non-stable ABI — one wheel per torch minor version required) |
| Python versions supported | 3.10, 3.11, 3.12 (declared in `pyproject.toml`) |
| Torch versions to target | 2.6.x, 2.7.x (latest two stable minor releases; extend as new versions ship) |
| OS targets | `ubuntu-latest` (Linux x86\_64), `macos-latest` (macOS arm64, Apple M-series), `windows-latest` (Windows x86\_64) |
| Wheel count | 3 Python × 2 torch × 3 OS = **18 wheels** per release |
| PyPI strategy | Single package name `philtorch`; wheels tagged with platform + cpython version; torch compatibility documented in README |

---

## Step 1 — Audit and fix `setup.py` for cross-platform wheel builds

**File**: `setup.py`

### 1.1 — Add an OpenMP opt-out escape hatch

**Why**: `setup.py` calls `torch.backends.openmp.is_available()` to decide
whether to enable OpenMP. On macOS this then calls `brew --prefix llvm` and
`brew --prefix libomp`, raising a hard `RuntimeError` if they are not
installed. GitHub-hosted `macos-latest` runners have Homebrew but do **not**
pre-install `llvm` or `libomp`, so the build would fail before any C++
compilation starts. While the CI install step (Step 3.3) handles the normal
case, an env-var escape hatch (`PHILTORCH_DISABLE_OPENMP=1`) provides a
safety valve for debugging, minimal builds, or future runners where OpenMP
is unavailable or undesirable without requiring a code change:

```python
def get_extensions():
    use_openmp = (
        torch.backends.openmp.is_available()
        and os.environ.get("PHILTORCH_DISABLE_OPENMP", "0") != "1"
    )
    ...
```

### 1.2 — Confirm CUDA is correctly skipped on CPU-only runners (no change needed)

**Why**: verifying this explicitly avoids surprises. `setup.py` line 70
already checks `torch.cuda.is_available() and CUDA_HOME is not None`.
On all GitHub-hosted runners (ubuntu/macos/windows), `torch.cuda.is_available()`
returns `False` and `CUDA_HOME` is `None`, so `.cu` sources are already
excluded and `CppExtension` is used instead of `CUDAExtension`. **No code
change needed**, but the CI matrix must install the `+cpu` variant of torch
(see Step 3) to guarantee this — if the CUDA variant of torch were
accidentally installed on a Linux runner, `torch.cuda.is_available()` could
return `True` even without a GPU, causing the CUDA build path to activate and
fail on missing `nvcc`.

### 1.3 — Fix the Windows OpenMP compiler flag

**Why**: the current code unconditionally passes `-fopenmp` as the C++
compile flag and linker flag. This is a GCC/Clang flag and is not recognised
by MSVC, which is the default compiler on `windows-latest`. MSVC uses
`/openmp` instead, and unlike GCC/Clang, does not require an explicit linker
flag (OpenMP is linked automatically). Without this fix the Windows build will
fail with an unrecognised compiler option error:

```python
if use_openmp:
    if sys.platform == "win32":
        extra_compile_args["cxx"] = ["/openmp"]
        # MSVC links OpenMP automatically — no extra_link_args entry needed
    else:
        extra_compile_args["cxx"] = ["-fopenmp"]
        extra_link_args.append("-fopenmp")
    if sys.platform == "darwin":
        ...  # existing macOS homebrew logic, unchanged
```

### 1.4 — Understand why Linux needs `cibuildwheel` (no code change, awareness only)

**Why**: a wheel built directly on `ubuntu-latest` is tagged
`linux_x86_64`, which PyPI rejects for public upload and which pip will
refuse to install on most Linux systems because it makes no guarantee about
glibc version compatibility. The standard is `manylinux`, a policy that
requires building inside a Docker container with an old enough glibc so the
resulting `.so` files run on any reasonably modern Linux distro. `cibuildwheel`
automates this — it pulls the correct `manylinux_2_28` Docker image and runs
the build inside it, producing a correctly tagged `manylinux_2_28_x86_64`
wheel automatically. macOS and Windows do not have this issue and can use
plain `python -m build` directly.

---

## Step 2 — Update `pyproject.toml`

**File**: `pyproject.toml`

### 2.1 — Remove `torch` and `numpy` from `build-system.requires`

**Why**: `build-system.requires` lists packages that pip installs into an
isolated build environment before running `setup.py`. Having `torch` here
means pip will always install the **latest** torch from PyPI into that
environment, regardless of which torch version the wheel is being built
against. For the CI matrix (Step 3), the workflow installs a specific pinned
torch version (`2.6.0+cpu`, `2.7.0+cpu`, etc.) before invoking the build.
If `torch` remains in `build-system.requires`, pip may overwrite that pinned
installation with the latest torch, breaking the version targeting. The fix
is to remove `torch` (and `numpy`, which has the same problem and is not
actually needed at C++ compile time) from here, and rely on the CI step to
provide the correct torch in the environment. The `--no-isolation` flag in
the build command (Step 3) tells `python -m build` to use the already-active
environment rather than creating a fresh isolated one, which is how the
pre-installed torch is picked up:

```toml
[build-system]
requires = [
    "setuptools >= 77.0.3",
    "setuptools-git-versioning>=2.0,<3",
    "wheel",
]
build-backend = "setuptools.build_meta:__legacy__"
```

### 2.2 — Verify whether Python 3.13 should be added

**Why**: `pyproject.toml` currently declares support for Python 3.10–3.12
only. Python 3.13 was released in October 2024 and torch 2.6+ ships `cp313`
wheels. If the torch versions in the matrix (`2.6.0`, `2.7.0`) provide
`cp313` wheels at `https://download.pytorch.org/whl/cpu`, then philtorch
should also ship `cp313` wheels and declare support in the classifiers —
otherwise users on Python 3.13 get no pre-built wheel and must compile from
source. Before adding it, verify with:

```bash
pip index versions torch --index-url https://download.pytorch.org/whl/cpu
```

and confirm `cp313` tags exist. If confirmed, add to `pyproject.toml`:

```toml
"Programming Language :: Python :: 3.13",
```

and include `"3.13"` in the CI matrix (Step 3). If not confirmed, leave
3.13 out of both the classifiers and the matrix.

---

## Step 3 — Create the wheel build workflow

**File**: `.github/workflows/build-wheels.yml` (new file)

**Why a new file rather than modifying existing workflows**: the existing
`python-publish-stable.yml` triggers on release events and pushes to `main`,
and is responsible for the sdist. Mixing wheel builds into it would make the
job graph much more complex and harder to debug. A dedicated
`build-wheels.yml` keeps concerns separate: wheels are built and published
here; the sdist continues to be built and published by the existing workflow.
Both upload to the same PyPI project — pip will prefer wheels over sdist
automatically when a compatible wheel is available.

### 3.1 — Workflow triggers

**Why `workflow_dispatch` in addition to tag push**: tag-triggered runs are
the production path, but `workflow_dispatch` lets you test the full matrix
(or a single OS job) without cutting a release tag. This is important during
initial setup when you need to iterate on the build configuration.

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

**Why `cibuildwheel` for Linux only and not macOS/Windows**: as explained in
Step 1.4, Linux requires the `manylinux` container environment to produce
installable wheels. `cibuildwheel` is the standard tool for this. macOS and
Windows do not have the same requirement and their platform tags
(`macosx_11_0_arm64`, `win_amd64`) are produced correctly by plain
`python -m build`, so using `cibuildwheel` there would add unnecessary
complexity without benefit.

**Why `manylinux_2_28`**: this image is based on AlmaLinux 8 / glibc 2.28
(released 2018). It is the minimum glibc version that torch's own Linux
wheels target, so it is consistent and ensures the philtorch wheel will
install alongside torch on any system that can already run torch.

**Why `--index-url` and not `--extra-index-url`**: `--extra-index-url` adds
PyTorch's wheel index as a secondary source but still tries PyPI first.
PyPI hosts a `torch` package too (the CUDA variant), and pip may pick that
one instead of the CPU-only `+cpu` variant. Using `--index-url` replaces the
default index entirely, ensuring only the PyTorch wheel server is consulted
for this install.

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
        # fetch-depth: 0 is required so setuptools-git-versioning can read
        # the full git history and tags to derive the package version number.

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

### 3.3 — macOS arm64 job

**Why `macos-latest` and not `macos-13` or an x86_64 runner**: as of late
2023, GitHub's `macos-latest` resolves to an M-series (arm64) runner.
Apple Silicon is now the dominant macOS platform for developers and ML
practitioners. x86_64 macOS (Intel) is a diminishing user base and can still
install from source. Starting with arm64 only keeps the matrix size down.

**Why `brew install llvm libomp`**: `setup.py`'s `get_macos_openmp_config()`
locates the OpenMP headers and compiler via `brew --prefix llvm` and
`brew --prefix libomp`. These formulae are not pre-installed on GitHub
runners. Without this step the build fails immediately with a Homebrew error
before any compilation starts.

**Why `MACOSX_DEPLOYMENT_TARGET=11.0`**: this env var tells the compiler and
linker the minimum macOS version the binary must run on. `11.0` (Big Sur) is
the minimum for Apple Silicon hardware. Setting it explicitly ensures the
wheel's platform tag is `macosx_11_0_arm64`, which is the broadest
compatible tag for arm64. If left unset, the runner's own OS version
(e.g., `14.x`) would be used, producing a wheel that pip refuses to install
on older macOS versions even if the binary would actually work.

```yaml
build-macos:
  runs-on: macos-latest
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

### 3.4 — Windows job

**Why no special toolchain setup for Windows**: `windows-latest` comes with
Visual Studio Build Tools (MSVC) pre-installed, which is what
`torch.utils.cpp_extension.BuildExtension` uses on Windows. After the
`setup.py` fix in Step 1.3, MSVC's `/openmp` flag is used correctly. No
additional compiler installation is needed.

**Why `.mm` files are not a concern on Windows**: `setup.py` adds
Objective-C++ `.mm` sources only inside `if sys.platform == "darwin"`.
Windows skips this branch automatically, so no special handling is required.

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

### 3.5 — Publish job

**Why a single publish job that waits for all three build jobs**: uploading
all wheels in one step ensures an atomic release — either all platform wheels
for a given tag appear on PyPI together, or none do. Publishing per-OS would
risk a partial release where, for example, Linux wheels are live but macOS
wheels failed.

**Why `if: startsWith(github.ref, 'refs/tags/v')`**: when the workflow is
triggered via `workflow_dispatch` for testing, you do not want it to publish
to PyPI. This guard ensures publishing only happens on actual version tag
pushes.

**Why reuse the existing `pypi` environment**: the `python-publish-stable.yml`
workflow already has trusted publishing configured for the `pypi` environment.
Reusing it means no new PyPI or GitHub environment configuration is needed.

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

---

## Step 4 — Keep existing publish workflows building the sdist

**Files**: `python-publish.yml`, `python-publish-stable.yml`

**Why keep the sdist**: the sdist is the fallback for users on platforms not
covered by the wheel matrix (e.g., Linux aarch64, older macOS Intel, custom
CUDA builds). PyPI allows both sdist and wheels under the same package name —
pip always prefers a compatible wheel and falls back to the sdist only if no
wheel matches. Removing the sdist would break source installs entirely for
those users. No changes to these files are needed.

---

## Step 5 — Verify torch+cpu index URLs before implementing

**Why do this before writing the matrix**: the exact version strings and `+cpu`
suffix must match what the PyTorch wheel server actually hosts. If a version
string is wrong the `pip install torch==x.y.z+cpu` step will fail silently or
with a confusing error, wasting CI time. Run this once locally first:

```bash
pip index versions torch --index-url https://download.pytorch.org/whl/cpu
```

Confirm:
- `torch==2.6.0+cpu` exists
- `torch==2.7.0+cpu` exists
- `cp313` wheels exist for both versions (needed if Python 3.13 is added per Step 2.2)

Adjust the matrix versions accordingly before creating the workflow file.

---

## Step 6 — Test the workflow manually before wiring to tags

**Why test via `workflow_dispatch` first**: each OS has different failure
modes (macOS OpenMP, Windows MSVC flags, Linux manylinux container). Testing
all three in one go on a live tag makes it hard to iterate quickly. Using
`workflow_dispatch` lets you trigger and fix one OS at a time without
polluting the release history. Suggested order:

1. Linux — fastest runner, most likely to succeed first, validates the
   `cibuildwheel` + `manylinux` setup
2. macOS arm64 — validates the `brew install llvm libomp` step and
   `MACOSX_DEPLOYMENT_TARGET`
3. Windows — validates the MSVC `/openmp` flag fix

After each job succeeds, download the artifact and run a minimal smoke test:

```bash
pip install philtorch-*.whl --force-reinstall
python -c "import philtorch; print(philtorch.__version__)"
```

---

## Step 7 — Update `pyproject.toml` classifiers and README

**Why**: PyPI displays classifier information to users browsing the package
page. If the classifiers still list only Python 3.10–3.12 but wheels for
3.13 are published, the package page is misleading. Similarly, users need to
know which torch versions have pre-built wheels so they know whether to
`pip install philtorch` directly or compile from source. Add to `README.md`:

```
Pre-built wheels are available for torch 2.6 and 2.7 on Linux (x86_64),
macOS (arm64), and Windows (x86_64) for Python 3.10–3.13.
For other configurations (CUDA, older torch, Linux aarch64), install from
source: pip install philtorch --no-binary philtorch
```

---

## Step 8 — Pin `cibuildwheel` to a specific version

**Why**: `cibuildwheel` releases frequently and occasionally introduces
breaking changes to environment variable names, default image versions, or
build behaviour. Pinning to a specific version (e.g., `2.23.3`) ensures the
workflow behaves identically on every run. Check the
[cibuildwheel releases page](https://github.com/pypa/cibuildwheel/releases)
for the latest stable version at implementation time and pin to that.

---

## Checklist for the implementing agent

- [ ] **Step 1.1** — Add `PHILTORCH_DISABLE_OPENMP` env-var escape hatch to `setup.py`
- [ ] **Step 1.3** — Add Windows OpenMP flag (`/openmp`) guard to `setup.py`
- [ ] **Step 2.1** — Remove `torch` and `numpy` from `build-system.requires` in `pyproject.toml`
- [ ] **Step 2.2** — Verify Python 3.13 torch+cpu wheel availability; add `cp313` to matrix and classifiers if confirmed
- [ ] **Step 5** — Verify torch+cpu index URLs for `2.6.0` and `2.7.0` before finalising matrix
- [ ] **Step 3** — Create `.github/workflows/build-wheels.yml` with Linux, macOS, Windows, and publish jobs
- [ ] **Step 4** — Confirm existing publish workflows still build sdist (no change expected)
- [ ] **Step 6** — Trigger workflow manually and validate each OS artifact
- [ ] **Step 7** — Update `README.md` and `pyproject.toml` classifiers
- [ ] **Step 8** — Pin `cibuildwheel` version in `build-wheels.yml`

---

## Files modified/created by this plan

| File | Action | Reason |
|---|---|---|
| `setup.py` | Modify | OpenMP escape hatch (Step 1.1) + Windows `/openmp` flag (Step 1.3) |
| `pyproject.toml` | Modify | Remove torch/numpy from build-system.requires (Step 2.1); add py313 classifier if applicable (Step 2.2) |
| `.github/workflows/build-wheels.yml` | **Create** | New wheel build + publish workflow (Step 3) |
| `README.md` | Modify | Document wheel availability and torch version compatibility (Step 7) |

---

## Deferred to future plan: CUDA wheels

CUDA wheel builds are explicitly out of scope for this plan. When ready, the
recommended approach is to add a `build-linux-cuda` job to the same
`build-wheels.yml` workflow using an online GPU-enabled CI service
(Namespace.so or Cirun.io as `runs-on` targets), building Linux CUDA wheels
only, against a matrix of CUDA versions (`cu121`, `cu124`) × torch minor
versions. No GPU runner is needed at compile time — only `nvcc` and the CUDA
toolkit. A GPU runner is only required for runtime smoke tests.
