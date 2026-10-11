# AGENTS.md

Guidance for coding agents working in this repository. It is the one shared source of instructions: Claude Code, GitHub Copilot, Antigravity and Muse Code all read it. Keep tool-specific notes in that tool's own file, which only adds to this one: `CLAUDE.md` (Claude Code), `.github/copilot-instructions.md` and `.github/instructions/` (Copilot), `GEMINI.md` (Antigravity).

## Commands

Pixi defines the environment (`pixi.toml`); the default environment installs philtorch editable, with its compiled extension. On Linux, wrap builds and full test runs in the resource limiter (defaults: 3 CPUs, 12 GiB, two native build jobs; override with `PHILTORCH_CPU_QUOTA`, `PHILTORCH_MEMORY_MAX`, `MAX_JOBS`, `OMP_NUM_THREADS`):

```bash
./scripts/run-limited pixi reinstall philtorch   # rebuild after changing philtorch/csrc/*
./scripts/run-limited pixi run pytest            # full suite

pixi run pytest tests/test_lti_lfilter.py                                 # one file
pixi run pytest tests/test_lti_lfilter.py::test_time_invariant_filter     # one test

pixi run ruff check .            # lint, as CI does (Ruff pinned in pixi.toml)
pixi run ruff format --check .   # format check; format only files you changed
pixi run docs                    # Sphinx with -W: warnings, unresolved references included, are errors
```

Benchmarks live in `benchmarks/` and run as `pixi run python benchmarks/<name>.py`.

## Architecture

- **`philtorch._C` is required.** `philtorch/__init__.py` checks that the installed torch matches the one the extension was built for (`_build_check.py`), imports `_C`, and registers the fake and autograd implementations of the native ops (`torch.ops.philtorch.*`: `recur2`, `recurN`, `lti_recur*`, `scan`, `lpc`, PararNN). There is no pure-PyTorch fallback if the extension is missing; `setup.py`/`build_support.py` compile `philtorch/csrc/*.cpp`/`*.cu` (CUDA when available or `PHILTORCH_FORCE_CUDA=1`) and the Metal source on macOS.
- **Filters: `lti/` (time-invariant) and `lpv/` (parameter-varying).** The user-facing functions (`lfilter`, `filtfilt`, `state_space`, ...) normalize coefficients into a state-space recursion. `lti/ssm.py` and `lpv/ssm.py` are the engines that pick the runner (`_select_recursion_runner`). With `unroll_factor == 1` and either state size M ≤ 2 or a CPU input (`extension_backend_indicator`), they call the native extension. Otherwise they run a PyTorch loop over time (`_recursion_loop`). The LPV `torchlpc` backend is vendored (`_torchlpc.py`, `csrc/torchlpc_*`).
- **`philtorch/_recur.py`** is one operator for the recurrence h[t] = A[t] h[t − 1] + x[t] of any state size M, not yet called by the filters. One autograd function defines its derivatives; its forward picks a native kernel by device, M and dtype, or, on CUDA, the Triton scan in `_recur_kernels.py`, and raises `NotImplementedError` where none applies, with no PyTorch fallback.
- **`estimation/` (Kalman, HMM) and `align/` (DTW, soft-DTW, CTC).**
  - Kalman is pure PyTorch with a parallel scan.
  - HMM and alignment run as hand-written Triton kernels, on CUDA only. Every entry point calls `philtorch._triton.check_cuda_triton`, and its docstring carries the shared `Note:` from that module. Triton is imported inside functions, so `philtorch.align` and `philtorch.estimation` import without it.
  - The kernels are wrapped as `torch.library.custom_op`s with `register_autograd`. Their backward passes are themselves differentiable ops, so gradients exist to any order.
  - `philtorch/_trace_kernels.py` is the chunked parallel traceback shared by `hmm_viterbi` and `forced_align`.
  - CTC reuses the DTW kernels: it is soft-DTW over a third step set, `"ctc"`.
  - The module docstring of `align/_dtw_kernels.py` explains the row-scan design.
- Triton kernels live in modules named `_*_kernels.py`: the CUDA workflow's path filter picks them out by that name, so a new kernel module must follow it, and its tests be added to the filter.
- Kernels are prototyped and tuned in Helion, a development tool the package never imports, and shipped as Triton with their configurations fixed: there is no autotuning at runtime. The algorithms in `estimation/` and `align/` ship only parallel implementations; sequential recursions belong in tests, as references.
- `philtorch/prototype/` holds unreleased research code.

## Conventions

- **Coefficients:** the denominator `a` excludes `a0`, so SciPy-equivalent calls prepend `1.0` (in tests, `[1.0] + a.tolist()`). `lfilter_zi` takes `(a, b)`, the reverse of SciPy's order.
- **Shapes:** LTI functions take static coefficients, e.g. `(N,)` or `(B, N)`. LPV functions take time-varying ones, e.g. `(B, T, M)`. The alignment and estimation APIs are batch first: `(B, T, C)`, not PyTorch's `(T, B, C)`.
- **Choices are explicit arguments:**
  - filter forms: `df2`, `tdf2`, `df1`, `tdf1`;
  - LTI `lfilter` backends: `"ssm"`, `"diag_ssm"`;
  - LPV `lfilter` backends: `"ssm"`, `"torchlpc"`.
- **Docstrings** follow PyTorch's conventions, as detailed in `CONTRIBUTING.md`:
  - Google style, lines of 80 characters or fewer;
  - math in reStructuredText;
  - `Raises:` lists the errors bad input triggers; argument errors are `ValueError`s, not asserts.
- **New public functions** need an entry on their page in `docs/api/*.rst`.
- **Docstring examples:** if a module's docstrings have examples, add the module to `tests/test_docstrings.py`: `MODULES`, or `CUDA_MODULES` for Triton code (checked only on CUDA machines). Load it with `importlib.import_module` when a function of the same name shadows it, as `philtorch.align.dtw` does.

## Tests and CI

- pytest runs with `--import-mode=importlib`, so the tests import the installed philtorch, with its compiled extension, rather than the source tree. A test that runs a subprocess should pass `cwd=tmp_path`, or the subprocess imports the checkout.
- Mark tests that need a CUDA device with `pytest.mark.cuda`, on the test or on its `"cuda"` parameter. `tests/conftest.py` skips them where no CUDA device is available, as on the CPU CI runners, and the CUDA workflow (`build-cuda-wheels.yml`) runs exactly them, `-m cuda`, on a Modal GPU, uploading their coverage to Codecov with the `gpu` flag, which `codecov.yml` carries forward to commits that don't run it. Coverage leaves out `@triton.jit` functions, which it can't trace, and `tests/conftest.py` runs autograd's backward passes on the calling thread, since coverage can't trace the thread autograd starts for CUDA.

## Branches and releases

- Branch from `dev` and open pull requests against `dev`; PR bodies follow `.github/pull_request_template.md`.
- Never merge into `main`. A release fast-forwards `main` to a `dev` commit and tags it with a three-part `vX.Y.Z` tag (setuptools_scm), as described in `CONTRIBUTING.md`.
