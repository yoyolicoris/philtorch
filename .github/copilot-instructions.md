# Copilot instructions for `philtorch`

## Build, test, and lint commands

Use pixi as the source of truth for environment/setup in this repository.

On Linux, run build and test commands through the repository resource limiter:

```bash
./scripts/run-limited pixi reinstall philtorch
./scripts/run-limited pixi run pytest
```

The wrapper defaults to 3 CPUs, 12 GiB RAM, no swap, and two parallel native
build jobs. Override these with `PHILTORCH_CPU_QUOTA`, `PHILTORCH_MEMORY_MAX`,
`PHILTORCH_MEMORY_SWAP_MAX`, `MAX_JOBS`, or `OMP_NUM_THREADS`.

```bash
# install/build project from pixi.toml instructions
pixi install philtorch
```

```bash
# run full tests (pytest options are configured in pyproject.toml)
pixi run pytest

# run a single test file
pixi run pytest tests/test_lti_lfilter.py

# run a single test function
pixi run pytest tests/test_lti_lfilter.py::test_time_invariant_filter
```

```bash
# lint command used by CI
pixi run python -m pip install flake8
pixi run flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
pixi run flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
```

## Terminal usage (avoid hangs)

- For quick Python checks, use one-liner `pixi run python -c "code"`; avoid `pixi run python - << 'PY'` heredocs and do not add `timeout=15000` wrappers, as they leave the terminal appearing finished without returning.

## High-level architecture

`philtorch` provides differentiable digital filtering APIs in PyTorch, with backend dispatch to compiled kernels when available.

1. `philtorch/lti/*` implements **time-invariant** filters and recurrences.
2. `philtorch/lpv/*` implements **time-varying (parameter-varying)** filters and recurrences.
3. `philtorch/lti/ssm.py` and `philtorch/lpv/ssm.py` are the core state-space engines; they route execution between compiled ops (`torch.ops.philtorch.*`), optional Helion/PararNN paths, and pure PyTorch fallbacks.
4. `philtorch/__init__.py` loads/registers extension ops and fake/autograd registrations. `setup.py` builds native sources from `philtorch/csrc/*.cpp` and `*.cu`.

Typical flow: API-level `lfilter`/`state_space` normalization -> backend/form selection -> recurrence execution backend.

## Key conventions in this codebase

- **Filter coefficient convention**: denominator `a` excludes `a0`; SciPy-equivalent calls prepend `1.0` in tests (`[1.0] + a.tolist()`).
- **Shape contracts are strict**: LTI functions accept static coefficients (`(N,)`, `(B,N)` style), while LPV functions encode time-varying coefficients (`(B,T,M)` style).
- **Filter forms are explicit API choices**: use only `df2`, `tdf2`, `df1`, `tdf1`; behavior and state handling depend on form.
- **Backend choice is explicit and meaningful**:
  - LTI `lfilter`: `backend="ssm"` or `backend="diag_ssm"`.
  - LPV `lfilter`: `backend="ssm"` or `backend="torchlpc"`.
- **`unroll_factor` affects backend path**: in state-space code, values greater than 1 force pure PyTorch unrolled recursion instead of extension-backed recursion.
- **Extension loading is optional**: code must keep fallback behavior working when compiled extensions are unavailable.

<!-- mermaid-ai-skills:start -->
## Mermaid Diagrams

When the user asks to create, edit, or visualize a diagram, follow the
instructions in `.github/instructions/mermaid.instructions.md`.
<!-- mermaid-ai-skills:end -->
