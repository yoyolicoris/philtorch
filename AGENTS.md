# AGENTS.md

Guidance for coding agents working in this repository. It is the one shared source of instructions: Claude Code, GitHub Copilot, Antigravity and Muse Code all read it. Keep tool-specific notes in that tool's own file, which only adds to this one: `CLAUDE.md` (Claude Code), `.github/copilot-instructions.md` and `.github/instructions/` (Copilot), `GEMINI.md` (Antigravity).

## Before starting a task

Agree on the task with the maintainer before carrying it out. Where the request leaves its details, scope or changes open, ask: don't assume the maintainer sees it the way you do, and don't settle it yourself by building on a guess. Questions are welcome; a question costs less than work done on the wrong reading. Ask in particular before:

- widening or narrowing the scope, or adding work that wasn't asked for;
- changing behaviour users see, such as the dtypes, devices or shapes a function takes, even in a run the maintainer left to you: ask, or leave the change out, rather than decide it and mention it in a report;
- renaming, moving or deleting files or public names;
- choosing between designs with real trade-offs: give the options and a recommendation, and let the maintainer decide.

For a task from an issue, put the plan on the issue as a `plan` artifact: briefly, what changes, what stays, the trade-offs and the tests. Point the maintainer to it and wait for agreement. Revise that artifact rather than adding another, so that its history shows the revisions. Issue artifacts are in public preview; agents write them with GitHub's MCP server, `gh issue artifact` (gh 2.103 or later) or the REST API (`/repos/{owner}/{repo}/issues/{number}/artifacts`).

## Delegating work

Once a task is agreed, the main agent (the one talking with the maintainer) delegates the parts big enough to be worth running in parallel. A small task it does itself.

- **Split the work into parts with disjoint files,** so that no two agents edit the same file. Give each part to a mid-tier subagent with exact instructions: the files, the target behaviour, any text to use verbatim, and the checks to run.
- **A mid-tier subagent may split its part further** over small-tier subagents, on the same terms: disjoint files, exact instructions, and a review of the results. That is the deepest level: three in all, counting the main agent.
- **Parts that depend on others,** such as the tests of new behaviour, start once those are final.
- **Subagents don't commit or push.** The main agent commits, and pushes only when the maintainer asks.
- **Open questions go up.** A subagent can't ask the maintainer. If its instructions leave something open, it reports the question to its caller instead of deciding it. The main agent asks the maintainer, as "Before starting a task" says.
- **One working tree, environment and build are shared,** even by parts with disjoint files. Nobody rebuilds the extension while subagents run; the main agent does, between rounds.
- **Tests:** a subagent runs the tests its part exercises, selected by name, when they don't strain the machine, and reports which it ran and why they cover its part, so that the main agent can pass that on, as "Commands" asks. Builds and long or GPU-heavy runs are left to the main agent.
- **Benchmarks** never run in parallel, and only when no other process competes for the machine, the GPU above all. The main agent runs them last, after the subagents have finished.
- **The main agent collects the results, reviews every diff, corrects what is wrong, and reports to the maintainer.** It is responsible for the result; a subagent's report is a claim to check.
- **Without subagents:** an agent whose tool can't spawn them, or whose tool file defines no tiers, does the work itself.

Which model each tier is depends on the tool; its own file says, such as `CLAUDE.md`.

## Commands

Pixi defines the environment (`pixi.toml`); the default environment installs philtorch editable, with its compiled extension. On Linux, wrap builds and full test runs in the resource limiter (defaults: 3 CPUs, 12 GiB, two native build jobs; override with `PHILTORCH_CPU_QUOTA`, `PHILTORCH_MEMORY_MAX`, `MAX_JOBS`, `OMP_NUM_THREADS`):

```bash
./scripts/run-limited pixi reinstall philtorch   # rebuild after changing philtorch/csrc/*
./scripts/run-limited pixi run pytest            # full suite, left to CI

pixi run pytest tests/test_lti_lfilter.py::test_time_invariant_filter     # one test
pixi run pytest tests/test_recur.py -k "integers or dispatch"             # tests by name

pixi run ruff check .            # lint, as CI does (Ruff pinned in pixi.toml)
pixi run ruff format --check .   # format check; format only files you changed
pixi run docs                    # Sphinx with -W: warnings, unresolved references included, are errors
```

Locally, run only the tests that exercise the change, selected by name: node IDs or `-k` expressions, not whole files. Find them by searching the tests for the changed function, op or kernel, and state which tests you ran and why they cover the change. The full suite runs on CI; run it locally only when the maintainer asks.

Benchmarks live in `benchmarks/` and run as `pixi run python benchmarks/<name>.py`.

Keep the measurements behind an issue, its benchmark or error tables and the scripts that produced them, on the issue as a `generic` artifact, or an `embedded_html` one for charts. Update it when they are run again, so that earlier versions stay comparable. Where a chart says more than a table, such as a speed-up or an error against a size, attach it to the pull request or comment as a PNG: `gh pr create`, `pr edit`, `issue comment` and the like take `--attach <file>#<alt text>` (gh 2.103 or later), and rewrite a body's `![alt](./plot.png)` to the upload.

## Architecture

- **`philtorch._C` is required.** `philtorch/__init__.py` checks that the installed torch matches the one the extension was built for (`_build_check.py`), imports `_C`, and `_ops.py` registers the fake and autograd implementations of the native ops (`torch.ops.philtorch.*`: `recur2`, `recur3`, `recurN`, `lti_recur*`, `scan`, `lpc`, PararNN). There is no pure-PyTorch fallback if the extension is missing; `setup.py`/`build_support.py` compile `philtorch/csrc/*.cpp`/`*.cu` (CUDA when available or `PHILTORCH_FORCE_CUDA=1`) and the Metal source on macOS.
- **Filters: `lti/` (time-invariant) and `lpv/` (parameter-varying).** The user-facing functions (`lfilter`, `filtfilt`, `state_space`, ...) normalize coefficients into a state-space recursion. `lti/ssm.py` and `lpv/ssm.py` validate their own layouts of A and share one engine, `philtorch/_ssm.py`, on the operator's (B or 1, T or 1, M, M): with `unroll_factor=None`, the default, it runs the operator of `philtorch/_recur.py`, whose `NotImplementedError` for want of a kernel reaches the user with a hint: there is no silent fallback. An integer `unroll_factor` runs PyTorch ops on any device, in the dtypes their matrix products take there (on CUDA, no integers): the sequential loop at 1 or from the sequence length on, and in between a block-unrolled recursion, by A's powers when time-invariant and by cumulative products when time-varying. Both modules' `linear_recurrence`, the scalar case, runs the same engine's `scalar_recursion`. The LPV `torchlpc` backend is vendored (`_torchlpc.py`, `csrc/torchlpc_*`).
- **`philtorch/_recur.py`** is one operator for the recurrence h[t] = A[t] h[t − 1] + x[t] of any state size M, which the filters' `state_space_recursion` runs. One autograd function defines its derivatives; its forward picks a native kernel by device, M and dtype, or, on CUDA, the Triton scan in `_recur_kernels.py`, and raises `NotImplementedError` where none applies, with no PyTorch fallback.
- **`estimation/` (Kalman, HMM) and `align/` (DTW, soft-DTW, CTC).**
  - Kalman is pure PyTorch with a parallel scan.
  - HMM and alignment run as hand-written Triton kernels, on CUDA only. Every entry point calls `philtorch._triton.check_cuda_triton`, and its docstring carries the shared `Note:` from that module. Triton is imported inside functions, so `philtorch.align` and `philtorch.estimation` import without it.
  - The kernels are wrapped as `torch.library.custom_op`s with `register_autograd`. Their backward passes are themselves differentiable ops, so gradients exist to any order.
  - `philtorch/_trace_kernels.py` is the chunked parallel traceback shared by `hmm_viterbi` and `forced_align`.
  - CTC reuses the DTW kernels: it is soft-DTW over a third step set, `"ctc"`.
  - The module docstring of `align/_dtw_kernels.py` explains the row-scan design.
- Triton kernels live in modules named `_*_kernels.py`: the CUDA workflow's path filter picks them out by that name, so a new kernel module must follow it. The filter also lists `_recur.py`, `_ops.py`, the CUDA-only modules `align/*.py` and `estimation/hmm.py`, and exactly the test modules with `cuda`-marked tests; `tests/test_ci_paths.py` fails when one is missing or another is listed. Keep CUDA cases out of mostly-CPU modules: any change to a listed module runs the CUDA build.
- Kernels are prototyped and tuned in Helion, a development tool the package never imports, and shipped with their configurations fixed: there is no autotuning at runtime. A new kernel ships as plain CUDA in `philtorch/csrc/`, as the native recurrence ops (`recur2`, `recur3`, `recurN`, `lti_recur*`) do, when that is practical; one that would be complex in CUDA and is much simpler in Triton ships as Triton. cuTile waits until it runs on every architecture the CUDA wheels target, Hopper included, so that no op needs two GPU kernels. The Triton kernels already shipped stay unless porting one to CUDA is practical. The algorithms in `estimation/` and `align/` ship only parallel implementations; sequential recursions belong in tests, as references.
- `philtorch/prototype/` holds unreleased research code.

## Native ops

- **Autograd in two places, one gradient.** An op's eager path is an `autograd.Function`, which also defines `jvp` and `vmap`; under `torch.compile` code calls the op itself, whose `torch.library.register_autograd` serves Dynamo, which can't trace an `autograd.Function` with a `jvp`. Both call one gradient function, as `_recur.py` does with `_gradients(run, ...)`, `run` being the function's `apply` or the op. Write a backward with the op and differentiable PyTorch ops, never a backward-only kernel, so that derivatives exist to any order.
- **Accumulation:** kernels read and write `scalar_t` and compute in `at::opmath_type<scalar_t>`, so float16 and bfloat16 are summed in float32. Dispatch with the `AT_DISPATCH_*` macros the op's dtypes need (no complex when it takes none), and report errors with `TORCH_CHECK`.
- **Portable C++:** C++17, as the oldest supported PyTorch builds use (no `<bit>`); include the `c10/cuda` headers, not `ATen/cuda/CUDAContext.h`, whose `cusparse.h` the CUDA CI image lacks.
- **Graph-capturable launches:** no host-to-device copy or synchronization in the launch path. Prefer arguments that are valid by construction, such as offsets computed from shapes, to device-side asserts, whose tests need a subprocess, as an assert ends the CUDA context.
- **Cached device tensors** (a filter table, an index):
  - Under Dynamo, wrap the `lru_cache` in `torch.compiler.assume_constant_result` through a lambda, or Dynamo traces into it.
  - Build them under `torch.inference_mode(False)`, or a later training call can't save them for backward.
  - Use the cache only for plain tensors (`type(t) is torch.Tensor`): traced code (`torch.compile`, `opcheck`) passes fake or functional tensors, which the cache would keep, and SymInt shapes, which it can't hash.
  - Make a CUDA entry complete before it is cached (synchronize its stream on a miss), and call `record_stream` on each use, so that other streams read it whole and its memory isn't reused under their work. A miss during CUDA graph capture is an error: warm the caches up first.
  - A test of another stream's use: autograd runs a backward on its forward's stream, and CUDA's lazy module loading synchronizes on a kernel's first launch and hides races, so launch the kernels once first.

## Conventions

- **Coefficients:** the denominator `a` excludes `a0`, so SciPy-equivalent calls prepend `1.0` (in tests, `[1.0] + a.tolist()`). `lfilter_zi` takes `(a, b)`, the reverse of SciPy's order.
- **Shapes:** LTI functions take static coefficients, e.g. `(N,)` or `(B, N)`. LPV functions take time-varying ones, e.g. `(B, T, M)`. The alignment and estimation APIs are batch first: `(B, T, C)`, not PyTorch's `(T, B, C)`. A function takes exactly the shapes its docstring lists: a batched coefficient (`b`, `a`) has the batch size of `x`, never a batch of one broadcast to it. Code that accepts an unlisted shape is a bug to report, not a feature to keep.
- **Choices are explicit arguments:**
  - filter forms: `df2`, `tdf2`, `df1`, `tdf1`;
  - LTI `lfilter` backends: `"ssm"`, `"diag_ssm"`;
  - LPV `lfilter` backends: `"ssm"`, `"torchlpc"`.
- **Docstrings** follow PyTorch's conventions, as detailed in `CONTRIBUTING.md`:
  - Google style, lines of 80 characters or fewer;
  - math in reStructuredText;
  - `Raises:` lists the errors bad input triggers; argument errors are `ValueError`s, not asserts.
- **New public functions** need an entry on their page in `docs/api/*.rst`.
- **Third-party code:** never port GPL or LGPL code, not even into a comment: philtorch is MIT. Take values measured from such a library, or derive them anew, and cite the source; squash-merge a branch whose history held such a port.
- **Docstring examples:** if a module's docstrings have examples, add the module to `MODULES` in `tests/test_docstrings.py`, or, for Triton code, to `CUDA_MODULES` in `tests/test_docstrings_cuda.py` (checked only on CUDA machines). Load it with `importlib.import_module` when a function of the same name shadows it, as `philtorch.align.dtw` does.

## Tests and CI

- When changing an op's registered backward or fake, test compiled code with `TORCHINDUCTOR_FX_GRAPH_CACHE=0 TORCHINDUCTOR_AUTOGRAD_CACHE=0`: the on-disk compile caches are keyed before tracing, so they can replay the old backward.
- pytest runs with `--import-mode=importlib`, so the tests import the installed philtorch, with its compiled extension, rather than the source tree. A test that runs a subprocess should pass `cwd=tmp_path`, or the subprocess imports the checkout.
- Mark tests that need a CUDA device with `pytest.mark.cuda`, on the test or on its `"cuda"` parameter. `tests/conftest.py` skips them where no CUDA device is available, as on the CPU CI runners, and the CUDA workflow (`build-cuda-wheels.yml`) runs exactly them, `-m cuda`, collecting only the modules that have them, on a Modal GPU, uploading their coverage to Codecov with the `gpu` flag, which `codecov.yml` carries forward to commits that don't run it. Coverage leaves out `@triton.jit` functions, which it can't trace, and `tests/conftest.py` runs autograd's backward passes on the calling thread, since coverage can't trace the thread autograd starts for CUDA.

## Branches and releases

- Branch from `dev` and open pull requests against `dev`; PR bodies follow `.github/pull_request_template.md`.
- Never merge into `main`. A release fast-forwards `main` to a `dev` commit and tags it with a three-part `vX.Y.Z` tag (setuptools_scm), as described in `CONTRIBUTING.md`.
- **Copilot reviews** arrive as comments: "Approval recommended" in the summary is its approval. The summary can also list issues it previously missed, outside any review thread, so read the review body as well as its threads.
