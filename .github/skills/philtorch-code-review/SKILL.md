---
name: philtorch-code-review
description: Review a philtorch pull request or diff for minimal, duplication-free and efficient code, tests and docs that follow AGENTS.md and CONTRIBUTING.md, and report findings by severity. Use for every code review in this repository.
---

# Reviewing philtorch changes

The rules are in `AGENTS.md` and, for docstrings, `CONTRIBUTING.md`; where this skill differs, they win. Read the pull request's description first: a choice it states and explains is not a finding, unless a rule forbids it. A plan on the issue, or an agent's brief, is a blueprint, not a rule: check the change against the rules even where it does what the plan said. Review what was asked: the pull request, every pull request of a stack, and uncommitted changes when they are included.

## Questions for code, tests, comments and docs alike

- **Minimal?** The fewest functions, arguments, branches and lines that keep the feature efficient. Flag:
  - what the feature doesn't need, such as an argument comparable libraries do without, a helper, a dispatch layer or a kernel variant;
  - new logic stacked on old logic instead of replacing it, and a speed-up that costs much code;
  - a condition that the general path covers or that can't happen, and loops or if-else chains that fold into fewer;
  - a helper used once, one helper split into two for no gain, and a refactor that buys nothing beyond removing duplication.
- **Duplicated?** Code, functions, variables, behaviour, tests, comments, docs, kernels, dispatchers, launchers, ops or autograd definitions. When you find one case, sweep the change for the same pattern.
- **Stale or sloppy?** Unused code and variables, leftovers of earlier iterations, comments and docs that no longer match the code, jargon or machinery a reader can't follow, inconsistent names.
- **Efficient?** Passes, allocations or buffers that a view, a reshape or one fused pass avoids. A memory saving must not cost speed, and a speed-up needs a benchmark of each step, kept on the issue.
- **Consistent?** Before accepting a fix, check how the sibling kernels and functions handle the same case. Carry over what earlier reviews and Copilot found in similar code.
- **Tests?**
  - One test per path, covering each new behaviour and each new `ValueError`. Each new tensor argument has a mismatch case in `tests/test_argument_errors.py`.
  - A new device or dtype extends the existing tests. Parametrize only over values that reach distinct paths, and cross two parameters only where their combinations are paths of their own.
  - No test of an impossible branch or that restates the implementation, such as one that patches a function to see which path runs: test what the paths return.
  - Gradient tests, a `gradcheck` or a comparison with a reference's gradients, only for native ops, `custom_op`s and `autograd.Function`s: autograd through PyTorch ops is PyTorch's to test.
  - A removed test names the test that still covers its path.
- **Docs?** Compact, with no tuning history or benchmark tables. After a trim, a first-time user can still tell what each argument does, what is returned and how the parts connect. A method from a paper cites it, as `CONTRIBUTING.md`'s "Citations" says.
- **Scope and plan?** No more than the pull request says; a port keeps the logic it ports. A departure from the plan that a rule calls for is not a finding; one that changes the agreed design or behaviour is a decision for the maintainer.
- **Regressions?** Compare behaviour before and after. A claim that speed or behaviour is unchanged needs a run that shows it.

## Checks by area

- **`philtorch/csrc/`:** `AGENTS.md`, "Native ops". Also reads and writes at the edges of a sequence, a chunk and a partial block, synchronization between a kernel's phases, and stream use.
- **Autograd, `_ops.py`, `_recur.py`:** "Native ops". No backward-only kernel; gradients to any order; the eager and compiled paths agree.
- **Public functions:** "Conventions" and "Architecture". Shapes, dtypes and devices, empty inputs, coefficients, explicit choices; for a function that mixes real and complex, run each combination its checks accept and confirm it is listed and computed right; no fallback that hides the `NotImplementedError`; an integer `unroll_factor` gives the kernels' results and gradients; functions, not stateful classes; a SciPy mirror tested against SciPy.
- **Triton kernels:** "Architecture". `_*_kernels.py` names, `check_cuda_triton`, `custom_op`s with differentiable backward passes.
- **Tests:** "Tests and CI". `cuda` marks and the CUDA path filter; subprocesses run with `cwd=tmp_path`.
- **Docstrings:** `CONTRIBUTING.md`, "Docstrings", as the `philtorch-docstring` skill applies it. A new public function is in `docs/api/*.rst`; a module with examples is in `tests/test_docstrings.py`.
- **Adapted code:** its licence allows it; no GPL or LGPL, not even in a comment.
- **Build and CI** (`setup.py`, `build_support.py`, `pixi.toml`, `.github/workflows/`): "Commands", "Architecture" and "Tests and CI". The torch floor holds, and no paid GPU job is added.
- **The pull request:** targets `dev`, follows `.github/pull_request_template.md`, and passes CI; read the logs of failed jobs.

## Report

Tie each finding to a file and line, with the input that fails or the rule it breaks. Drop what you can't, and what an earlier review raised and the pull request fixed.

- 🔴 **[blocking]**: must be fixed before merging, such as a wrong gradient, an out-of-bounds access, a race, or a failing or missing test of new behaviour.
- 🟡 **[important]**: should be fixed, or discussed if you disagree, such as a silent dtype promotion, a fallback that hides the `NotImplementedError`, or a CUDA test without the `cuda` mark.
- 🟢 **[nit]**: nice to have, such as naming or a comment.
- 💡 **[suggestion]**: an alternative to consider.

Show the fix of each 🔴 and 🟡 finding in a short snippet. A review of your own pull request goes to the maintainer in the session, not into a pull request comment. When asked to address the findings, fix them and review again, until only 🟢 and 💡 remain.
