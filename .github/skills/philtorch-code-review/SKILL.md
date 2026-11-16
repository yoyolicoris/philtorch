---
name: philtorch-code-review
description: Review a philtorch pull request or diff for minimal, duplication-free, current and efficient code, tests and docs, and against the repository's rules on native kernels and autograd, the dtype, device and empty-input contract, tests and CI paths, and docstrings; report findings by severity. Use for every code review in this repository.
---

# Reviewing philtorch changes

The rules are in `AGENTS.md`, and for docstrings in `CONTRIBUTING.md`, "Docstrings", which the `philtorch-docstring` skill applies. This skill says what to check and how to report, without restating them. Where the two differ, those files win. Read the pull request's description first: a choice it states and explains is not a finding.

## What every review answers

The maintainer asks the same questions of every pull request, of its code, tests, comments and docs alike. Answer each one.

- **Is it minimal?** The change should reach its feature with the fewest functions, arguments, branches and lines that keep it efficient.
  - Flag over-engineering: arguments, helpers, dispatch layers or kernel variants the feature doesn't need, and new logic stacked on old logic instead of replacing it. A speed-up that costs much code, new arguments or new functions is rarely worth it: say what it buys.
  - Flag a condition the general path already handles or that can't happen, such as a special case for a zero size, and loops or if-else chains that fold into fewer.
  - Flag a helper used once that could be inlined, and one helper split into two for no gain.
  - Flag an argument that comparable libraries do without; check what they offer.
  - For a refactor, say what it buys. Removing duplication alone doesn't justify code that is harder to follow.
- **Is anything duplicated or redundant?** Code, functions, variables, behaviour, tests, comments, docs, kernels, dispatchers, launchers, ops and autograd definitions. When you find one case, sweep the whole change for the same pattern.
- **Is anything stale or sloppy?** Unused code and variables, leftovers of earlier iterations, comments and docs that no longer match the code, jargon or machinery a reader can't follow, and inconsistent names.
- **Is it efficient?** Flag passes, allocations or buffers that a view, a reshape or one fused pass avoids, such as summing tangents from several forward passes where one pass takes them all. A change meant to save memory must not cost speed, and one meant to be faster needs a benchmark of each step, kept on the issue.
- **Is it consistent with its neighbours?** Before accepting a fix, check how the sibling kernels or functions handle the same case, such as clamping, streams or empty sizes. Carry over what earlier reviews and Copilot found in similar code.
- **Are the tests minimal and sufficient?**
  - One test per path.
  - A new device or dtype extends the existing tests rather than adding parallel ones. Parametrize only over values that reach distinct paths.
  - No test of a branch that can't happen, and no test that restates the implementation. Gradchecks only for native ops, `custom_op`s and `autograd.Function`s, not for compositions of PyTorch ops.
  - A removed test leaves no path uncovered: name the test that still covers it.
- **Do the docs serve the user?**
  - Comments and docstrings are compact, with no long descriptions, tuning history or benchmark tables. Measurements go on issue artifacts.
  - After a trim, a first-time user can still tell what each argument does, what is returned, and how its parts connect, such as which sections a filter has and how its initial state maps to them.
  - A method from a paper cites it.
- **Is it within scope and the plan?** The change does what the pull request says and no more; a port doesn't change the logic it ports. List each departure from the agreed plan as a decision for the maintainer, not a settled fact.
- **Does anything regress?** Compare behaviour before and after. A claim that speed or behaviour is unchanged needs a run that shows it.

## What to check by area

Check each changed file against the section of `AGENTS.md` that governs it.

- **`philtorch/csrc/`:** "Native ops": the stable ABI, the accumulation type, graph-capturable launches, the Metal sources' rules. Also check reads and writes at the edges of a sequence, a chunk and a partial block, the synchronization between a kernel's phases, and stream use: launches on the current stream, and cached tensors used on another.
- **Autograd and `_ops.py`, `_recur.py`:** "Native ops", "Autograd in two places, one gradient". Flag a backward-only kernel, a public function whose gradients stop short of any order, and a gradient that differs between the eager and compiled paths.
- **Public functions** (`lti/`, `lpv/`, `estimation/`, `align/`, `mat.py`, `poly.py`): "Conventions" and "Architecture".
  - Check shapes, dtypes and devices, empty inputs, coefficients and explicit choices.
  - Flag a fallback that hides the `NotImplementedError` for want of a kernel.
  - Flag a stateful class where a function would do: the package is functional.
  - Where a function mirrors SciPy's, check its output against SciPy's in a test.
- **Triton kernels:** "Architecture": the `_*_kernels.py` names, `check_cuda_triton`, `custom_op` with differentiable backward passes.
- **Tests:** "Tests and CI".
  - Each new behaviour and each new `ValueError` has a test.
  - CUDA tests are marked `cuda`, and the CUDA workflow's path filter lists their modules.
  - A subprocess runs with `cwd=tmp_path`.
- **Docstrings:** `CONTRIBUTING.md`, "Docstrings".
  - Check form, content and notation.
  - A new public function has an entry in `docs/api/*.rst`.
  - A module with examples is listed in `tests/test_docstrings.py`.
- **Anything ported or adapted** from SciPy, soxr or another library: check its licence; no GPL or LGPL code, not even in a comment.
- **Scope:** the pull request targets `dev` and follows `.github/pull_request_template.md`.

## Before reporting

- Review what was asked: the pull request, every pull request of a stack, and uncommitted changes when they are included.
- Tie each finding to a file and line. Give the input that fails or the rule it breaks.
- Drop a finding you can't tie to a concrete failure or a written rule.
- Don't repeat a finding from an earlier review that the pull request has addressed.

## Output

Label each finding by severity:

- 🔴 **[blocking]**: must be fixed before merging, such as a wrong gradient, an out-of-bounds access, a race, or a failing or missing test of new behaviour.
- 🟡 **[important]**: should be fixed, or discussed if you disagree, such as a silent dtype promotion, a fallback that hides the `NotImplementedError`, or a CUDA test without the `cuda` mark.
- 🟢 **[nit]**: nice to have, such as naming or a comment.
- 💡 **[suggestion]**: an alternative to consider.

For 🔴 and 🟡 findings, show the fix in a short code snippet.

## After reporting

- An agent's review of its own pull request goes to the maintainer in the session, not into a pull request comment.
- When asked to address the findings, fix them, then review again, until only 🟢 and 💡 findings remain.
