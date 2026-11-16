---
name: philtorch-code-review
description: Review a philtorch pull request or diff against the repository's rules, covering native kernels and autograd, the dtype, device and empty-input contract, tests and CI paths, and docstrings, and report findings by severity. Use for every code review in this repository.
---

# Reviewing philtorch changes

The rules are in `AGENTS.md`, and for docstrings in `CONTRIBUTING.md`, "Docstrings", which the `philtorch-docstring` skill applies. This skill says what to check and how to report, without restating them. Where the two differ, those files win. Read the pull request's description first: a choice it states and explains is not a finding.

## What to check

Check each changed file against the section of `AGENTS.md` that governs it.

- **`philtorch/csrc/`:** "Native ops": the stable ABI, the accumulation type, graph-capturable launches, the Metal sources' rules. Also check reads and writes at the edges of a sequence, a chunk and a partial block, and the synchronization between a kernel's phases.
- **Autograd and `_ops.py`, `_recur.py`:** "Native ops", "Autograd in two places, one gradient". Flag a backward-only kernel, and a gradient that differs between the eager and compiled paths.
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
- **Anything ported:** no GPL or LGPL code, not even in a comment.
- **Scope:** the pull request targets `dev`, follows `.github/pull_request_template.md`, and leaves out unrelated changes.

## Before reporting

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
