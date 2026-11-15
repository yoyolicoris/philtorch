---
name: philtorch-docstring
description: Write, revise or trim docstrings of philtorch's public functions so that they read well for a first-time user. Covers content and notation, which PyTorch's style rules don't, and checks the result with a fresh reader. Use for every docstring edit under philtorch/, trims included.
---

# philtorch docstrings

The rules are in `CONTRIBUTING.md`, "Docstrings": PyTorch's form (sections, roles, math, line length), then the content rules and the symbol table. Read them first. This skill is the workflow that applies them, for every coding agent: the file lives in `.agents/skills/`, which Copilot and Antigravity read, and `.claude/skills/philtorch-docstring` links to it for Claude Code.

## Steps

1. **Read the neighbours.** Read the docstrings of the related functions, the ones the target calls, sits beside on its `docs/api/*.rst` page, or shares arguments with. List their symbols and check them against the table in `CONTRIBUTING.md`. Match their notation and structure, or, where they break the table, say so to the maintainer.
2. **Check the reference.** For a filter equation, look the notation up in Smith's book (https://ccrma.stanford.edu/~jos/filters/), keeping `x[n]` brackets. For a SciPy counterpart, read SciPy's docstring: a Note states only the differences that change how a call is written or what it returns.
3. **Write or edit**, following the content rules: what, not how; no figures; Notes the user can act on.
4. **Form.** Apply the form rules in `CONTRIBUTING.md`, "Docstrings".
5. **Render.** Render the docstring through Napoleon and docutils with no warnings, or build with `pixi run docs`. Run its examples with `tests/test_docstrings.py` (`tests/test_docstrings_cuda.py` for Triton code). An example prints only results that are the same on every platform.
6. **Fresh reader.** Give a subagent only the signature and the rendered docstring, with no code and no history; an agent that can't spawn one asks the maintainer to read it instead. Ask:
   - which argument's effect, or which returned value, it can't work out from the text;
   - which sentence is a fragment, or refers to something not on the page;
   - which symbol is used before it is defined, or used for two things.

   Fix what it finds, and repeat until it finds nothing.
7. **Diff against the old version** (`git show <base>:<path>`). Every argument the old text explained is still explained, every citation keeps its target, and nothing that changes the output was cut.

## Don't

- Narrate the implementation: kernels, scans, trees, solves, chunks, launches.
- Quote measurements: errors, timings, memory figures, speed-ups.
- Use `k` as a gain, except in functions that mirror SciPy's.
- Leave a `Note:` about a limitation no argument avoids, such as a graph break.
- Show one equation with a parameter where an argument selects between two forms: show both.
