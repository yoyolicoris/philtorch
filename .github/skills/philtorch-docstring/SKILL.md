---
name: philtorch-docstring
description: Workflow for writing, revising or trimming the docstring of a philtorch public function so that a first-time user can follow it. Use for every docstring edit under philtorch/, trims included.
---

# philtorch docstrings

The rules are in `CONTRIBUTING.md`, "Docstrings": Form, Content and Notation. This skill is the order in which to apply them.

1. **Neighbours.** Read the docstrings of the related functions: those the target calls, shares arguments with, or sits beside on its `docs/api/*.rst` page. Match their notation and structure, and tell the maintainer about any that break the symbol table.
2. **Reference.** Look filter equations up in Smith's book (https://ccrma.stanford.edu/~jos/filters/). For a SciPy counterpart, read its docstring to find the differences worth stating.
3. **Write**, then check the result against every rule in Form, Content and Notation.
4. **Render and run.** `pixi run docs` builds with no warnings, and `tests/test_docstrings.py` (`tests/test_docstrings_cuda.py` for Triton code) runs the examples.
5. **Fresh reader.** Give a subagent, or the maintainer if you can't spawn one, only the signature and the rendered docstring. Ask which argument or returned value it can't work out, which sentence is a fragment or refers to something off the page, and which symbol is undefined or used for two things. Fix what it finds, and repeat until it finds nothing.
6. **Diff** against `git show <base>:<path>`. Every argument is still explained, every citation keeps its target, and nothing that changes the output was cut.
