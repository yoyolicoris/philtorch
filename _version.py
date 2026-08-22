# Kept at the repo root (not inside the `philtorch` package) so it can be
# imported by setuptools-git-versioning's `branch_formatter` without
# triggering `philtorch/__init__.py`, which requires torch to be installed.
# Isolated build environments (e.g. `python -m build`, cibuildwheel) only
# install `[build-system] requires`, not torch, so importing this from inside
# the package would fail and silently fall back to treating the dotted
# import path itself as a regexp.
import re


def format_branch_name(name):
    # "(fix|feat)/issue-name" or CICD's branch "HEAD"
    pattern = re.compile("^((fix|feat)\/(?P<branch>.+))|((head|HEAD))")

    match = pattern.search(name)
    if match:
        return f"dev+{match.group(0)}"  # => dev+"(fix|feat)/issue-name"

    # function is called even if branch name is not used in a current template
    # just left properly named branches intact
    if name in ["master", "dev", "main"]:
        return name

    # fail in case of wrong branch names like "bugfix/issue-unknown"
    raise ValueError(f"Wrong branch name: {name}")
