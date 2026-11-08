#!/usr/bin/env python3
"""Tell whether a push changed a file that build-cuda-wheels.yml GPU-tests on.

A pull request runs the CUDA workflow only when it changes a file of the
``pull_request`` ``paths:`` list in that workflow. A push to ``dev`` builds
the wheels every time, but needs the Modal GPU only for such a change. This
script reads the list from the workflow itself, so that it stays in one place,
and prints ``gpu_test=true`` or ``gpu_test=false`` for ``$GITHUB_OUTPUT``::

    python scripts/cuda_paths_changed.py --event "$GITHUB_EVENT_NAME" \\
        --before "$BEFORE" --sha "$GITHUB_SHA" >> "$GITHUB_OUTPUT"

Every other event, a new branch (``--before`` all zeros or empty) and a diff
that can't be computed, such as after a force push, print ``true``.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]
WORKFLOW = REPO_ROOT / ".github/workflows/build-cuda-wheels.yml"


def pull_request_paths(workflow: str) -> list[str]:
    """The entries of the ``paths:`` list under ``pull_request:``."""
    patterns = []
    state = "search"
    for line in workflow.splitlines():
        if state == "search" and line == "  pull_request:":
            state = "pull_request"
        elif state == "pull_request" and line == "    paths:":
            state = "paths"
        elif state == "paths":
            # The list ends at the next key, indented less than its entries.
            if line.strip() and len(line) - len(line.lstrip()) < 6:
                break
            entry = re.fullmatch(r'      - "(.+)"', line)
            if entry:
                patterns.append(entry[1])
            elif not line.lstrip().startswith("#"):
                raise ValueError(f"unexpected line in the pull_request paths: {line!r}")
    if not patterns:
        raise ValueError("no pull_request paths found in the workflow")
    return patterns


def pattern_regex(pattern: str) -> re.Pattern[str]:
    """GitHub's filter: ``*`` matches any characters but ``/``, ``**`` any.

    GitHub's other special characters, ``?+[]!``, raise a ``ValueError``.
    """
    special = sorted(set(pattern) & set("?+[]!"))
    if special:
        raise ValueError(f"{pattern!r}: unsupported special characters {''.join(special)}")
    regex = ""
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            # Zero directories too, as in philtorch/**/_*_kernels.py for
            # philtorch/_recur_kernels.py. At worst this runs the test too often.
            regex += "(?:.*/)?"
            i += 3
        elif pattern.startswith("**", i):
            regex += ".*"
            i += 2
        elif pattern[i] == "*":
            regex += "[^/]*"
            i += 1
        else:
            regex += re.escape(pattern[i])
            i += 1
    return re.compile(regex)


def touches(patterns: list[str], files: list[str]) -> bool:
    regexes = [pattern_regex(pattern) for pattern in patterns]
    return any(regex.fullmatch(file) for file in files for regex in regexes)


def gpu_test(event: str, before: str, sha: str) -> bool:
    if event != "push" or not before.strip("0"):
        return True
    try:
        diff = subprocess.run(
            ["git", "diff", "--name-only", before, sha],
            check=True,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
    except subprocess.CalledProcessError as error:
        print(
            f"::warning::git diff failed, running the GPU test: {error.stderr.strip()}",
            file=sys.stderr,
        )
        return True
    return touches(pull_request_paths(WORKFLOW.read_text()), diff.stdout.splitlines())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", required=True, help="the GitHub event name")
    parser.add_argument("--before", default="", help="the commit the push replaced")
    parser.add_argument("--sha", required=True, help="the commit pushed")
    arguments = parser.parse_args()
    print(f"gpu_test={str(gpu_test(arguments.event, arguments.before, arguments.sha)).lower()}")


if __name__ == "__main__":
    main()
