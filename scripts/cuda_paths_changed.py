#!/usr/bin/env python3
"""Tell whether a dev push needs the GPU test of build-cuda-wheels.yml.

A pull request runs the CUDA workflow only when it changes a file of the
``pull_request`` ``paths:`` list in that workflow. A push to ``dev`` builds
the wheels every time, but needs the Modal GPU only when such a file changed
since ``--base``, the commit whose wheels the ``cuda-dev`` pre-release holds.
A push whose GPU test fails publishes nothing, so the next push still tests
its change. This script reads the list from the workflow itself, so that it
stays in one place, matches it with git's glob pathspecs, which follow
GitHub's filter for the patterns it allows, and prints ``gpu_test=true`` or ``gpu_test=false`` for
``$GITHUB_OUTPUT``::

    python scripts/cuda_paths_changed.py --event "$GITHUB_EVENT_NAME" \\
        --base "$BASE" --sha "$GITHUB_SHA" >> "$GITHUB_OUTPUT"

Every other event, an empty ``--base`` (nothing published yet) and a diff
that can't be computed print ``true``.
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
    """The entries of the ``paths:`` list under ``pull_request:``.

    GitHub's special characters ``?+[]!`` mean something else to git, so they
    raise a ``ValueError``.
    """
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
                special = sorted(set(entry[1]) & set("?+[]!"))
                if special:
                    raise ValueError(
                        f"{entry[1]!r}: unsupported special characters {''.join(special)}"
                    )
                patterns.append(entry[1])
            elif not line.lstrip().startswith("#"):
                raise ValueError(f"unexpected line in the pull_request paths: {line!r}")
    if not patterns:
        raise ValueError("no pull_request paths found in the workflow")
    return patterns


def gpu_test(
    event: str,
    base: str,
    sha: str,
    patterns: list[str] | None = None,
    cwd: Path = REPO_ROOT,
) -> bool:
    if event != "push" or not base:
        return True
    if patterns is None:
        patterns = pull_request_paths(WORKFLOW.read_text())
    try:
        diff = subprocess.run(
            ["git", "diff", "--name-only", base, sha, "--"]
            + [f":(glob){pattern}" for pattern in patterns],
            check=True,
            capture_output=True,
            text=True,
            cwd=cwd,
        )
    except subprocess.CalledProcessError as error:
        print(
            f"::warning::git diff failed, running the GPU test: {error.stderr.strip()}",
            file=sys.stderr,
        )
        return True
    return bool(diff.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event", required=True, help="the GitHub event name")
    parser.add_argument(
        "--base", default="", help="the commit whose wheels the cuda-dev pre-release holds"
    )
    parser.add_argument("--sha", required=True, help="the commit pushed")
    arguments = parser.parse_args()
    print(f"gpu_test={str(gpu_test(arguments.event, arguments.base, arguments.sha)).lower()}")


if __name__ == "__main__":
    main()
