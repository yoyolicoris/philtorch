"""Tests for scripts/cuda_paths_changed.py."""

import re
import subprocess
import sys

import pytest

from ._cases import REPO_ROOT, load_script

SCRIPT = REPO_ROOT / "scripts" / "cuda_paths_changed.py"
PATHS = load_script("scripts/cuda_paths_changed.py")


def test_parsed_patterns_are_the_workflows_entries():
    text = PATHS.WORKFLOW.read_text()
    # Every quoted list entry of the file is a path: branches use the flow style.
    assert PATHS.pull_request_paths(text) == re.findall(r'^ +- "(.+)"$', text, re.MULTILINE)


def test_parsing_skips_comments_and_stops_at_the_end_of_the_list():
    text = (
        'on:\n  push:\n    paths:\n      - "ignored"\n  pull_request:\n    branches: ["dev"]\n'
        '    paths:\n      - "a/**"\n      # a comment\n      - "b.py"\n  workflow_call:\n'
        '      - "after"\n'
    )
    assert PATHS.pull_request_paths(text) == ["a/**", "b.py"]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (
            f'on:\n  pull_request:\n    paths:\n      - "a.py"\n{line}\n      - "c.py"\n',
            "unexpected line",
        )
        for line in ["", "      - 'b.py'", "      - b.py"]
    ]
    + [("on:\n  pull_request:\n    branches: [dev]\n", "no pull_request paths")]
    + [
        (
            f'on:\n  pull_request:\n    paths:\n      - "{pattern}"\n',
            "unsupported special characters",
        )
        for pattern in ["a?.py", "a+.py", "[ab].py", "!a.py"]
    ],
)
def test_parsing_rejects_an_invalid_workflow(text, message):
    with pytest.raises(ValueError, match=message):
        PATHS.pull_request_paths(text)


@pytest.mark.parametrize(
    ("patterns", "changed"),
    [
        (["philtorch/**/_*_kernels.py"], True),
        (["philtorch/*.py"], True),
        (["philtorch/align/**"], True),
        (["docs/x.rst"], True),
        (["philtorch/*/dtw.py"], False),
        (["philtorch/align/*.py"], False),
        (["docs/y.rst", "tests/**"], False),
    ],
)
def test_git_matches_the_patterns(tmp_path, patterns, changed):
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    for command in (["init", "-q"], ["commit", "-q", "--allow-empty", "-m", "base"]):
        subprocess.run(git + command, check=True, cwd=tmp_path)
    for name in ("philtorch/_trace_kernels.py", "philtorch/align/sub/dtw.py", "docs/x.rst"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).touch()
    for command in (["add", "."], ["commit", "-q", "-m", "files"]):
        subprocess.run(git + command, check=True, cwd=tmp_path)
    assert PATHS.gpu_test("push", "HEAD~", "HEAD", patterns, tmp_path) is changed


@pytest.mark.parametrize(
    ("event", "base"),
    [("pull_request", "abc"), ("release", ""), ("push", "")],
)
def test_runs_the_test_without_a_commit_to_compare(event, base):
    assert PATHS.gpu_test(event, base, "HEAD")


def test_runs_the_test_when_the_diff_fails(capsys):
    assert PATHS.gpu_test("push", "1" * 40, "HEAD")
    assert "git diff failed" in capsys.readouterr().err


def test_command_prints_the_output_line():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--event", "workflow_dispatch", "--sha", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout == "gpu_test=true\n"
