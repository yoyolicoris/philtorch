"""Tests for scripts/cuda_paths_changed.py."""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "cuda_paths_changed.py"
spec = importlib.util.spec_from_file_location("cuda_paths_changed", SCRIPT)
assert spec is not None and spec.loader is not None
PATHS = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PATHS)


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


@pytest.mark.parametrize("line", ["", "      - 'b.py'", "      - b.py"])
def test_parsing_rejects_an_unexpected_line_in_the_list(line):
    text = f'on:\n  pull_request:\n    paths:\n      - "a.py"\n{line}\n      - "c.py"\n'
    with pytest.raises(ValueError, match="unexpected line"):
        PATHS.pull_request_paths(text)


def test_parsing_rejects_a_workflow_without_paths():
    with pytest.raises(ValueError, match="no pull_request paths"):
        PATHS.pull_request_paths("on:\n  pull_request:\n    branches: [dev]\n")


@pytest.mark.parametrize(
    ("pattern", "file", "matches"),
    [
        ("setup.py", "setup.py", True),
        ("setup.py", "xsetup.py", False),
        ("setup.py", "setupxpy", False),
        ("philtorch/csrc/**", "philtorch/csrc/a.cu", True),
        ("philtorch/csrc/**", "philtorch/csrc/sub/dir/a.cu", True),
        ("philtorch/csrc/**", "philtorch/other.py", False),
        ("philtorch/align/*.py", "philtorch/align/dtw.py", True),
        ("philtorch/align/*.py", "philtorch/align/sub/dtw.py", False),
        ("philtorch/**/_*_kernels.py", "philtorch/align/_dtw_kernels.py", True),
        ("philtorch/**/_*_kernels.py", "philtorch/a/b/_x_kernels.py", True),
        ("philtorch/**/_*_kernels.py", "philtorch/_recur_kernels.py", True),
        ("philtorch/**/_*_kernels.py", "philtorch/align/dtw.py", False),
        ("tests/test_recur.py", "tests/test_recur_ext.py", False),
        ("a/(b).py", "a/(b).py", True),
    ],
)
def test_pattern_matching(pattern, file, matches):
    assert PATHS.touches([pattern], [file]) is matches


@pytest.mark.parametrize("pattern", ["a?.py", "a+.py", "[ab].py", "!a.py"])
def test_pattern_rejects_unsupported_special_characters(pattern):
    with pytest.raises(ValueError, match="unsupported special characters"):
        PATHS.touches([pattern], ["a.py"])


def test_touches_any_file_and_any_pattern():
    assert PATHS.touches(["a.py", "b/**"], ["docs/x.rst", "b/c/d.py"])
    assert not PATHS.touches(["a.py", "b/**"], ["docs/x.rst"])
    assert not PATHS.touches(["a.py"], [])


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
