"""Tests for scripts/torch_matrix.py and torch_matrix.json."""

import json
import re
import subprocess
import sys

import pytest

from ._cases import REPO_ROOT, entry, load_script

SCRIPT = REPO_ROOT / "scripts" / "torch_matrix.py"
MATRIX_PATH = REPO_ROOT / "torch_matrix.json"


TM = load_script("scripts/torch_matrix.py")
MIN_TORCH = load_script("build_support.py").MIN_TORCH
PYPROJECT = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
OLDEST = TM.oldest_python(PYPROJECT)
ARCHS = "7.5 8.0 8.6 9.0 10.0 12.0+PTX"
IMAGE = "quay.io/pypa/manylinux_2_28_x86_64:2026.10.26-1"


# The values of the matrix when this was written, so that the tests of the
# legs don't change when scripts/update_torch_matrix.py updates the file.
TORCH = [
    entry("2.12.1", ["12.9", "13.0", "13.2"]),
    entry("2.13.0", ["12.9", "13.0", "13.2"]),
    entry("2.14.1", ["13.0", "13.2"]),
    entry("2.15.0", ["13.2", "13.4"], "3.15"),
]


def _matrix(**fields):
    return {
        "build_image": IMAGE,
        "cuda_arch_lists": {"12": ARCHS, "13": ARCHS},
        "torch": TORCH,
        **fields,
    }


def _legs(event, **fields):
    return TM.legs(_matrix(**fields), event, "3.11")


def _pairs(legs):
    return [(leg["torch"], leg["cuda"], leg["python"]) for leg in legs]


def test_matrix_file_follows_its_policy():
    # Invariants of any matrix that follows its policy, so that
    # scripts/update_torch_matrix.py can change the entries without editing this.
    data = TM.validate_matrix(json.loads(MATRIX_PATH.read_text()), OLDEST, MATRIX_PATH)
    assert TM.torch_minor(data["torch"][0]["version"]) == MIN_TORCH
    assert len(data["torch"]) <= 4


def test_matrix_python_versions_match_package_classifiers():
    # The package advertises the CPython versions from requires-python to the
    # newest one the matrix tests on.
    classifiers = re.findall(r'"Programming Language :: Python :: (3\.\d+)"', PYPROJECT)
    pythons = [entry["python"] for entry in json.loads(MATRIX_PATH.read_text())["torch"]]
    newest = max(int(python.split(".")[1]) for python in pythons)
    assert classifiers == [f"3.{minor}" for minor in range(OLDEST, newest + 1)]


def test_oldest_python_reads_requires_python():
    assert TM.oldest_python('[project]\nrequires-python = ">=3.12"\n') == 12
    with pytest.raises(RuntimeError, match="requires-python must be like"):
        TM.oldest_python('[project]\nrequires-python = ">=3.11,<3.14"\n')


def test_validate_matrix_accepts_the_example_matrix():
    data = _matrix()
    assert TM.validate_matrix(data, 11) is data


@pytest.mark.parametrize(
    "fields, message",
    [
        # PyTorch still publishes cu126, but the CUDA sources don't build with it.
        pytest.param(
            {"torch": [entry("2.14.1", ["12.6", "13.0"])]},
            "CUDA 12.6 is older than 12.9",
            id="cuda-too-old",
        ),
        # 12.10 is newer than 12.9, so a string sort would get this wrong.
        pytest.param(
            {"torch": [entry("2.14.1", ["12.10", "12.9"])]}, "must be ascending", id="cuda-order"
        ),
        pytest.param(
            {"torch": [entry("2.14.1", ["13.0", "13.0"])]},
            "without duplicates",
            id="cuda-dup",
        ),
        pytest.param({"torch": [entry("2.14.1", ["13.0"], "3.9")]}, "older than requires-python"),
        # Two patches of one minor would test it twice.
        pytest.param(
            {"torch": [entry("2.14.0", ["13.0"]), entry("2.14.1", ["13.0"])]},
            "ascending, with one entry per minor",
            id="torch-minor-dup",
        ),
        pytest.param(
            {"torch": [entry("2.15.0", ["13.0"]), entry("2.14.1", ["13.0"])]},
            "ascending, with one entry per minor",
            id="torch-order",
        ),
        pytest.param(
            {"cuda_arch_lists": {"12": "8.0;9.0", "13": ARCHS}},
            "cuda_arch_lists\\['12'\\] must be",
            id="bad-arch",
        ),
        pytest.param(
            {"cuda_arch_lists": {"12": ARCHS}},
            "cuda_arch_lists\\['13'\\] must be",
            id="missing-arch-list",
        ),
        pytest.param(
            {"cuda_arch_lists": {"12": ARCHS, "13": "7.0 8.0"}},
            "CUDA 13 dropped 7.0",
            id="arch-dropped-by-cuda",
        ),
    ],
)
def test_validate_matrix_rejects_bad_matrix(fields, message):
    with pytest.raises(RuntimeError, match=message):
        TM.validate_matrix(_matrix(**fields), 11)


def test_validate_matrix_accepts_7_0_with_cuda_12():
    TM.validate_matrix(_matrix(cuda_arch_lists={"12": "7.0 8.0+PTX", "13": ARCHS}), 11)


def test_platform_derives_from_build_image():
    image = "example.com/pypa/manylinux_2_34_aarch64:2027.1.1-1"
    assert _legs("push", build_image=image)["platform"] == "manylinux_2_34_aarch64"
    assert _legs("push")["platform"] == "manylinux_2_28_x86_64"


def test_legs_of_a_pull_request():
    legs = _legs("pull_request")
    assert legs["newest"] == "2.15.0"
    # The floor on the build Python comes first: that leg uploads coverage.
    assert legs["cpu_tests"] == [
        {"torch": "2.12.1", "python": "3.11"},
        {"torch": "2.15.0", "python": "3.15"},
    ]
    assert _pairs(legs["cuda_builds"]) == [("2.12.1", "12.9", "3.11"), ("2.12.1", "13.0", "3.11")]
    assert _pairs(legs["cuda_tests"]) == [("2.15.0", "13.4", "3.15")]


def test_legs_of_a_push_are_those_of_a_pull_request_but_for_the_cpu_tests():
    pull_request, push = _legs("pull_request"), _legs("push")
    assert push["cuda_builds"] == pull_request["cuda_builds"]
    assert push["cuda_tests"] == pull_request["cuda_tests"]
    assert [(leg["torch"], leg["python"]) for leg in push["cpu_tests"]] == [
        ("2.12.1", "3.11"),
        ("2.15.0", "3.15"),
        ("2.14.1", "3.14"),
        ("2.13.0", "3.14"),
        ("2.12.1", "3.14"),
    ]


def test_legs_name_the_runners_of_each_event():
    for event in ("pull_request", "release", "push", "workflow_dispatch"):
        legs = _legs(event)
        assert legs["build_os"] == ["ubuntu-latest", "windows-latest", "macos-latest"]
        tested = event in ("pull_request", "release")
        assert legs["test_os"] == (
            ["ubuntu-latest", "macos-latest", "windows-latest"] if tested else ["ubuntu-latest"]
        )


def test_legs_of_a_release_test_every_torch_and_cuda_pair():
    legs = _legs("release")
    assert len(legs["cpu_tests"]) == 5
    assert _pairs(legs["cuda_tests"]) == [
        ("2.15.0", "13.2", "3.15"),
        ("2.15.0", "13.4", "3.15"),
        ("2.14.1", "13.0", "3.14"),
        ("2.14.1", "13.2", "3.14"),
        ("2.13.0", "12.9", "3.14"),
        ("2.13.0", "13.0", "3.14"),
        ("2.13.0", "13.2", "3.14"),
        ("2.12.1", "12.9", "3.14"),
        ("2.12.1", "13.0", "3.14"),
        ("2.12.1", "13.2", "3.14"),
    ]


def test_legs_carry_what_the_workflow_needs():
    legs = _legs("pull_request")
    assert legs["cuda_builds"][1] == {
        "label": "cu13",
        "torch": "2.12.1",
        "cuda": "13.0",
        "python": "3.11",
        "index_url": "https://download.pytorch.org/whl/cu130",
        "arch_list": ARCHS,
    }
    # A test names the wheel it tests by label, and has no arch list.
    assert legs["cuda_tests"][-1] == {
        "label": "cu13",
        "torch": "2.15.0",
        "cuda": "13.4",
        "python": "3.15",
        "index_url": "https://download.pytorch.org/whl/cu134",
    }


def test_a_cuda_major_only_in_a_newer_torch_builds_from_that_torch():
    torch = [
        entry("2.12.1", ["12.9"]),
        entry("2.13.0", ["12.9", "13.0"]),
        entry("2.14.1", ["13.0", "13.2"]),
    ]
    legs = _legs("pull_request", torch=torch)
    # The oldest torch that has CUDA 13, on the build Python, not the floor.
    assert _pairs(legs["cuda_builds"]) == [("2.12.1", "12.9", "3.11"), ("2.13.0", "13.0", "3.11")]
    # Every tested wheel is built.
    assert {leg["label"] for leg in legs["cuda_tests"]} <= {
        leg["label"] for leg in legs["cuda_builds"]
    }


def test_legs_of_a_single_torch_have_no_duplicate():
    legs = _legs("pull_request", torch=[entry("2.12.1", ["12.9"], "3.11")])
    assert legs["cpu_tests"] == [{"torch": "2.12.1", "python": "3.11"}]
    assert len(legs["cuda_tests"]) == 1


def _cli(*args, root=REPO_ROOT):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def _root(tmp_path, matrix):
    (tmp_path / "torch_matrix.json").write_text(json.dumps(matrix))
    (tmp_path / "pyproject.toml").write_text(PYPROJECT)
    return tmp_path


def test_cli_prints_legs_for_github_output(tmp_path):
    result = _cli("legs", "--event", "push", root=_root(tmp_path, _matrix()))
    assert result.returncode == 0, result.stderr
    # key=value lines, each value one line: a bare string or JSON.
    lines = result.stdout.splitlines()
    assert [line.split("=", 1)[0] for line in lines] == [
        "floor",
        "newest",
        "build_python",
        "build_image",
        "platform",
        "build_os",
        "test_os",
        "cpu_tests",
        "cuda_builds",
        "cuda_tests",
    ]
    values = dict(line.split("=", 1) for line in lines)
    assert values["floor"] == "2.12.1"
    assert values["build_image"] == IMAGE
    legs = _legs("push")
    for key in ("cpu_tests", "cuda_builds", "cuda_tests"):
        assert json.loads(values[key]) == legs[key]
