"""Tests for scripts/cuda_wheel_version.py and cuda_wheel_matrix.json."""

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from packaging.version import Version

SCRIPT = Path(__file__).parents[1] / "scripts" / "cuda_wheel_version.py"
SPEC = importlib.util.spec_from_file_location("cuda_wheel_version", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
VERSION_SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERSION_SCRIPT)

REPO_ROOT = Path(__file__).parents[1]
MATRIX_PATH = REPO_ROOT / "cuda_wheel_matrix.json"

# (torch, cuda) pairs PyTorch publishes per release, mirrored in the matrix,
# minus CUDA 12.6, which the CUDA sources don't build with unchanged.
EXPECTED_BUILDS = [
    ("2.12.1", "12.9"),
    ("2.12.1", "13.0"),
    ("2.12.1", "13.2"),
    ("2.13.0", "12.9"),
    ("2.13.0", "13.0"),
    ("2.13.0", "13.2"),
    ("2.14.1", "13.0"),
    ("2.14.1", "13.2"),
]
# PyTorch's x86_64 release arch lists (.ci/manywheel/build_env_setup.py).
EXPECTED_ARCHS = {
    "12.9": "7.5 8.0 8.6 9.0 10.0 12.0",
    "13.0": "7.5 8.0 8.6 9.0 10.0 12.0",
    "13.2": "7.5 8.0 8.6 9.0 10.0 12.0",
}


def _build(**fields):
    return {
        "torch": "2.14.1",
        "cuda": "13.0",
        "torch_index_url": "https://download.pytorch.org/whl/cu130",
        # Valid for every CUDA version, so tests can change "cuda" alone.
        "cuda_arch_list": "8.0 9.0+PTX",
        **fields,
    }


def _matrix(**fields):
    return {
        "schema_version": 2,
        "platform": "manylinux_2_28_x86_64",
        "build_image": "quay.io/pypa/manylinux_2_28_x86_64:2026.09.30-1",
        "python_versions": ["3.10"],
        "builds": [_build()],
        **fields,
    }


def test_matrix_file_validates_against_schema():
    data = VERSION_SCRIPT.validate_matrix(VERSION_SCRIPT.load_matrix(MATRIX_PATH), MATRIX_PATH)
    assert data["schema_version"] == 2
    assert data["platform"] == "manylinux_2_28_x86_64"
    assert data["build_image"].startswith("quay.io/pypa/manylinux_2_28_x86_64:")
    assert data["python_versions"] == ["3.10", "3.11", "3.12", "3.13"]
    # Latest patch of each of the three newest stable PyTorch minors, with one
    # build per CUDA version PyTorch publishes for that minor.
    assert [(build["torch"], build["cuda"]) for build in data["builds"]] == EXPECTED_BUILDS
    for build in data["builds"]:
        assert build["cuda_arch_list"] == EXPECTED_ARCHS[build["cuda"]], build


def test_version_label_is_deterministic_pep440():
    version = VERSION_SCRIPT.cuda_wheel_version("0.6.0", "2.14.1", "13.0")
    assert version == "0.6.0+torch2.14.1.cu130"
    parsed = Version(version)
    assert str(parsed) == version
    assert parsed.local == "torch2.14.1.cu130"
    # Deterministic: same inputs always give the same version.
    assert VERSION_SCRIPT.cuda_wheel_version("0.6.0", "2.14.1", "13.0") == version


def test_version_label_rejects_non_normalized_label():
    # packaging drops the leading zero, so the label would not be the one asked for.
    with pytest.raises(RuntimeError, match="does not normalize to itself"):
        VERSION_SCRIPT.cuda_wheel_version("0.6.0", "2.014.1", "13.0")


def test_base_version_strips_local_segment():
    base = VERSION_SCRIPT.base_version(override="0.6.0+cpu")
    assert base == "0.6.0"
    version = VERSION_SCRIPT.cuda_wheel_version(base, "2.12.1", "13.0")
    assert version == "0.6.0+torch2.12.1.cu130"


def test_base_version_from_setuptools_scm_is_public():
    pytest.importorskip("setuptools_scm")
    base = VERSION_SCRIPT.base_version()
    assert Version(base).public == base


def test_base_version_rejects_garbage():
    with pytest.raises(RuntimeError, match="not valid PEP 440"):
        VERSION_SCRIPT.base_version(override="not a version!!!")


def test_find_build_rejects_unknown_pair():
    data = VERSION_SCRIPT.validate_matrix(VERSION_SCRIPT.load_matrix(MATRIX_PATH))
    with pytest.raises(RuntimeError, match="not in the CUDA wheel matrix"):
        VERSION_SCRIPT.find_build(data, "2.11.0", "13.0")
    # 2.14.1 exists, but PyTorch never published it for CUDA 12.9.
    with pytest.raises(RuntimeError, match=r"supported: .*2\.14\.1\+cu130"):
        VERSION_SCRIPT.find_build(data, "2.14.1", "12.9")


def test_find_build_returns_exact_pair():
    data = VERSION_SCRIPT.validate_matrix(VERSION_SCRIPT.load_matrix(MATRIX_PATH))
    build = VERSION_SCRIPT.find_build(data, "2.14.1", "13.2")
    assert build["torch_index_url"] == "https://download.pytorch.org/whl/cu132"


def test_validate_matrix_accepts_minimal_matrix():
    data = _matrix()
    assert VERSION_SCRIPT.validate_matrix(data) is data


@pytest.mark.parametrize(
    "data, message",
    [
        pytest.param(_matrix(schema_version=1), "schema_version must be 2", id="schema"),
        pytest.param(_matrix(platform=""), "'platform' must be", id="platform"),
        # The image must match the platform auditwheel certifies.
        pytest.param(
            _matrix(build_image="nvidia/cuda:13.2.1-devel-ubuntu22.04"),
            "'build_image' must be an image for 'manylinux_2_28_x86_64'",
            id="build-image",
        ),
        pytest.param(_matrix(python_versions=[]), "'python_versions' must be", id="no-python"),
        pytest.param(
            _matrix(python_versions=["3.10\n"]), "'python_versions' must be", id="python-newline"
        ),
        pytest.param(
            _matrix(python_versions=["3.10", "3.10"]), "contains a duplicate", id="python-dup"
        ),
        pytest.param(_matrix(builds=[]), "'builds' must be a non-empty list", id="no-builds"),
        pytest.param(
            _matrix(builds=[_build(torch="2.14")]), "'torch' must be an exact X.Y.Z", id="torch"
        ),
        pytest.param(
            _matrix(builds=[_build(torch="2.14.1\n")]),
            "'torch' must be an exact X.Y.Z",
            id="torch-newline",
        ),
        pytest.param(_matrix(builds=[_build(cuda="13")]), "'cuda' must be an X.Y", id="cuda"),
        pytest.param(
            _matrix(builds=[_build(torch_index_url="https://download.pytorch.org/whl/cu126")]),
            "must end with '/cu130'",
            id="index-url",
        ),
        pytest.param(
            _matrix(builds=[_build(cuda_arch_list=" ")]), "'cuda_arch_list' must be", id="no-arch"
        ),
        pytest.param(
            _matrix(builds=[_build(cuda_arch_list="native")]),
            "'cuda_arch_list' must be",
            id="native-arch",
        ),
        pytest.param(
            _matrix(builds=[_build(cuda_arch_list="8.0;9.0")]),
            "'cuda_arch_list' must be",
            id="semicolon-arch",
        ),
        # PyTorch still publishes cu126, but recur2.cuh does not build with it.
        pytest.param(
            _matrix(
                builds=[
                    _build(cuda="12.6", torch_index_url="https://download.pytorch.org/whl/cu126")
                ]
            ),
            "CUDA 12.6 is older than 12.9",
            id="cuda-too-old",
        ),
        pytest.param(
            _matrix(builds=[_build(cuda_arch_list="7.0 8.0")]),
            "CUDA 13 dropped 7.0",
            id="arch-dropped-by-cuda",
        ),
        # A stray "python" key inside a build would otherwise flow through **build
        # in expand_matrix and silently override the matrix's python_versions.
        pytest.param(_matrix(builds=[_build(python="3.9")]), "unknown field", id="stray-field"),
        # One torch release ships several CUDA builds, but a repeated
        # (torch, CUDA) pair would emit the same wheel twice.
        pytest.param(
            _matrix(builds=[_build(), _build()]),
            "duplicate torch/CUDA combination",
            id="torch-cuda-dup",
        ),
    ],
)
def test_validate_matrix_rejects_bad_matrix(data, message):
    with pytest.raises(RuntimeError, match=message):
        VERSION_SCRIPT.validate_matrix(data)


def test_validate_matrix_accepts_same_torch_different_cuda():
    data = _matrix(
        builds=[
            _build(),
            _build(cuda="12.9", torch_index_url="https://download.pytorch.org/whl/cu129"),
        ]
    )
    assert len(VERSION_SCRIPT.validate_matrix(data)["builds"]) == 2


def test_validate_matrix_rejects_non_object_build():
    with pytest.raises(TypeError, match=r"builds\[0\] must be an object"):
        VERSION_SCRIPT.validate_matrix(_matrix(builds=["2.14.1"]))


def test_load_matrix_rejects_non_object(tmp_path):
    path = tmp_path / "matrix.json"
    path.write_text("[1, 2, 3]")
    with pytest.raises(TypeError, match="must be a JSON object"):
        VERSION_SCRIPT.load_matrix(path)


def test_load_matrix_rejects_invalid_json(tmp_path):
    path = tmp_path / "matrix.json"
    path.write_text("{")
    with pytest.raises(RuntimeError, match="is not valid JSON"):
        VERSION_SCRIPT.load_matrix(path)


def test_load_matrix_rejects_missing_file(tmp_path):
    with pytest.raises(RuntimeError, match="cannot read CUDA wheel matrix"):
        VERSION_SCRIPT.load_matrix(tmp_path / "missing.json")


def test_matrix_python_versions_match_package_classifiers():
    # CUDA wheels cover the same Python versions the package metadata advertises.
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    classifiers = re.findall(r'"Programming Language :: Python :: (3\.\d+)"', pyproject)
    assert VERSION_SCRIPT.load_matrix(MATRIX_PATH)["python_versions"] == classifiers


def test_expand_matrix_covers_python_x_torch_cuda():
    data = VERSION_SCRIPT.validate_matrix(VERSION_SCRIPT.load_matrix(MATRIX_PATH))
    expanded = VERSION_SCRIPT.expand_matrix(data)
    assert len(expanded) == 4 * len(EXPECTED_BUILDS)
    assert {(entry["python"], entry["torch"], entry["cuda"]) for entry in expanded} == {
        (python, torch, cuda)
        for python in ["3.10", "3.11", "3.12", "3.13"]
        for torch, cuda in EXPECTED_BUILDS
    }
    # Every leg carries what the workflow needs to pick its container and tag.
    for entry in expanded:
        assert entry["platform"] == data["platform"]
        assert entry["build_image"] == data["build_image"]


def test_cli_prints_version(tmp_path):
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps(_matrix()))
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--matrix",
            str(matrix),
            "--torch",
            "2.14.1",
            "--cuda",
            "13.0",
            "--base-version",
            "0.6.0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0.6.0+torch2.14.1.cu130"


def test_cli_prints_version_for_non_default_cuda(tmp_path):
    matrix = tmp_path / "matrix.json"
    matrix.write_text(
        json.dumps(
            _matrix(
                builds=[
                    _build(
                        cuda="12.9",
                        torch_index_url="https://download.pytorch.org/whl/cu129",
                    )
                ]
            )
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--matrix",
            str(matrix),
            "--torch",
            "2.14.1",
            "--cuda",
            "12.9",
            "--base-version",
            "0.6.0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0.6.0+torch2.14.1.cu129"


def test_cli_requires_cuda_with_torch():
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--matrix",
            str(MATRIX_PATH),
            "--torch",
            "2.14.1",
            "--base-version",
            "0.6.0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "--cuda is required" in result.stderr


def test_cli_list_prints_valid_json():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--matrix", str(MATRIX_PATH), "--list"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    # One line, so the workflow can write it to $GITHUB_OUTPUT as matrix=<json>.
    assert result.stdout.count("\n") == 1
    assert len(json.loads(result.stdout)) == 4 * len(EXPECTED_BUILDS)


def test_cli_fails_loudly_on_unknown_pair():
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--matrix",
            str(MATRIX_PATH),
            "--torch",
            "1.0.0",
            "--cuda",
            "13.0",
            "--base-version",
            "0.6.0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "not in the CUDA wheel matrix" in result.stderr
