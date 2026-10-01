"""Tests for scripts/cuda_wheel_version.py and cuda_wheel_matrix.json."""

import importlib.util
import json
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


def test_matrix_file_validates_against_schema():
    data = VERSION_SCRIPT.validate_matrix(VERSION_SCRIPT.load_matrix(MATRIX_PATH), MATRIX_PATH)
    assert data["schema_version"] == 1
    assert data["platform"] == "manylinux_2_28_x86_64"
    assert data["python_versions"] == ["3.10", "3.11", "3.12", "3.13"]
    # Latest patch of each of the three newest stable PyTorch minors.
    assert [build["torch"] for build in data["builds"]] == [
        "2.12.1",
        "2.13.0",
        "2.14.1",
    ]
    assert {build["cuda"] for build in data["builds"]} == {"13.0"}


def test_version_label_is_deterministic_pep440():
    version = VERSION_SCRIPT.cuda_wheel_version("0.6.0", "2.14.1", "13.0")
    assert version == "0.6.0+torch2.14.1.cu130"
    parsed = Version(version)
    assert str(parsed) == version
    assert parsed.local == "torch2.14.1.cu130"
    # Deterministic: same inputs always give the same version.
    assert VERSION_SCRIPT.cuda_wheel_version("0.6.0", "2.14.1", "13.0") == version


def test_base_version_strips_local_segment():
    base = VERSION_SCRIPT.base_version(override="0.6.0+cpu")
    assert base == "0.6.0"
    version = VERSION_SCRIPT.cuda_wheel_version(base, "2.12.1", "13.0")
    assert version == "0.6.0+torch2.12.1.cu130"


def test_base_version_rejects_garbage():
    with pytest.raises(RuntimeError, match="not valid PEP 440"):
        VERSION_SCRIPT.base_version(override="not a version!!!")


def test_find_build_rejects_unknown_torch():
    data = VERSION_SCRIPT.validate_matrix(VERSION_SCRIPT.load_matrix(MATRIX_PATH))
    with pytest.raises(RuntimeError, match="not in the CUDA wheel matrix"):
        VERSION_SCRIPT.find_build(data, "2.11.0")


def test_validate_matrix_rejects_bad_entries(tmp_path):
    bad = {
        "schema_version": 1,
        "platform": "manylinux_2_28_x86_64",
        "python_versions": ["3.10"],
        "builds": [
            {
                "torch": "2.14",
                "cuda": "13.0",
                "torch_index_url": "x",
                "cuda_arch_list": "8.0",
            }
        ],
    }
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(RuntimeError, match="'torch' must be an exact X.Y.Z"):
        VERSION_SCRIPT.validate_matrix(VERSION_SCRIPT.load_matrix(path), path)


def test_validate_matrix_rejects_non_object(tmp_path):
    path = tmp_path / "matrix.json"
    path.write_text("[1, 2, 3]")
    with pytest.raises(TypeError, match="must be a JSON object"):
        VERSION_SCRIPT.validate_matrix(VERSION_SCRIPT.load_matrix(path), path)


def test_validate_matrix_rejects_index_url_cuda_mismatch(tmp_path):
    bad = {
        "schema_version": 1,
        "platform": "manylinux_2_28_x86_64",
        "python_versions": ["3.10"],
        "builds": [
            {
                "torch": "2.14.1",
                "cuda": "13.0",
                "torch_index_url": "https://download.pytorch.org/whl/cu126",
                "cuda_arch_list": "8.0",
            }
        ],
    }
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(RuntimeError, match="must end with '/cu130'"):
        VERSION_SCRIPT.validate_matrix(VERSION_SCRIPT.load_matrix(path), path)


def test_validate_matrix_rejects_unknown_build_field(tmp_path):
    # A stray "python" key inside a build would otherwise flow through **build
    # in expand_matrix and silently override the matrix's python_versions.
    bad = {
        "schema_version": 1,
        "platform": "manylinux_2_28_x86_64",
        "python_versions": ["3.10"],
        "builds": [
            {
                "torch": "2.14.1",
                "cuda": "13.0",
                "torch_index_url": "https://download.pytorch.org/whl/cu130",
                "cuda_arch_list": "8.0",
                "python": "3.9",
            }
        ],
    }
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(RuntimeError, match="unknown field"):
        VERSION_SCRIPT.validate_matrix(VERSION_SCRIPT.load_matrix(path), path)


def test_validate_matrix_rejects_duplicate_torch_version(tmp_path):
    # find_build looks up by torch version alone, so the matrix must keep
    # torch versions unique even across different CUDA builds.
    bad = {
        "schema_version": 1,
        "platform": "manylinux_2_28_x86_64",
        "python_versions": ["3.10"],
        "builds": [
            {
                "torch": "2.14.1",
                "cuda": "13.0",
                "torch_index_url": "https://download.pytorch.org/whl/cu130",
                "cuda_arch_list": "8.0",
            },
            {
                "torch": "2.14.1",
                "cuda": "12.6",
                "torch_index_url": "https://download.pytorch.org/whl/cu126",
                "cuda_arch_list": "8.0",
            },
        ],
    }
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(RuntimeError, match="duplicate torch version"):
        VERSION_SCRIPT.validate_matrix(VERSION_SCRIPT.load_matrix(path), path)


def test_expand_matrix_covers_python_x_torch():
    data = VERSION_SCRIPT.validate_matrix(VERSION_SCRIPT.load_matrix(MATRIX_PATH))
    expanded = VERSION_SCRIPT.expand_matrix(data)
    assert len(expanded) == 4 * 3
    assert {(entry["python"], entry["torch"]) for entry in expanded} == {
        (python, torch)
        for python in ["3.10", "3.11", "3.12", "3.13"]
        for torch in ["2.12.1", "2.13.0", "2.14.1"]
    }


def test_cli_prints_version(tmp_path):
    matrix = tmp_path / "matrix.json"
    matrix.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "platform": "manylinux_2_28_x86_64",
                "python_versions": ["3.10"],
                "builds": [
                    {
                        "torch": "2.14.1",
                        "cuda": "13.0",
                        "torch_index_url": "https://download.pytorch.org/whl/cu130",
                        "cuda_arch_list": "8.0",
                    }
                ],
            }
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
            "--base-version",
            "0.6.0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0.6.0+torch2.14.1.cu130"


def test_cli_list_prints_valid_json():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--matrix", str(MATRIX_PATH), "--list"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert len(json.loads(result.stdout)) == 12


def test_cli_fails_loudly_on_unknown_torch():
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--matrix",
            str(MATRIX_PATH),
            "--torch",
            "1.0.0",
            "--base-version",
            "0.6.0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "not in the CUDA wheel matrix" in result.stderr
