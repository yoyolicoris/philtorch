"""Tests for scripts/cuda_wheel_version.py and cuda_wheel_matrix.json."""

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
from packaging.specifiers import SpecifierSet
from packaging.version import Version

SCRIPT = Path(__file__).parents[1] / "scripts" / "cuda_wheel_version.py"
SPEC = importlib.util.spec_from_file_location("cuda_wheel_version", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
VERSION_SCRIPT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERSION_SCRIPT)

REPO_ROOT = Path(__file__).parents[1]
MATRIX_PATH = REPO_ROOT / "cuda_wheel_matrix.json"


def _legs(data):
    return {
        (python, build["torch"], build["cuda"])
        for build in data["builds"]
        for python in build["python_versions"]
    }


def _requires_python():
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    return SpecifierSet(re.search(r'requires-python = "([^"]+)"', pyproject).group(1))


def _build(**fields):
    return {
        "torch": "2.14.1",
        "cuda": "13.0",
        "torch_index_url": "https://download.pytorch.org/whl/cu130",
        # Valid for every CUDA version, so tests can change "cuda" alone.
        "cuda_arch_list": "8.0 9.0+PTX",
        "python_versions": ["3.11"],
        **fields,
    }


def _matrix(**fields):
    return {
        "schema_version": 3,
        "platform": "manylinux_2_28_x86_64",
        "build_image": "quay.io/pypa/manylinux_2_28_x86_64:2026.09.30-1",
        "builds": [_build()],
        **fields,
    }


def test_matrix_file_validates_against_schema():
    # Invariants of any matrix that follows its policy, so that
    # scripts/update_cuda_matrix.py can change the builds without editing this.
    data = VERSION_SCRIPT.validate_matrix(VERSION_SCRIPT.load_matrix(MATRIX_PATH), MATRIX_PATH)
    assert data["schema_version"] == 3
    assert data["platform"] == "manylinux_2_28_x86_64"
    assert data["build_image"].startswith("quay.io/pypa/manylinux_2_28_x86_64:")
    by_minor = {}
    for build in data["builds"]:
        by_minor.setdefault(build["torch"].rsplit(".", 1)[0], []).append(build)
        tag = VERSION_SCRIPT._cuda_tag(build["cuda"])
        assert build["torch_index_url"] == f"https://download.pytorch.org/whl/{tag}", build
        pythons = build["python_versions"]
        assert pythons == sorted(pythons, key=lambda v: int(v.split(".")[1])), build
    # The latest patch of each of the (at most) three newest minors, with
    # every build of a minor on the same patch and the same Pythons.
    assert 1 <= len(by_minor) <= 3
    for builds in by_minor.values():
        assert len({build["torch"] for build in builds}) == 1
        assert len({tuple(build["python_versions"]) for build in builds}) == 1
    keys = [
        (Version(build["torch"]), VERSION_SCRIPT._cuda_tuple(build["cuda"]))
        for build in data["builds"]
    ]
    assert keys == sorted(keys), "builds are in ascending torch and CUDA order"


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


def _two_cuda_matrix():
    return VERSION_SCRIPT.validate_matrix(
        _matrix(
            builds=[
                _build(),
                _build(cuda="13.2", torch_index_url="https://download.pytorch.org/whl/cu132"),
            ]
        )
    )


def test_find_build_rejects_unknown_pair():
    data = _two_cuda_matrix()
    with pytest.raises(RuntimeError, match="not in the CUDA wheel matrix"):
        VERSION_SCRIPT.find_build(data, "2.11.0", "13.0")
    # 2.14.1 exists, but not for CUDA 12.9.
    with pytest.raises(RuntimeError, match=r"supported: .*2\.14\.1\+cu130"):
        VERSION_SCRIPT.find_build(data, "2.14.1", "12.9")


def test_find_build_returns_exact_pair():
    build = VERSION_SCRIPT.find_build(_two_cuda_matrix(), "2.14.1", "13.2")
    assert build["torch_index_url"] == "https://download.pytorch.org/whl/cu132"


def test_validate_matrix_accepts_minimal_matrix():
    data = _matrix()
    assert VERSION_SCRIPT.validate_matrix(data) is data


@pytest.mark.parametrize(
    "data, message",
    [
        pytest.param(_matrix(schema_version=1), "schema_version must be 3", id="schema"),
        pytest.param(_matrix(platform=""), "'platform' must be", id="platform"),
        # The image must match the platform auditwheel certifies.
        pytest.param(
            _matrix(build_image="nvidia/cuda:13.2.1-devel-ubuntu22.04"),
            "'build_image' must be an image for 'manylinux_2_28_x86_64'",
            id="build-image",
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
        pytest.param(
            _matrix(builds=[_build(python_versions=[])]),
            "'python_versions' must be",
            id="no-python",
        ),
        pytest.param(
            _matrix(builds=[_build(python_versions=["3.11\n"])]),
            "'python_versions' must be",
            id="python-newline",
        ),
        pytest.param(
            _matrix(builds=[_build(python_versions=["3.11", "3.11"])]),
            "contains a duplicate",
            id="python-dup",
        ),
        # A stray "python" key inside a build would otherwise flow through **build
        # in expand_matrix and silently override the build's python_versions.
        pytest.param(_matrix(builds=[_build(python="3.9")]), "unknown field", id="stray-field"),
        # One torch release ships several CUDA builds, but a repeated
        # (torch, CUDA) pair would emit the same wheel twice.
        pytest.param(
            _matrix(builds=[_build(), _build()]),
            "duplicate torch minor/CUDA combination",
            id="torch-cuda-dup",
        ),
        # Two patches of one minor would share that minor's package index.
        pytest.param(
            _matrix(builds=[_build(), _build(torch="2.14.0")]),
            r"duplicate torch minor/CUDA combination \('2.14', '13.0'\)",
            id="torch-minor-cuda-dup",
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
    # CUDA wheels cover the Python versions the package metadata advertises,
    # none older than requires-python.
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    classifiers = re.findall(r'"Programming Language :: Python :: (3\.\d+)"', pyproject)
    requires = _requires_python()
    builds = VERSION_SCRIPT.load_matrix(MATRIX_PATH)["builds"]
    pythons = {python for build in builds for python in build["python_versions"]}
    assert sorted(pythons, key=lambda v: int(v.split(".")[1])) == classifiers
    assert all(python in requires for python in pythons)


def test_expand_matrix_covers_each_builds_pythons():
    data = VERSION_SCRIPT.validate_matrix(VERSION_SCRIPT.load_matrix(MATRIX_PATH))
    expanded = VERSION_SCRIPT.expand_matrix(data)
    legs = _legs(data)
    assert len(expanded) == len(legs) == sum(len(b["python_versions"]) for b in data["builds"])
    assert {(entry["python"], entry["torch"], entry["cuda"]) for entry in expanded} == legs
    # Every leg carries what the workflow needs to pick its container and tag.
    for entry in expanded:
        assert entry["platform"] == data["platform"]
        assert entry["build_image"] == data["build_image"]
        assert "python_versions" not in entry


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
    data = VERSION_SCRIPT.load_matrix(MATRIX_PATH)
    assert len(json.loads(result.stdout)) == len(_legs(data))


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
