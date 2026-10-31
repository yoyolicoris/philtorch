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
BUILD_SPEC = importlib.util.spec_from_file_location("build_support", REPO_ROOT / "build_support.py")
assert BUILD_SPEC is not None and BUILD_SPEC.loader is not None
BUILD_SUPPORT = importlib.util.module_from_spec(BUILD_SPEC)
BUILD_SPEC.loader.exec_module(BUILD_SUPPORT)
MIN_TORCH = BUILD_SUPPORT.MIN_TORCH


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
    # The latest patch of every minor from the floor on, with every build of a
    # minor on the same patch and the same Pythons.
    assert min(by_minor, key=VERSION_SCRIPT._version_tuple) == MIN_TORCH
    for builds in by_minor.values():
        assert len({build["torch"] for build in builds}) == 1
        assert len({tuple(build["python_versions"]) for build in builds}) == 1
    keys = [
        (Version(build["torch"]), VERSION_SCRIPT._version_tuple(build["cuda"]))
        for build in data["builds"]
    ]
    assert keys == sorted(keys), "builds are in ascending torch and CUDA order"


def test_version_label_is_deterministic_pep440():
    version = VERSION_SCRIPT.cuda_wheel_version("0.9.0", "13.0")
    assert version == "0.9.0+cu13"
    parsed = Version(version)
    assert str(parsed) == version
    assert parsed.local == "cu13"
    # Deterministic: same inputs always give the same version.
    assert VERSION_SCRIPT.cuda_wheel_version("0.9.0", "13.0") == version
    # One label per CUDA major.
    assert VERSION_SCRIPT.cuda_wheel_version("0.9.0", "13.2") == version


def test_version_rejects_non_normalized_version():
    # packaging drops the leading zero, so the version would not be the one asked for.
    with pytest.raises(RuntimeError, match="does not normalize to itself"):
        VERSION_SCRIPT.cuda_wheel_version("0.06.0", "13.0")


def test_base_version_strips_local_segment():
    base = VERSION_SCRIPT.base_version(override="0.6.0+cpu")
    assert base == "0.6.0"
    version = VERSION_SCRIPT.cuda_wheel_version(base, "13.0")
    assert version == "0.6.0+cu13"


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


def test_check_cuda_rejects_unknown_cuda():
    data = _two_cuda_matrix()
    with pytest.raises(RuntimeError, match=r"CUDA 12\.9 is not in the CUDA wheel matrix"):
        VERSION_SCRIPT.check_cuda(data, "12.9")
    # CUDA 13 is in the matrix, but not 13.1.
    with pytest.raises(RuntimeError, match=r"supported: 13\.0, 13\.2"):
        VERSION_SCRIPT.check_cuda(data, "13.1")


def test_check_cuda_accepts_matrix_cuda():
    VERSION_SCRIPT.check_cuda(_two_cuda_matrix(), "13.2")


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
        # PyTorch still publishes cu126, but the CUDA sources don't build with it.
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


def test_expand_matrix_tests_every_build_on_its_newest_python():
    data = VERSION_SCRIPT.validate_matrix(VERSION_SCRIPT.load_matrix(MATRIX_PATH))
    legs = VERSION_SCRIPT.expand_matrix(data)["test"]
    assert [(leg["torch"], leg["cuda"]) for leg in legs] == [
        (build["torch"], build["cuda"]) for build in data["builds"]
    ]
    for leg, build in zip(legs, data["builds"]):
        assert leg["python"] == max(build["python_versions"], key=VERSION_SCRIPT._version_tuple)
        # Every leg carries what the workflow needs to pick its container and tag.
        assert leg["platform"] == data["platform"]
        assert leg["build_image"] == data["build_image"]
        assert leg["label"] == "cu" + build["cuda"].split(".")[0]
        assert "python_versions" not in leg


def test_expand_matrix_builds_one_wheel_per_cuda_major():
    data = VERSION_SCRIPT.validate_matrix(VERSION_SCRIPT.load_matrix(MATRIX_PATH))
    legs = VERSION_SCRIPT.expand_matrix(data)
    majors = sorted({int(build["cuda"].split(".")[0]) for build in data["builds"]})
    assert [leg["label"] for leg in legs["build"]] == [f"cu{major}" for major in majors]
    # Every tested build has a wheel to test.
    assert {leg["label"] for leg in legs["test"]} <= {leg["label"] for leg in legs["build"]}


def test_expand_matrix_builds_from_oldest_torch_then_oldest_cuda():
    def build(torch, cuda, pythons):
        return _build(
            torch=torch,
            cuda=cuda,
            torch_index_url="https://download.pytorch.org/whl/" + VERSION_SCRIPT._cuda_tag(cuda),
            python_versions=pythons,
        )

    data = VERSION_SCRIPT.validate_matrix(
        _matrix(
            builds=[
                build("2.13.0", "13.2", ["3.11", "3.12"]),
                build("2.14.1", "12.9", ["3.12", "3.13"]),
                build("2.14.1", "13.0", ["3.12", "3.13"]),
                build("2.14.1", "12.10", ["3.12", "3.13"]),
            ]
        )
    )
    legs = VERSION_SCRIPT.expand_matrix(data)["build"]
    # The oldest torch of a major wins over its oldest CUDA, and 12.10 is newer than 12.9.
    assert [(leg["label"], leg["torch"], leg["cuda"], leg["python"]) for leg in legs] == [
        ("cu12", "2.14.1", "12.9", "3.12"),
        ("cu13", "2.13.0", "13.2", "3.11"),
    ]
    for leg in legs:
        assert leg["torch_index_url"].endswith("/" + VERSION_SCRIPT._cuda_tag(leg["cuda"]))
        assert leg["cuda_arch_list"] == "8.0 9.0+PTX"
    assert [leg["python"] for leg in VERSION_SCRIPT.expand_matrix(data)["test"]] == [
        "3.12",
        "3.13",
        "3.13",
        "3.13",
    ]


def _cli(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_cli_prints_version(tmp_path):
    matrix = tmp_path / "matrix.json"
    matrix.write_text(json.dumps(_matrix()))
    result = _cli("--matrix", str(matrix), "--cuda", "13.0", "--base-version", "0.9.0")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0.9.0+cu13"


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
    result = _cli("--matrix", str(matrix), "--cuda", "12.9", "--base-version", "0.9.0")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0.9.0+cu12"


def test_cli_list_prints_valid_json():
    result = _cli("--matrix", str(MATRIX_PATH), "--list")
    assert result.returncode == 0, result.stderr
    # One line, so the workflow can write it to $GITHUB_OUTPUT as legs=<json>.
    assert result.stdout.count("\n") == 1
    data = VERSION_SCRIPT.load_matrix(MATRIX_PATH)
    legs = json.loads(result.stdout)
    assert set(legs) == {"build", "test"}
    assert len(legs["test"]) == len(data["builds"])


def test_cli_fails_loudly_on_unknown_cuda():
    result = _cli("--matrix", str(MATRIX_PATH), "--cuda", "11.8", "--base-version", "0.9.0")
    assert result.returncode == 1
    assert "not in the CUDA wheel matrix" in result.stderr
