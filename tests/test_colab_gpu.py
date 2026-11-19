"""Tests for the local parts of scripts/colab_gpu.py."""

import tarfile
from pathlib import Path

import pytest

from ._cases import entry, load_script

CG = load_script("scripts/colab_gpu.py")
MATRIX = {
    "torch": [
        entry("2.13.0", ["12.9", "13.0"]),
        entry("2.14.1", ["12.9", "13.0", "13.2"]),
        entry("2.15.0", ["13.2", "13.4"]),
    ]
}
WHEEL = Path("philtorch-0.8.1.dev44+cu13-cp311-abi3-manylinux_2_28_x86_64.whl")


@pytest.mark.parametrize(
    "version, torch, index",
    [
        ("0.8.1.dev44+cu13", "2.15.0", "cu134"),
        ("0.8.1+cu12", "2.14.1", "cu129"),
    ],
)
def test_pick_torch_takes_the_newest_torch_and_cuda_of_the_major(version, torch, index):
    assert CG.pick_torch(version, MATRIX) == (torch, CG.INDEX_URL + index)


@pytest.mark.parametrize("version", ["0.8.1", "0.8.1+cu11"])
def test_pick_torch_rejects_a_version_without_a_cuda_build_of_the_matrix(version):
    with pytest.raises(SystemExit):
        CG.pick_torch(version, MATRIX)


def test_wheel_version():
    assert CG.wheel_version(WHEEL) == "0.8.1.dev44+cu13"


def test_pack_takes_tests_benchmarks_and_pyproject_without_caches(tmp_path):
    path = tmp_path / "tree.tar.gz"
    CG.pack(path)
    with tarfile.open(path) as tar:
        names = tar.getnames()
    assert {name.split("/")[0] for name in names} == {"tests", "benchmarks", "pyproject.toml"}
    assert "tests/test_colab_gpu.py" in names
    assert not any("__pycache__" in name for name in names)
