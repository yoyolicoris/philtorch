from pathlib import Path
from runpy import run_path

import pytest

resolve_cuda_build = run_path(str(Path(__file__).resolve().parents[1] / "build_support.py"))[
    "resolve_cuda_build"
]


def test_default_cpu_selection_without_visible_gpu():
    assert resolve_cuda_build("0", "/opt/cuda", False) is False


def test_default_cuda_selection_with_visible_gpu():
    assert resolve_cuda_build("0", "/opt/cuda", True) is True


def test_default_cpu_selection_without_toolkit():
    assert resolve_cuda_build("0", None, True) is False


def test_force_cuda_selection_without_visible_gpu():
    assert resolve_cuda_build("1", "/opt/cuda", False, "8.0 8.6", torch_cuda_version="13.0") is True


def test_force_cuda_rejects_missing_toolkit():
    with pytest.raises(
        RuntimeError,
        match="PHILTORCH_FORCE_CUDA=1 was requested, but CUDA_HOME is not set",
    ):
        resolve_cuda_build("1", None, False)


@pytest.mark.parametrize("value", ["yes", "true", "2", " 1"])
def test_force_cuda_rejects_invalid_value(value):
    with pytest.raises(RuntimeError, match="PHILTORCH_FORCE_CUDA must be either '0' or '1'"):
        resolve_cuda_build(value, "/opt/cuda", False)


@pytest.mark.parametrize("cuda_available", [False, True])
def test_empty_value_selects_automatically(cuda_available):
    assert resolve_cuda_build("", "/opt/cuda", cuda_available) is cuda_available


def test_force_cuda_rejects_cpu_only_torch():
    with pytest.raises(RuntimeError, match="PyTorch has no CUDA support"):
        resolve_cuda_build("1", "/opt/cuda", False, "8.0 8.6", torch_cuda_version=None)


def test_force_cuda_with_visible_gpu_ignores_missing_arch_list():
    assert resolve_cuda_build("1", "/opt/cuda", True, torch_cuda_version="13.0") is True


@pytest.mark.parametrize("arch_list", [None, ""])
def test_force_cuda_without_visible_gpu_requires_arch_list(arch_list):
    with pytest.raises(RuntimeError, match="TORCH_CUDA_ARCH_LIST is unset or 'native'"):
        resolve_cuda_build("1", "/opt/cuda", False, arch_list, torch_cuda_version="13.0")


def test_force_cuda_without_visible_gpu_accepts_arch_list():
    assert resolve_cuda_build("1", "/opt/cuda", False, "8.0 8.6", torch_cuda_version="13.0") is True


def test_force_cuda_without_visible_gpu_rejects_native_arch_list():
    with pytest.raises(RuntimeError, match="TORCH_CUDA_ARCH_LIST is unset or 'native'"):
        resolve_cuda_build("1", "/opt/cuda", False, "native", torch_cuda_version="13.0")
