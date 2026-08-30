from pathlib import Path
from runpy import run_path

import pytest

resolve_cuda_build = run_path(
    str(Path(__file__).resolve().parents[1] / "build_support.py")
)["resolve_cuda_build"]


def test_default_cpu_selection_without_visible_gpu():
    assert resolve_cuda_build("0", "/opt/cuda", False) is False


def test_default_cuda_selection_with_visible_gpu():
    assert resolve_cuda_build("0", "/opt/cuda", True) is True


def test_default_cpu_selection_without_toolkit():
    assert resolve_cuda_build("0", None, True) is False


def test_force_cuda_selection_without_visible_gpu():
    assert resolve_cuda_build("1", "/opt/cuda", False) is True


def test_force_cuda_rejects_missing_toolkit():
    with pytest.raises(
        RuntimeError,
        match="PHILTORCH_FORCE_CUDA=1 was requested, but CUDA_HOME is not set",
    ):
        resolve_cuda_build("1", None, False)


def test_force_cuda_rejects_invalid_value():
    with pytest.raises(
        RuntimeError, match="PHILTORCH_FORCE_CUDA must be either '0' or '1'"
    ):
        resolve_cuda_build("yes", "/opt/cuda", False)
