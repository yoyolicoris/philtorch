import pytest

from philtorch._build_check import check_torch_build

CUDA_WHEEL = "0.9.0+cu13"


def test_unlabelled_build_is_not_checked():
    # Source builds and the CPU wheels are not checked.
    check_torch_build("0.5.1", "2.12.0+cpu", None)


def test_matching_cuda_major_passes():
    check_torch_build(CUDA_WHEEL, "2.13.0+cu130", "13.0")


def test_cpu_only_torch_is_rejected():
    with pytest.raises(ImportError, match="built for CUDA 13, .* no CUDA support"):
        check_torch_build(CUDA_WHEEL, "2.14.1+cpu", None)


def test_cuda_older_than_the_build_is_rejected():
    with pytest.raises(ImportError, match="needs CUDA 12.9 or newer, .* uses CUDA 12.6"):
        check_torch_build("0.9.0+cu12", "2.13.0+cu126", "12.6")


def test_other_cuda_major_is_rejected():
    with pytest.raises(ImportError, match="built for CUDA 13, .* uses CUDA 12.9"):
        check_torch_build(CUDA_WHEEL, "2.14.1+cu129", "12.9")


def test_torch_older_than_the_extension_targets_is_rejected():
    with pytest.raises(ImportError, match=r"needs PyTorch 2\.12 or newer, .* PyTorch 2\.11\.0"):
        check_torch_build("0.5.1", "2.11.0+cpu", None)
