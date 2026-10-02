import warnings

import pytest

from philtorch._build_check import check_torch_build

CUDA_WHEEL = "0.5.1+torch2.14.1.cu132"


@pytest.mark.parametrize(
    "philtorch_version",
    ["0.5.1", "0.5.2.dev3", "0.5.2.dev3+g1234abc", "0.4"],
)
def test_unlabelled_builds_are_not_checked(philtorch_version):
    # CPU wheels and source builds are compiled against the torch in use.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_torch_build(philtorch_version, "1.0.0", None)


@pytest.mark.parametrize("torch_cuda", ["13.2", "13.2.1"])
def test_matching_torch_and_cuda_pass(torch_cuda):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_torch_build(CUDA_WHEEL, "2.14.1+cu132", torch_cuda)


def test_other_torch_patch_passes():
    # The wheel pins torch == 2.14.*, so any 2.14 patch release is supported.
    check_torch_build(CUDA_WHEEL, "2.14.0", "13.2")


@pytest.mark.parametrize("torch_version", ["2.13.0+cu132", "3.14.1", "nightly"])
def test_other_torch_minor_is_rejected(torch_version):
    with pytest.raises(ImportError, match="built for PyTorch 2.14"):
        check_torch_build(CUDA_WHEEL, torch_version, "13.2")


def test_cpu_only_torch_is_rejected():
    with pytest.raises(ImportError, match="has no CUDA support"):
        check_torch_build(CUDA_WHEEL, "2.14.1+cpu", None)


def test_other_cuda_major_is_rejected():
    with pytest.raises(ImportError, match="installed PyTorch uses CUDA 12.9"):
        check_torch_build("0.5.1+torch2.14.1.cu130", "2.14.1+cu129", "12.9")


def test_other_cuda_minor_only_warns():
    with pytest.warns(RuntimeWarning, match="built for CUDA 13.2.*uses CUDA 13.0"):
        check_torch_build(CUDA_WHEEL, "2.14.1", "13.0")


def test_two_digit_label_cuda_minor_parses():
    # cu129 is CUDA 12.9, not 1.29.
    with pytest.warns(RuntimeWarning, match="built for CUDA 12.9"):
        check_torch_build("0.5.1+torch2.13.0.cu129", "2.13.0", "12.8")
