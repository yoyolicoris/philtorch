import pytest

from philtorch._build_check import check_torch_build

CUDA_WHEEL = "0.9.0+cu13"


@pytest.mark.parametrize("philtorch_version", ["0.5.1", "0.5.2.dev3+g1234abc", "0.4"])
def test_unlabelled_builds_are_not_checked(philtorch_version):
    # Source builds and the CPU wheels are not checked.
    check_torch_build(philtorch_version, "1.0.0", None)


@pytest.mark.parametrize(
    "torch_version, torch_cuda",
    [("2.13.0+cu130", "13.0"), ("2.20.1", "13.2.1"), ("nightly", "13.2")],
)
def test_matching_cuda_major_passes_on_any_torch(torch_version, torch_cuda):
    check_torch_build(CUDA_WHEEL, torch_version, torch_cuda)


def test_cpu_only_torch_is_rejected():
    with pytest.raises(ImportError, match="built for CUDA 13, .* no CUDA support"):
        check_torch_build(CUDA_WHEEL, "2.14.1+cpu", None)


def test_cuda_older_than_the_build_is_rejected():
    with pytest.raises(ImportError, match="needs CUDA 12.9 or newer, .* uses CUDA 12.6"):
        check_torch_build("0.9.0+cu12", "2.13.0+cu126", "12.6")


def test_other_cuda_major_is_rejected():
    with pytest.raises(ImportError, match="built for CUDA 13, .* uses CUDA 12.9"):
        check_torch_build(CUDA_WHEEL, "2.14.1+cu129", "12.9")
