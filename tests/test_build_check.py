import pytest

from philtorch._build_check import check_torch_build

CUDA_WHEEL = "0.9.0+cu13"


def test_unlabelled_build_gets_no_cuda_checks():
    # Source builds and the CPU wheels get only the torch floor check.
    check_torch_build("0.5.1", "2.12.0+cpu", None)


def test_matching_cuda_major_passes():
    check_torch_build(CUDA_WHEEL, "2.13.0+cu130", "13.0")


@pytest.mark.parametrize(
    ("version", "torch_version", "cuda", "message"),
    [
        (CUDA_WHEEL, "2.14.1+cpu", None, "built for CUDA 13, .* no CUDA support"),
        ("0.9.0+cu12", "2.13.0+cu126", "12.6", "needs CUDA 12.9 or newer, .* uses CUDA 12.6"),
        (CUDA_WHEEL, "2.14.1+cu129", "12.9", "built for CUDA 13, .* uses CUDA 12.9"),
        ("0.5.1", "2.11.0+cpu", None, r"needs PyTorch 2\.12 or newer, .* PyTorch 2\.11\.0"),
    ],
    ids=["cpu_only_torch", "cuda_older_than_build", "other_cuda_major", "torch_older_than_floor"],
)
def test_incompatible_torch_is_rejected(version, torch_version, cuda, message):
    with pytest.raises(ImportError, match=message):
        check_torch_build(version, torch_version, cuda)
