import pytest

from philtorch import _build_check
from philtorch._build_check import check_torch_build

CUDA_WHEEL = "0.9.0+cu13"


@pytest.mark.parametrize("philtorch_version", ["0.5.1", "0.5.2.dev3+g1234abc", "0.4"])
def test_unlabelled_builds_are_not_checked(philtorch_version):
    # Source builds and the unpinned wheels are compiled for the torch in use.
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


def test_other_cuda_major_is_rejected():
    with pytest.raises(ImportError, match="built for CUDA 13, .* uses CUDA 12.9"):
        check_torch_build(CUDA_WHEEL, "2.14.1+cu129", "12.9")


# The macOS wheel has no label; its requirement pins the torch minor instead.
PINNED_WHEEL = "0.6.1"


@pytest.mark.parametrize("torch_version", ["2.14.1", "2.14.0+cpu", "2.14.3"])
def test_pinned_wheel_with_matching_torch_minor_passes(torch_version):
    check_torch_build(PINNED_WHEEL, torch_version, None, "torch==2.14.*")


@pytest.mark.parametrize("torch_version", ["2.12.1+cpu", "2.15.0", "nightly"])
def test_pinned_wheel_with_other_torch_minor_is_rejected(torch_version):
    with pytest.raises(ImportError, match="built for PyTorch 2.14, but PyTorch .* is installed"):
        check_torch_build(PINNED_WHEEL, torch_version, None, "torch == 2.14.*")


@pytest.mark.parametrize("requirement", [None, "torch>=2.13.0", "torch==2.14.1"])
def test_unpinned_requirement_is_not_checked(requirement):
    # Source builds and the other wheels require torch >= 2.13.0 and are not checked.
    check_torch_build(PINNED_WHEEL, "2.12.1", None, requirement)


def test_installed_torch_requirement_reads_philtorch_metadata(monkeypatch):
    requirements = ['pytest>=6.0; extra == "dev"', "torchvision", "torch==2.14.*"]
    monkeypatch.setattr(_build_check.importlib.metadata, "requires", lambda name: requirements)
    assert _build_check.installed_torch_requirement() == "torch==2.14.*"


def test_installed_torch_requirement_without_metadata(monkeypatch):
    def missing(name):
        raise _build_check.importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(_build_check.importlib.metadata, "requires", missing)
    assert _build_check.installed_torch_requirement() is None
