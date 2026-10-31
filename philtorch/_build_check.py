"""Check that a philtorch wheel is loaded with a PyTorch it works with."""

import importlib.metadata
import re

# CUDA wheels carry their CUDA major in the local version label, e.g.
# 0.9.0+cu13 (see scripts/cuda_wheel_version.py). They are built on PyTorch's
# stable ABI, so they load on any torch from the minimum up.
_LABEL_RE = re.compile(r"\+cu(\d+)$")
_TORCH_RE = re.compile(r"(\d+)\.(\d+)")
# PyPI rejects local labels, so the macOS wheel, built for one torch minor,
# records it only in its requirement (see build_support.torch_requirement).
_PIN_RE = re.compile(r"torch\s*==\s*(\d+)\.(\d+)\.\*")


def installed_torch_requirement():
    """Return the torch requirement in philtorch's installed metadata, or None."""
    try:
        requirements = importlib.metadata.requires("philtorch") or []
    except importlib.metadata.PackageNotFoundError:
        return None
    for requirement in requirements:
        if ";" not in requirement and re.match(r"torch\b", requirement):
            return requirement
    return None


def check_torch_build(philtorch_version, torch_version, torch_cuda_version, torch_requirement=None):
    """Raise ImportError if this build cannot work with the installed PyTorch.

    A CPU-only torch or a different CUDA major makes the CUDA wheel's compiled
    extension fail to load with an obscure error, so this fails first with a
    clear one. The torch version and the CUDA minor are not checked: the
    wheel's ops use PyTorch's stable ABI, and a torch older than the minimum
    is rejected by the wheel's requirement.

    CUDA wheels are recognised by the version label ``+cu<major>``. Other
    builds are checked against ``torch_requirement`` when it pins a minor
    (``torch == X.Y.*``), as the macOS wheel does; source builds and the other
    wheels require ``torch >= ...`` and are not checked.
    """
    label = _LABEL_RE.search(philtorch_version)
    if label is None:
        _check_pinned_minor(philtorch_version, torch_version, torch_requirement)
        return
    cuda_major = label.group(1)
    fix = "Install the philtorch build for your PyTorch's CUDA major, or build from source."
    if torch_cuda_version is None:
        raise ImportError(
            f"philtorch {philtorch_version} was built for CUDA {cuda_major}, but the "
            f"installed PyTorch {torch_version} has no CUDA support. {fix}"
        )
    if torch_cuda_version.split(".")[0] != cuda_major:
        raise ImportError(
            f"philtorch {philtorch_version} was built for CUDA {cuda_major}, but the "
            f"installed PyTorch uses CUDA {torch_cuda_version}. {fix}"
        )


def _check_pinned_minor(philtorch_version, torch_version, torch_requirement):
    pin = _PIN_RE.fullmatch(torch_requirement.strip()) if torch_requirement else None
    if pin is None:
        return
    built_torch = ".".join(pin.groups())
    installed = _TORCH_RE.match(torch_version)
    if installed is None or ".".join(installed.groups()) != built_torch:
        raise ImportError(
            f"philtorch {philtorch_version} was built for PyTorch {built_torch}, "
            f"but PyTorch {torch_version} is installed. Install PyTorch {built_torch}, "
            "or build philtorch from source for your PyTorch (see the README)."
        )
