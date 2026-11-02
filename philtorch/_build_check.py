"""Check that a philtorch CUDA wheel is loaded with a PyTorch it works with."""

import re

# CUDA wheels carry their CUDA major in the local version label, e.g.
# 0.9.0+cu13 (see scripts/torch_matrix.py).
_LABEL_RE = re.compile(r"\+cu(\d+)$")
# The oldest CUDA the sources build with, so the oldest the wheels are built
# with and run on. scripts/torch_matrix.py reads it from here.
_MIN_CUDA = (12, 9)
# The torch the extension targets (build_support.MIN_TORCH); an older torch fails to load it.
_MIN_TORCH = (2, 12)


def check_torch_build(philtorch_version, torch_version, torch_cuda_version):
    """Raise ImportError if this build cannot work with the installed PyTorch.

    A CPU-only torch or a different CUDA major makes the CUDA wheel's compiled
    extension fail to load with an obscure error, so this fails first with a
    clear one, as does a CUDA older than the wheels are built with. A torch
    older than the extension targets fails the same way, for every build.

    CUDA wheels are recognised by the version label ``+cu<major>``; other
    builds are checked for the torch version only.
    """
    version = re.match(r"(\d+)\.(\d+)", torch_version)
    if version is not None and tuple(map(int, version.groups())) < _MIN_TORCH:
        minimum = ".".join(map(str, _MIN_TORCH))
        raise ImportError(
            f"philtorch {philtorch_version} needs PyTorch {minimum} or newer, but "
            f"PyTorch {torch_version} is installed. Install PyTorch {minimum} or newer."
        )
    label = _LABEL_RE.search(philtorch_version)
    if label is None:
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
    minimum = ".".join(map(str, _MIN_CUDA))
    if tuple(int(part) for part in torch_cuda_version.split(".")[:2]) < _MIN_CUDA:
        raise ImportError(
            f"philtorch {philtorch_version} needs CUDA {minimum} or newer, but the installed "
            f"PyTorch uses CUDA {torch_cuda_version}. Install a PyTorch built for CUDA "
            f"{minimum} or newer, or build philtorch from source."
        )
