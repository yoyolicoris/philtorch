"""Check that a philtorch CUDA wheel is loaded with a PyTorch it works with."""

import re

# CUDA wheels carry their CUDA major in the local version label, e.g.
# 0.9.0+cu13 (see scripts/cuda_wheel_version.py). They are built on PyTorch's
# stable ABI, so they load on any torch from the minimum up.
_LABEL_RE = re.compile(r"\+cu(\d+)$")


def check_torch_build(philtorch_version, torch_version, torch_cuda_version):
    """Raise ImportError if this build cannot work with the installed PyTorch.

    A CPU-only torch or a different CUDA major makes the CUDA wheel's compiled
    extension fail to load with an obscure error, so this fails first with a
    clear one. The torch version and the CUDA minor are not checked: the
    wheel's ops use PyTorch's stable ABI, and a torch older than the minimum
    is rejected by the wheel's requirement.

    CUDA wheels are recognised by the version label ``+cu<major>``; other
    builds are not checked.
    """
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
