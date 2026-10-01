"""Check that a CUDA wheel is loaded with the PyTorch it was built against."""

import re
import warnings

# CUDA wheels carry their build in the local version label, e.g.
# 0.5.1+torch2.14.1.cu132 (see scripts/cuda_wheel_version.py).
_LABEL_RE = re.compile(r"\+torch(\d+)\.(\d+)\.\d+\.cu(\d+)(\d)$")
_TORCH_RE = re.compile(r"(\d+)\.(\d+)")


def check_torch_build(philtorch_version, torch_version, torch_cuda_version):
    """Raise ImportError if this build cannot work with the installed PyTorch.

    A different torch minor, a CPU-only torch or a different CUDA major makes
    the compiled extension fail to load with an obscure error, so this fails
    first with a clear one. A different CUDA minor usually works and only
    warns. CPU wheels and source builds have no label and are not checked.
    """
    label = _LABEL_RE.search(philtorch_version)
    if label is None:
        return
    torch_major, torch_minor, cuda_major, cuda_minor = label.groups()
    built_torch = f"{torch_major}.{torch_minor}"
    built_cuda = f"{cuda_major}.{cuda_minor}"
    fix = "Install the philtorch build that matches your PyTorch, or build philtorch from source."

    installed = _TORCH_RE.match(torch_version)
    if installed is None or ".".join(installed.groups()) != built_torch:
        raise ImportError(
            f"philtorch {philtorch_version} was built for PyTorch {built_torch}, "
            f"but PyTorch {torch_version} is installed. {fix}"
        )
    if torch_cuda_version is None:
        raise ImportError(
            f"philtorch {philtorch_version} was built for CUDA {built_cuda}, but the "
            f"installed PyTorch {torch_version} has no CUDA support. {fix}"
        )
    if torch_cuda_version.split(".")[0] != cuda_major:
        raise ImportError(
            f"philtorch {philtorch_version} was built for CUDA {built_cuda}, but the "
            f"installed PyTorch uses CUDA {torch_cuda_version}. {fix}"
        )
    if ".".join(torch_cuda_version.split(".")[:2]) != built_cuda:
        warnings.warn(
            f"philtorch {philtorch_version} was built for CUDA {built_cuda}, but the "
            f"installed PyTorch uses CUDA {torch_cuda_version}. This usually works; "
            f"if CUDA calls fail, install the matching philtorch build.",
            RuntimeWarning,
            stacklevel=2,
        )
