"""The requirements of the functions that run only as Triton kernels on CUDA GPUs.

These functions parallelize a recursion over time in custom kernels, which
only pay off on a GPU; elsewhere they raise instead of falling back. Their
docstrings say so in a note:

    Note:
        Runs only on CUDA GPUs, as Triton kernels: the inputs must be CUDA
        tensors, and Triton must be installed, as it is with PyTorch's CUDA
        builds for Linux.
"""

from torch import Tensor


def check_cuda_triton(name: str, tensor: Tensor) -> None:
    """Raise unless ``tensor`` is on a CUDA device and Triton is installed.

    Args:
        name (str): the calling function's name, for the error message.
        tensor (Tensor): an input whose device decides where the call runs.

    Raises:
        ValueError: if the tensor is not on a CUDA device.
        RuntimeError: if Triton is not installed.
    """
    if not tensor.is_cuda:
        raise ValueError(
            f"{name} runs Triton kernels on CUDA GPUs only; got a tensor on {tensor.device}."
        )
    try:
        import triton  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            f"{name} runs Triton kernels, but Triton is not installed. PyTorch's CUDA builds "
            "for Linux install it; elsewhere, see https://github.com/triton-lang/triton."
        ) from error
