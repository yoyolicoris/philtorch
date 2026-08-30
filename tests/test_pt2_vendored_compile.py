from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from philtorch.lpv import allpole, linear_recurrence, state_space_recursion

_TORCHLPC_CUDA_OPS = ("philtorch::scan", "philtorch::lpc")
_PARARNN_OPS = (
    "parallel_reduce_cuda::parallel_reduce_block_diag_2x2_cuda",
    "parallel_reduce_cuda::parallel_reduce_block_diag_3x3_cuda",
)


def _has_dispatch_kernel(op: str, dispatch_key: str) -> bool:
    try:
        return torch._C._dispatch_has_kernel_for_dispatch_key(op, dispatch_key)
    except RuntimeError:
        return False


def _torchlpc_devices() -> list[str]:
    devices = ["cpu"]
    if torch.cuda.is_available() and all(
        _has_dispatch_kernel(op, "CUDA") for op in _TORCHLPC_CUDA_OPS
    ):
        devices.append("cuda")
    return devices


def _pararnn_available() -> bool:
    if not torch.cuda.is_available():
        return False
    return all(_has_dispatch_kernel(op, "CUDA") for op in _PARARNN_OPS)


def _assert_compiled_forward_and_backward(
    function: Callable[..., Tensor], inputs: tuple[Tensor, ...]
) -> None:
    eager_inputs = tuple(tensor.detach().clone().requires_grad_() for tensor in inputs)
    compiled_inputs = tuple(
        tensor.detach().clone().requires_grad_() for tensor in inputs
    )

    eager_output = function(*eager_inputs)
    compiled_output = torch.compile(function, fullgraph=True)(*compiled_inputs)
    torch.testing.assert_close(compiled_output, eager_output)

    grad_output = torch.randn_like(eager_output)
    eager_output.backward(grad_output)
    compiled_output.backward(grad_output)

    for compiled_input, eager_input in zip(compiled_inputs, eager_inputs):
        torch.testing.assert_close(compiled_input.grad, eager_input.grad)


def _run_allpole(a: Tensor, zi: Tensor, x: Tensor) -> Tensor:
    y, _ = allpole(a, x, zi)
    return y


@pytest.mark.parametrize("device", _torchlpc_devices())
def test_linear_recurrence_compile_forward_and_backward(device: str) -> None:
    a = torch.rand(2, 16, device=device) * 0.5
    init = torch.randn(2, device=device)
    x = torch.randn(2, 16, device=device)

    _assert_compiled_forward_and_backward(linear_recurrence, (a, init, x))


@pytest.mark.parametrize("device", _torchlpc_devices())
def test_allpole_compile_forward_and_backward(device: str) -> None:
    a = torch.randn(2, 16, 3, device=device) * 0.05
    zi = torch.randn(2, 3, device=device)
    x = torch.randn(2, 16, device=device)

    _assert_compiled_forward_and_backward(_run_allpole, (a, zi, x))


@pytest.mark.skipif(not _pararnn_available(), reason="CUDA ParaRNN kernels unavailable")
@pytest.mark.parametrize("state_size", [2, 3])
@pytest.mark.parametrize("share_A", [True, False])
def test_pararnn_compile_forward_and_backward(state_size: int, share_A: bool) -> None:
    A_shape = (
        (16, state_size, state_size) if share_A else (2, 16, state_size, state_size)
    )
    A = torch.randn(*A_shape, device="cuda") * 0.03
    zi = torch.randn(2, state_size, device="cuda")
    x = torch.randn(2, 16, state_size, device="cuda")

    _assert_compiled_forward_and_backward(state_space_recursion, (A, zi, x))
