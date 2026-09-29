from collections.abc import Callable
import shutil
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor

from philtorch import _pararnn_backward
from philtorch._torchlpc import lpc as vendored_lpc, scan as vendored_scan
from philtorch.lpv import allpole, lfilter, linear_recurrence, state_space_recursion
from philtorch.lpv.ssm import MatrixRecurrence, _matrix_recurrence

_WINDOWS_INDUCTOR_UNAVAILABLE = sys.platform == "win32" and shutil.which("cl") is None
_SKIP_WITHOUT_CPU_INDUCTOR = pytest.mark.skipif(
    _WINDOWS_INDUCTOR_UNAVAILABLE,
    reason="PyTorch Inductor requires cl, which is unavailable on Windows CI",
)
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


def _torchlpc_devices() -> list[object]:
    devices = [pytest.param("cpu", marks=_SKIP_WITHOUT_CPU_INDUCTOR)]
    if torch.cuda.is_available() and all(
        _has_dispatch_kernel(op, "CUDA") for op in _TORCHLPC_CUDA_OPS
    ):
        devices.append("cuda")
    return devices


def _pararnn_available() -> bool:
    if not torch.cuda.is_available():
        return False
    return all(_has_dispatch_kernel(op, "CUDA") for op in _PARARNN_OPS)


@pytest.fixture
def isolated_rng():
    """Seed locally and restore the global RNG so later tests see unchanged state."""
    devices = (
        list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    )
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(0)
        yield


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


@pytest.mark.usefixtures("isolated_rng")
@pytest.mark.parametrize("device", _torchlpc_devices())
@pytest.mark.parametrize(
    ("with_zi", "transpose", "contiguous_x"),
    [
        pytest.param(False, False, True, id="default"),
        pytest.param(False, True, True, id="transpose"),
        pytest.param(True, True, True, id="transpose-zi"),
        pytest.param(True, False, False, id="zi-noncontiguous-x"),
    ],
)
def test_allpole_variants_compile_forward_and_backward(
    device: str, with_zi: bool, transpose: bool, contiguous_x: bool
) -> None:
    a = torch.randn(2, 16, 3, device=device) * 0.05
    x = (
        torch.randn(2, 16, device=device)
        if contiguous_x
        else torch.randn(16, 2, device=device).t()
    )
    if with_zi:
        zi = torch.randn(2, 3, device=device)
        _assert_compiled_forward_and_backward(
            lambda a, zi, x: allpole(a, x, zi, transpose=transpose)[0], (a, zi, x)
        )
    else:
        _assert_compiled_forward_and_backward(
            lambda a, x: allpole(a, x, transpose=transpose), (a, x)
        )


@pytest.mark.usefixtures("isolated_rng")
@pytest.mark.parametrize("device", _torchlpc_devices())
def test_torchlpc_lfilter_compile_forward_and_backward(device: str) -> None:
    b = torch.randn(2, 16, 3, device=device)
    a = torch.randn(2, 16, 2, device=device) * 0.05
    x = torch.randn(2, 16, device=device)

    _assert_compiled_forward_and_backward(
        lambda b, a, x: lfilter(b, a, x, backend="torchlpc"), (b, a, x)
    )


def _state_space_cases() -> list[object]:
    # M == 1 routes to the vendored LPC op; M == 2 and 3 exercise the compiled
    # matrix-recurrence route on CPU (CUDA M == 2 and 3 use ParaRNN, tested below).
    cases = [
        pytest.param("cpu", M, marks=_SKIP_WITHOUT_CPU_INDUCTOR) for M in (1, 2, 3)
    ]
    if "cuda" in _torchlpc_devices():
        cases.append(("cuda", 1))
    return cases


@pytest.mark.usefixtures("isolated_rng")
@pytest.mark.parametrize(("device", "state_size"), _state_space_cases())
@pytest.mark.parametrize("share_A", [True, False])
def test_state_space_recursion_compile_forward_and_backward(
    device: str, state_size: int, share_A: bool
) -> None:
    A_shape = (
        (16, state_size, state_size) if share_A else (2, 16, state_size, state_size)
    )
    A = torch.randn(*A_shape, device=device) * 0.05
    zi = torch.randn(2, state_size, device=device)
    x = torch.randn(2, 16, state_size, device=device)

    _assert_compiled_forward_and_backward(state_space_recursion, (A, zi, x))


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


def _serial_pararnn(jac: Tensor, rhs: Tensor) -> Tensor:
    output = [rhs[:, 0]]
    for step in range(1, rhs.size(1)):
        output.append(
            rhs[:, step] - (jac[:, step] @ output[-1].unsqueeze(-1)).squeeze(-1)
        )
    return torch.stack(output, dim=1)


@pytest.mark.parametrize("state_size", [2, 3])
def test_pararnn_registered_backward_formula_on_cpu(state_size: int) -> None:
    jac = torch.linspace(
        -0.03, 0.03, 2 * 9 * state_size * state_size, dtype=torch.double
    ).reshape(2, 9, state_size, state_size)
    jac[:, 0] = 0
    rhs = torch.linspace(-0.8, 0.9, 2 * 9 * state_size, dtype=torch.double).reshape(
        2, 9, state_size
    )
    jac.requires_grad_()
    rhs.requires_grad_()

    output = _serial_pararnn(jac, rhs)
    grad_output = torch.linspace(
        0.9, -0.7, output.numel(), dtype=output.dtype
    ).reshape_as(output)
    expected = torch.autograd.grad(output, (jac, rhs), grad_output)

    ctx = SimpleNamespace(
        saved_tensors=(jac.detach(), rhs.detach(), output.detach()),
        needs_input_grad=(True, True),
    )
    actual = _pararnn_backward(_serial_pararnn)(ctx, grad_output)
    torch.testing.assert_close(actual, expected)

    ctx.needs_input_grad = (False, False)
    assert _pararnn_backward(_serial_pararnn)(ctx, grad_output) == (None, None)


def test_compile_dispatch_helpers_call_raw_operators(monkeypatch) -> None:
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    impulse = torch.linspace(-0.8, 0.9, 16).reshape(2, 8)
    decay = torch.linspace(0.05, 0.45, 16).reshape(2, 8)
    init = torch.tensor([-0.2, 0.3])
    torch.testing.assert_close(
        vendored_scan(impulse, decay, init),
        torch.ops.philtorch.scan(impulse, decay, init),
    )

    x = torch.linspace(-0.7, 0.8, 16).reshape(2, 8)
    A = torch.linspace(-0.03, 0.03, 48).reshape(2, 8, 3)
    zi = torch.linspace(-0.2, 0.3, 6).reshape(2, 3)
    torch.testing.assert_close(
        vendored_lpc(x, A, zi), torch.ops.philtorch.lpc(x, A, zi)
    )

    matrix_x = torch.linspace(-0.6, 0.7, 32).reshape(2, 8, 2)
    matrix_A = torch.linspace(-0.03, 0.03, 64).reshape(2, 8, 2, 2)
    matrix_zi = torch.linspace(-0.2, 0.3, 4).reshape(2, 2)
    torch.testing.assert_close(
        _matrix_recurrence(matrix_A, matrix_zi, matrix_x),
        MatrixRecurrence.forward(matrix_A, matrix_zi, matrix_x),
    )
