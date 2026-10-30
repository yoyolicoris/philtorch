import shutil
import sys
from collections.abc import Callable

import pytest
import torch
from torch import Tensor

from philtorch._torchlpc import lpc as vendored_lpc
from philtorch.lpv import allpole, lfilter, linear_recurrence, state_space_recursion

_WINDOWS_INDUCTOR_UNAVAILABLE = sys.platform == "win32" and shutil.which("cl") is None
_SKIP_WITHOUT_CPU_INDUCTOR = pytest.mark.skipif(
    _WINDOWS_INDUCTOR_UNAVAILABLE,
    reason="PyTorch Inductor requires cl, which is unavailable on Windows CI",
)
_TORCHLPC_CUDA_OPS = ("philtorch::scan", "philtorch::lpc")


@pytest.fixture(autouse=True)
def _fresh_dynamo():
    # Several tests here compile the same function, such as
    # state_space_recursion, each with a different device or shape, so each
    # adds a compiled version. Dynamo keeps at most 8 per function (its
    # recompile_limit), and past that a fullgraph=True compile fails. Start
    # each test from an empty cache.
    torch.compiler.reset()
    yield


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
        devices.append(pytest.param("cuda", marks=pytest.mark.cuda))
    return devices


@pytest.fixture
def isolated_rng():
    """Restore the global RNG afterwards so later tests see unchanged state."""
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        yield


def _assert_compiled_forward_and_backward(
    function: Callable[..., Tensor], inputs: tuple[Tensor, ...]
) -> None:
    eager_inputs = tuple(tensor.detach().clone().requires_grad_() for tensor in inputs)
    compiled_inputs = tuple(tensor.detach().clone().requires_grad_() for tensor in inputs)

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
    x = torch.randn(2, 16, device=device) if contiguous_x else torch.randn(16, 2, device=device).t()
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
    # matrix-recurrence route, on CUDA through recur2 and recur3.
    cases = [pytest.param("cpu", M, marks=_SKIP_WITHOUT_CPU_INDUCTOR) for M in (1, 2, 3)]
    if torch.cuda.is_available() and all(
        _has_dispatch_kernel(op, "CUDA") for op in _TORCHLPC_CUDA_OPS
    ):
        cases += [pytest.param("cuda", M, marks=pytest.mark.cuda) for M in (1, 2, 3)]
    return cases


@pytest.mark.usefixtures("isolated_rng")
@pytest.mark.parametrize(("device", "state_size"), _state_space_cases())
@pytest.mark.parametrize("share_A", [True, False])
def test_state_space_recursion_compile_forward_and_backward(
    device: str, state_size: int, share_A: bool
) -> None:
    A_shape = (16, state_size, state_size) if share_A else (2, 16, state_size, state_size)
    A = torch.randn(*A_shape, device=device) * 0.05
    zi = torch.randn(2, state_size, device=device)
    x = torch.randn(2, 16, state_size, device=device)

    _assert_compiled_forward_and_backward(state_space_recursion, (A, zi, x))


def test_compiled_lpc_calls_the_raw_operator(monkeypatch) -> None:
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    x = torch.linspace(-0.7, 0.8, 16).reshape(2, 8)
    A = torch.linspace(-0.03, 0.03, 48).reshape(2, 8, 3)
    zi = torch.linspace(-0.2, 0.3, 6).reshape(2, 3)
    torch.testing.assert_close(vendored_lpc(x, A, zi), torch.ops.philtorch.lpc(x, A, zi))
