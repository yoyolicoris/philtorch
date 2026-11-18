"""Tests for the vendored torchlpc ops (philtorch._torchlpc).

These tests are adapted from DiffAPF/torchlpc (MIT License, Copyright (c) 2023
Chin-Yun Yu), which is vendored into philtorch._C. They cover only the ops
redefined in philtorch's shim (AllPole), since that is what
philtorch owns and may revise in the future.

Original sources, in https://github.com/DiffAPF/torchlpc:
- torchlpc/tests/test_grad.py
- torchlpc/tests/test_vmap.py
"""

import subprocess
import sys

import pytest
import torch
import torch.autograd.forward_ad as fwAD
import torch.nn.functional as F
from torch.autograd.gradcheck import gradcheck, gradgradcheck
from torch.func import jacfwd

from philtorch._torchlpc import AllPole

from ._cases import DEVICES, DTYPE_TOLERANCES, MPS, random_int64, relative_error

# Each input on and off, and A both with and without zi: zi's gradient changes
# the adjoint recurrence that A's gradient is read from.
_REQUIRES_GRAD_CASES = [(True, True, True), (True, True, False), (False, False, True)]


def test_external_torchlpc_namespace_coexists(tmp_path):
    script = """
import torch

library = torch.library.Library("torchlpc", "DEF")
library.define("scan(Tensor a, Tensor b, Tensor c) -> Tensor")
library.define("lpc(Tensor a, Tensor b, Tensor c) -> Tensor")

import philtorch

assert hasattr(torch.ops.philtorch, "scan")
assert hasattr(torch.ops.philtorch, "lpc")
"""
    subprocess.run([sys.executable, "-c", script], cwd=tmp_path, check=True)


def get_random_biquads(cmplx=False):
    if cmplx:
        mag = torch.rand(2, dtype=torch.double)
        phase = torch.rand(2, dtype=torch.double) * 2 * torch.pi
        roots = mag * torch.exp(1j * phase)
        return torch.tensor([-roots[0] - roots[1], roots[0] * roots[1]], dtype=torch.complex128)
    mag = torch.rand(1, dtype=torch.double)
    phase = torch.rand(1, dtype=torch.double) * torch.pi
    return torch.tensor([-mag * torch.cos(phase) * 2, mag**2], dtype=torch.double)


def create_test_inputs(batch_size, samples, cmplx=False):
    start_coeffs = get_random_biquads(cmplx)
    end_coeffs = get_random_biquads(cmplx)
    dtype = torch.complex128 if cmplx else torch.double

    A = (
        torch.stack(
            [torch.linspace(start_coeffs[i], end_coeffs[i], samples, dtype=dtype) for i in range(2)]
        )
        .T.unsqueeze(0)
        .repeat(batch_size, 1, 1)
    )
    x = torch.randn(batch_size, samples, dtype=dtype)
    zi = torch.randn(batch_size, 2, dtype=dtype)
    return x, A, zi


@pytest.mark.parametrize(
    ("x_requires_grad", "a_requires_grad", "zi_requires_grad"),
    _REQUIRES_GRAD_CASES,
)
@pytest.mark.parametrize("cmplx", [True, False])
@pytest.mark.parametrize("device", DEVICES)
def test_allpole(
    x_requires_grad: bool,
    a_requires_grad: bool,
    zi_requires_grad: bool,
    cmplx: bool,
    device: str,
):
    batch_size = 4
    samples = 32
    x, A, zi = tuple(x.to(device) for x in create_test_inputs(batch_size, samples, cmplx))
    A.requires_grad = a_requires_grad
    x.requires_grad = x_requires_grad
    zi.requires_grad = zi_requires_grad

    assert gradcheck(AllPole.apply, (x, A, zi), check_forward_ad=True)
    assert gradgradcheck(AllPole.apply, (x, A, zi))


@pytest.mark.cuda
def test_allpole_float64_vs_32_cuda():
    batch_size = 4
    samples = 32
    x, A, zi = create_test_inputs(batch_size, samples)
    x = x.cuda()
    A = A.cuda()
    zi = zi.cuda()

    x32 = x.float()
    A32 = A.float()
    zi32 = zi.float()

    y64 = AllPole.apply(x, A, zi)
    y32 = AllPole.apply(x32, A32, zi32)

    assert torch.allclose(y64, y32.double(), atol=1e-6), torch.max(torch.abs(y64 - y32.double()))


@pytest.mark.parametrize("device", DEVICES)
def test_allpole_vmap(device: str):
    batch_size = 4
    samples = 40
    x, A, zi = tuple(x.to(device) for x in create_test_inputs(batch_size, samples, False))
    y = torch.randn_like(x)

    A = A[:, 0, :].clone()

    A.requires_grad = True
    zi.requires_grad = True
    x.requires_grad = True

    args = (x, A, zi)

    def func(x, A, zi):
        return F.mse_loss(AllPole.apply(x, A[:, None, :].expand(-1, samples, -1), zi), y)

    jacs = jacfwd(func, argnums=tuple(range(len(args))))(*args)

    loss = func(*args)
    loss.backward()
    for jac, arg in zip(jacs, args):
        assert torch.allclose(jac, arg.grad)


# CPU, CUDA where there is one, and MPS where there is one.
LPC_DEVICES = [*DEVICES, pytest.param("mps", marks=MPS)]


@pytest.mark.parametrize(
    "device", [pytest.param("cuda", marks=pytest.mark.cuda), pytest.param("mps", marks=MPS)]
)
@pytest.mark.parametrize(
    "on_cpu, match",
    [(1, "Coefficients must be on the same device"), (2, "Initial conditions must be on the same")],
)
def test_lpc_rejects_other_devices(device: str, on_cpu: int, match: str):
    # x, A and zi, one of them on the CPU.
    args = [torch.ones(2, 8), torch.zeros(2, 8, 2), torch.zeros(2, 2)]
    args = [arg if i == on_cpu else arg.to(device) for i, arg in enumerate(args)]
    with pytest.raises(RuntimeError, match=match):
        torch.ops.philtorch.lpc(*args)


@pytest.mark.parametrize("device", LPC_DEVICES)
@pytest.mark.parametrize(
    "a_shape, match",
    [((2, 8), "3-dimensional"), ((1, 8, 2), "Batch and time"), ((2, 4, 2), "Batch and time")],
)
def test_lpc_rejects_coefficients_of_another_shape(device: str, a_shape, match: str):
    x, zi = torch.ones(2, 8, device=device), torch.zeros(2, 2, device=device)
    with pytest.raises(RuntimeError, match=match):
        torch.ops.philtorch.lpc(x, torch.zeros(a_shape, device=device), zi)


@pytest.mark.parametrize(
    "device", [pytest.param("cuda", marks=pytest.mark.cuda), pytest.param("mps", marks=MPS)]
)
def test_lpc_rejects_orders_above_1024(device: str):
    x, zi = torch.ones(2, 8, device=device), torch.zeros(2, 1025, device=device)
    with pytest.raises(RuntimeError, match="less than or equal to 1024"):
        torch.ops.philtorch.lpc(x, torch.zeros(2, 8, 1025, device=device), zi)


@MPS
def test_lpc_mps_rejects_int64():
    x, zi = (
        torch.ones(2, 8, dtype=torch.int64, device="mps"),
        torch.zeros(2, 2, dtype=torch.int64, device="mps"),
    )
    with pytest.raises(RuntimeError, match="MPS lpc takes float32, float16, bfloat16"):
        torch.ops.philtorch.lpc(x, torch.zeros(2, 8, 2, dtype=torch.int64, device="mps"), zi)


@MPS
@pytest.mark.parametrize("B, T, order", [(0, 5, 2), (2, 0, 2), (2, 5, 0)])
def test_lpc_mps_empty_and_order_zero(B: int, T: int, order: int):
    x = torch.randn(B, T, device="mps")
    y = torch.ops.philtorch.lpc(x, torch.zeros(B, T, order, device="mps"), x.new_zeros(B, order))
    assert y.shape == (B, T) and y.device == x.device
    torch.testing.assert_close(y, x)


def _lpc_problem(B: int, T: int, order: int, dtype: torch.dtype):
    """x, a and zi in dtype, on CPU, with each step's coefficients summing to
    0.9 in magnitude, so that the filter is stable whatever they do over time."""
    gen = torch.Generator().manual_seed(order)
    wide = torch.complex128 if dtype.is_complex else torch.float64
    a = torch.randn(B, T, order, dtype=wide, generator=gen)
    a = a / a.abs().sum(2, keepdim=True) * 0.9
    x = torch.randn(B, T, dtype=wide, generator=gen)
    zi = torch.randn(B, order, dtype=wide, generator=gen)
    return x.to(dtype), a.to(dtype), zi.to(dtype), wide


def _check_lpc(device: str, dtype: torch.dtype, order: int, T: int):
    x, a, zi, wide = _lpc_problem(3, T, order, dtype)
    expected = torch.ops.philtorch.lpc(*(t.to(wide) for t in (x, a, zi)))
    actual = torch.ops.philtorch.lpc(*(t.to(device) for t in (x, a, zi)))
    assert actual.dtype == dtype
    assert relative_error(actual, expected) < DTYPE_TOLERANCES[dtype]


@pytest.mark.parametrize("device", LPC_DEVICES)
@pytest.mark.parametrize("dtype", DTYPE_TOLERANCES)
def test_lpc_dtypes(device: str, dtype: torch.dtype):
    _check_lpc(device, dtype, 33, 300)


@pytest.mark.parametrize("device", LPC_DEVICES)
@pytest.mark.parametrize(
    "dtype, order, T",
    # Orders around the 32 threads of a SIMD group and up to the largest of
    # CUDA and MPS, steps past the order, as the head reaches back into zi, and
    # fewer; the largest again in complex32, whose ring is complex.
    [(torch.float32, order, 300) for order in (1, 31, 32)]
    + [(torch.float32, 33, 5), (torch.float32, 1024, 1100), (torch.complex32, 1024, 1100)],
)
def test_lpc_orders(device: str, dtype: torch.dtype, order: int, T: int):
    _check_lpc(device, dtype, order, T)


# What each low-precision gradient is held to, relative to the largest.
_LPC_GRADIENT_DTYPES = [(torch.float32, 1e-4), (torch.float16, 1e-2), (torch.complex32, 1e-2)]


@pytest.mark.parametrize("device", LPC_DEVICES)
@pytest.mark.parametrize("dtype, tolerance", _LPC_GRADIENT_DTYPES)
def test_allpole_gradients_in_low_precision(device: str, dtype: torch.dtype, tolerance: float):
    # Against float64's on the same rounded inputs: the adjoint filter runs in
    # dtype, and complex32 runs the PyTorch ops it lacks on CPU in complex64.
    x, a, zi, wide = _lpc_problem(2, 40, 3, dtype)
    grad_y = torch.randn(2, 40, dtype=wide, generator=torch.Generator().manual_seed(1))

    def gradients(inputs, grad_y):
        inputs = [t.requires_grad_() for t in inputs]
        return torch.autograd.grad(AllPole.apply(*inputs), inputs, grad_y)

    expected = gradients([t.to(wide) for t in (x, a, zi)], grad_y)
    actual = gradients([t.to(device) for t in (x, a, zi)], grad_y.to(device, dtype))
    for got, want in zip(actual, expected):
        assert got.dtype == dtype
        assert relative_error(got, want) < tolerance


@pytest.mark.parametrize("device", LPC_DEVICES)
@pytest.mark.parametrize("dtype, tolerance", _LPC_GRADIENT_DTYPES)
def test_allpole_tangents_in_low_precision(device: str, dtype: torch.dtype, tolerance: float):
    x, a, zi, wide = _lpc_problem(2, 40, 3, dtype)
    gen = torch.Generator().manual_seed(2)
    tangents = [torch.randn(t.shape, dtype=wide, generator=gen) for t in (x, a, zi)]

    def tangent_out(primals, tangents):
        with fwAD.dual_level():
            duals = [fwAD.make_dual(p, t) for p, t in zip(primals, tangents)]
            return fwAD.unpack_dual(AllPole.apply(*duals)).tangent

    expected = tangent_out([t.to(wide) for t in (x, a, zi)], tangents)
    actual = tangent_out(
        [t.to(device) for t in (x, a, zi)], [t.to(device, dtype) for t in tangents]
    )
    assert actual.dtype == dtype
    assert relative_error(actual, expected) < tolerance


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.complex32, torch.int64])
def test_scan_takes_the_recurrence_dtypes(device: str, dtype: torch.dtype):
    # On CUDA, the recurrence sends time-varying M = 1 here in every dtype. On
    # CPU it sends them to recurN, but scan takes them too.
    B, T = 3, 50
    gen = torch.Generator().manual_seed(0)
    if dtype == torch.int64:
        impulse, decay, init = (random_int64(*s, generator=gen) for s in ((B, T), (B, T), (B,)))
        wide = dtype
    else:
        wide = torch.complex128 if dtype.is_complex else torch.float64
        impulse = torch.randn(B, T, dtype=wide, generator=gen).to(dtype)
        decay = (0.9 * torch.rand(B, T, generator=gen, dtype=torch.float64)).to(wide).to(dtype)
        init = torch.randn(B, dtype=wide, generator=gen).to(dtype)
    h, states = init.to(wide), []
    for t in range(T):
        h = decay[:, t].to(wide) * h + impulse[:, t].to(wide)
        states.append(h)
    expected = torch.stack(states, 1)
    actual = torch.ops.philtorch.scan(*(t.to(device) for t in (impulse, decay, init)))
    assert actual.dtype == dtype
    if dtype == torch.int64:
        assert torch.equal(actual.cpu(), expected)
    else:
        assert relative_error(actual, expected) < DTYPE_TOLERANCES[dtype]
