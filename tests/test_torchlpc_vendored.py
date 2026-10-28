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
import torch.nn.functional as F
from torch.autograd.gradcheck import gradcheck, gradgradcheck
from torch.func import jacfwd

from philtorch._torchlpc import AllPole

_DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.cuda,
    ),
]
_REQUIRES_GRAD_CASES = [
    (True, False, False),
    (False, True, False),
    (False, False, True),
    (True, True, False),
    (True, False, True),
    (False, True, True),
    (True, True, True),
]


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
@pytest.mark.parametrize("device", _DEVICES)
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


@pytest.mark.parametrize("device", _DEVICES)
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


def _random(*shape, dtype):
    return torch.randn(*shape, dtype=dtype) * 0.3


def _lpc_reference(x, A, zi):
    B, T, order = A.shape
    # y[-1 - k] = zi[k]; the list holds y[-order], ..., y[-1], then y[0], ...
    y = list(zi.flip(1).unbind(1))
    for t in range(T):
        past = sum(A[:, t, i] * y[order - 1 - i + t] for i in range(order))
        y.append(x[:, t] - past)
    return torch.stack(y[order:], 1)


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
@pytest.mark.parametrize("T, order", [(2, 5), (3, 4), (40, 3)])
@pytest.mark.parametrize("strided", [False, True])
def test_lpc_cpu_matches_sequential(T, order, strided, dtype):
    B = 3
    if strided:
        # Transposed views, and one with gaps between its elements.
        x = _random(T, B, dtype=dtype).T
        A = _random(order, T, B, dtype=dtype).permute(2, 1, 0)
        zi = _random(B, 2 * order, dtype=dtype)[:, ::2]
        assert not (x.is_contiguous() or A.is_contiguous() or zi.is_contiguous())
    else:
        x, A, zi = (_random(*shape, dtype=dtype) for shape in [(B, T), (B, T, order), (B, order)])
    torch.testing.assert_close(torch.ops.philtorch.lpc(x, A, zi), _lpc_reference(x, A, zi))


def _scan_reference(impulse, decay, init):
    h, out = init, []
    for t in range(impulse.size(1)):
        h = decay[:, t] * h + impulse[:, t]
        out.append(h)
    return torch.stack(out, 1)


@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
@pytest.mark.parametrize("B, T", [(3, 40), (1, 5000)])
def test_scan_cpu_matches_sequential(eight_threads, B, T, dtype):
    impulse, init = _random(B, T, dtype=dtype), _random(B, dtype=dtype)
    # Decays of magnitude 0.99 to 1, so that the state lasts across the 5000 steps.
    decay = torch.polar(
        0.99 + 0.01 * torch.rand(B, T, dtype=torch.float64), torch.randn(B, T, dtype=torch.float64)
    )
    decay = decay if dtype.is_complex else decay.real
    torch.testing.assert_close(
        torch.ops.philtorch.scan(impulse, decay, init), _scan_reference(impulse, decay, init)
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.int64])
def test_scan_cpu_rejects_dtype(dtype):
    impulse, decay, init = (
        torch.ones(2, 8, dtype=dtype),
        torch.ones(2, 8, dtype=dtype),
        torch.ones(2, dtype=dtype),
    )
    with pytest.raises(RuntimeError):
        torch.ops.philtorch.scan(impulse, decay, init)
