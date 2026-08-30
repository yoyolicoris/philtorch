"""Tests for the vendored torchlpc ops (philtorch._torchlpc).

These tests are adapted from DiffAPF/torchlpc (MIT License, Copyright (c) 2023
Chin-Yun Yu), which is vendored into philtorch._C. They cover only the ops
redefined in philtorch's shim (AllPole / ScanRecurrence), since those are what
philtorch owns and may revise in the future.

Original sources:
- third_party/torchlpc/torchlpc/tests/test_grad.py
- third_party/torchlpc/torchlpc/tests/test_vmap.py
"""

import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F
from torch.autograd.gradcheck import gradcheck, gradgradcheck
from torch.func import jacfwd

from philtorch._torchlpc import AllPole, ScanRecurrence

_DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA not available"
        ),
    ),
]
_GRAD_CASES = [(True, True), (True, False), (False, True), (False, False)]


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
        return torch.tensor(
            [-roots[0] - roots[1], roots[0] * roots[1]], dtype=torch.complex128
        )
    mag = torch.rand(1, dtype=torch.double)
    phase = torch.rand(1, dtype=torch.double) * torch.pi
    return torch.tensor([-mag * torch.cos(phase) * 2, mag**2], dtype=torch.double)


def create_test_inputs(batch_size, samples, cmplx=False):
    start_coeffs = get_random_biquads(cmplx)
    end_coeffs = get_random_biquads(cmplx)
    dtype = torch.complex128 if cmplx else torch.double

    A = (
        torch.stack(
            [
                torch.linspace(start_coeffs[i], end_coeffs[i], samples, dtype=dtype)
                for i in range(2)
            ]
        )
        .T.unsqueeze(0)
        .repeat(batch_size, 1, 1)
    )
    x = torch.randn(batch_size, samples, dtype=dtype)
    zi = torch.randn(batch_size, 2, dtype=dtype)
    return x, A, zi


@pytest.mark.parametrize(("a_requires_grad", "zi_requires_grad"), _GRAD_CASES)
@pytest.mark.parametrize("cmplx", [True, False])
@pytest.mark.parametrize("device", _DEVICES)
def test_allpole(
    a_requires_grad: bool,
    zi_requires_grad: bool,
    cmplx: bool,
    device: str,
):
    batch_size = 4
    samples = 32
    x, A, zi = tuple(
        x.to(device) for x in create_test_inputs(batch_size, samples, cmplx)
    )
    A.requires_grad = a_requires_grad
    x.requires_grad = True
    zi.requires_grad = zi_requires_grad

    assert gradcheck(AllPole.apply, (x, A, zi), check_forward_ad=True)
    assert gradgradcheck(AllPole.apply, (x, A, zi))


@pytest.mark.parametrize(("a_requires_grad", "zi_requires_grad"), _GRAD_CASES)
@pytest.mark.parametrize("cmplx", [True, False])
@pytest.mark.parametrize("device", _DEVICES)
def test_scan_recurrence(
    a_requires_grad: bool,
    zi_requires_grad: bool,
    cmplx: bool,
    device: str,
):
    batch_size = 2
    samples = 123
    dtype = torch.complex128 if cmplx else torch.double
    x = torch.randn(batch_size, samples, dtype=dtype, device=device)
    if cmplx:
        A = torch.rand(
            batch_size, samples, dtype=torch.double, device=device
        ).sqrt() * torch.exp(
            1j
            * torch.rand(batch_size, samples, dtype=torch.double, device=device)
            * 2
            * torch.pi
        )
    else:
        A = torch.rand(batch_size, samples, dtype=dtype, device=device) * 2 - 1
    zi = torch.randn(batch_size, dtype=dtype, device=device)

    A.requires_grad = a_requires_grad
    x.requires_grad = True
    zi.requires_grad = zi_requires_grad

    # ScanRecurrence takes (impulse, decay, init), i.e. (x, A, zi).
    assert gradcheck(ScanRecurrence.apply, (x, A, zi), check_forward_ad=True)
    assert gradgradcheck(ScanRecurrence.apply, (x, A, zi))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
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

    assert torch.allclose(y64, y32.double(), atol=1e-6), torch.max(
        torch.abs(y64 - y32.double())
    )


@pytest.mark.parametrize("device", _DEVICES)
def test_allpole_vmap(device: str):
    batch_size = 4
    samples = 40
    x, A, zi = tuple(
        x.to(device) for x in create_test_inputs(batch_size, samples, False)
    )
    y = torch.randn_like(x)

    A = A[:, 0, :].clone()

    A.requires_grad = True
    zi.requires_grad = True
    x.requires_grad = True

    args = (x, A, zi)

    def func(x, A, zi):
        return F.mse_loss(
            AllPole.apply(x, A[:, None, :].expand(-1, samples, -1), zi), y
        )

    jacs = jacfwd(func, argnums=tuple(range(len(args))))(*args)

    loss = func(*args)
    loss.backward()
    for jac, arg in zip(jacs, args):
        assert torch.allclose(jac, arg.grad)


@pytest.mark.parametrize("device", _DEVICES)
def test_scan_recurrence_vmap(device: str):
    batch_size = 3
    samples = 255
    x = torch.randn(batch_size, samples, dtype=torch.double, device=device)
    A = torch.rand(batch_size, samples, dtype=torch.double, device=device) * 2 - 1
    zi = torch.randn(batch_size, dtype=torch.double, device=device)
    y = torch.randn(batch_size, samples, dtype=torch.double, device=device)

    A.requires_grad = True
    x.requires_grad = True
    zi.requires_grad = True

    args = (x, A, zi)

    def func(x, A, zi):
        return F.mse_loss(ScanRecurrence.apply(x, A, zi), y)

    jacs = jacfwd(func, argnums=tuple(range(len(args))))(*args)

    loss = func(*args)
    loss.backward()
    for jac, arg in zip(jacs, args):
        assert torch.allclose(jac, arg.grad)
