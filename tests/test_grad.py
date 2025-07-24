import pytest
import torch
from torch.autograd.gradcheck import gradcheck, gradgradcheck
from philtorch.lpv.ssm import SecondOrderRecurrence


@pytest.mark.parametrize(
    "x_requires_grad",
    [True],
)
@pytest.mark.parametrize(
    "A_requires_grad",
    [True, False],
)
@pytest.mark.parametrize(
    "zi_requires_grad",
    [True, False],
)
@pytest.mark.parametrize(
    "samples",
    [11],
)
@pytest.mark.parametrize(
    "cmplx",
    [True, False],
)
@pytest.mark.parametrize(
    "device",
    [
        # "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ],
)
def test_second_order(
    x_requires_grad: bool,
    A_requires_grad: bool,
    zi_requires_grad: bool,
    samples: int,
    cmplx: bool,
    device: str,
):
    batch_size = 3
    order = 2
    x, A, zi = tuple(
        x.to(device)
        for x in [
            torch.randn(
                batch_size,
                samples,
                2,
                dtype=torch.double if not cmplx else torch.complex128,
            ),
            torch.randn(
                batch_size,
                samples,
                order,
                order,
                dtype=torch.double if not cmplx else torch.complex128,
            ),
            torch.randn(
                batch_size, order, dtype=torch.double if not cmplx else torch.complex128
            ),
        ]
    )
    A.requires_grad = A_requires_grad
    x.requires_grad = x_requires_grad
    zi.requires_grad = zi_requires_grad

    assert gradcheck(SecondOrderRecurrence.apply, (A, zi, x), check_forward_ad=True)
    assert gradgradcheck(SecondOrderRecurrence.apply, (A, zi, x))
