import pytest
import torch
from torch.autograd.gradcheck import gradcheck, gradgradcheck

from philtorch.lti.recur import LTIRecurrence

_REQUIRES_GRAD_CASES = (
    (True, False, False),
    (False, True, False),
    (False, False, True),
    (True, True, False),
    (True, False, True),
    (False, True, True),
    (True, True, True),
)


@pytest.mark.parametrize(
    ("x_requires_grad", "a_requires_grad", "zi_requires_grad"),
    _REQUIRES_GRAD_CASES,
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
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.cuda,
        ),
    ],
)
@pytest.mark.parametrize(
    "share_a",
    [True, False],
)
def test_first_order_lti(
    x_requires_grad: bool,
    a_requires_grad: bool,
    zi_requires_grad: bool,
    samples: int,
    cmplx: bool,
    device: str,
    share_a: bool,
):
    batch_size = 3
    x, a, zi = tuple(
        x.to(device)
        for x in [
            torch.randn(
                batch_size,
                samples,
                dtype=torch.double if not cmplx else torch.complex128,
            ),
            torch.randn(
                *((batch_size,) if not share_a else (1,)),
                dtype=torch.double if not cmplx else torch.complex128,
            )
            * 0.5,
            torch.randn(batch_size, dtype=torch.double if not cmplx else torch.complex128),
        ]
    )
    a.requires_grad = a_requires_grad
    x.requires_grad = x_requires_grad
    zi.requires_grad = zi_requires_grad

    assert gradcheck(LTIRecurrence.apply, (a, zi, x), check_forward_ad=True)
    assert gradgradcheck(LTIRecurrence.apply, (a, zi, x), check_fwd_over_rev=True)
