import pytest
import torch
from torch.autograd.gradcheck import gradcheck, gradgradcheck
from philtorch.lpv.ssm import MatrixRecurrence, PARARNN_AVAILABLE
from philtorch.lti.ssm import LTIMatrixRecurrence
from philtorch.lti.recur import LTIRecurrence
from philtorch import HELION_LOADED


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
    ("cmplx", "order", "device"),
    [
        (True, 2, "cpu"),
        (False, 2, "cpu"),
        (True, 3, "cpu"),
        (False, 3, "cpu"),
        pytest.param(
            True,
            2,
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
        pytest.param(
            False,
            2,
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
        pytest.param(
            False,
            3,
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available() or not PARARNN_AVAILABLE,
                reason="CUDA or `pararnn` not available",
            ),
        ),
        pytest.param(
            False,
            4,
            "cuda",
            marks=pytest.mark.skipif(not HELION_LOADED, reason="Helion not available"),
        ),
    ],
)
@pytest.mark.parametrize(
    "share_A",
    [True, False],
)
def test_N_order(
    x_requires_grad: bool,
    A_requires_grad: bool,
    zi_requires_grad: bool,
    share_A: bool,
    samples: int,
    cmplx: bool,
    device: str,
    order: int,
):
    batch_size = 3
    x, A, zi = tuple(
        x.to(device)
        for x in [
            torch.randn(
                batch_size,
                samples,
                order,
                dtype=torch.double if not cmplx else torch.complex128,
            ),
            torch.randn(
                *(
                    (batch_size, samples, order, order)
                    if not share_A
                    else (samples, order, order)
                ),
                dtype=torch.double if not cmplx else torch.complex128,
            )
            / order**2,
            torch.randn(
                batch_size, order, dtype=torch.double if not cmplx else torch.complex128
            ),
        ]
    )
    A.requires_grad = A_requires_grad
    x.requires_grad = x_requires_grad
    zi.requires_grad = zi_requires_grad

    assert gradcheck(MatrixRecurrence.apply, (A, zi, x), check_forward_ad=True)
    assert gradgradcheck(MatrixRecurrence.apply, (A, zi, x), check_fwd_over_rev=True)


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
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
    ],
)
@pytest.mark.parametrize(
    "share_A",
    [True, False],
)
def test_second_order_lti(
    x_requires_grad: bool,
    A_requires_grad: bool,
    zi_requires_grad: bool,
    samples: int,
    cmplx: bool,
    device: str,
    share_A: bool,
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
                *((batch_size, order, order) if not share_A else (order, order)),
                dtype=torch.double if not cmplx else torch.complex128,
            )
            * 0.25,
            torch.randn(
                batch_size, order, dtype=torch.double if not cmplx else torch.complex128
            ),
        ]
    )
    A.requires_grad = A_requires_grad
    x.requires_grad = x_requires_grad
    zi.requires_grad = zi_requires_grad

    assert gradcheck(LTIMatrixRecurrence.apply, (A, zi, x), check_forward_ad=True)
    assert gradgradcheck(LTIMatrixRecurrence.apply, (A, zi, x), check_fwd_over_rev=True)


@pytest.mark.parametrize(
    "x_requires_grad",
    [True],
)
@pytest.mark.parametrize(
    "a_requires_grad",
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
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
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
            torch.randn(
                batch_size, dtype=torch.double if not cmplx else torch.complex128
            ),
        ]
    )
    a.requires_grad = a_requires_grad
    x.requires_grad = x_requires_grad
    zi.requires_grad = zi_requires_grad

    assert gradcheck(LTIRecurrence.apply, (a, zi, x), check_forward_ad=True)
    assert gradgradcheck(LTIRecurrence.apply, (a, zi, x), check_fwd_over_rev=True)
