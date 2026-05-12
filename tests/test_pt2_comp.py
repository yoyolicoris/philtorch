import pytest
import torch

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
    "cmplx",
    [False],
)
@pytest.mark.parametrize(
    "device",
    [
        "cuda",
        # pytest.param(
        #     "cuda",
        #     marks=pytest.mark.skipif(
        #         not torch.cuda.is_available(), reason="CUDA not available"
        #     ),
        # ),
    ],
)
@pytest.mark.parametrize(
    "share_A",
    [True, False],
)
@pytest.mark.skipif(not HELION_LOADED, reason="Helion backend not loaded")
def test_hl_lti_recurN_pt2_compatibility(
    x_requires_grad: bool,
    A_requires_grad: bool,
    zi_requires_grad: bool,
    samples: int,
    cmplx: bool,
    device: str,
    share_A: bool,
):
    batch_size = 3
    order = 4
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
                *((batch_size, order, order) if not share_A else (order, order)),
                dtype=torch.double if not cmplx else torch.complex128,
            )
            * 0.25,
            torch.randn(
                batch_size,
                order,
                dtype=torch.double if not cmplx else torch.complex128,
            ),
        ]
    )
    A.requires_grad = A_requires_grad
    x.requires_grad = x_requires_grad
    zi.requires_grad = zi_requires_grad

    from philtorch import hl_lti_recurN

    torch.library.opcheck(hl_lti_recurN, (A, zi, x))


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
        (False, 4, "cuda"),
    ],
)
@pytest.mark.parametrize(
    "share_A",
    [True, False],
)
@pytest.mark.skipif(not HELION_LOADED, reason="Helion backend not loaded")
def test_hl_recurN_pt2_compatibility(
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
                batch_size,
                order,
                dtype=torch.double if not cmplx else torch.complex128,
            ),
        ]
    )
    A.requires_grad = A_requires_grad
    x.requires_grad = x_requires_grad
    zi.requires_grad = zi_requires_grad

    from philtorch import hl_recurN

    torch.library.opcheck(hl_recurN, (A, zi, x))


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
        (True, 3, "cpu"),
        (False, 3, "cpu"),
    ],
)
@pytest.mark.parametrize(
    "share_A",
    [True, False],
)
def test_recurN_pt2_compatibility(
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

    torch.library.opcheck(torch.ops.philtorch.recurN, (A, zi, x))


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
    ("cmplx", "device"),
    [
        (True, "cpu"),
        (False, "cpu"),
        pytest.param(
            True,
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA not available"
            ),
        ),
        pytest.param(
            False,
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
def test_recur2_pt2_compatibility(
    x_requires_grad: bool,
    A_requires_grad: bool,
    zi_requires_grad: bool,
    share_A: bool,
    samples: int,
    cmplx: bool,
    device: str,
):
    order = 2
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

    torch.library.opcheck(torch.ops.philtorch.recur2, (A, zi, x))


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
def test_lti_recur2_pt2_compatibility(
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

    torch.library.opcheck(torch.ops.philtorch.lti_recur2, (A, zi, x))


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
        # pytest.param(
        #     "cuda",
        #     marks=pytest.mark.skipif(
        #         not torch.cuda.is_available(), reason="CUDA not available"
        #     ),
        # ),
    ],
)
@pytest.mark.parametrize(
    "share_A",
    [True, False],
)
def test_lti_recurN_pt2_compatibility(
    x_requires_grad: bool,
    A_requires_grad: bool,
    zi_requires_grad: bool,
    samples: int,
    cmplx: bool,
    device: str,
    share_A: bool,
):
    batch_size = 3
    order = 4
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

    torch.library.opcheck(torch.ops.philtorch.lti_recurN, (A, zi, x))


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
def test_lti_recur_pt2_compatibility(
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

    torch.library.opcheck(torch.ops.philtorch.lti_recur, (a, zi, x))
