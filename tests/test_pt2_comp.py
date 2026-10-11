import pytest
import torch

import philtorch  # noqa: F401  (registers its ops)

_REQUIRES_GRAD_CASES = [
    (True, False, False),
    (False, True, False),
    (False, False, True),
    (True, True, False),
    (True, False, True),
    (False, True, True),
    (True, True, True),
]


@pytest.mark.parametrize(
    ("x_requires_grad", "A_requires_grad", "zi_requires_grad"),
    _REQUIRES_GRAD_CASES,
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
                *((batch_size, samples, order, order) if not share_A else (samples, order, order)),
                dtype=torch.double if not cmplx else torch.complex128,
            )
            / order**2,
            torch.randn(batch_size, order, dtype=torch.double if not cmplx else torch.complex128),
        ]
    )
    A.requires_grad = A_requires_grad
    x.requires_grad = x_requires_grad
    zi.requires_grad = zi_requires_grad

    torch.library.opcheck(torch.ops.philtorch.recurN, (A, zi, x))


@pytest.mark.parametrize(
    ("x_requires_grad", "A_requires_grad", "zi_requires_grad"),
    _REQUIRES_GRAD_CASES,
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
            marks=pytest.mark.cuda,
        ),
        pytest.param(
            False,
            "cuda",
            marks=pytest.mark.cuda,
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
                *((batch_size, samples, order, order) if not share_A else (samples, order, order)),
                dtype=torch.double if not cmplx else torch.complex128,
            )
            / order**2,
            torch.randn(batch_size, order, dtype=torch.double if not cmplx else torch.complex128),
        ]
    )
    A.requires_grad = A_requires_grad
    x.requires_grad = x_requires_grad
    zi.requires_grad = zi_requires_grad

    torch.library.opcheck(torch.ops.philtorch.recur2, (A, zi, x))


@pytest.mark.parametrize(
    ("x_requires_grad", "A_requires_grad", "zi_requires_grad"),
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
            torch.randn(batch_size, order, dtype=torch.double if not cmplx else torch.complex128),
        ]
    )
    A.requires_grad = A_requires_grad
    x.requires_grad = x_requires_grad
    zi.requires_grad = zi_requires_grad

    torch.library.opcheck(torch.ops.philtorch.lti_recur2, (A, zi, x))


@pytest.mark.parametrize(
    ("x_requires_grad", "A_requires_grad", "zi_requires_grad"),
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
            torch.randn(batch_size, order, dtype=torch.double if not cmplx else torch.complex128),
        ]
    )
    A.requires_grad = A_requires_grad
    x.requires_grad = x_requires_grad
    zi.requires_grad = zi_requires_grad

    torch.library.opcheck(torch.ops.philtorch.lti_recurN, (A, zi, x))


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
            torch.randn(batch_size, dtype=torch.double if not cmplx else torch.complex128),
        ]
    )
    a.requires_grad = a_requires_grad
    x.requires_grad = x_requires_grad
    zi.requires_grad = zi_requires_grad

    torch.library.opcheck(torch.ops.philtorch.lti_recur, (a, zi, x))


def _op_inputs(op, shared, dtype):
    """Stable inputs to a native recurrence op, in its own layout."""
    gen = torch.Generator().manual_seed(0)
    B, T = 2, 7

    def randn(*shape, scale=1.0):
        return (torch.randn(shape, generator=gen, dtype=torch.float64) * scale).to(dtype)

    if op == "scan":
        return randn(B, T), randn(B, T, scale=0.3), randn(B)
    if op == "lti_recur":
        return randn(*(() if shared else (B,)), scale=0.3), randn(B), randn(B, T)
    M = 2 if op.endswith("2") else 3
    lead = {
        "recur2": (T,) if shared else (B, T),
        "recurN": (T,) if shared else (B, T),
        "lti_recur2": () if shared else (B,),
        "lti_recurN": () if shared else (B,),
    }[op]
    return randn(*lead, M, M, scale=0.3 / M), randn(B, M), randn(B, T, M)


_RECURRENCE_OPS = ["recur2", "recurN", "lti_recur2", "lti_recurN", "lti_recur", "scan"]


@pytest.mark.parametrize("op", _RECURRENCE_OPS)
@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_registered_gradients(op, shared, dtype):
    # The ops' registered backwards, which compiled code differentiates, map
    # each op's layout to philtorch._recur's gradients: check them, and their
    # own derivatives, against finite differences.
    if op == "scan" and shared:
        pytest.skip("scan has no shared layout")
    fn = getattr(torch.ops.philtorch, op)
    inputs = tuple(t.requires_grad_() for t in _op_inputs(op, shared, dtype))
    assert torch.autograd.gradcheck(fn, inputs)
    assert torch.autograd.gradgradcheck(fn, inputs)
