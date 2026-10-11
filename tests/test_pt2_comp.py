from unittest import mock

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
    if op == "lpc":
        return randn(B, T), randn(B, T, 2, scale=0.2), randn(B, 2)
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


_RECURRENCE_OPS = ["recur2", "recurN", "lti_recur2", "lti_recur", "lti_recurN", "scan", "lpc"]


@pytest.mark.parametrize("op", _RECURRENCE_OPS)
@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_registered_gradients(op, shared, dtype):
    # The ops' registered backwards, which compiled code differentiates, map
    # each op's layout to philtorch._recur's gradients: check them, and their
    # own derivatives, against finite differences.
    if op in ("scan", "lpc") and shared:
        pytest.skip(f"{op} has no shared layout")
    fn = getattr(torch.ops.philtorch, op)
    inputs = tuple(t.requires_grad_() for t in _op_inputs(op, shared, dtype))
    assert torch.autograd.gradcheck(fn, inputs)
    assert torch.autograd.gradgradcheck(fn, inputs)


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)],
)
@pytest.mark.parametrize("op", ["scan", "lpc"])
def test_vendored_ops_opcheck(op, device):
    # Their fakes and registered gradients, as torch.compile dispatches them.
    inputs = tuple(t.to(device).requires_grad_() for t in _op_inputs(op, False, torch.float64))
    torch.library.opcheck(getattr(torch.ops.philtorch, op), inputs)


def _pararnn_inputs(M, dtype=torch.float64):
    gen = torch.Generator().manual_seed(0)
    A = torch.randn(2, 7, M, M, generator=gen, dtype=dtype) * 0.3 / M
    zi, x = (
        torch.randn(2, M, generator=gen, dtype=dtype),
        torch.randn(2, 7, M, generator=gen, dtype=dtype),
    )
    jac = torch.nn.functional.pad(-A, (0, 0, 0, 0, 1, 0))
    return [t.cuda().requires_grad_() for t in (jac, torch.cat([zi[:, None], x], 1))]


@pytest.mark.cuda
@pytest.mark.parametrize("M", [2, 3])
def test_pararnn_opcheck_and_registered_gradients(M):
    op = getattr(torch.ops.parallel_reduce_cuda, f"parallel_reduce_block_diag_{M}x{M}_cuda")
    inputs = tuple(_pararnn_inputs(M))
    torch.library.opcheck(op, inputs)
    assert torch.autograd.gradcheck(op, inputs)
    assert torch.autograd.gradgradcheck(op, inputs)


@pytest.mark.cuda
@pytest.mark.parametrize("time_invariant", [True, False])
def test_recur_triton_registered_gradients(time_invariant):
    # Compiled code differentiates the Triton op through its registered
    # backward: check it, and its own derivatives, against finite differences.
    from philtorch._recur import recur_triton

    gen = torch.Generator().manual_seed(0)
    A = torch.randn(2, 1 if time_invariant else 7, 3, 3, generator=gen, dtype=torch.float64) * 0.2
    zi, x = (
        torch.randn(2, 3, generator=gen, dtype=torch.float64),
        torch.randn(2, 7, 3, generator=gen, dtype=torch.float64),
    )
    inputs = tuple(t.cuda().requires_grad_() for t in (A, zi, x))
    assert torch.autograd.gradcheck(recur_triton, inputs)
    assert torch.autograd.gradgradcheck(recur_triton, inputs)


def _recorded_calls(module, name, run):
    """The arguments of every call to module.name while run() runs, as leaves."""
    calls, original = [], getattr(module, name)

    def record(*args):
        calls.append(
            tuple(
                a.detach().clone().requires_grad_(a.requires_grad)
                if isinstance(a, torch.Tensor)
                else a
                for a in args
            )
        )
        return original(*args)

    with mock.patch.object(module, name, record):
        run()
    return calls


def _dtw_dp_calls():
    from philtorch.align import _dtw_kernels, dtw

    cost = torch.rand(2, 5, 6, device="cuda", requires_grad=True)
    return _recorded_calls(_dtw_kernels, "dtw_dp", lambda: dtw(cost, gamma=0.1).sum().backward())


def _dtw_accumulate_calls():
    # Only the gradient's own gradient builds on the accumulation op.
    from philtorch.align import _dtw_kernels, dtw

    cost = torch.rand(2, 5, 6, device="cuda", requires_grad=True)

    def second_order():
        (grad,) = torch.autograd.grad(dtw(cost, gamma=0.1).sum(), cost, create_graph=True)
        grad.sum().backward()

    return _recorded_calls(_dtw_kernels, "accumulate", second_order)


def _weighted_contract_calls():
    # Shared transitions: their gradient sums over the batch and steps.
    from philtorch.estimation import _contract_kernels, hmm_filter

    log_emit = torch.randn(2, 6, 3, device="cuda").log_softmax(-1)
    log_trans = torch.randn(3, 3, device="cuda").log_softmax(-1).requires_grad_()
    log_init = torch.randn(3, device="cuda").log_softmax(-1)
    return _recorded_calls(
        _contract_kernels,
        "weighted_contract",
        lambda: hmm_filter(log_emit, log_trans, log_init)[0].sum().backward(),
    )


@pytest.mark.cuda
@pytest.mark.parametrize(
    "op, calls",
    [
        pytest.param(
            "dtw_dp",
            _dtw_dp_calls,
            id="dtw_dp",
            marks=pytest.mark.xfail(
                strict=True,
                reason="its first-order backward launches Triton kernels directly (#79)",
            ),
        ),
        pytest.param("dtw_accumulate", _dtw_accumulate_calls, id="dtw_accumulate"),
        pytest.param("weighted_contract", _weighted_contract_calls, id="weighted_contract"),
    ],
)
def test_alignment_and_hmm_ops_opcheck(op, calls):
    # The Triton custom ops of philtorch.align and philtorch.estimation, on the
    # arguments real calls give them.
    recorded = calls()
    assert recorded, f"{op} was not called"
    torch.library.opcheck(getattr(torch.ops.philtorch, op), recorded[0])
