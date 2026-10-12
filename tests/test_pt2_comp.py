from unittest import mock

import pytest
import torch

import philtorch  # noqa: F401  (registers its ops)


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


# The recurrence ops, with A shared over the batch or one per signal.
_RECURRENCE_OPS = ["recur2", "recurN", "lti_recur2", "lti_recurN", "lti_recur"]
# Each op with the layouts it takes: torchlpc's scan and lpc have per-signal
# coefficients only.
_LAYOUTS = [(op, shared) for op in _RECURRENCE_OPS for shared in (True, False)]
_LAYOUTS += [("scan", False), ("lpc", False)]


@pytest.mark.parametrize("op, shared", _LAYOUTS)
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
def test_registered_gradients(op, shared, dtype):
    # The ops' registered backwards, which compiled code differentiates, map
    # each op's layout to philtorch._recur's gradients: check them, and their
    # own derivatives, against finite differences.
    fn = getattr(torch.ops.philtorch, op)
    inputs = tuple(t.requires_grad_() for t in _op_inputs(op, shared, dtype))
    assert torch.autograd.gradcheck(fn, inputs)
    assert torch.autograd.gradgradcheck(fn, inputs)


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


def _dtw_dp_backward_calls():
    from philtorch.align import _dtw_kernels, dtw

    cost = torch.rand(2, 5, 6, device="cuda", requires_grad=True)
    return _recorded_calls(
        _dtw_kernels, "dtw_dp_backward", lambda: dtw(cost, gamma=0.1).sum().backward()
    )


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
        pytest.param("dtw_dp", _dtw_dp_calls, id="dtw_dp"),
        pytest.param("dtw_dp_backward", _dtw_dp_backward_calls, id="dtw_dp_backward"),
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


def _ctc_loss(log_probs):
    from philtorch.align import ctc_loss

    targets = torch.tensor([[1, 2, 2], [3, 1, 0]], device="cuda")
    return ctc_loss(log_probs.log_softmax(-1), targets, (7, 6), (3, 2), reduction="sum")


def _soft_dtw_divergence(x):
    from philtorch.align import soft_dtw_divergence

    y = x.flip(1)[:, :5]
    costs = [torch.cdist(a, b).square() for a, b in ((x, y), (x, x), (y, y))]
    return soft_dtw_divergence(*costs, 0.1).sum()


def _dtw(gamma):
    def loss(cost):
        from philtorch.align import dtw

        return dtw(cost, gamma, diagonal_weight=2.0).sum()

    return loss


@pytest.mark.cuda
@pytest.mark.parametrize(
    "loss, shape, fullgraph",
    [
        pytest.param(_dtw(0.0), (2, 6, 5), True, id="dtw"),
        pytest.param(_dtw(0.1), (2, 6, 5), True, id="soft_dtw"),
        pytest.param(_soft_dtw_divergence, (2, 7, 3), True, id="soft_dtw_divergence"),
        # ctc_loss checks its lengths' values, which breaks the graph.
        pytest.param(_ctc_loss, (2, 7, 4), False, id="ctc_loss"),
    ],
)
def test_compiled_alignment_gradients(loss, shape, fullgraph):
    # Compiled code differentiates dtw_dp through its registered backward,
    # the fused kernels as one op: the same gradients as eager.
    x = torch.randn(shape, device="cuda", generator=torch.Generator("cuda").manual_seed(0))
    x.requires_grad_()
    (expected,) = torch.autograd.grad(loss(x), x)
    (actual,) = torch.autograd.grad(torch.compile(loss, fullgraph=fullgraph)(x), x)
    torch.testing.assert_close(actual, expected)


# The ops with a CUDA kernel too; recurN and lti_recurN run on CPU only.
_CUDA_OPS = {"recur2", "lti_recur2", "lti_recur", "scan", "lpc"}


def _opcheck_cases():
    for op, shared in _LAYOUTS:
        for device in ("cpu", "cuda") if op in _CUDA_OPS else ("cpu",):
            marks = [pytest.mark.cuda] if device == "cuda" else []
            layout = "shared" if shared else "per-item"
            yield pytest.param(op, shared, device, marks=marks, id=f"{op}-{layout}-{device}")


@pytest.mark.parametrize("op, shared, device", list(_opcheck_cases()))
@pytest.mark.parametrize("dtype", [torch.float64, torch.complex128])
@pytest.mark.parametrize("grads", ["all", "first", "last"])
def test_opcheck(op, device, shared, dtype, grads):
    # Each op's fake, schema and registered gradients, as torch.compile
    # dispatches it. Gradients for every input, or for the first or the last
    # alone, take each backward's needs_input_grad branches both ways.
    inputs = [t.to(device) for t in _op_inputs(op, shared, dtype)]
    wanted = {"all": range(len(inputs)), "first": [0], "last": [len(inputs) - 1]}[grads]
    for i in wanted:
        inputs[i].requires_grad_()
    torch.library.opcheck(getattr(torch.ops.philtorch, op), tuple(inputs))


@pytest.mark.parametrize(
    "op, shared",
    [
        pytest.param(
            op,
            shared,
            marks=pytest.mark.xfail(
                strict=True, reason="the CPU kernel's output copies the input's strides (#80)"
            ),
        )
        if op == "scan"
        else (op, shared)
        for op, shared in _LAYOUTS
    ],
)
def test_opcheck_non_contiguous(op, shared):
    # The kernels return contiguous outputs whatever their inputs' strides, so
    # the fakes must too, or compiled code with permuted inputs fails.
    inputs = tuple(
        t.transpose(0, 1).contiguous().transpose(0, 1) if t.dim() > 1 else t
        for t in _op_inputs(op, shared, torch.float64)
    )
    torch.library.opcheck(getattr(torch.ops.philtorch, op), inputs)
