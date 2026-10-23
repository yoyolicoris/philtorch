import importlib
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


def _dtw_dp_backward_calls(loss, shape):
    # loss's first-order backward, through dtw_dp's.
    from philtorch.align import _dtw_kernels

    x = torch.rand(shape, device="cuda", requires_grad=True)
    return _recorded_calls(_dtw_kernels, "dtw_dp_backward", lambda: loss(x).backward())


def _dtw_accumulate_calls():
    # Only the gradient's own gradient builds on the accumulation op.
    from philtorch.align import _dtw_kernels, dtw

    cost = torch.rand(2, 5, 6, device="cuda", requires_grad=True)

    def second_order():
        (grad,) = torch.autograd.grad(dtw(cost, gamma=0.1).sum(), cost, create_graph=True)
        grad.sum().backward()

    return _recorded_calls(_dtw_kernels, "accumulate", second_order)


def _ctc_viterbi_calls():
    from philtorch.align import _dtw_kernels, forced_align

    log_probs = torch.randn(2, 7, 4, device="cuda").log_softmax(-1)
    targets = torch.tensor([[1, 2, 2], [3, 1, 0]], device="cuda")
    return _recorded_calls(
        _dtw_kernels, "ctc_viterbi", lambda: forced_align(log_probs, targets, (7, 6), (3, 2))
    )


def _features_inputs():
    x, y = (torch.rand(2, n, 3, device="cuda", requires_grad=True) for n in (5, 6))
    return x, y, torch.tensor([[[0, 5]] * 5] * 2, dtype=torch.int32, device="cuda")


def _dtw_frontier_calls():
    # The no-gradient feature form of dtw.
    from philtorch.align import _dtw_kernels, dtw

    x, y = (t.detach() for t in _features_inputs()[:2])
    with torch.no_grad():
        return _recorded_calls(_dtw_kernels, "frontier", lambda: dtw(x=x, y=y, gamma=0.5))


def _dtw_dp_features_calls(op, seeded=False):
    # The forward and, through its backward, dtw_dp_backward_features; seeded, the
    # last four rows of the grid, from a given row.
    from philtorch.align import _dtw_kernels

    x, y, window = _features_inputs()
    seed = torch.rand(2, 6, device="cuda", requires_grad=True)

    def run():
        options = ("symmetric", 2.0, False, False)
        if seeded:
            D = _dtw_kernels.dtw_dp_features(x, y, window, 1e4, 1.5, True, *options, 4, 6, seed, 1)
        else:
            D = _dtw_kernels.dtw_dp_features(x, y, window, 1e4, 1.5, True, *options, 5, 6)
        D.sum().backward()

    return _recorded_calls(_dtw_kernels, op, run)


def _dtw_grid_cost_calls():
    # From the second derivatives of soft-DTW from frames, with a shear.
    from philtorch.align import _dtw_kernels, dtw

    gen = torch.Generator("cuda").manual_seed(0)
    x, y = (torch.randn(2, n, 3, device="cuda", generator=gen, requires_grad=True) for n in (6, 4))

    def run():
        loss = dtw(x=x, y=y, gamma=0.5, step_pattern="asymmetric", band=2.0).sum()
        (grad,) = torch.autograd.grad(loss, x, create_graph=True)
        grad.sum().backward()

    return _recorded_calls(_dtw_kernels, "grid_cost", run)


def _dtw_backtrack_calls():
    from philtorch.align import _dtw_kernels, dtw_path

    cost = torch.rand(2, 9, 6, device="cuda")
    return _recorded_calls(_dtw_kernels, "backtrack", lambda: dtw_path(cost))


def _dtw_linear_path_calls():
    # The path search of DTW from frames, for its gradient.
    from philtorch.align import dtw

    module = importlib.import_module("philtorch.align.dtw")
    x, y = (torch.rand(2, n, 3, device="cuda", requires_grad=True) for n in (9, 6))
    return _recorded_calls(module, "linear_path", lambda: dtw(x=x, y=y).sum().backward())


def _hmm_viterbi_chain_calls():
    from philtorch.estimation import hmm_viterbi

    hmm = importlib.import_module("philtorch.estimation.hmm")
    log_emit = torch.randn(2, 6, 3, device="cuda").log_softmax(-1)
    log_trans = torch.randn(3, 3, device="cuda").log_softmax(-1)
    log_init = torch.randn(3, device="cuda").log_softmax(-1)
    return _recorded_calls(
        hmm, "_viterbi_chain", lambda: hmm_viterbi(log_emit, log_trans, log_init)
    )


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
        pytest.param(
            "dtw_dp_backward",
            lambda: _dtw_dp_backward_calls(_dtw(0.1, diagonal_weight=2.0), (2, 5, 6)),
            id="dtw_dp_backward-weighted",
        ),
        pytest.param(
            "dtw_dp_backward",
            lambda: _dtw_dp_backward_calls(_ctc_loss, (2, 7, 4)),
            id="dtw_dp_backward-ctc",
        ),
        pytest.param("dtw_accumulate", _dtw_accumulate_calls, id="dtw_accumulate"),
        pytest.param("ctc_viterbi", _ctc_viterbi_calls, id="ctc_viterbi"),
        pytest.param("dtw_backtrack", _dtw_backtrack_calls, id="dtw_backtrack"),
        pytest.param(
            "dtw_dp_features",
            lambda: _dtw_dp_features_calls("dtw_dp_features"),
            id="dtw_dp_features",
        ),
        pytest.param(
            "dtw_dp_features",
            lambda: _dtw_dp_features_calls("dtw_dp_features", seeded=True),
            id="dtw_dp_features-seeded",
        ),
        pytest.param(
            "dtw_dp_backward_features",
            lambda: _dtw_dp_features_calls("dtw_dp_backward_features"),
            id="dtw_dp_backward_features",
        ),
        pytest.param(
            "dtw_dp_backward_features",
            lambda: _dtw_dp_features_calls("dtw_dp_backward_features", seeded=True),
            id="dtw_dp_backward_features-seeded",
        ),
        pytest.param("dtw_frontier", _dtw_frontier_calls, id="dtw_frontier"),
        pytest.param("dtw_grid_cost", _dtw_grid_cost_calls, id="dtw_grid_cost"),
        pytest.param("dtw_linear_path", _dtw_linear_path_calls, id="dtw_linear_path"),
        pytest.param("hmm_viterbi_chain", _hmm_viterbi_chain_calls, id="hmm_viterbi_chain"),
        pytest.param("weighted_contract", _weighted_contract_calls, id="weighted_contract"),
    ],
)
def test_alignment_and_hmm_ops_opcheck(op, calls):
    # The Triton custom ops of philtorch.align and philtorch.estimation, on the
    # arguments real calls give them.
    recorded = calls()
    assert recorded, f"{op} was not called"
    torch.library.opcheck(getattr(torch.ops.philtorch, op), recorded[0])


@pytest.mark.cuda
@pytest.mark.parametrize(
    "name, op",
    [
        ("dtw_dp_strip", "dtw_dp_strip"),
        ("dtw_dp_backward_strip", "dtw_dp_backward_strip"),
        ("accumulate", "dtw_accumulate"),
    ],
)
def test_dtw_strip_ops_opcheck(monkeypatch, name, op):
    from philtorch.align import _dtw_kernels, dtw

    # Strips of two tile anti-diagonals over a grid of 7 by 3 tiles, a band, and
    # second derivatives, which reach the backward's own ops; for the forward,
    # a strip with a seed and the last one.
    monkeypatch.setattr(_dtw_kernels, "_tile", lambda *args: (3, 16))
    gen = torch.Generator("cuda").manual_seed(0)
    x, y = (
        torch.randn(2, n, 3, device="cuda", generator=gen, requires_grad=True) for n in (20, 40)
    )

    def run():
        loss = dtw(x=x, y=y, gamma=0.5, diagonal_weight=2.0, band=12.0, checkpoint=3).sum()
        (grad,) = torch.autograd.grad(loss, x, create_graph=True)
        grad.sum().backward()

    recorded = _recorded_calls(_dtw_kernels, name, run)
    assert recorded, f"{name} was not called"
    calls = (
        [recorded[1], max(recorded, key=lambda args: args[-1])]
        if name == "dtw_dp_strip"
        else recorded[:1]
    )
    for args in calls:
        torch.library.opcheck(getattr(torch.ops.philtorch, op), args)


def _ctc_loss(log_probs, lengths=((7, 6), (3, 2))):
    from philtorch.align import ctc_loss

    targets = torch.tensor([[1, 2, 2], [3, 1, 0]], device="cuda")
    return ctc_loss(log_probs.log_softmax(-1), targets, *lengths, reduction="sum")


def _ctc_loss_tensor_lengths(log_probs):
    return _ctc_loss(log_probs, (torch.tensor([7, 6]), torch.tensor([3, 2])))


def _soft_dtw_divergence(x):
    from philtorch.align import soft_dtw_divergence

    y = x.flip(1)[:, :5]
    costs = [torch.cdist(a, b).square() for a, b in ((x, y), (x, x), (y, y))]
    return soft_dtw_divergence(*costs, 0.1).sum()


def _soft_dtw_band(x):
    # From frames, whose band Inductor must round as eager does.
    from philtorch.align import dtw

    return dtw(x=x, y=x[:, :20].flip(1), gamma=0.5, band=4.0).sum()


def _dtw_features(x):
    # DTW from frames with gradients: the path's search is an op.
    from philtorch.align import dtw

    return dtw(x=x, y=x[:, :20].flip(1)).sum()


def _dtw(gamma, **options):
    def loss(cost):
        from philtorch.align import dtw

        return dtw(cost, gamma, **options).sum()

    return loss


_LENGTHS = (torch.tensor([6, 4]), torch.tensor([5, 3]))


@pytest.mark.cuda
@pytest.mark.parametrize(
    "loss, shape",
    [
        pytest.param(_dtw(0.0, diagonal_weight=2.0), (2, 6, 5), id="dtw"),
        pytest.param(_dtw(0.1, diagonal_weight=2.0), (2, 6, 5), id="soft_dtw"),
        pytest.param(_dtw(0.1, lengths=_LENGTHS), (2, 6, 5), id="soft_dtw-lengths"),
        pytest.param(_dtw(0.1, step_pattern="asymmetric"), (2, 6, 5), id="soft_dtw-asymmetric"),
        pytest.param(_soft_dtw_band, (2, 30, 3), id="soft_dtw-band"),
        pytest.param(_dtw_features, (2, 30, 3), id="dtw-features"),
        pytest.param(_soft_dtw_divergence, (2, 7, 3), id="soft_dtw_divergence"),
        pytest.param(_ctc_loss, (2, 7, 4), id="ctc_loss"),
        pytest.param(_ctc_loss_tensor_lengths, (2, 7, 4), id="ctc_loss-tensor_lengths"),
    ],
)
def test_compiled_alignment_gradients(loss, shape):
    # Compiled code differentiates dtw_dp through its registered backward,
    # the fused kernels as one op: the same gradients as eager, in one graph.
    x = torch.randn(shape, device="cuda", generator=torch.Generator("cuda").manual_seed(0))
    x.requires_grad_()
    (expected,) = torch.autograd.grad(loss(x), x)
    (actual,) = torch.autograd.grad(torch.compile(loss, fullgraph=True)(x), x)
    torch.testing.assert_close(actual, expected)


@pytest.mark.cuda
def test_compiled_dtw_strips_with_dynamic_shapes(monkeypatch):
    from philtorch.align import _dtw_kernels, dtw

    # Strips of two tile anti-diagonals, whose tile is chosen by the shapes:
    # compiled as a whole, with them dynamic.
    monkeypatch.setattr(_dtw_kernels, "_tile", lambda *args: (3, 16))
    gen = torch.Generator("cuda").manual_seed(0)
    x, y = (
        torch.randn(2, n, 3, device="cuda", generator=gen, requires_grad=True) for n in (12, 36)
    )

    def loss(x, y):
        return dtw(x=x, y=y, gamma=0.5, checkpoint=3).sum()

    expected = torch.autograd.grad(loss(x, y), (x, y))
    compiled = torch.compile(loss, fullgraph=True, dynamic=True)
    actual = torch.autograd.grad(compiled(x, y), (x, y))
    torch.testing.assert_close(actual, expected)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "soft, steps, diag_weight",
    [
        (True, "symmetric", 1.0),
        (True, "symmetric", 2.0),
        (True, "orthogonal", 1.0),
        (False, "symmetric", 2.0),
    ],
)
def test_dtw_dp_backward_registered_gradients(soft, steps, diag_weight):
    # dtw_dp_backward's registered backward, which defines dtw_dp's second and
    # higher derivatives, against finite differences of the fused kernel, with
    # D, cost and grad independent; and its own derivatives. DTW's hard shares
    # make its gradient piecewise constant in D and the cost.
    from philtorch.align._dtw_kernels import dtw_dp_backward

    gen = torch.Generator("cuda").manual_seed(0)
    cost = torch.rand(2, 4, 5, device="cuda", dtype=torch.float64, generator=gen)
    D = torch.ops.philtorch.dtw_dp(cost, None, soft, steps, diag_weight)
    grad = torch.rand(D.shape, device="cuda", dtype=torch.float64, generator=gen)
    inputs = tuple(t.requires_grad_() for t in (D, cost, grad))

    def fn(D, cost, grad):
        return dtw_dp_backward(D, cost, None, grad, soft, steps, diag_weight)

    assert torch.autograd.gradcheck(fn, inputs)
    assert torch.autograd.gradgradcheck(fn, inputs)


@pytest.mark.cuda
def test_soft_dtw_asymmetric_second_derivatives():
    # The asymmetric steps are PyTorch's shear around the orthogonal kernel.
    from philtorch.align import dtw

    gen = torch.Generator("cuda").manual_seed(0)
    cost = torch.rand(2, 5, 4, device="cuda", dtype=torch.float64, generator=gen)
    cost.requires_grad_()
    assert torch.autograd.gradgradcheck(lambda c: dtw(c, 0.5, step_pattern="asymmetric"), (cost,))


def _dtw_path(cost):
    from philtorch.align import dtw_path

    return dtw_path(cost, lengths=_LENGTHS)


def _dtw_path_features(x):
    from philtorch.align import dtw_path

    return dtw_path(x=x, y=x[:, :4].flip(1))


def _forced_align(log_probs):
    from philtorch.align import forced_align

    targets = torch.tensor([[1, 2, 2], [3, 1, 0]], device="cuda")
    return forced_align(log_probs.log_softmax(-1), targets, (7, 6), (3, 2))


def _hmm_viterbi(x):
    from philtorch.estimation import hmm_viterbi

    # The transitions and initial probabilities from x too, (7, 4) frames each.
    log_trans, log_init = x[0, :4].log_softmax(-1), x[1, 0].log_softmax(-1)
    return hmm_viterbi(x.log_softmax(-1), log_trans, log_init)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "fn, shape",
    [
        pytest.param(_dtw_path, (2, 6, 5), id="dtw_path"),
        pytest.param(_dtw_path_features, (2, 6, 5), id="dtw_path-features"),
        pytest.param(_forced_align, (2, 7, 4), id="forced_align"),
        pytest.param(_hmm_viterbi, (2, 7, 4), id="hmm_viterbi"),
    ],
)
def test_compiled_alignment_paths(fn, shape):
    # The paths' kernels run as ops, which compiled code calls without tracing
    # them: the same paths as eager, and their scores, in one graph.
    x = torch.randn(shape, device="cuda", generator=torch.Generator("cuda").manual_seed(0))
    for actual, expected in zip(torch.compile(fn, fullgraph=True)(x), fn(x), strict=True):
        torch.testing.assert_close(actual, expected)


@pytest.mark.cuda
def test_dtw_path_search_reduce_overhead():
    # The path search reads sizes back to the host, which a CUDA graph can't
    # capture: its op is left out of the graphs, and repeated calls of the
    # compiled code, which replay them, give eager's values and gradients.
    compiled = torch.compile(_dtw_features, mode="reduce-overhead")
    generator = torch.Generator("cuda").manual_seed(0)
    for _ in range(4):
        x = torch.randn(2, 30, 3, device="cuda", generator=generator).requires_grad_()
        (expected,) = torch.autograd.grad(_dtw_features(x), x)
        (actual,) = torch.autograd.grad(compiled(x), x)
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


@pytest.mark.parametrize("op, shared", _LAYOUTS)
def test_opcheck_non_contiguous(op, shared):
    # The kernels return contiguous outputs whatever their inputs' strides, so
    # the fakes must too, or compiled code with permuted inputs fails.
    inputs = tuple(
        t.transpose(0, 1).contiguous().transpose(0, 1) if t.dim() > 1 else t
        for t in _op_inputs(op, shared, torch.float64)
    )
    torch.library.opcheck(getattr(torch.ops.philtorch, op), inputs)
