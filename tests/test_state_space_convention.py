"""Every state-space entry point follows the textbook convention.

That is, with x the state and u the input,

    x[n + 1] = A[n] x[n] + B[n] u[n],
    y[n] = C[n] x[n] + D[n] u[n],
    x[0] = zi,

where the matrices at step n take the state from step n to n + 1. Each
function below is written as such a model and checked against one simulator,
so a structure that applies its coefficients a step early or late fails here.
"""

import warnings
from unittest import mock

import pytest
import torch

import philtorch._ssm as _ssm
import philtorch.lpv as lpv
import philtorch.lti as lti
from philtorch.mat import companion

# For the tests that pass unroll_factor=1 without checking its warning.
ignore_one = pytest.mark.filterwarnings("ignore:unroll_factor=1:FutureWarning")

BATCH, N, M = 2, 10, 3


def _simulate(A, B, C, D, u, x0):
    """Run the textbook model; return y and the states x[1], ..., x[N]."""
    x = x0
    ys, states = [], []
    for n in range(u.size(1)):
        ys.append((C[:, n] * x).sum(-1) + D[:, n] * u[:, n])
        x = (A[:, n] @ x.unsqueeze(-1)).squeeze(-1) + B[:, n] * u[:, n, None]
        states.append(x)
    return torch.stack(ys, 1), torch.stack(states, 1)


@pytest.fixture
def signals():
    gen = torch.Generator().manual_seed(0)

    def randn(*shape, scale=1.0):
        return torch.randn(*shape, dtype=torch.float64, generator=gen) * scale

    return {
        "u": randn(BATCH, N),
        "b": randn(BATCH, N, M + 1, scale=0.4),
        "a": randn(BATCH, N, M, scale=0.2),
        "A": randn(BATCH, N, M, M, scale=0.3),
        "B": randn(BATCH, N, M),
        "C": randn(BATCH, N, M),
        "D": randn(BATCH, N),
        "zi": randn(BATCH, M),
    }


def _ones(*shape):
    return torch.ones(*shape, dtype=torch.float64)


def _e1():
    e1 = torch.zeros(BATCH, N, M, dtype=torch.float64)
    e1[..., 0] = 1.0
    return e1


def _shift():
    """S with S x = [x_2, ..., x_M, 0]: S.mT shifts the other way."""
    return torch.diag(_ones(M - 1), 1).expand(BATCH, N, M, M)


def _zeros_state():
    return torch.zeros(BATCH, M, dtype=torch.float64)


# The filter structures as textbook models. a is the denominator without the
# leading 1, b the numerator, and both have the same order M here.


def _allpole_direct(a, u):
    # x[n] = [y[n - 1], ..., y[n - M]]
    return _simulate(companion(a), _e1(), -a, _ones(BATCH, N), u, _zeros_state())[0]


def _allpole_transposed(a, u):
    return _simulate(companion(a).mT, -a, _e1(), _ones(BATCH, N), u, _zeros_state())[0]


def _fir_direct(b, u):
    # x[n] = [u[n - 1], ..., u[n - M]]
    return _simulate(_shift().mT, _e1(), b[..., 1:], b[..., 0], u, _zeros_state())[0]


def _fir_transposed(b, u):
    return _simulate(_shift(), b[..., 1:], _e1(), b[..., 0], u, _zeros_state())[0]


def _direct_form_two(b, a, u):
    b0 = b[..., 0]
    return _simulate(companion(a), _e1(), b[..., 1:] - b0[..., None] * a, b0, u, _zeros_state())[0]


def _transposed_direct_form_two(b, a, u):
    b0 = b[..., 0]
    return _simulate(companion(a).mT, b[..., 1:] - b0[..., None] * a, _e1(), b0, u, _zeros_state())[
        0
    ]


def test_lpv_state_space(signals):
    s = signals
    y_ref, states = _simulate(s["A"], s["B"], s["C"], s["D"], s["u"], s["zi"])
    y, zf = lpv.state_space(s["A"], s["u"], B=s["B"], C=s["C"], D=s["D"], zi=s["zi"])
    assert torch.allclose(y, y_ref)
    assert torch.allclose(zf, states[:, -1])


# 4 leaves a partial last block of the unrolled recursions, 5 none.
@ignore_one
@pytest.mark.parametrize("unroll_factor", [None, 1, 4, 5, N])
def test_lpv_state_space_recursion_returns_the_next_states(signals, unroll_factor):
    s = signals
    _, states = _simulate(s["A"], s["B"], s["C"], s["D"], s["u"], s["zi"])
    Bu = s["B"] * s["u"][..., None]
    h = lpv.state_space_recursion(s["A"], s["zi"], Bu, unroll_factor=unroll_factor)
    assert torch.allclose(h, states)
    h1 = lpv.state_space_recursion(s["A"], s["zi"], Bu, unroll_factor=unroll_factor, out_idx=1)
    assert torch.allclose(h1, states[..., 1])


# 4 leaves a partial last block of the unrolled recursions, 5 none.
@ignore_one
@pytest.mark.parametrize("unroll_factor", [None, 1, 4, 5, N])
def test_lpv_linear_recurrence_returns_the_next_states(signals, unroll_factor):
    s = signals
    a = s["a"][..., 0]
    _, states = _simulate(
        a[..., None, None], _ones(BATCH, N, 1), s["C"][..., :1], s["D"], s["u"], s["zi"][:, :1]
    )
    h = lpv.linear_recurrence(a, s["zi"][:, 0], s["u"], unroll_factor=unroll_factor)
    assert torch.allclose(h, states[..., 0])


@pytest.mark.parametrize(
    ("form", "backend", "reference"),
    [
        ("df2", "ssm", "df2"),
        ("df2", "torchlpc", "df2"),
        ("tdf2", "ssm", "tdf2"),
        ("df1", "ssm", "df1"),
        ("df1", "torchlpc", "df1"),
        ("tdf1", "ssm", "tdf1"),
        ("tdf1", "torchlpc", "tdf1"),
    ],
)
def test_lpv_lfilter(signals, form, backend, reference):
    b, a, u = signals["b"], signals["a"], signals["u"]
    expected = {
        "df2": lambda: _direct_form_two(b, a, u),
        "tdf2": lambda: _transposed_direct_form_two(b, a, u),
        "df1": lambda: _allpole_direct(a, _fir_direct(b, u)),
        "tdf1": lambda: _fir_transposed(b, _allpole_transposed(a, u)),
    }[reference]()
    assert torch.allclose(lpv.lfilter(b, a, u, form=form, backend=backend), expected)


@pytest.mark.parametrize("transpose", [True, False])
def test_lpv_fir_and_allpole(signals, transpose):
    b, a, u = signals["b"], signals["a"], signals["u"]
    fir_ref = _fir_transposed(b, u) if transpose else _fir_direct(b, u)
    allpole_ref = _allpole_transposed(a, u) if transpose else _allpole_direct(a, u)
    assert torch.allclose(lpv.fir(b, u, transpose=transpose), fir_ref)
    assert torch.allclose(lpv.allpole(a, u, transpose=transpose), allpole_ref)


# With constant matrices the step a matrix is taken from cannot matter, but
# the state the recursions return and the final state still can.


@ignore_one
@pytest.mark.parametrize("unroll_factor", [None, 1, 4, 5, N])
def test_lti_state_space(signals, unroll_factor):
    s = signals
    A, B, C, D = s["A"][:, 0], s["B"][:, 0], s["C"][:, 0], s["D"][:, 0]

    def over_time(t):
        return t.unsqueeze(1).expand(-1, N, *t.shape[1:])

    y_ref, states = _simulate(*map(over_time, (A, B, C, D)), s["u"], s["zi"])
    y, zf = lti.state_space(A, s["u"], B=B, C=C, D=D, zi=s["zi"])
    assert torch.allclose(y, y_ref)
    assert torch.allclose(zf, states[:, -1])
    Bu = B.unsqueeze(1) * s["u"][..., None]
    h = lti.state_space_recursion(A, s["zi"], Bu, unroll_factor=unroll_factor)
    assert torch.allclose(h, states)
    h1 = lti.state_space_recursion(A, s["zi"], Bu, unroll_factor=unroll_factor, out_idx=1)
    assert torch.allclose(h1, states[..., 1])


# 4 leaves a partial last block of the unrolled recursions, 5 none.
@ignore_one
@pytest.mark.parametrize("unroll_factor", [None, 1, 4, 5, N])
def test_lti_linear_recurrence_returns_the_next_states(signals, unroll_factor):
    s = signals
    a = s["a"][:, 0, 0]
    _, states = _simulate(
        a[:, None, None, None].expand(-1, N, 1, 1),
        _ones(BATCH, N, 1),
        s["C"][..., :1],
        s["D"],
        s["u"],
        s["zi"][:, :1],
    )
    h = lti.linear_recurrence(a, s["zi"][:, 0], s["u"], unroll_factor=unroll_factor)
    assert torch.allclose(h, states[..., 0])


@ignore_one
def test_lpv_linear_recurrence_broadcasts_a_and_init():
    # a shared over the batch, (N), and a scalar init, as one per signal.
    x = torch.randn(2, 7, dtype=torch.double)
    a = torch.rand(7, dtype=torch.double)
    init = torch.tensor(0.5, dtype=torch.double)
    expected = lpv.linear_recurrence(a.expand_as(x), init.expand(x.size(0)), x)
    for unroll_factor in (None, 1, 3):
        actual = lpv.linear_recurrence(a, init, x, unroll_factor=unroll_factor)
        assert torch.allclose(actual, expected)


def _recursion_inputs(module, dtype):
    """Small integer A, zi and x in module's layout: A shared over the batch."""
    gen = torch.Generator().manual_seed(0)
    A = torch.randint(-1, 2, (M, M) if module is lti else (N, M, M), generator=gen)
    zi = torch.randint(-3, 4, (BATCH, M), generator=gen)
    x = torch.randint(-3, 4, (BATCH, N, M), generator=gen)
    return [t.to(dtype) for t in (A, zi, x)]


@ignore_one
@pytest.mark.parametrize("module", [lti, lpv], ids=["lti", "lpv"])
def test_state_space_recursion_without_a_kernel(module):
    # No kernel takes int32: the default raises and says what to do, and
    # unroll_factor=1 or more runs PyTorch ops: the sequential loop and the
    # block-unrolled recursion give the same states as the kernels' int64.
    with pytest.raises(NotImplementedError, match="pass unroll_factor=1"):
        module.state_space_recursion(*_recursion_inputs(module, torch.int32))
    expected = module.state_space_recursion(*_recursion_inputs(module, torch.int64))
    for unroll_factor in (1, 3, N):
        actual = module.state_space_recursion(
            *_recursion_inputs(module, torch.int32), unroll_factor=unroll_factor
        )
        assert actual.dtype == torch.int32
        assert torch.equal(actual.long(), expected)


@ignore_one
@pytest.mark.parametrize("module", [lti, lpv], ids=["lti", "lpv"])
def test_linear_recurrence_without_a_kernel(module):
    A, zi, x = _recursion_inputs(module, torch.int32)
    a = A[0, 0] if module is lti else A[:, 0, 0]
    with pytest.raises(NotImplementedError, match="pass unroll_factor=1"):
        module.linear_recurrence(a, zi[:, 0], x[..., 0])
    # The sequential loop (unroll_factor=1 or N) takes any dtype. (The scalar
    # block-unrolled recursion doesn't take integers: torch.cumprod and
    # torch.linalg.vecdot don't.)
    expected = module.linear_recurrence(a.long(), zi[:, 0].long(), x[..., 0].long())
    for unroll_factor in (1, N):
        actual = module.linear_recurrence(a, zi[:, 0], x[..., 0], unroll_factor=unroll_factor)
        assert torch.equal(actual.long(), expected)


@pytest.mark.parametrize("module", [lti, lpv], ids=["lti", "lpv"])
def test_state_space_recursion_rejects_bad_shapes(module):
    A, zi, x = _recursion_inputs(module, torch.float64)
    with pytest.raises(ValueError, match="2D or 3D"):
        module.state_space_recursion(A, zi, x[0, 0])
    with pytest.raises(ValueError, match="must be square"):
        module.state_space_recursion(A[..., :-1], zi, x)
    with pytest.raises(ValueError, match="Batch size of zi"):
        module.state_space_recursion(A, zi[:1], x)
    with pytest.raises(ValueError, match="Last dimension of zi"):
        module.state_space_recursion(A, zi[:, :-1], x)
    with pytest.raises(ValueError, match="zi must be 2D"):
        module.state_space_recursion(A, zi[0], x)
    with pytest.raises(ValueError, match="State matrix A must be"):
        module.state_space_recursion(A[..., 0], zi, x)
    with pytest.raises(ValueError, match="Last dimension of A must match"):
        module.state_space_recursion(A, zi, x[..., :-1])
    # A per signal, of the wrong batch size.
    with pytest.raises(ValueError, match="Batch size of A"):
        module.state_space_recursion(A.expand(BATCH + 1, *A.shape), zi, x)
    if module is lpv:
        with pytest.raises(ValueError, match="same time dimension"):
            module.state_space_recursion(A[:-1], zi, x)
    with pytest.raises(ValueError, match="must be None or at least 1"):
        module.state_space_recursion(A, zi, x, unroll_factor=0)


@pytest.mark.parametrize("module", [lti, lpv], ids=["lti", "lpv"])
def test_linear_recurrence_rejects_bad_shapes(module):
    x = torch.randn(BATCH, N, dtype=torch.float64)
    a = torch.rand(BATCH if module is lti else N, dtype=torch.float64)
    init = torch.randn(BATCH, dtype=torch.float64)
    with pytest.raises(ValueError, match="State matrix a must be 1D"):
        module.linear_recurrence(a[:-1] if module is lpv else a.repeat(2), init, x, unroll_factor=3)
    with pytest.raises(ValueError, match="Initial state init must be 1D"):
        module.linear_recurrence(a, init.repeat(2), x, unroll_factor=3)
    with pytest.raises(ValueError, match="must be None or at least 1"):
        module.linear_recurrence(a, init, x, unroll_factor=0)


def _recursion_calls(module):
    A, zi, x = _recursion_inputs(module, torch.float64)
    a = A[0, 0] if module is lti else A[:, 0, 0]
    return {
        "state_space_recursion": lambda **kw: module.state_space_recursion(A, zi, x, **kw),
        "linear_recurrence": lambda **kw: module.linear_recurrence(a, zi[:, 0], x[..., 0], **kw),
    }


@pytest.mark.parametrize("module", [lti, lpv], ids=["lti", "lpv"])
def test_unroll_factor_one_warns_and_runs_the_loop(module):
    calls = _recursion_calls(module)
    for call in calls.values():
        with pytest.warns(FutureWarning, match="since philtorch 0.8") as record:
            actual = call(unroll_factor=1)
        # The warning points at the caller, not at philtorch.
        assert record[0].filename == __file__
        torch.testing.assert_close(actual, call(unroll_factor=N))
        torch.testing.assert_close(actual, call())


@pytest.mark.parametrize("module", [lti, lpv], ids=["lti", "lpv"])
def test_unroll_factor_picks_the_loop_or_the_kernels(module):
    # Which engine runs, by spying on the ones recursion looks up by name.
    calls = _recursion_calls(module)
    engines = {
        "state_space_recursion": ("_loop", "_kernels"),
        "linear_recurrence": ("_scalar_loop", "_scalar_kernels"),
    }
    for name, (loop, kernels) in engines.items():
        for unroll_factor, (expected, other) in [(1, (loop, kernels)), (None, (kernels, loop))]:
            with (
                mock.patch.object(_ssm, loop, wraps=getattr(_ssm, loop)) as loop_spy,
                mock.patch.object(_ssm, kernels, wraps=getattr(_ssm, kernels)) as kernels_spy,
                warnings.catch_warnings(),
            ):
                warnings.simplefilter("ignore", FutureWarning)
                calls[name](unroll_factor=unroll_factor)
            spies = {loop: loop_spy, kernels: kernels_spy}
            spies[expected].assert_called_once()
            spies[other].assert_not_called()


@pytest.mark.parametrize("module", [lti, lpv], ids=["lti", "lpv"])
def test_unroll_factor_warns_only_for_one(module):
    calls = _recursion_calls(module)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for call in calls.values():
            for unroll_factor in (None, 3, N):
                call(unroll_factor=unroll_factor)


@pytest.mark.parametrize("module", [lti, lpv], ids=["lti", "lpv"])
def test_unroll_factor_one_compiles_in_one_graph(module):
    # The warning is skipped while tracing: fullgraph=True would fail on it.
    # Dynamo's tracing is what this checks, so aot_eager, which needs no C++
    # compiler: Windows CI has none for Inductor.
    A, zi, x = _recursion_inputs(module, torch.float64)
    a = A[0, 0] if module is lti else A[:, 0, 0]
    calls = [
        lambda x: module.state_space_recursion(A, zi, x, unroll_factor=1),
        lambda x: module.linear_recurrence(a, zi[:, 0], x[..., 0], unroll_factor=1),
    ]
    for call in calls:
        with pytest.warns(FutureWarning, match="since philtorch 0.8"):
            expected = call(x)
        compiled = torch.compile(call, backend="aot_eager", fullgraph=True)
        torch.testing.assert_close(compiled(x), expected)


@pytest.mark.parametrize("module", [lti, lpv], ids=["lti", "lpv"])
def test_unroll_factor_rejects_zero_and_negative(module):
    A, zi, x = _recursion_inputs(module, torch.float64)
    a = A[0, 0] if module is lti else A[:, 0, 0]
    for unroll_factor in (0, -1):
        with pytest.raises(ValueError, match="must be None or at least 1"):
            module.state_space_recursion(A, zi, x, unroll_factor=unroll_factor)
        with pytest.raises(ValueError, match="must be None or at least 1"):
            module.linear_recurrence(a, zi[:, 0], x[..., 0], unroll_factor=unroll_factor)


@ignore_one
def test_lti_linear_recurrence_takes_a_scalar_init():
    # A 0-D init, shared by the batch, through the unrolled recursion too.
    x = torch.randn(BATCH, N, dtype=torch.float64)
    a = torch.rand(BATCH, dtype=torch.float64)
    init = torch.tensor(0.5, dtype=torch.float64)
    expected = lti.linear_recurrence(a, init.expand(BATCH), x)
    for unroll_factor in (None, 1, 3):
        assert torch.allclose(
            lti.linear_recurrence(a, init, x, unroll_factor=unroll_factor), expected
        )


@pytest.mark.parametrize("module", [lti, lpv], ids=["lti", "lpv"])
def test_recursions_vmap(module):
    # vmap through the public recursions, which run the operator: the same as
    # one call per index.
    batches = [_recursion_inputs(module, torch.float64) for _ in range(3)]
    gen = torch.Generator().manual_seed(1)
    for inputs in batches:
        inputs[2].copy_(torch.randn(inputs[2].shape, generator=gen, dtype=torch.float64))
        inputs[0].mul_(0.3)
    A, zi, x = (torch.stack(t) for t in zip(*batches))
    expected = torch.stack([module.state_space_recursion(*inputs) for inputs in batches])
    torch.testing.assert_close(torch.func.vmap(module.state_space_recursion)(A, zi, x), expected)
    # The scalar coefficients: lti's per index (), lpv's (N).
    a = A[..., 0, 0]
    expected = torch.stack(
        [module.linear_recurrence(a[v], zi[v, :, 0], x[v, ..., 0]) for v in range(3)]
    )
    actual = torch.func.vmap(module.linear_recurrence)(a, zi[..., 0], x[..., 0])
    torch.testing.assert_close(actual, expected)


@ignore_one
@pytest.mark.parametrize("module", [lti, lpv], ids=["lti", "lpv"])
@pytest.mark.parametrize("unroll_factor", [None, 1, 3, 100])
@pytest.mark.parametrize(("batch", "steps"), [(2, 0), (0, 8), (0, 0)])
def test_recursions_take_empty_inputs(module, unroll_factor, batch, steps):
    # No steps or no signals: an empty result, by every algorithm. LPV's A is
    # per signal, which takes its block-unrolled recursion's batch path.
    # Their gradients exist, all zero, as for any other length.
    A = torch.eye(M, dtype=torch.float64) * 0.5
    if module is lpv:
        A = A.expand(batch, steps, M, M).clone()
    zi = torch.ones(batch, M, dtype=torch.float64)
    x = torch.ones(batch, steps, M, dtype=torch.float64)
    inputs = [t.requires_grad_() for t in (A, zi, x)]
    h = module.state_space_recursion(*inputs, unroll_factor=unroll_factor)
    assert h.shape == (batch, steps, M)
    for grad, t in zip(torch.autograd.grad(h.sum(), inputs), inputs, strict=True):
        torch.testing.assert_close(grad, torch.zeros_like(t))
    a = torch.tensor(0.5, dtype=torch.float64)
    if module is lpv:
        a = a.expand(batch, steps).clone()
    inputs = [t.detach().requires_grad_() for t in (a, zi[:, 0], x[..., 0])]
    h = module.linear_recurrence(*inputs, unroll_factor=unroll_factor)
    assert h.shape == (batch, steps)
    for grad, t in zip(torch.autograd.grad(h.sum(), inputs), inputs, strict=True):
        torch.testing.assert_close(grad, torch.zeros_like(t))


@pytest.mark.parametrize(("batch", "steps"), [(2, 0), (0, 8)])
def test_state_spaces_take_empty_inputs(batch, steps):
    # The outputs of no steps or no signals, and the initial state as the final.
    A = torch.tensor([[0.5, 0.2, 0.0], [0.0, -0.3, 0.1], [0.0, 0.0, 0.2]], dtype=torch.float64)
    zi = torch.ones(batch, M, dtype=torch.float64)
    x = torch.ones(batch, steps, dtype=torch.float64)
    results = [
        lti.state_space(A, x, zi=zi),
        lti.diag_state_space(x, A=A, zi=zi),
        lpv.state_space(A.expand(steps, M, M), x, zi=zi),
    ]
    for y, zf in results:
        assert y.shape == (batch, steps, M)
        torch.testing.assert_close(zf, zi.to(zf.dtype))
    assert lti.state_space(A, x).shape == (batch, steps, M)
    assert lti.state_space(A, x, out_idx=1).shape == (batch, steps)
    assert lti.diag_state_space(x, A=A, out_idx=1).shape == (batch, steps)

    b, a = torch.tensor([1.0, 0.5], dtype=torch.float64), torch.tensor([0.3], dtype=torch.float64)
    zi = torch.ones(batch, 1, dtype=torch.float64)
    y, zf = lti.lfilter(b, a, x, zi=zi)
    assert y.shape == (batch, steps)
    torch.testing.assert_close(zf, zi)
    for backend in ("ssm", "torchlpc"):
        y, zf = lpv.lfilter(
            b.expand(batch, steps, 2), a.expand(batch, steps, 1), x, zi=zi, backend=backend
        )
        assert y.shape == (batch, steps)
        torch.testing.assert_close(zf, zi)

    # lpv.fir's outputs: zero gradients, and zi's dtype among those promoted.
    inputs = [t.requires_grad_() for t in (b.expand(batch, steps, 2).clone(), x.clone())]
    (grad_b, grad_x) = torch.autograd.grad(lpv.fir(*inputs).sum(), inputs)
    torch.testing.assert_close(grad_b, torch.zeros_like(inputs[0]))
    torch.testing.assert_close(grad_x, torch.zeros_like(inputs[1]))
    y, _ = lpv.fir(inputs[0].detach(), x, zi=zi.to(torch.complex128))
    assert y.dtype == torch.complex128


@pytest.mark.parametrize("module", [lti, lpv], ids=["lti", "lpv"])
def test_state_space_rejects_D_with_other_outputs(module):
    # Four outputs from D, one from out_idx or a vector C.
    A = torch.eye(M, dtype=torch.float64) * 0.5
    if module is lpv:
        A = A.expand(N, M, M)
    x = torch.ones(BATCH, N, M, dtype=torch.float64)
    D = torch.ones(4, M, dtype=torch.float64)
    message = r"D x, of shape \(2, 10, 4\), has different outputs from C h, .* \(2, 10\)"
    with pytest.raises(ValueError, match=message):
        module.state_space(A, x, D=D, out_idx=0)
    with pytest.raises(ValueError, match=message):
        module.state_space(A, x, C=torch.ones(M, dtype=torch.float64), D=D)
    # One output row is not one scalar output either.
    with pytest.raises(ValueError, match=r"of shape \(2, 10, 1\)"):
        module.state_space(A, x, C=torch.ones(M, dtype=torch.float64), D=D[:1])
