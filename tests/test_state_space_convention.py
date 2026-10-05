"""Every state-space entry point follows the textbook convention.

That is, with x the state and u the input,

    x[n + 1] = A[n] x[n] + B[n] u[n],
    y[n] = C[n] x[n] + D[n] u[n],
    x[0] = zi,

where the matrices at step n take the state from step n to n + 1. Each
function below is written as such a model and checked against one simulator,
so a structure that applies its coefficients a step early or late fails here.
"""

import pytest
import torch

import philtorch.lpv as lpv
import philtorch.lti as lti
from philtorch.mat import companion

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


@pytest.mark.parametrize("unroll_factor", [1, 4, N])
def test_lpv_state_space_recursion_returns_the_next_states(signals, unroll_factor):
    s = signals
    _, states = _simulate(s["A"], s["B"], s["C"], s["D"], s["u"], s["zi"])
    h = lpv.state_space_recursion(
        s["A"], s["zi"], s["B"] * s["u"][..., None], unroll_factor=unroll_factor
    )
    assert torch.allclose(h, states)


@pytest.mark.parametrize("unroll_factor", [1, 4, N])
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


def test_lti_state_space(signals):
    s = signals
    A, B, C, D = s["A"][:, 0], s["B"][:, 0], s["C"][:, 0], s["D"][:, 0]

    def over_time(t):
        return t.unsqueeze(1).expand(-1, N, *t.shape[1:])

    y_ref, states = _simulate(*map(over_time, (A, B, C, D)), s["u"], s["zi"])
    y, zf = lti.state_space(A, s["u"], B=B, C=C, D=D, zi=s["zi"])
    assert torch.allclose(y, y_ref)
    assert torch.allclose(zf, states[:, -1])
    h = lti.state_space_recursion(A, s["zi"], B.unsqueeze(1) * s["u"][..., None])
    assert torch.allclose(h, states)


def test_lti_linear_recurrence_returns_the_next_states(signals):
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
    assert torch.allclose(lti.linear_recurrence(a, s["zi"][:, 0], s["u"]), states[..., 0])
