import itertools
import random
import warnings

import pytest
import torch

import philtorch.lti.delay as delay_module
from philtorch.lti import delay_state_space, state_space
from philtorch.lti.ssm import _ssm_B, _ssm_C_D

BATCH = 3
LINES = 4
FEATURES = 2
OUTPUTS = 5
DELAYS = (3, 7, 4, 5)
SAMPLES = 11


def _as_state_space(A, x, delays, B, C, D, zi, out_idx):
    """Evaluate the delay network as an equivalent sum(delays)-state state_space.

    Each delay line is a shift register whose head is read by C and A, and
    whose tail receives A @ r + B @ x.
    """
    M, S = len(delays), sum(delays)
    lengths = torch.tensor(delays)
    heads = torch.cumsum(lengths, 0) - lengths
    tails = heads + lengths - 1

    big_A = A.new_zeros(*A.shape[:-2], S, S)
    for head, delay in zip(heads.tolist(), delays):
        shift = torch.arange(head, head + delay - 1)
        big_A[..., shift, shift + 1] = 1
    big_A[..., tails[:, None], heads[None, :]] = A

    eye = torch.eye(M, dtype=x.dtype)
    if B is None:
        B = eye[:, 0] if x.dim() == 2 else eye
    B_dim = B.dim() - 1 if x.dim() == 2 else B.dim() - 2
    big_B = B.new_zeros(*B.shape[:B_dim], S, *B.shape[B_dim + 1 :])
    big_B.index_copy_(B_dim, tails, B)

    if out_idx is not None:
        C = eye[out_idx]
    elif C is None:
        C = eye
    big_C = C.new_zeros(*C.shape[:-1], S)
    big_C.index_copy_(C.dim() - 1, heads, C)

    if zi is None:
        big_zi = x.new_zeros(x.size(0), S)
    else:
        big_zi = torch.cat([state.expand(x.size(0), -1) for state in zi], dim=-1)

    y, zf = state_space(big_A, x, B=big_B, C=big_C, D=D, zi=big_zi)
    return y, zf.split(delays, dim=-1)


def _layout(name, dtype):
    def randn(*shape):
        return torch.randn(*shape, dtype=dtype)

    siso, mimo = randn(BATCH, SAMPLES), randn(BATCH, SAMPLES, FEATURES)
    M, b, F, P = LINES, BATCH, FEATURES, OUTPUTS
    layouts = {
        "B(M) C(M) D()": dict(x=siso, B=randn(M), C=randn(M), D=randn(())),
        "B(b,M) C(b,M) D(b)": dict(x=siso, B=randn(b, M), C=randn(b, M), D=randn(b)),
        "B(M) C(M) D(1)": dict(x=siso, B=randn(M), C=randn(M), D=randn(1)),
        "B(M) C(P,M) D(P)": dict(x=siso, B=randn(M), C=randn(P, M), D=randn(P)),
        "B(b,M) C(b,P,M) D(b,P)": dict(x=siso, B=randn(b, M), C=randn(b, P, M), D=randn(b, P)),
        "B(M) out_idx D()": dict(x=siso, B=randn(M), D=randn(()), out_idx=2),
        "defaults 2D": dict(x=siso),
        "B(M,F) C(M) D(F)": dict(x=mimo, B=randn(M, F), C=randn(M), D=randn(F)),
        "B(b,M,F) C(b,M) D(b,F)": dict(x=mimo, B=randn(b, M, F), C=randn(b, M), D=randn(b, F)),
        "B(M,F) C(P,M) D(P,F)": dict(x=mimo, B=randn(M, F), C=randn(P, M), D=randn(P, F)),
        "B(b,M,F) C(b,P,M) D(b,P,F)": dict(
            x=mimo, B=randn(b, M, F), C=randn(b, P, M), D=randn(b, P, F)
        ),
        "B(M,F) C(F,M) D()": dict(x=mimo, B=randn(M, F), C=randn(F, M), D=randn(())),
        "B(M,F) out_idx D(F)": dict(x=mimo, B=randn(M, F), D=randn(F), out_idx=0),
        "defaults 3D": dict(x=randn(BATCH, SAMPLES, LINES)),
    }
    return {"B": None, "C": None, "D": None, "out_idx": None, **layouts[name]}


def _initial_states(mode, dtype):
    if mode is None:
        return None
    if mode == "1d":
        return tuple(torch.randn(delay, dtype=dtype) for delay in DELAYS)
    return tuple(torch.randn(BATCH, delay, dtype=dtype) for delay in DELAYS)


def _assert_states_close(actual, expected):
    assert isinstance(actual, tuple) and len(actual) == len(expected)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e)


LAYOUTS = [
    "B(M) C(M) D()",
    "B(b,M) C(b,M) D(b)",
    "B(M) C(M) D(1)",
    "B(M) C(P,M) D(P)",
    "B(b,M) C(b,P,M) D(b,P)",
    "B(M) out_idx D()",
    "defaults 2D",
    "B(M,F) C(M) D(F)",
    "B(b,M,F) C(b,M) D(b,F)",
    "B(M,F) C(P,M) D(P,F)",
    "B(b,M,F) C(b,P,M) D(b,P,F)",
    "B(M,F) C(F,M) D()",
    "B(M,F) out_idx D(F)",
    "defaults 3D",
]


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("dtype", [torch.double, torch.cdouble])
@pytest.mark.parametrize("batched_A", [False, True])
@pytest.mark.parametrize("zi_mode", [None, "1d", "2d"])
@pytest.mark.parametrize("block_size", [None, 1, 2])
def test_delay_state_space_matches_expanded_state_space(
    layout, dtype, batched_A, zi_mode, block_size
):
    args = _layout(layout, dtype)
    A_shape = (BATCH, LINES, LINES) if batched_A else (LINES, LINES)
    A = 0.4 * torch.randn(*A_shape, dtype=dtype)
    zi = _initial_states(zi_mode, dtype)

    result = delay_state_space(A, delays=DELAYS, zi=zi, block_size=block_size, **args)
    expected_y, expected_zf = _as_state_space(A, delays=DELAYS, zi=zi, **args)

    if zi is None:
        assert isinstance(result, torch.Tensor)
        torch.testing.assert_close(result, expected_y)
    else:
        y, zf = result
        torch.testing.assert_close(y, expected_y)
        _assert_states_close(zf, expected_zf)


@pytest.mark.parametrize("chunk", [1, 2, 4, 5, 64])
@pytest.mark.parametrize("out_idx", [None, 1])
def test_delay_state_space_output_chunking(monkeypatch, chunk, out_idx):
    delays = (4, 6)
    x = torch.randn(2, 37, dtype=torch.double)
    A = 0.5 * torch.randn(2, 2, dtype=torch.double)
    C = None if out_idx is not None else torch.randn(3, 2, dtype=torch.double)
    zi = (torch.randn(4, dtype=torch.double), torch.randn(2, 6, dtype=torch.double))
    expected_y, expected_zf = _as_state_space(A, x, delays, None, C, None, zi, out_idx)

    monkeypatch.setattr(delay_module, "_OUTPUT_CHUNK", chunk)
    y, zf = delay_state_space(A, x, delays, C=C, zi=zi, out_idx=out_idx)

    torch.testing.assert_close(y, expected_y)
    _assert_states_close(zf, expected_zf)


@pytest.mark.parametrize("samples", [1, 3, 6, 7, 20])
def test_delay_state_space_short_and_long_signals(samples):
    delays = (3, 7)
    x = torch.randn(BATCH, samples, dtype=torch.double)
    A = 0.5 * torch.randn(2, 2, dtype=torch.double)
    zi = (torch.randn(BATCH, 3, dtype=torch.double), torch.randn(7, dtype=torch.double))

    y, zf = delay_state_space(A, x, delays, zi=zi)
    expected_y, expected_zf = _as_state_space(A, x, delays, None, None, None, zi, None)

    torch.testing.assert_close(y, expected_y)
    _assert_states_close(zf, expected_zf)


def test_delay_state_space_batched_matrices_are_not_conjugated_like_state_space():
    x = torch.randn(BATCH, SAMPLES, FEATURES, dtype=torch.cdouble)
    A = 0.4 * torch.randn(LINES, LINES, dtype=torch.cdouble)
    B = torch.randn(LINES, FEATURES, dtype=torch.cdouble)
    C = torch.randn(LINES, dtype=torch.cdouble)
    D = torch.randn(FEATURES, dtype=torch.cdouble)

    batched = delay_state_space(
        A,
        x,
        DELAYS,
        B=B.expand(BATCH, -1, -1),
        C=C.expand(BATCH, -1),
        D=D.expand(BATCH, -1),
    )
    unbatched = delay_state_space(A, x, DELAYS, B=B, C=C, D=D)

    torch.testing.assert_close(batched, unbatched)


def test_delay_state_space_single_line_is_a_feedback_comb():
    # y[n] = s[n - m], s[n] = g * s[n - m] + x[n]: impulse response g^(k-1) at n = k*m.
    delay, gain = 4, 0.5
    x = torch.zeros(1, 17, dtype=torch.double)
    x[0, 0] = 1.0
    A = torch.tensor([[gain]], dtype=torch.double)

    y = delay_state_space(A, x, (delay,), C=torch.ones(1, dtype=torch.double))

    expected = torch.zeros(17, dtype=torch.double)
    expected[delay::delay] = gain ** torch.arange(4, dtype=torch.double)
    torch.testing.assert_close(y[0], expected)


def test_delay_state_space_default_input_and_initial_state():
    x = torch.tensor([[1.0, 2.0, 3.0]])
    A = torch.zeros(2, 2)

    y, zf = delay_state_space(A, x, (1, 2), zi=(torch.zeros(1), torch.zeros(2)))

    assert torch.equal(y, torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]]))
    assert torch.equal(zf[0], torch.tensor([[3.0]]))
    assert torch.equal(zf[1], torch.zeros(1, 2))
    assert y.dtype == x.dtype
    assert y.device == x.device


@pytest.mark.parametrize(
    "delays",
    [
        pytest.param(lambda np: [np.int64(3), np.int32(7)], id="numpy-scalars"),
        pytest.param(lambda np: np.array([3, 7]), id="numpy-array"),
        pytest.param(lambda np: torch.tensor([3, 7]), id="tensor"),
        pytest.param(lambda np: (torch.tensor(3), 7), id="0d-tensor-and-int"),
        pytest.param(
            lambda np: (torch.tensor([3], dtype=torch.int32), np.uint8(7)),
            id="one-element-tensor-and-uint8",
        ),
    ],
)
@pytest.mark.parametrize(
    "block_size",
    [
        pytest.param(lambda np: None, id="default-block"),
        pytest.param(lambda np: np.int64(2), id="numpy-block"),
        pytest.param(lambda np: torch.tensor(2), id="tensor-block"),
    ],
)
def test_delay_state_space_accepts_integer_like_delays(delays, block_size):
    np = pytest.importorskip("numpy")
    x = torch.randn(2, 9, dtype=torch.double)
    A = 0.5 * torch.randn(2, 2, dtype=torch.double)
    zi = (torch.randn(3, dtype=torch.double), torch.randn(7, dtype=torch.double))
    block = block_size(np)

    y, zf = delay_state_space(A, x, delays(np), zi=zi, block_size=block)
    expected_y, expected_zf = delay_state_space(
        A, x, (3, 7), zi=zi, block_size=None if block is None else 2
    )

    torch.testing.assert_close(y, expected_y)
    _assert_states_close(zf, expected_zf)


@pytest.mark.parametrize(
    "delay",
    [
        pytest.param(lambda np: 5, id="int"),
        pytest.param(lambda np: torch.tensor(5), id="0d-tensor"),
        pytest.param(lambda np: torch.tensor(5, dtype=torch.int32), id="0d-int32"),
        pytest.param(lambda np: np.int64(5), id="numpy-scalar"),
        pytest.param(lambda np: np.array(5), id="0d-numpy-array"),
    ],
)
def test_delay_state_space_accepts_scalar_delay(delay):
    np = pytest.importorskip("numpy")
    x = torch.randn(2, 13, dtype=torch.double)
    A = torch.tensor([[0.5]], dtype=torch.double)
    zi = (torch.randn(5, dtype=torch.double),)

    y, zf = delay_state_space(A, x, delay(np), zi=zi)
    expected_y, expected_zf = delay_state_space(A, x, (5,), zi=zi)

    torch.testing.assert_close(y, expected_y)
    _assert_states_close(zf, expected_zf)


@pytest.mark.parametrize(
    "delay",
    [
        pytest.param(lambda np: True, id="bool"),
        pytest.param(lambda np: 0, id="zero"),
        pytest.param(lambda np: 5.0, id="float"),
        pytest.param(lambda np: None, id="none"),
        pytest.param(lambda np: torch.tensor(True), id="0d-bool-tensor"),
        pytest.param(lambda np: torch.tensor(5.0), id="0d-float-tensor"),
        pytest.param(lambda np: np.array(5.0), id="0d-float-array"),
    ],
)
def test_delay_state_space_rejects_invalid_scalar_delay(delay):
    np = pytest.importorskip("numpy")
    with pytest.raises(ValueError, match="positive integer"):
        delay_state_space(torch.zeros(1, 1), torch.zeros(1, 4), delay(np))


def test_delay_state_space_is_functional():
    delays = (2, 5, 3)
    x = torch.randn(2, 9, dtype=torch.double)
    A = 0.3 * torch.randn(3, 3, dtype=torch.double)
    B, C, D = (torch.randn(s, dtype=torch.double) for s in ((3,), (3,), ()))
    zi = tuple(torch.randn(2, delay, dtype=torch.double) for delay in delays)
    originals = [t.clone() for t in (x, A, B, C, D, *zi)]

    _, zf = delay_state_space(A, x, delays, B=B, C=C, D=D, zi=zi, block_size=2)

    for value, original in zip((x, A, B, C, D, *zi), originals):
        assert torch.equal(value, original)
    assert all(final.data_ptr() != initial.data_ptr() for final, initial in zip(zf, zi))


def test_delay_state_space_reuses_returned_states():
    delays = (2, 4)
    x = torch.randn(2, 7, dtype=torch.double)
    A = 0.4 * torch.randn(2, 2, dtype=torch.double)
    B, C, D = (torch.randn(s, dtype=torch.double) for s in ((2,), (2,), ()))
    zi = (torch.randn(2, 2, dtype=torch.double), torch.randn(2, 4, dtype=torch.double))

    expected_y, expected_zf = delay_state_space(A, x, delays, B=B, C=C, D=D, zi=zi)
    first_y, first_zf = delay_state_space(A, x[:, :3], delays, B=B, C=C, D=D, zi=zi)
    second_y, actual_zf = delay_state_space(A, x[:, 3:], delays, B=B, C=C, D=D, zi=first_zf)

    torch.testing.assert_close(torch.cat([first_y, second_y], dim=1), expected_y)
    _assert_states_close(actual_zf, expected_zf)


@pytest.mark.parametrize("out_idx", [None, 1])
@pytest.mark.parametrize("with_zi", [False, True])
def test_delay_state_space_zero_length(with_zi, out_idx):
    delays = (2, 3)
    x = torch.empty(2, 0)
    A = torch.eye(2)
    C = torch.ones(2) if out_idx is None else None
    zi = (
        (torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([5.0, 6.0, 7.0])) if with_zi else None
    )

    result = delay_state_space(A, x, delays, C=C, D=torch.tensor(0.5), zi=zi, out_idx=out_idx)

    y = result[0] if with_zi else result
    assert y.shape == (2, 0)
    assert y.dtype == x.dtype
    if with_zi:
        _assert_states_close(result[1], (zi[0], zi[1].expand(2, -1)))


@pytest.mark.parametrize(
    ("delays", "block_size", "message"),
    [
        ((), None, "at least one"),
        ((2, 0), None, "positive integer"),
        ((2, -1), None, "positive integer"),
        ((2, 3.0), None, "positive integer"),
        ((2, True), None, "positive integer"),
        ((2, torch.tensor(True)), None, "positive integer"),
        ((2, torch.tensor(3.0)), None, "positive integer"),
        ((2, torch.tensor([3, 4])), None, "positive integer"),
        ((2, torch.tensor(-3)), None, "positive integer"),
        ((2, 3), 0, "1 <= block_size"),
        ((2, 3), 3, "1 <= block_size"),
        ((2, 3), torch.tensor(-1), "1 <= block_size"),
        ((2, 3), 2.0, "must be an integer"),
        ((2, 3), True, "must be an integer"),
        ((2, 3), torch.tensor(2.0), "must be an integer"),
    ],
)
def test_delay_state_space_validates_delays_and_block_size(delays, block_size, message):
    M = max(len(delays), 1)
    with pytest.raises(ValueError, match=message):
        delay_state_space(torch.zeros(M, M), torch.zeros(1, 4), delays, block_size=block_size)


@pytest.mark.parametrize(
    ("delays", "message"),
    [
        (lambda np: (2, np.float64(3.0)), "positive integer"),
        (lambda np: (2, np.bool_(True)), "positive integer"),
        (lambda np: (2, np.int64(0)), "positive integer"),
        (lambda np: np.array([2.0, 3.0]), "positive integer"),
    ],
)
def test_delay_state_space_rejects_non_integer_numpy_delays(delays, message):
    np = pytest.importorskip("numpy")
    with pytest.raises(ValueError, match=message):
        delay_state_space(torch.zeros(2, 2), torch.zeros(1, 4), delays(np))


@pytest.mark.parametrize(
    ("A", "x", "message"),
    [
        (torch.zeros(2, 2), torch.zeros(4), "Input signal must be 2D or 3D"),
        (torch.zeros(2), torch.zeros(2, 4), "State matrix A must be 2D or 3D"),
        (torch.zeros(2, 3), torch.zeros(2, 4), "square"),
        (torch.zeros(3, 3), torch.zeros(2, 4), "number of delays"),
        (torch.zeros(3, 2, 2), torch.zeros(2, 4), "Batch size of A"),
    ],
)
def test_delay_state_space_validates_signal_and_feedback_matrix(A, x, message):
    with pytest.raises(ValueError, match=message):
        delay_state_space(A, x, (2, 3))


def test_delay_state_space_rejects_C_with_out_idx():
    with pytest.raises(ValueError, match="C and out_idx"):
        delay_state_space(torch.zeros(2, 2), torch.zeros(2, 4), (2, 3), C=torch.ones(2), out_idx=0)


@pytest.mark.parametrize(
    ("x", "B", "error", "message"),
    [
        (torch.zeros(2, 4, 3), None, ValueError, "number of delays when B is None"),
        (torch.zeros(2, 4, 1), torch.zeros(2), ValueError, "Input matrix B"),
        (torch.zeros(2, 4, 1), torch.zeros(2, 2), ValueError, "Input matrix B"),
        (torch.zeros(2, 4), torch.zeros(4), ValueError, "Input matrix B"),
        (torch.zeros(2, 4, 3), torch.zeros(2, 3, 3), ValueError, "Input matrix B"),
    ],
)
def test_delay_state_space_validates_input_matrix(x, B, error, message):
    with pytest.raises(error, match=message):
        delay_state_space(torch.zeros(2, 2), x, (2, 3), B=B)


def test_delay_state_space_validates_output_matrices():
    A, x = torch.zeros(2, 2), torch.zeros(2, 4)
    with pytest.raises(ValueError, match="Output matrix C"):
        delay_state_space(A, x, (2, 3), C=torch.zeros(4))
    with pytest.raises(ValueError, match="Input matrix D .* for 2D"):
        delay_state_space(A, x, (2, 3), D=torch.zeros(2, 2, 2, 2))
    with pytest.raises(ValueError, match="Input matrix D .* for 3D"):
        delay_state_space(A, torch.zeros(2, 4, 2), (2, 3), D=torch.zeros(3))


@pytest.mark.parametrize(
    ("zi", "error", "message"),
    [
        (torch.zeros(2, 2), ValueError, "one state"),
        ((torch.zeros(2, 2),), ValueError, "one state"),
        ((torch.zeros(2, 2), [0.0, 0.0, 0.0]), TypeError, r"zi\[1\] must be a Tensor"),
        ((torch.zeros(2, 2), torch.zeros(1, 2, 3)), ValueError, "1D or 2D"),
        ((torch.zeros(2, 2), torch.zeros(2, 4)), ValueError, "match delay"),
        ((torch.zeros(2, 2), torch.zeros(3, 3)), ValueError, "Batch size of zi"),
    ],
)
def test_delay_state_space_validates_initial_states(zi, error, message):
    with pytest.raises(error, match=message):
        delay_state_space(torch.zeros(2, 2), torch.zeros(2, 4), (2, 3), zi=zi)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"D": torch.ones(1, dtype=torch.float32)}, id="matrix"),
        pytest.param(
            {"zi": (torch.zeros(2, dtype=torch.float64), torch.zeros(3, dtype=torch.float32))},
            id="zi",
        ),
    ],
)
def test_delay_state_space_rejects_other_dtypes_than_x(kwargs):
    # The matmul would raise a RuntimeError, or the queues be promoted.
    kwargs = {"A": torch.zeros(2, 2, dtype=torch.float64), **kwargs}
    with pytest.raises(ValueError, match="dtype of x"):
        delay_state_space(x=torch.zeros(2, 4, dtype=torch.float64), delays=(2, 3), **kwargs)


# --- absorption filters ---------------------------------------------------

POSITIONS = [False, True]
POSITION_IDS = ["after-delay", "after-matrix"]


def _filter_coefficients(coefficients, orders, dtype):
    """Random per-line filters, as (b, a) in lfilter's convention."""
    kb, ka = orders
    b_batch, a_batch = {
        "shared": ((), ()),
        "batched": ((BATCH,), (BATCH,)),
        "mixed": ((BATCH,), ()),
    }[coefficients]
    b = 0.5 * torch.randn(*b_batch, LINES, kb + 1, dtype=dtype)
    a = 0.3 * torch.randn(*a_batch, LINES, ka, dtype=dtype)
    return b, a


def _absorption_state(mode, orders, dtype):
    if mode is None:
        return None
    shape = (LINES, max(orders[0], orders[1]))
    shape = shape if mode == "shared" else (BATCH, *shape)
    return torch.randn(*shape, dtype=dtype)


def _reference_absorption(A, x, delays, B, C, D, out_idx, zi, b, a, absorption_zi, after_matrix):
    """Run the network one sample at a time, with explicit queues and filters.

    The filters are written from y[n] = sum_k b_k x[n - k] - sum_k a_k y[n - k]
    in transposed direct form II, so z_k[n] = b_{k+1} x[n] - a_{k+1} y[n] +
    z_{k+1}[n - 1], as SciPy's ``zi`` holds them. The layouts of B, C and D are
    those of the expanded state-space test above, and are applied here by the
    same helpers as in ``delay_state_space``.
    """
    batch_size, samples = x.shape[:2]
    M = len(delays)
    Bx = _ssm_B(B, x, batch_size, M)
    if Bx.dim() == 2:
        Bx = torch.cat([Bx.unsqueeze(-1), Bx.new_zeros(batch_size, samples, M - 1)], -1)

    b = b.expand(batch_size, M, -1)
    a = a.expand(batch_size, M, -1)
    K = max(b.size(-1) - 1, a.size(-1))
    b = torch.cat([b, b.new_zeros(batch_size, M, K + 1 - b.size(-1))], -1)
    a = torch.cat([a, a.new_zeros(batch_size, M, K - a.size(-1))], -1)
    if absorption_zi is None:
        z = x.new_zeros(batch_size, M, K)
    else:
        z = absorption_zi.expand(batch_size, M, K)
    z = list(z.unbind(-1))  # K tensors of shape (B, M)

    def apply_filters(u):
        nonlocal z
        out = b[..., 0] * u + (z[0] if K else 0)
        nxt = [b[..., k + 1] * u - a[..., k] * out for k in range(K)]
        z = [nxt[k] + (z[k + 1] if k + 1 < K else 0) for k in range(K)]
        return out

    if zi is None:
        queues = [[x.new_zeros(batch_size)] * delay for delay in delays]
    else:
        queues = [list(state.expand(batch_size, -1).unbind(-1)) for state in zi]

    rs = []
    for n in range(samples):
        q = torch.stack([queue.pop(0) for queue in queues], dim=-1)  # (B, M)
        if after_matrix:
            r = q
            s = apply_filters((A @ r.unsqueeze(-1)).squeeze(-1)) + Bx[:, n]
        else:
            r = apply_filters(q)
            s = (A @ r.unsqueeze(-1)).squeeze(-1) + Bx[:, n]
        for queue, value in zip(queues, s.unbind(-1)):
            queue.append(value)
        rs.append(r)

    h = torch.stack(rs, dim=1) if rs else x.new_empty(batch_size, 0, M)
    if out_idx is not None:
        h = h[..., out_idx]
    y = _ssm_C_D(h, x, C, D, batch_size, M)
    zf = tuple(torch.stack(queue, dim=-1) for queue in queues)
    absorption_zf = torch.stack(z, dim=-1) if K else x.new_zeros(batch_size, M, 0)
    return y, zf, absorption_zf


_ABSORPTION_LAYOUTS = [
    "defaults 2D",
    "B(M) C(P,M) D(P)",
    "B(M) out_idx D()",
    "B(b,M,F) C(b,P,M) D(b,P,F)",
    "defaults 3D",
]
_ABSORPTION_ORDERS = [(0, 0), (1, 1), (2, 1), (0, 2)]


def _case_id(layout, after_matrix, dtype, orders, coefficients, batched_A, zi_mode, az_mode, block):
    return "-".join(
        [
            layout.replace(" ", ""),
            POSITION_IDS[after_matrix],
            str(dtype).removeprefix("torch."),
            f"orders{orders[0]}{orders[1]}",
            coefficients,
            "A3d" if batched_A else "A2d",
            f"zi-{zi_mode or 'none'}",
            f"az-{az_mode or 'none'}",
            "block-default" if block is None else f"block-{block}",
        ]
    )


_ABSORPTION_OPTIONS = list(
    itertools.product(
        _ABSORPTION_LAYOUTS,
        POSITIONS,
        [torch.double, torch.cdouble],
        _ABSORPTION_ORDERS,
        ["shared", "batched", "mixed"],
        [False, True],
        [None, "1d", "2d"],
        [None, "shared", "batched"],
        [None, 1, 2],
    )
)


def _absorption_cases():
    # The full product is 12960 cases. A seeded sample of 200 keeps every
    # value of every option.
    return [
        pytest.param(*case, id=_case_id(*case))
        for case in random.Random(0).sample(_ABSORPTION_OPTIONS, 200)
    ]


@pytest.mark.parametrize(
    "layout, after_matrix, dtype, orders, coefficients, batched_A, zi_mode, az_mode, block_size",
    _absorption_cases(),
)
def test_delay_state_space_absorption_matches_reference(
    layout, after_matrix, dtype, orders, coefficients, batched_A, zi_mode, az_mode, block_size
):
    args = _layout(layout, dtype)
    A_shape = (BATCH, LINES, LINES) if batched_A else (LINES, LINES)
    A = 0.4 * torch.randn(*A_shape, dtype=dtype)
    b, a = _filter_coefficients(coefficients, orders, dtype)
    zi = _initial_states(zi_mode, dtype)
    absorption_zi = _absorption_state(az_mode, orders, dtype)

    result = delay_state_space(
        A,
        delays=DELAYS,
        zi=zi,
        block_size=block_size,
        absorption=(b, a),
        absorption_zi=absorption_zi,
        absorption_after_matrix=after_matrix,
        **args,
    )
    expected_y, expected_zf, expected_azf = _reference_absorption(
        A, args["x"], DELAYS, args["B"], args["C"], args["D"], args["out_idx"],
        zi, b, a, absorption_zi, after_matrix,
    )  # fmt: skip

    if zi is None and absorption_zi is None:
        assert isinstance(result, torch.Tensor)
        y = result
    else:
        assert isinstance(result, tuple) and len(result) == 3
        y, zf, azf = result
        _assert_states_close(zf, expected_zf)
        assert azf.shape == (BATCH, LINES, max(orders))
        torch.testing.assert_close(azf, expected_azf)
    torch.testing.assert_close(y, expected_y)


@pytest.mark.parametrize("after_matrix", POSITIONS, ids=POSITION_IDS)
@pytest.mark.parametrize("batched_zi", [False, True])
def test_delay_state_space_identity_absorption_is_a_no_op(after_matrix, batched_zi):
    x = torch.randn(BATCH, SAMPLES, dtype=torch.double)
    A = 0.4 * torch.randn(LINES, LINES, dtype=torch.double)
    B, C, D = (torch.randn(shape, dtype=torch.double) for shape in ((LINES,), (LINES,), ()))
    zi = _initial_states("2d" if batched_zi else "1d", torch.double)
    expected_y, expected_zf = delay_state_space(A, x, DELAYS, B=B, C=C, D=D, zi=zi)

    ones, zeros = (
        torch.ones(LINES, 1, dtype=torch.double),
        torch.zeros(LINES, 1, dtype=torch.double),
    )
    identities = {
        "pure gain": (ones, torch.empty(LINES, 0, dtype=torch.double)),
        "padded": (torch.cat([ones, zeros], dim=1), zeros),
    }
    for b, a in identities.values():
        y, zf, azf = delay_state_space(
            A,
            x,
            DELAYS,
            B=B,
            C=C,
            D=D,
            zi=zi,
            absorption=(b, a),
            absorption_after_matrix=after_matrix,
        )
        torch.testing.assert_close(y, expected_y)
        _assert_states_close(zf, expected_zf)
        assert azf.shape == (BATCH, LINES, a.size(-1))
        assert torch.count_nonzero(azf) == 0


@pytest.mark.parametrize("after_matrix", POSITIONS, ids=POSITION_IDS)
@pytest.mark.parametrize("delay", [1, 4])
def test_delay_state_space_absorption_single_line_matches_scipy(after_matrix, delay):
    np = pytest.importorskip("numpy")
    signal = pytest.importorskip("scipy.signal")
    gain, b, a = 0.5, np.array([0.8, 0.3]), np.array([1.0, -0.4])
    x = torch.randn(2, 40, dtype=torch.double)
    x[0, 1:] = 0.0  # an impulse for the first row

    # Polynomials in z^-1, lowest power first. H = b / a and the delay is z^-m.
    shifted_b = np.concatenate([np.zeros(delay), b])
    loop = np.zeros(max(len(a), len(shifted_b)))
    loop[: len(a)] += a
    loop[: len(shifted_b)] -= gain * shifted_b
    if after_matrix:  # y = z^-m / (1 - g z^-m H)
        numerator = np.concatenate([np.zeros(delay), a])
    else:  # y = z^-m H / (1 - g z^-m H)
        numerator = shifted_b
    expected = signal.lfilter(numerator, loop, x.numpy(), axis=-1)

    y = delay_state_space(
        torch.tensor([[gain]], dtype=torch.double),
        x,
        (delay,),
        C=torch.ones(1, dtype=torch.double),
        absorption=(
            torch.from_numpy(b)[None],
            torch.from_numpy(a[1:])[None],
        ),
        absorption_after_matrix=after_matrix,
    )

    torch.testing.assert_close(y, torch.from_numpy(expected))


@pytest.mark.parametrize("after_matrix", POSITIONS, ids=POSITION_IDS)
@pytest.mark.parametrize("split", [2, 5, 9, 13])
@pytest.mark.parametrize("dtype", [torch.double, torch.cdouble])
def test_delay_state_space_absorption_streaming(after_matrix, split, dtype):
    # The splits fall below the shortest delay (3), between the delays, and
    # beyond the longest (7).
    samples, orders = 17, (2, 2)
    x = torch.randn(BATCH, samples, dtype=dtype)
    A = 0.4 * torch.randn(BATCH, LINES, LINES, dtype=dtype)
    B, C, D = (
        torch.randn(shape, dtype=dtype) for shape in ((LINES,), (OUTPUTS, LINES), (OUTPUTS,))
    )
    absorption = _filter_coefficients("batched", orders, dtype)
    zi = _initial_states("2d", dtype)
    absorption_zi = _absorption_state("batched", orders, dtype)
    kwargs = dict(B=B, C=C, D=D, absorption=absorption, absorption_after_matrix=after_matrix)

    expected_y, expected_zf, expected_azf = delay_state_space(
        A, x, DELAYS, zi=zi, absorption_zi=absorption_zi, **kwargs
    )
    first_y, zf, azf = delay_state_space(
        A, x[:, :split], DELAYS, zi=zi, absorption_zi=absorption_zi, **kwargs
    )
    second_y, final_zf, final_azf = delay_state_space(
        A, x[:, split:], DELAYS, zi=zf, absorption_zi=azf, **kwargs
    )

    torch.testing.assert_close(torch.cat([first_y, second_y], dim=1), expected_y)
    _assert_states_close(final_zf, expected_zf)
    torch.testing.assert_close(final_azf, expected_azf)


@pytest.mark.parametrize("after_matrix", POSITIONS, ids=POSITION_IDS)
def test_delay_state_space_absorption_return_contract(after_matrix):
    dtype, orders = torch.double, (1, 2)
    x = torch.randn(BATCH, SAMPLES, dtype=dtype)
    A = 0.4 * torch.randn(LINES, LINES, dtype=dtype)
    absorption = _filter_coefficients("shared", orders, dtype)
    zi = _initial_states("2d", dtype)
    absorption_zi = _absorption_state("shared", orders, dtype)
    kwargs = dict(absorption=absorption, absorption_after_matrix=after_matrix)
    zero_zi = tuple(torch.zeros(BATCH, delay, dtype=dtype) for delay in DELAYS)
    zero_az = torch.zeros(LINES, max(orders), dtype=dtype)

    plain = delay_state_space(A, x, DELAYS, **kwargs)
    assert isinstance(plain, torch.Tensor)

    # Only the filters' state: the queues start empty, and come back.
    y, zf, azf = delay_state_space(A, x, DELAYS, absorption_zi=absorption_zi, **kwargs)
    expected_y, expected_zf, expected_azf = delay_state_space(
        A, x, DELAYS, zi=zero_zi, absorption_zi=absorption_zi, **kwargs
    )
    torch.testing.assert_close(y, expected_y)
    _assert_states_close(zf, expected_zf)
    torch.testing.assert_close(azf, expected_azf)

    # Only the queues: the filters start at zero, and their state comes back.
    y, zf, azf = delay_state_space(A, x, DELAYS, zi=zi, **kwargs)
    expected_y, expected_zf, expected_azf = delay_state_space(
        A, x, DELAYS, zi=zi, absorption_zi=zero_az, **kwargs
    )
    torch.testing.assert_close(y, expected_y)
    _assert_states_close(zf, expected_zf)
    torch.testing.assert_close(azf, expected_azf)
    assert azf.shape == (BATCH, LINES, max(orders))

    # Zero states are the default.
    zero_y, _, _ = delay_state_space(A, x, DELAYS, zi=zero_zi, absorption_zi=zero_az, **kwargs)
    torch.testing.assert_close(zero_y, plain)


@pytest.mark.parametrize("after_matrix", POSITIONS, ids=POSITION_IDS)
@pytest.mark.parametrize("az_dim", [2, 3])
@pytest.mark.parametrize("with_zi", [False, True])
def test_delay_state_space_absorption_zero_length(after_matrix, az_dim, with_zi):
    delays, orders = (2, 3), (1, 2)
    x = torch.empty(2, 0)
    A = torch.eye(2)
    b, a = torch.randn(2, 2), torch.randn(2, 2)
    az_shape = (2, 2) if az_dim == 2 else (2, 2, 2)
    absorption_zi = torch.randn(*az_shape)
    zi = (
        (torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([5.0, 6.0, 7.0])) if with_zi else None
    )

    y, zf, azf = delay_state_space(
        A,
        x,
        delays,
        C=torch.ones(2),
        D=torch.tensor(0.5),
        zi=zi,
        absorption=(b, a),
        absorption_zi=absorption_zi,
        absorption_after_matrix=after_matrix,
    )

    assert y.shape == (2, 0)
    assert y.dtype == x.dtype
    assert azf.shape == (2, 2, max(orders))
    torch.testing.assert_close(azf, absorption_zi.expand(2, -1, -1))
    if with_zi:
        _assert_states_close(zf, (zi[0], zi[1].expand(2, -1)))
    else:
        _assert_states_close(zf, tuple(torch.zeros(2, delay) for delay in delays))


@pytest.mark.parametrize("after_matrix", POSITIONS, ids=POSITION_IDS)
@pytest.mark.parametrize("batched_coefficients", [False, True])
@pytest.mark.parametrize("orders", [(1, 1), (0, 0)], ids=["order-1", "pure-gain"])
def test_delay_state_space_absorption_gradcheck(after_matrix, batched_coefficients, orders):
    delays, batch, samples, M = (2, 3), 2, 8, 2
    double = dict(dtype=torch.double, requires_grad=True)
    x = torch.randn(batch, samples, **double)
    A = (0.4 * torch.randn(M, M, dtype=torch.double)).requires_grad_()
    coefficient_batch = (batch,) if batched_coefficients else ()
    b = (
        0.5 * torch.randn(*coefficient_batch, M, orders[0] + 1, dtype=torch.double)
    ).requires_grad_()
    a = (0.3 * torch.randn(*coefficient_batch, M, orders[1], dtype=torch.double)).requires_grad_()
    absorption_zi = torch.randn(batch, M, max(orders), **double)
    # The initial queues reach y through the indexed reads of the stream.
    zi = tuple(torch.randn(batch, delay, **double) for delay in delays)

    def run(A, b, a, absorption_zi, x, *zi):
        y, zf, azf = delay_state_space(
            A,
            x,
            delays,
            C=torch.ones(M, dtype=torch.double),
            zi=zi,
            absorption=(b, a),
            absorption_zi=absorption_zi,
            absorption_after_matrix=after_matrix,
        )
        return y, azf, *zf

    assert torch.autograd.gradcheck(run, (A, b, a, absorption_zi, x, *zi))


@pytest.mark.parametrize("after_matrix", POSITIONS, ids=POSITION_IDS)
@pytest.mark.parametrize("orders", [(1, 1), (0, 0)])
@pytest.mark.parametrize(
    ("delays", "block_size"), [((2, 37), 1), ((2, 37), 2), ((1, 50), 1)], ids=str
)
def test_delay_state_space_absorption_spread_delays(delays, block_size, orders, after_matrix):
    # A short and a long line read from chunks far apart in the stream.
    dtype, batch, samples, M = torch.double, 2, 120, 2
    x = torch.randn(batch, samples, M, dtype=dtype)
    A = 0.3 * torch.randn(M, M, dtype=dtype)
    b = 0.5 * torch.randn(M, orders[0] + 1, dtype=dtype)
    a = 0.3 * torch.randn(M, orders[1], dtype=dtype)
    zi = tuple(torch.randn(batch, delay, dtype=dtype) for delay in delays)

    y, zf, azf = delay_state_space(
        A,
        x,
        delays,
        zi=zi,
        block_size=block_size,
        absorption=(b, a),
        absorption_after_matrix=after_matrix,
    )
    expected_y, expected_zf, expected_azf = _reference_absorption(
        A, x, delays, None, None, None, None, zi, b, a, None, after_matrix
    )

    torch.testing.assert_close(y, expected_y)
    _assert_states_close(zf, expected_zf)
    torch.testing.assert_close(azf, expected_azf)


@pytest.mark.parametrize("after_matrix", POSITIONS, ids=POSITION_IDS)
def test_delay_state_space_absorption_unroll_factor(after_matrix):
    dtype, orders = torch.double, (2, 1)
    x = torch.randn(BATCH, SAMPLES, dtype=dtype)
    A = 0.4 * torch.randn(LINES, LINES, dtype=dtype)
    B, C, D = (
        torch.randn(shape, dtype=dtype) for shape in ((LINES,), (OUTPUTS, LINES), (OUTPUTS,))
    )
    absorption = _filter_coefficients("shared", orders, dtype)
    zi = _initial_states("2d", dtype)
    absorption_zi = _absorption_state("batched", orders, dtype)
    kwargs = dict(
        B=B,
        C=C,
        D=D,
        absorption=absorption,
        zi=zi,
        absorption_zi=absorption_zi,
        absorption_after_matrix=after_matrix,
        block_size=2,
    )
    expected_y, expected_zf, expected_azf = _reference_absorption(
        A, x, DELAYS, B, C, D, None, zi, *absorption, absorption_zi, after_matrix
    )

    # unroll_factor=1 is new here, so it runs the sequential loop quietly.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        runs = [
            delay_state_space(A, x, DELAYS, unroll_factor=factor, **kwargs)
            for factor in (None, 1, 2)
        ]

    for y, zf, azf in runs:
        torch.testing.assert_close(y, expected_y)
        _assert_states_close(zf, expected_zf)
        torch.testing.assert_close(azf, expected_azf)


@pytest.mark.parametrize("after_matrix", POSITIONS, ids=POSITION_IDS)
@pytest.mark.parametrize("x_dim", [2, 3])
@pytest.mark.parametrize("chunk", [1, 5, 8])
def test_delay_state_space_absorption_across_spans(monkeypatch, chunk, x_dim, after_matrix):
    # With block_size 4, chunks of 1, 5 and 8 samples make spans of 4, 8 and 8,
    # so the absorption state must carry over from one span to the next.
    dtype, delays, samples = torch.double, (4, 6), 37
    x = torch.randn(2, samples, *(2,) * (x_dim - 2), dtype=dtype)
    A = 0.5 * torch.randn(2, 2, dtype=dtype)
    b, a = 0.5 * torch.randn(2, 2, dtype=dtype), 0.3 * torch.randn(2, 1, dtype=dtype)
    zi = (torch.randn(4, dtype=dtype), torch.randn(2, 6, dtype=dtype))
    absorption_zi = torch.randn(2, 1, dtype=dtype)
    expected_y, expected_zf, expected_azf = _reference_absorption(
        A, x, delays, None, None, None, None, zi, b, a, absorption_zi, after_matrix
    )

    monkeypatch.setattr(delay_module, "_OUTPUT_CHUNK", chunk)
    y, zf, azf = delay_state_space(
        A,
        x,
        delays,
        zi=zi,
        absorption=(b, a),
        absorption_zi=absorption_zi,
        absorption_after_matrix=after_matrix,
    )

    torch.testing.assert_close(y, expected_y)
    _assert_states_close(zf, expected_zf)
    torch.testing.assert_close(azf, expected_azf)


@pytest.mark.parametrize("after_matrix", POSITIONS, ids=POSITION_IDS)
@pytest.mark.parametrize("factor", [None, 1, 2])
def test_delay_state_space_absorption_passes_unroll_factor(monkeypatch, factor, after_matrix):
    dtype, delays = torch.double, (4, 6)
    x = torch.randn(2, 22, dtype=dtype)  # blocks of 4 samples, the last of 2
    A = 0.5 * torch.randn(2, 2, dtype=dtype)
    absorption = 0.5 * torch.randn(2, 2, dtype=dtype), 0.3 * torch.randn(2, 1, dtype=dtype)
    calls = []

    def spy(A, x, *args, **kwargs):
        calls.append((x.size(1), kwargs["unroll_factor"]))
        return state_space(A, x, *args, **kwargs)

    monkeypatch.setattr(delay_module, "state_space", spy)
    delay_state_space(
        A,
        x,
        delays,
        absorption=absorption,
        absorption_after_matrix=after_matrix,
        unroll_factor=factor,
    )

    # unroll_factor=1 becomes one more than the block's length: any larger
    # factor runs the sequential loop, without state_space's FutureWarning.
    assert [length for length, _ in calls] == [4] * 5 + [2]
    expected = {None: lambda n: None, 1: lambda n: n + 1, 2: lambda n: 2}[factor]
    assert [value for _, value in calls] == [expected(length) for length, _ in calls]

    calls.clear()
    pure_gain = absorption[0][:, :1], absorption[1][:, :0]
    delay_state_space(A, x, delays, absorption=pure_gain, unroll_factor=factor)
    assert calls == []


@pytest.mark.parametrize(
    "factor",
    [
        pytest.param(lambda np: np.int64(2), id="numpy-scalar"),
        pytest.param(lambda np: torch.tensor(2), id="tensor"),
    ],
)
@pytest.mark.parametrize("container", [tuple, list])
def test_delay_state_space_absorption_accepts_integer_like_unroll_factor_and_list(
    factor, container
):
    np = pytest.importorskip("numpy")
    dtype, delays = torch.double, (4, 6)
    x = torch.randn(2, 22, dtype=dtype)
    A = 0.5 * torch.randn(2, 2, dtype=dtype)
    absorption = 0.5 * torch.randn(2, 2, dtype=dtype), 0.3 * torch.randn(2, 1, dtype=dtype)
    expected = delay_state_space(A, x, delays, absorption=absorption, unroll_factor=2)

    result = delay_state_space(
        A, x, delays, absorption=container(absorption), unroll_factor=factor(np)
    )

    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize("mode", ["plain", "after-delay", "after-matrix"])
@pytest.mark.parametrize("block_size", [None, 1])
def test_delay_state_space_returned_queues_are_contiguous(mode, block_size):
    dtype = torch.double
    x = torch.randn(BATCH, SAMPLES, dtype=dtype)
    A = 0.4 * torch.randn(LINES, LINES, dtype=dtype)
    kwargs = dict(zi=_initial_states("1d", dtype), block_size=block_size)
    if mode != "plain":
        kwargs.update(
            absorption=_filter_coefficients("shared", (1, 1), dtype),
            absorption_after_matrix=mode == "after-matrix",
        )

    zf = delay_state_space(A, x, DELAYS, **kwargs)[1]

    assert all(queue.is_contiguous() for queue in zf)


def _validation_inputs(dtype=torch.double):
    return dict(
        A=torch.zeros(2, 2, dtype=dtype),
        x=torch.zeros(BATCH, 9, dtype=dtype),
        delays=(2, 3),
        absorption=(torch.zeros(2, 2, dtype=dtype), torch.zeros(2, 1, dtype=dtype)),
    )


def _zeros(*shape):
    return torch.zeros(*shape, dtype=torch.double)


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    [
        pytest.param(
            dict(absorption=(_zeros(2, 2),)), ValueError, "pair of tensors", id="one-element"
        ),
        pytest.param(
            dict(absorption=(_zeros(2, 2), _zeros(2, 1), _zeros(2, 1))),
            ValueError,
            "pair of tensors",
            id="three-elements",
        ),
        pytest.param(dict(absorption=_zeros(2, 2)), ValueError, "pair of tensors", id="tensor"),
        pytest.param(
            dict(absorption=(_zeros(2), _zeros(2, 1))),
            ValueError,
            "absorption b must be 2D or 3D",
            id="b-1d",
        ),
        pytest.param(
            dict(absorption=(_zeros(2, 2), _zeros(1, BATCH, 2, 1))),
            ValueError,
            "absorption a must be 2D or 3D",
            id="a-4d",
        ),
        pytest.param(
            dict(absorption=(_zeros(3, 2), _zeros(2, 1))),
            ValueError,
            "absorption b must match the number of delays",
            id="b-lines",
        ),
        pytest.param(
            dict(absorption=(_zeros(2, 2), _zeros(BATCH, 3, 1))),
            ValueError,
            "absorption a must match the number of delays",
            id="a-lines",
        ),
        pytest.param(
            dict(absorption=(_zeros(BATCH + 1, 2, 2), _zeros(2, 1))),
            ValueError,
            "Batch size of absorption b",
            id="b-batch",
        ),
        pytest.param(
            dict(absorption=(_zeros(2, 2), _zeros(BATCH + 1, 2, 1))),
            ValueError,
            "Batch size of absorption a",
            id="a-batch",
        ),
        pytest.param(
            dict(absorption_zi=_zeros(2)), ValueError, "absorption_zi must be 2D or 3D", id="zi-1d"
        ),
        pytest.param(
            dict(absorption_zi=_zeros(1, BATCH, 2, 1)),
            ValueError,
            "absorption_zi must be 2D or 3D",
            id="zi-4d",
        ),
        pytest.param(
            dict(absorption_zi=_zeros(3, 1)),
            ValueError,
            "absorption_zi must match the number of delays",
            id="zi-lines",
        ),
        pytest.param(
            dict(absorption_zi=_zeros(2, 2)),
            ValueError,
            "Last dimension of absorption_zi must be",
            id="zi-order",
        ),
        pytest.param(
            dict(absorption_zi=_zeros(BATCH + 1, 2, 1)),
            ValueError,
            "Batch size of absorption_zi",
            id="zi-batch",
        ),
        pytest.param(
            dict(absorption=([[0.0, 0.0]] * 2, _zeros(2, 1))),
            TypeError,
            "absorption b must be a Tensor",
            id="b-list",
        ),
        pytest.param(
            dict(absorption=(_zeros(2, 2), [[0.0]] * 2)),
            TypeError,
            "absorption a must be a Tensor",
            id="a-list",
        ),
        pytest.param(
            dict(absorption_zi=[[0.0], [0.0]]),
            TypeError,
            "absorption_zi must be a Tensor",
            id="zi-list",
        ),
        pytest.param(
            dict(absorption=None, absorption_zi=_zeros(2, 1)),
            ValueError,
            "absorption_zi needs absorption",
            id="zi-without-absorption",
        ),
        pytest.param(
            dict(absorption=None, absorption_after_matrix=True),
            ValueError,
            "absorption_after_matrix needs absorption",
            id="after-matrix-without-absorption",
        ),
        pytest.param(
            dict(absorption=None, unroll_factor=2),
            ValueError,
            "unroll_factor needs absorption",
            id="unroll-without-absorption",
        ),
        pytest.param(
            dict(absorption=(_zeros(2, 2).float(), _zeros(2, 1))),
            ValueError,
            "must have the dtype and device of x",
            id="b-dtype",
        ),
        pytest.param(
            dict(absorption=(_zeros(2, 2), _zeros(2, 1).float())),
            ValueError,
            "must have the dtype and device of x",
            id="a-dtype",
        ),
        pytest.param(
            dict(absorption_zi=_zeros(2, 1).float()),
            ValueError,
            "must have the dtype and device of x",
            id="zi-dtype",
        ),
        pytest.param(
            dict(absorption=(_zeros(2, 2).to("meta"), _zeros(2, 1))),
            ValueError,
            "dtype and device",
            id="b-device",
        ),
        pytest.param(
            dict(absorption=(_zeros(2, 2), _zeros(2, 1).to("meta"))),
            ValueError,
            "dtype and device",
            id="a-device",
        ),
        pytest.param(
            dict(absorption_zi=_zeros(2, 1).to("meta")),
            ValueError,
            "dtype and device",
            id="zi-device",
        ),
        pytest.param(
            dict(absorption=(_zeros(2, 0), _zeros(2, 1))),
            ValueError,
            "absorption b must have at least one coefficient",
            id="b-empty",
        ),
        pytest.param(
            dict(unroll_factor=0),
            ValueError,
            "unroll_factor must be None or at least 1, got 0",
            id="unroll-zero",
        ),
        pytest.param(
            dict(absorption=(_zeros(2, 1), _zeros(2, 0)), unroll_factor=-3),
            ValueError,
            "unroll_factor must be None or at least 1, got -3",
            id="unroll-negative-pure-gain",
        ),
        pytest.param(
            dict(unroll_factor=1.5),
            ValueError,
            "unroll_factor must be None or at least 1, got 1.5",
            id="unroll-float",
        ),
        pytest.param(
            dict(unroll_factor=True),
            ValueError,
            "unroll_factor must be None or at least 1, got True",
            id="unroll-bool",
        ),
    ],
)
def test_delay_state_space_validates_absorption(overrides, error, message):
    inputs = {**_validation_inputs(), **overrides}
    A, x, delays = inputs.pop("A"), inputs.pop("x"), inputs.pop("delays")
    with pytest.raises(error, match=message):
        delay_state_space(A, x, delays, **inputs)


@pytest.mark.parametrize("samples", [0, 2, 9], ids=["empty", "shorter-than-delays", "longer"])
@pytest.mark.parametrize(
    "absorption",
    [None, ("after-delay", 1), ("after-matrix", 1), ("after-delay", 0), ("after-matrix", 0)],
    ids=["none", "after-delay", "after-matrix", "after-delay-gain", "after-matrix-gain"],
)
@pytest.mark.parametrize("x_dim", [2, 3])
def test_delay_state_space_every_input_gets_a_gradient(samples, absorption, x_dim):
    # Shortcuts must not drop inputs from the graph: no steps, a pure gain or
    # an output read only from the initial queues still gives zero gradients,
    # not None.
    delays, batch, M = (3, 4), 2, 2
    f64 = dict(dtype=torch.double, requires_grad=True)
    A = (0.4 * torch.randn(batch, M, M, dtype=torch.double)).requires_grad_()
    x = torch.randn(batch, samples, *((M,) if x_dim == 3 else ()), **f64)
    B = torch.randn(M, M, **f64) if x_dim == 3 else torch.randn(M, **f64)
    zi = tuple(torch.randn(batch, delay, **f64) for delay in delays)
    inputs = [A, x, B, *zi]
    kwargs = dict(B=B, C=torch.ones(M, dtype=torch.double), zi=zi)
    if absorption is not None:
        position, order = absorption
        b = torch.randn(M, order + 1, **f64)
        a = (0.3 * torch.randn(M, order, dtype=torch.double)).requires_grad_()
        absorption_zi = torch.randn(batch, M, order, **f64)
        inputs += [b, a, absorption_zi]
        kwargs.update(
            absorption=(b, a),
            absorption_zi=absorption_zi,
            absorption_after_matrix=position == "after-matrix",
        )
    y, zf, *azf = delay_state_space(A, x, delays, **kwargs)
    total = y.sum() + sum(q.sum() for q in zf) + sum(t.sum() for t in azf)
    for grad in torch.autograd.grad(total, inputs, allow_unused=True):
        assert grad is not None
