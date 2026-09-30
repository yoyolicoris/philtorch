import pytest
import torch

import philtorch.lti.delay as delay_module
from philtorch.lti import delay_state_space, state_space

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
        "B(b,M) C(b,P,M) D(b,P)": dict(
            x=siso, B=randn(b, M), C=randn(b, P, M), D=randn(b, P)
        ),
        "B(M) out_idx D()": dict(x=siso, B=randn(M), D=randn(()), out_idx=2),
        "defaults 2D": dict(x=siso),
        "B(M,F) C(M) D(F)": dict(x=mimo, B=randn(M, F), C=randn(M), D=randn(F)),
        "B(b,M,F) C(b,M) D(b,F)": dict(
            x=mimo, B=randn(b, M, F), C=randn(b, M), D=randn(b, F)
        ),
        "B(M,F) C(P,M) D(P,F)": dict(
            x=mimo, B=randn(M, F), C=randn(P, M), D=randn(P, F)
        ),
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
    torch.manual_seed(0)
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
    torch.manual_seed(1)
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
    torch.manual_seed(2)
    delays = (3, 7)
    x = torch.randn(BATCH, samples, dtype=torch.double)
    A = 0.5 * torch.randn(2, 2, dtype=torch.double)
    zi = (torch.randn(BATCH, 3, dtype=torch.double), torch.randn(7, dtype=torch.double))

    y, zf = delay_state_space(A, x, delays, zi=zi)
    expected_y, expected_zf = _as_state_space(A, x, delays, None, None, None, zi, None)

    torch.testing.assert_close(y, expected_y)
    _assert_states_close(zf, expected_zf)


def test_delay_state_space_batched_matrices_are_not_conjugated_like_state_space():
    torch.manual_seed(3)
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
    torch.manual_seed(4)
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


def test_delay_state_space_is_functional():
    torch.manual_seed(5)
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
    torch.manual_seed(6)
    delays = (2, 4)
    x = torch.randn(2, 7, dtype=torch.double)
    A = 0.4 * torch.randn(2, 2, dtype=torch.double)
    B, C, D = (torch.randn(s, dtype=torch.double) for s in ((2,), (2,), ()))
    zi = (torch.randn(2, 2, dtype=torch.double), torch.randn(2, 4, dtype=torch.double))

    expected_y, expected_zf = delay_state_space(A, x, delays, B=B, C=C, D=D, zi=zi)
    first_y, first_zf = delay_state_space(A, x[:, :3], delays, B=B, C=C, D=D, zi=zi)
    second_y, actual_zf = delay_state_space(
        A, x[:, 3:], delays, B=B, C=C, D=D, zi=first_zf
    )

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
        (torch.tensor([[1.0, 2.0], [3.0, 4.0]]), torch.tensor([5.0, 6.0, 7.0]))
        if with_zi
        else None
    )

    result = delay_state_space(
        A, x, delays, C=C, D=torch.tensor(0.5), zi=zi, out_idx=out_idx
    )

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
        delay_state_space(
            torch.zeros(M, M), torch.zeros(1, 4), delays, block_size=block_size
        )


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
    with pytest.raises(AssertionError, match=message):
        delay_state_space(A, x, (2, 3))


def test_delay_state_space_rejects_C_with_out_idx():
    with pytest.raises(ValueError, match="C and out_idx"):
        delay_state_space(
            torch.zeros(2, 2), torch.zeros(2, 4), (2, 3), C=torch.ones(2), out_idx=0
        )


@pytest.mark.parametrize(
    ("x", "B", "error", "message"),
    [
        (torch.zeros(2, 4, 3), None, AssertionError, "number of delays when B is None"),
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
        ((torch.zeros(2, 2), torch.zeros(1, 2, 3)), AssertionError, "1D or 2D"),
        ((torch.zeros(2, 2), torch.zeros(2, 4)), AssertionError, "match delay"),
        ((torch.zeros(2, 2), torch.zeros(3, 3)), AssertionError, "Batch size of zi"),
    ],
)
def test_delay_state_space_validates_initial_states(zi, error, message):
    with pytest.raises(error, match=message):
        delay_state_space(torch.zeros(2, 2), torch.zeros(2, 4), (2, 3), zi=zi)
