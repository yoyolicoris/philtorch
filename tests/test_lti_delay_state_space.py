import pytest
import torch
from torch.autograd import gradcheck

from philtorch.lti import delay_state_space

_REQUIRES_GRAD_CASES = (
    (True, False, False),
    (False, True, False),
    (False, False, True),
    (True, True, False),
    (True, False, True),
    (False, True, True),
    (True, True, True),
)


def _sample_reference(A, x, B, C, D, zi):
    states = tuple(zi)
    outputs = []
    for xn in x.unbind(1):
        delay_outputs = torch.stack([state[:, 0] for state in states], dim=-1)
        feedback = (delay_outputs.unsqueeze(-2) @ A.mT).squeeze(-2)
        if x.dim() == 2:
            delay_inputs = feedback + xn.unsqueeze(-1) * B
        else:
            delay_inputs = feedback + (xn.unsqueeze(-2) @ B.mT).squeeze(-2)

        if C.dim() == 1:
            yn = delay_outputs @ C
        else:
            yn = (delay_outputs.unsqueeze(-2) @ C.mT).squeeze(-2)
        if D.dim() == 0:
            yn = yn + D * xn
        else:
            yn = yn + (xn.unsqueeze(-2) @ D.mT).squeeze(-2)
        outputs.append(yn)
        states = tuple(
            torch.cat([state[:, 1:], delay_inputs[:, index, None]], dim=-1)
            for index, state in enumerate(states)
        )

    if outputs:
        output = torch.stack(outputs, dim=1)
    else:
        output_shape = (x.size(0), 0) if C.dim() == 1 else (x.size(0), 0, C.size(-2))
        output = x.new_empty(output_shape)
    return output, states


def _flatten_result(result):
    y, zf = result
    return torch.cat([y.reshape(-1), *(state.reshape(-1) for state in zf)])


def test_delay_state_space_matches_sample_reference_and_is_functional():
    delays = (2, 5, 3)
    x = torch.arange(18, dtype=torch.double).reshape(2, 9) / 10
    A = torch.tensor(
        [[0.1, -0.2, 0.3], [0.2, 0.05, -0.1], [-0.1, 0.15, 0.2]],
        dtype=torch.double,
    )
    B = torch.tensor([0.4, -0.2, 0.1], dtype=torch.double)
    C = torch.tensor([0.3, 0.5, -0.4], dtype=torch.double)
    D = torch.tensor(0.2, dtype=torch.double)
    zi = (
        torch.arange(4, dtype=torch.double).reshape(2, 2) / 7,
        torch.arange(10, dtype=torch.double).reshape(2, 5) / 11,
        torch.arange(6, dtype=torch.double).reshape(2, 3) / 13,
    )
    originals = (x.clone(), A.clone(), B.clone(), C.clone(), D.clone()) + tuple(
        state.clone() for state in zi
    )

    actual = delay_state_space(A, x, delays, B=B, C=C, D=D, zi=zi, block_size=2)
    expected = _sample_reference(A, x, B, C, D, zi)

    assert torch.allclose(actual[0], expected[0])
    assert all(torch.allclose(a, e) for a, e in zip(actual[1], expected[1]))
    for value, original in zip((x, A, B, C, D) + zi, originals):
        assert torch.equal(value, original)
    assert all(
        actual_state.data_ptr() != initial_state.data_ptr()
        for actual_state, initial_state in zip(actual[1], zi)
    )


def test_delay_state_space_batched_mimo_matches_sample_reference():
    delays = (2, 4)
    x = torch.arange(20, dtype=torch.double).reshape(2, 5, 2) / 20
    A = torch.tensor(
        [[[0.1, 0.2], [-0.1, 0.3]], [[0.2, -0.2], [0.15, 0.1]]],
        dtype=torch.double,
    )
    B = torch.tensor(
        [[[0.2, 0.1], [-0.3, 0.4]], [[-0.1, 0.3], [0.2, 0.25]]],
        dtype=torch.double,
    )
    C = torch.tensor(
        [[[0.5, -0.2], [0.1, 0.4]], [[0.3, 0.2], [-0.1, 0.6]]],
        dtype=torch.double,
    )
    D = torch.tensor(
        [[[0.1, 0.0], [0.0, -0.2]], [[0.05, 0.1], [-0.1, 0.2]]],
        dtype=torch.double,
    )
    zi = (
        torch.tensor([[0.1, 0.2], [-0.1, 0.3]], dtype=torch.double),
        torch.tensor(
            [[0.2, -0.2, 0.1, 0.0], [0.3, 0.1, -0.1, 0.2]],
            dtype=torch.double,
        ),
    )

    actual = delay_state_space(A, x, delays, B=B, C=C, D=D, zi=zi, block_size=2)
    expected = _sample_reference(A, x, B, C, D, zi)

    assert torch.allclose(actual[0], expected[0])
    assert all(torch.allclose(a, e) for a, e in zip(actual[1], expected[1]))


def test_delay_state_space_zero_length_preserves_states():
    delays = (2, 3)
    x = torch.empty(2, 0, dtype=torch.float32)
    A = torch.eye(2, dtype=torch.float32)
    B = torch.ones(2, dtype=torch.float32)
    C = torch.ones(2, dtype=torch.float32)
    D = torch.tensor(0.0, dtype=torch.float32)
    zi = (
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        torch.tensor([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]),
    )

    y, zf = delay_state_space(A, x, delays, B=B, C=C, D=D, zi=zi)

    assert y.shape == (2, 0)
    assert y.dtype == x.dtype
    assert y.device == x.device
    assert all(torch.equal(final, initial) for final, initial in zip(zf, zi))


def test_delay_state_space_default_input_and_initial_state():
    x = torch.tensor([[1.0, 2.0, 3.0]])
    A = torch.zeros(2, 2)

    y, zf = delay_state_space(A, x, (1, 2), block_size=1)

    assert torch.equal(y, torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]]))
    assert torch.equal(zf[0], torch.tensor([[3.0]]))
    assert torch.equal(zf[1], torch.zeros(1, 2))
    assert y.dtype == x.dtype
    assert y.device == x.device


def test_delay_state_space_reuses_returned_states():
    delays = (2, 4)
    x = torch.arange(14, dtype=torch.double).reshape(2, 7) / 10
    A = torch.tensor([[0.1, -0.2], [0.3, 0.05]], dtype=torch.double)
    B = torch.tensor([0.4, -0.1], dtype=torch.double)
    C = torch.tensor([0.25, 0.6], dtype=torch.double)
    D = torch.tensor(-0.2, dtype=torch.double)
    zi = (
        torch.tensor([[0.1, -0.1], [0.2, 0.3]], dtype=torch.double),
        torch.tensor(
            [[0.0, 0.1, 0.2, 0.3], [-0.2, -0.1, 0.0, 0.1]],
            dtype=torch.double,
        ),
    )

    expected_y, expected_zf = delay_state_space(
        A, x, delays, B=B, C=C, D=D, zi=zi, block_size=2
    )
    first_y, first_zf = delay_state_space(
        A, x[:, :3], delays, B=B, C=C, D=D, zi=zi, block_size=2
    )
    second_y, actual_zf = delay_state_space(
        A, x[:, 3:], delays, B=B, C=C, D=D, zi=first_zf, block_size=2
    )

    assert torch.allclose(torch.cat([first_y, second_y], dim=1), expected_y)
    assert all(torch.allclose(a, e) for a, e in zip(actual_zf, expected_zf))


@pytest.mark.parametrize(
    ("delays", "block_size", "error", "message"),
    [
        ((), None, ValueError, "at least one"),
        ((2, 0), None, ValueError, "positive integer"),
        ((2, 3), 0, ValueError, "1 <= block_size"),
        ((2, 3), 3, ValueError, "1 <= block_size"),
    ],
)
def test_delay_state_space_validates_delays_and_block_size(
    delays, block_size, error, message
):
    x = torch.zeros(1, 4)
    A = torch.zeros(max(len(delays), 1), max(len(delays), 1))

    with pytest.raises(error, match=message):
        delay_state_space(A, x, delays, block_size=block_size)


def test_delay_state_space_validates_matrix_and_states():
    x = torch.zeros(2, 4)
    A = torch.zeros(2, 2)

    with pytest.raises(AssertionError, match="square"):
        delay_state_space(torch.zeros(2, 3), x, (2, 3))
    with pytest.raises(ValueError, match="Input matrix B"):
        delay_state_space(A, x, (2, 3), B=torch.zeros(4))
    with pytest.raises(ValueError, match="Output matrix C"):
        delay_state_space(A, x, (2, 3), C=torch.zeros(4))
    with pytest.raises(ValueError, match="Input matrix D"):
        delay_state_space(A, x, (2, 3), D=torch.zeros(2, 2, 2, 2))

    with pytest.raises(AssertionError, match="number of delays"):
        delay_state_space(A, x, (2, 3, 4))
    with pytest.raises(ValueError, match="one state"):
        delay_state_space(A, x, (2, 3), zi=(torch.zeros(2, 2),))
    with pytest.raises(AssertionError, match="match delay"):
        delay_state_space(
            A,
            x,
            (2, 3),
            zi=(torch.zeros(2, 2), torch.zeros(2, 4)),
        )


@pytest.mark.parametrize(
    ("x_requires_grad", "A_requires_grad", "zi_requires_grad"),
    _REQUIRES_GRAD_CASES,
)
def test_delay_state_space_gradcheck(
    x_requires_grad, A_requires_grad, zi_requires_grad
):
    delays = (2, 3)
    x = torch.tensor([[0.1, -0.2, 0.3]], dtype=torch.double).requires_grad_(
        x_requires_grad
    )
    A = torch.tensor([[0.1, -0.2], [0.25, 0.05]], dtype=torch.double).requires_grad_(
        A_requires_grad
    )
    B = torch.tensor([0.3, -0.1], dtype=torch.double)
    C = torch.tensor([0.4, 0.2], dtype=torch.double)
    D = torch.tensor(0.15, dtype=torch.double)
    zi = (
        torch.tensor([[0.2, -0.1]], dtype=torch.double).requires_grad_(
            zi_requires_grad
        ),
        torch.tensor([[0.05, 0.1, -0.2]], dtype=torch.double).requires_grad_(
            zi_requires_grad
        ),
    )

    def run(x_value, A_value, *zi_value):
        return _flatten_result(
            delay_state_space(
                A_value,
                x_value,
                delays,
                B=B,
                C=C,
                D=D,
                zi=zi_value,
                block_size=2,
            )
        )

    assert gradcheck(run, (x, A, *zi))
    if x_requires_grad and A_requires_grad and zi_requires_grad:
        run(x, A, *zi).square().sum().backward()
        assert x.grad is not None
        assert A.grad is not None
        assert all(state.grad is not None for state in zi)


def test_delay_state_space_parameter_gradcheck():
    delays = (2, 3)
    x = torch.tensor([[0.1, -0.2, 0.3]], dtype=torch.double)
    A = torch.tensor([[0.1, -0.2], [0.25, 0.05]], dtype=torch.double)
    B = torch.tensor([0.3, -0.1], dtype=torch.double, requires_grad=True)
    C = torch.tensor([0.4, 0.2], dtype=torch.double, requires_grad=True)
    D = torch.tensor(0.15, dtype=torch.double, requires_grad=True)
    zi = (
        torch.tensor([[0.2, -0.1]], dtype=torch.double),
        torch.tensor([[0.05, 0.1, -0.2]], dtype=torch.double),
    )

    def run(B_value, C_value, D_value):
        return _flatten_result(
            delay_state_space(
                A,
                x,
                delays,
                B=B_value,
                C=C_value,
                D=D_value,
                zi=zi,
                block_size=2,
            )
        )

    assert gradcheck(run, (B, C, D))
    run(B, C, D).square().sum().backward()
    assert B.grad is not None
    assert C.grad is not None
    assert D.grad is not None
