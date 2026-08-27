import pytest
import torch
from torch.autograd import gradcheck

import philtorch._torchlpc as torchlpc
from philtorch.lpv import linear_recurrence


def _scan_reference(impulse, decay, init):
    output = []
    state = init
    for impulse_t, decay_t in zip(impulse.unbind(1), decay.unbind(1)):
        state = decay_t * state + impulse_t
        output.append(state)
    return torch.stack(output, dim=1)


def _allpole_reference(x, A, zi):
    history = zi.flip(1)
    output = []
    for t in range(x.shape[1]):
        previous = history[:, t : t + A.shape[2]].flip(1)
        y = x[:, t] - torch.sum(A[:, t] * previous, dim=1)
        output.append(y)
        history = torch.cat([history, y.unsqueeze(1)], dim=1)
    return torch.stack(output, dim=1)


@pytest.mark.parametrize("extension_loaded", [True, False])
def test_scan_forward_and_gradcheck(monkeypatch, extension_loaded):
    monkeypatch.setattr(torchlpc, "EXTENSION_LOADED", extension_loaded)
    impulse = torch.tensor(
        [[0.2, -0.3, 0.7, 0.1], [-0.4, 0.8, 0.5, -0.2]],
        dtype=torch.double,
        requires_grad=True,
    )
    decay = torch.tensor(
        [[0.1, 0.4, -0.2, 0.6], [0.7, -0.5, 0.3, 0.2]],
        dtype=torch.double,
        requires_grad=True,
    )
    init = torch.tensor([0.9, -0.6], dtype=torch.double, requires_grad=True)

    expected = _scan_reference(impulse, decay, init)
    actual = torchlpc.ScanRecurrence.apply(impulse, decay, init)
    torch.testing.assert_close(actual, expected)
    assert gradcheck(
        torchlpc.ScanRecurrence.apply,
        (impulse, decay, init),
        check_forward_ad=True,
    )


@pytest.mark.parametrize("extension_loaded", [True, False])
def test_allpole_forward_and_gradcheck(monkeypatch, extension_loaded):
    monkeypatch.setattr(torchlpc, "EXTENSION_LOADED", extension_loaded)
    x = torch.randn(2, 4, dtype=torch.double, requires_grad=True)
    A = (torch.randn(2, 4, 2, dtype=torch.double) * 0.1).requires_grad_()
    zi = torch.randn(2, 2, dtype=torch.double, requires_grad=True)

    expected = _allpole_reference(x, A, zi)
    actual = torchlpc.AllPole.apply(x, A, zi)
    torch.testing.assert_close(actual, expected)
    assert gradcheck(torchlpc.AllPole.apply, (x, A, zi), check_forward_ad=True)


def test_scan_and_allpole_vmap():
    outer, batch, samples, order = 2, 2, 4, 2
    impulse = torch.randn(outer, batch, samples, dtype=torch.double)
    decay = torch.rand(outer, batch, samples, dtype=torch.double) * 0.5
    init = torch.randn(batch, dtype=torch.double)
    actual_scan = torch.vmap(
        lambda impulse_i, decay_i: torchlpc.ScanRecurrence.apply(
            impulse_i, decay_i, init
        )
    )(impulse, decay)
    expected_scan = torch.stack(
        [_scan_reference(impulse[i], decay[i], init) for i in range(outer)]
    )
    torch.testing.assert_close(actual_scan, expected_scan)

    x = torch.randn(outer, batch, samples, dtype=torch.double)
    A = torch.randn(outer, batch, samples, order, dtype=torch.double) * 0.1
    zi = torch.randn(batch, order, dtype=torch.double)
    actual_allpole = torch.vmap(lambda x_i, A_i: torchlpc.AllPole.apply(x_i, A_i, zi))(
        x, A
    )
    expected_allpole = torch.stack(
        [_allpole_reference(x[i], A[i], zi) for i in range(outer)]
    )
    torch.testing.assert_close(actual_allpole, expected_allpole)


def test_scan_backward_without_init_gradient():
    impulse = torch.randn(2, 4, dtype=torch.double, requires_grad=True)
    decay = (torch.rand(2, 4, dtype=torch.double) * 0.5).requires_grad_()
    init = torch.randn(2, dtype=torch.double)
    actual = torch.autograd.grad(
        torchlpc.ScanRecurrence.apply(impulse, decay, init).sum(),
        (impulse, decay),
    )
    expected = torch.autograd.grad(
        _scan_reference(impulse, decay, init).sum(),
        (impulse, decay),
    )
    for actual_grad, expected_grad in zip(actual, expected):
        torch.testing.assert_close(actual_grad, expected_grad)


def test_linear_recurrence_normalizes_supported_shapes():
    x = torch.randn(2, 5, dtype=torch.double)
    decay = torch.linspace(0.1, 0.5, 5, dtype=torch.double)
    expected = _scan_reference(x, decay.expand_as(x), torch.full((2,), 0.25))

    torch.testing.assert_close(
        linear_recurrence(decay, torch.tensor(0.25, dtype=torch.double), x), expected
    )
    torch.testing.assert_close(
        linear_recurrence(
            decay,
            torch.tensor([0.25], dtype=torch.double),
            x,
            unroll_factor=2,
        ),
        expected,
    )


@pytest.mark.parametrize("unroll_factor", [0, -1])
def test_linear_recurrence_rejects_invalid_unroll_factor(unroll_factor):
    with pytest.raises(ValueError, match="Unroll factor must be >= 1"):
        linear_recurrence(
            torch.ones(3),
            torch.tensor(0.0),
            torch.ones(2, 3),
            unroll_factor=unroll_factor,
        )
