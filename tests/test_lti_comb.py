import pytest
import torch

from philtorch.lti import comb_filter, lfilter


@pytest.mark.parametrize(
    ("delay", "zi_shape"),
    [
        (1, None),
        (5, None),
        (13, (13,)),
        (5, (5, 5)),
    ],
)
@pytest.mark.parametrize("batch_a", [True, False])
def test_comb_filter(delay, batch_a, zi_shape):
    B = 5
    T = 97

    # Generate random coefficients and signal
    if batch_a:
        a = torch.rand(B).double() * 2 - 1
    else:
        a = torch.rand(1).double() * 2 - 1
    x = torch.randn(B, T).double()
    zi = torch.randn(*zi_shape).double() if zi_shape else None

    if not batch_a:
        padded_a = torch.cat([torch.zeros(delay - 1), a])
        a = a[0]
    else:
        padded_a = torch.cat([torch.zeros((B, delay - 1)), a.unsqueeze(1)], dim=-1)

    comb_y = comb_filter(a, delay, x, zi=zi)
    lfilter_y = lfilter(torch.ones(B, 1).double(), padded_a, x, zi=zi, form="df2")
    if zi is not None:
        lfilter_y, lfilter_zf = lfilter_y
        comb_y, comb_zf = comb_y
        assert torch.allclose(comb_zf, lfilter_zf), torch.max(torch.abs(comb_zf - lfilter_zf))

    assert torch.allclose(comb_y, lfilter_y), torch.max(torch.abs(comb_y - lfilter_y))


def test_delay_one_returns_final_state():
    x = torch.randn(2, 10, dtype=torch.float64)
    a = torch.tensor([0.4, -0.3], dtype=torch.float64)
    y, zf = comb_filter(a, 1, x, zi=torch.zeros(2, 1, dtype=torch.float64))
    assert torch.allclose(y, comb_filter(a, 1, x))
    assert torch.equal(zf, y[:, -1:])


def test_delay_zero_raises():
    with pytest.raises(AssertionError, match="at least 1"):
        comb_filter(torch.tensor(0.4), 0, torch.randn(2, 10))
