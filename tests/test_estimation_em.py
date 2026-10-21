import pytest
import torch

from philtorch.estimation import em


def test_em_stops_once_the_rise_is_below_tol():
    # Rises of 1, 1/2, 1/4, ...: the third, 1/4, is below tol.
    def step(k):
        return torch.tensor([-(2.0**-k)]), (k + 1,)

    params, history = em(step, (-1,), tol=0.3)
    assert params == (3,)
    assert history.tolist() == [-2.0, -1.0, -0.5, -0.25]


def test_em_needs_a_step():
    with pytest.raises(ValueError, match="max_iter"):
        em(lambda: None, (), max_iter=0)
