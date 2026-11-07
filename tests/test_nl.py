import pytest
import torch

from philtorch.nl import _newton_solve


@pytest.mark.parametrize("is_scalar", [True, False])
@pytest.mark.parametrize("use_fprime", [False, True])
def test_newton_solve_forward_against_sequential_ref(is_scalar: bool, use_fprime: bool):
    B, T = 2, 8
    torch.manual_seed(0)
    if is_scalar:
        step_fn = lambda y, x, w: torch.tanh(w * y + x)
        fprime = (
            (lambda y, x, w: (1.0 - step_fn(y, x, w).square()) * w)
            if use_fprime
            else None
        )
        x = torch.randn(B, T, dtype=torch.float64)
        init = torch.randn(B, dtype=torch.float64)
        w = torch.tensor(0.5, dtype=torch.float64)
        params = (w,)
        # Sequential ref
        y_ref = torch.empty(B, T, dtype=torch.float64)
        curr = init
        for t in range(T):
            curr = step_fn(curr, x[:, t], w)
            y_ref[:, t] = curr
    else:
        M = 2
        step_fn = lambda y, x, A: torch.tanh(y @ A.T + x)
        fprime = (
            (lambda y, x, A: (1.0 - step_fn(y, x, A).square()).unsqueeze(-1) * A)
            if use_fprime
            else None
        )
        x = torch.randn(B, T, M, dtype=torch.float64)
        init = torch.randn(B, M, dtype=torch.float64)
        A = torch.tensor([[0.3, 0.1], [0.0, 0.2]], dtype=torch.float64)
        params = (A,)
        y_ref = torch.empty(B, T, M, dtype=torch.float64)
        curr = init
        for t in range(T):
            curr = step_fn(curr, x[:, t], A)
            y_ref[:, t] = curr

    y_sol, iters = _newton_solve(
        step_fn,
        x,
        init,
        params=params,
        fprime=fprime,
        y0=None,
        max_iter=100,
        max_halvings=8,
        atol=None,
        rtol=None,
        recur_kwargs={},
    )
    assert torch.allclose(y_sol, y_ref, atol=1e-10, rtol=1e-10)


def test_newton_solve_linear_system_convergence():
    B, T, M = 2, 6, 2
    torch.manual_seed(1)
    A = torch.tensor([[0.4, 0.1], [0.0, 0.3]], dtype=torch.float64)
    x = torch.randn(B, T, M, dtype=torch.float64)
    init = torch.randn(B, M, dtype=torch.float64)
    step_fn = lambda y, x, A: y @ A.T + x
    fprime = lambda y, x, A: A.expand(y.shape[0], y.shape[1], M, M)

    y_sol, iters = _newton_solve(
        step_fn,
        x,
        init,
        params=(A,),
        fprime=fprime,
        y0=None,
        max_iter=100,
        max_halvings=8,
        atol=None,
        rtol=None,
        recur_kwargs={},
    )
    assert iters <= 2
