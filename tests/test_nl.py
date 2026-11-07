import pytest
import torch

from philtorch.nl import newton_solve


@pytest.mark.parametrize("is_scalar", [True, False])
@pytest.mark.parametrize("use_fprime", [False, True])
def test_newton_solve_forward_against_sequential_ref(is_scalar: bool, use_fprime: bool):
    B, T = 2, 8
    torch.manual_seed(0)
    if is_scalar:

        def step_fn(y, x, w):
            return torch.tanh(w * y + x)

        def fprime_fn(y, x, w):
            return (1.0 - step_fn(y, x, w).square()) * w

        fprime = fprime_fn if use_fprime else None
        x = torch.randn(B, T, dtype=torch.float64)
        init = torch.randn(B, dtype=torch.float64)
        params = (torch.tensor(0.5, dtype=torch.float64),)
    else:
        M = 2

        def step_fn(y, x, A):
            return torch.tanh(y @ A.T + x)

        def fprime_fn(y, x, A):
            return (1.0 - step_fn(y, x, A).square()).unsqueeze(-1) * A

        fprime = fprime_fn if use_fprime else None
        x = torch.randn(B, T, M, dtype=torch.float64)
        init = torch.randn(B, M, dtype=torch.float64)
        params = (torch.tensor([[0.3, 0.1], [0.0, 0.2]], dtype=torch.float64),)

    y_ref = torch.empty_like(x)
    curr = init
    for t in range(T):
        curr = step_fn(curr, x[:, t], *params)
        y_ref[:, t] = curr

    y_sol, _ = newton_solve(step_fn, x, init, params=params, fprime=fprime)
    assert torch.allclose(y_sol, y_ref, atol=1e-10, rtol=1e-10)


def test_newton_solve_linear_system_convergence():
    B, T, M = 2, 6, 2
    A = torch.tensor([[0.4, 0.1], [0.0, 0.3]], dtype=torch.float64)
    x = torch.randn(B, T, M, dtype=torch.float64)
    init = torch.randn(B, M, dtype=torch.float64)

    def step_fn(y, x, A):
        return y @ A.T + x

    def fprime(y, x, A):
        return A.expand(y.shape[0], y.shape[1], M, M)

    _, iters = newton_solve(step_fn, x, init, params=(A,), fprime=fprime)
    assert iters <= 2


@pytest.mark.parametrize("is_scalar", [True, False])
def test_newton_solve_gradcheck(is_scalar: bool):
    B, T = 2, 4
    if is_scalar:

        def step_fn(y, x, w):
            return torch.tanh(w * y + x)

        x = torch.randn(B, T, dtype=torch.float64, requires_grad=True)
        init = torch.randn(B, dtype=torch.float64, requires_grad=True)
        w = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
        inputs = (x, init, w)

        def fn(xi, ii, wi):
            return newton_solve(step_fn, xi, ii, params=(wi,))[0]
    else:
        M = 2

        def step_fn(y, x, A):
            return torch.tanh(y @ A.T + x)

        x = torch.randn(B, T, M, dtype=torch.float64, requires_grad=True)
        init = torch.randn(B, M, dtype=torch.float64, requires_grad=True)
        A = torch.tensor([[0.3, 0.1], [0.0, 0.2]], dtype=torch.float64, requires_grad=True)
        inputs = (x, init, A)

        def fn(xi, ii, Ai):
            return newton_solve(step_fn, xi, ii, params=(Ai,))[0]

    assert torch.autograd.gradcheck(fn, inputs, eps=1e-6, atol=1e-4)


def test_newton_solve_tuple_inputs():
    def step_fn(y, x):
        return torch.tanh(0.2 * y + x[0] + 0.3 * x[1])

    B, T = 2, 4
    x1 = torch.randn(B, T, dtype=torch.float64, requires_grad=True)
    x2 = torch.randn(B, T, dtype=torch.float64, requires_grad=True)
    init = torch.zeros(B, dtype=torch.float64)
    y, _ = newton_solve(step_fn, (x1, x2), init)
    y.sum().backward()
    assert x1.grad is not None and x2.grad is not None


def test_newton_solve_divergence():
    def step_fn(y, x):
        return 10.0 * (y**2) + x

    B, T = 2, 10
    x = torch.full((B, T), 10.0, dtype=torch.float64)
    init = torch.full((B,), 10.0, dtype=torch.float64)
    with pytest.warns(RuntimeWarning, match="diverged"):
        y, _ = newton_solve(step_fn, x, init, max_iter=3)
        assert torch.isnan(y).any()


def test_newton_solve_invalid_shapes():
    def step_fn(y, x):
        return y + x

    with pytest.raises(ValueError, match="at least 2 dimensions"):
        newton_solve(step_fn, torch.zeros(4), torch.zeros(4))
    with pytest.raises(ValueError, match="matching"):
        newton_solve(step_fn, (torch.zeros(2, 4), torch.zeros(2, 5)), torch.zeros(2))
    with pytest.raises(ValueError, match="Batch size of init"):
        newton_solve(step_fn, torch.zeros(2, 4), torch.zeros(3))
