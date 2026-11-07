import warnings

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

    def step_fn(y, x, A):
        return y @ A.T + x

    def fprime(y, x, A):
        return A.expand(B, T, M, M)

    x = torch.randn(B, T, M, dtype=torch.float64)
    init = torch.randn(B, M, dtype=torch.float64)
    _, iters = newton_solve(step_fn, x, init, params=(A,), fprime=fprime)
    assert iters <= 2


@pytest.mark.parametrize("is_scalar", [True, False])
def test_newton_solve_gradcheck(is_scalar: bool):
    B, T = 2, 4
    torch.manual_seed(0)
    if is_scalar:

        def step_fn(y, x, w):
            return torch.tanh(w * y + x)

        x = torch.randn(B, T, dtype=torch.float64, requires_grad=True)
        init = torch.randn(B, dtype=torch.float64, requires_grad=True)
        w = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
        params = (w,)
    else:
        M = 2

        def step_fn(y, x, A):
            return torch.tanh(y @ A.T + x)

        x = torch.randn(B, T, M, dtype=torch.float64, requires_grad=True)
        init = torch.randn(B, M, dtype=torch.float64, requires_grad=True)
        A = torch.tensor([[0.3, 0.1], [0.0, 0.2]], dtype=torch.float64, requires_grad=True)
        params = (A,)

    def func_to_check(init_, x_, *p_):
        return newton_solve(step_fn, x_, init_, params=p_)[0]

    assert torch.autograd.gradcheck(
        func_to_check,
        (init, x, *params),
        eps=1e-6,
        atol=1e-4,
    )


@pytest.mark.parametrize("is_scalar", [True, False])
def test_newton_solve_multi_step_gradgradcheck(is_scalar: bool):
    B, T = 2, 3
    torch.manual_seed(0)
    if is_scalar:

        def step_fn(y, x, w):
            return torch.tanh(w * y + x)

        x = torch.randn(B, T, dtype=torch.float64, requires_grad=True)
        init = torch.randn(B, dtype=torch.float64, requires_grad=True)
        w = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
        params = (w,)
    else:
        M = 2

        def step_fn(y, x, A):
            return torch.tanh(y @ A.T + x)

        x = torch.randn(B, T, M, dtype=torch.float64, requires_grad=True)
        init = torch.randn(B, M, dtype=torch.float64, requires_grad=True)
        A = torch.tensor([[0.3, 0.1], [0.0, 0.2]], dtype=torch.float64, requires_grad=True)
        params = (A,)

    def func_to_check(init_, x_, *p_):
        return newton_solve(step_fn, x_, init_, params=p_)[0]

    assert torch.autograd.gradgradcheck(
        func_to_check,
        (init, x, *params),
        eps=1e-6,
        atol=1e-4,
    )


@torch.inference_mode()
def test_newton_solve_inference_mode():
    def func(y, x):
        return y + x

    x = torch.ones(1, 101)
    init = torch.zeros(1)
    y, iters = newton_solve(func, x, init)
    assert iters <= 2
    assert torch.isclose(y[0, -1], torch.tensor(101.0))

    def func_v(y, x):
        return 0.5 * y + x

    x_v = torch.ones(2, 5, 3)
    init_v = torch.zeros(2, 3)
    y_v, iters_v = newton_solve(func_v, x_v, init_v)
    assert iters_v <= 2
    assert y_v.shape == (2, 5, 3)


def test_newton_solve_tolerance_exit_preserves_trajectory():
    def func(y, x):
        return 100 * y + x

    init = torch.zeros(1, dtype=torch.float64)
    x = torch.tensor([[5e-7, 0.0]], dtype=torch.float64)
    y, _ = newton_solve(func, x, init, atol=1e-6, rtol=0.0)
    assert torch.allclose(y, torch.zeros_like(x))
    res = func(torch.cat([init.unsqueeze(1), y[:, :-1]], dim=1), x) - y
    assert (res.abs() <= 1e-6).all()


def test_newton_solve_state_dtype_and_vector_dispatch():
    A = torch.tensor([[0.5, 0.1], [0.0, 0.3]], dtype=torch.float64)

    def func(y, x, A):
        return y @ A.T + x.to(torch.float64)

    x = torch.randn(2, 4, 2, dtype=torch.float32)
    init = torch.randn(2, 2, dtype=torch.float64)
    y, _ = newton_solve(func, x, init, params=(A,))
    assert y.dtype == torch.float64

    def func_2d(y, x):
        return 0.5 * y + x.unsqueeze(-1)

    x_2d = torch.randn(2, 4)
    init_2d = torch.randn(2, 1)
    y_2d, _ = newton_solve(func_2d, x_2d, init_2d)
    assert y_2d.shape == (2, 4, 1)


def test_newton_solve_non_differentiable_inputs():
    def step_fn(y, xs, factor):
        return y + xs[0] * xs[1].float() * factor

    B, T = 2, 4
    x = torch.randn(B, T, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]])
    factor = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    init = torch.zeros(B, dtype=torch.float64, requires_grad=True)

    y, _ = newton_solve(step_fn, (x, mask), init, params=(factor,))
    y.sum().backward()
    assert x.grad is not None
    assert factor.grad is not None
    assert init.grad is not None


def test_newton_solve_tuple_inputs():
    def step_fn(y, xs):
        return torch.tanh(y + xs[0] + 0.5 * xs[1])

    B, T = 2, 4
    x1 = torch.randn(B, T, dtype=torch.float64, requires_grad=True)
    x2 = torch.randn(B, T, dtype=torch.float64, requires_grad=True)
    init = torch.zeros(B, dtype=torch.float64)
    y, _ = newton_solve(step_fn, (x1, x2), init)
    y.sum().backward()
    assert x1.grad is not None and x2.grad is not None

    # Singleton tuple input
    def step_single(y, xs):
        return torch.tanh(y + xs[0])

    x_single = torch.randn(B, T, dtype=torch.float64, requires_grad=True)
    y_s, _ = newton_solve(step_single, (x_single,), init)
    y_s.sum().backward()
    assert x_single.grad is not None


def test_newton_solve_state_independent():
    def step_fn(y, x):
        return x

    B, T = 2, 4
    x = torch.randn(B, T, dtype=torch.float64, requires_grad=True)
    init = torch.randn(B, dtype=torch.float64, requires_grad=True)
    y, iters = newton_solve(step_fn, x, init)
    assert iters <= 2
    assert torch.allclose(y, x)
    y.sum().backward()
    assert x.grad is not None

    # Constant output where out does not require grad
    x_const = torch.ones(B, T)
    init_const = torch.zeros(B)
    y_c, _ = newton_solve(lambda y, x: x, x_const, init_const)
    assert torch.allclose(y_c, x_const)


def test_newton_solve_divergence():
    def step_fn(y, x):
        return torch.full_like(y, float("nan"))

    B, T = 2, 4
    x = torch.randn(B, T, dtype=torch.float64)
    init = torch.randn(B, dtype=torch.float64)
    with pytest.warns(RuntimeWarning, match="non-finite step"):
        y, iters = newton_solve(step_fn, x, init, max_iter=3)
        assert y.shape == (B, T)

    # Overflow in delta during recurrence
    with pytest.warns(RuntimeWarning, match="non-finite step"):
        newton_solve(
            lambda y, x: torch.tanh(y + x),
            x,
            init,
            fprime=lambda y, x: torch.full_like(y, float("inf")),
        )

    # Rejected line-search halvings
    with pytest.warns(RuntimeWarning):
        newton_solve(
            lambda y, x: torch.tanh(y + x),
            x,
            init,
            fprime=lambda y, x: -100.0 * torch.ones_like(y),
            max_halvings=0,
            max_iter=2,
        )

    with pytest.warns(RuntimeWarning, match="did not converge in 1 iterations"):
        y2, iters2 = newton_solve(
            lambda y, x: torch.sin(y) + x,
            torch.randn(B, T, dtype=torch.float64),
            torch.zeros(B, dtype=torch.float64),
            max_iter=1,
            atol=1e-15,
            rtol=1e-15,
        )
        assert iters2 == 1
        assert y2.shape == (B, T)

    # Converges on the final iteration without early break
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, iters3 = newton_solve(
            lambda y, x: 0.5 * y + x,
            torch.ones(B, T, dtype=torch.float64),
            torch.zeros(B, dtype=torch.float64),
            max_iter=1,
        )
        assert iters3 == 1


def test_newton_solve_y0_and_grad_selection():
    def step_fn(y, x):
        return torch.tanh(0.5 * y + x)

    x = torch.randn(2, 4, dtype=torch.float64, requires_grad=False)
    init = torch.randn(2, dtype=torch.float64, requires_grad=True)
    y0 = torch.zeros(2, 4, dtype=torch.float64)
    y, iters = newton_solve(step_fn, x, init, y0=y0)
    assert iters >= 1
    y.sum().backward()
    assert init.grad is not None

    # Backward when only y0 requires grad (needs_init and needs_all are False)
    init_nograd = torch.randn(2, dtype=torch.float64, requires_grad=False)
    y0_grad = torch.zeros(2, 4, dtype=torch.float64, requires_grad=True)
    y_g, _ = newton_solve(step_fn, x, init_nograd, y0=y0_grad)
    y_g.sum().backward()
    assert y0_grad.grad is None


def test_newton_solve_invalid_shapes():
    def step_fn(y, x):
        return y + x

    with pytest.raises(ValueError, match="at least 2 dimensions"):
        newton_solve(step_fn, torch.zeros(4), torch.zeros(4))
    with pytest.raises(ValueError, match="Sequence length T"):
        newton_solve(step_fn, torch.zeros(2, 0), torch.zeros(2))
    with pytest.raises(ValueError, match="matching"):
        newton_solve(step_fn, (torch.zeros(2, 4), torch.zeros(2, 5)), torch.zeros(2))
    with pytest.raises(ValueError, match="Batch size of init"):
        newton_solve(step_fn, torch.zeros(2, 4), torch.zeros(3))
    with pytest.raises(ValueError, match="must be 1-D"):
        newton_solve(step_fn, torch.zeros(2, 4), torch.zeros(2, 3, 4))
    with pytest.raises(ValueError, match="max_iter must be positive"):
        newton_solve(step_fn, torch.zeros(2, 4), torch.zeros(2), max_iter=0)
    with pytest.raises(ValueError, match="max_halvings must be non-negative"):
        newton_solve(step_fn, torch.zeros(2, 4), torch.zeros(2), max_halvings=-1)
    with pytest.raises(ValueError, match="y0 must have shape"):
        newton_solve(step_fn, torch.zeros(2, 4), torch.zeros(2), y0=torch.zeros(2, 5))
