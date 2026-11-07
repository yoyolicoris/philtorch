import warnings

import pytest
import torch

from philtorch.nl import newton_solve


def _sequential(step_fn, x, init, *params):
    y = torch.empty_like(x)
    curr = init
    for t in range(x.size(1)):
        curr = step_fn(curr, x[:, t], *params)
        y[:, t] = curr
    return y


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

    y_sol, _ = newton_solve(step_fn, x, init, params=params, fprime=fprime)
    assert torch.allclose(y_sol, _sequential(step_fn, x, init, *params), atol=1e-10, rtol=1e-10)


def test_newton_solve_linear_system_convergence():
    B, T, M = 2, 6, 2
    torch.manual_seed(0)
    A = torch.tensor([[0.4, 0.1], [0.0, 0.3]], dtype=torch.float64)

    def step_fn(y, x, A):
        return y @ A.T + x

    def fprime(y, x, A):
        return A.expand(*y.shape[:2], M, M)

    x = torch.randn(B, T, M, dtype=torch.float64)
    init = torch.randn(B, M, dtype=torch.float64)
    y, iters = newton_solve(step_fn, x, init, params=(A,), fprime=fprime)
    assert iters.tolist() == [1, 1]
    assert torch.allclose(y, _sequential(step_fn, x, init, A), atol=1e-12, rtol=1e-12)


def test_newton_solve_exact_initial_guess():
    torch.manual_seed(0)

    def step_fn(y, x):
        return torch.tanh(0.5 * y + x)

    x = torch.randn(2, 5, dtype=torch.float64)
    init = torch.randn(2, dtype=torch.float64)
    y_ref = _sequential(step_fn, x, init)
    y, iters = newton_solve(step_fn, x, init, y0=y_ref)
    assert iters.tolist() == [0, 0]
    assert torch.equal(y, y_ref)


@pytest.mark.parametrize("is_scalar", [True, False])
def test_newton_solve_batch_independence(is_scalar: bool):
    B, T = 3, 6
    torch.manual_seed(0)
    shape = (B, T) if is_scalar else (B, T, 2)
    A = torch.tensor(0.5 if is_scalar else [[0.3, 0.1], [0.0, 0.2]], dtype=torch.float64)

    def step_fn(y, x, A):
        return torch.tanh((A * y if is_scalar else y @ A.T) + x)

    scale = torch.tensor([0.1, 1.0, 3.0], dtype=torch.float64).view(B, 1, *[1] * (len(shape) - 2))
    x = torch.randn(*shape, dtype=torch.float64) * scale
    init = torch.randn(*shape[:1], *shape[2:], dtype=torch.float64)
    x[1, 2] = float("nan")
    y_ref = _sequential(step_fn, x, init, A)
    with pytest.warns(RuntimeWarning, match="non-finite step or residual") as w:
        y, iters = newton_solve(step_fn, x, init, params=(A,))
    assert w[0].filename == __file__
    assert "1 of 3 batch elements" in str(w[0].message)
    assert len(w) == 1
    for b in (0, 2):
        assert torch.allclose(y[b], y_ref[b], atol=1e-10, rtol=1e-10)
    assert iters[1] == 0 and (iters[[0, 2]] > 0).all()
    # The diverged element stops updating: it keeps its initial guess
    assert torch.equal(y[1], init[1].expand(T, *init.shape[1:]))


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
    assert iters.tolist() == [1]
    assert torch.isclose(y[0, -1], torch.tensor(101.0))

    def func_v(y, x):
        return 0.5 * y + x

    x_v = torch.ones(2, 5, 3)
    init_v = torch.zeros(2, 3)
    y_v, iters_v = newton_solve(func_v, x_v, init_v)
    assert iters_v.tolist() == [1, 1]
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
        return torch.tanh(y + xs[0] * xs[1].float() * factor)

    B, T = 2, 4
    torch.manual_seed(0)
    x = torch.randn(B, T, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]])
    factor = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    init = torch.zeros(B, dtype=torch.float64, requires_grad=True)

    def func_to_check(init_, x_, factor_):
        return newton_solve(step_fn, (x_, mask), init_, params=(factor_,))[0]

    assert torch.autograd.gradcheck(func_to_check, (init, x, factor), eps=1e-6, atol=1e-4)


def test_newton_solve_tuple_inputs():
    def step_fn(y, xs):
        return torch.tanh(y + xs[0] + 0.5 * xs[1])

    B, T = 2, 4
    torch.manual_seed(0)
    x1 = torch.randn(B, T, dtype=torch.float64, requires_grad=True)
    x2 = torch.randn(B, T, dtype=torch.float64, requires_grad=True)
    init = torch.zeros(B, dtype=torch.float64)
    assert torch.autograd.gradcheck(
        lambda a, b: newton_solve(step_fn, (a, b), init)[0], (x1, x2), eps=1e-6, atol=1e-4
    )

    # Singleton tuple input
    def step_single(y, xs):
        return torch.tanh(y + xs[0])

    x_single = torch.randn(B, T, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda a: newton_solve(step_single, (a,), init)[0], (x_single,), eps=1e-6, atol=1e-4
    )


def test_newton_solve_state_independent():
    def step_fn(y, x):
        return x

    B, T = 2, 4
    x = torch.randn(B, T, dtype=torch.float64, requires_grad=True)
    init = torch.randn(B, dtype=torch.float64, requires_grad=True)
    y, iters = newton_solve(step_fn, x, init)
    assert iters.tolist() == [1, 1]
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
    torch.manual_seed(0)
    x = torch.randn(B, T, dtype=torch.float64)
    init = torch.randn(B, dtype=torch.float64)
    with pytest.warns(RuntimeWarning, match="non-finite step") as w:
        y, iters = newton_solve(step_fn, x, init, max_iter=3)
        assert y.shape == (B, T)
    assert iters.tolist() == [0, 0]
    assert w[0].filename == __file__

    # Overflow in delta during recurrence
    with pytest.warns(RuntimeWarning, match="non-finite step"):
        newton_solve(
            lambda y, x: torch.tanh(y + x),
            x,
            init,
            fprime=lambda y, x: torch.full_like(y, float("inf")),
        )

    with pytest.warns(RuntimeWarning, match="did not converge in 1 iterations") as w:
        y2, iters2 = newton_solve(
            lambda y, x: torch.sin(y) + x,
            torch.randn(B, T, dtype=torch.float64),
            torch.zeros(B, dtype=torch.float64),
            max_iter=1,
            atol=1e-15,
            rtol=1e-15,
        )
        assert iters2.tolist() == [1, 1]
        assert y2.shape == (B, T)
    assert w[0].filename == __file__

    # Converges on the final iteration without early break
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, iters3 = newton_solve(
            lambda y, x: 0.5 * y + x,
            torch.ones(B, T, dtype=torch.float64),
            torch.zeros(B, dtype=torch.float64),
            max_iter=1,
        )
        assert iters3.tolist() == [1, 1]


def test_newton_solve_rejected_line_search():
    B, T = 2, 4
    torch.manual_seed(0)

    def step_fn(y, x):
        return torch.tanh(y + x)

    def fprime(y, x):
        return -100.0 * torch.ones_like(y)

    x = torch.randn(B, T, dtype=torch.float64)
    init = torch.randn(B, dtype=torch.float64)
    y_ref = _sequential(step_fn, x, init)
    kwargs = dict(fprime=fprime, max_halvings=0)
    # Every step is rejected: only the first positions, one per iteration, are exact
    with pytest.warns(RuntimeWarning, match="did not converge in 2 iterations"):
        y, iters = newton_solve(step_fn, x, init, max_iter=2, **kwargs)
    assert iters.tolist() == [2, 2]
    assert torch.allclose(y[:, :2], y_ref[:, :2])
    assert torch.equal(y[:, 2:], init.unsqueeze(1).expand(B, T - 2))
    y, iters = newton_solve(step_fn, x, init, max_iter=T, **kwargs)
    assert iters.tolist() == [T, T]
    assert torch.allclose(y, y_ref)


def test_newton_solve_y0_and_grad_selection():
    torch.manual_seed(0)

    def step_fn(y, x):
        return torch.tanh(0.5 * y + x)

    x = torch.randn(2, 4, dtype=torch.float64, requires_grad=False)
    init = torch.randn(2, dtype=torch.float64, requires_grad=True)
    y0 = torch.zeros(2, 4, dtype=torch.float64)
    y, iters = newton_solve(step_fn, x, init, y0=y0)
    assert (iters >= 1).all()
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


def test_newton_solve_rtol_scales_with_state_size():
    B, T, M = 2, 2000, 32
    torch.manual_seed(0)
    W = torch.randn(M, M) / M**0.5

    def step_fn(y, x):
        return torch.tanh(y @ W.T + x)

    x = torch.randn(B, T, M)
    init = torch.randn(B, M)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        y, iters = newton_solve(step_fn, x, init)
    assert (iters < 100).all()
    assert torch.allclose(y, _sequential(step_fn, x, init), atol=1e-5)


def test_newton_solve_invalid_params():
    with pytest.raises(ValueError, match="params must be a tuple of Tensors"):
        newton_solve(lambda y, x, w: y + w * x, torch.zeros(2, 4), torch.zeros(2), params=(0.5,))


def _batched_param_problem():
    B, T, M = 3, 10, 2
    torch.manual_seed(0)

    def step_fn(y, xs, A, g, c):
        return y @ A.transpose(-1, -2) + g.view(-1, 1, 1) * torch.sin(y) + xs[0] + 0.5 * xs[1] + c

    x = tuple(torch.randn(B, T, M, dtype=torch.float64, requires_grad=True) for _ in range(2))
    init = torch.randn(B, M, dtype=torch.float64, requires_grad=True)
    A = (0.3 * torch.randn(B, M, M, dtype=torch.float64)).requires_grad_()
    g = torch.tensor([0.0, 0.3, 1.0], dtype=torch.float64, requires_grad=True)
    c = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    return step_fn, x, init, (A, g, c)


def test_newton_solve_batched_params_and_counts():
    step_fn, x, init, params = _batched_param_problem()
    with torch.no_grad():
        y_ref = torch.empty_like(x[0])
        curr = init
        for t in range(x[0].size(1)):
            curr = step_fn(curr.unsqueeze(1), tuple(xi[:, t : t + 1] for xi in x), *params)[:, 0]
            y_ref[:, t] = curr
        y, iters = newton_solve(step_fn, x, init, params=params)
    assert torch.allclose(y, y_ref, atol=1e-10, rtol=1e-10)
    assert iters.dtype == torch.long and not iters.requires_grad
    assert iters[0] == 1
    assert 1 < iters[1] < iters[2]


def test_newton_solve_batched_params_gradcheck():
    step_fn, x, init, params = _batched_param_problem()

    def func_to_check(init_, x1, x2, *p_):
        return newton_solve(step_fn, (x1, x2), init_, params=p_)[0]

    assert torch.autograd.gradcheck(func_to_check, (init, *x, *params), eps=1e-6, atol=1e-4)
