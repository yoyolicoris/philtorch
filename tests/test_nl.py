import warnings

import pytest
import torch

from philtorch.nl import newton_solve


def _sequential(step_fn, x, init, *params):
    xs = x if isinstance(x, tuple) else (x,)
    y = []
    curr = init
    for t in range(xs[0].size(1)):
        x_t = tuple(xi[:, t : t + 1] for xi in xs) if isinstance(x, tuple) else x[:, t : t + 1]
        curr = step_fn(curr.unsqueeze(1), x_t, *params).squeeze(1)
        y.append(curr)
    return torch.stack(y, 1)


def _tanh_problem(is_scalar, requires_grad=False, T=8):
    B, M = 2, 2
    torch.manual_seed(0)
    if is_scalar:

        def step_fn(y, x, w):
            return torch.tanh(w * y + x)

        def fprime(y, x, w):
            return (1.0 - step_fn(y, x, w).square()) * w

        x = torch.randn(B, T, dtype=torch.float64)
        init = torch.randn(B, dtype=torch.float64)
        param = torch.tensor(0.5, dtype=torch.float64)
    else:

        def step_fn(y, x, A):
            return torch.tanh(y @ A.T + x)

        def fprime(y, x, A):
            return (1.0 - step_fn(y, x, A).square()).unsqueeze(-1) * A

        x = torch.randn(B, T, M, dtype=torch.float64)
        init = torch.randn(B, M, dtype=torch.float64)
        param = torch.tensor([[0.3, 0.1], [0.0, 0.2]], dtype=torch.float64)
    x.requires_grad_(requires_grad)
    init.requires_grad_(requires_grad)
    param.requires_grad_(requires_grad)
    return step_fn, fprime, x, init, (param,)


@pytest.mark.parametrize("is_scalar", [True, False])
@pytest.mark.parametrize("use_fprime", [False, True])
def test_newton_solve_forward_against_sequential_ref(is_scalar: bool, use_fprime: bool):
    step_fn, fprime, x, init, params = _tanh_problem(is_scalar)
    y, _ = newton_solve(step_fn, x, init, params=params, fprime=fprime if use_fprime else None)
    assert torch.allclose(y, _sequential(step_fn, x, init, *params), atol=1e-10, rtol=1e-10)


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

    x = torch.randn(*shape, dtype=torch.float64)
    init = torch.randn(*shape[:1], *shape[2:], dtype=torch.float64)
    x[1, 2] = float("nan")
    y_ref = _sequential(step_fn, x, init, A)
    with pytest.warns(RuntimeWarning, match="non-finite step or residual") as w:
        y, iters = newton_solve(step_fn, x, init, params=(A,))
    assert len(w) == 1
    assert "1 of 3 batch elements" in str(w[0].message)
    for b in (0, 2):
        assert torch.allclose(y[b], y_ref[b], atol=1e-10, rtol=1e-10)
    assert iters[1] == 0 and (iters[[0, 2]] > 0).all()
    # The diverged element stops updating: it keeps its initial guess
    assert torch.equal(y[1], init[1].expand(T, *init.shape[1:]))


@pytest.mark.parametrize("is_scalar", [True, False])
def test_newton_solve_gradcheck(is_scalar: bool):
    step_fn, _, x, init, params = _tanh_problem(is_scalar, requires_grad=True, T=3)

    def func_to_check(init_, x_, *p_):
        return newton_solve(step_fn, x_, init_, params=p_)[0]

    args = (init, x, *params)
    assert torch.autograd.gradcheck(func_to_check, args, eps=1e-6, atol=1e-4)
    assert torch.autograd.gradgradcheck(func_to_check, args, eps=1e-6, atol=1e-4)


def test_newton_solve_gradient_to_init_only():
    step_fn, _, x, init, params = _tanh_problem(True, T=4)
    init.requires_grad_()
    assert torch.autograd.gradcheck(
        lambda i: newton_solve(step_fn, x, i, params=params)[0], (init,), eps=1e-6, atol=1e-4
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
    y, iters = newton_solve(func, x, init, atol=1e-6, rtol=0.0)
    assert iters.tolist() == [0]
    assert torch.equal(y, torch.zeros_like(x))


def test_newton_solve_state_size_one():
    def func(y, x):
        return 0.5 * y + x.unsqueeze(-1)

    x = torch.randn(2, 4)
    init = torch.randn(2, 1)
    y, _ = newton_solve(func, x, init)
    assert torch.allclose(y, _sequential(func, x, init), atol=1e-6)


def test_newton_solve_tuple_inputs():
    B, T = 2, 4
    torch.manual_seed(0)
    x = torch.randn(B, T, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]])
    init = torch.zeros(B, dtype=torch.float64)

    def step_masked(y, xs):
        return torch.tanh(y + 2.0 * xs[0] * xs[1])

    def step_single(y, xs):
        return torch.tanh(y + xs[0])

    assert torch.autograd.gradcheck(
        lambda a: newton_solve(step_masked, (a, mask), init)[0], (x,), eps=1e-6, atol=1e-4
    )
    assert torch.autograd.gradcheck(
        lambda a: newton_solve(step_single, (a,), init)[0], (x,), eps=1e-6, atol=1e-4
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
    assert torch.equal(x.grad, torch.ones_like(x))

    # A constant step has no gradient with respect to the state
    x_const = torch.ones(B, T)
    y_c, _ = newton_solve(lambda y, x: x, x_const, torch.zeros(B))
    assert torch.allclose(y_c, x_const)


def test_newton_solve_divergence():
    B, T = 2, 4
    torch.manual_seed(0)
    x = torch.randn(B, T, dtype=torch.float64)
    init = torch.randn(B, dtype=torch.float64)
    with pytest.warns(RuntimeWarning, match="non-finite step") as w:
        _, iters = newton_solve(lambda y, x: torch.full_like(y, float("nan")), x, init)
    assert iters.tolist() == [0, 0]
    assert w[0].filename == __file__

    with pytest.warns(RuntimeWarning, match="non-finite step"):
        newton_solve(
            lambda y, x: torch.tanh(y + x),
            x,
            init,
            fprime=lambda y, x: torch.full_like(y, float("inf")),
        )


def test_newton_solve_max_iter():
    B, T = 2, 4
    torch.manual_seed(0)
    init = torch.zeros(B, dtype=torch.float64)
    with pytest.warns(RuntimeWarning, match="did not converge in 1 iterations") as w:
        newton_solve(
            lambda y, x: torch.sin(y) + x,
            torch.randn(B, T, dtype=torch.float64),
            init,
            max_iter=1,
            atol=1e-15,
            rtol=1e-15,
        )
    assert w[0].filename == __file__

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, iters = newton_solve(
            lambda y, x: 0.5 * y + x, torch.ones(B, T, dtype=torch.float64), init, max_iter=1
        )
    assert iters.tolist() == [1, 1]


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


def test_newton_solve_no_gradient_to_y0():
    def step_fn(y, x):
        return torch.tanh(0.5 * y + x)

    x = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    init = torch.randn(2, dtype=torch.float64)
    y0 = torch.zeros(2, 4, dtype=torch.float64, requires_grad=True)
    y, _ = newton_solve(step_fn, x, init, y0=y0)
    y.sum().backward()
    assert x.grad is not None and y0.grad is None


def test_newton_solve_invalid_arguments():
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
    with pytest.raises(ValueError, match="params must be a tuple of Tensors"):
        newton_solve(step_fn, torch.zeros(2, 4), torch.zeros(2), params=(0.5,))


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


def _per_element_problem():
    # A and g differ per batch element, so they travel in x, expanded over time
    B, T, M = 3, 10, 2
    torch.manual_seed(0)

    def step_fn(y, xs, c):
        x1, x2, A_t, g_t = xs
        Ay = (y.unsqueeze(-2) @ A_t.transpose(-1, -2)).squeeze(-2)
        return Ay + g_t.unsqueeze(-1) * torch.sin(y) + x1 + 0.5 * x2 + c

    def expand(x1, x2, A, g):
        return x1, x2, A.unsqueeze(1).expand(B, T, M, M), g.unsqueeze(1).expand(B, T)

    x = tuple(torch.randn(B, T, M, dtype=torch.float64, requires_grad=True) for _ in range(2))
    init = torch.randn(B, M, dtype=torch.float64, requires_grad=True)
    A = (0.3 * torch.randn(B, M, M, dtype=torch.float64)).requires_grad_()
    g = torch.tensor([0.0, 0.3, 1.0], dtype=torch.float64, requires_grad=True)
    c = torch.tensor(0.1, dtype=torch.float64, requires_grad=True)
    return step_fn, expand, x, init, A, g, c


def test_newton_solve_per_element_inputs_and_counts():
    step_fn, expand, x, init, A, g, c = _per_element_problem()
    with torch.no_grad():
        xs = expand(*x, A, g)
        y_ref = _sequential(step_fn, xs, init, c)
        y, iters = newton_solve(step_fn, xs, init, params=(c,))
    assert torch.allclose(y, y_ref, atol=1e-10, rtol=1e-10)
    assert iters.dtype == torch.long and not iters.requires_grad
    assert iters[0] == 1
    assert 1 < iters[1] < iters[2]


def test_newton_solve_per_element_inputs_gradcheck():
    step_fn, expand, x, init, A, g, c = _per_element_problem()

    def func_to_check(init_, x1, x2, A_, g_, c_):
        return newton_solve(step_fn, expand(x1, x2, A_, g_), init_, params=(c_,))[0]

    args = (init, *x, A, g, c)
    assert torch.autograd.gradcheck(func_to_check, args, eps=1e-6, atol=1e-4)


def test_newton_solve_one_step_matches_newton():
    # One step linearizes f about the initial guess: delta[t] = J[t] delta[t-1] + r[t]
    B, T = 2, 6
    torch.manual_seed(0)

    def step_fn(y, x):
        return torch.tanh(2.0 * y + x)

    x = torch.randn(B, T, dtype=torch.float64)
    init = torch.randn(B, dtype=torch.float64)
    # Near the solution, so that the full step is accepted
    y0 = _sequential(step_fn, x, init) + 0.05 * torch.randn(B, T, dtype=torch.float64)
    y_prev = torch.cat([init[:, None], y0[:, :-1]], 1)
    r = step_fn(y_prev, x) - y0
    jac = 2.0 * (1.0 - step_fn(y_prev, x).square())
    delta = torch.empty_like(r)
    delta[:, 0] = r[:, 0]
    for t in range(1, T):
        delta[:, t] = jac[:, t] * delta[:, t - 1] + r[:, t]
    with pytest.warns(RuntimeWarning, match="did not converge in 1 iterations"):
        y, _ = newton_solve(step_fn, x, init, y0=y0, max_iter=1, max_halvings=0)
    assert torch.allclose(y, y0 + delta, atol=1e-12, rtol=1e-12)
