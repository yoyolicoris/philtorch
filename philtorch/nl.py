"""Non-linear recurrent sequence solvers."""

import warnings
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function

from philtorch.lpv.recur import linear_recurrence
from philtorch.lpv.ssm import state_space_recursion

__all__ = ["newton_solve"]


def _jacobian(
    func: Callable[..., Tensor],
    y: Tensor,
    x: Tensor | tuple[Tensor, ...],
    params: tuple[Tensor, ...],
    fprime: Callable[..., Tensor] | None = None,
) -> Tensor:
    if fprime is not None:
        return fprime(y, x, *params)
    if y.dim() == 2:
        with torch.enable_grad():
            y_in = y.detach().requires_grad_(True)
            out = func(y_in, x, *params)
            (jac,) = torch.autograd.grad(
                out, y_in, grad_outputs=torch.ones_like(out), create_graph=True
            )
            return jac
    return torch.autograd.functional.jacobian(
        lambda y_: func(y_, x, *params).sum((0, 1)),
        y,
        create_graph=True,
        vectorize=True,
    ).permute(1, 2, 0, 3)


def _newton_solve(
    func: Callable[..., Tensor],
    x: Tensor | tuple[Tensor, ...],
    init: Tensor,
    *,
    params: tuple[Tensor, ...] = (),
    fprime: Callable[..., Tensor] | None = None,
    y0: Tensor | None = None,
    max_iter: int = 100,
    max_halvings: int = 8,
    atol: float | None = None,
    rtol: float | None = None,
    recur_kwargs: dict | None = None,
) -> tuple[Tensor, int]:
    recur_kwargs = recur_kwargs or {}
    x_prim = x[0] if isinstance(x, tuple) else x
    B, T = x_prim.size(0), x_prim.size(1)
    eps = torch.finfo(x_prim.dtype).eps
    atol_eff = eps if atol is None else atol
    rtol_eff = 8.0 * eps if rtol is None else rtol

    is_scalar = (init.dim() == 1) or (init.dim() == 2 and init.size(-1) == 1 and x_prim.dim() == 2)
    recur_fn = linear_recurrence if is_scalar else state_space_recursion

    init_in = init.squeeze(-1) if (is_scalar and init.dim() == 2) else init
    y = x_prim.new_empty(B, T + 1, *init_in.shape[1:])
    y[:, 0] = init_in
    y[:, 1:] = (
        init_in.unsqueeze(1).expand(B, T, *init_in.shape[1:])
        if y0 is None
        else (y0.squeeze(-1) if (is_scalar and y0.dim() == 3) else y0)
    )

    slice_x = (
        (lambda s: tuple(xi[:, s:] for xi in x)) if isinstance(x, tuple) else (lambda s: x[:, s:])
    )

    for i in range(max_iter):
        y_prev = y[:, i:-1]
        x_curr = slice_x(i)
        next_y = func(y_prev, x_curr, *params)
        res = next_y - y[:, i + 1 :]
        y[:, i + 1] = next_y[:, 0]

        if i + 2 > T:
            break

        if (res.abs() <= atol_eff + rtol_eff * y[:, i + 1 :].abs()).all():
            break

        jac = _jacobian(func, y_prev, x_curr, params, fprime)
        delta = recur_fn(jac[:, 1:], res[:, 0], res[:, 1:], **recur_kwargs)

        if not torch.isfinite(delta).all():
            warnings.warn(
                "newton_solve diverged (non-finite step or residual); "
                "the returned state is set to NaN",
                RuntimeWarning,
                stacklevel=2,
            )
            y[:, 1:] = float("nan")
            break

        base = y[:, i + 2 :].clone()
        merit = torch.linalg.vector_norm(res[:, 1:])
        t = 1.0
        for _ in range(max_halvings + 1):
            cand = base + t * delta
            y[:, i + 2 :] = cand
            cand_res = func(y[:, i + 1 : -1], slice_x(i + 1), *params) - cand
            if (
                torch.linalg.vector_norm(cand_res) < merit
                or (cand_res.abs() <= atol_eff + rtol_eff * cand.abs()).all()
            ):
                break
            t /= 2.0
    else:
        warnings.warn(
            f"newton_solve diverged (did not converge in {max_iter} iterations); "
            "the returned state is set to NaN",
            RuntimeWarning,
            stacklevel=2,
        )
        y[:, 1:] = float("nan")

    return y[:, 1:], i + 1


class _NewtonSolve(Function):
    @staticmethod
    def forward(
        func,
        fprime,
        max_iter,
        max_halvings,
        atol,
        rtol,
        y0,
        recur_kwargs,
        num_x,
        init,
        *all_tensors,
    ):
        x = all_tensors[0] if num_x == 1 else all_tensors[:num_x]
        return _newton_solve(
            func,
            x,
            init,
            params=all_tensors[num_x:],
            fprime=fprime,
            y0=y0,
            max_iter=max_iter,
            max_halvings=max_halvings,
            atol=atol,
            rtol=rtol,
            recur_kwargs=recur_kwargs,
        )

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        (
            func,
            fprime,
            max_iter,
            max_halvings,
            atol,
            rtol,
            y0,
            recur_kwargs,
            num_x,
            init,
            *all_tensors,
        ) = inputs
        y, _ = outputs

        ctx.set_materialize_grads(False)
        ctx.func = func
        ctx.fprime = fprime
        ctx.recur_kwargs = recur_kwargs
        ctx.num_x = num_x
        ctx.save_for_backward(y, init, *all_tensors)

    @staticmethod
    def backward(ctx, grad_y, _):
        if grad_y is None:
            return (None,) * (10 + len(ctx.saved_tensors) - 2)

        needs_init = ctx.needs_input_grad[9]
        needs_all = ctx.needs_input_grad[10 : 10 + len(ctx.saved_tensors) - 2]
        if not (needs_init or any(needs_all)):
            return (None,) * (10 + len(needs_all))

        y, init, *all_tensors = ctx.saved_tensors
        num_x = ctx.num_x
        x = all_tensors[0] if num_x == 1 else all_tensors[:num_x]
        params = all_tensors[num_x:]
        x_prim = all_tensors[0]
        B = x_prim.size(0)

        is_scalar = (init.dim() == 1) or (
            init.dim() == 2 and init.size(-1) == 1 and x_prim.dim() == 2
        )
        init_in = init.squeeze(-1) if (is_scalar and init.dim() == 2) else init
        y_prev = torch.cat([init_in.unsqueeze(1), y[:, :-1]], dim=1)

        jac = _jacobian(ctx.func, y_prev, x, params, ctx.fprime)

        g_rev = grad_y.flip(1)
        if is_scalar:
            a_rev = F.pad(jac.flip(1)[:, :-1], (1, 0))
            lam_rev = linear_recurrence(a_rev, g_rev.new_zeros(B), g_rev, **ctx.recur_kwargs)
        else:
            jac_rev = jac.flip(1).transpose(-1, -2)
            M = init.size(-1)
            a_rev = torch.cat([jac_rev.new_zeros(B, 1, M, M), jac_rev[:, :-1]], dim=1)
            lam_rev = state_space_recursion(a_rev, g_rev.new_zeros(B, M), g_rev, **ctx.recur_kwargs)
        lam = lam_rev.flip(1)

        if needs_init:
            if is_scalar:
                grad_init = (lam[:, 0] * jac[:, 0]).reshape_as(init)
            else:
                grad_init = (lam[:, 0].unsqueeze(1) @ jac[:, 0]).squeeze(1).reshape_as(init)
        else:
            grad_init = None

        if any(needs_all):
            diff_tensors = [
                t.detach().requires_grad_(True) if needs else t
                for t, needs in zip(all_tensors, needs_all)
            ]
            diff_inputs = [t for t, needs in zip(diff_tensors, needs_all) if needs]
            x_diff = diff_tensors[0] if num_x == 1 else tuple(diff_tensors[:num_x])
            params_diff = tuple(diff_tensors[num_x:])

            with torch.enable_grad():
                out = ctx.func(y_prev, x_diff, *params_diff)
                grads = torch.autograd.grad(
                    out,
                    diff_inputs,
                    grad_outputs=lam,
                    create_graph=True,
                    allow_unused=True,
                )
            grad_iter = iter(grads)
            grad_all = [next(grad_iter) if needs_all[k] else None for k in range(len(all_tensors))]
        else:
            grad_all = [None] * len(all_tensors)

        return (None,) * 9 + (grad_init,) + tuple(grad_all)


def newton_solve(
    func: Callable[..., Tensor],
    x: Tensor | tuple[Tensor, ...],
    init: Tensor,
    *,
    params: tuple[Tensor, ...] = (),
    fprime: Callable[..., Tensor] | None = None,
    y0: Tensor | None = None,
    max_iter: int = 100,
    max_halvings: int = 8,
    atol: float | None = None,
    rtol: float | None = None,
    **recur_kwargs,
) -> tuple[Tensor, int]:
    r"""Solve for the state trajectory of a non-linear recurrence using Newton's method.

    Computes the solution of the non-linear sequence recursion

    .. math::
        \mathbf{y}[t] = f(\mathbf{y}[t-1], \mathbf{x}[t]; \theta)

    in parallel across time using damped Newton-Raphson iterations.

    Args:
        func (Callable): step transition function with signature
            ``func(y_prev, x_curr, *params) -> Tensor``.
        x (Tensor or tuple of Tensor): time-varying inputs of shape
            :math:`(B, T, *)` or :math:`(B, T)`.
        init (Tensor): boundary condition :math:`\mathbf{y}[-1]` before the
            sequence starts, of shape :math:`(B, M)` or :math:`(B,)`.
        params (tuple of Tensor, optional): time-invariant parameters
            :math:`\theta`. Default: ``()``.
        fprime (Callable, optional): analytical Jacobian of :attr:`func` w.r.t.
            ``y_prev``. Default: ``None``.
        y0 (Tensor, optional): initial guess for the trajectory.
            Default: ``None``.
        max_iter (int, optional): maximum Newton iterations. Default: 100.
        max_halvings (int, optional): maximum line search step halvings.
            Default: 8.
        atol (float, optional): absolute tolerance. Default: ``1.0 * eps``.
        rtol (float, optional): relative tolerance. Default: ``8.0 * eps``.
        **recur_kwargs: keyword arguments (e.g. ``unroll_factor``) passed to
            the underlying linear recurrence.

    Returns:
        tuple of (Tensor, int): the solved trajectory of shape
        :math:`(B, T, M)` or :math:`(B, T)`, and the number of iterations.

    Raises:
        ValueError: if input shapes do not match.

    Note:
        This algorithm is based on ParaRNN:

        Botev et al., *ParaRNN: Unlocking Parallel Training of Nonlinear RNNs
        for Large Language Models*, 2024.

    Example::

        >>> from philtorch.nl import newton_solve
        >>> step_fn = lambda y, x: torch.tanh(0.5 * y + x)
        >>> x = torch.zeros(2, 4)
        >>> init = torch.zeros(2)
        >>> y, iters = newton_solve(step_fn, x, init)
        >>> y.shape
        torch.Size([2, 4])
    """
    x_tensors = x if isinstance(x, tuple) else (x,)
    for xi in x_tensors:
        if xi.dim() < 2:
            raise ValueError(
                f"Input tensor x must have at least 2 dimensions (batch, time), got {xi.shape}"
            )
        if xi.size(0) != x_tensors[0].size(0) or xi.size(1) != x_tensors[0].size(1):
            raise ValueError(
                "All input tensors in x must have matching (batch, time) dimensions, "
                f"got {xi.shape} vs {x_tensors[0].shape}"
            )

    if init.size(0) != x_tensors[0].size(0):
        raise ValueError(
            f"Batch size of init ({init.size(0)}) must match x ({x_tensors[0].size(0)})"
        )

    all_tensors = x_tensors + tuple(params)
    return _NewtonSolve.apply(
        func,
        fprime,
        max_iter,
        max_halvings,
        atol,
        rtol,
        y0,
        recur_kwargs,
        len(x_tensors),
        init,
        *all_tensors,
    )
