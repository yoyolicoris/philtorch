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


def _slice_x(x: Tensor | tuple[Tensor, ...], start: int) -> Tensor | tuple[Tensor, ...]:
    if isinstance(x, tuple):
        return tuple(xi[:, start:] for xi in x)
    return x[:, start:]


def _jacobian(
    func: Callable[..., Tensor],
    y: Tensor,
    x: Tensor | tuple[Tensor, ...],
    params: tuple[Tensor, ...],
    is_scalar: bool,
    fprime: Callable[..., Tensor] | None = None,
) -> Tensor:
    if fprime is not None:
        return fprime(y, x, *params)
    if is_scalar:
        with torch.enable_grad():
            y_in = y.detach().requires_grad_(True)
            out = func(y_in, x, *params)
            (jac,) = torch.autograd.grad(
                out,
                y_in,
                grad_outputs=torch.ones_like(out),
                create_graph=False,
            )
            return jac
    else:

        def func_sum(y_: Tensor) -> Tensor:
            return func(y_, x, *params).sum((0, 1))

        return torch.autograd.functional.jacobian(
            func_sum,
            y,
            create_graph=False,
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
    dtype = x_prim.dtype
    eps = torch.finfo(dtype).eps
    atol_eff = 1.0 * eps if atol is None else atol
    rtol_eff = 8.0 * eps if rtol is None else rtol

    is_scalar = (init.dim() == 1) or (init.dim() == 2 and init.size(-1) == 1 and x_prim.dim() == 2)

    if is_scalar:
        y = x_prim.new_empty(B, T + 1)
        y[:, 0] = init.squeeze(-1) if init.dim() == 2 else init
        if y0 is None:
            y[:, 1:] = y[:, :1].expand(B, T)
        else:
            y[:, 1:] = y0.squeeze(-1) if y0.dim() == 3 else y0
    else:
        M = init.size(-1)
        y = x_prim.new_empty(B, T + 1, M)
        y[:, 0] = init
        if y0 is None:
            y[:, 1:] = init.unsqueeze(1).expand(B, T, M)
        else:
            y[:, 1:] = y0

    for i in range(max_iter):
        y_prev = y[:, i:-1]
        x_curr = _slice_x(x, i)
        next_y = func(y_prev, x_curr, *params)
        res = next_y - y[:, i + 1 :]
        y[:, i + 1] = next_y[:, 0]

        if i + 2 > T:
            break

        converged = (res.abs() <= atol_eff + rtol_eff * y[:, i + 1 :].abs()).all()
        if converged:
            break

        jac = _jacobian(func, y_prev, x_curr, params, is_scalar, fprime)

        if is_scalar:
            delta = linear_recurrence(jac[:, 1:], res[:, 0], res[:, 1:], **recur_kwargs)
        else:
            delta = state_space_recursion(jac[:, 1:], res[:, 0], res[:, 1:], **recur_kwargs)

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
            cand_next_y = func(y[:, i + 1 : -1], _slice_x(x, i + 1), *params)
            cand_res = cand_next_y - cand
            cand_norm = torch.linalg.vector_norm(cand_res)
            cand_conv = (cand_res.abs() <= atol_eff + rtol_eff * cand.abs()).all()
            if cand_norm < merit or cand_conv:
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
        x_tensors = all_tensors[:num_x]
        params = all_tensors[num_x:]
        x = x_tensors[0] if num_x == 1 else x_tensors

        with torch.no_grad():
            y, iters = _newton_solve(
                func,
                x,
                init,
                params=params,
                fprime=fprime,
                y0=y0,
                max_iter=max_iter,
                max_halvings=max_halvings,
                atol=atol,
                rtol=rtol,
                recur_kwargs=recur_kwargs,
            )

        return y, iters

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

        y, init, *all_tensors = ctx.saved_tensors
        num_x = ctx.num_x
        x_tensors = all_tensors[:num_x]
        params = all_tensors[num_x:]
        x = x_tensors[0] if num_x == 1 else x_tensors
        func = ctx.func
        fprime = ctx.fprime
        recur_kwargs = ctx.recur_kwargs

        x_prim = x_tensors[0]
        B = x_prim.size(0)
        is_scalar = (init.dim() == 1) or (
            init.dim() == 2 and init.size(-1) == 1 and x_prim.dim() == 2
        )

        init_in = init.squeeze(-1) if (is_scalar and init.dim() == 2) else init
        init_prev = init_in.unsqueeze(1)
        y_prev_detached = y[:, :-1].detach()
        y_prev = torch.cat([init_prev, y_prev_detached], dim=1)

        jac = _jacobian(func, y_prev, x, params, is_scalar, fprime)

        if is_scalar:
            jac_rev = jac.flip(1)
            g_rev = grad_y.flip(1)
            a_rev = F.pad(jac_rev[:, :-1], (1, 0))
            lam_rev = linear_recurrence(a_rev, g_rev.new_zeros(B), g_rev, **recur_kwargs)
            lam = lam_rev.flip(1)
        else:
            jac_rev = jac.flip(1).transpose(-1, -2)
            g_rev = grad_y.flip(1)
            M = init.size(-1)
            a_rev = torch.cat([jac_rev.new_zeros(B, 1, M, M), jac_rev[:, :-1]], dim=1)
            lam_rev = state_space_recursion(a_rev, g_rev.new_zeros(B, M), g_rev, **recur_kwargs)
            lam = lam_rev.flip(1)

        needs_init = ctx.needs_input_grad[9]
        needs_all = ctx.needs_input_grad[10 : 10 + len(all_tensors)]

        if not (needs_init or any(needs_all)):
            return (None,) * (10 + len(all_tensors))

        with torch.enable_grad():
            diff_inputs = []
            if needs_init:
                init_diff = init.detach().requires_grad_(True)
                diff_inputs.append(init_diff)
                init_prev_diff = (
                    init_diff.unsqueeze(1) if init_diff.dim() == 1 else init_diff.unsqueeze(1)
                )
            else:
                init_prev_diff = init_prev

            y_prev_eval = torch.cat([init_prev_diff, y_prev_detached], dim=1)

            x_diff_list = []
            for i, xt in enumerate(x_tensors):
                if needs_all[i]:
                    xt_d = xt.detach().requires_grad_(True)
                    diff_inputs.append(xt_d)
                    x_diff_list.append(xt_d)
                else:
                    x_diff_list.append(xt)
            x_diff = x_diff_list[0] if num_x == 1 else tuple(x_diff_list)

            params_diff = []
            for j, p in enumerate(params):
                idx = num_x + j
                if needs_all[idx]:
                    p_d = p.detach().requires_grad_(True)
                    diff_inputs.append(p_d)
                    params_diff.append(p_d)
                else:
                    params_diff.append(p)

            out = func(y_prev_eval, x_diff, *params_diff)
            grads = torch.autograd.grad(
                out,
                diff_inputs,
                grad_outputs=lam,
                create_graph=False,
                allow_unused=True,
            )

        grad_iter = iter(grads)
        grad_init = next(grad_iter) if needs_init else None
        grad_all = [next(grad_iter) if needs_all[k] else None for k in range(len(all_tensors))]

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
                f"All input tensors in x must have matching (batch, time) dimensions, "
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
