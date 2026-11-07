"""Non-linear recurrent sequence solvers."""

from collections.abc import Callable
import warnings

import torch
from torch import Tensor

from philtorch.lpv.recur import linear_recurrence
from philtorch.lpv.ssm import state_space_recursion


def _slice_x(
    x: Tensor | tuple[Tensor, ...], start: int
) -> Tensor | tuple[Tensor, ...]:
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
                out, y_in, grad_outputs=torch.ones_like(out), create_graph=False
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

    is_scalar = (init.dim() == 1) or (
        init.dim() == 2 and init.size(-1) == 1 and x_prim.dim() == 2
    )

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

        converged = (
            res.abs() <= atol_eff + rtol_eff * y[:, i + 1 :].abs()
        ).all()
        if converged:
            break

        jac = _jacobian(func, y_prev, x_curr, params, is_scalar, fprime)

        if is_scalar:
            delta = linear_recurrence(
                jac[:, 1:], res[:, 0], res[:, 1:], **recur_kwargs
            )
        else:
            delta = state_space_recursion(
                jac[:, 1:], res[:, 0], res[:, 1:], **recur_kwargs
            )

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
            cand_conv = (
                cand_res.abs() <= atol_eff + rtol_eff * cand.abs()
            ).all()
            if cand_norm < merit or cand_conv:
                break
            t /= 2.0

    return y[:, 1:], i + 1
