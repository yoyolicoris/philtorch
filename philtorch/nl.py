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
    create_graph = torch.is_grad_enabled()
    with torch.inference_mode(False):
        y_in = y.clone() if y.is_inference() else y
        x_in = (
            tuple(xi.clone() if xi.is_inference() else xi for xi in x)
            if isinstance(x, tuple)
            else (x.clone() if x.is_inference() else x)
        )
        params_in = tuple(p.clone() if p.is_inference() else p for p in params)
        if y.dim() == 2:
            with torch.enable_grad():
                y_diff = y_in if y_in.requires_grad else y_in.detach().requires_grad_()
                out = func(y_diff, x_in, *params_in)
                if not out.requires_grad:
                    return torch.zeros_like(y)
                grads = torch.autograd.grad(
                    out.sum(),
                    y_diff,
                    create_graph=create_graph,
                    allow_unused=True,
                )
                return torch.zeros_like(y) if grads[0] is None else grads[0]
        return torch.autograd.functional.jacobian(
            lambda y_: func(y_, x_in, *params_in).sum((0, 1)),
            y_in,
            create_graph=create_graph,
            vectorize=True,
        ).permute(1, 2, 0, 3)


def _finite(t: Tensor) -> Tensor:
    return torch.isfinite(t).flatten(1).all(1)


def _converged(res: Tensor, y: Tensor, atol: float, rtol: float) -> Tensor:
    return (res.abs() <= atol + rtol * y.abs()).flatten(1).all(1)


def _norm(res: Tensor) -> Tensor:
    return torch.linalg.vector_norm(res.flatten(1), dim=1)


class _NewtonSolve(Function):
    @staticmethod
    def forward(
        func,
        fprime,
        y0,
        max_iter,
        max_halvings,
        atol,
        rtol,
        recur_kwargs,
        is_tuple_x,
        num_x,
        init,
        *all_tensors,
    ):
        x = tuple(all_tensors[:num_x]) if is_tuple_x else all_tensors[0]
        params = all_tensors[num_x:]
        B, T = init.size(0), all_tensors[0].size(1)
        eps = torch.finfo(init.dtype).eps
        is_scalar = init.dim() == 1
        atol = eps if atol is None else atol
        rtol = 8.0 * eps * (1 if is_scalar else init.size(-1)) if rtol is None else rtol
        recur_fn = linear_recurrence if is_scalar else state_space_recursion

        y = init.new_empty(B, T + 1, *init.shape[1:])
        y[:, 0] = init
        y[:, 1:] = init.unsqueeze(1).expand(B, T, *init.shape[1:]) if y0 is None else y0

        # The batch elements still being solved: their indices, trajectories
        # (y itself while all are), inputs and parameters
        rows, ys, xs, ps = torch.arange(B, device=y.device), y, x, params

        def inputs_from(s):
            return tuple(xi[:, s:] for xi in xs) if is_tuple_x else xs[:, s:]

        def residual(s):
            return func(ys[:, s:-1], inputs_from(s), *ps) - ys[:, s + 1 :]

        iters = torch.zeros(B, dtype=torch.long, device=y.device)
        step_failed = torch.zeros(B, dtype=torch.bool, device=y.device)
        n_diverged = 0
        res = residual(0)
        for i in range(max_iter + 1):  # pragma: no branch - the last pass breaks
            diverged = ~_finite(res) | step_failed
            done = diverged | _converged(res, ys[:, i + 1 :], atol, rtol)
            n_diverged += int(diverged.sum())
            if done.any():
                if ys is not y:
                    y[rows[done]] = ys[done]
                rows, ys, res = rows[~done], ys[~done], res[~done]
                xs = tuple(xi[rows] for xi in x) if is_tuple_x else x[rows]
                ps = tuple(p[rows] if p.dim() and p.size(0) == B else p for p in params)
            n_unconverged = rows.numel()
            if i == max_iter or n_unconverged == 0:
                break

            # y[i + 1] follows from the exact y[i]; the last one needs no step
            ys[:, i + 1] += res[:, 0]
            if i == T - 1:
                iters[rows] += 1
                n_unconverged = 0
                break

            jac = _jacobian(func, ys[:, i + 1 : -1], inputs_from(i + 1), ps, fprime)
            delta = recur_fn(jac, res[:, 0], res[:, 1:], **recur_kwargs)
            step_failed = ~_finite(delta)
            iters[rows] += ~step_failed
            per_row = (rows.numel(),) + (1,) * (y.dim() - 1)
            delta = torch.where(step_failed.view(per_row), 0.0, delta)

            # Halve the step of each element until its residual norm decreases
            base = ys[:, i + 2 :].clone()
            norm = _norm(res[:, 1:])
            t = torch.ones(per_row, dtype=y.dtype, device=y.device)
            pending = ~step_failed
            for _ in range(max_halvings + 1):
                ys[:, i + 2 :] = base + t * delta
                res = residual(i + 1)
                pending &= ~((_norm(res) < norm) | _converged(res, ys[:, i + 2 :], atol, rtol))
                if not pending.any():
                    break
                t = torch.where(pending.view(per_row), t * 0.5, t)
            else:
                t = torch.where(pending.view(per_row), 0.0, t)
                ys[:, i + 2 :] = base + t * delta
                res = residual(i + 1)

        if ys is not y:
            y[rows] = ys
        return y[:, 1:], iters, n_diverged, n_unconverged

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        func, fprime, *_, recur_kwargs, is_tuple_x, num_x, init = inputs[:11]
        ctx.set_materialize_grads(False)
        ctx.mark_non_differentiable(outputs[1])
        ctx.func = func
        ctx.fprime = fprime
        ctx.recur_kwargs = recur_kwargs
        ctx.is_tuple_x = is_tuple_x
        ctx.num_x = num_x
        ctx.save_for_backward(outputs[0], init, *inputs[11:])

    @staticmethod
    def backward(ctx, grad_y, *_):
        if grad_y is None:
            return (None,) * len(ctx.needs_input_grad)

        needs_init = ctx.needs_input_grad[10]
        needs_all = ctx.needs_input_grad[11:]
        if not (needs_init or any(needs_all)):
            return (None,) * len(ctx.needs_input_grad)

        func = ctx.func
        fprime = ctx.fprime
        recur_kwargs = ctx.recur_kwargs
        is_tuple_x = ctx.is_tuple_x
        num_x = ctx.num_x

        y, init, *all_tensors = ctx.saved_tensors
        x = tuple(all_tensors[:num_x]) if is_tuple_x else all_tensors[0]
        params = all_tensors[num_x:]
        B = init.size(0)

        is_scalar = init.dim() == 1
        y_prev = torch.cat([init.unsqueeze(1), y[:, :-1]], dim=1)

        jac = _jacobian(func, y_prev, x, params, fprime)

        g_rev = grad_y.flip(1)
        if is_scalar:
            a_rev = F.pad(jac.flip(1)[:, :-1], (1, 0))
            lam = linear_recurrence(a_rev, g_rev.new_zeros(B), g_rev, **recur_kwargs).flip(1)
            grad_init = (lam[:, 0] * jac[:, 0]).reshape_as(init) if needs_init else None
        else:
            jac_rev = jac.flip(1).transpose(-1, -2)
            M = init.size(-1)
            a_rev = torch.cat([jac_rev.new_zeros(B, 1, M, M), jac_rev[:, :-1]], dim=1)
            lam = state_space_recursion(a_rev, g_rev.new_zeros(B, M), g_rev, **recur_kwargs).flip(1)
            grad_init = (
                (lam[:, 0].unsqueeze(1) @ jac[:, 0]).squeeze(1).reshape_as(init)
                if needs_init
                else None
            )

        grads = [None] * len(all_tensors)
        diff_indices = [k for k, need in enumerate(needs_all) if need]
        if diff_indices:
            diff_inputs = tuple(all_tensors[k] for k in diff_indices)

            def step_eval(*diff_args):
                full_args = list(all_tensors)
                for idx, arg in zip(diff_indices, diff_args):
                    full_args[idx] = arg
                x_in = tuple(full_args[:num_x]) if is_tuple_x else full_args[0]
                return func(y_prev, x_in, *full_args[num_x:])

            _, diff_grads = torch.autograd.functional.vjp(
                step_eval,
                diff_inputs,
                v=lam,
                create_graph=torch.is_grad_enabled(),
            )
            for idx, g in zip(diff_indices, diff_grads):
                grads[idx] = g

        return (None,) * 10 + (grad_init,) + tuple(grads)


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
) -> tuple[Tensor, Tensor]:
    r"""Solve a non-linear recurrence over a whole sequence by Newton's method.

    This computes

    .. math::
        y[t] = f(y[t - 1], x[t]; \theta), \quad t = 0, \dots, T - 1,

    starting from :math:`y[-1] = \text{init}`, for all :math:`t` at once, as
    in `ParaRNN: Unlocking Parallel Training of Nonlinear RNNs for Large
    Language Models`_ (Danieli et al., 2026): each Newton step linearizes
    :math:`f` about the current trajectory and solves the linear recurrence
    that results with :func:`philtorch.lpv.linear_recurrence` or
    :func:`philtorch.lpv.state_space_recursion`. A step makes one more
    sample exact, so :math:`T` steps always suffice; a contractive :math:`f`
    needs far fewer. A step is halved until it decreases the norm of the
    residual :math:`f(y[t - 1], x[t]; \theta) - y[t]`. The gradients come
    from one more linear recurrence, backward in time, and are themselves
    differentiable.

    Batch elements are solved independently. One is done when it converges,
    once :math:`|f(y[t - 1], x[t]; \theta) - y[t]| \le \text{atol} +
    \text{rtol}\, |y[t]|` for every :math:`t`, or diverges, when its
    residual or step is not finite. A ``RuntimeWarning`` reports the
    elements that diverged or did not converge within :attr:`max_iter`
    steps; they return their last iterate.

    Args:
        func (Callable): the step :math:`f`, called as
            ``func(y_prev, x_prev, *params)`` on whole sequences or their
            ends: ``y_prev``, of shape :math:`(B, T', M)` or :math:`(B, T')`,
            holds :math:`y[t - 1]` and ``x_prev`` the matching
            :math:`x[t]`, that is :attr:`x` cut to :math:`(B, T', *)` (a
            tuple stays a tuple). It must act on each sample separately and
            return a tensor of the shape of ``y_prev``.
        x (Tensor or tuple of Tensor): the inputs, of shape :math:`(B, T, *)`,
            :math:`*` being any number of dimensions.
        init (Tensor): the initial state :math:`y[-1]`, of shape
            :math:`(B, M)`, or :math:`(B)` for a scalar recurrence.
        params (tuple of Tensor, optional): the parameters :math:`\theta`,
            the same at every sample. One whose first dimension is :math:`B`
            is taken as one per batch element. Default: ``()``.
        fprime (Callable, optional): the Jacobian of :attr:`func` with
            respect to ``y_prev``, called with the same arguments: of shape
            :math:`(B, T')` for a 1-D :attr:`init`, otherwise
            :math:`(B, T', M, M)` with entry :math:`(i, j)` equal to
            :math:`\partial f_i / \partial y_j`. ``None`` differentiates
            :attr:`func` with autograd. Default: ``None``.
        y0 (Tensor, optional): the initial guess of the trajectory, of the
            shape of the output. ``None`` repeats :attr:`init`.
            Default: ``None``.
        max_iter (int, optional): the maximum number of Newton steps.
            Default: 100.
        max_halvings (int, optional): the maximum number of halvings of a
            step. Default: 8.
        atol (float, optional): the absolute tolerance. ``None`` takes
            :math:`\epsilon`, the machine epsilon of :attr:`init`'s dtype.
            Default: ``None``.
        rtol (float, optional): the relative tolerance. ``None`` takes
            :math:`8 M \epsilon`, as the rounding error of :math:`f` grows
            with the state size :math:`M` (1 for a 1-D :attr:`init`).
            Default: ``None``.
        **recur_kwargs: keyword arguments of the linear recurrences, such
            as ``unroll_factor``.

    Returns:
        tuple of Tensor: the trajectory :math:`y[0], \dots, y[T - 1]`, of
        shape :math:`(B, T, M)` or :math:`(B, T)`, and the number of Newton
        steps each batch element took, of shape :math:`(B)` (0 if :attr:`y0`
        already solves the recurrence). The counts are not differentiable.

    Raises:
        ValueError: if :attr:`max_iter` is not positive,
            :attr:`max_halvings` is negative, :attr:`params` holds anything
            but tensors, :attr:`init` is not 1-D or 2-D, a tensor of
            :attr:`x` has fewer than 2 dimensions or a length :math:`T` of
            0, the batch sizes or lengths of :attr:`x` and :attr:`init`
            differ, or :attr:`y0` is not of the shape of the output.

    Example::

        >>> from philtorch.nl import newton_solve
        >>> step = lambda y, x: torch.tanh(0.5 * y + x)
        >>> x = torch.full((1, 8), 0.5)
        >>> y, steps = newton_solve(step, x, torch.zeros(1))
        >>> y
        tensor([[0.4621, 0.6237, 0.6706, 0.6833, 0.6867, 0.6876, 0.6878, 0.6879]])
        >>> steps
        tensor([3])

    .. _ParaRNN\: Unlocking Parallel Training of Nonlinear RNNs for Large Language Models:
        https://proceedings.iclr.cc/paper_files/paper/2026/file/57d8ebf4c2f050a6485f370d47656a9e-Paper-Conference.pdf
    """
    if max_iter <= 0:
        raise ValueError(f"max_iter must be positive, got {max_iter}")
    if max_halvings < 0:
        raise ValueError(f"max_halvings must be non-negative, got {max_halvings}")
    if init.dim() not in (1, 2):
        raise ValueError(f"init must be 1-D (B,) or 2-D (B, M), got {init.dim()}-D")

    x_tensors = x if isinstance(x, tuple) else (x,)
    for xi in x_tensors:
        if xi.dim() < 2:
            raise ValueError(
                f"Input tensor x must have at least 2 dimensions (batch, time), got {xi.shape}"
            )
        if xi.size(1) == 0:
            raise ValueError("Sequence length T must be greater than 0")
        if xi.size(0) != x_tensors[0].size(0) or xi.size(1) != x_tensors[0].size(1):
            raise ValueError(
                "All input tensors in x must have matching (batch, time) dimensions, "
                f"got {xi.shape} vs {x_tensors[0].shape}"
            )

    if init.size(0) != x_tensors[0].size(0):
        raise ValueError(
            f"Batch size of init ({init.size(0)}) must match x ({x_tensors[0].size(0)})"
        )
    if y0 is not None:
        expected_shape = (init.size(0), x_tensors[0].size(1), *init.shape[1:])
        if y0.shape != expected_shape:
            raise ValueError(f"y0 must have shape {expected_shape}, got {y0.shape}")

    if not all(isinstance(p, Tensor) for p in params):
        raise ValueError("params must be a tuple of Tensors")

    all_tensors = x_tensors + tuple(params)
    y, iters, n_diverged, n_unconverged = _NewtonSolve.apply(
        func,
        fprime,
        y0,
        max_iter,
        max_halvings,
        atol,
        rtol,
        recur_kwargs,
        isinstance(x, tuple),
        len(x_tensors),
        init,
        *all_tensors,
    )
    B = init.size(0)
    if n_diverged:
        warnings.warn(
            "newton_solve diverged (non-finite step or residual) in "
            f"{n_diverged} of {B} batch elements",
            RuntimeWarning,
            stacklevel=2,
        )
    if n_unconverged:
        warnings.warn(
            f"newton_solve did not converge in {max_iter} iterations "
            f"in {n_unconverged} of {B} batch elements",
            RuntimeWarning,
            stacklevel=2,
        )
    return y, iters
