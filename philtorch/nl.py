"""Non-linear operators."""

import warnings
from collections.abc import Callable
from typing import Any, NamedTuple

import torch
from torch import Tensor
from torch.autograd import Function

from .lpv.recur import linear_recurrence
from .lpv.ssm import state_space_recursion

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

        def usable(t):
            return t.clone() if t.is_inference() else t

        y_in = usable(y)
        x_in = tuple(usable(xi) for xi in x) if isinstance(x, tuple) else usable(x)
        params_in = tuple(usable(p) for p in params)
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


@torch.compiler.disable
def _solve(options, init, x, params):
    func, fprime, y0, max_iter, max_halvings, atol, rtol, recur_kwargs, is_tuple_x, _ = options
    B, T = init.size(0), (x[0] if is_tuple_x else x).size(1)
    eps = torch.finfo(init.dtype).eps
    is_scalar = init.dim() == 1
    atol = eps if atol is None else atol
    rtol = 8.0 * eps * (1 if is_scalar else init.size(-1)) if rtol is None else rtol
    recur_fn = linear_recurrence if is_scalar else state_space_recursion

    y = init.new_empty(B, T + 1, *init.shape[1:])
    y[:, 0] = init
    y[:, 1:] = init.unsqueeze(1).expand(B, T, *init.shape[1:]) if y0 is None else y0

    # The batch elements still being solved: their indices, trajectories
    # (y itself while all are) and inputs
    rows, ys, xs = torch.arange(B, device=y.device), y, x

    def inputs_from(s):
        return tuple(xi[:, s:] for xi in xs) if is_tuple_x else xs[:, s:]

    def residual(s):
        return func(ys[:, s:-1], inputs_from(s), *params) - ys[:, s + 1 :]

    iters = torch.zeros(B, dtype=torch.long, device=y.device)
    n_diverged = 0
    res = residual(0)
    converged = _converged(res, ys[:, 1:], atol, rtol)
    for i in range(max_iter + 1):  # pragma: no branch - the last pass breaks
        diverged = ~_finite(res)
        done = diverged | converged
        n_diverged += int(diverged.sum())
        if done.any():
            if ys is not y:
                y[rows[done]] = ys[done]
            rows, ys, res = rows[~done], ys[~done], res[~done]
            xs = tuple(xi[rows] for xi in x) if is_tuple_x else x[rows]
        n_unconverged = rows.numel()
        if i == max_iter or n_unconverged == 0:
            break

        # y[i + 1] follows from the exact y[i]; the last one needs no step
        if i == T - 1:
            ys[:, T] += res[:, 0]
            iters[rows] += 1
            n_unconverged = 0
            break

        # Linearize about the current trajectory before y[i + 1] moves
        jac = _jacobian(func, ys[:, i + 1 : -1], inputs_from(i + 1), params, fprime)
        delta = recur_fn(jac, res[:, 0], res[:, 1:], **recur_kwargs)
        ys[:, i + 1] += res[:, 0]
        iters[rows] += 1
        # A step that is not finite is rejected, as a failed line search is
        per_row = (rows.numel(),) + (1,) * (y.dim() - 1)
        pending = _finite(delta)
        if not pending.all():
            delta = torch.where(pending.view(per_row), delta, 0.0)

        # Halve the step of each element until its residual norm decreases;
        # the norm before the step includes y[i + 1]'s, now 0
        base = ys[:, i + 2 :].clone()
        norm = _norm(res)
        t = torch.ones(per_row, dtype=y.dtype, device=y.device)
        for _ in range(max_halvings + 1):
            torch.addcmul(base, t, delta, out=ys[:, i + 2 :])
            res = residual(i + 1)
            converged = _converged(res, ys[:, i + 2 :], atol, rtol)
            pending &= ~((_norm(res) < norm) | converged)
            if not pending.any():
                break
            t = torch.where(pending.view(per_row), t * 0.5, t)
        else:
            t = torch.where(pending.view(per_row), 0.0, t)
            torch.addcmul(base, t, delta, out=ys[:, i + 2 :])
            res = residual(i + 1)
            converged = _converged(res, ys[:, i + 2 :], atol, rtol)

    if ys is not y:
        y[rows] = ys
    return y[:, 1:], iters, n_diverged, n_unconverged


class _Options(NamedTuple):
    func: Callable[..., Tensor]
    fprime: Callable[..., Tensor] | None
    y0: Tensor | None
    max_iter: int
    max_halvings: int
    atol: float | None
    rtol: float | None
    recur_kwargs: dict[str, Any]
    is_tuple_x: bool
    num_x: int


class _NewtonSolve(Function):
    @staticmethod
    def forward(options, init, *tensors):
        x = tuple(tensors[: options.num_x]) if options.is_tuple_x else tensors[0]
        return _solve(options, init, x, tensors[options.num_x :])

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        options, init, *tensors = inputs
        ctx.set_materialize_grads(False)
        ctx.mark_non_differentiable(outputs[1])
        ctx.options = options
        ctx.save_for_backward(outputs[0], init, *tensors)

    @staticmethod
    def backward(ctx, grad_y, *_):
        if grad_y is None:
            return (None,) * len(ctx.needs_input_grad)

        needs_init = ctx.needs_input_grad[1]
        needs_all = ctx.needs_input_grad[2:]

        func, fprime, recur_kwargs = ctx.options.func, ctx.options.fprime, ctx.options.recur_kwargs
        is_tuple_x, num_x = ctx.options.is_tuple_x, ctx.options.num_x

        y, init, *all_tensors = ctx.saved_tensors
        x = tuple(all_tensors[:num_x]) if is_tuple_x else all_tensors[0]
        params = all_tensors[num_x:]
        is_scalar = init.dim() == 1
        y_prev = torch.cat([init.unsqueeze(1), y[:, :-1]], dim=1)

        jac = _jacobian(func, y_prev, x, params, fprime)

        a_rev = jac[:, 1:].flip(1)
        if not is_scalar:
            a_rev = a_rev.mT
        a_rev = torch.cat([torch.zeros_like(jac[:, :1]), a_rev], dim=1)
        recur = linear_recurrence if is_scalar else state_space_recursion
        lam = recur(a_rev, torch.zeros_like(init), grad_y.flip(1), **recur_kwargs).flip(1)
        grad_init = None
        if needs_init:
            grad_init = (
                lam[:, 0] * jac[:, 0]
                if is_scalar
                else (lam[:, 0].unsqueeze(1) @ jac[:, 0]).squeeze(1)
            )

        grads = [None] * len(all_tensors)
        diff_indices = [k for k, need in enumerate(needs_all) if need]
        if diff_indices:
            create_graph = torch.is_grad_enabled()
            with torch.enable_grad():
                # Views, so that only the direct dependence on each input is
                # differentiated, not the one through y_prev
                args = list(all_tensors)
                for k in diff_indices:
                    args[k] = args[k].view_as(args[k])
                out = func(y_prev, tuple(args[:num_x]) if is_tuple_x else args[0], *args[num_x:])
                diff_grads = torch.autograd.grad(
                    out,
                    [args[k] for k in diff_indices],
                    lam,
                    create_graph=create_graph,
                    allow_unused=True,
                )
            for idx, g in zip(diff_indices, diff_grads):
                grads[idx] = g

        return (None, grad_init, *grads)


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
    differentiable. They are those of the exact solution, taken at the
    returned trajectory, so they hold for the elements that converged.

    :attr:`init`, :attr:`x` and :attr:`params` are real floating-point tensors
    with the dtype and device of :attr:`init`.

    Batch elements are solved independently. One is done when it converges,
    once :math:`|f(y[t - 1], x[t]; \theta) - y[t]| \le \text{atol} +
    \text{rtol}\, |y[t]|` for every :math:`t`, or diverges, when its
    residual is not finite. A ``RuntimeWarning`` reports the
    elements that diverged or did not converge within :attr:`max_iter`
    steps; they return their last iterate.

    Args:
        func (Callable): the step :math:`f`, called as
            ``func(y_prev, x_t, *params)`` on whole sequences or their
            ends, with ``y_prev`` holding :math:`y[t - 1]` and ``x_t`` the
            matching :math:`x[t]`: :attr:`x` cut to :math:`(B, T', *)`, the
            leading dimensions of ``y_prev`` (a tuple stays a tuple). For a
            1-D :attr:`init`, it acts elementwise on ``y_prev`` of any shape.
            Otherwise ``y_prev`` has shape :math:`(*, M)`, with any leading
            dimensions (:math:`(B, T', M)` here), and it acts on each sample
            separately and returns the shape of ``y_prev``.
        x (Tensor or tuple of Tensor): the inputs, of shape :math:`(B, T, *)`,
            :math:`*` being any number of dimensions.
        init (Tensor): the initial state :math:`y[-1]`, of shape
            :math:`(B, M)`, or :math:`(B)` for a scalar recurrence.
        params (tuple of Tensor, optional): the parameters :math:`\theta`,
            passed to :attr:`func` and :attr:`fprime` as they are and shared
            by every batch element and step. Values that differ per batch
            element go in :attr:`x`, expanded over time if constant.
            Default: ``()``.
        fprime (Callable, optional): the Jacobian of :attr:`func` with
            respect to ``y_prev``, called with the same arguments: of shape
            :math:`(*)` for a 1-D :attr:`init`, otherwise
            :math:`(*, M, M)` with entry :math:`(i, j)` equal to
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
            :attr:`max_halvings` is negative, :attr:`params` is not a tuple of
            tensors, :attr:`init` is not 1-D or 2-D, a tensor of
            :attr:`x` has fewer than 2 dimensions or a length :math:`T` of
            0, the tensors of :attr:`x` differ in batch size or length,
            :attr:`init`'s batch size differs from :attr:`x`'s, or
            :attr:`y0` is not of the shape of the output.

    Example::

        >>> from philtorch.nl import newton_solve
        >>> step = lambda y, x: torch.tanh(0.5 * y + x)
        >>> x = torch.full((1, 6), 0.5)
        >>> y, steps = newton_solve(step, x, torch.zeros(1))
        >>> y
        tensor([[0.4621, 0.6237, 0.6706, 0.6833, 0.6867, 0.6876]])
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

    if not isinstance(params, tuple) or not all(isinstance(p, Tensor) for p in params):
        raise ValueError("params must be a tuple of Tensors")

    all_tensors = x_tensors + tuple(params)
    options = _Options(
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
    )
    y, iters, n_diverged, n_unconverged = _NewtonSolve.apply(options, init, *all_tensors)
    B = init.size(0)
    if n_diverged:
        warnings.warn(
            f"newton_solve diverged (non-finite residual) in {n_diverged} of {B} batch elements",
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
