"""State-space models with time-invariant coefficients."""

from functools import partial
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Function

from .. import HELION_LOADED
from ..mat import find_eigenvectors, matrix_power_accumulate
from .recur import LTIRecurrence, linear_recurrence


def extension_backend_indicator(x: Tensor, M: int) -> bool:
    r"""Return whether to dispatch this input to the native extension.

    This is a dispatch heuristic, not a check that a kernel exists: it
    assumes kernels for :math:`M \le 2` on every device. Some are still
    missing for a device or dtype, such as :math:`M = 2` on MPS, and calling
    one raises an error; the README lists them.

    Args:
        x (Tensor): the input, whose device is checked.
        M (int): the state size.

    Returns:
        bool: ``True`` on CPU, or when :math:`M \le 2`.
    """
    return M <= 2 or x.is_cpu


def helion_backend_indicator(x: Tensor) -> bool:
    """Return whether the Helion kernels can run on this input.

    Args:
        x (Tensor): the input, whose device and dtype are checked.

    Returns:
        bool: ``True`` when Helion is loaded and :attr:`x` is a real tensor on
        a CUDA device.
    """
    return HELION_LOADED and x.is_cuda and not x.is_complex()


class LTIMatrixRecurrence(Function):
    r"""Autograd function for the matrix recurrence of :func:`_ext_ss_recur`.

    The forward pass runs the native ``lti_recur2`` kernel for
    :math:`M = 2`, the Helion kernel on CUDA when it is available, and the
    ``lti_recurN`` kernel otherwise. The backward pass runs the recurrence
    backward in time with :math:`A^H`, and the JVP runs it forward on the
    tangents, so both reverse- and forward-mode differentiation work.
    """

    @staticmethod
    def forward(A: Tensor, zi: Tensor, x: Tensor) -> Tensor:
        if x.size(-1) == 2:
            return torch.ops.philtorch.lti_recur2(A, zi, x)
        elif helion_backend_indicator(x):
            from .. import hl_lti_recurN

            return hl_lti_recurN(A, zi, x)
        return torch.ops.philtorch.lti_recurN(A, zi, x)

    @staticmethod
    def setup_context(ctx: Any, inputs: list[Any], output: Any) -> Any:
        A, zi, _ = inputs
        y = output
        ctx.save_for_backward(A, zi, y)
        ctx.save_for_forward(A, zi, y)

    @staticmethod
    def backward(
        ctx: Any, grad_y: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        A, zi, y = ctx.saved_tensors
        grad_x = grad_A = grad_zi = None

        AmT = A.mT.conj_physical()

        flipped_grad_x = LTIMatrixRecurrence.apply(AmT, torch.zeros_like(zi), grad_y.flip(1))

        if ctx.needs_input_grad[1]:
            grad_zi = (AmT @ flipped_grad_x[:, -1, :, None]).squeeze(-1)

        if ctx.needs_input_grad[2]:
            grad_x = flipped_grad_x.flip(1)

        if ctx.needs_input_grad[0]:
            valid_y = y[:, :-1]
            padded_y = torch.cat([zi.unsqueeze(1), valid_y], dim=1)
            if A.dim() == 2:
                grad_A = flipped_grad_x.flip(1).flatten(0, 1).T @ padded_y.conj_physical().flatten(
                    0, 1
                )
            else:
                grad_A = flipped_grad_x.flip(1).mT @ padded_y.conj_physical()

        return grad_A, grad_zi, grad_x

    @staticmethod
    def jvp(
        ctx: Any, grad_A: torch.Tensor, grad_zi: torch.Tensor, grad_x: torch.Tensor
    ) -> torch.Tensor:
        A, zi, y = ctx.saved_tensors

        fwd_zi = grad_zi if grad_zi is not None else torch.zeros_like(zi)
        fwd_x = grad_x if grad_x is not None else torch.zeros_like(y)

        if grad_A is not None:
            padded_y = torch.cat([zi.unsqueeze(1), y[:, :-1]], dim=1)
            fwd_A = (
                (grad_A if grad_A.dim() == 2 else grad_A.unsqueeze(-3)) @ padded_y.unsqueeze(-1)
            ).squeeze(-1)
            fwd_x = fwd_x + fwd_A

        return LTIMatrixRecurrence.apply(A, fwd_zi, fwd_x)


def _recursion_loop(
    A: Tensor,
    zi: Tensor,
    x: Tensor,
    out_idx: int | None = None,
) -> Tensor:
    """Run :math:`h[n] = A h[n - 1] + x[n]` with a loop over time steps.

    Args:
        A (Tensor): state matrices, of shape :math:`(M, M)` or
            :math:`(B, M, M)`.
        zi (Tensor): the initial state, of shape :math:`(B, M)`.
        x (Tensor): inputs, of shape :math:`(B, N, M)`, or :math:`(B, N)` to
            feed the first state only.
        out_idx (int, optional): return only this state. Default: ``None``.

    Returns:
        Tensor: the states, of shape :math:`(B, N, M)`, or :math:`(B, N)`
        with :attr:`out_idx`.
    """
    results = []
    AT = A.mT
    if x.dim() == 2:
        M = A.size(-1)
        x = torch.cat([x.unsqueeze(-1), x.new_zeros(*x.shape, M - 1)], dim=-1)  # (batch, time, M)
    if A.dim() == 2:
        h = zi
        for xn in x.unbind(1):
            h = torch.addmm(xn, h, AT)
            results.append(h if out_idx is None else h[:, out_idx])
        output = torch.stack(results, dim=1)
    else:
        h = zi.unsqueeze(1)
        for xn in x.unbind(1):
            h = torch.baddbmm(xn.unsqueeze(1), h, AT)
            results.append(h if out_idx is None else h[:, :, out_idx])
        output = torch.cat(results, dim=1)

    return output


def _ext_ss_recur(A: Tensor, zi: Tensor, x: Tensor, *, out_idx: int | None = None, **_) -> Tensor:
    """Run :math:`h[n] = A h[n - 1] + x[n]` with the native kernels.

    Takes the same arguments and returns the same as :func:`_recursion_loop`.
    """
    if x.dim() == 2 and A.size(-1) == 1:
        y = LTIRecurrence.apply(A[..., 0, 0], zi.squeeze(-1), x).unsqueeze(-1)
    elif A.size(-1) == 1:
        y = LTIRecurrence.apply(A[..., 0, 0], zi.squeeze(-1), x.squeeze(-1)).unsqueeze(-1)
    else:
        x = (
            torch.cat([x.unsqueeze(-1), x.new_zeros(*x.shape, A.size(-1) - 1)], dim=-1)
            if x.dim() == 2
            else x
        )
        y = LTIMatrixRecurrence.apply(A, zi, x)
    if out_idx is not None and y.dim() == 3:
        y = y[:, :, out_idx]
    return y


def _select_recursion_runner(x: Tensor, M: int, unroll_factor: int):
    if unroll_factor == 1 and extension_backend_indicator(x, M):
        return _ext_ss_recur
    return _recursion_loop


def state_space_recursion(
    A: Tensor,
    zi: Tensor,
    x: Tensor,
    *,
    unroll_factor: int = 1,
    out_idx: int | None = None,
) -> Tensor:
    r"""Compute the states of a linear time-invariant recurrence.

    This computes

    .. math::
        \mathbf{h}[n] = A \mathbf{h}[n - 1] + \mathbf{x}[n],
        \quad n = 0, \dots, N - 1,

    starting from :math:`\mathbf{h}[-1] = \mathbf{z}_i`. A 2-D input feeds
    the first state only: :math:`\mathbf{x}[n] = x[n] \mathbf{e}_1`.

    With ``unroll_factor=1``, this calls the native extension on CPU, and on
    other devices when :math:`M \le 2`, and otherwise runs a loop over time
    steps. On other devices the extension may lack a kernel for the device or
    dtype, such as :math:`M = 2` on MPS, which raises an error; the README
    lists them. A larger :attr:`unroll_factor` runs a block-unrolled PyTorch
    recursion instead, on any device.

    Args:
        A (Tensor): state matrices, of shape :math:`(M, M)` or
            :math:`(B, M, M)`.
        zi (Tensor): the initial state :math:`\mathbf{h}[-1]`, of shape
            :math:`(B, M)`.
        x (Tensor): inputs, of shape :math:`(B, N, M)` or :math:`(B, N)`.
        unroll_factor (int, optional): ``1`` for the native kernels, or the
            block length of the unrolled recursion. Default: ``1``.
        out_idx (int, optional): return only this state. Default: ``None``.

    Returns:
        Tensor: the states :math:`\mathbf{h}[0], \dots, \mathbf{h}[N - 1]`,
        of shape :math:`(B, N, M)`, or :math:`(B, N)` with :attr:`out_idx`.

    Raises:
        AssertionError: if the shapes do not match.
        ValueError: if :attr:`unroll_factor` is less than 1.

    Note:
        Unlike :func:`state_space`, the state at step :math:`n` already
        includes :math:`\mathbf{x}[n]`.

    Example::

        >>> from philtorch.lti import state_space_recursion
        >>> A = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
        >>> x = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
        >>> state_space_recursion(A, torch.zeros(1, 2), x, out_idx=0)
        tensor([[1., 1., 2., 3., 5.]])
    """
    assert x.dim() in (
        2,
        3,
    ), f"Input signal must be 2D or 3D (batch, time, [features]), got {x.shape}"
    assert A.dim() in (2, 3), f"State matrix A must be 2D or 3D, got {A.shape}"
    assert A.size(-2) == A.size(-1), f"State matrix A must be square, got {A.shape}"
    if A.dim() == 3:
        assert x.size(0) == A.size(0), (
            f"Batch size of A must match batch size of x, got A: {A.size(0)}, x: {x.size(0)}"
        )

    if x.dim() == 3:
        assert A.size(-1) == x.size(-1), (
            f"Last dimension of A must match last dimension of x, "
            f"got A: {A.size(-1)}, x: {x.size(-1)}"
        )

    batch_size, N = x.size(0), x.size(1)
    M = A.size(-1)

    assert zi.dim() == 2, f"Initial conditions zi must be 2D, got {zi.shape}"
    assert zi.size(0) == batch_size, (
        f"Batch size of zi must match batch size of x, got zi: {zi.size(0)}, x: {batch_size}"
    )
    assert zi.size(1) == M, (
        f"Last dimension of zi must match last dimension of A, got zi: {zi.size(1)}, A: {M}"
    )

    if unroll_factor < 1:
        raise ValueError("Unroll factor must be >= 1")
    else:
        block_size = unroll_factor

    runner = _select_recursion_runner(x, M, block_size)

    # boundary condition
    if block_size == 1 or block_size >= N:
        return runner(A, zi, x, out_idx=out_idx)

    remainder = N % block_size
    if remainder != 0:
        x = F.pad(x, (0, 0) * (x.dim() - 2) + (0, block_size - remainder))
        N = x.size(1)  # Update T after padding

    unrolled_x = x.unflatten(1, (-1, block_size)).flatten(2, -1)

    A_powers = matrix_power_accumulate(A, block_size)
    A_powered = A_powers[..., -1, :, :]
    A_powers_plus_I = torch.cat(
        [
            A_powers[..., :-1, :, :].flip(-3),
            torch.eye(M, device=A.device, dtype=A.dtype).broadcast_to(A.shape).unsqueeze(-3),
        ],
        dim=-3,
    )

    mat1 = (
        A_powers_plus_I.transpose(-2, -1).flatten(-3, -2)
        if x.dim() == 3
        else A_powers_plus_I[..., 0]  # assume x -> x * [1, 0, 0, ...] in the 2D case
    )
    z = unrolled_x @ mat1

    initials = torch.cat(
        [
            zi.unsqueeze(1),
            state_space_recursion(A_powered, zi, z, unroll_factor=unroll_factor),
        ],
        dim=1,
    )

    # prepare the augmented matrix and input for all the remaining steps
    aug_x = torch.cat([initials[:, :-1], unrolled_x[..., : -(1 if x.dim() == 2 else M)]], dim=2)

    if out_idx is None:
        mat2 = A_powers[..., :-1, :, :].flatten(-3, -2)
        if x.dim() == 3:
            mat3 = (
                torch.cat(
                    [
                        A_powers_plus_I[..., 1:, :, :],
                        A_powers_plus_I.new_zeros(
                            A_powers_plus_I.shape[:-3] + (block_size - 2, M, M)
                        ),
                    ],
                    dim=-3,
                )
                .unfold(-3, block_size - 1, 1)
                .transpose(-2, -1)
                .flip(-4)
                .flatten(-4, -3)
                .flatten(-2, -1)
            )
        else:
            mat3 = (
                torch.cat(
                    [
                        A_powers_plus_I[..., 1:, :, 0],
                        A_powers_plus_I.new_zeros(A_powers_plus_I.shape[:-3] + (block_size - 2, M)),
                    ],
                    dim=-2,
                )
                .unfold(-2, block_size - 1, 1)
                .flip(-3)
                .flatten(-3, -2)
            )
    else:
        mat2 = A_powers[..., :-1, out_idx, :]
        if x.dim() == 3:
            mat3 = (
                torch.cat(
                    [
                        A_powers_plus_I[..., 1:, out_idx, :],
                        A_powers_plus_I.new_zeros(A_powers_plus_I.shape[:-3] + (block_size - 2, M)),
                    ],
                    dim=-2,
                )
                .unfold(-2, block_size - 1, 1)
                .transpose(-2, -1)
                .flip(-3)
                .flatten(-2, -1)
            )
        else:
            mat3 = (
                torch.cat(
                    [
                        A_powers_plus_I[..., 1:, out_idx, 0],
                        A_powers_plus_I.new_zeros(A_powers_plus_I.shape[:-3] + (block_size - 2,)),
                    ],
                    dim=-1,
                )
                .unfold(-1, block_size - 1, 1)
                .flip(-2)
            )

    aug_A = torch.cat([mat2.mT, mat3.mT], dim=-2)
    output = aug_x @ aug_A

    # concat the first M - 1 outputs with the last one
    if out_idx is None:
        output = torch.cat([output, initials[:, 1:, :]], dim=2).unflatten(2, (-1, M)).flatten(1, 2)
    else:
        output = torch.cat([output, initials[:, 1:, out_idx, None]], dim=2).flatten(1, 2)
    if remainder != 0:
        # if we padded the input, we need to remove the padding from the output
        output = output[:, : -(block_size - remainder)]
    return output


def state_space(
    A: Tensor,
    x: Tensor,
    B: Tensor | None = None,
    C: Tensor | None = None,
    D: Tensor | None = None,
    zi: Tensor | None = None,
    unroll_factor: int = 1,
    out_idx: int | None = None,
    # **kwargs,
):
    r"""Compute the outputs of a linear time-invariant state-space model.

    This computes

    .. math::
        \mathbf{h}[n + 1] &= A \mathbf{h}[n] + B \mathbf{x}[n], \\
        \mathbf{y}[n] &= C \mathbf{h}[n] + D \mathbf{x}[n],

    starting from :math:`\mathbf{h}[0] = \mathbf{z}_i`. The recursion runs as
    in :func:`state_space_recursion`.

    Args:
        A (Tensor): state matrices, of shape :math:`(M, M)` or
            :math:`(B, M, M)`.
        x (Tensor): inputs, of shape :math:`(B, N, F)`, or :math:`(B, N)` for
            a scalar input.
        B (Tensor, optional): the input matrix, of shape :math:`(M)` or
            :math:`(B, M)` for a 2-D :attr:`x`, and :math:`(M, F)` or
            :math:`(B, M, F)` for a 3-D one. Default: ``None``: a 2-D
            :attr:`x` feeds the first state, and a 3-D one needs
            :math:`F = M`.
        C (Tensor, optional): the output matrix, of shape :math:`(M)` or
            :math:`(B, M)` for a scalar output, or :math:`(P, M)` or
            :math:`(B, P, M)` for :math:`P` outputs. Default: ``None``, which
            outputs the states.
        D (Tensor, optional): the feedthrough matrix. For a 2-D :attr:`x`, of
            shape :math:`()`, :math:`(1)`, :math:`(B)`, :math:`(P)` or
            :math:`(B, P)`; for a 3-D one, :math:`()`, :math:`(1)`,
            :math:`(F)`, :math:`(B, F)`, :math:`(P, F)` or :math:`(B, P, F)`.
            When two readings fit, such as :math:`P = B`, the batched one wins.
            Default: ``None``, no feedthrough.
        zi (Tensor, optional): the initial state, of shape :math:`(B, M)` or
            :math:`(M)`. Default: ``None``, all zero.
        unroll_factor (int, optional): see :func:`state_space_recursion`.
            Default: ``1``.
        out_idx (int, optional): output only this state, instead of using
            :attr:`C`. Default: ``None``.

    Returns:
        Tensor or tuple of Tensor: the outputs, of shape :math:`(B, N)`,
        :math:`(B, N, P)`, or :math:`(B, N, M)` without :attr:`C`, and with
        :attr:`zi`, the final state :math:`\mathbf{h}[N]`, of shape
        :math:`(B, M)`.

    Raises:
        ValueError: if both :attr:`C` and :attr:`out_idx` are given, or
            :attr:`B`, :attr:`C` or :attr:`D` has an unsupported shape.

    Note:
        This computes the same as :func:`scipy.signal.dlsim` with
        ``x0=zi``, batched, without the time vector or the state sequence.

    Example::

        >>> from philtorch.lti import state_space
        >>> # Fibonacci numbers from an impulse.
        >>> A = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
        >>> x = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        >>> state_space(A, x, C=torch.tensor([1.0, 0.0]))
        tensor([[0., 1., 1., 2., 3., 5., 8.]])
    """
    assert x.dim() in (
        2,
        3,
    ), f"Input signal must be 2D or 3D (batch, time, [features]), got {x.shape}"

    assert A.dim() in (2, 3), f"State matrix A must be 2D or 3D, got {A.shape}"
    assert A.size(-2) == A.size(-1), f"State matrix A must be square, got {A.shape}"
    if A.dim() == 3:
        assert x.size(0) == A.size(0), (
            f"Batch size of A must match batch size of x, got A: {A.size(0)}, x: {x.size(0)}"
        )

    if not (C is None or out_idx is None):
        raise ValueError("C and out_idx cannot be used together. Use either C or out_idx.")

    batch_size, N, *_ = x.shape
    M = A.size(-1)

    return_zf = True
    if zi is None:
        return_zf = False
        zi = x.new_zeros(batch_size, M)
    elif zi.dim() == 1:
        zi = zi.unsqueeze(0).expand(batch_size, -1)

    Bx = _ssm_B(B, x, batch_size, M)

    if return_zf or out_idx is None:
        h = state_space_recursion(A, zi, Bx, unroll_factor=unroll_factor, out_idx=None)
        zf = h[:, -1, :] if return_zf else None
        h = (
            torch.cat([zi.unsqueeze(1), h[:, :-1]], dim=1)
            if out_idx is None
            else torch.cat([zi[:, None, out_idx], h[:, :-1, out_idx]], dim=1)
        )
    else:
        zf = None
        h = state_space_recursion(A, zi, Bx, unroll_factor=unroll_factor, out_idx=out_idx)
        h = torch.cat([zi[:, None, out_idx], h[:, :-1]], dim=1)

    y = _ssm_C_D(h, x, C, D, batch_size, M)

    if return_zf:
        return y, zf
    return y


def _ssm_B(B, x, batch_size, M):
    r"""Return :math:`B \mathbf{x}[n]` for every step.

    This takes the shapes :func:`state_space` accepts, and returns :attr:`x`
    itself without :attr:`B`.
    """
    if B is None:
        return x

    # Matching on x.dim() first makes every case unambiguous, e.g. an (M, F)
    # matrix is never mistaken for a batched (B, M) vector when B == M.
    features = x.size(-1)
    match x.dim(), tuple(B.shape):
        case 2, (m,) if m == M:
            return x.unsqueeze(-1) * B
        case 2, (b, m) if (b, m) == (batch_size, M):
            return x.unsqueeze(-1) * B.unsqueeze(1)
        case 3, (m, f) if (m, f) == (M, features):
            return x @ B.T
        case 3, (b, m, f) if (b, m, f) == (batch_size, M, features):
            return x @ B.mT
    raise _ssm_B_shape_error(B, x, batch_size, M)


def _ssm_B_shape_error(B, x, batch_size, M):
    if x.dim() == 2:
        allowed = f"({M},) or ({batch_size}, {M})"
    else:
        features = x.size(-1)
        allowed = f"({M}, {features}) or ({batch_size}, {M}, {features})"
    return ValueError(
        f"Input matrix B must be of shape {allowed} for {x.dim()}D input x, got {tuple(B.shape)}"
    )


def _ssm_D(D, x, batch_size):
    r"""Return :math:`D \mathbf{x}[n]` for every step.

    This takes the shapes :func:`state_space` accepts.
    """
    features = x.size(-1)
    match x.dim(), tuple(D.shape):
        case 2, (b,) if b == batch_size:
            return D.unsqueeze(1) * x
        case 2, () | (1,):
            return x * D
        case 2, (_,):
            return x.unsqueeze(-1) * D
        case 2, (b, _) if b == batch_size:
            return x.unsqueeze(-1) * D.unsqueeze(1)
        case 3, (f,) if f == features:
            return x @ D
        case 3, () | (1,):
            return x * D
        case 3, (b, f) if (b, f) == (batch_size, features):
            return (x @ D.unsqueeze(-1)).squeeze(-1)
        case 3, (_, f) if f == features:
            return x @ D.T
        case 3, (b, _, f) if (b, f) == (batch_size, features):
            return x @ D.mT
        case 2, _:
            allowed = f"(), (1,), ({batch_size},), (P,), or ({batch_size}, P)"
        case _:
            allowed = (
                f"(), (1,), ({features},), ({batch_size}, {features}), "
                f"(P, {features}), or ({batch_size}, P, {features})"
            )
    raise ValueError(
        f"Input matrix D must be of shape {allowed} for {x.dim()}D input x, got {tuple(D.shape)}"
    )


def _ssm_C_D(h, x, C, D, batch_size, M):
    r"""Return :math:`C \mathbf{h}[n] + D \mathbf{x}[n]` for every step.

    This takes the shapes :func:`state_space` accepts. Where a batched and an
    unbatched shape coincide, such as :math:`P = B`, the batched reading
    wins.
    """
    Dx = None if D is None else _ssm_D(D, x, batch_size)

    if C is not None:
        match tuple(C.shape):
            case (m,) if m == M:
                Ch = h @ C
            case (b, m) if (b, m) == (batch_size, M):
                Ch = (h @ C.unsqueeze(-1)).squeeze(-1)
            case (_, m) if m == M:
                Ch = h @ C.T
            case (b, _, m) if (b, m) == (batch_size, M):
                Ch = h @ C.mT
            case _:
                raise ValueError(
                    f"Output matrix C must be of shape ({M},), ({batch_size}, {M}), "
                    f"(P, {M}), or ({batch_size}, P, {M}), got {tuple(C.shape)}"
                )
    else:
        Ch = h

    if Dx is not None:
        y = Ch + Dx
    else:
        y = Ch
    return y


def diag_state_space(
    x: Tensor,
    L: Tensor | None = None,
    V: Tensor | None = None,
    Vinv: Tensor | None = None,
    A: Tensor | None = None,
    B: Tensor | None = None,
    C: Tensor | None = None,
    D: Tensor | None = None,
    zi: Tensor | None = None,
    out_idx: int | None = None,
    unroll_factor: int = 1,
):
    r"""Compute a state-space model's outputs through its eigendecomposition.

    With :math:`A = V \operatorname{diag}(\boldsymbol{\lambda}) V^{-1}`, the
    state :math:`\mathbf{z} = V^{-1} \mathbf{h}` evolves as :math:`M`
    independent first-order recurrences,

    .. math::
        z_i[n + 1] = \lambda_i z_i[n] + (V^{-1} B \mathbf{x}[n])_i,

    which run with :func:`linear_recurrence`. The outputs are those of
    :func:`state_space` for a diagonalizable :math:`A`.

    Give the decomposition in one of these ways:

    - :attr:`A` alone: the eigenvalues come from
      :func:`torch.linalg.eigvals` and the eigenvectors from
      :func:`~philtorch.mat.find_eigenvectors`.
    - :attr:`L` with :attr:`A`, :attr:`V` or :attr:`Vinv`: the missing
      matrices are computed from it.
    - :attr:`L` with both :attr:`V` and :attr:`Vinv`: used as given.
    - :attr:`L` alone: :math:`A` is diagonal and :math:`V = I`.

    Args:
        x (Tensor): inputs, of shape :math:`(B, N, F)` or :math:`(B, N)`.
        L (Tensor, optional): the eigenvalues :math:`\boldsymbol{\lambda}`,
            of shape :math:`(M)` or :math:`(B, M)`. Batched eigenvalues need
            :attr:`V` or :attr:`Vinv`. Default: ``None``.
        V (Tensor, optional): the eigenvectors as columns, of shape
            :math:`(M, M)` or :math:`(B, M, M)`. Default: ``None``.
        Vinv (Tensor, optional): the inverse of :attr:`V`, of the same shape.
            Default: ``None``.
        A (Tensor, optional): the state matrix, of shape :math:`(M, M)` or
            :math:`(B, M, M)`. Default: ``None``.
        B (Tensor, optional): see :func:`state_space`. Default: ``None``.
        C (Tensor, optional): see :func:`state_space`. Default: ``None``.
        D (Tensor, optional): see :func:`state_space`. Default: ``None``.
        zi (Tensor, optional): see :func:`state_space`. Default: ``None``.
        out_idx (int, optional): see :func:`state_space`. Default: ``None``.
        unroll_factor (int, optional): see :func:`linear_recurrence`.
            Default: ``1``.

    Returns:
        Tensor or tuple of Tensor: as :func:`state_space`. With complex
        eigenvalues, for example a real filter with complex poles, the
        computation is complex: a real :attr:`C` returns the real part, and
        otherwise the outputs are complex.

    Raises:
        AssertionError: if neither :attr:`L` nor :attr:`A` is given, or the
            shapes do not match.
        ValueError: if both :attr:`C` and :attr:`out_idx` are given.

    Example::

        >>> from philtorch.lti import diag_state_space, state_space
        >>> A = torch.tensor([[0.5, 0.2], [0.0, -0.3]])
        >>> C = torch.tensor([1.0, 1.0])
        >>> x = torch.randn(1, 8, 2)
        >>> y = diag_state_space(x, A=A, C=C)
        >>> torch.allclose(y, state_space(A, x, C=C), atol=1e-6)
        True
    """
    assert x.dim() in (
        2,
        3,
    ), f"Input signal must be 2D or 3D (batch, time, [features]), got {x.shape}"
    if not (C is None or out_idx is None):
        raise ValueError("C and out_idx cannot be used together. Use either C or out_idx.")

    batch_size = x.size(0)

    if L is None:
        assert A is not None, "Either L or A must be provided"
        assert A.dim() in (2, 3), f"State matrix A must be 2D or 3D, got {A.shape}"
        assert A.size(-2) == A.size(-1), f"State matrix A must be square, got {A.shape}"
        if A.dim() == 3:
            assert x.size(0) == A.size(0), (
                f"Batch size of A must match batch size of x, got A: {A.size(0)}, x: {x.size(0)}"
            )
        L = torch.linalg.eigvals(A)
        V = Vinv = None
        M = A.size(-1)
    else:
        match L.shape:
            case (_,):
                M = L.size(0)
                if V is not None:
                    assert V.dim() == 2, f"P must be 2D, got {V.shape}"
                    assert V.size(0) == V.size(1) == M, (
                        f"P must be square with size {M}, got {V.shape}"
                    )
                if Vinv is not None:
                    assert Vinv.dim() == 2, f"Vinv must be 2D, got {Vinv.shape}"
                    assert Vinv.size(0) == Vinv.size(1) == M, (
                        f"Vinv must be square with size {M}, got {Vinv.shape}"
                    )

                if A is not None:
                    assert A.dim() == 2, f"A must be 2D, got {A.shape}"
                    assert A.size(0) == A.size(1) == M, (
                        f"A must be square with size {M}, got {A.shape}"
                    )

            case (L_batch, _) if L_batch == batch_size:
                M = L.size(1)
                assert not (V is None and Vinv is None), (
                    "P and Vinv cannot both be None when L is a batch of vectors"
                )
                if V is not None:
                    assert V.dim() == 3, f"P must be 3D, got {V.shape}"
                    assert V.size(0) == batch_size and V.size(1) == V.size(2) == M, (
                        f"P must be a batch of square matrices with size {M}, got {V.shape}"
                    )
                if Vinv is not None:
                    assert Vinv.dim() == 3, f"Vinv must be 3D, got {Vinv.shape}"
                    assert Vinv.size(0) == batch_size and Vinv.size(1) == Vinv.size(2) == M, (
                        f"Vinv must be a batch of square matrices with size {M}, got {Vinv.shape}"
                    )

                if A is not None:
                    assert A.dim() == 3, f"A must be 3D, got {A.shape}"
                    assert A.size(0) == batch_size and A.size(1) == A.size(2) == M, (
                        f"A must be a batch of square matrices with size {M}, got {A.shape}"
                    )

            case _:
                raise ValueError(f"L must be a vector or a batch of vectors, got {L.shape}")

    match (V, Vinv, A):
        case (None, None, None):
            # scalar case
            V = Vinv = torch.eye(M, device=x.device, dtype=x.dtype)
        case (None, None, _):
            V = find_eigenvectors(A, L)
            Vinv = torch.linalg.inv(V)
        case (None, _, _):
            V = torch.linalg.inv(Vinv)
        case (_, None, _):
            Vinv = torch.linalg.inv(V)
        case (_, _, _):
            pass
        case _:
            raise ValueError("Only one of V, Vinv, or A can be provided at a time.")
    assert Vinv is not None, "Vinv must be provided or computed from A or V"
    assert V is not None, "V must be provided or computed from A or Vinv"

    return_zf = True
    if zi is None:
        return_zf = False
        zi = x.new_zeros(batch_size, M)
    elif zi.dim() == 1:
        zi = zi.unsqueeze(0).expand(batch_size, -1)

    x_orig = x
    if Vinv.is_complex():
        if not zi.is_complex():
            zi = zi + 0j  # Ensure zi is complex if Vinv is complex
        if not x.is_complex():
            x = x + 0j  # Ensure x is complex if Vinv is complex
        if B is not None and not B.is_complex():
            B = B + 0j

    match Vinv.dim():
        case 2:
            Vinvzi = zi @ Vinv.T
        case 3:
            Vinvzi = torch.linalg.vecdot(Vinv.conj(), zi.unsqueeze(1))
        case _:
            assert False, f"Vinv must be 2D or 3D, got {Vinv.shape}"

    if B is not None:
        # Same dispatch as _ssm_B, but Vinv is folded into B before touching x.
        features = x.size(-1)
        match x.dim(), tuple(B.shape):
            case 2, (m,) if m == M:
                VinvB = Vinv @ B
                if VinvB.dim() == 2:
                    VinvB = VinvB.unsqueeze(1)
                VinvBx = x.unsqueeze(-1) * VinvB
            case 2, (b, m) if (b, m) == (batch_size, M):
                VinvB = (Vinv @ B.unsqueeze(-1)).squeeze(-1)
                VinvBx = x.unsqueeze(-1) * VinvB.unsqueeze(1)
            case 3, (m, f) if (m, f) == (M, features):
                VinvBx = x @ (Vinv @ B).mT
            case 3, (b, m, f) if (b, m, f) == (batch_size, M, features):
                VinvBx = x @ (Vinv @ B).mT
            case _:
                raise _ssm_B_shape_error(B, x, batch_size, M)
    elif x.dim() == 2 and Vinv.dim() == 2:
        VinvBx = x.unsqueeze(-1) * Vinv[:, 0]
    elif x.dim() == 2 and Vinv.dim() == 3:
        VinvBx = x.unsqueeze(-1) * Vinv[:, None, :, 0]
    elif x.dim() == 3:
        VinvBx = x @ Vinv.mT
    else:
        assert False, f"Input signal x must be 2D or 3D, got {x.shape}"

    recur_runner = (
        LTIRecurrence.apply
        if unroll_factor == 1
        else partial(linear_recurrence, unroll_factor=unroll_factor)
    )

    Vinvh = recur_runner(
        L.broadcast_to((batch_size, M)).flatten(0, 1),
        Vinvzi.flatten(),
        VinvBx.mT.flatten(0, 1),
    ).unflatten(0, (batch_size, M))
    if not return_zf and out_idx is not None:
        h = torch.cat(
            [
                zi[:, None, out_idx],
                (V[..., out_idx, None, :] @ Vinvh[..., :-1]).squeeze(-2),
            ],
            dim=1,
        )
        zf = None
    else:
        h = (V @ Vinvh).mT
        zf = h[:, -1, :] if return_zf else None
        h = torch.cat(
            (
                [zi.unsqueeze(1), h[:, :-1]]
                if out_idx is None
                else [zi[:, None, out_idx], h[:, :-1, out_idx]]
            ),
            dim=1,
        )

    if C is not None and not C.is_complex():
        h = h.real
        zf = zf.real if return_zf else None

    y = _ssm_C_D(h, x_orig, C, D, batch_size, M)
    if return_zf:
        return y, zf
    return y
