"""First-order linear recurrences with time-varying coefficients."""

from torch import Tensor

from .. import _ssm


def linear_recurrence(a: Tensor, init: Tensor, x: Tensor, *, unroll_factor: int = 1) -> Tensor:
    r"""Compute batched first-order recurrences with time-varying coefficients.

    This computes

    .. math::
        h[n + 1] = a[n]\, h[n] + x[n], \quad n = 0, \dots, N - 1,

    starting from :math:`h[0] = \text{init}`, and returns
    :math:`h[1], \dots, h[N]`.

    Args:
        a (Tensor): the coefficients, of shape :math:`(N)` to share them or
            :math:`(B, N)` for one sequence per signal.
        init (Tensor): the initial value :math:`h[0]`, of shape :math:`()`,
            :math:`(1)` or :math:`(B)`.
        x (Tensor): input sequences, of shape :math:`(B, N)`.
        unroll_factor (int, optional): ``1`` runs a kernel chosen by device
            and dtype, or a loop where none takes the input. A
            value greater than 1 and less than :math:`N` runs a block-unrolled
            PyTorch recursion with blocks of this length, and :math:`N` or more
            runs a plain PyTorch loop; see the README for guidance.
            Default: ``1``.

    Returns:
        Tensor: :math:`h[1], \dots, h[N]`, of shape :math:`(B, N)`.

    Raises:
        ValueError: if :attr:`unroll_factor` is less than 1.
        ValueError or RuntimeError: if :attr:`a` or :attr:`init` does not
            match :attr:`x`; which one depends on :attr:`unroll_factor`.

    Example::

        >>> from philtorch.lpv import linear_recurrence
        >>> a = torch.tensor([0.5, 0.5, 0.0, 0.5])
        >>> x = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
        >>> linear_recurrence(a, torch.tensor(0.0), x)
        tensor([[1.0000, 0.5000, 1.0000, 0.5000]])
    """

    if unroll_factor == 1:
        return _ssm.scalar_recursion(a.broadcast_to(x.shape), init.broadcast_to(x.shape[0]), x)

    assert x.dim() == 2, f"Input x must be 2D, got {x.shape}"
    assert a.dim() in (1, 2), f"State matrix a must be 1D or 2D, got {a.shape}"
    if a.dim() == 1 and a.size(0) != x.size(1):
        raise ValueError(
            f"State matrix a must be 1D with the same length as x, "
            f"got a: {a.size(0)}, x: {x.size(1)}"
        )

    batch_size = x.size(0)

    assert init.dim() in (
        0,
        1,
    ), f"Initial state init must be 1D or 0D, got {init.shape}"
    if init.dim() == 1 and init.size(0) > 1 and init.size(0) != batch_size:
        raise ValueError(
            f"Initial state init must be 1D with the same batch size as x, "
            f"got init: {init.size(0)}, x: {batch_size}"
        )

    if init.dim() == 0:
        init = init.expand(1)

    return _ssm.scalar_recursion(
        a[None] if a.dim() == 1 else a, init, x, unroll_factor=unroll_factor
    )
