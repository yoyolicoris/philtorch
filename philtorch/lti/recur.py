"""First-order linear recurrences with time-invariant coefficients."""

from torch import Tensor

from .. import _ssm


def linear_recurrence(a: Tensor, init: Tensor, x: Tensor, *, unroll_factor: int = 1) -> Tensor:
    r"""Compute batched first-order linear recurrences.

    This computes

    .. math::
        h[n + 1] = a\, h[n] + x[n], \quad n = 0, \dots, N - 1,

    starting from :math:`h[0] = \text{init}`, and returns
    :math:`h[1], \dots, h[N]`.

    Args:
        a (Tensor): the coefficient, of shape :math:`()` or :math:`(1)` to
            share it, or :math:`(B)` for one per signal.
        init (Tensor): the initial value :math:`h[0]`, of shape :math:`()`,
            :math:`(1)` or :math:`(B)`.
        x (Tensor): input sequences, of shape :math:`(B, N)`.
        unroll_factor (int, optional): ``1`` runs a kernel chosen by device
            and dtype, and raises where none takes the input. A value
            greater than 1 and less than :math:`N` runs a block-unrolled
            PyTorch recursion with blocks of this length, which parallelizes
            over blocks, and :math:`N` or more runs a plain PyTorch loop; see
            the README for guidance. Default: ``1``.

    Returns:
        Tensor: :math:`h[1], \dots, h[N]`, of shape :math:`(B, N)`.

    Raises:
        ValueError: if :attr:`unroll_factor` is less than 1, or, with
            ``unroll_factor > 1``, :attr:`a` or :attr:`init` has a batch size
            other than 1 or that of :attr:`x`.
        RuntimeError: with ``unroll_factor=1``, for the same batch-size
            mismatch.
        NotImplementedError: with ``unroll_factor=1``, if no kernel takes the
            input; an :attr:`unroll_factor` greater than 1 runs PyTorch ops,
            on any device.

    Example::

        >>> from philtorch.lti import linear_recurrence
        >>> x = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        >>> linear_recurrence(torch.tensor(0.5), torch.tensor(0.0), x)
        tensor([[1.0000, 0.5000, 0.2500, 0.1250]])
    """
    if unroll_factor == 1:
        batch_size = x.shape[0]
        return _ssm.scalar_recursion(
            a.broadcast_to(batch_size)[:, None], init.broadcast_to(batch_size), x
        )

    if x.dim() != 2:
        raise ValueError(f"Input x must be 2D, got {x.shape}")
    if a.dim() not in (0, 1):
        raise ValueError(f"State matrix a must be 1D or 0D, got {a.shape}")
    if a.dim() == 1 and a.size(0) > 1 and a.size(0) != x.size(0):
        raise ValueError(
            f"State matrix a must be 1D with the same batch size as x, "
            f"got a: {a.size(0)}, x: {x.size(0)}"
        )

    if a.dim() == 0:
        a = a.expand(1)

    batch_size = x.size(0)

    if init.dim() not in (
        0,
        1,
    ):
        raise ValueError(f"Initial state init must be 1D or 0D, got {init.shape}")
    if init.dim() == 1 and init.size(0) > 1 and init.size(0) != batch_size:
        raise ValueError(
            f"Initial state init must be 1D with the same batch size as x, "
            f"got init: {init.size(0)}, x: {batch_size}"
        )

    if init.dim() == 0:
        init = init.expand(1)

    return _ssm.scalar_recursion(a[:, None], init, x, unroll_factor=unroll_factor)
