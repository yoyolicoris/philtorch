"""Cubic B-spline interpolation."""

import torch
import torch.nn.functional as F
from torch import Tensor

from .recur import LTIRecurrence


def _first_order_filt(
    x: Tensor, a: Tensor, zi: Tensor, b: Tensor | None = None, **kwargs
) -> Tensor:
    xb = x if b is None else x * b
    return LTIRecurrence.apply(a.broadcast_to(x.shape[0]), zi.broadcast_to(x.shape[0]), xb)


def _cubic_coeff(x: Tensor, parallel_form: bool, scipy_padding: bool, **kwargs) -> Tensor:
    r = torch.tensor(3**0.5 - 2, device=x.device, dtype=x.dtype)
    # k_0 = min(14, x.shape[-1] - 1)

    if scipy_padding:
        # in scipy, a sequence [1, 2, 3] is mirrored to [3, 2, 1, 1, 2, 3, 3, 2, 1]
        powers = r ** torch.arange(x.shape[-1], device=x.device, dtype=x.dtype)
        causal_zi = x @ powers
    else:
        # while in torch reflect padding, [1, 2, 3] is padded to [2, 1, 2, 3, 2]
        powers = r ** torch.arange(x.shape[-1] - 1, device=x.device, dtype=x.dtype)
        causal_zi = x[..., 1:] @ powers

    if parallel_form:
        mirrored_x = torch.cat([x, x.flip(-1) if scipy_padding else x[..., :-1].flip(-1)], dim=-1)

        h = _first_order_filt(mirrored_x, r, causal_zi, **kwargs)
        causal_h, anticausal_h = h[..., : x.shape[-1]], h[..., -x.shape[-1] :].flip(-1)
        c = -6 * r / (1 - r * r) * (causal_h + anticausal_h - x)
    else:
        # causal inverse filtering
        h = _first_order_filt(x, r, causal_zi, **kwargs)
        if scipy_padding:
            zi = r / (r - 1) * h[..., -1]
        else:
            zi = -r / (1 - r * r) * (2 * h[..., -1] - x[..., -1])

        # anticausal inverse filtering
        c_flip = _first_order_filt(h[..., :-1].flip(-1), r, zi, -r, **kwargs).flip(-1)
        c = torch.cat([c_flip, zi.unsqueeze(-1)], dim=-1) * 6
    return c


def _cubic_spline_kernel(x: Tensor) -> Tensor:
    abs_x = x.abs()
    mask1 = abs_x <= 1
    mask2 = ~mask1 & (abs_x < 2)
    return torch.where(
        mask1,
        (4 - 6 * abs_x**2 + 3 * abs_x**3) / 6,
        torch.where(
            mask2,
            (2 - abs_x) ** 3 / 6,
            0.0,
        ),
    )


def cspline(
    x: Tensor,
    parallel_form: bool = True,
    scipy_padding: bool = False,
    lamb: float = 0.0,
    **kwargs,
) -> Tensor:
    """Compute the cubic B-spline coefficients of signals.

    The coefficients come from recursive filtering, as in M. Unser, "B-Spline
    Signal Processing: Part II---Efficient Design and Applications," IEEE
    Transactions on Signal Processing, 1993.

    Args:
        x (Tensor): signals, of shape :math:`(B, L)`.
        parallel_form (bool, optional): sum a causal and an anticausal filter
            if ``True``, or cascade them if ``False``; both give the same
            coefficients. Default: ``True``.
        scipy_padding (bool, optional): see :func:`cubic_spline`.
            Default: ``False``.
        lamb (float, optional): the smoothing coefficient; only ``0.0`` is
            implemented. Default: ``0.0``.
        **kwargs: unused.

    Returns:
        Tensor: the coefficients, of shape :math:`(B, L)`.

    Raises:
        NotImplementedError: if :attr:`lamb` is not zero.
    """
    if lamb != 0.0:
        raise NotImplementedError(
            "Regularization for cubic spline interpolation is not implemented."
        )
    return _cubic_coeff(x, parallel_form, scipy_padding, **kwargs)


def cubic_spline(x: Tensor, m: int, scipy_padding: bool = False, **kwargs) -> Tensor:
    """Upsample signals by an integer factor with cubic B-spline interpolation.

    This computes the cubic B-spline coefficients of :attr:`x` by recursive
    filtering, as in M. Unser, "B-Spline Signal Processing: Part II---Efficient
    Design and Applications," IEEE Transactions on Signal Processing, 1993,
    and evaluates the spline at :attr:`m` points per sample interval. The
    spline passes through the original samples.

    Args:
        x (Tensor): signals, of shape :math:`(B, L)`. Unlike SciPy, the batch
            dimension is required.
        m (int): the upsampling factor, at least 1.
        scipy_padding (bool, optional): the boundary condition. If ``True``,
            the signal is extended by mirror symmetry as in
            :func:`scipy.signal.cspline1d`, and the output equals
            :func:`scipy.signal.cspline1d_eval` on the upsampled grid. If
            ``False``, it is extended like the ``"reflect"`` mode of
            :func:`torch.nn.functional.pad`. Default: ``False``.
        **kwargs: options for the coefficients: ``parallel_form`` (bool)
            sums a causal and an anticausal filter if ``True``, the default,
            or cascades them if ``False``, with the same result; ``lamb``
            (float), the smoothing coefficient, must be ``0.0``.

    Returns:
        Tensor: the upsampled signals, of shape :math:`(B, (L - 1) m + 1)`.
        With ``m=1``, :attr:`x` itself.

    Raises:
        AssertionError: if :attr:`m` is not an integer of at least 1.
        ValueError: if :attr:`x` is not 2-D.
        NotImplementedError: if ``lamb`` is not zero.

    Example::

        >>> from philtorch.lti import cubic_spline
        >>> x = torch.randn(1, 16, dtype=torch.float64)
        >>> y = cubic_spline(x, 4)
        >>> y.shape
        torch.Size([1, 61])
        >>> torch.allclose(y[:, ::4], x)
        True
    """
    assert m >= 1 and isinstance(m, int), "Interpolation factor m must be an integer >= 1."
    if x.dim() != 2:
        raise ValueError(f"Input signal x must be 2D (batch, time), got {x.shape}")
    if m == 1:
        return x

    c = cspline(x, scipy_padding=scipy_padding, **kwargs)

    kernel_idx = torch.arange(-2, 2, 1 / m, device=x.device, dtype=x.dtype).reshape(4, m)
    kernel = _cubic_spline_kernel(kernel_idx).flip(0).T

    interped = F.conv1d(
        (
            torch.cat([c[:, :1], c, c[:, -2:].flip(-1)], dim=1).unsqueeze(1)
            if scipy_padding
            else F.pad(c.unsqueeze(1), (1, 2), mode="reflect")
        ),
        kernel.unsqueeze(1),
    ).mT.flatten(1, 2)
    return interped[..., : -(m - 1)]
