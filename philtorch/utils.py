"""General-purpose helpers."""

from collections.abc import Callable
from functools import reduce

import torch
from torch import Tensor


def chain_functions(*functions: Callable) -> Callable:
    """Compose callables from left to right into a single callable.

    The returned function calls the first callable with its arguments and
    passes each result on to the next callable. A tuple result is unpacked into
    positional arguments; any other result is passed as the single argument.

    Args:
        *functions (Callable): the callables, in the order they are applied.

    Returns:
        Callable: a function that returns the last callable's result.

    Example::

        >>> from philtorch.utils import chain_functions
        >>> # (x + y)(x - y) at x = 3, y = 2: the tuple is unpacked into s, d
        >>> f = chain_functions(lambda x, y: (x + y, x - y), lambda s, d: s * d)
        >>> f(3, 2)
        5
    """

    def closure(*args: tuple) -> tuple:
        return reduce(
            lambda acc, func: func(*acc) if isinstance(acc, tuple) else func(acc),
            functions,
            args,
        )

    return closure


def _squeeze_first_result(
    y: Tensor | tuple[Tensor, Tensor], squeeze_first: bool
) -> Tensor | tuple[Tensor, Tensor]:
    """Drop the batch axis of lfilter's output, and of its final state if any,
    when ``squeeze_first``."""
    if not squeeze_first:
        return y
    if isinstance(y, tuple):
        return y[0].squeeze(0), y[1].squeeze(0)
    return y.squeeze(0)


def _accumulation_dtype(dtype: torch.dtype) -> torch.dtype:
    """The dtype that values of ``dtype`` accumulate in: float32 for float16 and
    bfloat16, complex64 for complex32, and ``dtype`` itself otherwise."""
    if dtype == torch.complex32:
        return torch.complex64
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


def _check_shared(precision: bool = False, **tensors: Tensor | None) -> None:
    """Check that the given tensors share a dtype and a device.

    Args:
        precision: compare only the real dtype, for functions that let real
            and complex tensors mix.
        **tensors: the tensors by their argument names; ``None`` is skipped.

    Raises:
        ValueError: if the tensors differ in dtype (or precision) or device.
    """
    named = {name: t for name, t in tensors.items() if t is not None}
    kind = "precision" if precision else "dtype"
    if len({(t.dtype.to_real() if precision else t.dtype, t.device) for t in named.values()}) > 1:
        *head, last = named
        got = ", ".join(f"{name}: {t.dtype} on {t.device}" for name, t in named.items())
        raise ValueError(
            f"{', '.join(head)} and {last} must share a {kind} and a device, got {got}"
        )


def _check_integer(name: str, t: Tensor) -> None:
    """Raise a ValueError unless ``t`` is an integer tensor (not bool)."""
    if t.is_floating_point() or t.is_complex() or t.dtype == torch.bool:
        raise ValueError(f"{name} must be an integer tensor, got {t.dtype}")


def _check_scalar_batch(t: Tensor, label: str, name: str, x: Tensor) -> None:
    """Check a scalar recurrence's ``a`` or ``init``, 0-D or 1-D, against x.

    Args:
        t: the tensor.
        label: its description in the message, such as "State matrix a".
        name: its name, such as "a".
        x: the input, (B, N).

    Raises:
        ValueError: if ``t`` is not 0-D or 1-D, or its size is neither 1 nor B.
    """
    if t.dim() not in (0, 1):
        raise ValueError(f"{label} {name} must be 1D or 0D, got {t.shape}")
    if t.dim() == 1 and t.size(0) > 1 and t.size(0) != x.size(0):
        raise ValueError(
            f"{label} {name} must be 1D with the same batch size as x, "
            f"got {name}: {t.size(0)}, x: {x.size(0)}"
        )
