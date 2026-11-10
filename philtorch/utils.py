"""General-purpose helpers."""

from collections.abc import Callable
from functools import reduce

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
