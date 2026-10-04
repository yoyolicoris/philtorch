"""General-purpose helpers."""

from collections.abc import Callable
from functools import reduce


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
