from functools import reduce
from typing import Callable, Tuple


def chain_functions(*functions: Callable) -> Callable:
    """Chain multiple callables into a single callable.

    The returned function will invoke the first callable with the provided
    arguments, then pass its result(s) to the next callable, and so on. If a
    callable returns a tuple, that tuple is spread as positional arguments to
    the next callable.

    Args:
        *functions: Callables to chain in invocation order.

    Returns:
        Callable: A function that executes the chain when called.
    """

    def closure(*args: Tuple) -> Tuple:
        return reduce(
            lambda acc, func: func(*acc) if isinstance(acc, tuple) else func(acc),
            functions,
            args,
        )

    return closure
